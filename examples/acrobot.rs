use std::{
    f64::consts::{PI, TAU},
    iter::once,
    sync::Arc,
    time::{Duration, Instant},
};

use argmin::core::Executor;
use ndarray::{Array1, Array2, ArrayView1, array};

use ndarray_linalg::Solve;
use plotters::prelude::*;
use simple_model_predictive_control::{
    dynamics_optimizer::{DynamicsOptimizer, DynamicsOptimizerSettings},
    dynamics_problem::DynamicsFunction,
    prelude::*,
};

// link 1 angle CCW from right (rad), link 2 angle of deflection CCW *from link 1* (rad), link 1 angular velocity (rad/s), link 2 angular velocity (rad/s)
// const STATE_SIZE: usize = 4;

// torque on link 1 (N*m)
const INPUT_SIZE: usize = 1;

// timestep (s)
const DT: f64 = 0.3; // (s)

const CONTROLLER_TIME_LIMIT: f64 = 0.1;

const PLOT_SUBDIVISION: usize = 1;

// lookahead time (s)
const LOOKAHEAD: f64 = 10.0;

// start pointing straight down
// const INITIAL_STATE: [f64; 4] = [-PI / 2., 0., 0., 0.];
// both arms pointing straight up
const GOAL: [f64; 4] = [PI / 2., 0., 0., 0.];

// acrobot parameters
const INPUT_MAX: f64 = 50.; // N*m
const GRAVITY: f64 = -9.80665; // m/s^2
const L1: f64 = 2.0; // m
// length to center of 1
const LC1: f64 = L1 / 2.0;
const L2: f64 = 1.8; // m
// length to center of 2
const LC2: f64 = L2 / 2.0;
const M1: f64 = 3.0; // kg
const M2: f64 = 2.0; // kg
// moments of inertia of a rod about its end
const I1: f64 = 1. / 3. * M1 * L1 * L1; // kg m^2
const I2: f64 = 1. / 3. * M2 * L2 * L2; // kg m^2

// get capital M (2x2) as a function of the state
fn mass_matrix(state: &Array1<f64>) -> Array2<f64> {
    let theta2 = state[1];
    array![
        [
            M1 * LC1.powi(2)
                + M2 * (L1.powi(2) + LC2.powi(2) + 2. * L1 * LC2 * theta2.cos())
                + I1
                + I2,
            M2 * (LC2.powi(2) + L1 * LC2 * theta2.cos()) + I2
        ],
        [
            M2 * (LC2.powi(2) + L1 * LC2 * theta2.cos()) + I2,
            M2 * LC2.powi(2) + I2
        ]
    ]
}

// get the contribution from derivatives (2x1) as a function of the state
fn coriolis_matrix(state: &Array1<f64>) -> Array2<f64> {
    let theta2 = state[1];
    let omega1 = state[2];
    let omega2 = state[3];

    array![
        [-M2 * L1 * LC2 * theta2.sin() * omega2.powi(2)
            - 2. * M2 * L1 * LC2 * theta2.sin() * omega2 * omega1],
        [M2 * L1 * LC2 * theta2.sin() * omega1.powi(2)]
    ]
}

// get the torque due to gravity, tau (2x1), as a function of the state
fn gravity_torque_matrix(state: &Array1<f64>) -> Array2<f64> {
    let theta1 = state[0];
    let theta2 = state[1];
    array![
        [(M1 * LC1 + M2 * L1) * GRAVITY * theta1.cos()
            + M2 * LC2 * GRAVITY * (theta1 + theta2).cos()],
        [M2 * LC2 * GRAVITY * (theta1 + theta2).cos()]
    ]
}

// return function's derivatives at a given state and with a given input
#[allow(unused, reason = "Optimized but left as a simpler reference")]
fn unoptimized_state_derivative(state: &Array1<f64>, input: ArrayView1<f64>) -> Array1<f64> {
    let fx = input[0].clamp(-INPUT_MAX, INPUT_MAX);
    // input mapping matrix
    let b = array![[0.], [1.]];
    let m = mass_matrix(&state);
    let c = coriolis_matrix(&state);
    let tau = gravity_torque_matrix(&state);

    // right-hand side of M * second_derivative equation (2x1)
    let rhs = tau + b * fx - c;

    let second_derivative = m.solve_into(rhs.column(0).to_owned()).unwrap();

    let friction_coeff = 0.;

    array![
        state[2],
        state[3],
        second_derivative[0] - friction_coeff * state[2],
        second_derivative[1] - friction_coeff * state[3],
    ]
}

fn state_derivative(state: &Array1<f64>, input: ArrayView1<f64>) -> Array1<f64> {
    let theta2 = state[1];
    let omega1 = state[2];
    let omega2 = state[3];
    let fx = input[0].clamp(-INPUT_MAX, INPUT_MAX);

    // Pre-compute common trigonometric values
    let cos_theta2 = theta2.cos();
    let sin_theta2 = theta2.sin();
    let theta1_plus_theta2 = state[0] + theta2;
    let cos_theta1 = state[0].cos();
    let cos_theta1_plus_theta2 = theta1_plus_theta2.cos();

    // Pre-compute common terms
    let m2_lc2_sq = M2 * LC2.powi(2);
    let m2_l1_lc2 = M2 * L1 * LC2;
    let m2_l1_lc2_cos_theta2 = m2_l1_lc2 * cos_theta2;
    let i1_plus_i2 = I1 + I2;

    // Mass matrix elements (symmetric, so only compute 3 unique values)
    let m11 = M1 * LC1.powi(2)
        + M2 * (L1.powi(2) + LC2.powi(2) + 2. * L1 * LC2 * cos_theta2)
        + i1_plus_i2;
    let m12 = m2_lc2_sq + m2_l1_lc2_cos_theta2 + I2;
    let m22 = m2_lc2_sq + I2;

    // Coriolis terms
    let c1 = -m2_l1_lc2 * sin_theta2 * (omega2.powi(2) + 2. * omega2 * omega1);
    let c2 = m2_l1_lc2 * sin_theta2 * omega1.powi(2);

    // Gravity torques
    let tau1 =
        (M1 * LC1 + M2 * L1) * GRAVITY * cos_theta1 + M2 * LC2 * GRAVITY * cos_theta1_plus_theta2;
    let tau2 = M2 * LC2 * GRAVITY * cos_theta1_plus_theta2;

    // forcing terms
    let rhs1 = tau1 + fx - c1;
    let rhs2 = tau2 - c2;

    // Solve 2x2 system analytically
    let det = m11 * m22 - m12 * m12;
    let det_inv = 1.0 / det;

    let alpha1 = det_inv * (m22 * rhs1 - m12 * rhs2);
    let alpha2 = det_inv * (-m12 * rhs1 + m11 * rhs2);

    array![omega1, omega2, alpha1, alpha2,]
}

fn euler_step(state: &Array1<f64>, input: ArrayView1<f64>, dt: f64) -> Array1<f64> {
    state + dt * state_derivative(state, input)
}

// dynamics for this example are from https://courses.ece.ucsb.edu/ECE179/179D_S12Byl/hw/acrobot_swingup.pdf
fn dynamics_function(state: &Array1<f64>, input: ArrayView1<f64>, dt: Duration) -> Array1<f64> {
    let dt = dt.as_secs_f64();
    if dt <= 0. {
        return state.clone();
    }
    euler_step(&state, input.view(), dt)
}

fn state_cost(state: &Array1<f64>, _setpoint: &Array1<f64>) -> f64 {
    let theta1 = state[0];
    let theta2 = state[1];
    let omega1 = state[2];
    let omega2 = state[3];
    let g = -GRAVITY;
    // purely potential energy
    let target_energy = LC1 * M1 * g + (L1 + LC2) * M2 * g;

    let yc1 = LC1 * theta1.sin();
    let yc2 = L1 * theta1.sin() + LC2 * (theta1 + theta2).sin();

    let potential_energy =
    // potential
    yc1 * M1 * g + yc2 * M2 * g;

    let kinetic_energy =
    // kinetic for link 1
    0.5 * I1 * omega1.powi(2)
    // kinetic for link 2
    + 0.5 * (M2 * L1.powi(2) + I2 + 2. * M2 * L1 * LC2 * theta2.cos()) * omega1.powi(2)
    + 0.5 * I2 * omega2.powi(2) + (I2 + M2 * L1 * LC2 * theta2.cos()) * omega1 * omega2;

    let pe_diff = (potential_energy - target_energy).abs();

    // punish excess kinetic energy and not being at target PE
    (kinetic_energy - pe_diff).max(0.) + pe_diff + 0.01 * kinetic_energy.powi(2)
}

fn simple_dynamics_cost_function(
    states: &Array1<Array1<f64>>,
    inputs: &Array1<f64>,
    setpoint: &Array1<f64>,
) -> f64 {
    let n = inputs.len() as f64;
    // normalize but add 1 to penalize short solutions
    (inputs.map(|input| input.abs()).sum()
        + states.map(|state| state_cost(state, setpoint)).sum()
        + 1.)
        / n
}

fn get_mpc_problem(
    initial_conditions: Array1<f64>,
    setpoint: Array1<f64>,
) -> (MPCProblem, DynamicsOptimizer) {
    let mpc_problem = MPCProblem::new(
        setpoint,
        initial_conditions,
        Duration::from_secs_f64(DT),
        Duration::from_secs_f64(LOOKAHEAD),
        DynamicsFunction::Discrete(Arc::new(dynamics_function)),
        INPUT_SIZE,
        Arc::new(state_cost),
        Box::new(simple_dynamics_cost_function),
    );
    let dynamics_optimizer = DynamicsOptimizer::new(
        array![-INPUT_MAX],
        array![INPUT_MAX],
        &mpc_problem,
        1e-2,
        DynamicsOptimizerSettings {
            time_limit: Some(Duration::from_secs_f64(CONTROLLER_TIME_LIMIT)),
            ..Default::default()
        },
    );
    (mpc_problem, dynamics_optimizer)
}

fn plot(trajectory: Array1<Array1<f64>>) -> Result<(), Box<dyn std::error::Error>> {
    // don't render any faster than 100 fps; if we're simulating faster than that this will result in a little slow-mo, which is ok
    let frame_time = ((DT / (PLOT_SUBDIVISION as f64) * 1000.).round() as u32).max(10);
    let root = BitMapBackend::gif(OUT_FILE_NAME, (1280, 720), frame_time)?.into_drawing_area();

    // step by 0.1 / DT to target ~100 fps
    for i in
        (0..trajectory.len()).step_by(((0.1 / (DT / (PLOT_SUBDIVISION as f64))) as usize).max(1))
    {
        root.fill(&WHITE)?;

        let point = trajectory[i].view();

        let aspect_ratio = 1280. / 720.;
        let y_bound = 4.0;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!(
                    "Acrobot MPC Trajectory (t = {:.1})",
                    (i as f64) * (DT / (PLOT_SUBDIVISION as f64))
                ),
                ("sans-serif", 50),
            )
            .build_cartesian_2d(
                -y_bound * aspect_ratio..y_bound * aspect_ratio,
                -y_bound..y_bound,
            )?;

        // draw both links as lines

        // draw first link as thick red
        chart.draw_series(LineSeries::new(
            std::iter::once((0., 0.)).chain(std::iter::once((point[0], point[1]))),
            RED.filled().stroke_width(6),
        ))?;

        // draw second link as thinner blue (proportionally sized so its mass makes sense)
        chart.draw_series(LineSeries::new(
            std::iter::once((point[0], point[1])).chain(std::iter::once((point[2], point[3]))),
            BLUE.filled().stroke_width(4),
        ))?;

        root.present()?;
    }

    Ok(())
}

// change the trajectory from (theta1, theta2, omega1, omega2) to (x_end1, y_end1, x_end2, y_end2)
fn trajectory_to_plot_format(trajectory: &mut Array1<Array1<f64>>) {
    trajectory.iter_mut().for_each(|state| {
        // tip of first link
        let x1 = L1 * state[0].cos();
        let y1 = L1 * state[0].sin();

        // tip of second link
        let x2 = x1 + L2 * (state[0] + state[1]).cos();
        let y2 = y1 + L2 * (state[0] + state[1]).sin();

        state[0] = x1;
        state[1] = y1;
        state[2] = x2;
        state[3] = y2;
    });
}

fn plot_tree(tree_segments: Vec<([f64; 2], [f64; 2])>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("tree.bmp", (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("MPC Tree", ("sans-serif", 50))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-PI..PI, -PI..PI)
        .unwrap();
    chart.configure_mesh().draw()?;
    for (point_1, point_2) in tree_segments {
        chart.draw_series(std::iter::once(Circle::new(
            (point_2[0], point_2[1]),
            1,
            BLUE.filled(),
        )))?;
        chart.draw_series(LineSeries::new(
            once((point_1[0], point_1[1])).chain(once((point_2[0], point_2[1]))),
            ShapeStyle::from(&RED.mix(0.1)).stroke_width(1),
        ))?;
    }

    chart.draw_series(std::iter::once(Circle::new(
        (GOAL[0], GOAL[2]),
        5,
        GREEN.filled(),
    )))?;

    root.present()?;
    Ok(())
}

fn signed_angle_difference(theta1: f64, theta2: f64) -> f64 {
    let mut diff = (theta2 - theta1) % (2.0 * std::f64::consts::PI);

    if diff > std::f64::consts::PI {
        diff -= 2.0 * std::f64::consts::PI;
    } else if diff < -std::f64::consts::PI {
        diff += 2.0 * std::f64::consts::PI;
    }

    diff
}

fn subdivide_trajectory(
    trajectory: ArrayView1<Array1<f64>>,
    subdivision: usize,
) -> Array1<Array1<f64>> {
    assert!(subdivision > 0, "Cannot subdivide 0 times");
    if subdivision <= 1 {
        return trajectory.to_owned();
    }
    // interpolate between each trajectory point
    // we will add `subdivision` - 1 equispaced points between each pair
    let interpoland_step = 1. / (subdivision as f64);
    Array1::from_iter(
        (0..(trajectory.len() - 1))
            .flat_map(|i| {
                let p1 = &trajectory[i];
                let p2 = &trajectory[i + 1];
                let delta = array![
                    signed_angle_difference(p1[0], p2[0]),
                    signed_angle_difference(p1[1], p2[1]),
                    p2[2] - p1[2],
                    p2[3] - p1[3]
                ];
                (0..subdivision).map(move |interp_index| {
                    let mut out_state =
                        p1 + (interp_index as f64) * interpoland_step * delta.clone();
                    out_state[0] = out_state[0].rem_euclid(TAU);
                    out_state[1] = out_state[1].rem_euclid(TAU);
                    out_state
                })
            })
            // this would otherwise miss the last point in the traj
            .chain(once(trajectory.last().unwrap().clone())),
    )
}

const OUT_FILE_NAME: &str = "acrobot.gif";

// see plotters animation example for reference:
// https://github.com/plotters-rs/plotters/blob/master/plotters/examples/animation.rs
pub fn main() {
    println!("Running acrobot MPC simulation...");
    let now = Instant::now();

    let mut initial_state = array![-PI / 2., 0., 0., 0.];
    let mut trajectory = array![initial_state.clone()];
    let mut line_segments = vec![];

    let goal = Array1::from_iter(GOAL.into_iter());
    let num_chunks = (5. / CONTROLLER_TIME_LIMIT).ceil() as usize;
    for _ in 0..num_chunks {
        let (mut mpc_problem, solver) = get_mpc_problem(initial_state, goal.clone());
        // Run solver
        let res = Executor::new(mpc_problem, solver).run().unwrap();
        let mut this_segments = res.solver.get_line_segments(0, 1);
        line_segments.append(&mut this_segments);
        let optimized_input = array![*res.state.best_param.unwrap().first().unwrap()];

        mpc_problem = res.problem.problem.unwrap();
        let this_trajectory = mpc_problem.calculate_trajectory(optimized_input.view());
        trajectory
            .append(ndarray::Axis(0), this_trajectory.view())
            .unwrap();
        initial_state = this_trajectory.last().unwrap().clone();
    }

    let elapsed = now.elapsed();
    println!(
        "MPC simulation complete in {:.2} seconds, now plotting...",
        elapsed.as_secs_f64()
    );

    #[cfg(feature = "profiling")]
    {
        println!("Profiling mode: skipping plotting");
        return;
    }

    let now = Instant::now();

    plot_tree(line_segments).unwrap();

    // plot with subdivisions for smoother visualizations
    let mut trajectory = subdivide_trajectory(trajectory.view(), PLOT_SUBDIVISION);
    trajectory_to_plot_format(&mut trajectory);
    plot(trajectory).unwrap();

    let elapsed = now.elapsed();
    println!(
        "Plotting took {:.1} seconds. Result has been saved to {}",
        elapsed.as_secs_f64(),
        OUT_FILE_NAME
    );
}

#[cfg(test)]
mod bench {
    use super::*;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    fn bench_function<F>(name: &str, iterations: usize, mut f: F) -> Duration
    where
        F: FnMut(),
    {
        let mut total_duration = Duration::ZERO;

        for _ in 0..iterations {
            let start = Instant::now();
            black_box(f());
            total_duration += start.elapsed();
        }

        let avg = total_duration / iterations as u32;
        println!(
            "{}: avg {:?}/call over {} iterations",
            name, avg, iterations
        );
        avg
    }

    // run with `cargo test --release --package simple_model_predictive_control --example acrobot -- bench::benchmark_state_derivative --exact --nocapture`
    #[test]
    pub fn benchmark_state_derivative() {
        #[cfg(debug_assertions)]
        let num_iterations = 1000;
        #[cfg(not(debug_assertions))]
        let num_iterations = 100_000_000;

        let test_state = array![-PI / 4., PI / 6., 0.5, -0.3];
        let test_input = array![25.0];

        bench_function("state_derivative", num_iterations, || {
            state_derivative(black_box(&test_state), black_box(test_input.view()));
        });
    }

    // run with `cargo test --release --package simple_model_predictive_control --example acrobot -- bench::benchmark_state_cost --exact --nocapture`
    #[test]
    pub fn benchmark_state_cost() {
        #[cfg(debug_assertions)]
        let num_iterations = 1000;
        #[cfg(not(debug_assertions))]
        let num_iterations = 100_000_000;

        let test_state = array![-PI / 4., PI / 6., 0.5, -0.3];
        let setpoint = Array1::from_iter(GOAL.into_iter());

        bench_function("state_cost", num_iterations, || {
            state_cost(black_box(&test_state), black_box(&setpoint));
        });
    }

    // run with `cargo test --release --package simple_model_predictive_control --example acrobot -- bench::benchmark_unoptimized_state_derivative --exact --nocapture`
    #[test]
    pub fn benchmark_unoptimized_state_derivative() {
        #[cfg(debug_assertions)]
        let num_iterations = 1000;
        #[cfg(not(debug_assertions))]
        let num_iterations = 100_000_000;

        let test_state = array![-PI / 4., PI / 6., 0.5, -0.3];
        let test_input = array![25.0];

        bench_function("unoptimized_state_derivative", num_iterations, || {
            unoptimized_state_derivative(black_box(&test_state), black_box(test_input.view()));
        });
    }
}
