use std::{
    f64::consts::{PI, TAU},
    iter::once,
    sync::Arc,
    time::{Duration, Instant},
};

use argmin::core::Executor;
use ndarray::{Array1, ArrayView1, array};

use ndarray_linalg::Norm;
use plotters::{prelude::*, style::full_palette::GREY};
use simple_model_predictive_control::{
    dynamics_optimizer::{DynamicsOptimizer, DynamicsOptimizerSettings},
    dynamics_problem::DynamicsFunction,
    prelude::*,
};

// cart position (m), cart velocity (m/s), angle (rad), angular velocity (rad)
// the angle state is positive CCW and
// const STATE_SIZE: usize = 4;

// x force on cart (N)
const INPUT_SIZE: usize = 1;

// timestep (s)
const DT: f64 = 0.1;

const PLOT_SUBDIVISION: usize = 10;

const CONTROLLER_TIME_LIMIT: f64 = 0.01;

// lookahead time (s)
const LOOKAHEAD: f64 = 5.;

// cart in center, rod pointing straight up
const GOAL: [f64; 4] = [0., 0., PI, 0.];

// cartpole parameters
const CART_MASS: f64 = 1.; // kg
const POLE_MASS: f64 = 0.1; // kg
const POLE_LENGTH: f64 = 0.2; // m
const GRAVITY: f64 = 9.80665; // m/s^2
const CART_RAIL_BOUNDS: (f64, f64) = (-1., 1.); // (N, N)
const INPUT_MAX: f64 = 50.; // N

// dynamics for this example are from https://underactuated.mit.edu/acrobot.html#cart_pole
fn dynamics_function(state: &Array1<f64>, input: ArrayView1<f64>, dt: Duration) -> Array1<f64> {
    let dt = dt.as_secs_f64();
    if dt <= 0. {
        return state.clone();
    }
    let fx = input[0].clamp(-INPUT_MAX, INPUT_MAX);

    let n_euler_steps = 5;
    let dt = dt / (n_euler_steps as f64);
    let mut state = state.clone();

    let angular_damping_coefficient = 1.5;
    let linear_damping_coefficient = 0.5;

    // fine euler integration
    for _ in 0..n_euler_steps {
        let x = state[0];
        let vx = state[1];
        let theta = state[2];
        let omega = state[3].min((PI - 0.01) / DT); // clamp this to just under half the aliasing rate

        // now, see if our x velocity is going to carry us past the rail extents and if it is, add the force to stop the cart to fx
        let next_x_position = x + dt * vx;
        if next_x_position < CART_RAIL_BOUNDS.0 || next_x_position > CART_RAIL_BOUNDS.1 {
            // collision
            let next_x_position = next_x_position.clamp(CART_RAIL_BOUNDS.0, CART_RAIL_BOUNDS.1);
            let next_vx = 0.;
            // F = dp/dt, the force necessary to stop the cart
            // clamp this so if dt is small this doesn't explode the math, though this isn't physically accurate
            let fx = (-vx * (CART_MASS + POLE_MASS * theta.sin().powi(2)) / dt)
                .clamp(-INPUT_MAX, INPUT_MAX);
            // theta accel
            let theta_accel = 1. / (POLE_LENGTH * (CART_MASS + POLE_MASS * theta.sin().powi(2)))
                * (-fx * theta.cos()
                    - POLE_MASS * POLE_LENGTH * omega.powi(2) * theta.cos() * theta.sin()
                    - (CART_MASS + POLE_MASS) * GRAVITY * theta.sin())
                - omega * angular_damping_coefficient;

            state = array![
                next_x_position,
                next_vx,
                (theta + dt * omega).rem_euclid(TAU), // wrap angle to 0-2*pi value
                omega + dt * theta_accel,
            ];
        } else {
            // x accel
            let x_accel = 1. / (CART_MASS + POLE_MASS * theta.sin().powi(2))
                * (fx
                    + POLE_MASS
                        * theta.sin()
                        * (POLE_LENGTH * omega.powi(2) + GRAVITY * theta.cos()))
                - vx * linear_damping_coefficient;

            // theta accel
            let theta_accel = 1. / (POLE_LENGTH * (CART_MASS + POLE_MASS * theta.sin().powi(2)))
                * (-fx * theta.cos()
                    - POLE_MASS * POLE_LENGTH * omega.powi(2) * theta.cos() * theta.sin()
                    - (CART_MASS + POLE_MASS) * GRAVITY * theta.sin())
                - omega * angular_damping_coefficient;

            // euler integration
            state = array![
                next_x_position,
                vx + dt * x_accel,
                (theta + dt * omega).rem_euclid(TAU), // wrap angle to 0-2*pi value
                omega + dt * theta_accel,
            ];
        }
    }
    state
}

fn state_cost(state: &Array1<f64>, setpoint: &Array1<f64>) -> f64 {
    let weights = array![5., 1., 7.5, 1.];
    ((state - setpoint) * weights).norm()
}

fn simple_dynamics_cost_function(
    _state: &Array1<Array1<f64>>,
    inputs: &Array1<f64>,
    _setpoint: &Array1<f64>,
) -> f64 {
    inputs.map(|input| input.abs()).sum()
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
        array![-100.],
        array![100.],
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

    for i in
        (0..trajectory.len()).step_by(((0.1 / (DT / (PLOT_SUBDIVISION as f64))) as usize).max(1))
    {
        root.fill(&WHITE)?;

        let point = trajectory[i].clone();

        let aspect_ratio = 1280. / 720.;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!(
                    "Cartpole MPC Trajectory (t = {:.1})",
                    (i as f64) * DT / (PLOT_SUBDIVISION as f64)
                ),
                ("sans-serif", 50),
            )
            .build_cartesian_2d(-1.0 * aspect_ratio..1.0 * aspect_ratio, -1.0..1.0)?;

        // draw a blue circle for the cart position
        chart.draw_series(std::iter::once(Circle::new(
            (point[0], point[1]),
            5,
            BLUE.filled(),
        )))?;

        // and a red circle for the pole tip position
        chart.draw_series(std::iter::once(Circle::new(
            (point[2], point[3]),
            5,
            RED.filled(),
        )))?;

        // and a line between the cart and pole tip
        chart.draw_series(LineSeries::new(
            std::iter::once((point[0], point[1])).chain(std::iter::once((point[2], point[3]))),
            GREY.filled(),
        ))?;

        // and a green circle for the goal position
        chart.draw_series(std::iter::once(Circle::new(
            (0., POLE_LENGTH),
            5,
            GREEN.filled(),
        )))?;

        root.present()?;
    }

    Ok(())
}

// change the trajectory from (x_cart, vx_cart, theta, omega) to (x_cart, y_cart, x_pole_tip, y_pole_tip)
fn trajectory_to_plot_format(trajectory: &mut Array1<Array1<f64>>) {
    trajectory.iter_mut().for_each(|state| {
        let x_cart = state[0];
        let theta = state[2];
        // arbitrarily choose y position of cart to be 0
        let y_cart = 0.;

        let x_pole_tip = x_cart + POLE_LENGTH * theta.sin();
        let y_pole_tip = y_cart - POLE_LENGTH * theta.cos(); // since down is theta=0

        // update trajectory
        state[1] = y_cart;
        state[2] = x_pole_tip;
        state[3] = y_pole_tip;
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
        .build_cartesian_2d(-1.1..1.1, -0.2..3.3)
        .unwrap();
    chart.configure_mesh().draw()?;

    chart.draw_series(std::iter::once(Circle::new(
        (GOAL[0], GOAL[2]),
        5,
        GREEN.filled(),
    )))?;
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
    // interpolate between each trajectory point
    // we will add `subdivision` - 1 equispaced points between each pair
    let interpoland_step = 1. / (subdivision as f64);
    Array1::from_iter(
        (0..(trajectory.len() - 1))
            .flat_map(|i| {
                let p1 = &trajectory[i];
                let p2 = &trajectory[i + 1];
                let delta = array![
                    p2[0] - p1[0],
                    p2[1] - p1[1],
                    signed_angle_difference(p1[2], p2[2]),
                    p2[3] - p1[3]
                ];
                (0..subdivision).map(move |interp_index| {
                    let mut out_state =
                        p1 + (interp_index as f64) * interpoland_step * delta.clone();
                    out_state[2] = out_state[2].rem_euclid(TAU);
                    out_state
                })
            })
            // this would otherwise miss the last point in the traj
            .chain(once(trajectory.last().unwrap().clone())),
    )
}

const OUT_FILE_NAME: &str = "cartpole.gif";

pub fn main() {
    println!("Running acrobot MPC simulation...");
    let now = Instant::now();

    let mut initial_state = array![0., 0., 0., 0.];
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

    // run with `cargo test --release --package simple_model_predictive_control --example cartpole -- bench::benchmark_dynamics --exact --nocapture`
    #[test]
    pub fn benchmark_dynamics() {
        #[cfg(debug_assertions)]
        let num_iterations = 1000;
        #[cfg(not(debug_assertions))]
        let num_iterations = 100_000_000;

        let test_state = array![1., 2., PI / 2., 0.123];
        let test_input = array![5.];

        bench_function("state_derivative", num_iterations, || {
            dynamics_function(
                black_box(&test_state),
                black_box(test_input.view()),
                black_box(Duration::from_secs_f32(0.1)),
            );
        });
    }

    // run with `cargo test --release --package simple_model_predictive_control --example cartpole -- bench::benchmark_state_cost --exact --nocapture`
    #[test]
    pub fn benchmark_state_cost() {
        #[cfg(debug_assertions)]
        let num_iterations = 1000;
        #[cfg(not(debug_assertions))]
        let num_iterations = 100_000_000;

        let test_state = array![1., 2., PI / 2., 0.123];
        let setpoint = Array1::from_iter(GOAL.into_iter());

        bench_function("state_cost", num_iterations, || {
            state_cost(black_box(&test_state), black_box(&setpoint));
        });
    }
}
