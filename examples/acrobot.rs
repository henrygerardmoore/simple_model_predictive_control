use std::{
    f64::consts::{PI, TAU},
    iter::once,
    sync::Arc,
    time::{Duration, Instant},
};

use argmin::core::{Executor, observers::ObserverMode};
use argmin_observer_slog::SlogLogger;
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
fn state_derivative(state: &Array1<f64>, input: ArrayView1<f64>) -> Array1<f64> {
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

fn euler_step(state: &Array1<f64>, input: ArrayView1<f64>, dt: f64) -> Array1<f64> {
    let mut new_state = state + dt * state_derivative(state, input);
    // wrap angles
    new_state[0] = new_state[0].rem_euclid(TAU);
    new_state[1] = new_state[1].rem_euclid(TAU);
    // don't let angular velocity get too high
    new_state[2] = new_state[2].clamp(-20., 20.);
    new_state[3] = new_state[3].clamp(-20., 20.);
    new_state
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
    (kinetic_energy - pe_diff).max(0.) + pe_diff
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
        array![-INPUT_MAX],
        array![INPUT_MAX],
        &mpc_problem,
        1e-2,
        DynamicsOptimizerSettings {
            time_limit: Some(Duration::from_secs_f32(5.)),
            ..Default::default()
        },
    );
    (mpc_problem, dynamics_optimizer)
}

fn plot(trajectory: Array1<Array1<f64>>) -> Result<(), Box<dyn std::error::Error>> {
    let now = Instant::now();
    // don't render any faster than 100 fps; if we're simulating faster than that this will result in a little slow-mo, which is ok
    let frame_time = ((DT * 1000.).round() as u32).max(50);
    let root = BitMapBackend::gif(OUT_FILE_NAME, (1280, 720), frame_time)?.into_drawing_area();

    // step by 0.1 / DT to target ~100 fps
    for i in (0..trajectory.len()).step_by(((0.1 / DT) as usize).max(1)) {
        root.fill(&WHITE)?;

        let point = trajectory[i].view();

        let aspect_ratio = 1280. / 720.;
        let y_bound = 4.0;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Acrobot MPC Trajectory (t = {:.1})", (i as f64) * DT),
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

    let elapsed = now.elapsed();
    println!(
        "Plotting took {:.1} seconds. Result has been saved to {}",
        elapsed.as_secs_f64(),
        OUT_FILE_NAME
    );

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
        .build_cartesian_2d(0.0..TAU, 0.0..TAU)
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
            ShapeStyle::from(&RED.mix(0.5)).stroke_width(1),
        ))?;
    }

    root.present()?;
    Ok(())
}

const OUT_FILE_NAME: &str = "acrobot.gif";

// see plotters animation example for reference:
// https://github.com/plotters-rs/plotters/blob/master/plotters/examples/animation.rs
pub fn main() {
    println!("Running acrobot MPC simulation...");
    let now = Instant::now();

    let initial_state = array![-PI / 2., 0., 0., 0.];

    // how many lookahead periods we should do
    let goal = Array1::from_iter(GOAL.into_iter());

    let (mut mpc_problem, solver) = get_mpc_problem(initial_state, goal.clone());
    // Run solver
    let res = Executor::new(mpc_problem, solver)
        .configure(|state| state.max_iters(1001))
        .add_observer(SlogLogger::term(), ObserverMode::Every(100))
        .run()
        .unwrap();

    let elapsed = now.elapsed();
    println!(
        "MPC simulation of {:.1} seconds complete in {:.1} seconds, now plotting...",
        LOOKAHEAD,
        elapsed.as_secs_f64()
    );

    #[cfg(feature = "profiling")]
    {
        println!("Profiling mode: skipping plotting");
        return;
    }

    plot_tree(res.solver.get_line_segments(0, 1)).unwrap();

    mpc_problem = res.problem.problem.unwrap();
    let mut trajectory = mpc_problem.calculate_trajectory(res.state.best_param.unwrap().view());

    trajectory_to_plot_format(&mut trajectory);
    plot(trajectory).unwrap();
}
