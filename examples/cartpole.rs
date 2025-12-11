use std::{
    f64::consts::{PI, TAU},
    sync::Arc,
    time::{Duration, Instant},
};

use argmin::core::{Executor, observers::ObserverMode};
use argmin_observer_slog::SlogLogger;
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

// lookahead time (s)
const LOOKAHEAD: f64 = 2.5;

// cart in center, rod pointing straight up
const GOAL: [f64; 4] = [0., 0., PI, 0.];

// cartpole parameters
const CART_MASS: f64 = 1.; // kg
const POLE_MASS: f64 = 0.1; // kg
const POLE_LENGTH: f64 = 0.2; // m
const GRAVITY: f64 = 9.80665; // m/s^2
const CART_RAIL_BOUNDS: (f64, f64) = (-1., 1.); // (N, N)
const INPUT_MAX: f64 = 200.; // N

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
    let weights = array![5., 1., 10., 1.];
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
            time_limit: Some(Duration::from_secs_f32(2.)),
            ..Default::default()
        },
    );
    (mpc_problem, dynamics_optimizer)
}

fn plot(trajectory: Array1<Array1<f64>>) -> Result<(), Box<dyn std::error::Error>> {
    let now = Instant::now();
    // don't render any faster than 100 fps; if we're simulating faster than that this will result in a little slow-mo, which is ok
    let frame_time = ((DT * 1000.).round() as u32).max(10);
    let root = BitMapBackend::gif(OUT_FILE_NAME, (1280, 720), frame_time)?.into_drawing_area();

    for i in 0..trajectory.len() {
        root.fill(&WHITE)?;

        let point = trajectory[i].clone();

        let aspect_ratio = 1280. / 720.;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Cartpole MPC Trajectory (t = {:.1})", (i as f64) * DT),
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

    let elapsed = now.elapsed();
    println!(
        "Plotting took {:.1} seconds. Result has been saved to {}",
        elapsed.as_secs_f64(),
        OUT_FILE_NAME
    );

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
        .build_cartesian_2d(CART_RAIL_BOUNDS.0..CART_RAIL_BOUNDS.1, 0.0..3.1415926535)
        .unwrap();
    chart.configure_mesh().draw()?;

    chart.draw_series(std::iter::once(Circle::new(
        (GOAL[0], GOAL[2]),
        5,
        GREEN.filled(),
    )))?;
    for (_point_1, point_2) in tree_segments {
        chart.draw_series(std::iter::once(Circle::new(
            (point_2[0], point_2[1]),
            1,
            BLUE.filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

const OUT_FILE_NAME: &str = "cartpole.gif";

pub fn main() {
    println!("Running cartpole MPC simulation...");
    let now = Instant::now();
    let mut trajectory = Array1::<Array1<f64>>::default(0);

    let mut initial_state = array![0., 0., 0., 0.];

    // how many lookahead periods we should do
    let num_chunks = 1;
    let goal = Array1::from_iter(GOAL.into_iter());

    for _ in 0..num_chunks {
        let (mut mpc_problem, solver) = get_mpc_problem(initial_state.clone(), goal.clone());
        // Run solver
        let res = Executor::new(mpc_problem, solver)
            .configure(|state| state.max_iters(1000))
            .add_observer(SlogLogger::term(), ObserverMode::Every(50))
            .run()
            .unwrap();

        plot_tree(res.solver.get_line_segments(0, 2)).unwrap();

        mpc_problem = res.problem.problem.unwrap();
        let this_trajectory =
            mpc_problem.calculate_trajectory(res.state.best_param.unwrap().view());
        trajectory
            .append(ndarray::Axis(0), this_trajectory.view())
            .unwrap();
        initial_state = this_trajectory.last().unwrap().clone();
    }

    let elapsed = now.elapsed();
    println!(
        "MPC simulation of {:.1} seconds complete in {:.1} seconds, now plotting...",
        (num_chunks as f64) * LOOKAHEAD,
        elapsed.as_secs_f64()
    );

    #[cfg(feature = "profiling")]
    {
        println!("Profiling mode: skipping plotting");
        return;
    }

    trajectory_to_plot_format(&mut trajectory);
    plot(trajectory).unwrap();
}
