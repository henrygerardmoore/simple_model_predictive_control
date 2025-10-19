use std::{
    f64::consts::{PI, TAU},
    time::{Duration, Instant},
};

use argmin::{core::Executor, solver::neldermead::NelderMead};
use ndarray::{Array1, ArrayView1};

use mpc_rs::prelude::*;
use plotters::{prelude::*, style::full_palette::GREY};

// cart position (m), cart velocity (m/s), angle (rad), angular velocity (rad)
// the angle state is positive CCW and
const STATE_SIZE: usize = 4;

// x force on cart (N)
const INPUT_SIZE: usize = 1;

// timestep (s)
const DT: f64 = 0.05;

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
fn dynamics_function(
    state: &[f64; STATE_SIZE],
    input: &ArrayView1<f64>,
    dt: Duration,
) -> [f64; STATE_SIZE] {
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

            state = [
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
            state = [
                next_x_position,
                vx + dt * x_accel,
                (theta + dt * omega).rem_euclid(TAU), // wrap angle to 0-2*pi value
                omega + dt * theta_accel,
            ];
        }
    }
    state
}

fn terminal_cost(state: &[f64; STATE_SIZE], setpoint: &[f64; STATE_SIZE]) -> f64 {
    let weight: [f64; STATE_SIZE] = [1., 1., 1., 1.];
    // for terminal cost, penalize velocity off setpoint too
    (0..STATE_SIZE).fold(0., |acc, i| {
        acc + weight[i] * (state[i] - setpoint[i]).powi(2)
    })
}

fn state_cost(
    state: &[f64; STATE_SIZE],
    setpoint: &[f64; STATE_SIZE],
    _command: &ArrayView1<f64>,
) -> f64 {
    let angle_tolerance = 0.1;
    // penalize off-target x position
    2. * (state[0] - setpoint[0]).powi(2)
    // penalize off-target angle, but only if it's above a tolerable deviation so that the cart can move to the x target without cost
    + 3. * ((state[2] - setpoint[2]).abs() - angle_tolerance).max(0.).powi(2)
    // don't penalize commands in this case to prefer a better solution
}

fn get_mpc_problem(
    initial_conditions: [f64; STATE_SIZE],
    setpoint: [f64; STATE_SIZE],
) -> MPCProblem<STATE_SIZE, INPUT_SIZE> {
    MPCProblemBuilder::<STATE_SIZE, INPUT_SIZE>::new()
        .dynamics_function(DynamicsFunction::Discrete(Box::new(&dynamics_function)))
        .state_cost(&state_cost)
        .terminal_cost(&terminal_cost)
        .lookahead_duration(Duration::from_secs_f64(LOOKAHEAD))
        .sample_period(Duration::from_secs_f64(DT))
        .setpoint(setpoint)
        .initial_conditions(initial_conditions)
        .build()
        .unwrap()
}

fn plot(trajectory: Array1<[f64; STATE_SIZE]>) -> Result<(), Box<dyn std::error::Error>> {
    let now = Instant::now();
    // don't render any faster than 100 fps; if we're simulating faster than that this will result in a little slow-mo, which is ok
    let frame_time = ((DT * 1000.).round() as u32).max(10);
    let root = BitMapBackend::gif(OUT_FILE_NAME, (1280, 720), frame_time)?.into_drawing_area();

    for i in 0..trajectory.len() {
        root.fill(&WHITE)?;

        let point = trajectory[i];

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
fn trajectory_to_plot_format(trajectory: &mut Array1<[f64; 4]>) {
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

#[cfg(debug_assertions)]
const OUT_FILE_NAME: &str = "cartpole_debug.gif";
#[cfg(not(debug_assertions))]
const OUT_FILE_NAME: &str = "cartpole_release.gif";
// see plotters animation example for reference:
// https://github.com/plotters-rs/plotters/blob/master/plotters/examples/animation.rs
pub fn main() {
    println!("Running cartpole MPC simulation...");
    let now = Instant::now();
    let mut trajectory = Array1::<[f64; 4]>::default(0);

    let mut initial_state = [0., 0., 0., 0.];

    // how many lookahead periods we should do
    let num_chunks = 5;

    for _ in 0..num_chunks {
        let mpc_problem = get_mpc_problem(initial_state, GOAL);

        let solver = NelderMead::new(mpc_problem.parameter_vector(1.));
        // Run solver
        // plotting is actually the slowest part when in debug mode, but solving is also much slower of course
        #[cfg(debug_assertions)]
        let res = Executor::new(mpc_problem, solver)
            .configure(|state| state.max_iters(1000))
            .run()
            .unwrap();
        #[cfg(not(debug_assertions))]
        let res = Executor::new(mpc_problem, solver)
            .configure(|state| state.max_iters(10000))
            .run()
            .unwrap();

        let mpc_problem = get_mpc_problem(initial_state, GOAL);

        // update start position and append to overall trajectory
        let this_trajectory =
            mpc_problem.calculate_trajectory(&res.state.best_param.unwrap().view());
        // let this_trajectory = mpc_problem.calculate_trajectory(&Array1::from_vec(vec![10.; n]).view());
        trajectory
            .append(ndarray::Axis(0), this_trajectory.view())
            .unwrap();
        initial_state = *this_trajectory.last().unwrap();
    }

    trajectory_to_plot_format(&mut trajectory);

    let elapsed = now.elapsed();
    println!(
        "MPC simulation of {:.1} seconds complete in {:.1} seconds, now plotting...",
        (num_chunks as f64) * LOOKAHEAD,
        elapsed.as_secs_f64()
    );
    plot(trajectory).unwrap();
}
