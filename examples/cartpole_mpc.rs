use std::{
    f64::consts::{PI, TAU},
    time::Duration,
};

use argmin::core::Executor;
use argmin::solver::neldermead::NelderMead;
use ndarray::{Array1, ArrayView1};

use ego_mpc::prelude::*;
use plotters::{prelude::*, style::full_palette::GREY};

// cart position (m), cart velocity (m/s), angle (rad), angular velocity (rad)
// the angle state is positive CCW and
const STATE_SIZE: usize = 4;

// x force on cart
const INPUT_SIZE: usize = 1;

// timestep
const DT: f64 = 0.05;

const LOOKAHEAD: f64 = 1.00;

// cart in center, rod pointing straight up
const GOAL: [f64; 4] = [0., 0., PI, 0.];

// cartpole parameters
const CART_MASS: f64 = 10.; // kg
const POLE_MASS: f64 = 1.; // kg
const POLE_LENGTH: f64 = 0.2; // m
const GRAVITY: f64 = 9.80665; // m/s^2
const CART_RAIL_BOUNDS: (f64, f64) = (-0.5, 0.5);

// dynamics for this example are from https://underactuated.mit.edu/acrobot.html
fn dynamics_function(
    state: &[f64; STATE_SIZE],
    input: &ArrayView1<f64>,
    dt: Duration,
) -> [f64; STATE_SIZE] {
    let dt = dt.as_secs_f64();
    if dt <= 0. {
        return state.clone();
    }

    let x = state[0];
    let vx = state[1];
    let theta = state[2];
    let omega = state[3].clamp(-50., 50.); // don't let omega get crazy high

    // now, see if our x velocity is going to carry us past the rail extents and if it is, add the force to stop the cart to fx
    let next_x_position = x + dt * vx;
    let fx = if next_x_position < CART_RAIL_BOUNDS.0 || next_x_position > CART_RAIL_BOUNDS.1 {
        // clamp next x position and velocity
        // F = dp/dt, so apply the necessary force to reduce the cart's velocity to 0 and any additional force from the cart in the other direction
        let wall_force = -vx * CART_MASS / dt;
        if input[0].signum() == wall_force.signum() {
            wall_force + input[0]
        } else {
            // this is essentially equivalent to the wall also stopping the external cart force
            wall_force
        }
    } else {
        input[0]
    };

    // x accel
    let x_accel = 1. / (CART_MASS + POLE_MASS * theta.sin().powi(2))
        * (fx + POLE_MASS * theta.sin() * (POLE_LENGTH * omega.powi(2) + GRAVITY * theta.cos()));

    // theta accel
    let theta_accel = 1. / (POLE_LENGTH * (CART_MASS + POLE_MASS * theta.sin().powi(2)))
        * (-fx
            - POLE_MASS * POLE_LENGTH * omega.powi(2) * theta.cos() * theta.sin()
            - (CART_MASS + POLE_MASS) * GRAVITY * theta.sin());

    // euler integration and clamp x position to rail bounds
    [
        next_x_position.clamp(CART_RAIL_BOUNDS.0, CART_RAIL_BOUNDS.1),
        vx + dt * x_accel,
        (theta + dt * omega).rem_euclid(TAU), // wrap angle to 0-2*pi value
        omega + dt * theta_accel,
    ]
}

fn distance_cost(state: &[f64; STATE_SIZE], setpoint: &[f64; STATE_SIZE]) -> f64 {
    let weight: [f64; 4] = [2.0, 0.1, 1.0, 0.1];
    // l2 distance squared
    (0..STATE_SIZE).fold(0., |acc, index| {
        acc + weight[index] * (state[index] - setpoint[index]).powi(2)
    })
}

fn state_cost(
    state: &[f64; STATE_SIZE],
    _setpoint: &[f64; STATE_SIZE],
    command: &ArrayView1<f64>,
) -> f64 {
    // penalize high thrust and any angular velocity
    0.01 * command.dot(command) + 0.001 * state[3].abs()
}

fn get_mpc_problem(
    initial_conditions: [f64; STATE_SIZE],
    setpoint: [f64; STATE_SIZE],
) -> MPCProblem<STATE_SIZE, INPUT_SIZE> {
    MPCControllerBuilder::<STATE_SIZE, INPUT_SIZE>::new()
        .dynamics_function(DynamicsFunction::Discrete(Box::new(&dynamics_function)))
        .terminal_cost(&distance_cost)
        .state_cost(&state_cost)
        .lookahead_duration(Duration::from_secs_f64(LOOKAHEAD))
        .sample_period(Duration::from_secs_f64(DT))
        .setpoint(setpoint)
        .initial_conditions(initial_conditions)
        // .input_bounds([(-1., 1.), (-1., 1.)])
        .build()
        .unwrap()
}

fn plot(trajectory: Array1<[f64; STATE_SIZE]>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::gif(OUT_FILE_NAME, (1280, 720), (DT * 1000.).round() as u32)?
        .into_drawing_area();

    for i in 0..trajectory.len() {
        root.fill(&WHITE)?;

        let point = trajectory[i];

        let aspect_ratio = 1280. / 720.;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Cartpole MPC Trajectory (t = {:.1})", (i as f64) * DT),
                ("sans-serif", 50),
            )
            .build_cartesian_2d(-1.0 * aspect_ratio..1.0 * aspect_ratio, -0.75..0.75)?;

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

    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}

// change the trajectory from (x_cart, vx_cart, theta, omega) to (x_cart, y_cart, x_pole_tip, y_pole_tip)
fn trajectory_to_cartesian(trajectory: &mut Array1<[f64; 4]>) {
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
    let mut trajectory = Array1::<[f64; 4]>::default(0);

    let mut initial_state = [0.; STATE_SIZE];

    // how many lookahead periods we should do
    let num_chunks = 40;

    for _ in 0..num_chunks {
        let mpc_problem = get_mpc_problem(initial_state, GOAL);

        let solver = NelderMead::new(mpc_problem.parameter_vector());

        // Run solver
        // plotting is actually the slowest part when in debug mode, but solving is also much slower of course
        #[cfg(debug_assertions)]
        let res = Executor::new(mpc_problem, solver)
            .configure(|state| state.max_iters(1000))
            .run()
            .unwrap();
        #[cfg(not(debug_assertions))]
        let res = Executor::new(mpc_problem, solver)
            .configure(|state| state.max_iters(100000))
            .run()
            .unwrap();

        let mpc_problem = get_mpc_problem(initial_state, GOAL);

        // update start position and append to overall trajectory
        let this_trajectory =
            mpc_problem.calculate_trajectory(&res.state.best_param.unwrap().view());
        trajectory
            .append(ndarray::Axis(0), this_trajectory.view())
            .unwrap();
        initial_state = *this_trajectory.last().unwrap();
    }

    trajectory_to_cartesian(&mut trajectory);

    println!("MPC simulation complete, now plotting...");
    plot(trajectory).unwrap();
}
