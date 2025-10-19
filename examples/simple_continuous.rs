use std::time::Duration;

use argmin::core::Executor;
use argmin::solver::neldermead::NelderMead;
use ndarray::{Array1, ArrayView1};

use mpc_rs::prelude::*;
use plotters::prelude::*;

// x, x velocity, y, y velocity
const STATE_SIZE: usize = 4;

// x thrust, y thrust
const INPUT_SIZE: usize = 2;

// timestep
const DT: f64 = 0.1;

const LOOKAHEAD: f64 = 1.0;

const GOAL: [f64; 4] = [1., 0., 2., 0.];

// simple dynamics, frictionless plane where the input applies a force
fn dynamics_function(state: &[f64; STATE_SIZE], input: &ArrayView1<f64>) -> [f64; STATE_SIZE] {
    let mut derivative = [0.; STATE_SIZE];
    // xdot = x velocity
    derivative[0] = state[1];
    // x acceleration is given by the input only
    derivative[1] = input[0];

    // same as above but for y
    derivative[2] = state[3];
    derivative[3] = input[1];

    derivative
}

fn distance_cost(state: &[f64; STATE_SIZE], setpoint: &[f64; STATE_SIZE]) -> f64 {
    // l2 distance squared
    (0..STATE_SIZE).fold(0., |acc, index| {
        acc + (state[index] - setpoint[index]).powi(2)
    })
}

fn state_cost(
    state: &[f64; STATE_SIZE],
    setpoint: &[f64; STATE_SIZE],
    command: &ArrayView1<f64>,
) -> f64 {
    let mut zero_vel_state = state.clone();
    zero_vel_state[1] = 0.;
    zero_vel_state[3] = 0.;
    // penalize high thrust and y velocity to encourage a curved trajectory
    0.01 * command.dot(command)
        + 0.1 * (state[3] - 0.1).max(0.).exp()
        + 0.01 * distance_cost(&zero_vel_state, setpoint)
}

fn get_mpc_problem(
    initial_conditions: [f64; STATE_SIZE],
    setpoint: [f64; STATE_SIZE],
) -> MPCProblem<STATE_SIZE, INPUT_SIZE> {
    MPCControllerBuilder::<STATE_SIZE, INPUT_SIZE>::new()
        .dynamics_function(DynamicsFunction::Continuous(Box::new(&dynamics_function)))
        .terminal_cost(&distance_cost)
        .state_cost(&state_cost)
        .lookahead_duration(Duration::from_secs_f64(LOOKAHEAD))
        .sample_period(Duration::from_secs_f64(DT))
        .setpoint(setpoint)
        .initial_conditions(initial_conditions)
        .build()
        .unwrap()
}

fn plot(trajectory: Array1<[f64; STATE_SIZE]>) -> Result<(), Box<dyn std::error::Error>> {
    let cartesian_trajectory: Vec<(f64, f64)> = trajectory
        .iter()
        .map(|state| (state[0], state[2]))
        .collect();
    let root = BitMapBackend::gif(OUT_FILE_NAME, (1280, 720), (DT * 1000.).round() as u32)?
        .into_drawing_area();

    for i in 0..trajectory.len() {
        root.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("MPC Trajectory (t = {:.1})", (i as f64) * DT),
                ("sans-serif", 50),
            )
            .build_cartesian_2d(-5.0..5.0, -5.0..5.0)?;

        // draw light red for the trajectory so far
        chart.draw_series(LineSeries::new(
            cartesian_trajectory[0..i].iter().cloned(),
            ShapeStyle::from(&RED.mix(0.5)).stroke_width(2),
        ))?;

        // and a darker red circle for current position
        chart.draw_series(std::iter::once(Circle::new(
            cartesian_trajectory[i],
            5,
            RED.filled(),
        )))?;

        // and a green circle for the goal position
        chart.draw_series(std::iter::once(Circle::new(
            (GOAL[0], GOAL[2]),
            5,
            GREEN.filled(),
        )))?;

        root.present()?;
    }

    println!("Result has been saved to {}", OUT_FILE_NAME);

    Ok(())
}

#[cfg(debug_assertions)]
const OUT_FILE_NAME: &str = "simple_continuous_debug.gif";
#[cfg(not(debug_assertions))]
const OUT_FILE_NAME: &str = "simple_continuous_release.gif";
// see plotters animation example for reference:
// https://github.com/plotters-rs/plotters/blob/master/plotters/examples/animation.rs
pub fn main() {
    println!("Running simple continuous MPC simulation...");
    let mut trajectory = Array1::<[f64; 4]>::default(0);

    let mut initial_state = [0.; STATE_SIZE];

    // how many lookahead periods we should do
    let num_chunks = 10;

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

    println!("MPC simulation complete, now plotting...");
    plot(trajectory).unwrap();
}
