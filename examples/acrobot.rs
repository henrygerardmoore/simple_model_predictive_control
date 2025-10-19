use std::{
    f64::consts::{PI, TAU},
    time::{Duration, Instant},
};

use argmin::{core::Executor, solver::neldermead::NelderMead};
use ndarray::{Array1, ArrayView1};

use mpc_rs::prelude::*;
use plotters::{prelude::*, style::full_palette::GREY};

// link 1 angle CCW from down (rad), link 2 angle of deflection CCW *from link 1* (rad), link 1 angular velocity (rad/2), link 2 angular velocity (rad/s)
const STATE_SIZE: usize = 4;

// torque on link 1 (N*m)
const INPUT_SIZE: usize = 1;

// timestep (s)
const DT: f64 = 0.05; // (s)

// lookahead time (s)
const LOOKAHEAD: f64 = 2.5;

// both arms pointing straight up
const GOAL: [f64; 4] = [PI, 0., 0., 0.];

// acrobot parameters
const INPUT_MAX: f64 = 200.; // N*m

// dynamics for this example are from https://underactuated.mit.edu/acrobot.html
// converted into our form using https://underactuated.mit.edu/multibody.html
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

    // fine euler integration
    for _ in 0..n_euler_steps {
        todo!()
    }
    state
}

fn terminal_cost(state: &[f64; STATE_SIZE], setpoint: &[f64; STATE_SIZE]) -> f64 {
    todo!()
}

fn state_cost(
    state: &[f64; STATE_SIZE],
    setpoint: &[f64; STATE_SIZE],
    _command: &ArrayView1<f64>,
) -> f64 {
    todo!()
}

fn get_mpc_problem(
    initial_conditions: [f64; STATE_SIZE],
    setpoint: [f64; STATE_SIZE],
) -> MPCProblem<STATE_SIZE, INPUT_SIZE> {
    MPCControllerBuilder::<STATE_SIZE, INPUT_SIZE>::new()
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
    let frame_time = ((DT * 1000.).round() as u32).max(50);
    let root = BitMapBackend::gif(OUT_FILE_NAME, (1280, 720), frame_time)?.into_drawing_area();

    for i in 0..trajectory.len() {
        root.fill(&WHITE)?;

        let point = trajectory[i];

        let aspect_ratio = 1280. / 720.;

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Acrobot MPC Trajectory (t = {:.1})", (i as f64) * DT),
                ("sans-serif", 50),
            )
            .build_cartesian_2d(-1.0 * aspect_ratio..1.0 * aspect_ratio, -0.75..0.75)?;

        // draw both links as lines
        todo!();

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
fn trajectory_to_plot_format(trajectory: &mut Array1<[f64; 4]>) {
    trajectory.iter_mut().for_each(|state| todo!());
}

#[cfg(debug_assertions)]
const OUT_FILE_NAME: &str = "cartpole_debug.gif";
#[cfg(not(debug_assertions))]
const OUT_FILE_NAME: &str = "cartpole_release.gif";
// see plotters animation example for reference:
// https://github.com/plotters-rs/plotters/blob/master/plotters/examples/animation.rs
pub fn main() {
    let now = Instant::now();
    let mut trajectory = Array1::<[f64; 4]>::default(0);

    let mut initial_state = [0., 0., 0., 0.];

    // how many lookahead periods we should do
    let num_chunks = 5;

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
