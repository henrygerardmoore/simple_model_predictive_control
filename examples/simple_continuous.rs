use std::{iter::once, sync::Arc, time::Duration};

use argmin::core::{Executor, observers::ObserverMode};
use argmin_observer_slog::SlogLogger;
use ndarray::{Array1, ArrayView1, array};

use ndarray_linalg::Norm;
use plotters::prelude::*;
use simple_model_predictive_control::{
    dynamics_optimizer::DynamicsOptimizer, dynamics_problem::DynamicsFunction, prelude::*,
};

// x, x velocity, y, y velocity
const STATE_SIZE: usize = 4;

// x thrust, y thrust
// const INPUT_SIZE: usize = 2;

// timestep
const DT: f64 = 0.5;

const LOOKAHEAD: f64 = 5.;

const GOAL: [f64; STATE_SIZE] = [1., 0., 1., 0.];

// simple dynamics, frictionless plane where the input applies a force
fn dynamics_function(state: &Array1<f64>, input: ArrayView1<f64>) -> Array1<f64> {
    let mut derivative = array![0., 0., 0., 0.];
    // xdot = x velocity
    derivative[0] = state[1];
    // x acceleration is given by the input only
    derivative[1] = input[0];

    // same as above but for y
    derivative[2] = state[3];
    derivative[3] = input[1];

    derivative
}

fn state_cost(state: &Array1<f64>, setpoint: &Array1<f64>) -> f64 {
    (state - setpoint).norm()
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
        DynamicsFunction::Continuous(Arc::new(dynamics_function)),
        2,
        Arc::new(state_cost),
        Box::new(simple_dynamics_cost_function),
    );
    let dynamics_optimizer =
        DynamicsOptimizer::new(array![-1., -1.], array![1., 1.], &mpc_problem, 1e-3);
    (mpc_problem, dynamics_optimizer)
}

fn plot(trajectory: Array1<Array1<f64>>) -> Result<(), Box<dyn std::error::Error>> {
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

fn plot_tree(tree_segments: Vec<([f64; 2], [f64; 2])>) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("tree.bmp", (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;
    let (x_extent, y_extent) = tree_segments
        .iter()
        .fold((0.0_f64, 0.0_f64), |acc, (p1, p2)| {
            (
                acc.0.max(p1[0].abs()).max(p2[0].abs()),
                acc.1.max(p1[1].abs()).max(p2[1].abs()),
            )
        });

    let x_extent = x_extent.max(GOAL[0] + 0.01);
    let y_extent = y_extent.max(GOAL[2] + 0.01);
    let mut chart = ChartBuilder::on(&root)
        .caption("MPC Tree", ("sans-serif", 50))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d((-x_extent)..x_extent, (-y_extent)..y_extent)
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
            ShapeStyle::from(&RED.mix(0.5)).stroke_width(1),
        ))?;
    }
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    chart.draw_series(std::iter::once(Circle::new(
        (GOAL[0], GOAL[2]),
        5,
        GREEN.filled(),
    )))?;

    root.present()?;
    Ok(())
}

const OUT_FILE_NAME: &str = "simple_continuous.gif";
// see plotters animation example for reference:
// https://github.com/plotters-rs/plotters/blob/master/plotters/examples/animation.rs
pub fn main() {
    println!("Running simple continuous MPC simulation...");
    let mut trajectory = Array1::<Array1<f64>>::default(0);

    let mut initial_state = array![0., 0., 0., 0.];

    // how many lookahead periods we should do
    let num_chunks = 10;
    let goal = Array1::from_iter(GOAL.into_iter());

    for _ in 0..num_chunks {
        let (mpc_problem, solver) = get_mpc_problem(initial_state.clone(), goal.clone());

        // Run solver
        let res = Executor::new(mpc_problem, solver)
            .configure(|state| state.max_iters(100))
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()
            .unwrap();

        plot_tree(res.solver.get_line_segments()).unwrap();

        let mpc_problem = res.problem.problem.unwrap();

        // update start position and append to overall trajectory
        let this_trajectory =
            mpc_problem.calculate_trajectory(res.state.best_param.unwrap().view());
        trajectory
            .append(ndarray::Axis(0), this_trajectory.view())
            .unwrap();
        initial_state = this_trajectory.last().unwrap().clone();
    }

    println!("MPC simulation complete, now plotting...");
    plot(trajectory).unwrap();
}
