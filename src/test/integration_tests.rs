mod simple_continuous_mpc {
    use std::time::Duration;

    use argmin::core::Executor;
    use argmin::core::observers::ObserverMode;
    use argmin::solver::neldermead::NelderMead;
    use argmin_observer_slog::SlogLogger;
    use ndarray::ArrayView1;

    use crate::{
        mpc_controller::DynamicsFunction, mpc_controller_builder::MPCControllerBuilder,
        prelude::MPCProblem,
    };

    // x, x velocity, y, y velocity
    const STATE_SIZE: usize = 4;

    // x thrust, y thrust
    const INPUT_SIZE: usize = 2;

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

    // (&[f64; STATE_SIZE], &[f64; STATE_SIZE]) -> f64
    fn distance_cost(state: &[f64; STATE_SIZE], setpoint: &[f64; STATE_SIZE]) -> f64 {
        // distance squared
        (0..STATE_SIZE).fold(0., |acc, index| {
            acc + (state[index] - setpoint[index]).powi(2)
        })
    }

    fn state_cost(_state: &[f64; STATE_SIZE], command: &ArrayView1<f64>) -> f64 {
        // penalize high thrust
        0.00001 * command.dot(command)
    }

    fn get_mpc_problem(
        initial_conditions: [f64; STATE_SIZE],
        setpoint: [f64; STATE_SIZE],
    ) -> MPCProblem<STATE_SIZE, INPUT_SIZE> {
        MPCControllerBuilder::<STATE_SIZE, INPUT_SIZE>::new()
            .dynamics_function(DynamicsFunction::Continuous(Box::new(&dynamics_function)))
            .terminal_cost(&distance_cost)
            .state_cost(&state_cost)
            .lookahead_duration(Duration::from_secs_f64(1.0))
            .sample_period(Duration::from_secs_f64(0.1))
            .setpoint(setpoint)
            .initial_conditions(initial_conditions)
            // .input_bounds([(-1., 1.), (-1., 1.)])
            .build()
            .unwrap()
    }

    #[test]
    fn test_simple_continuous_mpc() {
        // try to move to 1, 2 with 0 velocity, starting at the origin
        let mpc_problem = get_mpc_problem([0.; STATE_SIZE], [1., 0., 2., 0.]);

        let solver = NelderMead::new(mpc_problem.parameter_vector());

        // Run solver
        let res = Executor::new(mpc_problem, solver)
            .configure(|state| state.max_iters(1000))
            .add_observer(SlogLogger::term(), ObserverMode::Always)
            .run()
            .unwrap();

        // print result
        println!("{res}");

        let mpc_problem = get_mpc_problem([0.; STATE_SIZE], [1., 0., 2., 0.]);

        let trajectory = mpc_problem.calculate_trajectory(&res.state.best_param.unwrap().view());

        print!("Trajectory: {:?}", trajectory);
    }
}

mod simple_discrete_mpc {
    #[test]
    fn test_simple_discrete_mpc() {}
}
