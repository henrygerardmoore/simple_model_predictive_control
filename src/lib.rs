pub mod mpc_controller;
pub mod mpc_controller_builder;
// WIP add unit tests
#[cfg(test)]
mod integration_tests {
    // WIP add integration tests
    mod simple_continuous_mpc {
        use std::time::Duration;

        use crate::{
            mpc_controller::DynamicsFunction, mpc_controller_builder::MPCControllerBuilder,
        };

        const STATE_SIZE: usize = 2;
        const INPUT_SIZE: usize = 2;

        // dyn Fn(&[f64; STATE_SIZE], &[f64; INPUT_SIZE]) -> [f64; 2] + Send + Sync
        fn dynamics_function(state: &[f64; STATE_SIZE], input: &[f64; INPUT_SIZE]) -> [f64; 2] {
            todo!()
        }

        // (&[f64; STATE_SIZE], &[f64; STATE_SIZE]) -> f64
        fn distance_cost(state: &[f64; STATE_SIZE], setpoint: &[f64; STATE_SIZE]) -> f64 {
            todo!()
        }

        #[test]
        fn test_simple_continuous_mpc() {
            let _mpc_controller = MPCControllerBuilder::<STATE_SIZE, INPUT_SIZE>::new()
                .dynamics_function(DynamicsFunction::Continuous(Box::new(&dynamics_function)))
                .terminal_cost(&distance_cost)
                .lookahead_duration(Duration::from_secs_f64(0.5))
                .sample_period(Duration::from_secs_f64(0.005))
                .setpoint([1., 2.])
                .initial_conditions([0., 0.])
                .build()
                .unwrap();
        }
    }

    #[test]
    fn test_simple_discrete_mpc() {}
}
