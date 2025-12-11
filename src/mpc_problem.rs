use core::f64;
use std::{sync::Arc, time::Duration};

use argmin::core::CostFunction;
use ndarray::{
    Array, Array1,
    parallel::prelude::{IntoParallelRefIterator, ParallelIterator},
};

use crate::dynamics_problem::{
    DynamicsCostFunction, DynamicsFunction, DynamicsProblem, StateCostFunction,
};

/// Trait indicating that an argmin problem has a dynamics subproblem, used in DynamicsOptimizer Solver
pub trait DynamicsSubProblem {
    fn get_dynamics(&self, state: Array1<f64>) -> DynamicsProblem;
    fn get_state(&self) -> &Array1<f64>;
    fn get_lookahead(&self) -> &Duration;
}

pub struct MPCProblem {
    pub(crate) setpoint: Arc<Array1<f64>>,
    pub(crate) current_state: Array1<f64>,
    pub(crate) sample_period: Duration,
    pub(crate) lookahead_duration: Duration,
    pub(crate) dynamics_function: DynamicsFunction,

    pub(crate) input_size: usize,

    pub(crate) state_cost_function: Arc<StateCostFunction>,
    pub(crate) dynamics_cost_function: Box<DynamicsCostFunction>,
}

impl DynamicsSubProblem for MPCProblem {
    fn get_dynamics(&self, state: Array1<f64>) -> DynamicsProblem {
        DynamicsProblem {
            dynamics_function: self.dynamics_function.clone(),
            state_cost_function: self.state_cost_function.clone(),
            state,
            set_point: self.setpoint.clone(),
            dt: self.sample_period,
        }
    }

    fn get_state(&self) -> &Array1<f64> {
        &self.current_state
    }

    fn get_lookahead(&self) -> &Duration {
        &self.lookahead_duration
    }
}

impl MPCProblem {
    pub fn calculate_trajectory(&self, inputs: &Array1<f64>) -> Array1<Array1<f64>> {
        let mut current_state = self.current_state.clone();

        Array::from_iter(
            inputs
                .exact_chunks(self.input_size)
                .into_iter()
                .map(|input| {
                    let next_state = self.dynamics_function.get_next_state(
                        &current_state,
                        input,
                        self.sample_period,
                    );
                    current_state = next_state;
                    current_state.clone()
                }),
        )
    }

    /// Calculates the cost of a given trajectory
    pub(crate) fn calculate_trajectory_cost(
        &self,
        trajectory: &Array1<Array1<f64>>,
        inputs: &Array1<f64>,
    ) -> f64 {
        trajectory
            .iter()
            .zip(inputs.exact_chunks(self.input_size))
            .fold(0., |accumulated_cost, (state, input)| {
                accumulated_cost
                    + (self.dynamics_cost_function)(state, input.view(), &self.setpoint)
            })
    }
}

/// Implement `argmin`'s `CostFunction` type.
/// This allows usage of any gradient-free optimizer from `argmin`.
impl CostFunction for MPCProblem {
    type Param = Array1<f64>;

    type Output = f64;

    /// Calculate the trajectory for a given series of inputs,
    /// then calculate the cost of that trajectory.
    fn cost(&self, inputs: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let trajectory = self.calculate_trajectory(inputs);
        let trajectory_cost = self.calculate_trajectory_cost(&trajectory, inputs);
        Ok(trajectory_cost)
    }

    /// Only particle swarm optimization in argmin currently supports this.
    /// See the [argmin book page on parallel evaluation](https://www.argmin-rs.org/book/defining_optimization_problem.html?highlight=rayon#parallel-evaluation-with-bulk_-methods).
    fn bulk_cost<P>(&self, params: &[P]) -> Result<Vec<Self::Output>, argmin_math::Error>
    where
        P: std::borrow::Borrow<Self::Param> + argmin::core::SyncAlias,
        Self::Output: argmin::core::SendAlias,
        Self: argmin::core::SyncAlias,
    {
        params.par_iter().map(|p| self.cost(p.borrow())).collect()
    }
}
