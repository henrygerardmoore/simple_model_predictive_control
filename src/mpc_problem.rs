use core::f64;
use std::{sync::Arc, time::Duration};

use argmin::core::CostFunction;
use ndarray::{
    Array, Array1, ArrayView1,
    parallel::prelude::{IntoParallelRefIterator, ParallelIterator},
};

use crate::dynamics_problem::{DynamicsFunction, DynamicsProblem, StateCostFunction};

/// f(trajectory, inputs, setpoint) -> cost
pub type TrajectoryCostFunction =
    dyn Fn(&Array1<Array1<f64>>, &Array1<f64>, &Array1<f64>) -> f64 + Send + Sync;
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
    pub(crate) dynamics_cost_function: Box<TrajectoryCostFunction>,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        setpoint: Array1<f64>,
        current_state: Array1<f64>,
        sample_period: Duration,
        lookahead_duration: Duration,
        dynamics_function: DynamicsFunction,
        input_size: usize,
        state_cost_function: Arc<StateCostFunction>,
        dynamics_cost_function: Box<TrajectoryCostFunction>,
    ) -> Self {
        Self {
            setpoint: Arc::new(setpoint),
            current_state,
            sample_period,
            lookahead_duration,
            dynamics_function,
            input_size,
            state_cost_function,
            dynamics_cost_function,
        }
    }
    pub fn calculate_trajectory(&self, inputs: ArrayView1<f64>) -> Array1<Array1<f64>> {
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
        (self.dynamics_cost_function)(trajectory, inputs, &self.setpoint)
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
        let trajectory = self.calculate_trajectory(inputs.view());
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

#[cfg(test)]
mod test {
    use std::{sync::Arc, time::Duration};

    use argmin::{
        core::{CostFunction, Executor},
        solver::neldermead::NelderMead,
    };
    use ndarray::{Array1, ArrayView1, array};
    use ndarray_linalg::Norm;

    use crate::{dynamics_problem::DynamicsFunction, prelude::MPCProblem};

    const DT: f64 = 1.;
    const LOOKAHEAD: f64 = 10.;
    const INITIAL_POS: f64 = 1.0;
    const INITIAL_VEL: f64 = 0.0;

    // input is x acceleration, state is (x, vx)
    fn simple_continuous_dynamics(state: &Array1<f64>, input: ArrayView1<f64>) -> Array1<f64> {
        array![state[1], input[0]]
    }

    fn simple_state_cost_function(state: &Array1<f64>, setpoint: &Array1<f64>) -> f64 {
        (state - setpoint).norm()
    }

    // punish deviation of the last state from the goal and the input magnitude slightly
    fn simple_dynamics_cost_function(
        state: &Array1<Array1<f64>>,
        inputs: &Array1<f64>,
        setpoint: &Array1<f64>,
    ) -> f64 {
        (state.last().unwrap() - setpoint).norm()
            + 0.0000001 * inputs.map(|input| input.abs()).sum()
    }

    fn get_simple_problem(goal: Array1<f64>) -> MPCProblem {
        MPCProblem {
            setpoint: Arc::new(goal),
            current_state: array![INITIAL_POS, INITIAL_VEL],
            sample_period: Duration::from_secs_f64(DT),
            lookahead_duration: Duration::from_secs_f64(LOOKAHEAD),
            dynamics_function: DynamicsFunction::Continuous(Arc::new(&simple_continuous_dynamics)),
            input_size: 1,
            state_cost_function: Arc::new(&simple_state_cost_function),
            dynamics_cost_function: Box::new(&simple_dynamics_cost_function),
        }
    }
    #[test]
    fn test_trajectory_calculation() {
        let input = -0.314159;
        let mpc_problem = get_simple_problem(array![0., 0.]);
        let trajectory = mpc_problem.calculate_trajectory(array![input].view());
        let expected_endpoint = array![
            INITIAL_POS + 0.5 * input * DT.powi(2),
            INITIAL_VEL + input * DT
        ];
        assert!((trajectory.last().unwrap() - expected_endpoint).norm() < 1e-3);
    }

    #[test]
    fn test_trajectory_cost() {
        let mpc_problem = get_simple_problem(array![INITIAL_POS, INITIAL_VEL]);
        let cost = mpc_problem.cost(&array![0.]).unwrap();
        assert_eq!(cost, 0.);

        let mpc_problem = get_simple_problem(array![0., 0.]);
        let cost = mpc_problem.cost(&array![0.]).unwrap();
        assert_eq!(cost, 1.);
    }

    #[test]
    fn test_optimization() {
        let goal = array![0., 0.];
        let mpc_problem = get_simple_problem(goal.clone());

        let n_points = 10;
        // this problem can be solved by hand
        let optimal_inputs = Array1::from_iter((0..n_points).map(|subindex| {
            if subindex < (n_points / 2) {
                -0.04
            } else {
                0.04
            }
        }));

        // this is chosen so that the optimal value isn't quite in the simplex (though it will contain 0.05 and 0.00)
        let max_input = 0.5;

        let simplex = Vec::from_iter((0..(n_points + 1)).map(|index| {
            // make the param vec a bang-bang input scaled by the max value
            let max_value = max_input * (index as f64) / (n_points as f64);
            Array1::from_iter((0..n_points).map(|subindex| {
                if subindex < (n_points / 2) {
                    -max_value
                } else {
                    max_value
                }
            }))
        }));
        let solver = NelderMead::new(simplex);
        let res = Executor::new(mpc_problem, solver)
            .configure(|state| state.max_iters(1000))
            .run()
            .unwrap();
        let optimized_inputs = res.state.best_param.unwrap();
        let mpc_problem = get_simple_problem(goal.clone());
        let trajectory = mpc_problem.calculate_trajectory(optimized_inputs.view());
        let last_state = trajectory.last().unwrap();

        assert!((goal - last_state).norm() < 1e-5);
        assert!((optimized_inputs - optimal_inputs).norm() < 1e-5);
    }
}
