use argmin::core::{Error, IterState, KV, Problem, Solver, TerminationStatus};
use ego_tree::Tree;
use ndarray::Array1;

use crate::{dynamics_problem::DynamicsProblem, prelude::MPCProblem};

#[derive(Clone)]
pub struct DynamicsOptimizer {
    dynamics_tree: Tree<DynamicsProblem>,
    n: usize,
}

impl DynamicsOptimizer {}

#[allow(unused)]
impl Solver<MPCProblem, IterState<Vec<Array1<f64>>, (), (), (), (), f64>> for DynamicsOptimizer {
    fn name(&self) -> &str {
        "Tree-based dynamics optimization"
    }

    fn init(
        &mut self,
        problem: &mut Problem<MPCProblem>,
        state: IterState<Vec<Array1<f64>>, (), (), (), (), f64>,
    ) -> Result<(IterState<Vec<Array1<f64>>, (), (), (), (), f64>, Option<KV>), Error> {
        let mpc_problem = problem.problem.as_ref().unwrap();
        self.dynamics_tree = Tree::new(DynamicsProblem {
            dynamics_function: mpc_problem.dynamics_function.clone(),
            dynamics_cost_function: mpc_problem.dynamics_cost_function.clone(),
            state: mpc_problem.current_state.clone(),
            set_point: mpc_problem.setpoint.clone(),
            dt: mpc_problem.sample_period,
        });
        self.n = (mpc_problem.lookahead_duration.as_secs_f64()
            / mpc_problem.sample_period.as_secs_f64())
        .ceil() as usize;

        todo!()
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<MPCProblem>,
        state: IterState<Vec<Array1<f64>>, (), (), (), (), f64>,
    ) -> Result<(IterState<Vec<Array1<f64>>, (), (), (), (), f64>, Option<KV>), Error> {
        todo!()
    }

    fn terminate(
        &mut self,
        _state: &IterState<Vec<Array1<f64>>, (), (), (), (), f64>,
    ) -> TerminationStatus {
        todo!()
    }
}
