use std::fmt::{self, Display, Formatter};

use argmin::{
    core::{Error, Executor, IterState, KV, Problem, Solver, TerminationReason, TerminationStatus},
    kv,
    solver::newton::Newton,
};
use ego_tree::{NodeId, NodeRef, Tree};
use ndarray::Array1;

use crate::{dynamics_problem::DynamicsProblem, prelude::MPCProblem};

#[derive(Clone)]
pub struct DynamicsOptimizer {
    dynamics_tree: Tree<DynamicsProblem>,
    max_depth: usize,
    // input sequence, cost
    solutions: Vec<(Array1<f64>, f64)>,
    input_size: usize,
    input_min_middle_max: (f64, f64, f64),

    // tree size we try to maintain
    target_size: usize,

    // node IDs we can reuse instead of inserting
    orphans: Vec<NodeId>,
}

impl DynamicsOptimizer {
    pub fn new(input_min: f64, input_max: f64, mpc_problem: &MPCProblem) -> Self {
        let max_depth = (mpc_problem.lookahead_duration.as_secs_f64()
            / mpc_problem.sample_period.as_secs_f64())
        .ceil() as usize;
        let input_size = mpc_problem.input_size;

        let target_size = max_depth * input_size * 1000;

        let dynamics_tree = Tree::with_capacity(
            DynamicsProblem {
                dynamics_function: mpc_problem.dynamics_function.clone(),
                state_cost_function: mpc_problem.state_cost_function.clone(),
                state: mpc_problem.current_state.clone(),
                set_point: mpc_problem.setpoint.clone(),
                dt: mpc_problem.sample_period,
            },
            target_size * 2,
        );

        Self {
            dynamics_tree,
            max_depth,
            solutions: vec![],
            input_size: mpc_problem.input_size,
            input_min_middle_max: (input_min, (input_min + input_max) / 2., input_max),
            target_size,
            orphans: vec![],
        }
    }

    fn depth(mut node_ref: NodeRef<'_, DynamicsProblem>) -> usize {
        let mut depth = 0;

        while let Some(parent) = node_ref.parent() {
            node_ref = parent;
            depth += 1;
        }

        depth
    }

    // find the `num_nodes` best leaves and add children to them
    fn grow_nodes(&mut self, num_nodes: usize) {
        let mut leaves = self.leaves().collect::<Vec<NodeRef<'_, DynamicsProblem>>>();

        leaves.sort_by(|n1, n2| {
            let n1_val = (n1.value().state_cost_function)(&n1.value().state, &n1.value().set_point);
            let n2_val = (n2.value().state_cost_function)(&n2.value().state, &n2.value().set_point);
            n1_val.total_cmp(&n2_val)
        });

        let best_leaf_ids: Vec<NodeId> = leaves
            .into_iter()
            .rev()
            .take(num_nodes)
            .filter_map(|node_ref| {
                let depth = Self::depth(node_ref);
                if depth < self.max_depth {
                    Some(node_ref.id())
                } else {
                    None
                }
            })
            .collect();

        best_leaf_ids.into_iter().for_each(|id_to_grow| {
            self.grow_node(id_to_grow);
        });
    }

    fn grow_node(&mut self, node_id: NodeId) {
        let ids: Vec<NodeId> = Self::generate_children(
            self.input_min_middle_max,
            self.input_size,
            &self.dynamics_tree.get(node_id).unwrap().value().clone(),
        )
        .map(|dynamics_problem| self.add_node(dynamics_problem))
        .collect();

        ids.into_iter().for_each(|id| {
            self.dynamics_tree.get_mut(node_id).unwrap().append_id(id);
        });
    }

    /// the most important function in the DynamicsOptimizer
    /// *must* connect to the goal if it is possible
    fn generate_children(
        input_values: (f64, f64, f64),
        input_size: usize,
        dynamics: &DynamicsProblem,
    ) -> impl Iterator<Item = DynamicsProblem> {
        // Set up solver
        let solver: Newton<f64> = Newton::new();

        let input = Array1::from_vec(vec![input_values.1; input_size]);
        // Run solver
        let res = Executor::new(dynamics.clone(), solver)
            .configure(|state| state.param(input).max_iters(10))
            .run()
            .unwrap();
        let optimized_inputs = res.state.best_param.unwrap();

        let min_input = Array1::from_vec(vec![input_values.0; input_size]);
        let max_input = Array1::from_vec(vec![input_values.0; input_size]);

        // turn our 3 inputs into the next states they create
        [min_input, optimized_inputs, max_input]
            .map(|input| {
                let mut new_dynamics = dynamics.clone();
                new_dynamics.state = new_dynamics.dynamics_function.get_next_state(
                    &new_dynamics.state,
                    input.view(),
                    new_dynamics.dt,
                );
                new_dynamics
            })
            .into_iter()
    }

    fn add_node(&mut self, dynamics: DynamicsProblem) -> NodeId {
        // either overwrite an orphan or add a new
        if let Some(node_id) = self.orphans.pop() {
            *self.dynamics_tree.get_mut(node_id).unwrap().value() = dynamics;
            node_id
        } else {
            self.dynamics_tree.orphan(dynamics).id()
        }
    }

    // find the `num_nodes` worst leaves and prune them
    fn prune_nodes(&mut self, num_nodes: usize) {
        let mut leaves = self.leaves().collect::<Vec<NodeRef<'_, DynamicsProblem>>>();

        leaves.sort_by(|n1, n2| {
            let n1_val = (n1.value().state_cost_function)(&n1.value().state, &n1.value().set_point);
            let n2_val = (n2.value().state_cost_function)(&n2.value().state, &n2.value().set_point);
            n1_val.total_cmp(&n2_val)
        });

        let ids_to_prune: Vec<NodeId> = leaves
            .into_iter()
            .take(num_nodes)
            .map(|node_ref| node_ref.id())
            .collect();

        ids_to_prune.into_iter().for_each(|id_to_prune| {
            self.prune_node(id_to_prune);
        });
    }

    fn prune_node(&mut self, node: NodeId) {
        self.dynamics_tree.get_mut(node).unwrap().detach();
        self.orphans.push(node);
    }

    fn leaves(&self) -> impl Iterator<Item = NodeRef<'_, DynamicsProblem>> {
        self.dynamics_tree
            .nodes()
            .filter(|node| !node.has_children())
    }
}

#[derive(Debug)]
enum TreeOptimizationAction {
    /// adding children to a node
    Grow,
    /// removing low-value nodes
    Prune,
}

impl Display for TreeOptimizationAction {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match *self {
            TreeOptimizationAction::Grow => write!(f, "Grow"),
            TreeOptimizationAction::Prune => write!(f, "Prune"),
        }
    }
}

#[allow(unused)]
impl Solver<MPCProblem, IterState<Array1<f64>, (), (), (), (), f64>> for DynamicsOptimizer {
    fn name(&self) -> &str {
        "Tree-based dynamics optimization"
    }

    fn init(
        &mut self,
        problem: &mut Problem<MPCProblem>,
        state: IterState<Array1<f64>, (), (), (), (), f64>,
    ) -> Result<(IterState<Array1<f64>, (), (), (), (), f64>, Option<KV>), Error> {
        // we did our initialization of the solver in `new()`

        Ok((state, None))
    }

    fn next_iter(
        &mut self,
        problem: &mut Problem<MPCProblem>,
        state: IterState<Array1<f64>, (), (), (), (), f64>,
    ) -> Result<(IterState<Array1<f64>, (), (), (), (), f64>, Option<KV>), Error> {
        let mpc_problem = problem.problem.as_ref().unwrap();

        let action = if self.leaves().count() < self.target_size {
            self.grow_nodes(10);
            TreeOptimizationAction::Grow
        } else {
            self.prune_nodes(10);
            TreeOptimizationAction::Prune
        };

        Ok((state, Some(kv!("action" => format!("{action}");))))
    }

    fn terminate(
        &mut self,
        _state: &IterState<Array1<f64>, (), (), (), (), f64>,
    ) -> TerminationStatus {
        // TODO: look for solutions with cost under a certain threshold and only terminate then
        if !self.solutions.is_empty() {
            TerminationStatus::Terminated(TerminationReason::SolverConverged)
        } else {
            TerminationStatus::NotTerminated
        }
    }
}
