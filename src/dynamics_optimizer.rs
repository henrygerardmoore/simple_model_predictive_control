use std::{
    fmt::{self, Display, Formatter},
    iter::once,
};

use argmin::{
    core::{Error, IterState, KV, Problem, Solver, TerminationReason, TerminationStatus},
    kv,
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

    // tree size we try to maintain
    target_size: usize,

    // node IDs we can reuse instead of inserting
    orphans: Vec<NodeId>,
}

impl DynamicsOptimizer {
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
            .map(|node_ref| node_ref.id())
            .collect();

        best_leaf_ids.into_iter().for_each(|id_to_grow| {
            self.grow_node(id_to_grow);
        });
    }

    fn grow_node(&mut self, node_id: NodeId) {
        let ids: Vec<NodeId> =
            Self::generate_children(&self.dynamics_tree.get(node_id).unwrap().value().clone())
                .map(|dynamics_problem| self.add_node(dynamics_problem))
                .collect();

        ids.into_iter().for_each(|id| {
            self.dynamics_tree.get_mut(node_id).unwrap().append_id(id);
        });
    }

    /// the most important function in the DynamicsOptimizer
    /// *must* connect to the goal if it is possible
    fn generate_children(dynamics: &DynamicsProblem) -> impl Iterator<Item = DynamicsProblem> {
        // TODO: generate children with cost function
        once(dynamics.clone())
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
        let mpc_problem = problem.problem.as_ref().unwrap();
        self.max_depth = (mpc_problem.lookahead_duration.as_secs_f64()
            / mpc_problem.sample_period.as_secs_f64())
        .ceil() as usize;
        self.input_size = mpc_problem.input_size;

        self.target_size = self.max_depth * self.input_size * 1000;

        self.dynamics_tree = Tree::with_capacity(
            DynamicsProblem {
                dynamics_function: mpc_problem.dynamics_function.clone(),
                state_cost_function: mpc_problem.state_cost_function.clone(),
                state: mpc_problem.current_state.clone(),
                set_point: mpc_problem.setpoint.clone(),
                dt: mpc_problem.sample_period,
            },
            self.target_size * 2,
        );

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
