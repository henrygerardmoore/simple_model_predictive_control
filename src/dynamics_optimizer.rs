use std::{
    collections::BTreeSet,
    fmt::{self, Display, Formatter},
    iter::once,
};

use argmin::{
    core::{Error, Executor, IterState, KV, Problem, Solver, TerminationReason, TerminationStatus},
    kv,
    solver::{neldermead::NelderMead, particleswarm::ParticleSwarm},
};
use ego_tree::{NodeId, NodeRef, Tree};
use ndarray::Array1;
use rand::distr::{Distribution, weighted::WeightedIndex};

use crate::{dynamics_problem::DynamicsProblem, prelude::MPCProblem};

// the state/dynamics problem, the inputs to get from parent to this node, and this node's cost
type DynamicsNodeType = (DynamicsProblem, Array1<f64>, f64);

#[derive(Clone, Copy)]
pub struct NodeIDAndCost(NodeId, f64);

// we don't care about node id in comparison
impl PartialEq for NodeIDAndCost {
    fn eq(&self, other: &Self) -> bool {
        if self.1 == other.1 {
            // tie-break on node ID
            self.0 == other.0
        } else {
            false
        }
    }
}

impl Eq for NodeIDAndCost {}

impl PartialOrd for NodeIDAndCost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for NodeIDAndCost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.1.total_cmp(&other.1) {
            // tie-break on node ID
            std::cmp::Ordering::Equal => self.0.cmp(&other.0),
            _ => self.1.total_cmp(&other.1),
        }
    }
}

#[derive(Clone)]
pub struct DynamicsOptimizer {
    // the dynamics at a state and the inputs to get there
    dynamics_tree: Tree<DynamicsNodeType>,
    max_depth: usize,
    // input sequence, cost
    solutions: Vec<(Array1<f64>, f64)>,
    solution_nodes: Vec<NodeId>,
    input_min_max: (Array1<f64>, Array1<f64>),

    // tree size we try to maintain
    target_size: usize,

    // node IDs we can reuse instead of inserting
    orphans: Vec<NodeId>,

    state_cost_epsilon: f64,

    // TODO: try including all nodes in this
    leaves: BTreeSet<NodeIDAndCost>,
}

impl DynamicsOptimizer {
    pub fn new(
        input_min: Array1<f64>,
        input_max: Array1<f64>,
        mpc_problem: &MPCProblem,
        solution_cost_tolerance: f64,
    ) -> Self {
        let max_depth = (mpc_problem.lookahead_duration.as_secs_f64()
            / mpc_problem.sample_period.as_secs_f64())
        .ceil() as usize;
        let input_size = mpc_problem.input_size;

        let target_size = max_depth * input_size * 100;

        let root_dynamics = DynamicsProblem {
            dynamics_function: mpc_problem.dynamics_function.clone(),
            state_cost_function: mpc_problem.state_cost_function.clone(),
            state: mpc_problem.current_state.clone(),
            set_point: mpc_problem.setpoint.clone(),
            dt: mpc_problem.sample_period,
        };

        let root_cost =
            (root_dynamics.state_cost_function)(&root_dynamics.state, &root_dynamics.set_point);

        let dynamics_tree = Tree::with_capacity(
            (root_dynamics, Array1::<f64>::zeros(0), root_cost),
            target_size * 2,
        );
        let root_id = dynamics_tree.root().id();

        assert_eq!(input_min.len(), input_max.len());
        assert_eq!(input_min.len(), mpc_problem.input_size);

        Self {
            dynamics_tree,
            max_depth,
            solutions: vec![],
            input_min_max: (input_min, input_max),
            target_size,
            orphans: vec![],
            solution_nodes: vec![],
            state_cost_epsilon: solution_cost_tolerance,
            leaves: BTreeSet::from([NodeIDAndCost(root_id, root_cost)]),
        }
    }

    // each element of the vector is a line segment, only works for 2D
    pub fn get_line_segments(&self) -> Vec<([f64; 2], [f64; 2])> {
        self.line_segments_recursive(self.dynamics_tree.root().id())
    }

    fn line_segments_recursive(&self, node_id: NodeId) -> Vec<([f64; 2], [f64; 2])> {
        let mut to_return = vec![];
        let node = self.dynamics_tree.get(node_id).unwrap();
        let self_position = node.value().0.get_2d_state_array();

        for child in node.children() {
            to_return.push((self_position, child.value().0.get_2d_state_array()));
            to_return.append(&mut self.line_segments_recursive(child.id()));
        }

        to_return
    }

    fn depth(mut node_ref: NodeRef<'_, DynamicsNodeType>) -> usize {
        let mut depth = 0;

        while let Some(parent) = node_ref.parent() {
            node_ref = parent;
            depth += 1;
        }

        depth
    }

    fn calculate_and_sort_solutions(&mut self, mpc_problem: &MPCProblem) {
        while let Some(solution_node) = self.solution_nodes.pop() {
            self.solutions
                .push(self.get_inputs_and_trajectory_cost_to_node(mpc_problem, solution_node));
        }
        self.solutions.sort_by(|s1, s2| s1.1.total_cmp(&s2.1));
    }

    // traverse up the tree to the root, collecting all the inputs into one and then calculating the trajectory cost from that
    fn get_inputs_and_trajectory_cost_to_node(
        &self,
        mpc_problem: &MPCProblem,
        node: NodeId,
    ) -> (Array1<f64>, f64) {
        let mut node = self.dynamics_tree.get(node);
        let mut inputs = vec![];
        let mut trajectory = vec![];
        while let Some(node_ref) = node.take() {
            inputs.push(node_ref.value().1.clone());
            trajectory.push(node_ref.value().0.state.clone());

            node = node_ref.parent();
        }
        // the root node's inputs are empty and meaningless, don't include them
        let num_inputs = inputs.len() - 1;
        // flatten inputs
        let inputs = Array1::from_iter(
            inputs
                .iter()
                // the root node's inputs are the last item in the iter
                .take(num_inputs)
                // we want them in chronological order, not leaf->root order
                .rev()
                // other things use the input as one large 1D array, return it as such
                .flat_map(|input_chunk| input_chunk.iter().cloned()),
        );
        let trajectory_cost =
            mpc_problem.calculate_trajectory_cost(&Array1::from_vec(trajectory), &inputs);
        (inputs, trajectory_cost)
    }

    // find `num_nodes` leaves via importance sampling and grow them
    fn grow_nodes(&mut self, num_nodes: usize) {
        let (ids, weights): (Vec<_>, Vec<_>) = self
            .leaves
            .iter()
            .filter_map(|id_and_cost| {
                let node_ref = self.dynamics_tree.get(id_and_cost.0).unwrap();
                let depth = Self::depth(node_ref);
                if depth < self.max_depth {
                    Some((id_and_cost.0, id_and_cost.1))
                } else {
                    None
                }
            })
            .unzip();

        let dist = WeightedIndex::new(weights.iter()).unwrap();

        let best_leaf_ids: Vec<NodeId> = (0..num_nodes)
            .map(|_| ids[dist.sample(&mut rand::rng())])
            .collect();

        best_leaf_ids.into_iter().for_each(|id_to_grow| {
            self.grow_node(id_to_grow);
        });
    }

    fn grow_node(&mut self, node_id: NodeId) {
        // consider costs below this to have made it to the goal
        let ids: Vec<NodeId> = Self::generate_children(
            self.input_min_max.clone(),
            &self.dynamics_tree.get(node_id).unwrap().value().0.clone(),
            self.state_cost_epsilon,
            4,
        )
        .map(|(cost, dynamics_problem, inputs)| {
            let id = self.add_node(dynamics_problem, inputs, cost);
            if cost < self.state_cost_epsilon {
                self.solution_nodes.push(id);
            }
            id
        })
        .collect();

        // now remove node_id from leaves since it has children
        let this_cost = self.dynamics_tree.get(node_id).unwrap().value().2;
        self.leaves.remove(&NodeIDAndCost(node_id, this_cost));

        ids.into_iter().for_each(|id| {
            self.dynamics_tree.get_mut(node_id).unwrap().append_id(id);
        });
    }

    /// the most important function in the DynamicsOptimizer
    /// *must* connect to the goal if it is possible
    fn generate_children(
        input_min_max: (Array1<f64>, Array1<f64>),
        parent_dynamics: &DynamicsProblem,
        epsilon: f64,
        branch_factor: usize,
    ) -> impl Iterator<Item = (f64, DynamicsProblem, Array1<f64>)> {
        let solver = ParticleSwarm::new((input_min_max.0.clone(), input_min_max.1.clone()), 10000);

        let res = Executor::new(parent_dynamics.clone(), solver)
            .configure(|state| state.max_iters(1).target_cost(epsilon))
            .run()
            .unwrap();

        // now importance sample to get the other `branch_factor - 1` inputs
        let population = res.state.population.unwrap();
        let dist = WeightedIndex::new(population.iter().map(|particle| {
            let particle_weight = particle.cost.recip();
            if particle_weight.is_finite() {
                particle_weight
            } else {
                // particle cost is probably 0 or very close to 0, just return a huge weight so it gets selected
                1e12
            }
        }))
        .unwrap();

        // Nelder-Mead optimization
        let num_inputs = input_min_max.0.len();
        let importance_sample_simplex: Vec<Array1<f64>> = (0..(num_inputs + 1))
            .map(|_| population[dist.sample(&mut rand::rng())].position.clone())
            .collect();
        let nm_solver = NelderMead::new(importance_sample_simplex);

        let res = Executor::new(parent_dynamics.clone(), nm_solver)
            .configure(|state| state.max_iters(1000).target_cost(epsilon))
            .run()
            .unwrap();

        let nm_optimized_inputs = res.state.best_param.unwrap();

        let importance_sample_iter = (0..(branch_factor - 1))
            .map(move |_| population[dist.sample(&mut rand::rng())].position.clone());

        let inputs_iter = once(nm_optimized_inputs).chain(importance_sample_iter);

        // turn our selected inputs into the next states they create
        inputs_iter.map(|input| {
            let mut new_dynamics = parent_dynamics.clone();
            new_dynamics.state = new_dynamics.dynamics_function.get_next_state(
                &new_dynamics.state,
                input.view(),
                new_dynamics.dt,
            );
            (
                (new_dynamics.state_cost_function)(&new_dynamics.state, &new_dynamics.set_point),
                new_dynamics,
                input.clone(),
            )
        })
    }

    fn add_node(&mut self, dynamics: DynamicsProblem, inputs: Array1<f64>, cost: f64) -> NodeId {
        // either overwrite an orphan or add a new
        if let Some(node_id) = self.orphans.pop() {
            *self.dynamics_tree.get_mut(node_id).unwrap().value() = (dynamics, inputs, cost);
            self.leaves.insert(NodeIDAndCost(node_id, cost));
            node_id
        } else {
            let id = self.dynamics_tree.orphan((dynamics, inputs, cost)).id();
            self.leaves.insert(NodeIDAndCost(id, cost));
            id
        }
    }

    // find the `num_nodes` worst leaves and prune them
    fn prune_nodes(&mut self, num_nodes: usize) {
        let ids_to_prune: Vec<NodeId> = self
            .leaves
            .iter()
            .rev()
            .take(num_nodes)
            .map(|node_ref| node_ref.0)
            .collect();

        ids_to_prune.into_iter().for_each(|id_to_prune| {
            self.prune_node(id_to_prune);
        });
    }

    fn prune_node(&mut self, node_id: NodeId) {
        let mut node = self.dynamics_tree.get_mut(node_id).unwrap();
        // this may make the parent a leaf, store the parent ID for later
        let Some(parent_id) = node.parent().map(|parent_node| parent_node.id()) else {
            panic!("Trying to prune an orphan. This indicates a bug");
        };

        self.leaves.remove(&NodeIDAndCost(node_id, node.value().2));
        node.detach();
        self.orphans.push(node_id);
        let parent = self.dynamics_tree.get(parent_id).unwrap();
        if !parent.has_children() {
            // it's now a leaf, add it
            self.leaves
                .insert(NodeIDAndCost(parent_id, parent.value().2));
        }
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
        mut state: IterState<Array1<f64>, (), (), (), (), f64>,
    ) -> Result<(IterState<Array1<f64>, (), (), (), (), f64>, Option<KV>), Error> {
        let mpc_problem = problem.problem.as_ref().unwrap();

        let action = if self.leaves.len() < self.target_size {
            self.grow_nodes(10);
            TreeOptimizationAction::Grow
        } else {
            self.prune_nodes(10);
            TreeOptimizationAction::Prune
        };

        self.calculate_and_sort_solutions(mpc_problem);

        if !self.solutions.is_empty() {
            // calculate the solutions
            state.best_param = Some(self.solutions[0].0.clone());
            state.best_cost = self.solutions[0].1;
        } else {
            let (best_param, best_cost) = self.get_inputs_and_trajectory_cost_to_node(
                mpc_problem,
                self.leaves.first().unwrap().0,
            );
            state.best_cost = best_cost;
            state.best_param = Some(best_param);
        }

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

#[cfg(test)]
mod test {
    use std::{sync::Arc, time::Duration};

    use argmin::core::Executor;
    use ndarray::{Array1, ArrayView1, array};
    use ndarray_linalg::Norm;

    use crate::{
        dynamics_optimizer::DynamicsOptimizer, dynamics_problem::DynamicsFunction,
        prelude::MPCProblem,
    };

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

    // just punish input magnitude to discriminate between solutions that all reach the goal
    fn simple_dynamics_cost_function(
        _state: &Array1<Array1<f64>>,
        inputs: &Array1<f64>,
        _setpoint: &Array1<f64>,
    ) -> f64 {
        inputs.map(|input| input.abs()).sum()
    }

    fn get_simple_optimizer(goal: Array1<f64>) -> (MPCProblem, DynamicsOptimizer) {
        let mpc_problem = MPCProblem {
            setpoint: Arc::new(goal),
            current_state: array![INITIAL_POS, INITIAL_VEL],
            sample_period: Duration::from_secs_f64(DT),
            lookahead_duration: Duration::from_secs_f64(LOOKAHEAD),
            dynamics_function: DynamicsFunction::Continuous(Arc::new(&simple_continuous_dynamics)),
            input_size: 1,
            state_cost_function: Arc::new(&simple_state_cost_function),
            dynamics_cost_function: Box::new(&simple_dynamics_cost_function),
        };
        let dynamics_optimizer =
            DynamicsOptimizer::new(array![-10.], array![10.], &mpc_problem, 1e-3);
        (mpc_problem, dynamics_optimizer)
    }

    #[test]
    fn test_grow_node_almost_always_finds_goal() {
        let optimal_force = 3.14159265358979;
        let goal = array![
            INITIAL_POS + 0.5 * optimal_force * DT.powi(2),
            INITIAL_VEL + optimal_force * DT
        ];

        // run many times to be sure
        let num_trials = 1000;
        let num_successes = (0..num_trials)
            .map(|_| {
                let (mpc_problem, mut dynamics_optimizer) = get_simple_optimizer(goal.clone());
                dynamics_optimizer.grow_nodes(1);
                dynamics_optimizer.calculate_and_sort_solutions(&mpc_problem);

                // ensure that we found at least one solution
                dynamics_optimizer.solutions.len() > 0
            })
            .fold(0, |acc, succeeded| if succeeded { acc + 1 } else { acc });
        let success_rate = (num_successes as f64) / (num_trials as f64);

        println!("success rate after {} trials: {}", num_trials, success_rate);
        assert!(success_rate > 0.98);
    }

    #[test]
    fn test_node_logic() {
        let goal = array![0., 0.];
        let (_mpc_problem, mut dynamics_optimizer) = get_simple_optimizer(goal);
        while dynamics_optimizer.dynamics_tree.nodes().len() < 50 {
            dynamics_optimizer.grow_nodes(10);
        }

        let num_leaves = dynamics_optimizer.leaves.len();
        let size = dynamics_optimizer.dynamics_tree.nodes().count();

        dynamics_optimizer.prune_nodes(num_leaves);

        // check that the count of nodes in the tree is correct
        assert_eq!(
            dynamics_optimizer
                .dynamics_tree
                .nodes()
                // filter out orphans that have no children (i.e. non-root orphans)
                .filter(|node| { node.parent().is_some() || node.has_children() })
                .count(),
            size - num_leaves
        );

        // also check that all the previous leaves are now orphans
        assert_eq!(dynamics_optimizer.orphans.len(), num_leaves);

        // now grow num_leaves times to ensure we don't have any more orphans left
        // growing by num_leaves once could be limited by how many leaves we have currently
        for _ in 0..num_leaves {
            dynamics_optimizer.grow_nodes(1);
        }

        assert_eq!(dynamics_optimizer.orphans.len(), 0);
    }

    #[test]
    fn test_find_goal_moderate_difficulty() {
        let goal = array![0., 0.];
        let (mpc_problem, mut dynamics_optimizer) = get_simple_optimizer(goal.clone());
        let mut num_iter = 0;
        let max_iter = 10000;
        let target_count = 1000;
        while num_iter < max_iter && dynamics_optimizer.solution_nodes.len() == 0 {
            dynamics_optimizer.grow_nodes(3);
            let count = dynamics_optimizer
                .dynamics_tree
                .nodes()
                // filter out orphans that have no children (i.e. non-root orphans)
                .filter(|node| node.parent().is_some() || node.has_children())
                .count();
            let num_to_prune = count as i32 - target_count as i32;
            if num_to_prune > 0 {
                dynamics_optimizer.prune_nodes((num_to_prune + 5) as usize);
            }
            num_iter += 1;
        }
        dynamics_optimizer.calculate_and_sort_solutions(&mpc_problem);

        // ensure that we found at least one solution
        assert!(dynamics_optimizer.solutions.len() > 0);

        let solution_trajectory =
            mpc_problem.calculate_trajectory(dynamics_optimizer.solutions[0].0.view());

        let last_point = solution_trajectory.last().unwrap();

        println!("last_point: {}", last_point);

        assert!((last_point - goal).norm() < 1e-3);
    }

    #[test]
    fn test_optimizer_argmin() {
        let goal = array![0., 0.];
        let (mpc_problem, dynamics_optimizer) = get_simple_optimizer(goal.clone());
        let res = Executor::new(mpc_problem, dynamics_optimizer)
            .configure(|state| state.max_iters(10000))
            .run()
            .unwrap();

        let optimal_inputs = res.state.best_param.unwrap();
        let (mpc_problem, _) = get_simple_optimizer(goal.clone());
        let solution_trajectory = mpc_problem.calculate_trajectory(optimal_inputs.view());

        let last_point = solution_trajectory.last().unwrap();

        assert!((last_point - goal).norm() < 1e-3);
    }
}
