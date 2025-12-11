use std::{
    fmt::{self, Display, Formatter},
    iter::once,
};

use argmin::{
    core::{Error, Executor, IterState, KV, Problem, Solver, TerminationReason, TerminationStatus},
    kv,
    solver::{linesearch::HagerZhangLineSearch, neldermead::NelderMead, quasinewton::LBFGS},
};
use ego_tree::{NodeId, NodeRef, Tree};
use ndarray::Array1;

use crate::{dynamics_problem::DynamicsProblem, prelude::MPCProblem};

type DynamicsNodeType = (DynamicsProblem, Array1<f64>);
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
}

impl DynamicsOptimizer {
    pub fn new(input_min: Array1<f64>, input_max: Array1<f64>, mpc_problem: &MPCProblem) -> Self {
        let max_depth = (mpc_problem.lookahead_duration.as_secs_f64()
            / mpc_problem.sample_period.as_secs_f64())
        .ceil() as usize;
        let input_size = mpc_problem.input_size;

        let target_size = max_depth * input_size * 1000;

        let dynamics_tree = Tree::with_capacity(
            (
                DynamicsProblem {
                    dynamics_function: mpc_problem.dynamics_function.clone(),
                    state_cost_function: mpc_problem.state_cost_function.clone(),
                    state: mpc_problem.current_state.clone(),
                    set_point: mpc_problem.setpoint.clone(),
                    dt: mpc_problem.sample_period,
                },
                Array1::<f64>::zeros(0),
            ),
            target_size * 2,
        );

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
            state_cost_epsilon: 1e-5,
        }
    }

    fn depth(mut node_ref: NodeRef<'_, DynamicsNodeType>) -> usize {
        let mut depth = 0;

        while let Some(parent) = node_ref.parent() {
            node_ref = parent;
            depth += 1;
        }

        depth
    }

    fn calculate_solutions(&mut self, mpc_problem: &MPCProblem) {
        while let Some(solution_node) = self.solution_nodes.pop() {
            self.solutions
                .push(self.get_inputs_and_trajectory_cost_to_node(mpc_problem, solution_node));
        }
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

    // find the `num_nodes` best leaves and add children to them
    fn grow_nodes(&mut self, num_nodes: usize) {
        let mut leaves = self
            .leaves()
            .collect::<Vec<NodeRef<'_, DynamicsNodeType>>>();

        leaves.sort_by(|n1, n2| {
            let dynamics_1 = &n1.value().0;
            let dynamics_2 = &n2.value().0;
            let n1_state_cost =
                (dynamics_1.state_cost_function)(&dynamics_1.state, &dynamics_1.set_point);
            let n2_state_cost =
                (dynamics_2.state_cost_function)(&dynamics_2.state, &dynamics_2.set_point);
            n1_state_cost.total_cmp(&n2_state_cost)
        });

        let best_leaf_ids: Vec<NodeId> = leaves
            .into_iter()
            .filter_map(|node_ref| {
                let depth = Self::depth(node_ref);
                if depth < self.max_depth {
                    Some(node_ref.id())
                } else {
                    None
                }
            })
            .take(num_nodes)
            .collect();

        let best_leaf = &self.dynamics_tree.get(best_leaf_ids[0]).unwrap().value().0;
        let best_cost = (best_leaf.state_cost_function)(&best_leaf.state, &best_leaf.set_point);

        println!("Best leaf cost: {}", best_cost);

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
        )
        .map(|(cost, dynamics_problem, inputs)| {
            let id = self.add_node(dynamics_problem, inputs);
            if cost < self.state_cost_epsilon {
                self.solution_nodes.push(id);
            }
            id
        })
        .collect();

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
    ) -> impl Iterator<Item = (f64, DynamicsProblem, Array1<f64>)> {
        // Set up solver
        let linesearch = HagerZhangLineSearch::new();
        let initial_solver = LBFGS::new(linesearch, 7);

        let middle_input = (input_min_max.0.clone() + input_min_max.1.clone()) / 2.;
        // Run solver
        let res = Executor::new(parent_dynamics.clone(), initial_solver)
            .configure(|state| {
                state
                    .param(middle_input.clone())
                    .max_iters(10)
                    .target_cost(epsilon)
            })
            .run()
            .unwrap();
        let newton_optimized_inputs = res.state.best_param.unwrap();

        let num_inputs = input_min_max.0.len();

        // construct the n+1 points of the nelder-mead simplex, using the newton optimized answer as the +1
        let one_min_simplex: Vec<Array1<f64>> = once(newton_optimized_inputs.clone())
            .chain((0..num_inputs).map(|nonzero_index| {
                Array1::from_iter((0..num_inputs).map(|index| {
                    if index == nonzero_index {
                        input_min_max.0[index]
                    } else {
                        0.
                    }
                }))
            }))
            .collect();
        // let one_max_simplex: Vec<Array1<f64>> = once(newton_optimized_inputs.clone())
        //     .chain((0..num_inputs).map(|nonzero_index| {
        //         Array1::from_iter((0..num_inputs).map(|index| {
        //             if index == nonzero_index {
        //                 input_min_max.1[index]
        //             } else {
        //                 0.
        //             }
        //         }))
        //     }))
        //     .collect();
        // let most_min_simplex: Vec<Array1<f64>> = once(newton_optimized_inputs.clone())
        //     .chain((0..num_inputs).map(|zero_index| {
        //         Array1::from_iter((0..num_inputs).map(|index| {
        //             if index == zero_index {
        //                 0.
        //             } else {
        //                 input_min_max.0[index]
        //             }
        //         }))
        //     }))
        //     .collect();
        // let most_max_simplex: Vec<Array1<f64>> = once(newton_optimized_inputs.clone())
        //     .chain((0..num_inputs).map(|zero_index| {
        //         Array1::from_iter((0..num_inputs).map(|index| {
        //             if index == zero_index {
        //                 0.
        //             } else {
        //                 input_min_max.1[index]
        //             }
        //         }))
        //     }))
        //     .collect();

        let [one_min] = [one_min_simplex].map(|params| {
            let nelder_mead_solver = NelderMead::new(params);

            let res = Executor::new(parent_dynamics.clone(), nelder_mead_solver)
                .configure(|state| state.max_iters(100))
                .run()
                .unwrap();
            res.state.best_param.unwrap()
        });

        // turn our inputs into the next states they create
        let mut next_states =
            [one_min, input_min_max.0, input_min_max.1, middle_input].map(|input| {
                let mut new_dynamics = parent_dynamics.clone();
                new_dynamics.state = new_dynamics.dynamics_function.get_next_state(
                    &new_dynamics.state,
                    input.view(),
                    new_dynamics.dt,
                );
                (
                    (new_dynamics.state_cost_function)(
                        &new_dynamics.state,
                        &new_dynamics.set_point,
                    ),
                    new_dynamics,
                    input.clone(),
                )
            });

        let branch_factor = 4;
        // return the `branch_factor` lowest-cost children
        next_states.sort_by(|(cost1, _, _), (cost2, _, _)| cost1.total_cmp(cost2));
        next_states.into_iter().take(branch_factor)
    }

    fn add_node(&mut self, dynamics: DynamicsProblem, inputs: Array1<f64>) -> NodeId {
        // either overwrite an orphan or add a new
        if let Some(node_id) = self.orphans.pop() {
            *self.dynamics_tree.get_mut(node_id).unwrap().value() = (dynamics, inputs);
            node_id
        } else {
            self.dynamics_tree.orphan((dynamics, inputs)).id()
        }
    }

    // find the `num_nodes` worst leaves and prune them
    fn prune_nodes(&mut self, num_nodes: usize) {
        let mut leaves = self
            .leaves()
            .collect::<Vec<NodeRef<'_, DynamicsNodeType>>>();

        leaves.sort_by(|n1, n2| {
            let dynamics_1 = &n1.value().0;
            let dynamics_2 = &n2.value().0;
            let n1_val = (dynamics_1.state_cost_function)(&dynamics_1.state, &dynamics_1.set_point);
            let n2_val = (dynamics_2.state_cost_function)(&dynamics_2.state, &dynamics_2.set_point);
            n1_val.total_cmp(&n2_val)
        });

        let ids_to_prune: Vec<NodeId> = leaves
            .into_iter()
            .rev()
            .take(num_nodes)
            .map(|node_ref| node_ref.id())
            .collect();

        let worst_leaf = &self.dynamics_tree.get(ids_to_prune[0]).unwrap().value().0;
        let worst_cost = (worst_leaf.state_cost_function)(&worst_leaf.state, &worst_leaf.set_point);

        println!("Pruning worst leaf of cost: {}", worst_cost);

        ids_to_prune.into_iter().for_each(|id_to_prune| {
            self.prune_node(id_to_prune);
        });
    }

    fn prune_node(&mut self, node: NodeId) {
        self.dynamics_tree.get_mut(node).unwrap().detach();
        self.orphans.push(node);
    }

    fn leaves(&self) -> impl Iterator<Item = NodeRef<'_, DynamicsNodeType>> {
        let root_id = self.dynamics_tree.root().id();
        self.dynamics_tree
            .nodes()
            // filter out non-root orphans
            .filter(move |node| node.id() == root_id || node.parent().is_some())
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

        self.calculate_solutions(mpc_problem);

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

    use ndarray::{Array1, ArrayView1, array};

    use crate::{
        dynamics_optimizer::DynamicsOptimizer, dynamics_problem::DynamicsFunction,
        prelude::MPCProblem,
    };

    const DT: f64 = 0.1;
    const LOOKAHEAD: f64 = 500.0;
    const INITIAL_POS: f64 = 1.0;
    const INITIAL_VEL: f64 = 0.0;

    // input is x acceleration, state is (x, vx)
    fn simple_continuous_dynamics(state: &Array1<f64>, input: ArrayView1<f64>) -> Array1<f64> {
        array![state[1], input[0]]
    }

    // simple L2 distance to setpoint
    fn simple_state_cost_function(state: &Array1<f64>, setpoint: &Array1<f64>) -> f64 {
        state
            .iter()
            .zip(setpoint.iter())
            .fold(0., |acc, x| acc + (x.0 - x.1).powi(2))
    }

    // the sum of input^2 over all the inputs is used to discriminate solutions that made it to the goal
    fn simple_dynamics_cost_function(
        _state: &Array1<f64>,
        input: ArrayView1<f64>,
        _setpoint: &Array1<f64>,
    ) -> f64 {
        input[0].powi(2)
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
        let dynamics_optimizer = DynamicsOptimizer::new(array![-1.], array![1.], &mpc_problem);
        (mpc_problem, dynamics_optimizer)
    }

    #[test]
    fn test_grow_node_finds_goal() {
        let optimal_force = 3.14159265358979;
        let goal = array![
            INITIAL_POS + 0.5 * optimal_force * DT.powi(2),
            INITIAL_VEL + optimal_force * DT
        ];
        let (mpc_problem, mut dynamics_optimizer) = get_simple_optimizer(goal);
        dynamics_optimizer.grow_nodes(1);
        dynamics_optimizer.calculate_solutions(&mpc_problem);

        // ensure that we found at least one solution
        assert!(dynamics_optimizer.solutions.len() > 0);
    }

    #[test]
    fn test_node_logic() {
        let goal = array![0., 0.];
        let (_mpc_problem, mut dynamics_optimizer) = get_simple_optimizer(goal);
        while dynamics_optimizer.dynamics_tree.nodes().len() < 50 {
            dynamics_optimizer.grow_nodes(10);
        }

        let num_leaves = dynamics_optimizer.leaves().count();
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
    fn test_find_goal_moderate() {
        let goal = array![0., 0.];
        let (mpc_problem, mut dynamics_optimizer) = get_simple_optimizer(goal.clone());
        let mut num_iter = 0;
        let max_iter = 10000;
        let target_count = 1000;
        while num_iter < max_iter && dynamics_optimizer.solution_nodes.len() == 0 {
            println!("iteration {}", num_iter + 1);
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

            println!("Number of leaves: {}", dynamics_optimizer.leaves().count());

            println!(
                "Total number: {}",
                dynamics_optimizer.dynamics_tree.nodes().count() - dynamics_optimizer.orphans.len()
            );
            num_iter += 1;
            println!("");
        }
        dynamics_optimizer.calculate_solutions(&mpc_problem);

        // ensure that we found at least one solution
        assert!(dynamics_optimizer.solutions.len() > 0);

        let solution_trajectory =
            mpc_problem.calculate_trajectory(&dynamics_optimizer.solutions[0].0);

        let last_point = solution_trajectory.last().unwrap();

        println!("last_point: {}", last_point);

        assert!(last_point.relative_eq(&goal, 0.1, 0.1));
    }
}
