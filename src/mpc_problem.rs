use core::f64;
use std::time::Duration;

use argmin::core::CostFunction;
use ndarray::{
    Array, Array1, ArrayView1, ArrayView2,
    parallel::prelude::{IntoParallelRefIterator, ParallelIterator},
};

// the MPCController can take a function that returns the derivative and integrate it internally as needed
// or a function that takes a state and a dt and returns the next state
// if you want fine control of the integration or have dynamics that cannot be captured by a derivative, then you should use Discrete
#[allow(clippy::type_complexity)]
pub enum DynamicsFunction<const STATE_SIZE: usize, const INPUT_SIZE: usize> {
    // f(x, u) -> xdot
    Continuous(
        Box<dyn Fn(&[f64; STATE_SIZE], &ArrayView1<f64>) -> [f64; STATE_SIZE] + Send + Sync>,
    ),
    // f(x_k, u_k, dt) -> x_{k+1}
    Discrete(
        Box<
            dyn Fn(&[f64; STATE_SIZE], &ArrayView1<f64>, Duration) -> [f64; STATE_SIZE]
                + Send
                + Sync,
        >,
    ),
}

#[allow(clippy::type_complexity)]
pub struct MPCProblem<const STATE_SIZE: usize, const INPUT_SIZE: usize> {
    pub(crate) setpoint: [f64; STATE_SIZE],
    pub(crate) current_state: [f64; STATE_SIZE],
    pub(crate) sample_period: Duration,
    pub(crate) lookahead_duration: Duration,
    pub(crate) dynamics_function: DynamicsFunction<STATE_SIZE, INPUT_SIZE>,

    // MPC controller must have at least one of the below cost functions
    // optional state/input cost function, J(x, u) -> f64
    pub(crate) state_cost: Option<
        Box<dyn Fn(&[f64; STATE_SIZE], &[f64; STATE_SIZE], &ArrayView1<f64>) -> f64 + Send + Sync>,
    >,

    // optional terminal cost function, J(x, x_setpoint) -> f64
    pub(crate) terminal_cost:
        Option<Box<dyn Fn(&[f64; STATE_SIZE], &[f64; STATE_SIZE]) -> f64 + Send + Sync>>,
}

impl<const STATE_SIZE: usize, const INPUT_SIZE: usize> CostFunction
    for MPCProblem<STATE_SIZE, INPUT_SIZE>
{
    type Param = Array1<f64>;

    type Output = f64;

    fn cost(&self, x: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        let x_view = x.view();
        let trajectory = self.calculate_trajectory(&x_view);
        let trajectory_cost = self.calculate_trajectory_cost(&trajectory, &x_view);
        Ok(trajectory_cost)
    }

    // only particle swarm optimization in argmin currently supports this
    fn bulk_cost<P>(&self, params: &[P]) -> Result<Vec<Self::Output>, argmin_math::Error>
    where
        P: std::borrow::Borrow<Self::Param> + argmin::core::SyncAlias,
        Self::Output: argmin::core::SendAlias,
        Self: argmin::core::SyncAlias,
    {
        params.par_iter().map(|p| self.cost(p.borrow())).collect()
    }
}

impl<const STATE_SIZE: usize, const INPUT_SIZE: usize> MPCProblem<STATE_SIZE, INPUT_SIZE> {
    pub(crate) fn integrate_dynamics(
        &self,
        current_state: &[f64; STATE_SIZE],
        derivatives: &[f64; STATE_SIZE],
    ) -> [f64; STATE_SIZE] {
        // just do a simple Euler integration for now
        let mut next_state = [0_f64; STATE_SIZE];
        for i in 0..STATE_SIZE {
            next_state[i] = current_state[i] + self.sample_period.as_secs_f64() * derivatives[i];
        }
        next_state
    }

    // maps the inputs to the trajectory they would result in
    // useful for visualization
    pub fn calculate_trajectory(&self, inputs: &ArrayView1<f64>) -> Array1<[f64; STATE_SIZE]> {
        let mut current_state = self.current_state;
        let input_chunks = Self::inputs_to_chunks(inputs);
        let trajectory_iter = input_chunks.rows().into_iter().map(|input| {
            let next_state = match &self.dynamics_function {
                DynamicsFunction::Continuous(continuous_dynamics_function) => {
                    let derivatives = continuous_dynamics_function(&current_state, &input);
                    self.integrate_dynamics(&current_state, &derivatives)
                }
                DynamicsFunction::Discrete(discrete_dynamics_function) => {
                    discrete_dynamics_function(&current_state, &input, self.sample_period)
                }
            };
            current_state = next_state;
            current_state
        });

        Array::from_iter(trajectory_iter)
    }

    // changes the inputs from a 1D array to a 2D array where every row is a step and every column is an input
    pub(crate) fn inputs_to_chunks<'a>(inputs: &ArrayView1<'a, f64>) -> ArrayView2<'a, f64> {
        assert!(
            inputs.len().is_multiple_of(INPUT_SIZE),
            "inputs must be of length N * INPUT_SIZE where N is some nonnegative integer\nactual {} != N * {}",
            inputs.len(),
            INPUT_SIZE
        );

        let n_steps = inputs.len() / INPUT_SIZE;

        unsafe { ArrayView2::from_shape_ptr((n_steps, INPUT_SIZE), inputs.as_ptr()) }
    }

    // calculates the cost of a given trajectory
    pub(crate) fn calculate_trajectory_cost(
        &self,
        trajectory: &Array1<[f64; STATE_SIZE]>,
        inputs: &ArrayView1<f64>,
    ) -> f64 {
        let input_chunks = Self::inputs_to_chunks(inputs);
        let state_cost = if let Some(state_cost_function) = &self.state_cost {
            trajectory.iter().zip(input_chunks.rows()).fold(
                0.,
                |accumulated_cost, (state, input)| {
                    accumulated_cost + state_cost_function(state, &self.setpoint, &input)
                },
            )
        } else {
            0.
        };
        // terminal cost
        state_cost
            + trajectory
                .last()
                .map(|terminal_state| {
                    if let Some(terminal_cost_function) = &self.terminal_cost {
                        terminal_cost_function(terminal_state, &self.setpoint)
                    } else {
                        0.
                    }
                })
                .unwrap_or(0.)
    }

    // set the duration over which the control problem will be optimized
    pub fn set_lookahead(&mut self, lookahead_duration: Duration) {
        self.lookahead_duration = lookahead_duration;
    }

    // set the timestep of the controller
    // the lookahead duration divided by the sample period (rounded up) gives the number of control inputs returned
    pub fn set_sample_period(&mut self, sample_period: Duration) {
        self.sample_period = sample_period;
    }

    pub fn set_setpoint(&mut self, setpoint: [f64; STATE_SIZE]) {
        self.setpoint = setpoint;
    }

    // this function provides a sample parameter vector for use with the Nelder Mead optimizer
    pub fn parameter_vector(&self, max_input: f64) -> Vec<Array1<f64>> {
        // TODO add warm start
        let num_steps = (self.lookahead_duration.as_secs_f64() / self.sample_period.as_secs_f64())
            .ceil() as usize;
        let dimension = INPUT_SIZE * num_steps;

        let mut simplex = Vec::with_capacity(dimension + 1);
        let zero_vec = Array1::<f64>::zeros(dimension);
        simplex.push(zero_vec.clone());

        // have each simplex vector simply be a unit in a given direction
        for i in 0..dimension {
            let mut next_vec = zero_vec.clone();
            next_vec[i] = max_input;
            simplex.push(next_vec);
        }
        simplex
    }
}
