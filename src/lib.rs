use core::f64;
use std::{
    iter::{once, repeat_n},
    time::Duration,
};

use egobox_ego::{
    CorrelationSpec, EgorBuilder, EgorState, GroupFunc, InfillOptimizer, InfillStrategy,
    RegressionSpec,
};
use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Ix1, Zip};

// the MPCController can take a function that returns the derivative and integrate it internally as needed
// or a function that takes a state and a dt and returns the next state
// if you want fine control of the integration or have dynamics that cannot be captured by a derivative, then you should use Discrete
#[allow(clippy::type_complexity)]
pub enum DynamicsFunction<const STATE_SIZE: usize, const INPUT_SIZE: usize> {
    // f(x, u) -> xdot
    Continuous(
        Box<dyn Fn(&[f64; STATE_SIZE], &[f64; INPUT_SIZE]) -> [f64; STATE_SIZE] + Send + Sync>,
    ),
    // f(x_k, u_k, dt) -> x_{k+1}
    Discrete(
        Box<
            dyn Fn(&[f64; STATE_SIZE], &[f64; INPUT_SIZE], Duration) -> [f64; STATE_SIZE]
                + Send
                + Sync,
        >,
    ),
}

#[allow(clippy::type_complexity)]
pub struct MPCController<const STATE_SIZE: usize, const INPUT_SIZE: usize> {
    setpoint: [f64; STATE_SIZE],
    current_state: [f64; STATE_SIZE],
    sample_period: Duration,
    lookahead_duration: Duration,
    dynamics_function: DynamicsFunction<STATE_SIZE, INPUT_SIZE>,

    // MPC controller must have at least one of the below cost functions
    // optional state/input cost function, J(x, u) -> f64
    state_cost: Option<Box<dyn Fn(&[f64; STATE_SIZE], &[f64; INPUT_SIZE]) -> f64 + Send + Sync>>,

    // optional terminal cost function, J(x, x_setpoint) -> f64
    terminal_cost: Option<Box<dyn Fn(&[f64; STATE_SIZE], &[f64; STATE_SIZE]) -> f64 + Send + Sync>>,

    // can be an empty vector if this is an unconstrained problem
    // C(trajectory) -> c, where c is negative if the constraint is violated
    constraints: Vec<Box<dyn Fn(&[f64; STATE_SIZE]) -> f64 + Send + Sync>>,

    // min and max value of each control input
    input_bounds: [(f64, f64); INPUT_SIZE],
}

// will only be read by users and in tests
#[allow(dead_code)]
pub struct MPCOutput {
    control_inputs: Array<f64, Ix1>,
    expected_cost: Array<f64, Ix1>,
    state: EgorState<f64>,
}

// Fn(&ArrayView2<f64>) -> Array2<f64>
impl<const STATE_SIZE: usize, const INPUT_SIZE: usize> MPCController<STATE_SIZE, INPUT_SIZE> {
    pub fn optimize(&mut self) -> Result<MPCOutput, String> {
        let number_steps = self
            .lookahead_duration
            .div_duration_f32(self.sample_period)
            .ceil();
        if number_steps < 1. {
            return Err(format!(
                "lookahead {} divided by sample_period {} results in {} MPC steps",
                self.lookahead_duration.as_secs_f32(),
                self.sample_period.as_secs_f32(),
                number_steps
            ));
        }
        let number_steps = number_steps as usize;
        let xlimits = Array::from_shape_vec(
            (number_steps * INPUT_SIZE, 2),
            // xlimits should be a vec [lower, upper, lower, upper ...]
            // first 2 * INPUT_SIZE elements are step 1, next 2 * INPUT_SIZE are step 2, etc.
            // for a total of 2 * INPUT_SIZE * number_steps where each step is the same
            repeat_n(
                self.input_bounds
                    .iter()
                    .flat_map(|(lower, upper)| once(*lower).chain(once(*upper)))
                    .collect::<Vec<f64>>(),
                number_steps,
            )
            .flatten()
            .collect::<Vec<f64>>(),
        )
        .unwrap();
        let res = EgorBuilder::optimize(self.cost_function())
            .configure(|config| {
                config
                    .configure_gp(|gp| {
                        gp.regression_spec(RegressionSpec::CONSTANT)
                            .correlation_spec(CorrelationSpec::ABSOLUTEEXPONENTIAL)
                    })
                    .infill_strategy(InfillStrategy::WB2S)
                    .infill_optimizer(InfillOptimizer::Slsqp)
                    .n_start(50)
                    .max_iters(300)
            })
            .min_within(&xlimits)
            .run()
            .expect("Minimize failure");
        Ok(MPCOutput {
            control_inputs: res.x_opt,
            expected_cost: res.y_opt,
            state: res.state,
        })
    }

    fn cost_function(&self) -> impl GroupFunc {
        |x: &ArrayView2<f64>| -> Array2<f64> {
            let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
            Zip::from(y.rows_mut())
                .and(x.rows())
                .par_for_each(|mut yi, xi| {
                    let trajectory = self.calculate_trajectory(&xi);
                    let trajectory_cost = self.calculate_trajectory_cost(&trajectory, &xi);
                    let constraints = self.calculate_trajectory_constraints(&trajectory);
                    let cost_iter = std::iter::once(trajectory_cost).chain(constraints);
                    let output = Array::from_iter(cost_iter);
                    yi.assign(&output)
                });
            y
        }
    }

    fn integrate_dynamics(
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
    fn calculate_trajectory(&self, inputs: &ArrayView1<f64>) -> Array1<[f64; STATE_SIZE]> {
        let mut current_state = self.current_state;
        let trajectory_iter = Self::inputs_to_chunks(inputs).map(|input| {
            let next_state = match &self.dynamics_function {
                DynamicsFunction::Continuous(continuous_dynamics_function) => {
                    let derivatives = continuous_dynamics_function(&current_state, input);
                    self.integrate_dynamics(&current_state, &derivatives)
                }
                DynamicsFunction::Discrete(discrete_dynamics_function) => {
                    discrete_dynamics_function(&current_state, input, self.sample_period)
                }
            };
            current_state = next_state;
            current_state
        });

        Array::from_iter(trajectory_iter)
    }

    fn inputs_to_chunks<'a>(
        inputs: &'a ArrayView1<f64>,
    ) -> impl Iterator<Item = &'a [f64; INPUT_SIZE]> {
        inputs
            .as_slice()
            .unwrap()
            .chunks_exact(INPUT_SIZE)
            .map(|chunk| TryInto::<&[f64; INPUT_SIZE]>::try_into(chunk).unwrap())
    }

    fn calculate_trajectory_cost(
        &self,
        trajectory: &Array1<[f64; STATE_SIZE]>,
        inputs: &ArrayView1<f64>,
    ) -> f64 {
        let state_cost = if let Some(state_cost_function) = &self.state_cost {
            trajectory.iter().zip(Self::inputs_to_chunks(inputs)).fold(
                0.,
                |accumulated_cost, (state, input)| {
                    accumulated_cost + state_cost_function(state, input)
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

    fn calculate_trajectory_constraints(
        &self,
        trajectory: &Array1<[f64; STATE_SIZE]>,
    ) -> impl Iterator<Item = f64> {
        self.constraints.iter().map(|constraint_function| {
            // map each constraint to the minimum value they take on during the trajectory
            trajectory
                .iter()
                .map(constraint_function)
                .fold(f64::INFINITY, |acc, x| acc.min(x))
        })
    }

    pub fn set_lookahead(&mut self, lookahead_duration: Duration) {
        self.lookahead_duration = lookahead_duration;
    }

    pub fn set_sample_period(&mut self, sample_period: Duration) {
        self.sample_period = sample_period;
    }

    pub fn set_setpoint(&mut self, setpoint: [f64; STATE_SIZE]) {
        self.setpoint = setpoint;
    }
}

// WIP add unit tests
// WIP add integration tests
