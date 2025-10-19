use std::{
    iter::{once, repeat_n},
    time::Duration,
};

use egobox_ego::{
    CorrelationSpec, Cstr, EgorBuilder, EgorState, GroupFunc, InfillOptimizer, InfillStrategy,
    RegressionSpec,
};
use ndarray::{Array, Array2, ArrayView2, Ix1, Zip, array};

// WIP
#[allow(dead_code, clippy::type_complexity)]
pub struct MPCController<const STATE_SIZE: usize, const INPUT_SIZE: usize> {
    setpoint: [f64; STATE_SIZE],
    sample_period: Duration,
    lookahead_duration: Duration,
    // f(x, u) -> xdot
    dynamics_function:
        Box<dyn Fn(&[f64; STATE_SIZE], &[f64; INPUT_SIZE]) -> [f64; INPUT_SIZE] + Send>,

    // MPC controller must have at least one of the below cost functions
    // optional state cost function, J(x, u) -> f64
    state_cost: Option<Box<dyn Fn(&[f64; STATE_SIZE], &[f64; INPUT_SIZE]) -> f64 + Send>>,

    // optional terminal cost function, J(x) -> f64
    terminal_cost: Option<Box<dyn Fn(&[f64; STATE_SIZE]) -> f64 + Send>>,

    // can be an empty vector if this is an unconstrained problem
    // WIP: use this
    constraints: Vec<Cstr>,

    // min and max value of each control input
    input_bounds: [(f64, f64); INPUT_SIZE],
}

// WIP
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
        // fn cost(&self, x: &ArrayView2<f64>) -> Array2<f64> {
        |x: &ArrayView2<f64>| -> Array2<f64> {
            let mut y: Array2<f64> = Array2::zeros((x.nrows(), 1));
            Zip::from(y.rows_mut())
                .and(x.rows())
                .par_for_each(|mut yi, xi| yi.assign(&array![xi.sum()]));
            y
        }
    }

    pub fn set_lookahead(&mut self, lookahead_duration: Duration) {
        self.lookahead_duration = lookahead_duration;
    }

    pub fn set_sample_period(&mut self, sample_period: Duration) {
        self.sample_period = sample_period;
    }
}
