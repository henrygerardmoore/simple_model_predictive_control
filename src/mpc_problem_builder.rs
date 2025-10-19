use core::f64;
use std::time::Duration;

use ndarray::ArrayView1;

use crate::mpc_problem::{DynamicsFunction, MPCProblem};

#[allow(clippy::type_complexity)]
#[derive(Default)]
pub struct MPCProblemBuilder<const STATE_SIZE: usize, const INPUT_SIZE: usize> {
    setpoint: Option<[f64; STATE_SIZE]>,
    current_state: Option<[f64; STATE_SIZE]>,
    sample_period: Option<Duration>,
    lookahead_duration: Option<Duration>,
    dynamics_function: Option<DynamicsFunction<STATE_SIZE, INPUT_SIZE>>,
    state_cost: Option<
        Box<dyn Fn(&[f64; STATE_SIZE], &[f64; STATE_SIZE], &ArrayView1<f64>) -> f64 + Send + Sync>,
    >,
    terminal_cost: Option<Box<dyn Fn(&[f64; STATE_SIZE], &[f64; STATE_SIZE]) -> f64 + Send + Sync>>,
    constraints: Vec<Box<dyn Fn(&[f64; STATE_SIZE]) -> f64 + Send + Sync>>,
}

impl<const STATE_SIZE: usize, const INPUT_SIZE: usize> MPCProblemBuilder<STATE_SIZE, INPUT_SIZE> {
    pub fn new() -> Self {
        Self {
            setpoint: None,
            current_state: None,
            sample_period: None,
            lookahead_duration: None,
            dynamics_function: None,
            state_cost: None,
            terminal_cost: None,
            constraints: Vec::new(),
        }
    }

    pub fn setpoint(mut self, setpoint: [f64; STATE_SIZE]) -> Self {
        self.setpoint = Some(setpoint);
        self
    }

    pub fn initial_conditions(mut self, initial_conditions: [f64; STATE_SIZE]) -> Self {
        self.current_state = Some(initial_conditions);
        self
    }

    pub fn sample_period(mut self, sample_period: Duration) -> Self {
        self.sample_period = Some(sample_period);
        self
    }

    pub fn lookahead_duration(mut self, lookahead_duration: Duration) -> Self {
        self.lookahead_duration = Some(lookahead_duration);
        self
    }

    pub fn dynamics_function(
        mut self,
        dynamics_function: DynamicsFunction<STATE_SIZE, INPUT_SIZE>,
    ) -> Self {
        self.dynamics_function = Some(dynamics_function);
        self
    }

    pub fn state_cost<F>(mut self, state_cost: F) -> Self
    where
        F: Fn(&[f64; STATE_SIZE], &[f64; STATE_SIZE], &ArrayView1<f64>) -> f64
            + Send
            + Sync
            + 'static,
    {
        self.state_cost = Some(Box::new(state_cost));
        self
    }

    pub fn terminal_cost<F>(mut self, terminal_cost: F) -> Self
    where
        F: Fn(&[f64; STATE_SIZE], &[f64; STATE_SIZE]) -> f64 + Send + Sync + 'static,
    {
        self.terminal_cost = Some(Box::new(terminal_cost));
        self
    }

    pub fn add_constraint<F>(mut self, constraint: F) -> Self
    where
        F: Fn(&[f64; STATE_SIZE]) -> f64 + Send + Sync + 'static,
    {
        self.constraints.push(Box::new(constraint));
        self
    }

    pub fn build(self) -> Result<MPCProblem<STATE_SIZE, INPUT_SIZE>, String> {
        if self.state_cost.is_none() && self.terminal_cost.is_none() {
            return Err("at least one of state_cost or terminal_cost must be provided".to_string());
        }
        Ok(MPCProblem {
            setpoint: self.setpoint.ok_or("setpoint is required")?,
            current_state: self.current_state.ok_or("current_state is required")?,
            sample_period: self.sample_period.ok_or("sample_period is required")?,
            lookahead_duration: self
                .lookahead_duration
                .ok_or("lookahead_duration is required")?,
            dynamics_function: self
                .dynamics_function
                .ok_or("dynamics_function is required")?,
            state_cost: self.state_cost,
            terminal_cost: self.terminal_cost,
        })
    }
}
