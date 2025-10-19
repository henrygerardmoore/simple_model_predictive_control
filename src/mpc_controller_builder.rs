use core::f64;
use std::time::Duration;

use crate::mpc_controller::{DynamicsFunction, MPCController};

#[allow(clippy::type_complexity)]
#[derive(Default)]
pub struct MPCControllerBuilder<const STATE_SIZE: usize, const INPUT_SIZE: usize> {
    setpoint: Option<[f64; STATE_SIZE]>,
    current_state: Option<[f64; STATE_SIZE]>,
    sample_period: Option<Duration>,
    lookahead_duration: Option<Duration>,
    dynamics_function: Option<DynamicsFunction<STATE_SIZE, INPUT_SIZE>>,
    state_cost: Option<Box<dyn Fn(&[f64; STATE_SIZE], &[f64; INPUT_SIZE]) -> f64 + Send + Sync>>,
    terminal_cost: Option<Box<dyn Fn(&[f64; STATE_SIZE], &[f64; STATE_SIZE]) -> f64 + Send + Sync>>,
    constraints: Vec<Box<dyn Fn(&[f64; STATE_SIZE]) -> f64 + Send + Sync>>,
    input_bounds: Option<[(f64, f64); INPUT_SIZE]>,
}

impl<const STATE_SIZE: usize, const INPUT_SIZE: usize>
    MPCControllerBuilder<STATE_SIZE, INPUT_SIZE>
{
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
            input_bounds: None,
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
        F: Fn(&[f64; STATE_SIZE], &[f64; INPUT_SIZE]) -> f64 + Send + Sync + 'static,
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

    pub fn input_bounds(mut self, input_bounds: [(f64, f64); INPUT_SIZE]) -> Self {
        self.input_bounds = Some(input_bounds);
        self
    }

    pub fn build(self) -> Result<MPCController<STATE_SIZE, INPUT_SIZE>, String> {
        if self.state_cost.is_none() && self.terminal_cost.is_none() {
            return Err("at least one of state_cost or terminal_cost must be provided".to_string());
        }
        Ok(MPCController {
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
            constraints: self.constraints,
            input_bounds: self
                .input_bounds
                .unwrap_or([(-f64::INFINITY, f64::INFINITY); INPUT_SIZE]),
        })
    }
}
