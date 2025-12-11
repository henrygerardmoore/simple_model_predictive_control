use core::f64;
use std::{sync::Arc, time::Duration};

use argmin::core::{Gradient, Hessian, Jacobian};
use argmin_math::ArgminL2Norm;
use finitediff::ndarr;
use ndarray::{
    Array1, Array2, ArrayView1,
    parallel::prelude::{IntoParallelRefIterator, ParallelIterator},
};

#[allow(
    clippy::type_complexity,
    reason = "dynamics function closures have complex types"
)]
#[derive(Clone)]
pub enum DynamicsFunction {
    /// f(x, u) -> xdot
    Continuous(Arc<dyn Fn(&Array1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync>),
    /// f(x_k, u_k, dt) -> x_{k+1}
    Discrete(Arc<dyn Fn(&Array1<f64>, ArrayView1<f64>, Duration) -> Array1<f64> + Send + Sync>),
}

impl DynamicsFunction {
    pub fn get_next_state(
        &self,
        state: &Array1<f64>,
        input: ArrayView1<f64>,
        dt: Duration,
    ) -> Array1<f64> {
        match self {
            DynamicsFunction::Continuous(continuous_function) => {
                let deriv = continuous_function(state, input);
                state + dt.as_secs_f64() * deriv
            }
            DynamicsFunction::Discrete(discrete_function) => discrete_function(state, input, dt),
        }
    }
}

/// f(state, setpoint) -> cost
pub type StateCostFunction = dyn Fn(&Array1<f64>, &Array1<f64>) -> f64 + Send + Sync;

/// f(state, input, setpoint) -> cost
pub type DynamicsCostFunction =
    dyn Fn(&Array1<f64>, ArrayView1<f64>, &Array1<f64>) -> f64 + Send + Sync;

#[derive(Clone)]
pub struct DynamicsProblem {
    pub dynamics_function: DynamicsFunction,

    /// f(state, setpoint) -> cost
    pub state_cost_function: Arc<StateCostFunction>,
    pub state: Array1<f64>,
    pub set_point: Arc<Array1<f64>>,
    pub dt: Duration,
}

// gradient of distance from setpoint wrt
impl Gradient for DynamicsProblem {
    type Param = Array1<f64>;

    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin_math::Error> {
        let distance_from_setpoint = |inputs: &Self::Param| {
            let next_state =
                self.dynamics_function
                    .get_next_state(&self.state, inputs.view(), self.dt);
            Ok((next_state - (*self.set_point).clone()).l2_norm())
        };
        let grad_forward = ndarr::forward_diff(&distance_from_setpoint);
        grad_forward(param)
    }
}

impl Jacobian for DynamicsProblem {
    type Param = Array1<f64>;

    type Jacobian = Array2<f64>;

    fn jacobian(&self, param: &Self::Param) -> Result<Self::Jacobian, argmin_math::Error> {
        let dynamics_closure = |inputs: &Self::Param| {
            Ok(self
                .dynamics_function
                .get_next_state(&self.state, inputs.view(), self.dt))
        };
        let j_forward = ndarr::forward_jacobian(&dynamics_closure);
        j_forward(param)
    }

    fn bulk_jacobian<P>(&self, params: &[P]) -> Result<Vec<Self::Jacobian>, argmin_math::Error>
    where
        P: std::borrow::Borrow<Self::Param> + argmin::core::SyncAlias,
        Self::Jacobian: argmin::core::SendAlias,
        Self: argmin::core::SyncAlias,
    {
        params
            .par_iter()
            .map(|p| self.jacobian(p.borrow()))
            .collect()
    }
}

impl Hessian for DynamicsProblem {
    type Param = Array1<f64>;

    type Hessian = Array2<f64>;

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, argmin_math::Error> {
        let gradient_closure = |inputs: &Self::Param| self.gradient(inputs);
        let h_forward = ndarr::forward_hessian(&gradient_closure);
        h_forward(param)
    }

    fn bulk_hessian<P>(&self, params: &[P]) -> Result<Vec<Self::Hessian>, argmin_math::Error>
    where
        P: std::borrow::Borrow<Self::Param> + argmin::core::SyncAlias,
        Self::Hessian: argmin::core::SendAlias,
        Self: argmin::core::SyncAlias,
    {
        params
            .par_iter()
            .map(|p| self.hessian(p.borrow()))
            .collect()
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn test_gradient_continuous() {}
    #[test]
    fn test_jacobian_continuous() {}
    #[test]
    fn test_hessian_continuous() {}
    #[test]
    fn test_dynamics_optimization_continuous() {}
    #[test]
    fn test_gradient_discrete() {}
    #[test]
    fn test_jacobian_discrete() {}
    #[test]
    fn test_hessian_discrete() {}
    #[test]
    fn test_dynamics_optimization_discrete() {}
}
