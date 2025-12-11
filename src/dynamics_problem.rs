use core::f64;
use std::{sync::Arc, time::Duration};

use argmin::core::{CostFunction, Hessian, Jacobian};
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

/// f(state, input, setpoint) -> cost
pub type DynamicsCostFunction =
    dyn Fn(&Array1<f64>, ArrayView1<f64>, &Array1<f64>) -> f64 + Send + Sync;

#[derive(Clone)]
pub struct DynamicsProblem {
    pub dynamics_function: DynamicsFunction,

    /// f(state, input, setpoint) -> cost
    pub dynamics_cost_function: Arc<DynamicsCostFunction>,
    pub state: Array1<f64>,
    pub set_point: Arc<Array1<f64>>,
    pub dt: Duration,
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
        let dynamics_closure = |inputs: &Self::Param| {
            Ok(self
                .dynamics_function
                .get_next_state(&self.state, inputs.view(), self.dt))
        };
        let h_forward = ndarr::forward_hessian(&dynamics_closure);
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

impl CostFunction for DynamicsProblem {
    type Param = Array1<f64>;

    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin_math::Error> {
        let next_state = self
            .dynamics_function
            .get_next_state(&self.state, param.view(), self.dt);
        Ok((self.dynamics_cost_function)(
            &next_state,
            param.view(),
            &self.set_point,
        ))
    }

    fn bulk_cost<P>(&self, params: &[P]) -> Result<Vec<Self::Output>, argmin_math::Error>
    where
        P: std::borrow::Borrow<Self::Param> + argmin::core::SyncAlias,
        Self::Output: argmin::core::SendAlias,
        Self: argmin::core::SyncAlias,
    {
        params.par_iter().map(|p| self.cost(p.borrow())).collect()
    }
}
