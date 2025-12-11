use core::f64;
use std::{sync::Arc, time::Duration};

use argmin::core::{CostFunction, Gradient, Hessian, Jacobian};
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

/// RK4 integration
#[allow(
    clippy::type_complexity,
    reason = "derivative function has a necessarily complex type"
)]
pub fn rk4_step(
    state: &Array1<f64>,
    input: ArrayView1<f64>,
    dt: f64,
    derivative_function: &Arc<dyn Fn(&Array1<f64>, ArrayView1<f64>) -> Array1<f64> + Send + Sync>,
) -> Array1<f64> {
    let k1 = derivative_function(state, input);
    let s2 = state + (&k1 * (dt / 2.0));
    let k2 = derivative_function(&s2, input);
    let s3 = state + (&k2 * (dt / 2.0));
    let k3 = derivative_function(&s3, input);
    let s4 = state + (&k3 * dt);
    let k4 = derivative_function(&s4, input);

    state + (k1 + (k2 * 2.0) + (k3 * 2.0) + k4) * (dt / 6.0)
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
                rk4_step(state, input, dt.as_secs_f64(), continuous_function)
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

impl CostFunction for DynamicsProblem {
    type Param = Array1<f64>;

    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin_math::Error> {
        let next_state = self
            .dynamics_function
            .get_next_state(&self.state, param.view(), self.dt);
        Ok((next_state - (*self.set_point).clone()).l2_norm())
    }
}

// gradient of distance from setpoint wrt input
impl Gradient for DynamicsProblem {
    type Param = Array1<f64>;

    type Gradient = Array1<f64>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin_math::Error> {
        let distance_from_setpoint = |inputs: &Self::Param| self.cost(inputs);
        let grad_forward = ndarr::forward_diff(&distance_from_setpoint);
        grad_forward(param)
    }
}

// jacobian of dynamics function
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

// hessian of distance from setpoint wrt input
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
    mod continuous {
        use std::{sync::Arc, time::Duration};

        use approx::assert_relative_eq;
        use argmin::{
            core::{CostFunction, Executor, Gradient, Hessian, Jacobian},
            solver::{linesearch::HagerZhangLineSearch, quasinewton::LBFGS},
        };
        use argmin_math::ArgminL2Norm;
        use finitediff::ndarr;
        use ndarray::{Array1, ArrayView1, array};

        use crate::dynamics_problem::{DynamicsFunction, DynamicsProblem};

        const DT: f64 = 0.0001;
        const INITIAL_POS: f64 = 1.0;
        const INITIAL_VEL: f64 = 0.0;
        const GOAL_POS: f64 = 0.0;
        const GOAL_VEL: f64 = 0.0;

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

        // we start at INITIAL_POS, INITIAL_VEL and want to go to GOAL_POS, GOAL_VEL
        fn simple_continuous_dynamics_problem() -> DynamicsProblem {
            DynamicsProblem {
                dynamics_function: DynamicsFunction::Continuous(Arc::new(
                    &simple_continuous_dynamics,
                )),
                state_cost_function: Arc::new(&simple_state_cost_function),
                state: array![INITIAL_POS, INITIAL_VEL],
                set_point: Arc::new(array![GOAL_POS, GOAL_VEL]),
                dt: Duration::from_secs_f64(DT),
            }
        }

        #[test]
        fn test_dynamics_continuous() {
            let problem = simple_continuous_dynamics_problem();
            let input_ax = -2.;
            let next_state = problem.dynamics_function.get_next_state(
                &problem.state,
                array![input_ax].view(),
                problem.dt,
            );

            let next_state_expected = array![
                INITIAL_POS + 0.5 * input_ax * DT.powi(2),
                INITIAL_VEL + DT * input_ax
            ];

            // generally you shouldn't expect strict equality, but rk4 can exactly integrate our simple system
            assert_eq!(next_state, next_state_expected);
        }

        #[test]
        fn test_dynamics_cost_continuous() {
            let problem = simple_continuous_dynamics_problem();
            let input_ax = -1.;
            let next_state_expected = array![
                INITIAL_POS + 0.5 * input_ax * DT.powi(2),
                INITIAL_VEL + input_ax * DT
            ];
            let cost_expected = next_state_expected.l2_norm();
            let cost = problem.cost(&array![-1.]).expect("Cost calculation failed");

            assert_eq!(cost, cost_expected);
        }

        #[test]
        fn test_gradient_continuous() {
            let problem = simple_continuous_dynamics_problem();

            let input_ax = 1.;

            let gradient = problem
                .gradient(&array![input_ax])
                .expect("Gradient calculation failed");

            let du = 1e-4;
            let dcost = problem.cost(&array![input_ax + du]).unwrap()
                - problem.cost(&array![input_ax]).unwrap();
            let gradient_expected = array![dcost / du];

            // since our input size is 1
            assert_eq!(gradient.len(), 1);

            // the forward gradient approximation isn't perfect
            assert_relative_eq!(gradient, gradient_expected, epsilon = 1e-6);
        }

        #[test]
        fn test_jacobian_continuous() {
            let problem = simple_continuous_dynamics_problem();
            let input_ax = 1.;
            let jacobian = problem
                .jacobian(&array![input_ax])
                .expect("Jacobian calculation failed");

            // jacobian will in general be of size STATE_SIZE x INPUT_SIZE
            assert_eq!(jacobian.dim(), (2, 1));

            let jacobian_state = problem.dynamics_function.get_next_state(
                &problem.state,
                array![input_ax].view(),
                problem.dt,
            );

            // calculate expected with finite difference
            // we don't divide by dt since our function doesn't either
            let jacobian_expected = (array![[jacobian_state[0]], [jacobian_state[1]]]
                - array![[problem.state[0]], [problem.state[1]]]);

            assert_relative_eq!(jacobian, jacobian_expected, epsilon = 1e-6);
        }
        #[test]
        fn test_hessian_continuous() {
            let problem = simple_continuous_dynamics_problem();
            let input_ax = 1.;
            let input = array![input_ax];
            let hessian = problem.hessian(&input).expect("Hessian calculation failed");

            assert_eq!(hessian.dim(), (1, 1));
            let gradient_closure = |input: &Array1<f64>| problem.gradient(input);
            let hessian_expected_f = ndarr::forward_jacobian(&gradient_closure);
            let hessian_expected = hessian_expected_f(&input).unwrap();

            assert_relative_eq!(hessian, hessian_expected, epsilon = 1e-6);
        }

        #[test]
        fn test_dynamics_optimization_continuous() {
            let mut problem = simple_continuous_dynamics_problem();
            // make the set point something achievable in one step
            let optimal_ax = 3.14159265358979;
            problem.set_point = Arc::new(array![
                INITIAL_POS + 0.5 * optimal_ax * DT.powi(2),
                INITIAL_VEL + optimal_ax * DT
            ]);
            let input_ax = 1.;
            let input = array![input_ax];
            let linesearch = HagerZhangLineSearch::new();
            let solver = LBFGS::new(linesearch, 7);
            let res = Executor::new(problem.clone(), solver)
                .configure(|state| state.param(input).max_iters(10))
                .run()
                .unwrap();
            let optimized_input = res.state.best_param.unwrap();
            assert!((optimized_input[0] - optimal_ax).abs() < 1e-4);
        }
    }

    mod discrete {
        #[test]
        fn test_dynamics_discrete() {}
        #[test]
        fn test_gradient_discrete() {}
        #[test]
        fn test_jacobian_discrete() {}
        #[test]
        fn test_hessian_discrete() {}
        #[test]
        fn test_dynamics_optimization_discrete() {}
    }
}
