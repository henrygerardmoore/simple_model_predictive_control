// Copyright 2025 Henry Gerard Moore
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! This is a rust implementation of Model Predictive Control (MPC).
//! It intends to provide a simple framework to define MPC as an [argmin](https://github.com/argmin-rs/argmin) problem.
//! I created this library for a simple, pure-rust MPC implementation with no code generation.
//! See the [repository](github.com/henrygerardmoore/simple_model_predictive_control/) for more information.

pub mod mpc_problem;
pub mod mpc_problem_builder;
pub mod test;

/// Things that will be most commonly `use`'d in one convenient export
pub mod prelude;
