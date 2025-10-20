# simple_model_predictive_control

This is a rust implementation of Model Predictive Control (MPC).

It intends to provide a simple framework to define MPC as an [argmin](https://github.com/argmin-rs/argmin) problem.

I created this library for a simple, pure-rust MPC implementation with no code generation.

I originally intended to use `std::autodiff` (check out its [tracking issue](https://github.com/rust-lang/rust/issues/124509) if you're interested in this) to enable usage of gradient-based optimizers, but found the necessity of using an experimental compiler to be prohibitive to development.
Thus, I use the `finitediff` crate (also created by `argmin`) for this purpose.

## Benefits

`simple_model_predictive_control` is lightweight, simple to set up, requires no hand calculation of gradients, and can be used out of the box with `argmin` optimizers.

It supports both simple continuous dynamics (Euler integration of an ODE) and custom discrete dynamics.
Discrete dynamics functions can be used for dynamics that cannot be expressed as ODEs, using external physics simulation as the dynamics, or enforcing constraints.
See the [acrobot](examples/acrobot.rs) and [cartpole](examples/cartpole.rs) examples for examples of how to use discrete dynamics functions for both non-ODE dynamics.
The [acrobot](examples/acrobot.rs) example also uses RK4 as its integrator for better numerical stability.

Please feel encouraged to use this project as inspiration or a starting point for your own MPC implementations!

I plan to implement a wrapper for this as a controller in [Copper](https://github.com/copper-project/copper-rs) along with using it for [their existing cartpole example](https://github.com/copper-project/copper-rs/tree/master/examples/cu_rp_balancebot), so stay tuned for a more realistic simulated example!
I will provide a link to that example from this README when it is complete.

I also plan to make a dynamics-aware optimizer for `argmin`. You may note that the existing example for acrobot performs only decently, same as for the cartpole example.
This is because the optimizers I am using are simply optimizing the entire string of inputs over the whole lookahead duration, so dynamics is a black box to them.
I will modify the examples and remove this note when that optimizer is ready.

## Areas for Improvement

### Cost function

Cost or even dynamics functions are difficult to create and tune.
The [acrobot](examples/acrobot.rs) example's cost function, for instance, took quite a long time to formulate in a way that would produce the desired outcome.
This is partially a drawback to MPC in general, but could be improved with tooling to support visualization of costs.
Such visualization would be relatively easy to make in `plotters`, for example, but having it built-in would be a bonus.

### Constraints

This library does not support hard constraints.
Despite this, there are some workarounds.
For example, for input constraints, clamping the input to the valid range works well (and is done in the [cartpole](examples/cartpole.rs) and [acrobot](examples/acrobot.rs) examples).
Formulating constraints as high-penalty soft constraints is another way to work around this, but it obviously isn't a complete substitute for handling hard constraints.

### Performance

I'm sure there are a ton of areas for optimizations.
The most obvious of these is to use a parallel optimizer, though!
I added the `bulk_cost` function to the MPCProblem (see the [parallel evaluation section of the argmin book](https://www.argmin-rs.org/book/defining_optimization_problem.html?highlight=rayon#parallel-evaluation-with-bulk_-methods)), but only the particle swarm optimization solver in `argmin` currently supports this.
I did not have much luck with PSO when setting up the examples, but please let me know if you try it out and have a different experience!

There may be lower-hanging fruit in the form of utilizing more (or less) ndarray. I started out by using exclusively `f64` arrays and then ended up needing to use `ndarray`s, so it is entirely possible that removing the generics from MPCProblem entirely is the way to go.

### Gradient

I have not experimented with using central finite difference as opposed to forward difference. It would be nice to provide a way to choose which is used so that the user can do so.

## Alternatives

### Rust

Check out [optimization-engine](https://github.com/alphaville/optimization-engine) for a more powerful, more complete, and more production-ready MPC implementation!
It has Rust and python bindings and works with ROS.

### Other

I've used [MuJoCo MPC](https://github.com/google-deepmind/mujoco_mpc) quite a bit at my work, and it is great to use!
The MuJoCo simulator as a whole is quite powerful and user-friendly.

## License

You can use this project under either the [Apache License (Version 2.0)](./LICENSE-APACHE) or the [MIT license](./LICENSE-MIT).

## Contributing

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

Please feel free to open an issue or pull request!
Any problems you run into are likely something I encountered while building this library.
I'm always happy to provide help, and anything that's unclear likely needs better documentation.
