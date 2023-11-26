use std::f64::consts::E;

pub trait Activation {
	fn function(x: f64) -> f64;
	fn derivative(x: f64) -> f64;
}

pub struct Sigmoid;

impl Activation for Sigmoid {
	fn function(x: f64) -> f64 {
		1.0 / (1.0 + E.powf(-x))
	}

	fn derivative(x: f64) -> f64 {
		x * (1.0 - x)
	}
}

/*
#[derive(Clone)]
pub struct Activation<'a> {
	pub function: &'a dyn Fn(f64) -> f64,
	pub derivative: &'a dyn Fn(f64) -> f64,
}

pub const IDENTITY: Activation = Activation {
	function: &|x| x,
	derivative: &|_| 1.0,
};

pub const SIGMOID: Activation = Activation {
	function: &|x| 1.0 / (1.0 + E.powf(-x)),
	derivative: &|x| x * (1.0 - x),
};

pub const TANH: Activation = Activation {
	function: &|x| x.tanh(),
	derivative: &|x| 1.0 - (x.powi(2)),
};

pub const RELU: Activation = Activation {
	function: &|x| x.max(0.0),
	derivative: &|x| if x > 0.0 { 1.0 } else { 0.0 },
};
 */
