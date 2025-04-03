use ndarray::*;
use std::ops::{Add, Sub, Mul, Div};

pub trait Function {
    fn fval(&self, _x:&Array1<f64>) -> f64 {
        panic!("Value computation is not implemented for this function type");
    }
    fn grad(&self, _x:&Array1<f64>) -> Array1<f64> {
        panic!("Gradient computation is not implemented for this function type");
    }
    fn hessian(&self, _x:&Array1<f64>) -> Array2<f64> {
        panic!("Hessian computation is not implemented for this function type");
    }
}

pub struct GeneralizedRosenbrock {
    dim:usize
}
impl GeneralizedRosenbrock {
    pub fn new(dim:usize) -> GeneralizedRosenbrock {
        return GeneralizedRosenbrock {dim: dim};
    }
}

impl Function for GeneralizedRosenbrock {
    fn fval(&self, x:&Array1<f64>) -> f64 {
        let mut result = 0.0;
        for i in 0..self.dim-1 {
            result += 100. * (x[i+1] - x[i] * x[i]) * (x[i+1] - x[i]*x[i]) + (1. - x[i]) * (1. - x[i]);
        }
        result
    }
    fn grad(&self, x:&Array1<f64>) -> Array1<f64> {
        let mut result = Array1::<f64>::zeros(x.len());
        for i in 1..self.dim-1 {
            result[i] = -200. * (x[i+1] - x[i]) + 200. * (x[i] - x[i-1]) - 2. * (1. - x[i]);
        }
        result[0] = -200. * (x[1] - x[0]) - 2. * (1. - x[0]);
        result[self.dim-1] = 200. * (x[self.dim-1] - x[self.dim-2]);
        result
    }
}



pub struct Quadratic {
    dim:usize,
    mat:Array2<f64>,
    b:Array1<f64>
}

impl Quadratic {
    pub fn new(dim:usize, mat:Array2<f64>, b:Array1<f64>) -> Quadratic {
        return Quadratic {dim: dim, mat:mat, b:b};
    }
}

impl Function for Quadratic {
    fn fval(&self, x:&Array1<f64>) -> f64 {
        let result = 0.5 * x.dot(&self.mat.dot(x)) + self.b.dot(x);
        
        result
    }
    fn grad(&self, x:&Array1<f64>) -> Array1<f64> {
        let result =self.mat.dot(x)+ &self.b;
        result
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct D64 {
    x:f64,
    y:f64
}
impl Add for D64 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        return Self {x: self.x + other.x, y: self.y + other.y};
    }

}

impl Sub for D64 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        return Self {x: self.x - other.x, y: self.y - other.y};
    }
}
impl Mul<D64> for D64 {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        return Self {x: self.x * other.x, y: self.x * other.y + self.y * other.x};
    }
    
}
impl Mul<f64> for D64 {
    type Output = Self;

    fn mul(self, other: f64) -> Self {
        return Self {x: self.x * other, y: self.y * other};
    }
    
}
impl D64 {
    fn pow(&self, power:f64) -> D64 {
        if power == 1.0 {
            return self.clone();
        }
        return D64 { x: self.x.powf(power), y: power * self.y * self.x.powf(power-1.) }
    }

    fn tanh(&self, power:f64) -> D64 {
        if power == 1.0 {
            return self.clone();
        }
        return D64 { x: self.x.powf(power), y: power * self.y * self.x.powf(power-1.) }
    }
    fn eval() {

    }
    
}
