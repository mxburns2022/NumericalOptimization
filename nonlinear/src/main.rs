mod function_interface;
mod nelder_mead;
use ndarray as nd;
fn main() {
    let rosen = function_interface::GeneralizedRosenbrock::new(2); 

    let A = nd::arr2(&[[4.,1.], [1., 2.]]);
    let b = nd::arr1(&[1., -3.]);
    let quad = function_interface::Quadratic::new(2, A, b);
    let result = nelder_mead::nelder_mead(60, &quad, 2);
    println!("{} {}", result.0, result.1);
}
