pub use crate::function_interface::Function;
use ndarray as nd;


struct Simplex {
    dim:usize,
    vertices:nd::Array2<f64>,
    fvals:nd::Array1<f64>
}

impl Simplex {
    pub fn new(dim:usize, length_scale:f64, f:&impl Function) -> Simplex {
        let mut verts = nd::Array2::<f64>::zeros((dim+1, dim));
        let mut fvals = nd::Array1::<f64>::zeros(dim+1);
        for i in 1..dim+1 {
            verts[[i, i-1]] = length_scale;
            fvals[i] = f.fval(&verts.row(i).to_owned())

        }
        return Simplex { dim: dim, vertices: verts, fvals: fvals }
    }

    pub fn from_point(origin:&nd::Array1<f64>, length_scale:f64, f:&impl Function) -> Simplex {
        let dim = origin.len();
        let mut verts = nd::Array2::<f64>::zeros((dim+1, dim));
        let mut fvals = nd::Array1::<f64>::zeros(dim+1);
        verts.row_mut(0).assign(origin);
        fvals[0] = f.fval(origin);
        for i in 1..dim+1 {
            verts.row_mut(i).assign(origin);
            verts[[i, i-1]] += length_scale;
            fvals[i] = f.fval(&verts.row(i).to_owned())
        }
        return Simplex { dim: dim, vertices: verts, fvals: fvals}
    }
    pub fn eval_points(&mut self, f:&impl Function) {
        let mut fvals = nd::Array1::<f64>::zeros(self.dim + 1);
        for i in 1..self.dim {
            let point = self.vertices.row(i);
            self.fvals[i] = f.fval(&point.to_owned());
        }
    }
    pub fn get_vertices(&mut self) -> nd::Array2<f64> {
        self.vertices.to_owned()
    }
    pub fn get_vertex(&mut self, index:usize) -> nd::Array1<f64> {
        return self.vertices.row(index).to_owned();
    }
    pub fn get_fvals(&mut self) -> nd::Array1<f64> {
        return self.fvals.to_owned();
    }
    pub fn fval(&self, index:usize) -> f64 {
        return self.fvals[index];
    }
    pub fn reflect(&mut self, f:&impl Function, index:usize, factor:f64) -> f64 {
        let v =   self.vertices.row(index).to_owned();
        let psum =   (self.vertices.sum_axis(nd::Axis(0))-&v) / ((self.dim) as f64);
        let fac1 = (1. - factor) as f64;
        // let fac2 = fac1 - factor;
        let new_v = (&psum-&v) * fac1 + &v;
        let new_fval = f.fval(&new_v);

        if new_fval < self.fvals[index] {
            self.vertices.row_mut(index).assign(&new_v);
            self.fvals[index] = new_fval;
        }
        return self.fvals[index]
    }
    pub fn contract(&mut self, f:&impl Function, index:usize) {
        let v: &ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 1]>> =   &self.vertices.row(index).to_owned();
        for i in 0..self.dim+1 {
            if i == index {
                continue;
            }
            let u = &self.vertices.row(i);
            let z = 0.5 * u + 0.5 * v;
            self.vertices.row_mut(i).assign(&z);
            self.fvals[i] = f.fval(&z);
        }
    }
    pub fn get_iteration_indices(&mut self) -> (usize, usize, usize) {
        let mut low = 0;
        let mut high = 0;
        let mut sechigh = 0;
        let mut flow = f64::INFINITY;
        let mut fhigh = -f64::INFINITY;
        let mut fsechigh = -f64::INFINITY;

        for (i, fv) in self.fvals.iter().enumerate() {
            let fvloc = *fv;
            if fvloc > fhigh {
                fhigh = fvloc;
                high = i;
            } else if fvloc > fsechigh {
                fsechigh = fvloc;
                sechigh = i;
            }
            if fvloc < flow {
                flow = fvloc;
                low = i;
            }
        }
        return (low, sechigh, high);
    }
}

pub fn nelder_mead(iters:usize, f:&impl Function, dim:usize) -> (f64, nd::Array1<f64> ){
    let mut simplex = Simplex::from_point(&nd::arr1(&[1.5, -0.5]), 1., f);
    let ftol = 1e-8;
    let xtol = 1e-5;
    let mut solution = nd::Array1::<f64>::zeros(dim);
    let mut low_index = 0;
    for i in 0..iters {
        if i % 1 == 0 {
            println!("{}", simplex.vertices.flatten());
        }
        let indices = simplex.get_iteration_indices();
        low_index = indices.0;
        let high_index = indices.2;
        let sechigh_index = indices.1;
        let new_try = simplex.reflect(f, high_index, -1.0);
        if new_try < simplex.fval(low_index) {
            let _ = simplex.reflect(f, high_index, 2.0);
        } else if new_try >= simplex.fval(sechigh_index) {
            let y_prev = simplex.fval(high_index);
            let y_new = simplex.reflect(f, high_index, 0.5);
            if y_new >= y_prev {
                simplex.contract(f, low_index);
            }
        }
    }
    solution = simplex.get_vertex(low_index);
    return (f.fval(&solution), solution);

}
