use ndarray::{Array1, Array2};
use peroxide::fuga::GaussLegendre;
use peroxide::fuga::G7K15;
use peroxide::numerical::integral::integrate;
use roots::{find_root_brent, SimpleConvergency};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

// GRID GENERATION
pub fn generate_grids(grid_type: &str, b_min: f64, b_max: f64, lmax_min: f64, lmax_max: f64, dim: usize, b_decrease: f64) -> (Array2<f64>, Array2<f64>) {
    let grid = generate_grid(grid_type, b_min, b_max, lmax_min, lmax_max, dim);
    let reduced_b_grid = compute_reduced_b_grid(grid.clone(), b_decrease);
    (grid, reduced_b_grid)
}

pub fn generate_grid(grid_type: &str, b_min: f64, b_max: f64, lmax_min: f64, lmax_max: f64, dim: usize) -> Array2<f64> {
    let mut grid = Array2::<f64>::zeros((dim * dim, 2));

    if grid_type == "regular" {
        let b_grid: Vec<f64> = linspace(b_min, b_max, dim);
        let lmax_grid: Vec<f64> = linspace(lmax_min, lmax_max, dim);

        for (i, &b) in b_grid.iter().enumerate() {
            for (j, &l) in lmax_grid.iter().enumerate() {
                grid[[i * dim + j, 0]] = b;
                grid[[i * dim + j, 1]] = l;
            }
        }
    }

    grid
}

fn compute_reduced_b_grid(mut grid: Array2<f64>, b_decrease: f64) -> Array2<f64> {
    for i in 0..grid.rows() {
        grid[[i, 0]] *= b_decrease;
    }
    grid
}

// Helper function to mimic numpy's linspace
fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    let step = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + step * i as f64).collect()
}

// FITNESS LANDSCAPE

pub fn survival_l(x:f64, aging_intermediate_closure: &dyn Fn(f64, &[f64], &[f64], &[f64]) -> f64, aging_parameters: &[f64], learning_parameters: &[f64], growth_parameters: &[f64]) -> f64 {
    let cumulative_hazard = integrate(
        |t: f64| -> f64 {
            aging_intermediate_closure(
                t,
                &aging_parameters,
                &learning_parameters,
                &growth_parameters,
            )
        },
        (0.0, x),
        GaussLegendre(16),
        // G7K15(1e-3, 100),
    );
    (-cumulative_hazard).exp()
}

pub fn euler_lotka_function(r: f64, aging_intermediate_closure: &dyn Fn(f64, &[f64], &[f64], &[f64]) -> f64, aging_parameters: &[f64], learning_parameters: &[f64], growth_parameters: &[f64], fertility_closure: &dyn Fn(f64) -> f64) -> f64 {
    let integral = integrate(
        |t: f64| -> f64 {
            fertility_closure(t) * survival_l(
                t,
                aging_intermediate_closure,
                aging_parameters,
                learning_parameters,
                growth_parameters,
            ) * (-r * t).exp()
        },
        (0.0, 100.0),
        GaussLegendre(16),
    );
    integral - 1.0
}

pub fn compute_fitness(aging_intermediate_closure: &dyn Fn(f64, &[f64], &[f64], &[f64]) -> f64, aging_parameters: &[f64], learning_parameters: &[f64], growth_parameters: &[f64], fertility_closure: &dyn Fn(f64) -> f64) -> f64 {

    let euler_lotka_closure = |r: f64| -> f64 {
                euler_lotka_function(
                    r,
                    aging_intermediate_closure,
                    aging_parameters,
                    learning_parameters,
                    growth_parameters,
                    fertility_closure,
                )
            };
    
    // Attempt to find the root using Brent's method
    let mut convergency = SimpleConvergency { eps:1e-15f64, max_iter:100};
    match find_root_brent(-1f64, 1f64, &euler_lotka_closure, &mut convergency) {
        Ok(root) => return root.max(0.0),
        Err(e) => println!("Failed to find root: {:?}", e),
    }
    return 0.0;
}

pub fn compute_fitness_grid_point(grid_point: Vec<f64>, aging_intermediate_closure: &dyn Fn(f64, &[f64], &[f64], &[f64]) -> f64, aging_parameters: &[f64], learning_parameters: &[f64], growth_parameters: &[f64], fertility_closure: &dyn Fn(f64) -> f64) -> f64 {
    let new_aging_parameters = [aging_parameters[0], grid_point[0], aging_parameters[2]];
    let new_learning_parameters = [grid_point[1], learning_parameters[1], learning_parameters[2]];
    compute_fitness(
        aging_intermediate_closure,
        &new_aging_parameters,
        &new_learning_parameters,
        &growth_parameters,
        fertility_closure,
    )
}

pub fn compute_fitness_grid_parallel<F, T>(
    grid: &Array2<f64>,
    aging_intermediate_closure: &F,
    aging_parameters: &[f64],
    learning_parameters: &[f64],
    growth_parameters: &[f64],
    fertility_closure: &T,
) -> Array1<f64>
where
    F: Fn(f64, &[f64], &[f64], &[f64]) -> f64 + Send + Sync,
    T: Fn(f64) -> f64 + Send + Sync,
{
    let n_rows = grid.rows();
    let pb = ProgressBar::new(n_rows as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:50.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let mut results: Vec<(usize, f64)> = grid.axis_iter(ndarray::Axis(0))
        .enumerate()
        .par_bridge()
        .map(|(index, row)| {
            let res = compute_fitness_grid_point(
                row.to_vec(),
                aging_intermediate_closure,
                aging_parameters,
                learning_parameters,
                growth_parameters,
                fertility_closure,
            );
            pb.inc(1); // Manually increment the progress bar
            (index, res)
        }).collect();

    // Sort by the original index
    results.sort_by_key(|&(index, _)| index);

    // Extract the results in order
    let results_in_order: Vec<_> = results.into_iter().map(|(_, res)| res).collect();

    Array1::from_vec(results_in_order)
}