mod gla_package;
use crate::gla_package::{gla::{
    aging_gompertz_makeham, fertility_brass_polynomial, gla_model,
    growth_function, learning_function,
}, landscape::{generate_grids, compute_fitness_grid_parallel, compute_fitness_grid_point}};
use ndarray::{Array2, Array1, stack, Axis, s};
use std::error::Error;
use std::fs::File;
use csv::WriterBuilder;

fn main() -> Result<(), Box<dyn Error>>{
    let dim = 500;
    
    let (base_b_grid, reduced_b_grid) = generate_grids(
        "regular",
        1e-5, 
        0.14, 
        0.0, 
        0.20,
        dim,
        0.95,
    );

    let minimum_mortality = 1e-5f64;
    let aging_parameters = [0.00275961297460256,0.04326224872667336,0.025201676835511704] ;
    let learning_parameters = [0.01606792505529796,39.006865144958745,0.11060749334680318];
    let growth_parameters: [f64; 2] = [0.05168141300917714,0.08765165352033985];

    let aging_intermediate_closure = |x: f64,
                                      aging_parameters: &[f64],
                                      learning_parameters: &[f64],
                                      growth_parameters: &[f64]|
     -> f64 {
        gla_model(
            x,
            aging_gompertz_makeham as fn(f64, &[f64]) -> f64,
            learning_function,
            growth_function,
            &aging_parameters,
            &learning_parameters,
            &growth_parameters,
            minimum_mortality,
        )
    };

    let fertility_parameters = [2.445e-5, 14.8, 32.836];
    let fertility_function = fertility_brass_polynomial;

    let fertility_closure = |x: f64| -> f64 {
        fertility_function(x, &fertility_parameters)
    };

    let fitness_base_b = compute_fitness_grid_parallel(&base_b_grid, &aging_intermediate_closure, &aging_parameters, &learning_parameters, &growth_parameters, &fertility_closure);
    let fitness_reduced_b = compute_fitness_grid_parallel(&reduced_b_grid, &aging_intermediate_closure, &aging_parameters, &learning_parameters, &growth_parameters, &fertility_closure);

    let fitness_difference = fitness_reduced_b - fitness_base_b;

    // save result

    // Combine the grid and fitness difference into one array
    let fitness_difference_reshaped = fitness_difference.insert_axis(Axis(1));
     let combined_data = stack![Axis(1), base_b_grid, fitness_difference_reshaped];

    // Write to CSV
    let file = File::create("../output/fitness_difference.csv")?;
    let mut writer = WriterBuilder::new().has_headers(true).from_writer(file);

    // Write headers
    writer.write_record(&["b", "lmax", "fitness_difference"]);

    // Write data
    for row in combined_data.genrows() {
        writer.serialize(row.to_vec());
    }

    writer.flush();

    Ok(())
}