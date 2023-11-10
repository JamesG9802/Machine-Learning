use dragon_math as math;
use math::matrix::{self, Matrix};

pub const LEARNING_RATE: usize = 0;
pub const MAX_ITERATIONS: usize = 1;
pub const BATCH_SIZE: usize = 2;

/// A model that can be trained with weights and biases.
pub struct Model {
    pub input_size: usize,

    /// A d-dimensional vector to multiply with inputs to predict a value.
    pub weights: Matrix,

    /// A scalar that offsets the predicted value.
    pub bias: f64,

    /// Uses the current weights to predict an output value.
    pub predict_fn: fn(&Model, &Matrix) -> f64,

    /// Uses the current inputs to determine how the model's weights should be updated compared to actual outputs.
    pub update_fn: fn(&mut Model, &Matrix, &Matrix, &Vec<f64>),
}
impl Model {
    
    /// Creates a  model for a specific input size.
    pub fn new(input_size: usize, predict_fn: fn(&Model, &Matrix) -> f64, 
        update_fn: fn(&mut Model, &Matrix, &Matrix, &Vec<f64>)) -> Model {
        let weights = Matrix::new(1, input_size);
        return Model {
            input_size: input_size,
            weights: weights,
            bias: 0.0,
            predict_fn: predict_fn,
            update_fn: update_fn
        };
    }

    /// Uses the current weights to predict an output value. Throws an error if inputs do not match weight dimensions.
    pub fn predict(&self, inputs: &Matrix) -> f64 {
        return (self.predict_fn)(&self as &Model, inputs);
    }

    /// Uses the current weights to predict an output matrix. Throws an error if inputs do not match weight dimensions.
    pub fn predict_multiple(&self, inputs: &Matrix) -> Matrix {
        //  dimensionality does not match
        if self.weights.cols != inputs.cols {
            panic!("Trying to predict when inputs' dimensionality ({}x{}) does not match weights ({}x{})",
                inputs.rows, inputs.cols,
                self.weights.rows, self.weights.cols);
        }
        //  y = xwT
        let mut outputs = Matrix::new(inputs.rows, 1);
        for row in 0..inputs.rows {
            outputs.set(row, 0, self.predict(&inputs.submatrix(row, 0, row + 1, inputs.cols)));
        }
        return outputs;
    }

    /// Returns the cost of predicted outputs compared to actual outputs.
    pub fn get_loss(&self, inputs: &Matrix, outputs: 
        &Matrix,loss_function: fn(f64, f64) -> f64) -> f64 {
        let mut cost: f64 = 0.0;
        let predicted_outputs: Matrix = self.predict_multiple(inputs);
        for row in 0..predicted_outputs.rows {
            cost += loss_function(predicted_outputs.at(row, 0), outputs.at(row, 0));
        }
        return cost;
    }
  
    /// Trains the linear regression model with the given training function.
    pub fn train(
        &mut self,
        training_inputs: Matrix,
        training_outputs: Matrix,
        hyper_parameters: Vec<f64>,
        training_function: fn(model: &mut Model, &Matrix, &Matrix, Vec<f64>),
    ) 
    {
        training_function(self as &mut Model, &training_inputs, &training_outputs, hyper_parameters);
    }
}

/// Loss function for linear regression; returns the squared error.
pub fn loss_squared(predicted_value: f64, actual_value: f64) -> f64 {
    return (actual_value - predicted_value) * (actual_value - predicted_value);
}

/// Loss function for linear regression; returns the absolute error.
pub fn loss_absolute(predicted_value: f64, actual_value: f64) -> f64 {
    return (actual_value - predicted_value).abs();
}

/// Runs the batch gradient descent algorithm for (L2) least square deviation regression.
pub fn batch_gradient_descent_l2(
    model: &mut Model,
    training_inputs: &Matrix,
    training_outputs: &Matrix,
    hyper_parameters: Vec<f64>,
) {
    let mut iteration_count: i64 = 0;
    while hyper_parameters[MAX_ITERATIONS] == -1.0 
        || iteration_count < hyper_parameters[MAX_ITERATIONS] as i64 {
        iteration_count += 1;
        println!("Iteration #{iteration_count}");
        
        (model.update_fn)(model, &training_inputs, &training_outputs, &hyper_parameters);
        
        let mut training_error: f64 = 0.0;
        let prediction: Matrix = model.predict_multiple(&training_inputs);
        for i in 0..training_outputs.rows {
            training_error += loss_squared(training_outputs.at(i, 0), prediction.at(i, 0));
        }
        println!("\tTraining error: {}", training_error);
        println!("\tWeights: \n{}", model.weights.to_string());
        println!("\tBias: \n{}", model.bias);
    }
}

/// Runs the stochastic gradient descent algorithm for (L2) least square deviation regression.
pub fn stochastic_gradient_descent_l2(
    model: &mut Model,
    training_inputs: &Matrix,
    training_outputs: &Matrix,
    hyper_parameters: Vec<f64>,
) {
    let mut cloned_inputs: Matrix = training_inputs.clone();
    let mut cloned_outputs: Matrix = training_outputs.clone();

    let mut iteration_count: i64 = 0;
    while hyper_parameters[MAX_ITERATIONS] == -1.0 
        || iteration_count < hyper_parameters[MAX_ITERATIONS] as i64 {
        iteration_count += 1;
        println!("Iteration #{iteration_count}");
        //  randomize order of inputs and outputs
        matrix::randomize_rows_together(&mut cloned_inputs, &mut cloned_outputs);
        for _row in 0..cloned_inputs.rows {
            (model.update_fn)(model, &training_inputs, &training_outputs, &hyper_parameters);
        }

        let prediction: Matrix = model.predict_multiple(&cloned_inputs);
        let mut training_error: f64 = 0.0;
        for i in 0..cloned_outputs.rows {
            training_error += loss_squared(cloned_outputs.at(i, 0), prediction.at(i, 0));
        }
        println!("\tTraining error: {}", training_error);
        println!("\tWeights: \n{}", model.weights.to_string());
        println!("\tBias: \n{}", model.bias);
    }
}

/// Runs the mini-batch gradient descent algorithm for (L2) least square deviation regression.
pub fn mini_batch_gradient_descent_l2(
    model: &mut Model,
    training_inputs: &Matrix,
    training_outputs: &Matrix,
    hyper_parameters: Vec<f64>,
) {
    let mut cloned_inputs: Matrix = training_inputs.clone();
    let mut cloned_outputs: Matrix = training_outputs.clone();

    let mut iteration_count: i64 = 0;
    let mut batch_size: f64 = hyper_parameters[BATCH_SIZE];
    if batch_size <= 0.0 {
        batch_size = 50.0;
    }
    while hyper_parameters[MAX_ITERATIONS] == -1.0 
        || iteration_count < hyper_parameters[MAX_ITERATIONS] as i64 {
        iteration_count += 1;
        println!("Iteration #{iteration_count}");
        //  randomize order of inputs and outputs
        matrix::randomize_rows_together(&mut cloned_inputs, &mut cloned_outputs);
        let mut start_row = 0;
        let mut is_looping= true;
        while is_looping {
            let batch_inputs: Matrix;
            let batch_outputs: Matrix;
            if start_row + batch_size as i64 >= cloned_inputs.rows as i64{
                batch_inputs = cloned_inputs.submatrix(start_row as usize, 0, cloned_inputs.rows, cloned_inputs.cols);
                batch_outputs = cloned_outputs.submatrix(start_row as usize, 0, cloned_inputs.rows, cloned_outputs.cols);
                is_looping = false;
            }
            else {
                batch_inputs = cloned_inputs.submatrix(start_row as usize, 0, 
                    (start_row + batch_size as i64) as usize, cloned_inputs.cols);
                batch_outputs = cloned_outputs.submatrix(start_row as usize, 0, 
                    (start_row + batch_size as i64) as usize, cloned_outputs.cols);
            }
            start_row += batch_size as i64;
            (model.update_fn)(model, &batch_inputs, &batch_outputs, &hyper_parameters);
        }
        let prediction: Matrix = model.predict_multiple(&cloned_inputs);
        let mut training_error: f64 = 0.0;
        for i in 0..cloned_outputs.rows {
            training_error += loss_squared(cloned_outputs.at(i, 0), prediction.at(i, 0));
        }
        println!("\tTraining error: {}", training_error);
        println!("\tWeights: \n{}", model.weights.to_string());
        println!("\tBias: \n{}", model.bias);
    }    
}
