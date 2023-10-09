use dragon_math as math;
use math::matrix::{self, Matrix};

/// <summary>
/// A model for performing linear regression training.
/// </summary>
pub struct LinearRegression {
    pub input_size: usize,

    /// <summary>
    /// A d-dimensional vector to multiply with inputs to predict a value.
    /// </summary>
    weights: Matrix,

    /// <summary>
    /// A scalar that offsets the predicted value.
    /// </summary>
    bias: f64,
}

impl LinearRegression {
    /// <summary>
    /// Creates a Linear Regression model for a specific input size.
    /// </summary>
    pub fn new(input_size: usize) -> LinearRegression {
        let weights = Matrix::new(1, input_size);
        return LinearRegression {
            input_size: input_size,
            weights: weights,
            bias: 0.0,
        };
    }

    /// <summary>
    /// Trains the linear regression model with the given training function.
    /// </summary>
    pub fn train(
        &mut self,
        training_inputs: Matrix,
        training_outputs: Matrix,
        hyper_parameters: Vec<f64>,
        training_function: fn(&mut LinearRegression, &Matrix, &Matrix, Vec<f64>),
    ) {
        training_function(self, &training_inputs, &training_outputs, hyper_parameters);
    }

    /// <summary>
    /// Uses the current weights to predict an output matrix. Throws an error if inputs do not match weight dimensions.
    /// </summary>
    pub fn predict_outputs(&self, inputs: &Matrix) -> Matrix {
        //  dimensionality does not match
        if self.weights.cols != inputs.cols {
            panic!("Trying to predict when inputs' dimensionality ({}x{}) does not match weights ({}x{})",
                inputs.rows, inputs.cols,
                self.weights.rows, self.weights.cols);
        }
        //  y = xwT
        return inputs.dot(&self.weights.transpose()).add_scalar(self.bias);
    }

    /// <summary>
    /// Uses the current weights to predict an output value. Throws an error if inputs do not match weight dimensions.
    /// </summary>
    pub fn predict(&self, inputs: &Matrix) -> f64 {
        //  dimensionality does not match
        if self.weights.cols != inputs.cols {
            panic!("Trying to predict when inputs' dimensionality ({}x{}) does not match weights ({}x{})",
                inputs.rows, inputs.cols,
                self.weights.rows, self.weights.cols);
        }
        //  y = xwT
        return inputs.dot(&self.weights.transpose()).at(0, 0) + self.bias;
    }

    /// <summary>
    /// Returns the cost of predicted outputs compared to actual outputs.
    /// </summary>
    pub fn get_loss(
        &self,
        inputs: &Matrix,
        outputs: &Matrix,
        loss_function: fn(f64, f64) -> f64,
    ) -> f64 {
        let mut cost: f64 = 0.0;
        let predicted_outputs: Matrix = self.predict_outputs(inputs);
        for row in 0..predicted_outputs.rows {
            cost += loss_function(predicted_outputs.at(row, 0), outputs.at(row, 0));
        }
        return cost;
    }
}

/// <summary>
/// Loss function for linear regression; returns the squared error.
/// </summary>
pub fn loss_squared(predicted_value: f64, actual_value: f64) -> f64 {
    return (actual_value - predicted_value) * (actual_value - predicted_value);
}

/// <summary>
/// Loss function for linear regression; returns the absolute error.
/// </summary>
pub fn loss_absolute(predicted_value: f64, actual_value: f64) -> f64 {
    return (actual_value - predicted_value).abs();
}

/// <summary>
/// Runs the batch gradient descent algorithm for (L2) least square deviation regression.
/// </summary>
/// <param name="hyper_parameter">Hyper Paramaters: The first value is the learning rate.
/// The second value is the maximum number of training iterations. -1 for infinite iterations or until perfect prediction.</param>
pub fn batch_gradient_descent_l2(
    model: &mut LinearRegression,
    training_inputs: &Matrix,
    training_outputs: &Matrix,
    hyper_parameters: Vec<f64>,
) {
    let mut iteration_count: f64 = 0.0;
    while hyper_parameters[1] == -1.0 || iteration_count < hyper_parameters[1] {
        iteration_count += 1.0;
        println!("Iteration #{iteration_count}");
        let gradient_weights: Matrix;
        let gradient_bias: f64;
        //  prediction: y = w1x1 + w2x2 + ... wnxn + bias
        //  mean squared error: 1/n(actual-predicted)^2
        //  gradient (weights): -2/n Σ(actual-predicted)x
        //  gradient (bias): -2/n Σ(actual-predicted)
        let prediction: Matrix = model.predict_outputs(training_inputs);
        let difference: Matrix = training_outputs.add(&prediction.multiply_scalar(-1.0));
        //  Σ(actual-predicted)x
        gradient_weights = training_inputs.transpose().dot(&difference);
        //  Σ(actual-predicted)
        gradient_bias = difference.sum(0, 0, difference.rows, difference.cols);
        //  weights = weights - gradient
        model.weights = model.weights.add(
            &gradient_weights
                .multiply_scalar(2.0 * hyper_parameters[0] / training_inputs.rows as f64),
        );
        model.bias =
            model.bias + gradient_bias * 2.0 * hyper_parameters[0] / training_inputs.rows as f64;

        let mut training_error: f64 = 0.0;
        for i in 0..training_outputs.rows {
            training_error += loss_squared(training_outputs.at(i, 0), prediction.at(i, 0));
        }
        println!("\tTraining error: {}", training_error);
        println!("\tWeights: \n{}", model.weights.to_string());
        println!("\tBias: \n{}", model.bias);
    }
}

/// <summary>
/// Runs the stochastic gradient descent algorithm for (L2) least square deviation regression.
/// </summary>
/// <param name="hyper_parameter">Hyper Paramaters: The first value is the learning rate.
/// The second value is the maximum number of training iterations. -1 for infinite iterations or until perfect prediction.
/// </param>
pub fn stochastic_gradient_descent_l2(
    model: &mut LinearRegression,
    training_inputs: &Matrix,
    training_outputs: &Matrix,
    hyper_parameters: Vec<f64>,
) {
    let mut cloned_inputs: Matrix = training_inputs.clone();
    let mut cloned_outputs: Matrix = training_outputs.clone();

    let mut iteration_count: f64 = 0.0;
    while hyper_parameters[1] == -1.0 || iteration_count < hyper_parameters[1] {
        iteration_count += 1.0;
        println!("Iteration #{iteration_count}");
        //  randomize order of inputs and outputs
        matrix::randomize_rows_together(&mut cloned_inputs, &mut cloned_outputs);
        for row in 0..cloned_inputs.rows {
            let gradient_weights: Matrix;
            let gradient_bias: f64;
            let single_input: Matrix = cloned_inputs.submatrix(row, 0, row + 1, cloned_inputs.cols);
            let single_output: Matrix = cloned_outputs.submatrix(row, 0, row + 1, cloned_outputs.cols);
            //  prediction: y = w1x1 + w2x2 + ... wnxn + bias
            //  mean squared error: 1/n(actual-predicted)^2
            //  gradient (weights): -2/n Σ(actual-predicted)x
            //  gradient (bias): -2/n Σ(actual-predicted)

            let prediction: Matrix = model.predict_outputs(&single_input);
            let difference: Matrix = single_output.add(&prediction.multiply_scalar(-1.0));
            //  Σ(actual-predicted)x
            gradient_weights = single_input.transpose().dot(&difference);
            //  Σ(actual-predicted)
            gradient_bias = difference.sum(0, 0, difference.rows, difference.cols);
            //  weights = weights - gradient
            model.weights = model.weights.add(&gradient_weights.multiply_scalar(2.0 * hyper_parameters[0]));
            model.bias = model.bias + gradient_bias * 2.0 * hyper_parameters[0];
        }

        let prediction: Matrix = model.predict_outputs(&cloned_inputs);
        let mut training_error: f64 = 0.0;
        for i in 0..cloned_outputs.rows {
            training_error += loss_squared(cloned_outputs.at(i, 0), prediction.at(i, 0));
        }
        println!("\tTraining error: {}", training_error);
        println!("\tWeights: \n{}", model.weights.to_string());
        println!("\tBias: \n{}", model.bias);
    }
}

/// <summary>
/// Runs the mini-batch gradient descent algorithm for (L2) least square deviation regression.
/// </summary>
/// <param name="hyper_parameter">Hyper Paramaters: The first value is the learning rate.
/// The second value is the maximum number of training iterations. -1 for infinite iterations or until perfect prediction.
/// The third value is the desired size of the mini batch.
/// </param>
pub fn mini_batch_gradient_descent_l2(
    model: &mut LinearRegression,
    training_inputs: &Matrix,
    training_outputs: &Matrix,
    hyper_parameters: Vec<f64>,
) {
    let mut cloned_inputs: Matrix = training_inputs.clone();
    let mut cloned_outputs: Matrix = training_outputs.clone();

    let mut iteration_count: f64 = 0.0;
    let mut batch_size: f64 = hyper_parameters[2];
    if batch_size <= 0.0 {
        batch_size = 50.0;
    }
    while hyper_parameters[1] == -1.0 || iteration_count < hyper_parameters[1] {
        iteration_count += 1.0;
        println!("Iteration #{iteration_count}");
        //  randomize order of inputs and outputs
        matrix::randomize_rows_together(&mut cloned_inputs, &mut cloned_outputs);
        let mut start_row = 0;
        let mut is_looping= true;
        while is_looping {
            let gradient_weights: Matrix;
            let gradient_bias: f64;
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
            //  prediction: y = w1x1 + w2x2 + ... wnxn + bias
            //  mean squared error: 1/n(actual-predicted)^2
            //  gradient (weights): -2/n Σ(actual-predicted)x
            //  gradient (bias): -2/n Σ(actual-predicted)

            let prediction: Matrix = model.predict_outputs(&batch_inputs);
            let difference: Matrix = batch_outputs.add(&prediction.multiply_scalar(-1.0));
            //  Σ(actual-predicted)x
            gradient_weights = batch_inputs.transpose().dot(&difference);
            //  Σ(actual-predicted)
            gradient_bias = difference.sum(0, 0, difference.rows, difference.cols);
            //  weights = weights - gradient
            model.weights = model.weights.add(&gradient_weights.multiply_scalar(2.0 * hyper_parameters[0] / cloned_inputs.rows as f64));
            model.bias = model.bias + gradient_bias * 2.0 * hyper_parameters[0] / cloned_inputs.rows as f64 ;
        }
        let prediction: Matrix = model.predict_outputs(&cloned_inputs);
        let mut training_error: f64 = 0.0;
        for i in 0..cloned_outputs.rows {
            training_error += loss_squared(cloned_outputs.at(i, 0), prediction.at(i, 0));
        }
        println!("\tTraining error: {}", training_error);
        println!("\tWeights: \n{}", model.weights.to_string());
        println!("\tBias: \n{}", model.bias);
    }    
}
