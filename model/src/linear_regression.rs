use dragon_math as math;
use math::matrix::Matrix;

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
    fn predict_outputs(&self, inputs: &Matrix) -> Matrix {
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
}

/// <summary>
/// Loss function for linear regression; returns the squared error.
/// </summary>
fn loss_squared(predicted_value: f64, actual_value: f64) -> f64 {
    return (actual_value - predicted_value) * (actual_value - predicted_value);
}

/// <summary>
/// Loss function for linear regression; returns the absolute error.
/// </summary>
fn loss_absolute(predicted_value: f64, actual_value: f64) -> f64 {
    return (actual_value - predicted_value).abs();
}

/// <summary>
/// Runs the batch gradient descent algorithm for (L2) least square deviation regression.
/// </summary>
/// <param name="hyper_parameter">Hyper Paramaters: The first value is the learning rate.
/// The second value is the maximum number of training iterations. -1 for infinite iterations or until perfect prediction.</param>
pub fn batch_gradient_descent_l1(
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
        model.weights = model.weights.add(&gradient_weights.multiply_scalar(2.0 * hyper_parameters[0] / training_inputs.rows as f64));
        model.bias = model.bias + gradient_bias * 2.0 * hyper_parameters[0] / training_inputs.rows as f64;

        let mut training_error: f64 = 0.0;
        for i in 0..training_outputs.rows {
            training_error += loss_squared(training_outputs.at(i, 0), prediction.at(i, 0));
        }
        println!("\tTraining error: {}", training_error);
        println!("\tWeights: \n{}", model.weights.to_string());
        println!("\tBias: \n{}", model.bias);
        
    }
}
