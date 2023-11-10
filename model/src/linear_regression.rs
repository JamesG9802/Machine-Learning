use dragon_math;
use dragon_math::matrix::Matrix;
use crate::model as model;
use model::Model as Model;

/// A model for performing linear regression training.
pub struct LinearRegression {
    pub model: Model 
}

impl LinearRegression {
    /// Creates a Linear Regression model for a specific input size.
    pub fn new(input_size: usize) -> LinearRegression {
        let model: Model = Model::new(input_size, linear_regression_predict, linear_regression_update);
        return LinearRegression { model: model };
    }
}
/// Calculates the dot product of the weights and the inputs plus the bias.
fn linear_regression_predict(model: &Model, inputs: &Matrix) -> f64 {
    //  dimensionality does not match
    if model.weights.cols != inputs.cols {
        panic!("Trying to predict when inputs' dimensionality ({}x{}) does not match weights ({}x{})",
            inputs.rows, inputs.cols,
            model.weights.rows, model.weights.cols);
    }
    //  y = xwT
    return inputs.dot(&model.weights.transpose()).at(0, 0) + model.bias;
}

/// Uses the current inputs to determine how the model's weights should be updated.
fn linear_regression_update(model: &mut Model, 
    training_inputs: &Matrix, training_outputs: &Matrix, 
    hyper_parameters: &Vec<f64>) {
    let gradient_weights: Matrix;
    let gradient_bias: f64;
    //  prediction: y = w1x1 + w2x2 + ... wnxn + bias
    //  mean squared error: 1/n(actual-predicted)^2
    //  gradient (weights): -2/n Σ(actual-predicted)x
    //  gradient (bias): -2/n Σ(actual-predicted)
    let prediction: Matrix = model.predict_multiple(training_inputs);
    let difference: Matrix = training_outputs.add(&prediction.multiply_scalar(-1.0));
    //  Σ(actual-predicted)x
    gradient_weights = training_inputs.transpose().dot(&difference);
    //  Σ(actual-predicted)
    gradient_bias = difference.sum(0, 0, difference.rows, difference.cols);
    //  weights = weights - gradient
    model.weights = model.weights.add(&gradient_weights
        .multiply_scalar(2.0 * hyper_parameters[model::LEARNING_RATE] / training_inputs.rows as f64),
    );
    model.bias = model.bias + gradient_bias * 2.0 * hyper_parameters[model::LEARNING_RATE] / training_inputs.rows as f64;
}
