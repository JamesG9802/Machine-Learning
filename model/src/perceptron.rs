use std::collections::HashMap;

use dragon_math;
use dragon_math::matrix::Matrix;
use crate::model::{self as model, LEARNING_RATE};
use model::Model as Model;
/// A model for performing perceptron training. Expects data to be normalized and outputs to be -1 or +1.
pub struct Perceptron {
    pub model: Model,
}
impl Perceptron {
    pub fn new(input_size: usize) -> Perceptron {
        let model: Model = Model::new(input_size, perceptron_predict, perceptron_update);
        return Perceptron { model: model };
    }
}
/// Computes the dot product of the weights and inputs + bias to classify as -1 or 1. 
/// Throws an error if inputs do not match weight dimensions.
fn perceptron_predict(model: &Model, inputs: &Matrix) -> f64 {
    //  dimensionality does not match
    if model.weights.cols != inputs.cols {
        panic!("Trying to predict when inputs' dimensionality ({}x{}) does not match weights ({}x{})",
            inputs.rows, inputs.cols,
            model.weights.rows, model.weights.cols);
    }
    //  y = xwT
    if inputs.dot(&model.weights.transpose()).at(0, 0) + model.bias > 0.0
    {
        return 1.0;
    }
    else {
        return -1.0;
    }
}

/// Uses the current inputs to determine how the model's weights should be updated.
fn perceptron_update(model: &mut Model, 
    training_inputs: &Matrix, training_outputs: &Matrix, 
    hyper_parameters: &HashMap<&str, f64>) {

    let mut learning_rate: f64 = 0.01;

    if hyper_parameters.contains_key(LEARNING_RATE) {
        learning_rate = hyper_parameters.get(LEARNING_RATE).unwrap().clone();
    }
    //  prediction: y = w1x1 + w2x2 + ... wnxn + bias > 0 ? 1 : -1
    //  weights <- (y_actual - y_predict) * learning rate * input
    //  bias <- (y_actual - y_predict) * learning_rate
    let prediction: Matrix = model.predict_multiple(training_inputs);
    let difference: Matrix = training_outputs.add(&prediction.multiply_scalar(-1.0));
    
    model.weights = model.weights.add(&training_inputs.transpose().dot(&difference)
        .multiply_scalar(learning_rate / training_inputs.rows as f64),
    );
    model.bias = model.bias + learning_rate * 
        difference.sum(0, 0, difference.rows, difference.cols) / training_inputs.rows as f64;
}
