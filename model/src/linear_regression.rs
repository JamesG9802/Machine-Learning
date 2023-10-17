use dragon_math;
use dragon_math::matrix::Matrix;
use crate::model as model;
use model::Model as Model;

/// <summary>
/// A model for performing linear regression training.
/// </summary>
pub struct LinearRegression {
    pub model: Model 
}

impl LinearRegression {
    /// <summary>
    /// Creates a Linear Regression model for a specific input size.
    /// </summary>
    pub fn new(input_size: usize) -> LinearRegression {
        let model: Model = Model::new(input_size, linear_regression_predict);
        return LinearRegression { model: model };
    }
}
/// <summary>
/// Uses the current weights to predict an output value. Throws an error if inputs do not match weight dimensions.
/// </summary>
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
