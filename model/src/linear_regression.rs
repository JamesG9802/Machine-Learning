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
}

impl LinearRegression {

    /// <summary>
    /// Creates a Linear Regression model for a specific input size.
    /// </summary>
    pub fn new(input_size: usize) -> LinearRegression{
        let weights = Matrix::new(1, input_size);
        
        return LinearRegression {
            input_size: input_size,
            weights: weights,
        };
    }

    pub fn train(&mut self, training_data: Matrix) {

    }

    /// <summary>
    /// Uses the current weights to predict a value. Throws an error if inputs do not match weight dimensions.
    /// </summary>
    pub fn predict(&self, inputs: Matrix) {
        //  dimensionality does not match
        if self.weights.cols != inputs.cols {
            panic!("Trying to predict when inputs' dimensionality ({}x{}) does not match weights ({}x{})",
                inputs.rows, inputs.cols,
                self.weights.rows, self.weights.cols);
        }
    }
}