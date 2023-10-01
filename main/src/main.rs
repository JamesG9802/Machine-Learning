use dragon_math as math;
use math::matrix::Matrix as Matrix;

use dragon_model as model;
use model::linear_regression::LinearRegression as LinearRegression;
fn main() {
    let mut matrix: Matrix = Matrix::new(3, 3);
    matrix.set(0, 0, 1.0);
    matrix.set(0, 1, 2.0);
    matrix.set(0, 2, 3.0);
    matrix.set(1, 0, 4.0);
    matrix.set(1, 1, 5.0);
    matrix.set(1, 2, 6.0);
    matrix.set(2, 0, 7.0);
    matrix.set(2, 1, 8.0);
    matrix.set(2, 2, 9.0);
    
    let mut matrix2: Matrix = Matrix::new(3, 3);
    matrix2.set(0, 0, 2.0);
    matrix2.set(0, 1, 2.0);
    matrix2.set(0, 2, 2.0);
    matrix2.set(1, 0, 3.0);
    matrix2.set(1, 1, 3.0);
    matrix2.set(1, 2, 3.0);
    matrix2.set(2, 0, 4.0);
    matrix2.set(2, 1, 5.0);
    matrix2.set(2, 2, 0.0);

    println!("{}", matrix.to_string());
    println!("{}", matrix2.to_string());
    matrix = matrix.dot(matrix2);
    println!("{}", matrix.to_string());

    let lr = LinearRegression::new(5);
    lr.predict(matrix);
}
