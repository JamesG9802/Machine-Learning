pub mod matrix;
#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn at_set_test() {
        let mut matrix = Matrix::new(1, 2);
        matrix.set(0, 0, 20.0);
        matrix.set(0, 1, 50.0);
        assert_eq!(matrix.at(0, 0) + matrix.at(0, 1), 20.0 + 50.0);
    }

    #[test]
    fn submatrix_test() {
        let mut matrix = Matrix::new(3, 3);
        matrix.set(0, 0, 10.0);
        matrix.set(0, 1, 20.0);
        matrix.set(0, 2, 30.0);
        matrix.set(1, 0, 40.0);
        matrix.set(1, 1, 50.0);
        matrix.set(1, 2, 60.0);
        matrix.set(2, 0, 70.0);
        matrix.set(2, 1, 80.0);
        matrix.set(2, 2, 90.0);
        let submatrix = matrix.submatrix(0, 0, 2, 2);
        let mut matrix_check = Matrix::new(2, 2);
        matrix_check.set(0, 0, 10.0);
        matrix_check.set(0, 1, 20.0);
        matrix_check.set(1, 0, 40.0);
        matrix_check.set(1, 1, 50.0);

        println!("{}", submatrix.to_string());
        println!("{}", matrix_check.to_string());

        assert_eq!(submatrix.to_string(), matrix_check.to_string());
    }

    #[test]
    fn transpose_test() {
        let mut matrix = Matrix::new(2, 3);
        matrix.set(0, 0, 10.0);
        matrix.set(0, 1, 20.0);
        matrix.set(0, 2, 30.0);
        matrix.set(1, 0, 40.0);
        matrix.set(1, 1, 50.0);
        matrix.set(1, 2, 60.0);

        let transpose_matrix = matrix.transpose();

        let mut matrix_check = Matrix::new(3, 2);
        matrix_check.set(0, 0, 10.0);
        matrix_check.set(0, 1, 40.0);
        matrix_check.set(1, 0, 20.0);
        matrix_check.set(1, 1, 50.0);
        matrix_check.set(2, 0, 30.0);
        matrix_check.set(2, 1, 60.0);

        println!("{}", transpose_matrix.to_string());
        println!("{}", matrix_check.to_string());

        assert_eq!(transpose_matrix.to_string(), matrix_check.to_string());
    }

    #[test]
    fn dot_test() {
        let mut matrix = Matrix::new(2, 3);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(0, 2, 3.0);
        matrix.set(1, 0, 4.0);
        matrix.set(1, 1, 5.0);
        matrix.set(1, 2, 6.0);

        let mut matrix_other = Matrix::new(3, 2);
        matrix_other.set(0, 0, 7.0);
        matrix_other.set(0, 1, 8.0);
        matrix_other.set(1, 0, 9.0);
        matrix_other.set(1, 1, 10.0);
        matrix_other.set(2, 0, 11.0);
        matrix_other.set(2, 1, 12.0);

        let dot_matrix = matrix.dot(&matrix_other);
        let mut matrix_check = Matrix::new(2, 2);
        matrix_check.set(0, 0, 58.0);
        matrix_check.set(0, 1, 64.0);
        matrix_check.set(1, 0, 139.0);
        matrix_check.set(1, 1, 154.0);
        println!("{}", dot_matrix.to_string());
        println!("{}", matrix_check.to_string());

        assert_eq!(dot_matrix.to_string(), matrix_check.to_string());
    }

    #[test]
    fn add_test() {
        let mut matrix = Matrix::new(2, 2);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);

        let mut matrix_other = Matrix::new(2, 2);
        matrix_other.set(0, 0, 1.0);
        matrix_other.set(0, 1, 2.0);
        matrix_other.set(1, 0, 3.0);
        matrix_other.set(1, 1, 4.0);

        let add_matrix = matrix.add(&matrix_other);
        let mut matrix_check = Matrix::new(2, 2);
        matrix_check.set(0, 0, 2.0);
        matrix_check.set(0, 1, 4.0);
        matrix_check.set(1, 0, 6.0);
        matrix_check.set(1, 1, 8.0);
        println!("{}", add_matrix.to_string());
        println!("{}", matrix_check.to_string());

        assert_eq!(add_matrix.to_string(), matrix_check.to_string());
    }

    #[test]
    fn multiply_scalar_test() {
        let mut matrix = Matrix::new(2, 2);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);

        let multiply_scalar_matrix = matrix.multiply_scalar(2.0);
        let mut matrix_check = Matrix::new(2, 2);
        matrix_check.set(0, 0, 2.0);
        matrix_check.set(0, 1, 4.0);
        matrix_check.set(1, 0, 6.0);
        matrix_check.set(1, 1, 8.0);
        println!("{}", multiply_scalar_matrix.to_string());
        println!("{}", matrix_check.to_string());

        assert_eq!(multiply_scalar_matrix.to_string(), matrix_check.to_string());
    }

    #[test]
    fn add_scalar_test() {
        let mut matrix = Matrix::new(2, 2);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);

        let multiply_scalar_matrix = matrix.add_scalar(-2.0);
        let mut matrix_check = Matrix::new(2, 2);
        matrix_check.set(0, 0, -1.0);
        matrix_check.set(0, 1, 0.0);
        matrix_check.set(1, 0, 1.0);
        matrix_check.set(1, 1, 2.0);
        println!("{}", multiply_scalar_matrix.to_string());
        println!("{}", matrix_check.to_string());

        assert_eq!(multiply_scalar_matrix.to_string(), matrix_check.to_string());
    }

    #[test]
    fn sum_test() {
        let mut matrix = Matrix::new(2, 2);
        matrix.set(0, 0, 1.0);
        matrix.set(0, 1, 2.0);
        matrix.set(1, 0, 3.0);
        matrix.set(1, 1, 4.0);

        let sum = matrix.sum(0, 0, matrix.rows, matrix.cols);
        let sum_check = 10.0;
        println!("{}", {sum});
        println!("{}", {sum_check});

        assert_eq!(sum, sum_check);
    }
    
    #[test]
    fn randomize_rows_together_test() {
        let mut matrix1 = Matrix::new(4, 1);
        let mut matrix2 = Matrix::new(4, 1);
        matrix1.set(0, 0, 1.0);
        matrix1.set(1, 0, 2.0);
        matrix1.set(2, 0, 3.0);
        matrix1.set(3, 0, 4.0);

        matrix2.set(0, 0, 1.0);
        matrix2.set(1, 0, 2.0);
        matrix2.set(2, 0, 3.0);
        matrix2.set(3, 0, 4.0);

        crate::matrix::randomize_rows_together(&mut matrix1, &mut matrix2);

        println!("Matrix 1:\n{}\nMatrix 2:\n{}", matrix1.to_string(), matrix2.to_string());

        assert_eq!(matrix1.to_string(), matrix2.to_string());
    }
}
