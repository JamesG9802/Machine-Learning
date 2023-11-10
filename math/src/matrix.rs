use std::clone::Clone;
use rand::{self, seq::SliceRandom};

/// A data structure to handle matrices.
#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,

    /// <summary>
    /// The matrix is stored as 1D array.
    /// </summary>
    data: Vec<f64>,
}

impl Matrix {

    /// Creates an MxN Matrix object. Does not check if the matrix is valid.
    /// # Example
    /// ```rust
    /// use dragon_math::matrix::Matrix; 
    /// let matrix: Matrix = Matrix::new(2,4);  //  Create a 2x4 matrix.
    /// ```
    pub fn new(rows: usize, cols: usize) -> Matrix {
        let mut data_list: Vec<f64> = Vec::new();
        for _i in 0..rows * cols {
            data_list.push(0.0);
        }
        return Matrix {
            rows: rows,
            cols: cols,
            data: data_list,
        };
    }

    /// Returns a copy of the matrix.
    pub fn clone(&self) -> Self {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.clone(), // Clone the internal Vec<f64>
        }
    }

    /// Returns the value at Matrix[row][col]. Does not check if indices are in bounds.
    /// # Example
    /// ```rust
    /// use dragon_math::matrix::Matrix; 
    /// let mut matrix: Matrix = Matrix::new(2,4);
    /// matrix.set(0, 0, 1.0);
    /// assert_eq!(matrix.at(0, 0), 1.0);
    /// ```
    pub fn at(&self, row: usize, col: usize) -> f64 {
        return self.data[row * self.cols + col];
    }

    /// Sets the value at Matrix[row][col]. Does not check if indices are in bounds.
    /// # Example
    /// ```rust
    /// use dragon_math::matrix::Matrix; 
    /// let mut matrix: Matrix = Matrix::new(2,4);
    /// matrix.set(0, 0, 1.0);
    /// assert_eq!(matrix.at(0, 0), 1.0);
    /// ```
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }
    
    /// Returns a new matrix that is a submatrix of this matrix. Does not check if indices are in bounds.
    /// # Example
    /// ```rust
    /// use dragon_math::matrix::Matrix; 
    /// let mut matrix: Matrix = Matrix::new(2,4);
    /// matrix.set(0, 0, 1.0);
    /// let mut submatrix: Matrix = matrix.submatrix(0, 0, 1, 2);   //  Returns a 1x2 matrix
    /// assert_eq!(submatrix.rows, 1);
    /// assert_eq!(submatrix.cols, 2);
    /// assert_eq!(submatrix.at(0, 0), 1.0);
    /// ```
    pub fn submatrix(&self, start_row: usize, start_col: usize, end_row: usize, end_col: usize) -> Matrix {
        let mut new_data: Vec<f64> = Vec::new();
        for i in start_row..end_row {
            for j in start_col..end_col {
                new_data.push(self.at(i, j));
            }
        }
        return Matrix {
            rows: (end_row - start_row),
            cols: (end_col - start_col),
            data: new_data,
        }
    }

    /// Returns the transpose of the matrix.
    /// # Example
    /// ```rust
    /// use dragon_math::matrix::Matrix; 
    /// let mut matrix: Matrix = Matrix::new(2,2);
    /// matrix.set(1, 0, 1.0);
    /// matrix.set(0, 1, 2.0);
    /// matrix = matrix.transpose();
    /// assert_eq!(matrix.at(1, 0), 2.0);
    /// assert_eq!(matrix.at(0, 1), 1.0);
    /// ```
    pub fn transpose(&self) -> Matrix {
        let mut new_data: Vec<f64> = Vec::new();
        for j in 0..self.cols {
            for i in 0..self.rows {
                new_data.push(self.at(i, j));
            }
        }
        return Matrix {
            rows: self.cols,
            cols: self.rows,
            data: new_data,
        }
    }

    /// Returns the rows of the matrix in a random order. Column order is preserved.
    /// # Example
    /// ```rust
    /// use dragon_math::matrix::Matrix; 
    /// let mut matrix: Matrix = Matrix::new(2,2);
    /// matrix = matrix.randomize_rows();
    /// ```
    pub fn randomize_rows(&self) -> Matrix {
        let mut shuffled_rows: Vec<usize> = (0..self.rows).collect();
        shuffled_rows.shuffle(&mut rand::thread_rng());
        let mut new_data: Vec<f64> = Vec::new();
        for row in shuffled_rows {
            for col in 0..self.cols {
                new_data.push(self.at(row, col))
            }
        }
        return Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data,
        }
    }

    /// Creates an matrix equal to the dot product of this and the other matrix. Does not check if it is possible.
    /// # Example
    /// ```rust
    /// use dragon_math::matrix::Matrix; 
    /// let mut A : Matrix = Matrix::new(2,2);
    /// let mut B : Matrix = Matrix::new(2,2);
    /// let mut C : Matrix = A.dot(&B);
    /// ```
    pub fn dot(&self, other: &Matrix) -> Matrix {
        let mut new_matrix = Matrix::new(self.rows, other.cols);
        for i in 0..new_matrix.rows {
            for j in 0..new_matrix.cols {
                //  Each index is corresponds to the ith row of matrix1 * the jth column of matrix2.
                let mut value: f64 = new_matrix.at(i, j);
                for rowcol in 0..self.cols {
                    value += self.at(i, rowcol) * other.at(rowcol, j);
                }
                new_matrix.set(i, j, value);
            }
        }
        return new_matrix;
    }

    /// Creates an matrix equal to the sum of this and the other matrix. Does not check if it is possible.
    /// # Example
    /// ```rust
    /// use dragon_math::matrix::Matrix; 
    /// let mut A : Matrix = Matrix::new(2,2);
    /// let mut B : Matrix = Matrix::new(2,2);
    /// let mut C : Matrix = A.add(&B);
    /// ```
    pub fn add(&self, other: &Matrix) -> Matrix {
        let mut new_matrix = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                new_matrix.set(i, j, self.at(i, j) + other.at(i, j));
            }
        }
        return new_matrix;
    }

    /// Returns a matrix where each value in the matrix is multiplied by a scalar value.
    /// # Example
    /// ```rust
    /// use dragon_math::matrix::Matrix; 
    /// let mut matrix : Matrix = Matrix::new(2,2);
    /// matrix = matrix.multiply_scalar(2.0);
    /// ```
    pub fn multiply_scalar(&self, scalar: f64) -> Matrix {
        let mut new_data = self.data.clone();
        for i in 0..self.rows * self.cols {
            new_data[i] = new_data[i] * scalar;
        }
        return Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data
        }
    }

    /// Returns a matrix where each value in the matrix is multiplied by a scalar value.
    /// # Example
    /// ```rust
    /// use dragon_math::matrix::Matrix; 
    /// let mut matrix : Matrix = Matrix::new(2,2);
    /// matrix = matrix.add_scalar(-1.0);
    /// ```
    pub fn add_scalar(&self, scalar: f64) -> Matrix {
        let mut new_data = self.data.clone();
        for i in 0..self.rows * self.cols {
            new_data[i] = new_data[i] + scalar;
        }
        return Matrix {
            rows: self.rows,
            cols: self.cols,
            data: new_data
        }
    }

    /// Returns the sum of all indices from [start_row][start_col] to [end_row][end_col]
    /// ```rust
    /// use dragon_math::matrix::Matrix; 
    /// let mut matrix : Matrix = Matrix::new(2,2);
    /// let sum: f64 = matrix.sum(0, 0, 2, 2);
    /// ```
    pub fn sum(&self, start_row: usize, start_col: usize, end_row: usize, end_col: usize) -> f64 {
        let mut sum: f64 = 0.0;
        for i in start_row..end_row {
            for j in start_col..end_col {
                sum += self.data[i * self.cols + j];
            }
        }
        return sum;
    }
    /// Returns the matrix as a String.
    /// # Example
    /// ```rust
    /// use dragon_math::matrix::Matrix; 
    /// let mut matrix : Matrix = Matrix::new(2,2);
    /// println!("{}", matrix.to_string());
    /// ```
    pub fn to_string(&self) -> String {
        let mut string : String = String::from("");
        for row in 0..self.rows {
            for col in 0..self.cols {
                string.push_str(self.at(row, col).to_string().as_str());
                string.push_str(" ");
            }
            string.push_str("\n");
        }
        return string;
    }
}

/// Randomizes the rows of both matrices in the same way. Does not check if the matrices have the same number of rows.
/// # Example
/// ```rust
/// use dragon_math::matrix::Matrix; 
/// use dragon_math::matrix;
/// let mut A : Matrix = Matrix::new(2,2);
/// let mut B : Matrix = Matrix::new(2,2);
/// matrix::randomize_rows_together(&mut A, &mut B);
/// ```
pub fn randomize_rows_together(matrix1: &mut Matrix, matrix2: &mut Matrix) {
    let mut shuffled_rows: Vec<usize> = (0..matrix1.rows).collect();
    shuffled_rows.shuffle(&mut rand::thread_rng());
    let mut matrix1_new_data: Vec<f64> = Vec::new();
    let mut matrix2_new_data: Vec<f64> = Vec::new();

    for row in shuffled_rows {
        for col in 0..matrix1.cols {
            matrix1_new_data.push(matrix1.at(row, col));
        }
        for col in 0..matrix2.cols {
            matrix2_new_data.push(matrix2.at(row, col));
        }
    }
    matrix1.data = matrix1_new_data;
    matrix2.data = matrix2_new_data;
}