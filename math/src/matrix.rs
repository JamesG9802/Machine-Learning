/// <summary>
/// A data structure to handle matrices.
/// </summary> 
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,

    data: Vec<f64>,
}

impl Matrix {

    /// <summary>
    /// Creates an MxN Matrix object. Does not check if the matrix is valid.
    /// </summary>
    pub fn new(rows: usize, cols: usize) -> Matrix {
        let mut data_list: Vec<f64> = Vec::new();
        for _i in 0..rows * cols {
            data_list.push(0.0);
        }
        return Matrix {
            data: data_list,
            rows: rows,
            cols: cols,
        };
    }
    
    /// <summary>
    /// Returns the value at Matrix[row][col]. Does not check if indices are in bounds.
    /// </summary>
    pub fn at(&self, row: usize, col: usize) -> f64 {
        return self.data[row * self.cols + col];
    }

    /// <summary>
    /// Sets the value at Matrix[row][col]. Does not check if indices are in bounds.
    /// </summary>
    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[row * self.cols + col] = value;
    }
    
    /// <summary>
    /// Creates an matrix equal to the dot product of matrix1 and matrix2.
    /// </summary>
    pub fn dot(&self, other: Matrix) -> Matrix{
        let mut new_matrix = Matrix::new(self.cols, other.rows);
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

    /// <summary>
    /// Returns the matrix as a String.
    /// </summary>
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