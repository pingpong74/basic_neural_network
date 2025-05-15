use rand::Rng;

#[derive(Clone)]
pub struct Matrix {
	pub rows: usize,
	pub cols: usize,
	pub data: Vec<Vec<f64>>
}

impl Matrix {
	pub fn zero_matrix(rows: usize, cols: usize) -> Matrix{
		return Matrix { rows: rows, cols: cols, data: vec![vec![0.0; cols]; rows]};
	}

	pub fn rand_matrix(rows: usize, cols: usize) -> Matrix{
		let mut rng = rand::thread_rng();

		let mut res = Matrix::zero_matrix(rows, cols);

		for i in 0..rows {
			for j in 0..cols {
				res.data[i][j] = rng.gen_range(-1.0..1.0);
			}
		}

		return res;
	}

	pub fn from_data(data: Vec<Vec<f64>>) -> Matrix{
		return Matrix{rows: data.len(), cols: data[0].len(), data: data};
	}

	pub fn add(a: &Matrix, b: &Matrix) -> Matrix{
		if a.rows != b.rows || a.cols != b.cols {
			panic!("Incorrect matrix addition");
		}

		let mut res = Matrix::zero_matrix(a.rows, a.cols);

		for i in 0..a.rows {
			for j in 0..a.cols {
				res.data[i][j] = a.data[i][j] + b.data[i][j];
			}
		}

		return res;
	}

	pub fn subtract(a: &Matrix, b: &Matrix) -> Matrix{
		if a.rows != b.rows || a.cols != b.cols {
			panic!("Incorrect matrix subtraction");
		}

		let mut res = Matrix::zero_matrix(a.rows, a.cols);

		for i in 0..a.rows {
			for j in 0..a.cols {
				res.data[i][j] = a.data[i][j] - b.data[i][j];
			}
		}

		return res;
	}

	pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix{
		if a.cols != b.rows {
			panic!("Incorrect matrix multiplication");
		}

		let mut res = Matrix::zero_matrix(a.rows, b.cols);

		for i in 0..a.rows{
			for j in 0..b.cols{
				let mut s: f64 = 0.0;

				for k in 0..a.cols {
					s += a.data[i][k] * b.data[k][j];
				}

				res.data[i][j] = s;
			}
		}

		return res;
	}

	pub fn apply_func(&self, function: &dyn Fn(f64) -> f64) -> Matrix{
		let mut res = Matrix::zero_matrix(self.rows, self.cols);

		for i in 0..self.rows {
			for j in 0..self.cols {
				res.data[i][j] = function(self.data[i][j]);
			}
		}

		return res;
	}

	pub fn transpose(&self) -> Matrix{
		let mut res = Matrix::zero_matrix(self.cols, self.rows);

		for i in 0..self.rows {
			for j in 0..self.cols {
				res.data[j][i] = self.data[i][j];
			}
		}

		return res;
	}

	pub fn dot(a: &Matrix, b: &Matrix) -> Matrix {
		if a.rows != b.rows || a.cols != b.cols {
			panic!("Incorrect matrix hadmard");
		}

		let mut res = Matrix::zero_matrix(a.rows, a.cols);

		for i in 0..a.rows {
			for j in 0..a.cols {
				res.data[i][j] = a.data[i][j] * b.data[i][j];
			}
		}

		return res;
	}
}