use super::matrix::Matrix;

use std::{
	fs::File,
	io::{Read, Write},
};

use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};

pub struct Network{
	layers : Vec<usize>,
	data: Vec<Matrix>,

	//Keep in mind that weights[0] will convert layers[0] to layers[1]
	weights: Vec<Matrix>,
	biases: Vec<Matrix>,
	learning_rate: f64
}

#[derive(Serialize, Deserialize)]
struct SaveData{
	weights: Vec<Vec<Vec<f64>>>,
	biases: Vec<Vec<Vec<f64>>>,
}

impl Network {

	pub fn create(layers: Vec<usize>, learning_rate: f64) -> Network{
		let mut weights: Vec<Matrix> = Vec::new();
		let mut biases: Vec<Matrix> = Vec::new();
		let data: Vec<Matrix> = Vec::new();

		for i in 0..layers.len() - 1 {
			weights.push(Matrix::rand_matrix(layers[i + 1], layers[i]));
			biases.push(Matrix::rand_matrix(layers[i + 1], 1));
		}

		return Network{ layers: layers, data: data, weights: weights, biases: biases, learning_rate: learning_rate * 0.01 };
	}

	pub fn run(&mut self, input: Vec<f64>) -> Vec<f64> {
		if input.len() != self.layers[0] {
			panic!("Wrong input lenght");
		}

		let mut cur = Matrix::from_data(vec![input]).transpose();

		self.data.clear();
		self.data.push(cur.clone());

		for i in 0..self.weights.len() {
			cur = Matrix::add(&Matrix::multiply(&self.weights[i], &cur), &self.biases[i]).apply_func(&|x| 1.0 / ( 1.0 + (-x).exp() ) );

			self.data.push(cur.clone());
		}

		return cur.transpose().data[0].to_owned();
	}

	pub fn train(&mut self, input: &Vec<Vec<f64>>, expected: &Vec<Vec<f64>>){
		for i in 0..input.len() {
			let output = self.run(input[i].clone());
			self.backpropagate(output, expected[i].clone());
		}
	}

	pub fn backpropagate(&mut self, outputs: Vec<f64>, expected: Vec<f64>){
		let out = Matrix::from_data(vec![outputs]).transpose();
		let mut err = Matrix::subtract(&Matrix::from_data(vec![expected]).transpose(), &out).apply_func(&|x| 2.0 * x); // Del(cost)/del (activation of last layer)
		let mut grad = out.apply_func(&|x| (1.0 - x) * x );

		for i in (0..self.layers.len() - 1).rev() {

			grad = Matrix::dot(&grad, &err);

			self.weights[i] = Matrix::subtract(&self.weights[i], &Matrix::multiply(&grad, &self.data[i].transpose()).apply_func(&|x| self.learning_rate * x));
			self.biases[i] = Matrix::subtract(&self.biases[i], &grad.apply_func(&|x| self.learning_rate * x));

			grad = self.data[i].apply_func(&|x| (1.0 - x) * x );
			err = Matrix::multiply(&self.weights[i].transpose(), &err);
		}
	}


	// ignore this 
	pub fn back_prop(&mut self, outputs: Vec<Vec<f64>>, expected: Vec<Vec<f64>>){

		let w_c_avg: Vec<Matrix> = self.weights.clone().into_iter().map(|mat: Matrix| Matrix::zero_matrix(mat.rows, mat.cols) ).collect();
		let b_c_avg: Vec<Matrix> = self.biases.clone().into_iter().map(|mat: Matrix| Matrix::zero_matrix(mat.rows, mat.cols) ).collect();

		for i in 0..outputs.len() {
			let out = Matrix::from_data(vec![outputs[i]]).transpose();
			let mut err = Matrix::subtract(&Matrix::from_data(vec![expected[i]]).transpose(), &out).apply_func(&|x| 2.0 * x);

			let mut grad = out.apply_func(&|x| (1.0 - x) * x);

			for j in 0..self.layers.len()
		}
	}

	pub fn save(&self){
		let mut file = File::create("parameters.json").expect("Failed to create file");

		file.write_all(
			json!({
				"weights": self.weights.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>(),
				"biases": self.biases.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>()
			}).to_string().as_bytes(),
		).expect("Unable to write to save file");
	}

	pub fn load(&mut self){
		let mut file = File::open("parameters.json").expect("Failed to create file");

		let mut buffer = String::new();

		file.read_to_string(&mut buffer).expect("Unable to read save file");

		let save_data: SaveData = from_str(&buffer).expect("Unable to serialize save data");

		let mut weights = vec![];
		let mut biases = vec![];

		for i in 0..self.layers.len() - 1 {
			weights.push(Matrix::from_data(save_data.weights[i].clone()));
			biases.push(Matrix::from_data(save_data.biases[i].clone()));
		}

		self.weights = weights;
		self.biases = biases;
	}
}
