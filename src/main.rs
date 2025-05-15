mod matrix;

mod network;
use network::Network;

use std::io;

use mnist::{Mnist, MnistBuilder};
use rand::Rng;

fn create_training_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>){
	let Mnist {
        trn_img, trn_lbl,
        ..
    } = MnistBuilder::new().label_format_digit().training_set_length(60_000).finalize();

    let input: Vec<Vec<f64>> = trn_img.chunks(784).map(|chunk| {
    	chunk.iter().map(|&x| x as f64 / 255.0).collect()
    } ).collect();

    let expected: Vec<Vec<f64>> = trn_lbl.iter().map( |&val|{
    	let mut temp: Vec<f64> = vec![0.0; 10];
    	temp[val as usize] = 1.0;
    	temp
    }).collect();

    (input, expected)
}

fn create_eval_data() -> (Vec<Vec<f64>>, Vec<u8>){
	let Mnist {
        tst_img, tst_lbl,
        ..
    } = MnistBuilder::new().label_format_digit().test_set_length(10_000).finalize();

    let input: Vec<Vec<f64>> = tst_img.chunks(784).map(|chunk| {
    	chunk.iter().map(|&x| x as f64 / 255.0).collect()
    } ).collect();

    (input, tst_lbl)
}

fn main() {
	let layers: Vec<usize> = vec![784, 16, 16, 10];
    let learning_rate = 0.005;
	let epochs = 10;
    let mut network = Network::create(layers.clone(), learning_rate);

    println!("Type train to train or eval to evaluate. If the number of layers have been changed, type new and then eval");

    loop {

        let mut job = String::new();
        io::stdin().read_line(&mut job).expect("Failed to read line");
        let job = job.trim();

        if job == "train" {
    	   let (mut input,mut expected) = create_training_data();

           let mut rng = rand::thread_rng();

           print!("Percentage training done: 0%");

            for i in 0..epochs {

                for i in 0..input.len() {
                    for j in 0..input.len() {
                        if rng.gen_range( 0.0..=1.0) > 0.5 {
                            expected.swap(i, j);
                            input.swap(i, j);
                        }
                    }
                }

                network.train(&input, &expected);

                print!("\rPercentage training done: {:?}%   ", (i + 1) * 100 / epochs);
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }

            println!("Training finished");

    	   network.save();
        }
        else if job == "new" {
            network = Network::create(layers.clone(), learning_rate);
            network.save();
        }
        else if job == "eval" {
    	   network.load();

    	   let (input, target) = create_eval_data();

    	   let mut correct = 0;

            for i in 0..200 {
    	   	   let output = network.run(input[i].clone());

    		   let mut guess = 0;

    		    for i in 0..output.len() {
    			    if output[i] > output[guess] {
    				    guess = i;
    			    }
    		    }

    		    println!("Predicted: { }     Expected: { }", guess, target[i]);

    		    if guess as u8 == target[i] {
    			    correct += 1;
    		    }
            }

            let accuracy = (correct as f64) * 0.5;
            println!("{:?}", accuracy);
    	}
        else if job == "exit" {
            break;
        }
        else {
    	   println!("Invalid input");
        }
    }
}
