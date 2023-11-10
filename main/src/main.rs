use std::env;
use std::fs;
use std::io;

use dragon_math;
use dragon_math::matrix::Matrix;

use dragon_model;
use dragon_model::model as model;
use dragon_model::linear_regression::LinearRegression as LinearRegression;

/// Returns the index of the first occurence of the substring in the string. 
/// Returns -1 if it does not exist.
fn index_of(string: &str, substring: &str) -> i64{
    for i in string.match_indices(substring).map(|(index, _)| index).collect::<Vec<usize>>(){
        return i as i64;
    }
    return -1;
}
/// Returns a new substring from the beginning index to the end_index - 1.
/// Panics if indices are out of bounds
fn substring(string: &str, begin_index: usize, end_index: usize) -> String {
    let mut new_string: String = String::new();
    if begin_index >= end_index || end_index > string.chars().count() {
        panic!("Substring: Index out of bounds. begin_index={}, end_index={}", begin_index, end_index);
    }
    for i in begin_index..end_index {
        new_string.push(string.chars().nth(i).unwrap());
    }
    return new_string;
}
fn main() {
    let args: Vec<String> = env::args().collect();
    let mut input_file: String = String::new();
    let mut learning_rate: f64 = 0.01;
    let mut max_iterations: f64 = 10.0;
    let mut batch_size: f64 = 50.0;
    
    //  Parse Command Line Arguments
    for i in 0..args.len() {
        let argument: &String = &args[i];
        if argument == "-h" || argument == "--help" {
            println!("{}{}{}{}{}{}", "usage: [-h | --help]\n",
                "[-f:<input_file> | --file:<input_file>]\n",
                "[-l:<learning_rate> | --learningrate:<learning_rate>]\n",
                "[-n:<max_iterations> | --maxiterations:<max_iterations>]\n",
                "[-b:<batch_size> | --batchsize:<batch_size>]\n",
                "[-m:<model> | --model:<model>]");
            return;
        }
        else if argument.starts_with("-f:") || argument.starts_with("--file:") {
            let index: usize = index_of(&argument, ":") as usize + 1;
            if index >= argument.chars().count() {
                println!("Expecting [-l:<learning_rate> | --learningrate:<learning_rate>]");
                return;
            }
            input_file = substring(
                argument, 
                index_of(&argument, ":") as usize + 1, 
                argument.chars().count());
        }
        else if argument.starts_with("-l:") || argument.starts_with("--learningrate:") {
            let index: usize = index_of(&argument, ":") as usize + 1;
            if index >= argument.chars().count() {
                println!("Expecting [-l:<learning_rate> | --learningrate:<learning_rate>]");
                return;
            }
            learning_rate = substring(
                argument, 
                index_of(&argument, ":") as usize + 1, 
                argument.chars().count())
            .parse::<f64>().unwrap();   
        }
        else if argument.starts_with("-n:") || argument.starts_with("--maxiterations:") {
            let index: usize = index_of(&argument, ":") as usize + 1;
            if index >= argument.chars().count() {
                println!("Expecting [-n:<max_iterations> | --maxiterations:<max_iterations>]");
                return;
            }
            max_iterations = substring(
                argument, 
                index_of(&argument, ":") as usize + 1, 
                argument.chars().count())
            .parse::<f64>().unwrap();   
        }
        else if argument.starts_with("-b:") || argument.starts_with("--batchsize:") {
            let index: usize = index_of(&argument, ":") as usize + 1;
            if index >= argument.chars().count() {
                println!("Expecting [-b:<batch_size> | --batchsize:<batch_size>]");
                return;
            }
            batch_size = substring(
                argument, 
                index_of(&argument, ":") as usize + 1, 
                argument.chars().count())
            .parse::<f64>().unwrap();   
        }
    }

    //  Test with IRIS dataset.
    let mut binding = fs::read_to_string(input_file).expect("Something went wrong reading the file.");
    binding = binding.trim().to_string();
    let contents = binding.split("\n");
    let mut first_line: bool = true;
    let mut line_count = 0;

    let mut dataset: Matrix = Matrix::new(contents.clone().collect::<Vec<&str>>().len() - 1, 4);
    for line in contents {
        if first_line {
            first_line = false;
            continue;
        }
        let features = line.split(",").collect::<Vec<&str>>();
        dataset.set(line_count, 0, features[0].parse::<f64>().unwrap());
        dataset.set(line_count, 1, features[1].parse::<f64>().unwrap());
        dataset.set(line_count, 2, features[2].parse::<f64>().unwrap());
        dataset.set(line_count, 3, features[3].parse::<f64>().unwrap());
        line_count += 1;
    }

    let mut inputs: Matrix = Matrix::new(dataset.rows, 3);
    let mut outputs: Matrix = Matrix::new(dataset.rows, 1);

    dataset = dataset.randomize_rows();
    for i in 0..dataset.rows {
        inputs.set(i, 0, dataset.at(i, 0));
        inputs.set(i, 1, dataset.at(i, 1));
        inputs.set(i, 2, dataset.at(i, 2));
        outputs.set(i, 0, dataset.at(i, 3));
    }
    let training_inputs = inputs.submatrix(0, 0, ((inputs.rows as f64) * 0.8) as usize, inputs.cols);
    let training_outputs = outputs.submatrix(0, 0, ((outputs.rows as f64) * 0.8) as usize, outputs.cols);
    let test_inputs = inputs.submatrix(((inputs.rows as f64) * 0.8) as usize, 0, inputs.rows, inputs.cols);
    let test_outputs = outputs.submatrix(((outputs.rows as f64) * 0.8) as usize, 0, outputs.rows, outputs.cols);


    let linear_regression: LinearRegression = LinearRegression::new(3);
    let mut model = linear_regression.model;
    model.train(
        training_inputs,
        training_outputs,
        vec![learning_rate, max_iterations, batch_size],
        model::stochastic_gradient_descent_l2,
    );

    println!("Loss against test dataset: {}", model.get_loss(&test_inputs, &test_outputs, model::loss_squared));

    let mut line: String;
    let mut inputs: Matrix;
    println!("Model training complete. Entering test mode:");
    loop {
        line = String::new();
        inputs = Matrix::new(1, 3);
        println!("Enter a sepal_length: ");
        io::stdin().read_line(&mut line).unwrap();
        inputs.set(0, 0, line.trim().parse::<f64>().unwrap());
        println!("Enter a sepal_width: ");
        line = String::new();
        io::stdin().read_line(&mut line).unwrap();
        inputs.set(0, 1, line.trim().parse::<f64>().unwrap());

        println!("Enter a petal_length: ");
        line = String::new();
        io::stdin().read_line(&mut line).unwrap();
        inputs.set(0, 2, line.trim().parse::<f64>().unwrap());

        println!("Predicting petal_width: {}", model.predict(&inputs));
    }
}
