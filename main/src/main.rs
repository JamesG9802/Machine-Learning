use std::env;
use std::fs;
use std::io;

use dragon_math;
use dragon_math::matrix::Matrix;

use dragon_model;
use dragon_model::model as model;
use dragon_model::linear_regression::LinearRegression as LinearRegression;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut learning_rate: f64 = 0.01;
    let mut max_iterations: f64 = 10.0;
    let mut batch_size: f64 = 50.0;
    if args.len() <= 1 {
        return;
    }
    if args.len() > 2 {
        learning_rate = (&args[2]).parse::<f64>().unwrap();
    }
    if args.len() > 3 {
        max_iterations = (&args[3]).parse::<f64>().unwrap();
    }
    if args.len() > 4 {
        batch_size = (&args[4]).parse::<f64>().unwrap();
    }

    //  Test with IRIS dataset.
    let file: &String = &args[1];
    println!("{}", file);
    let mut binding = fs::read_to_string(file).expect("Something went wrong reading the file.");
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
        model::batch_gradient_descent_l2,
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
