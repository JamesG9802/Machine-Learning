use std::env;
use std::fs;
use std::io;

use dragon_math as math;
use math::matrix::Matrix;

use dragon_model as model;
use model::linear_regression::LinearRegression;
fn main() {
    let args: Vec<String> = env::args().collect();
    let mut learning_rate: f64 = 0.01;
    let mut max_iterations: f64 = 10.0;
    if args.len() <= 1 {
        return;
    }
    if args.len() > 2 {
        learning_rate = (&args[2]).parse::<f64>().unwrap();
    }
    if args.len() > 3 {
        max_iterations = (&args[3]).parse::<f64>().unwrap();
    }

    //  Test with IRIS dataset.
    let file: &String = &args[1];
    println!("{}", file);
    let mut binding = fs::read_to_string(file).expect("Something went wrong reading the file.");
    binding = binding.trim().to_string();
    let contents = binding.split("\n");
    let mut inputs: Matrix = Matrix::new(contents.clone().collect::<Vec<&str>>().len() - 1, 3);
    let mut outputs: Matrix = Matrix::new(inputs.rows, 1);
    let mut first_line: bool = true;
    let mut line_count = 0;

    for line in contents {
        if first_line {
            first_line = false;
            continue;
        }
        let features = line.split(",").collect::<Vec<&str>>();
        inputs.set(line_count, 0, features[0].parse::<f64>().unwrap());
        inputs.set(line_count, 1, features[1].parse::<f64>().unwrap());
        inputs.set(line_count, 2, features[2].parse::<f64>().unwrap());
        //    inputs.set(line_count, 2, features[2]);
        //    inputs.set(line_count, 3, features[3]);
        outputs.set(line_count, 0, features[3].parse::<f64>().unwrap());
        //    println!("+{}| {} {} {} {} {}+", line_count, features[0],features[1],features[2],features[3],features[4]);
        line_count += 1;
    }
    //  println!("{}", inputs.to_string());
    //  println!("{}", outputs.to_string());

    let mut model: LinearRegression = LinearRegression::new(3);
    model.train(
        inputs,
        outputs,
        vec![learning_rate, max_iterations],
        model::linear_regression::batch_gradient_descent_l1,
    );

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
