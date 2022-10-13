mod matrix; //self made "library" for handling matrices (I especially like the customizability of display, so you can really visualize them)
use matrix::*; //use all functions and structs and whatever from here
use std::{fmt, ops::Range, fs, io::{Read, Write}, thread, time::Instant}; //display, arguments as ranges, arguments as paths, read-file, read-file trait needed
use rand::{thread_rng, Rng}; //random numbers to initialize the weights and biases
use serde::{Serialize, Deserialize};


fn main() {
    let images = open("t10k-images.idx3-ubyte");
    let labels = open("t10k-labels.idx1-ubyte");

    let until: usize = 9999; 
    let batch_size: usize = 100;
    let threads: usize = 10;
    let batch_num = until / batch_size;
    let bits = batch_size / threads;

    let input = make_samples(&images, 0..until, 28*28, 17); // 17 is because the first seventeen numbers are just basically data about the file, and there are 28 by 28 images
    let labels = make_labels(&labels, 0..until);
    
    println!("Done making labels and inputs");
    
    //let mut guesser = Network::new(vec![784, 16, 16, 10]);
    let mut guesser = load_network("guesser9000");
    println!("Done loading the network");
    
    let now = Instant::now();

    //for batch in 0..batch_num {    
    //    for _ in 0..1000 { // How many times to do this
//
    //        let mut handles = Vec::new();
    //        for i in 1..=threads {
    //            
    //            let inp = input.clone();
    //            let lab = labels.clone();
    //            let mut gues = guesser.clone();
//
    //            let handle = thread::spawn(move || {
    //                let range = (bits * (i - 1) + (batch_size * batch) ) .. ((bits * i) + (batch_size * batch));
    //                calculate_batch(
    //                    inp[range.clone()].to_vec(), 
    //                    lab[range].to_vec(), 
    //                    &mut gues
    //                )
    //            });
    //            handles.push(handle);
    //        }
//
    //        let mut optimizations: Vec<Derivatives> = Vec::new();
//
    //        for handle in handles {
    //            optimizations.append(&mut handle.join().unwrap());
    //        }
//
    //        let derivative_average = average_derivatives(optimizations);
//
    //        guesser.update(derivative_average);
    //    }
    //}

    let elapsed = now.elapsed();
    

    ////------------------------ 2
    //test(input[1].clone(), 2, &mut guesser);
    //
    ////------------------------ 7
    //test(input[0].clone(), 7, &mut guesser);
//
    ////------------------------ 1
    //test(input[2].clone(), 1, &mut guesser);
    //
    //test(input[3].clone(), 1, &mut guesser);
    //
    //test(input[4].clone(), 1, &mut guesser);
    
    print!("Please give an index number out of the training data to test the network on: ");
    std::io::stdout().flush().unwrap();
    let mut select = String::new();
    std::io::stdin().read_line(&mut select).unwrap();
    let answer: usize = select.trim().parse().unwrap();
    println!("You've selected the {}th sample", answer);
    test(input[answer].clone(), labels[answer], &mut guesser);
    

    println!("Time to execute: {:.2?}", elapsed);

    //-- save
    //save_network("guesser9000_2", &guesser);
}

fn test(input: Matrix, exp: u8, network: &mut Network) {
    let expectation = network.make_expected_vector(exp);
    input.print_image();
    network.propagate(input.pancake());
    
    println!("Expected: {}:", exp);
    network.print_output();
    
    let cost = Network::calculate_cost(network.h_o.last().unwrap().main.apply_new(&sigmoid).entries.clone(), expectation.clone(), &Network::cost_function);
    println!("Cost: {}", cost);
}

fn calculate_batch(batch: Vec<Matrix>, labels: Vec<u8>, network: &mut Network) -> Vec<Derivatives> {
    let mut optimizations: Vec<Derivatives> = Vec::new();
    for i in 0..batch.len() {
        let data = batch[i].clone();
        let expectation = network.make_expected_vector(labels[i]);
        
        network.propagate(data.pancake());
        
        //let cost = Network::calculate_cost(guesser.h_o.last().unwrap().main.apply_new(&sigmoid).entries.clone(), expectation.clone(), &Network::cost_function);

        let derivs = network.calculate_derivatives(Matrix::new(expectation.clone(), expectation.len(), 1));

        optimizations.push(derivs);
    }
    optimizations
}

fn average_derivatives(derivs: Vec<Derivatives>) -> Derivatives {
    let mut opt = derivs[0].clone();
    let len = derivs.len();
    for i in 1..len {
        //println!("current id: {}", i);
        opt = Derivatives::add(opt, derivs[i].clone());
    }
    let opt = Derivatives::apply_division(opt, (len as f64) * 10.0);
    opt
}

#[derive(Serialize, Deserialize, Debug)]
struct GuessAndTruth(Network, u8);

fn save_network(name: &str, network: &Network) {
    let saved = serde_json::to_string(network).unwrap();
    let mut file = fs::File::create(String::from("saves/") + name).expect("creation of save file failed");
    file.write_all(saved.as_bytes()).expect("writing to save file failed");
}

fn load_network(name: &str) -> Network {
    let mut file = fs::File::open(String::from("saves/") + name).expect("Unable to open file"); 
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Unable to load");
    serde_json::from_str(&contents).unwrap()
}

fn open(name: &str) -> Vec<u8> {
    let mut contents = Vec::new();
    let mut file = fs::File::open(String::from("assets/") + name).expect("Unable to open file"); // read as bytes instead of string because it's not UTF-8 I guess?
    file.read_to_end(&mut contents).expect("Unable to read");
    contents
}

fn make_labels(contents: &Vec<u8>, range: Range<usize>) -> Vec<u8> {
    let mut labels = Vec::new();
    for x in range {
        labels.push(contents[x + 8]);
    }
    labels
}

fn make_samples(contents: &Vec<u8>, range: Range<usize>, dimensions: usize, start: usize) -> Vec<Matrix> {
    let mut input = Vec::<Matrix>::new();
    let mut nums: Vec<usize> = Vec::new();
    for x in range {nums.push(x * dimensions + start)}
    for elem in nums.iter() {
        let mut im: Vec<f64> = Vec::new();
        for e in *elem..*elem + dimensions {
            im.push(contents[e] as f64);
        }
        input.push(Matrix {entries: im, row: 28, col: 28});
    }
    input
}

fn max(a: Vec<f64>) -> u8 {
    let mut i = 0;
    for id in 0..a.len() {
        if a[id] >= a[i] {
            i = id.clone();
        }
    }
    i as u8
}

fn sigmoid(a: f64) -> f64 { // sigmoid function which is a logistic function that in this case approaches 0 with an argument of -infinity and 1 with an argument of infinity. It it exactly 0.5 with an argument of 0.
    1.0 / (1.0 + std::f64::consts::E.powf(-a))
}

fn sigmoid_derivative(a: f64) -> f64 {
    let x = sigmoid(a);
    x * (1.0 - x)
}

fn gen_rands(range: Range<f64>, n: usize) -> Vec<f64> { //the range to generate in and the number of elements it should generate
    let mut vec: Vec<f64> = Vec::new();
    for _ in 0..n {
        vec.push(thread_rng().gen_range(range.clone()));
    }
    vec
}

struct Rac(usize, usize);

#[derive(Serialize, Deserialize, Debug)]
struct Derivatives { //struct for storing the derivatives with respect to the cost function
    dzl: Vec<Matrix>,
    dwl: Vec<Matrix>,
    dbl: Vec<Matrix>
}

impl Derivatives {
    fn new(n: usize) -> Derivatives {
        Derivatives { 
            dzl: vec![Matrix::new_empty(n, 1); n],
            dwl: vec![Matrix::new_empty(n, 1); n], 
            dbl: vec![Matrix::new_empty(n, 1); n] 
        }
    }

    fn add(a: Derivatives, b: Derivatives) -> Derivatives {
        if a.dzl.len() != b.dzl.len() {
            println!("a dzl: {:?}", a.dzl);
            println!("b dzl: {:?}", b.dzl);
            panic!("not the same amount of elements in a and b derivative structs. a = {}, b = {}", a.dzl.len(), b.dzl.len());
        }
        let z_len = a.dzl.len();
        let w_len = a.dwl.len();
        let b_len = a.dbl.len();
        let mut dzl: Vec<Matrix> = Vec::new();
        let mut dwl: Vec<Matrix> = Vec::new();
        let mut dbl: Vec<Matrix> = Vec::new();
        for i in 0..z_len {
            let new = Matrix::add(&a.dzl[i], &b.dzl[i]);
            dzl.push(new)
        }
        for i in 0..w_len {
            let new = Matrix::add(&a.dwl[i], &b.dwl[i]);
            dwl.push(new)
        }
        for i in 0..b_len {
            let new = Matrix::add(&a.dbl[i], &b.dbl[i]);
            dbl.push(new)
        }
        Derivatives { dzl, dwl, dbl }
    }

    fn apply_division(a: Derivatives, div: f64) -> Derivatives {
        let z_len = a.dzl.len();
        let w_len = a.dwl.len();
        let b_len = a.dbl.len();
        let mut dzl: Vec<Matrix> = Vec::new();
        let mut dwl: Vec<Matrix> = Vec::new();
        let mut dbl: Vec<Matrix> = Vec::new();

        for i in 0..z_len {
            let new = a.dzl[i].apply_new_2(div, &divide);
            dzl.push(new)
        }
        for i in 0..w_len {
            let new = a.dwl[i].apply_new_2(div, &divide);
            dwl.push(new)
        }
        for i in 0..b_len {
            let new = a.dbl[i].apply_new_2(div, &divide);
            dbl.push(new)
        }
        Derivatives { dzl, dwl, dbl }
    }
}

impl Clone for Derivatives {
    fn clone(&self) -> Derivatives {
        let z_len = self.dzl.len();
        let w_len = self.dwl.len();
        let b_len = self.dbl.len();
        let mut dzl: Vec<Matrix> = Vec::new();
        let mut dwl: Vec<Matrix> = Vec::new();
        let mut dbl: Vec<Matrix> = Vec::new();

        for i in 0..z_len {
            dzl.push(self.dzl[i].clone());
        }
        for i in 0..w_len {
            dwl.push(self.dwl[i].clone());
        }
        for i in 0..b_len {
            dbl.push(self.dbl[i].clone());
        }
        Derivatives { dzl, dwl, dbl }
    }
}

fn divide(a: f64, b: f64) -> f64 {
    a / b
}

#[derive(Serialize, Deserialize, Debug)]
struct Layer {
    main: Matrix,
    bias: Matrix,
    weights: Matrix,
}

#[derive(Serialize, Deserialize, Debug)]
struct Network {
    input: Matrix,
    h_o: Vec<Layer>
}

impl Layer {
    fn new(prev_layer_row: usize, layer_row: usize, layer_col: usize) -> Layer {
        let bias = Matrix { entries: vec![0.0; layer_row], row: layer_row, col: 1};

        let weights = gen_rands(-1.0..1.0, prev_layer_row * layer_row);
        let weights = Matrix { entries: weights, row: layer_row, col: prev_layer_row};

        Layer {main: Matrix::new_empty(layer_row, layer_col), bias, weights}
    }

    fn propagate(&mut self, prev_main: &Matrix) { //Constructs the main part of the hidden layer from the previous layer
        let weight = &self.weights;
        let main = prev_main.apply_new(&sigmoid);
        let bias = &self.bias;
        //println!("Propagation{} {} {}", weight, main, bias);
        let weighted_sums = weight.multiply(&main);

        let biased_weighted_sums = Matrix::add(&weighted_sums, bias);
        //biased_weighted_sums.apply(&sigmoid);
        self.main = biased_weighted_sums;
    }

    fn clone(&self) -> Layer {
        let main = self.main.clone();
        let bias = self.bias.clone();
        let weights = self.weights.clone();
        Layer { main, bias, weights }
    }
}

impl Network {
    fn new(all_row: Vec<usize>) -> Network {
        let input = Matrix::new_empty(all_row[0], 1);
        
        let mut h_o: Vec<Layer> = Vec::new();
        for id in 1..all_row.len() {
            let new = Layer::new(all_row[id-1], all_row[id], 1);
            h_o.push(new);
            }
        
        Network { input, h_o }
    }

    fn propagate(&mut self, input: Matrix ) {
        self.input = input;
        self.h_o[0].propagate(&self.input);
        for id in 1..self.h_o.len() {
            //println!("id: {}", id);
            let previous = self.h_o[id-1].main.clone();
            self.h_o[id].propagate(&previous)
        }
    }

    fn cost_function(out: &f64, exp: &f64) -> f64 {
        -(exp * (out).ln() + (1.0 - exp) * (1.0 - out).ln())
    }

    fn calculate_cost(output: Vec<f64>, expected: Vec<f64>, cost_function: &dyn Fn(&f64, &f64) -> f64) -> f64 {
        let mut sum: f64 = 0.0;
        for id in 0..=output.len()-1 {
            sum += cost_function(&output[id], &expected[id]);
        }
        sum
    }

    fn make_expected_vector(&self, expected: u8) -> Vec<f64> {
        let mut a = vec![0.0; expected as usize];
        a.append(&mut vec![1.0]);
        a.append(&mut vec![0.0; self.h_o.last().unwrap().main.row - 1 - expected as usize]);
        a
    }

    fn clone(&self) -> Network {
        let input = self.input.clone();
        let mut h_o: Vec<Layer> = Vec::new();
        for id in 0..self.h_o.len() {
            h_o.push( self.h_o[id].clone() )
        }
        Network { input, h_o }
    }

    fn calculate_derivatives(&self, expected: Matrix) -> Derivatives {
        let L: usize = self.h_o.len() - 1;

        let mut dzl: Vec<Matrix> = Vec::new();
        let mut dwl: Vec<Matrix> = Vec::new();
        let mut dbl: Vec<Matrix> = Vec::new();

        match L {
            0 => {
                let dzL: Matrix = Matrix::sub(&self.h_o[L].main.apply_new(&sigmoid), &expected);
                let dwL: Matrix = dzL.multiply(&self.input.apply_new(&sigmoid).transpose());
                let dbL: Matrix = dzL.clone();
                dzl.push(dzL);
                dwl.push(dwL);
                dbl.push(dbL);
            }
            1 => {
                let dzL: Matrix = Matrix::sub(&self.h_o[L].main.apply_new(&sigmoid), &expected);
                //Implement Match for if there is only input and output
                //println!("{} {}", dzL, &self.h_o[L-1].main.apply_new(&sigmoid).transpose());
                let dwL: Matrix = dzL.multiply(&self.h_o[L-1].main.apply_new(&sigmoid).transpose()); //
                let dbL: Matrix = dzL.clone();
                dzl.push(dzL);
                dwl.push(dwL);
                dbl.push(dbL);

                //Skipped iterator
                //Last one handling
                let dzl_cur = 
                Matrix::hadamard_product(
                &(self.h_o[1].weights.transpose()).multiply(&dzl[0]), //
                &(self.h_o[0].main).apply_new(&sigmoid_derivative)
                );

                let dwl_cur = dzl_cur.multiply(&self.input.apply_new(&sigmoid).transpose()); //

                let dbl_cur = dzl_cur.clone();

                dzl.insert(0, dzl_cur);
                dwl.insert(0, dwl_cur);
                dbl.insert(0, dbl_cur);
            }
            2.. => {   //the general case   
                let dzL: Matrix = Matrix::sub(&self.h_o[L].main.apply_new(&sigmoid), &expected);
                //Implement Match for if there is only input and output
                let dwL: Matrix = dzL.multiply(&self.h_o[L-1].main.apply_new(&sigmoid).transpose());
                let dbL: Matrix = dzL.clone();
                dzl.push(dzL);
                dwl.push(dwL);
                dbl.push(dbL);

                for l in (1..L).rev() {
                    let dzl_cur = 
                        Matrix::hadamard_product(
                        &(self.h_o[l+1].weights.transpose()).multiply(&dzl[0]),
                        &(self.h_o[l].main).apply_new(&sigmoid_derivative)
                    );

                    let dwl_cur = dzl_cur.multiply(&self.h_o[l-1].main.apply_new(&sigmoid).transpose());

                    let dbl_cur = dzl_cur.clone();

                    dzl.insert(0, dzl_cur);
                    dwl.insert(0, dwl_cur);
                    dbl.insert(0, dbl_cur);
                }

                let dzl_cur = 
                Matrix::hadamard_product(
                &(self.h_o[1].weights.transpose()).multiply(&dzl[0]),
                &(self.h_o[0].main).apply_new(&sigmoid_derivative)
                );

                let dwl_cur = dzl_cur.multiply(&self.input.apply_new(&sigmoid).transpose());

                let dbl_cur = dzl_cur.clone();

                dzl.insert(0, dzl_cur);
                dwl.insert(0, dwl_cur);
                dbl.insert(0, dbl_cur);
            }
            _ => {
                panic!("The number of hidden or output layers you have was not 1, 2, 3, or more. It was: '{}' Could not do the derivatives.", L);
            }
        }
        Derivatives {dzl, dwl, dbl}
    }

    fn update(&mut self, derivative: Derivatives) {
        for l in 0..self.h_o.len() {
            self.h_o[l].bias = Matrix::sub(&self.h_o[l].bias, &derivative.dbl[l]);
            self.h_o[l].weights = Matrix::sub(&self.h_o[l].weights, &derivative.dwl[l]);
        }
    }

    fn print_output(&self) {
        println!("output: {}", self.h_o.last().unwrap().main); //printing the output
        println!("the network thinks it's a: {:?}", max(self.h_o.last().unwrap().main.entries.clone()))
    }
}
    

// The Display for Network, is not my best work :)
impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let w = 45;
        
        write!(f, "sigmoid input: {}", self.input.apply_new(&sigmoid));
        for (id, hidden) in self.h_o.iter().enumerate() {
            write!(f, "\x1B[{}A\x1B[{}C{id}:{}\x1B[u\x1B[{}C", self.input.row, w, hidden.main, w);
        }
        write!(f, "\x1B[{}B\n", self.input.row);
        
        write!(f, "Biases:\n");
        for bias in self.h_o.iter() {
            write!(f, "{}", bias.bias);
        }

        write!(f, "Weights:\n");
        for weight in self.h_o.iter() {
            write!(f, "{}", weight.weights);
        }

        Ok(())
    }
}