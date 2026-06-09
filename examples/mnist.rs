// To use this example download the MNIST dataset:
// http://yann.lecun.com/exdb/mnist/

use copper_mind::Perceptron;
use std::{array, fs, io};

const MNIST_PATH: &str = "data";
const IMG_BUF_SIZE: usize = 28 * 28;
const HIDDEN_LAYERS: [usize; 1] = [50];
const EPOCHS: usize = 20;
const LEARNING_RATE: f32 = 1.;
const MINI_BATCHES: usize = 10;

fn main() {
    let train_data = read_mnist_data("train").unwrap();
    let train_labels = read_mnist_labels("train").unwrap();
    let test_data = read_mnist_data("t10k").unwrap();
    let test_labels = read_mnist_labels("t10k").unwrap();

    let mut perceptron = Perceptron::<IMG_BUF_SIZE, 10>::new(&HIDDEN_LAYERS);

    perceptron.train(
        &train_data,
        &train_labels,
        EPOCHS,
        LEARNING_RATE,
        MINI_BATCHES,
    );

    let predictions = perceptron.predict(&test_data);
}

fn read_mnist_labels(dataset: &str) -> Result<Vec<usize>, io::Error> {
    let bytes = fs::read(format!("{}/{}-labels.idx1-ubyte", MNIST_PATH, dataset))?;
    Ok(bytes[8..].iter().map(|&b| b as usize).collect())
}

fn read_mnist_data(dataset: &str) -> Result<Vec<[f32; IMG_BUF_SIZE]>, io::Error> {
    let bytes = fs::read(format!("{}/{}-images.idx3-ubyte", MNIST_PATH, dataset))?;
    Ok(bytes[16..]
        .chunks_exact(IMG_BUF_SIZE)
        .map(|chunk| array::from_fn(|i| chunk[i] as f32 / 255.))
        .collect())
}
