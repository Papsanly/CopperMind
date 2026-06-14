// To use this example download the MNIST dataset:
// https://www.kaggle.com/datasets/hojjatk/mnist-dataset

use copper_mind::Perceptron;
use std::{fs, io};
use tensor::Tensor;

const MNIST_PATH: &str = "data";
const IMG_BUF_SIZE: usize = 28 * 28;
const SHAPE: [usize; 1] = [50];
const EPOCHS: usize = 20;
const LEARNING_RATE: f32 = 1.;
const BATCH_SIZE: usize = 32;

fn main() {
    let train_data = read_mnist_data("train").unwrap();
    let train_labels = read_mnist_labels("train").unwrap();
    let val_data = read_mnist_data("t10k").unwrap();
    let val_labels = read_mnist_labels("t10k").unwrap();

    let mut perceptron = Perceptron::new(&SHAPE);

    perceptron
        .train(train_data.as_ref(), train_labels.as_ref())
        .epochs(EPOCHS)
        .learn_rate(LEARNING_RATE)
        .batch_size(BATCH_SIZE)
        .on_epoch(|perceptron| todo!())
        .call();
}

fn read_mnist_labels(dataset: &str) -> Result<Tensor<f32>, io::Error> {
    let bytes = fs::read(format!("{}/{}-labels.idx1-ubyte", MNIST_PATH, dataset))?;
    Ok(bytes.into_iter().skip(8).map(|b| b as f32).collect())
}

fn read_mnist_data(dataset: &str) -> Result<Tensor<f32>, io::Error> {
    let bytes = fs::read(format!("{}/{}-images.idx3-ubyte", MNIST_PATH, dataset))?;
    Ok(
        Tensor::from_iter(bytes.into_iter().skip(16).map(|b| b as f32 / 255.))
            .reshape_infer([None, Some(IMG_BUF_SIZE)])
            .expect("invalid mnist dataset"),
    )
}
