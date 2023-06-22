// To use this example download the MNIST dataset:
// http://yann.lecun.com/exdb/mnist/

use copper_mind::Perceptron;
use std::fs::File;
use std::io::{self, BufReader, Read};

const MNIST_PATH: &str = "data";
const IMG_BUF_SIZE: usize = 28 * 28;
const HIDDEN_LAYERS: [usize; 1] = [50];
const EPOCHS: usize = 20;

fn main() {
    let train_data = read_mnist_data("train").unwrap();
    let train_labels = read_mnist_labels("train").unwrap();
    let test_data = read_mnist_data("t10k").unwrap();
    let test_labels = read_mnist_labels("t10k").unwrap();

    let perceptron = Perceptron::<IMG_BUF_SIZE, 10>::new(&HIDDEN_LAYERS);

    perceptron.fit(&train_data, &train_labels, EPOCHS);

    let predictions = perceptron.predict(&test_data);
}

fn read_mnist_labels(dataset: &str) -> Result<Vec<usize>, io::Error> {
    let f = File::open(format!("{}/{}-labels.idx1-ubyte", MNIST_PATH, dataset))?;
    let mut reader = BufReader::new(f).bytes().skip(8);

    let mut res = Vec::new();
    loop {
        let label = match reader.next() {
            Some(label) => label?,
            None => break,
        };
        res.push(label as usize)
    }

    Ok(res)
}

fn read_mnist_data(dataset: &str) -> Result<Vec<[f32; IMG_BUF_SIZE]>, io::Error> {
    let f = File::open(format!("{}/{}-images.idx3-ubyte", MNIST_PATH, dataset))?;
    let mut reader = BufReader::new(f).bytes().skip(16);

    let mut res = Vec::new();
    let mut buf = [0.; IMG_BUF_SIZE];
    'outer: loop {
        for val in buf.iter_mut() {
            *val = match reader.next() {
                Some(byte) => byte? as f32 / 255.,
                None => break 'outer,
            };
        }
        res.push(buf);
    }

    Ok(res)
}
