// To use this example download the MNIST dataset:
// http://yann.lecun.com/exdb/mnist/

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

const MNIST_PATH: &str = "data";
const IMG_WIDTH: usize = 28;
const IMG_HEIGHT: usize = 28;

fn main() {
    let train_data = read_mnist("train");
    let test_data = read_mnist("t10k");

    dbg!(&train_data[0..1]);
    dbg!(&test_data[0..1]);
}

fn read_mnist(dataset: &str) -> Vec<([u8; IMG_WIDTH * IMG_HEIGHT], u8)> {
    let f = File::open(format!("{MNIST_PATH}/{dataset}-images.idx3-ubyte")).unwrap();
    let mut img_reader = BufReader::new(f);
    img_reader.seek(SeekFrom::Start(16)).unwrap();

    let f = File::open(format!("{MNIST_PATH}/{dataset}-labels.idx1-ubyte")).unwrap();
    let mut labels_reader = BufReader::new(f);
    labels_reader.seek(SeekFrom::Start(8)).unwrap();

    let mut res = Vec::new();
    let mut img_buf = [0; IMG_WIDTH * IMG_HEIGHT];
    let mut labels_buf = [0; 1];
    loop {
        if img_reader.read(&mut img_buf).unwrap() == 0 {
            break;
        }

        if labels_reader.read(&mut labels_buf).unwrap() == 0 {
            break;
        }

        res.push((img_buf, labels_buf[0]));
    }

    res
}
