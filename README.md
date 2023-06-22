# CopperMind

CopperMind is a Rust-based implementation of a perceptron neural network, designed with a primary focus on deepening
understanding of both neural networks and the Rust programming language. This project aims to provide a comprehensive
learning experience by building a neural network from the ground up, without relying on external libraries or
frameworks, while simultaneously exploring the intricacies of Rust's syntax and features.

## Features

- Implementation of the perceptron neural network algorithm.
- todo!()

## Installation

To use CopperMind in your project, follow these steps:

1. Install Rust programming language and Cargo package manager. Visit [rust-lang.org](https://www.rust-lang.org) for
   installation instructions.

2. Add CopperMind as a dependency in your Cargo.toml file:

```toml
[dependencies]
coppermind = "0.1.0"
```

## Usage

Here's a basic example demonstrating how to use CopperMind:

```rust
use copper_mind::Perceptron;

fn main() {
   let train_data = read_mnist_data("train").unwrap();
   let train_labels = read_mnist_labels("train").unwrap();
   let test_data = read_mnist_data("t10k").unwrap();
   let test_labels = read_mnist_labels("t10k").unwrap();
   
   let perceptron = Perceptron::<IMG_BUF_SIZE, 10>::new(&HIDDEN_LAYERS);
   perceptron.fit(&train_data, &train_labels, EPOCHS);
   let predictions = perceptron.predict(&test_data);
   
   println!("MSE: {}", mse(&predictions, &test_labels));
}
```

Full example can be found at `examples/mnist.rs`

## Contributing

Contributions are welcome! If you'd like to contribute to CopperMind, please fork the repository and submit a pull
request. For major changes, please open an issue to discuss your ideas beforehand.

## Acknowledgements

I would like to thank the open-source community for their invaluable contributions and inspiration.

### *Happy coding with CopperMind!*