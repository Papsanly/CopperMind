use std::iter::zip;

#[derive(Debug)]
struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

#[derive(Debug)]
pub struct Perceptron<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    network: Vec<Vec<Neuron>>,
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> Perceptron<INPUT_SIZE, OUTPUT_SIZE> {
    pub fn new(shape: &[usize]) -> Self {
        let mut network: Vec<Vec<Neuron>> = Vec::with_capacity(shape.len());
        let mut weight_counts = vec![INPUT_SIZE];
        weight_counts.extend_from_slice(shape);
        let mut neuron_counts = shape.to_vec();
        neuron_counts.push(OUTPUT_SIZE);

        for (weight_count, neuron_count) in zip(weight_counts, neuron_counts) {
            network.push(
                (0..neuron_count)
                    .map(|_| Neuron {
                        weights: vec![0.; weight_count],
                        bias: 0.,
                    })
                    .collect(),
            );
        }

        Perceptron { network }
    }

    pub fn fit(&self, data: &[[f32; INPUT_SIZE]], labels: &[u32], epochs: usize) {
        todo!()
    }

    pub fn predict(&self, data: &[[f32; INPUT_SIZE]]) -> &[u32] {
        todo!()
    }
}
