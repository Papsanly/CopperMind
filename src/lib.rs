use rand::Rng;
use std::iter::zip;

#[derive(Debug)]
pub struct Perceptron<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> {
    network: Vec<Vec<Neuron>>,
}

#[derive(Debug)]
struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

impl<const INPUT_SIZE: usize, const OUTPUT_SIZE: usize> Perceptron<INPUT_SIZE, OUTPUT_SIZE> {
    pub fn new(hidden_layer: &[usize]) -> Self {
        let mut weight_counts = vec![INPUT_SIZE];
        weight_counts.extend(hidden_layer);

        let mut neuron_counts = hidden_layer.to_vec();
        neuron_counts.push(OUTPUT_SIZE);

        let mut rng = rand::thread_rng();
        let network = zip(weight_counts, neuron_counts)
            .map(|(weight_count, neuron_count)| {
                (0..neuron_count)
                    .map(|_| Neuron {
                        weights: (0..weight_count)
                            .map(|_| rng.gen_range(-1.0..1.0))
                            .collect(),
                        bias: rng.gen_range(-1.0..1.0),
                    })
                    .collect()
            })
            .collect();

        Perceptron { network }
    }

    pub fn fit(&self, data: &[[f32; INPUT_SIZE]], labels: &[usize], epochs: usize) {
        todo!()
    }

    pub fn predict(&self, data: &[[f32; INPUT_SIZE]]) -> Vec<usize> {
        data.iter()
            .map(|input| {
                self.evaluate(input)
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .map(|(index, _)| index)
                    .unwrap()
            })
            .collect()
    }

    pub fn evaluate(&self, input: &[f32; INPUT_SIZE]) -> Vec<f32> {
        let mut activations = input.to_vec();
        for layer in &self.network {
            activations = layer
                .iter()
                .map(|neuron| {
                    Self::activation_fn(
                        zip(&neuron.weights, &activations)
                            .map(|(w, a)| w * a)
                            .sum::<f32>()
                            + neuron.bias,
                    )
                })
                .collect();
        }
        activations
    }

    fn activation_fn(val: f32) -> f32 {
        f32::max(0., val)
    }
}
