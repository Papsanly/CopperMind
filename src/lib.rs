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

    pub fn fit(
        &mut self,
        data: &[[f32; INPUT_SIZE]],
        labels: &[usize],
        epochs: usize,
        learn_rate: f32,
        mini_batch_count: usize,
    ) {
        let mini_batches = self.mini_batch(data, labels, mini_batch_count);
        for _ in 0..epochs {
            for mini_batch in &mini_batches {
                self.update_mini_batch(mini_batch, learn_rate);
            }
        }
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

    fn update_mini_batch(
        &mut self,
        mini_batch: &(&[[f32; INPUT_SIZE]], &[usize]),
        learn_rate: f32,
    ) {
        for (layer_deltas, layer) in zip(self.backprop(mini_batch), &mut self.network) {
            for ((weights_delta, bias_delta), neuron) in zip(layer_deltas, layer) {
                for (weight_delta, weight) in zip(weights_delta, &mut neuron.weights) {
                    *weight -= learn_rate * weight_delta;
                }
                neuron.bias -= learn_rate * bias_delta;
            }
        }
    }

    fn backprop(&self, mini_batch: &(&[[f32; INPUT_SIZE]], &[usize])) -> Vec<Vec<(Vec<f32>, f32)>> {
        todo!()
    }

    fn mini_batch<'a>(
        &self,
        data: &'a [[f32; INPUT_SIZE]],
        labels: &'a [usize],
        count: usize,
    ) -> Vec<(&'a [[f32; INPUT_SIZE]], &'a [usize])> {
        let mut res = Vec::new();
        let len = INPUT_SIZE / count;
        for i in 0..count {
            res.push((
                &data[i * len..(i + 1) * len],
                &labels[i * len..(i + 1) * len],
            ));
        }
        res
    }

    fn activation_fn(val: f32) -> f32 {
        f32::max(0., val)
    }
}

#[cfg(test)]
mod tests {
    use super::Perceptron;
    use crate::Neuron;

    fn get_perceptron() -> Perceptron<4, 1> {
        Perceptron::<4, 1>::new(&[2])
    }

    fn assert_mini_batches(mini_batches: Vec<(&[[f32; 4]], &[usize])>) {
        assert_eq!(
            mini_batches,
            vec![
                ([[0.; 4], [1.; 4]].as_slice(), [1, 2].as_slice()),
                ([[2.; 4], [3.; 4]].as_slice(), [3, 4].as_slice())
            ]
        )
    }

    #[test]
    fn test_network_structure() {
        let perceptron = get_perceptron();
        let network_structure = perceptron
            .network
            .iter()
            .map(|layer| {
                layer
                    .iter()
                    .map(|neuron| neuron.weights.len())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(network_structure, vec![vec![4; 2], vec![2; 1]]);
    }

    #[test]
    fn test_evaluate() {
        let perceptron: Perceptron<2, 1> = Perceptron {
            network: vec![
                vec![Neuron {
                    weights: vec![1., 2.],
                    bias: 1.,
                }],
                vec![Neuron {
                    weights: vec![1.],
                    bias: -1.,
                }],
            ],
        };
        let data = [1., 2.];
        let res = perceptron.evaluate(&data);
        assert_eq!(res, vec![5.]);
    }

    #[test]
    fn test_mini_batching() {
        let perceptron = get_perceptron();
        let mini_batches =
            perceptron.mini_batch(&[[0.; 4], [1.; 4], [2.; 4], [3.; 4]], &[1, 2, 3, 4], 2);
        assert_mini_batches(mini_batches);
    }

    #[test]
    fn test_mini_batching_uneven() {
        let perceptron = get_perceptron();
        let mini_batches = perceptron.mini_batch(
            &[[0.; 4], [1.; 4], [2.; 4], [3.; 4], [4.; 4]],
            &[1, 2, 3, 4, 5],
            2,
        );
        assert_mini_batches(mini_batches);
    }
}
