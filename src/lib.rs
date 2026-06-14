use bon::bon;
use tensor::{Tensor, TensorRef};

#[derive(Debug)]
pub struct Perceptron {
    weights: Tensor<f32>,
    biases: Tensor<f32>,
}

#[bon]
impl Perceptron {
    pub fn new(shape: &[usize]) -> Self {
        todo!()
    }

    #[builder]
    pub fn train(
        &mut self,
        #[builder(start_fn)] data: TensorRef<'_, f32>,
        #[builder(start_fn)] labels: TensorRef<'_, f32>,
        epochs: usize,
        learn_rate: f32,
        batch_size: usize,
        #[builder(with = |v: impl Fn(Perceptron) + 'static| Box::new(v))] on_epoch: Option<
            Box<dyn Fn(Perceptron)>,
        >,
    ) {
        todo!()
    }

    pub fn predict(&self, data: TensorRef<f32>) -> Tensor<f32> {
        todo!()
    }
}

#[cfg(test)]
mod tests {}
