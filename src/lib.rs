pub struct Perceptron {}

impl Perceptron {
    pub fn new(shape: &[usize]) -> Self {
        Perceptron {}
    }

    pub fn fit<T, const S: usize>(&self, data: &[([T; S], T)], epochs: usize) {
        todo!()
    }

    pub fn predict<T, const S: usize>(&self, data: &[([T; S], T)]) -> &[T] {
        todo!()
    }
}
