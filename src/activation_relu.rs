pub struct ActivationReLu {
    pub output: Vec<Vec<f32>>,
}

impl ActivationReLu {
    pub fn build_layer() -> ActivationReLu {
        ActivationReLu { output: Vec::new() }
    }

    pub fn forward(&mut self, inputs: &[Vec<f32>]) {
        // iterate through batches
        for batch_input in inputs.iter() {
            // iterate trough inputs
            let batch_output: Vec<f32> = batch_input.iter().map(|&x| x.max(0.0)).collect();
            self.output.push(batch_output);
        }
    }
}
