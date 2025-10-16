use rand::prelude::*;

pub struct LayerDense {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
    pub output: Vec<Vec<f32>>,
}

impl LayerDense {
    pub fn build_layer(n_inputs: usize, n_neurons: usize) -> LayerDense {
        let mut rng = rand::rng();
        let mut weights = vec![vec![0.0; n_inputs]; n_neurons];

        for neuron_weights in weights.iter_mut() {
            for weight in neuron_weights.iter_mut() {
                *weight = 0.1 * rng.random::<f32>();
            }
        }

        LayerDense {
            weights,
            biases: vec![0.0; n_neurons],
            output: Vec::new(),
        }
    }

    pub fn forward(&mut self, inputs: &[Vec<f32>]) {
        // iterate through batches
        for input in inputs.iter() {
            let mut batch_output = Vec::with_capacity(self.weights.len());

            // iterate through neurons
            for (neuron_weights, neuron_bias) in self.weights.iter().zip(&self.biases) {
                // iterate through inputs
                let neuron_output: f32 = input
                    .iter()
                    .zip(neuron_weights)
                    .map(|(neuron_input, weight)| neuron_input * weight)
                    .sum();

                batch_output.push(neuron_output + neuron_bias);
            }

            self.output.push(batch_output);
        }
    }
}
