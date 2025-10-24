mod activation_relu;
mod layer_dense;

use activation_relu::ActivationReLu;
use layer_dense::LayerDense;

fn main() {
    // Inputs: 3 batches, each with 4 inputs
    let x = [
        vec![1.0, 2.0, 3.0, 2.5],
        vec![2.0, 5.0, -1.0, 2.0],
        vec![-1.5, 2.7, 3.3, -0.8],
    ];

    let mut layer_one = LayerDense::build_layer(x[0].len(), 5);
    let mut layer_two = LayerDense::build_layer(layer_one.weights.len(), 2);
    let mut activation_one = ActivationReLu::build_layer();

    layer_one.forward(&x);
    layer_two.forward(&layer_one.output);
    activation_one.forward(&layer_two.output);

    println!("{:?}", activation_one.output);
}
