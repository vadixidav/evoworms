use af;
use af::{Array, Dim4, MatProp};

struct GRUTanh {
    hidden_matrix: Array,
    input_matrix: Array,
    biases: Array,
}

impl GRUTanh {
    fn new_random(inputs: u64, outputs: u64) -> GRUTanh {
        GRUTanh {
            hidden_matrix: af::randu::<f64>(Dim4::new(&[outputs, outputs, 1, 1])),
            input_matrix: af::randu::<f64>(Dim4::new(&[outputs, inputs, 1, 1])),
            biases: af::randu::<f64>(Dim4::new(&[outputs, 1, 1, 1])),
        }
    }

    fn apply(&self, hiddens: &Array, inputs: &Array) -> Array {
        af::tanh(
            &(&af::matmul(&self.hidden_matrix, hiddens, MatProp::NONE, MatProp::NONE)
                + &af::matmul(&self.input_matrix, inputs, MatProp::NONE, MatProp::NONE)
                + &self.biases),
        )
    }
}

struct GRUGate {
    hidden_matrix: Array,
    input_matrix: Array,
    biases: Array,
}

impl GRUGate {
    fn new_random(inputs: u64, outputs: u64) -> GRUGate {
        GRUGate {
            hidden_matrix: af::randu::<f64>(Dim4::new(&[outputs, outputs, 1, 1])),
            input_matrix: af::randu::<f64>(Dim4::new(&[outputs, inputs, 1, 1])),
            biases: af::randu::<f64>(Dim4::new(&[outputs, 1, 1, 1])),
        }
    }

    fn apply(&self, hiddens: &Array, inputs: &Array) -> Array {
        af::sigmoid(
            &(&af::matmul(&self.hidden_matrix, hiddens, MatProp::NONE, MatProp::NONE)
                + &af::matmul(&self.input_matrix, inputs, MatProp::NONE, MatProp::NONE)
                + &self.biases),
        )
    }
}

pub struct GRU {
    reset_gate: GRUGate,
    update_gate: GRUGate,
    output_layer: GRUTanh,
    hiddens: Array,
}

impl GRU {
    pub fn new_rand(inputs: u64, outputs: u64) -> GRU {
        GRU {
            reset_gate: GRUGate::new_random(inputs, outputs),
            update_gate: GRUGate::new_random(inputs, outputs),
            output_layer: GRUTanh::new_random(inputs, outputs),
            hiddens: af::randu::<f64>(Dim4::new(&[outputs, 1, 1, 1])),
        }
    }

    pub fn apply(&mut self, inputs: &Array) -> Array {
        // Compute reset coefficients.
        let r = self.reset_gate.apply(&self.hiddens, inputs);
        // Compute update coefficients.
        let z = self.update_gate.apply(&self.hiddens, inputs);

        let outputs: Array =
            (&z + -1) * self.output_layer.apply(&(r * &self.hiddens), inputs) + z * &self.hiddens;

        self.hiddens = outputs.clone();
        outputs
    }
}
