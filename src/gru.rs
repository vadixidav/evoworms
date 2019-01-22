use af;
use af::{Array, Dim4, MatProp};

/// Creates a random array with the dimensions and elements in the range [-1, 1].
fn rand_sig(dims: Dim4) -> Array<f32> {
    af::randu::<f32>(dims) * 2 - 1
}

/// Mutates all array elements to random values with probability `lambda`.
fn mutate_rand(array: &mut Array<f32>, lambda: f32) {
    let choices: Array<bool> =
        unsafe { ::std::mem::transmute(af::le(&af::randu::<f32>(array.dims()), &lambda, false)) };
    *array = af::select(&rand_sig(array.dims()), &choices, array);
}

struct GRUTanh {
    hidden_matrix: Array<f32>,
    input_matrix: Array<f32>,
    biases: Array<f32>,
}

impl GRUTanh {
    fn new_random(inputs: u64, outputs: u64) -> GRUTanh {
        GRUTanh {
            hidden_matrix: rand_sig(Dim4::new(&[outputs, outputs, 1, 1])),
            input_matrix: rand_sig(Dim4::new(&[outputs, inputs, 1, 1])),
            biases: rand_sig(Dim4::new(&[outputs, 1, 1, 1])),
        }
    }

    fn apply(&self, hiddens: &Array<f32>, inputs: &Array<f32>) -> Array<f32> {
        af::tanh(
            &(&af::matmul(&self.hidden_matrix, hiddens, MatProp::NONE, MatProp::NONE)
                + &af::matmul(&self.input_matrix, inputs, MatProp::NONE, MatProp::NONE)
                + &self.biases),
        )
    }

    /// Mutate each matrix element with a probability lambda
    fn mutate(&mut self, lambda: f32) {
        mutate_rand(&mut self.hidden_matrix, lambda);
        mutate_rand(&mut self.input_matrix, lambda);
        mutate_rand(&mut self.biases, lambda);
    }
}

struct GRUGate {
    hidden_matrix: Array<f32>,
    input_matrix: Array<f32>,
    biases: Array<f32>,
}

impl GRUGate {
    fn new_random(inputs: u64, outputs: u64) -> GRUGate {
        GRUGate {
            hidden_matrix: rand_sig(Dim4::new(&[outputs, outputs, 1, 1])),
            input_matrix: rand_sig(Dim4::new(&[outputs, inputs, 1, 1])),
            biases: rand_sig(Dim4::new(&[outputs, 1, 1, 1])),
        }
    }

    fn apply(&self, hiddens: &Array<f32>, inputs: &Array<f32>) -> Array<f32> {
        af::sigmoid(
            &(&af::matmul(&self.hidden_matrix, hiddens, MatProp::NONE, MatProp::NONE)
                + &af::matmul(&self.input_matrix, inputs, MatProp::NONE, MatProp::NONE)
                + &self.biases),
        )
    }

    /// Mutate each matrix element with a probability lambda
    fn mutate(&mut self, lambda: f32) {
        mutate_rand(&mut self.hidden_matrix, lambda);
        mutate_rand(&mut self.input_matrix, lambda);
        mutate_rand(&mut self.biases, lambda);
    }
}

pub struct GRU {
    reset_gate: GRUGate,
    update_gate: GRUGate,
    output_layer: GRUTanh,
    hiddens: Array<f32>,
}

impl GRU {
    pub fn new_rand(inputs: u64, outputs: u64) -> GRU {
        GRU {
            reset_gate: GRUGate::new_random(inputs, outputs),
            update_gate: GRUGate::new_random(inputs, outputs),
            output_layer: GRUTanh::new_random(inputs, outputs),
            hiddens: af::randu::<f32>(Dim4::new(&[outputs, 1, 1, 1])),
        }
    }

    pub fn apply(&mut self, inputs: &Array<f32>) -> Array<f32> {
        // Compute reset coefficients.
        let r = self.reset_gate.apply(&self.hiddens, inputs);
        // Compute update coefficients.
        let z = self.update_gate.apply(&self.hiddens, inputs);

        let outputs: Array<f32> = (-z.clone() + 1)
            * self.output_layer.apply(&(r * &self.hiddens), inputs)
            + z * &self.hiddens;

        self.hiddens = outputs.clone();
        outputs
    }

    pub fn mutate(&mut self, lambda: f32) {
        self.reset_gate.mutate(lambda);
        self.update_gate.mutate(lambda);
        self.output_layer.mutate(lambda);
    }
}
