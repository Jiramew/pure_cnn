import Layer from './layer';
import Mat from '../util/mat';
import Shape from '../util/mat_shape';
import {
    INIT_ZEROS,
    INIT_RANDN,
    LAYER_FULLY_CONNECTED,
    ACTIVATION_TANH
} from '../util/constant';

export default class FCLayer extends Layer {
    constructor(name, units, activation) {
        super(name, units);
        this.type = LAYER_FULLY_CONNECTED;
        this.output_shape = new Shape(1, 1, units);
        this.activation = activation;

        this.weight = new Array(this.units);
        this.biases = new Float32Array(units);
        this.weight_grad = new Array(this.units);
        this.biases_grad = new Float32Array(this.units);
    }

    set_input_layer(inputLayer) {
        super.set_input_layer(inputLayer);
        for (let j = 0; j < this.units; j++) {
            this.weight[j] = new Mat(this.input_shape, INIT_RANDN);
            this.weight_grad[j] = new Mat(this.input_shape, INIT_ZEROS);

        }
    }

    set_output_layer(outputLayer) {
        super.set_output_layer(outputLayer);
    }

    set_params(weight, bias) {
        for (let j = 0; j < this.units; j++) {
            this.weight[j].set_value(weight[j]);
        }
        this.biases = new Float32Array(bias);
    }

    forward() {
        this.input = this.pre_layer.output;
        this.output = new Array(this.network.mini_batch_size);

        const size = this.input_shape.get_size();

        for (let i = 0; i < this.network.mini_batch_size; i++) {
            this.output[i] = new Mat(this.output_shape, INIT_ZEROS);

            for (let j = 0; j < this.units; j++) {
                for (let k = 0; k < size; k++) {
                    this.output[i].values[j] += this.weight[j].values[k] * this.input[i].values[k];
                }
                this.output[i].values[j] += this.biases[j];
            }
            this.output[i].activate(this.activation);
        }
    }

    backward() {
        this.back_error = new Array(this.network.mini_batch_size);

        for (let i = 0; i < this.network.mini_batch_size; ++i) {
            this.back_error[i] = new Mat(this.input_shape, INIT_ZEROS);

            for (let j = 0; j < this.units; ++j) {
                let error;
                if (this.is_last_layer()) {
                    error = this.output[i].values[j] - this.network.label_list_one_hot[i][j];
                }
                else {
                    error = this.next_layer.back_error[i].get_value_by_coordinate(0, 0, j);
                }

                if (this.activation === ACTIVATION_TANH) {
                    error *= (1 - Math.pow(this.output[i].values[j], 2));
                }

                this.network.training_error += Math.abs(error);

                let grad = -this.network.batch_learning_rate * error;
                this.weight_grad[j].operation_add_scaled_mat(grad, this.input[i]);
                this.biases_grad[j] += grad;

                this.back_error[i].operation_add_scaled_mat(error, this.weight[j]);
            }
        }
    }

    batch_update() {
        const l2_regularization = 1.0 - this.network.batch_learning_rate * this.network.l2;

        for (let j = 0; j < this.units; j++) {
            this.weight[j].operation_scale_and_add_mat(l2_regularization, this.weight_grad[j]);
            this.biases[j] += this.biases_grad[j];

            this.weight_grad[j].operation_scale_mat(this.network.momentum);
            this.biases_grad[j] *= this.network.momentum;
        }
    }
}