import Layer from './layer'
import Mat from '../util/mat'
import Shape from '../util/mat_shape'

import {
    INIT_ZEROS,
    INIT_RANDN,
    LAYER_CONV,
    ACTIVATION_RELU
} from '../util/constant';

export default class ConvLayer extends Layer {
    constructor(name, units, kernel_width, kernel_height, kernel_stride_x, kernel_stride_y, padding) {
        super(name, units);

        this.type = LAYER_CONV;
        this.activation = ACTIVATION_RELU;

        this.kernel_width = kernel_width;
        this.kernel_height = kernel_height;
        this.kernel_depth = undefined;

        this.kernel_stride_x = 1;
        this.kernel_stride_y = 1;
        this.pad_x = 0;
        this.pad_y = 0;

        this.kernel = new Array(units);
        this.biases = new Float32Array(units);
        this.kernel_grad = new Array(this.units);
        this.biases_grad = new Float32Array(this.units);

        this.back_error = undefined;
    }

    set_input_layer(inputLayer) {
        super.set_input_layer(inputLayer);

        const output_width = Math.floor((this.input_shape.width + this.pad_x * 2 - this.kernel_width) / this.kernel_stride_x + 1);
        const output_height = Math.floor((this.input_shape.height + this.pad_y * 2 - this.kernel_height) / this.kernel_stride_y + 1);

        this.output_shape = new Shape(output_width, output_height, this.units);
        this.kernel_depth = this.input_shape.depth;

        const kernel_shape = new Shape(this.kernel_width, this.kernel_height, this.kernel_depth);

        for (let j = 0; j < this.units; j++) {
            this.kernel[j] = new Mat(kernel_shape, INIT_RANDN);
            this.kernel_grad[j] = new Mat(kernel_shape, INIT_ZEROS);
        }
    }

    set_params(weight, bias) {
        for (let i = 0; i < this.units; i++) {
            this.kernel[i].set_value(weight[i]);
        }
        this.biases = new Float32Array(bias);
    }

    forward() {
        this.input = this.pre_layer.output;
        this.output = new Array(this.network.mini_batch_size);

        for (let i = 0; i < this.network.mini_batch_size; i++) {
            this.output[i] = new Mat(this.output_shape, INIT_ZEROS);

            for (let j = 0; j < this.units; j++) {
                for (let out_y = 0; out_y < this.output_shape.height; out_y++) {
                    for (let out_x = 0; out_x < this.output_shape.width; out_x++) {
                        for (let ker_y = 0; ker_y < this.kernel_height; ker_y++) {
                            let input_y = out_y + ker_y;
                            for (let ker_x = 0; ker_x < this.kernel_width; ker_x++) {
                                let input_x = out_x + ker_x;
                                for (let ker_d = 0; ker_d < this.kernel_depth; ker_d++) {
                                    this.output[i].add_value_by_coordinate(out_x, out_y, j, this.kernel[j].get_value_by_coordinate(ker_x, ker_y, ker_d) * this.input[i].get_value_by_coordinate(input_x, input_y, ker_d));
                                }
                            }
                        }
                        this.output[i].add_value_by_coordinate(out_x, out_y, j, this.biases[j]);
                    }
                }
            }
            this.output[i].activate(this.activation);
        }
    }

    backward() {
        let next_layer = this.next_layer;

        this.back_error = new Array(this.network.mini_batch_size);

        for (let i = 0; i < this.network.mini_batch_size; i++) {
            this.back_error[i] = new Mat(this.input_shape, INIT_ZEROS);

        }

        for (let i = 0; i < this.network.mini_batch_size; i++) {
            for (let j = 0; j < this.units; j++) {
                for (let out_y = 0; out_y < this.output_shape.height; out_y++) {
                    for (let out_x = 0; out_x < this.output_shape.width; out_x++) {
                        let error_delta = next_layer.back_error[i].get_value_by_coordinate(out_x, out_y, j) * ( ( this.output[i].get_value_by_coordinate(out_x, out_y, j) > 0 ) ? 1 : 0 );
                        let error_delta_with_learning_rate = -this.network.batch_learning_rate * error_delta;
                        this.biases_grad[j] += error_delta_with_learning_rate;
                        for (let ker_d = 0; ker_d < this.kernel_depth; ++ker_d) {
                            for (let ker_y = 0; ker_y < this.kernel_height; ++ker_y) {
                                let input_y = ker_y + out_y;
                                for (let ker_x = 0; ker_x < this.kernel_width; ++ker_x) {
                                    let input_x = ker_x + out_x;
                                    this.kernel_grad[j].add_value_by_coordinate(ker_x, ker_y, ker_d, error_delta_with_learning_rate * this.input[i].get_value_by_coordinate(input_x, input_y, ker_d));
                                    this.back_error[i].add_value_by_coordinate(input_x, input_y, ker_d, error_delta * this.kernel[j].get_value_by_coordinate(ker_x, ker_y, ker_d));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    batch_update() {
        const l2_regularization = 1.0 - this.network.batch_learning_rate * this.network.l2;

        for (let j = 0; j < this.units; j++) {
            this.kernel[j].operation_scale_and_add_mat(l2_regularization, this.kernel_grad[j]);
            this.biases[j] += this.biases_grad[j];

            this.kernel_grad[j].operation_scale_mat(this.network.momentum);
            this.biases_grad[j] *= this.network.momentum;
        }
    }
}