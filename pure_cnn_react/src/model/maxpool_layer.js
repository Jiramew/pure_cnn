import Layer from './layer';
import Mat from '../util/mat';
import Shape from '../util/mat_shape';
import {INIT_CHARS, INIT_ZEROS, LAYER_MAXPOOL} from '../util/constant';


export default class MaxPoolLayer extends Layer {
    constructor(name, pool_width, pool_height, pool_stride_x, pool_stride_y) {
        super(name, 0);
        this.type = LAYER_MAXPOOL;
        this.pool_width = pool_width;
        this.pool_height = pool_height;
        this.pool_stride_x = pool_stride_x;
        this.pool_stride_y = pool_stride_y;
        this.max_info = undefined
    }

    set_input_layer(inputLayer) {
        super.set_input_layer(inputLayer);

        this.output_width = Math.floor((this.input_shape.width - this.pool_width) / this.pool_stride_x + 1);
        this.output_height = Math.floor((this.input_shape.height - this.pool_height) / this.pool_stride_y + 1);
        this.output_depth = this.input_shape.depth;

        this.output_shape = new Shape(this.output_width, this.output_height, this.output_depth);
    }

    forward() {
        this.input = this.pre_layer.output;
        this.output = new Array(this.network.mini_batch_size);
        this.max_info = new Array(this.network.mini_batch_size);

        for (let i = 0; i < this.network.mini_batch_size; ++i) {
            this.max_info[i] = new Mat(this.output_shape, INIT_CHARS);
            this.output[i] = new Mat(this.output_shape, INIT_ZEROS);

            for (let out_d = 0; out_d < this.output_depth; out_d++) {
                for (let out_y = 0; out_y < this.output_height; out_y++) {
                    for (let out_x = 0; out_x < this.output_width; out_x++) {
                        let max_value = 0;
                        let max_index = [0, 0, 0];
                        for (let input_y = 0; input_y < this.pool_height; input_y++) {
                            let stride_y = out_y * this.pool_stride_y + input_y;
                            for (let input_x = 0; input_x < this.pool_width; input_x++) {
                                let stride_x = out_x * this.pool_stride_x + input_x;
                                let compare_input = this.input[i].get_value_by_coordinate(stride_x, stride_y, out_d);
                                if (compare_input >= max_value) {
                                    max_value = compare_input;
                                    max_index = [stride_x, stride_y, out_d];
                                }
                            }
                        }
                        this.max_info[i].set_value_by_coordinate(out_x, out_y, out_d, max_index.join("-"));
                        this.output[i].set_value_by_coordinate(out_x, out_y, out_d, max_value);
                    }
                }
            }
        }
    }

    backward() {
        let next_layer = this.next_layer;
        this.back_error = new Array(this.network.mini_batch_size);

        for (let i = 0; i < this.network.mini_batch_size; ++i) {
            this.back_error[i] = new Mat(this.input_shape, INIT_ZEROS);

            let max_infos = this.max_info[i].get_value();
            for (let mi = 0; mi < max_infos.length; mi++) {
                let info = max_infos[mi].split("-");
                let input_x = parseInt(info[0], 10);
                let input_y = parseInt(info[1], 10);
                let input_d = parseInt(info[2], 10);
                this.back_error[i].set_value_by_coordinate(
                    input_x,
                    input_y,
                    input_d,
                    next_layer.back_error[i].get_value_by_coordinate(
                        Math.floor(input_x / this.pool_width),
                        Math.floor(input_y / this.pool_height),
                        input_d));
            }
        }
    }
}