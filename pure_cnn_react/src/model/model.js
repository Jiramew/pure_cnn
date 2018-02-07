import {
    LAYER_CONV,
    LAYER_MAXPOOL,
    LAYER_FULLY_CONNECTED,
    LAYER_INPUT,
} from '../util/constant';

import InputLayer from './input_layer';
import ConvLayer from './conv_layer';
import MaxPoolLayer from './maxpool_layer';
import FCLayer from './fc_layer';

export default class PureCnn {
    constructor(mini_batch_size) {
        this.layers = [];
        this.next_layer_index = 0;

        this.learning_rate = 0.01;
        this.momentum = 0.9;

        this.l2 = 0.0;

        this.mini_batch_size = mini_batch_size;
        this.training_error = 0;

        this.forward_time = 0;
        this.backward_time = 0;
    }

    set_momentum(momentum) {
        this.momentum = momentum;
    }

    set_learning_rate(rate) {
        this.learning_rate = rate;
    }

    set_l2(l2) {
        this.l2 = l2;
    }

    add_layer(layer_info) {
        console.log("Creating layer " + layer_info.name);

        let new_layer;

        switch (layer_info.type) {
            case LAYER_INPUT: {
                new_layer = new InputLayer(layer_info.name, layer_info.width, layer_info.height, layer_info.depth);
                break;
            }

            case LAYER_CONV: {
                new_layer = new ConvLayer(layer_info.name, layer_info.units, layer_info.kernel_width, layer_info.kernel_height,
                    layer_info.pool_stride_x, layer_info.pool_stride_y, layer_info.padding);
                break;
            }

            case LAYER_MAXPOOL: {
                new_layer = new MaxPoolLayer(layer_info.name, layer_info.pool_width, layer_info.pool_height,
                    layer_info.pool_stride_x, layer_info.pool_stride_y);
                break;
            }

            case LAYER_FULLY_CONNECTED: {
                new_layer = new FCLayer(layer_info.name, layer_info.units, layer_info.activation);
                break;
            }

            default: {
                break
            }
        }

        new_layer.network = this;

        if (this.next_layer_index === 0) {
            if (new_layer.type !== LAYER_INPUT) {
                throw new Error("First layer should be input layer.");
            }
        }
        else {
            let pre_layer = this.layers[this.next_layer_index - 1];
            pre_layer.set_output_layer(new_layer);
            new_layer.set_input_layer(pre_layer);
        }

        new_layer.layerIndex = this.next_layer_index;
        this.layers[this.next_layer_index] = new_layer;
        this.next_layer_index++;
    }

    train(imageDataArray, imageLabelsArray) {
        this.batch_learning_rate = this.learning_rate / this.mini_batch_size;

        this.training_error = 0;

        this._one_hot(imageLabelsArray);

        let t0 = Date.now();
        this._forward(imageDataArray);
        let t1 = Date.now();
        this._backward();
        this._mini_batch();

        this.training_error /= this.mini_batch_size;
        this.forward_time = (t1 - t0) / this.mini_batch_size;
        this.backward_time = (Date.now() - t1) / this.mini_batch_size;
    }

    _one_hot(image_label_list) {
        this.label_list_one_hot = new Array(this.mini_batch_size);
        for (let i = 0; i < this.mini_batch_size; ++i) {
            this.label_list_one_hot[i] = new Array(this.layers[this.layers.length - 1].units);
            for (let j = 0; j < this.layers[this.layers.length - 1].units; ++j) {
                this.label_list_one_hot[i][j] = (j === image_label_list[i]) ? 1 : 0;
            }
        }
    }

    _forward(image_data_list) {
        this.layers[0].forward(image_data_list);
        for (let i = 1; i < this.layers.length; ++i) {
            this.layers[i].forward();
        }
    }

    _backward() {
        for (let i = this.layers.length - 1; i > 0; --i) {
            this.layers[i].backward();
        }
    }

    _mini_batch() {
        for (let i = this.layers.length - 1; i > 0; --i) {
            this.layers[i].batch_update();
        }
    }

    predict(image_data_list) {
        this.mini_batch_size = image_data_list.length;
        this.batch_learning_rate = this.learning_rate;
        this.layers[0].forward(image_data_list);
        for (let i = 1; i < this.layers.length; ++i) {
            this.layers[i].forward();
        }
        let outputLayer = this.layers[this.layers.length - 1];
        return outputLayer.output;
    }
}