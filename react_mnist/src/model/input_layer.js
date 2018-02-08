import Layer from './layer'
import Mat from '../util/mat'
import Shape from '../util/mat_shape'
import {INIT_ZEROS, LAYER_INPUT} from "../util/constant";


export default class InputLayer extends Layer {
    constructor(layerName, imageWidth, imageHeight, imageDepth) {
        super(layerName, 1);
        this.type = LAYER_INPUT;
        this.output_shape = new Shape(imageWidth, imageHeight, imageDepth);
        this.output = [];
    }

    forward(image_data_list) {
        for (let i = 0; i < this.network.mini_batch_size; i++) {
            this.output[i] = new Mat(this.output_shape, INIT_ZEROS);
            this.output[i].set_value_by_image(image_data_list[i], this.output_shape.depth);
        }
    }

    backward() {
    }
}