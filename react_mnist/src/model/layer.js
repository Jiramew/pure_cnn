import {ACTIVATION_LINEAR} from "../util/constant";

export default class Layer {

    constructor(name, units) {
        this.name = name;
        this.units = units;
        this.activation = ACTIVATION_LINEAR;
    }

    is_last_layer() {
        return this.next_layer === undefined;
    }

    set_input_layer(input_layer) {
        this.pre_layer = input_layer;
        this.input_shape = input_layer.output_shape;
    }

    set_output_layer(output_layer) {
        this.next_layer = output_layer;
    }

    batch_update() {
    }
}