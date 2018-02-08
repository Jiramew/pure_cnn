import Randn from './randn';
import {ACTIVATION_RELU, ACTIVATION_SOFTMAX, ACTIVATION_TANH, INIT_CHARS, INIT_RANDN} from "./constant";


export default class Mat {
    constructor(shape, initType) {
        this.shape = shape;
        this.size = shape.get_size();
        this.values = new Float32Array(this.size);

        if (initType === INIT_RANDN) {
            let sd = 1.0 / Math.sqrt(this.shape.width * this.shape.height * this.shape.depth);
            let rng = new Randn();
            for (let i = 0; i < this.size; ++i) {
                this.values[i] = sd * rng.getNextRandom();
            }
        }
        else if (initType === INIT_CHARS) {
            this.values = new Array(this.size)
        }
    }

    operation_scale_and_add_mat(scale, add_mat) {
        for (let i = 0; i < this.size; ++i) {
            this.values[i] *= scale;
            this.values[i] += add_mat.values[i];
        }
    }

    operation_add_scaled_mat(scale, add_mat) {
        for (let i = 0; i < this.size; ++i) {
            this.values[i] += scale * add_mat.values[i];
        }
    }

    operation_scale_mat(scale) {
        for (let i = 0; i < this.size; ++i) {
            this.values[i] *= scale;
        }
    }

    activate(activationType) {
        switch (activationType) {
            case ACTIVATION_RELU: {
                for (let i = 0; i < this.size; ++i) {
                    this.values[i] = Math.max(0, this.values[i]);
                }

                break;
            }

            case ACTIVATION_TANH: {
                for (let i = 0; i < this.size; ++i) {
                    this.values[i] = Math.tanh(this.values[i]);
                }

                break;
            }

            case ACTIVATION_SOFTMAX: {
                let max_value = 0;
                let sum_value = 0;

                for (let i = 0; i < this.size; ++i) {
                    max_value = Math.max(max_value, this.values[i]);
                }

                for (let i = 0; i < this.size; ++i) {
                    this.values[i] = Math.exp(this.values[i] - max_value);
                    sum_value += this.values[i];
                }

                for (let i = 0; i < this.size; ++i) {
                    this.values[i] /= sum_value;
                }
                break;
            }

            default: {
                alert("No such activation type" + activationType + ".");
            }
        }
    }

    get_value() {
        return this.values;
    }

    get_value_array() {
        return Array.from(this.values);
    }

    set_value(array) {
        this.values = new Float32Array(array);
    }

    set_value_by_image(image_data, depth) {
        const scale = 1.0 / 255;

        const area = image_data.width * image_data.height;

        for (let d = 0; d < depth; d++) {
            for (let h = 0; h < image_data.height; h++) {
                for (let w = 0; w < image_data.width; w++) {

                    let img_index = 4 * (image_data.width * h + w);
                    let max_index = area * d + image_data.width * h + w;

                    this.values[max_index] = image_data.data[img_index + d] * scale;
                }
            }
        }
    }

    set_value_by_coordinate(x, y, z, v) {
        const mat_index = (z * this.shape.height + y) * this.shape.width + x;
        this.values[mat_index] = v;
    }

    add_value_by_coordinate(x, y, z, v) {
        const mat_index = (z * this.shape.height + y) * this.shape.width + x;
        this.values[mat_index] += v;
    }

    get_value_by_coordinate(x, y, z) {
        const mat_index = (z * this.shape.height + y) * this.shape.width + x;
        return this.values[mat_index];
    }
}