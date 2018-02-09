import React from 'react';
import PureCnn from '../model/model';
import model_predict from '../model/model.json';
import {LAYER_CONV, LAYER_FULLY_CONNECTED} from '../util/constant';

class PureCnnPredicting extends React.Component {
    constructor(props) {
        super(props);
        this.type = -1;
        this.state = {guess: undefined, guess_mat: undefined};
    };

    componentDidMount() {
        let web_url = this.props.web_url;
        let base64 = this.props.base64;

        if (web_url !== undefined) {
            this.type = 1;
        } else if (base64 !== undefined) {
            this.type = 0;
        } else {
            this.type = -1;
        }

        if (this.type === -1) {
            throw new Error("Not supported origin image type.")
        }
        this._load_model_from_json(model_predict);
    };

    predict_do() {
        this.preprocess(this.predict.bind(this));
    }

    preprocess(callback) {
        let canvas;
        if (this.props.show_result) {
            canvas = document.getElementById("canvas_ori");
        }
        else {
            canvas = document.createElement("canvas");
        }

        let ctx = canvas.getContext("2d");
        let image = new Image();

        let result_image_data = undefined;

        switch (this.type) {
            case 0: {
                image.src = this.props.base64;
                break;
            }
            case 1: {
                image.src = this.props.web_url;
                break;
            }
            default: {
                throw new Error("Not supported origin image type.");
            }
        }

        image.onload = function () {
            canvas.width = image.width;
            canvas.height = image.height;
            ctx.drawImage(image, 0, 0);

            let image_data = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);

            for (let i = 0; i < image_data.data.length; i += 4) {
                image_data.data[i] = 255 - image_data.data[i];
                image_data.data[i + 1] = 255 - image_data.data[i + 1];
                image_data.data[i + 2] = 255 - image_data.data[i + 2];
            }

            let canvas2 = document.createElement("canvas");
            canvas2.width = image_data.width;
            canvas2.height = image_data.height;
            let ctx2 = canvas2.getContext("2d");
            ctx2.mozImageSmoothingEnabled = false;
            ctx2.webkitImageSmoothingEnabled = false;
            ctx2.msImageSmoothingEnabled = false;
            ctx2.imageSmoothingEnabled = false;
            ctx2.putImageData(image_data, 0, 0);

            let canvas_result;
            if (this.props.show_result) {
                canvas_result = document.getElementById("canvas_res");
            } else {
                canvas_result = document.createElement("canvas");
            }

            canvas_result.width = 24;
            canvas_result.height = 24;
            let ctx_result = canvas_result.getContext("2d");
            ctx_result.mozImageSmoothingEnabled = false;
            ctx_result.webkitImageSmoothingEnabled = false;
            ctx_result.msImageSmoothingEnabled = false;
            ctx_result.imageSmoothingEnabled = false;

            ctx_result.drawImage(canvas2, 0, 0, canvas2.width, canvas2.height, 0, 0, 24, 24);
            result_image_data = ctx_result.getImageData(0, 0, 24, 24);

            this.setState({canvas_ori: canvas, canvas_res: canvas_result});

            callback(result_image_data);
        }.bind(this);
        image.onerror = function () {
            console.log("No image could be loaded.")
        }
    };

    _load_model_from_json(model_json) {
        this.model = new PureCnn();

        if (model_json.momentum !== undefined) this.model.set_momentum(model_json.momentum);
        if (model_json.l2 !== undefined) this.model.set_l2(model_json.l2);
        if (model_json.learning_rate !== undefined) this.model.set_learning_rate(model_json.learning_rate);

        for (let layer_index = 0; layer_index < model_json.layers.length; ++layer_index) {
            let layerDesc = model_json.layers[layer_index];
            this.model.add_layer(layerDesc);
        }

        for (let layer_index = 0; layer_index < model_json.layers.length; ++layer_index) {
            let layer_info = model_json.layers[layer_index];

            switch (model_json.layers[layer_index].type) {
                case LAYER_CONV:
                case LAYER_FULLY_CONNECTED: {
                    if (layer_info.weight !== undefined && layer_info.biases !== undefined) {
                        this.model.layers[layer_index].set_params(layer_info.weight, layer_info.biases);
                    }
                    break;
                }
                default:
                    break;
            }
        }
    };

    predict(input_image_data) {
        if (this.model === undefined) {
            throw new Error("Model did not generated.");
        }

        const result = this.model.predict([input_image_data]);

        let guess = 0;
        let max = 0;
        for (let i = 0; i < 10; ++i) {
            if (result[0].get_value_by_coordinate(0, 0, i) > max) {
                max = result[0].get_value_by_coordinate(0, 0, i);
                guess = i;
            }
        }
        this.setState({guess: guess, guess_mat: result});
    };

    render() {
        console.log(this.state);
        let list_items;
        if (this.state.guess_mat !== undefined && this.state.canvas_ori !== undefined) {
            list_items = Array.prototype.slice.call(this.state.guess_mat[0].values).map((number, index) =>
                <li key={index}>
                    {index} : {number}
                </li>);
        } else {
            list_items = null;
        }

        let result = null;

        if (this.props.show_result) {
            result = (<div>
                <canvas id="canvas_ori"/>
                <canvas id="canvas_res"/>
                <ul>{this.state.guess}</ul>
                <ul>{list_items}</ul>
            </div>);
        }
        return result
    }
}

export default PureCnnPredicting;