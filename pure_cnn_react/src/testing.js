import React, {Component} from 'react';
import model_test from './mnist.json';
import PureCnn from './model/model';
import {LAYER_CONV, LAYER_FULLY_CONNECTED} from './util/constant';

import './testing.css';

const PI2 = Math.PI * 2;


class Testing extends Component {
    reset_draw = () => {
        if (this.ctx_draw === undefined)
            return;

        this.drawing_dots_index = 0;
        this.drawing_dot_list = [];
        this.fresh_canvas = true;
        this.ctx_draw.fillStyle = "white";
        this.ctx_draw.fillRect(0, 0, this.draw_canvas.width, this.draw_canvas.height);
        this.ctx_draw.fillStyle = "rgb(190,190,190)";
        this.ctx_draw.font = "20px Arial";

        this.ctx_draw.fillText("Paint 0-9 here", 88, 156);

    };
    on_mouse_down = (e) => {
        if (this.fresh_canvas) {
            this.fresh_canvas = false;
            this.ctx_draw.fillStyle = "white";
            this.ctx_draw.fillRect(0, 0, this.draw_canvas.width, this.draw_canvas.height);
        }
        this.drawing = true;
        this.drawing_path_index++;
        this.drawing_path_list[this.drawing_path_index] = [];
        this.drawing_dot_list[this.drawing_dots_index] = [e.nativeEvent.offsetX, e.nativeEvent.offsetY];
        this.drawing_dots_index++;
        this.ctx_draw.strokeStyle = "black";
        this.ctx_draw.fillStyle = "black";
        this.ctx_draw.lineCap = "round";
        this.ctx_draw.lineJoin = "round";
        this.ctx_draw.beginPath();
        this.ctx_draw.arc(e.nativeEvent.offsetX, e.nativeEvent.offsetY, this.lineWidth / 2, 0, PI2);
        this.ctx_draw.fill();
    };
    on_mouse_up = () => {
        if (this.drawing) {
            this.guessNumber();
            this.drawing = false;
            this.last_position = undefined;
        }
    };
    on_mouse_out = () => {
        this.drawing = false;
        this.last_position = undefined;
    };
    on_mouse_over = (e) => {

    };
    on_mouse_leave = () => {
        this.drawing = false;
        this.last_position = undefined;
    };
    on_mouse_move = (e) => {
        if (!this.drawing) return;

        let x = Math.max(0, Math.min(this.draw_canvas.width, e.nativeEvent.offsetX));
        let y = Math.max(0, Math.min(this.draw_canvas.height, e.nativeEvent.offsetY));

        if (e.nativeEvent.offsetX > 0 && e.nativeEvent.offsetX < e.target.width && e.nativeEvent.offsetY > 0 && e.nativeEvent.offsetY < e.target.height) {
            this.ctx_draw.lineWidth = this.lineWidth;

            if (this.last_position !== undefined) {
                this.ctx_draw.beginPath();
                this.ctx_draw.moveTo(this.last_position[0], this.last_position[1]);
                this.ctx_draw.lineTo(x, y);
                this.ctx_draw.stroke();
            }
            else {
                this.drawing_path_index++;
                this.ctx_draw.beginPath();
                this.ctx_draw.arc(x, y, this.lineWidth / 2, 0, PI2);
                this.ctx_draw.fill();
            }

            if (this.drawing_path_list[this.drawing_path_index] === undefined) {
                this.drawing_path_list[this.drawing_path_index] = [];
            }

            this.drawing_path_list[this.drawing_path_index].push([x, y]);
            this.last_position = [x, y];
        }
        else {
            this.last_position = undefined;
        }
    };
    guessNumber = () => {
        const inputImageData = this.preProcessDrawing();

        if (this.model === undefined) return;

        const result = this.model.predict([inputImageData]);

        let guess = 0;
        let max = 0;
        for (let i = 0; i < 10; ++i) {
            if (result[0].get_value_by_coordinate(0, 0, i) > max) {
                max = result[0].get_value_by_coordinate(0, 0, i);
                guess = i;
            }
        }

        document.getElementById("guessNumberDiv").innerHTML = ( max > 0.666667 ) ? String(guess) : "?";
        document.getElementById("confidence").innerHTML = String(Math.min(100, Math.floor(1000 * ( max + 0.1 )) / 10.0)) + "% it's a " + String(guess);
    };
    preProcessDrawing = () => {
        let drawnImageData = this.ctx_draw.getImageData(0, 0, this.ctx_draw.canvas.width, this.ctx_draw.canvas.height);

        let xmin = this.ctx_draw.canvas.width - 1;
        let xmax = 0;
        let ymin = this.ctx_draw.canvas.height - 1;
        let ymax = 0;
        let w = this.ctx_draw.canvas.width;
        let h = this.ctx_draw.canvas.height;

        for (let i = 0; i < drawnImageData.data.length; i += 4) {
            let x = Math.floor(i / 4) % w;
            let y = Math.floor(i / ( 4 * w ));

            if (drawnImageData.data[i] < 255 || drawnImageData.data[i + 1] < 255 || drawnImageData.data[i + 2] < 255) {
                xmin = Math.min(xmin, x);
                xmax = Math.max(xmax, x);
                ymin = Math.min(ymin, y);
                ymax = Math.max(ymax, y);
            }
        }

        const cropWidth = xmax - xmin;
        const cropHeight = ymax - ymin;

        if (cropWidth > 0 && cropHeight > 0 && ( cropWidth < w || cropHeight < h )) {
            const scaleX = cropWidth / w;
            const scaleY = cropHeight / h;
            const scale = Math.max(scaleX, scaleY);
            const scaledLineWidth = Math.max(1, Math.floor(this.lineWidth * scale));
            const scaledDotWidth = Math.max(1, Math.floor(scaledLineWidth / 2));

            const tempCanvas = document.createElement("canvas");

            tempCanvas.width = w;
            tempCanvas.height = h;
            const ctx_temp = tempCanvas.getContext("2d");

            ctx_temp.strokeStyle = "black";
            ctx_temp.fillStyle = "black";
            ctx_temp.lineCap = "round";
            ctx_temp.lineJoin = "round";
            ctx_temp.lineWidth = scaledLineWidth;

            for (let pathIndex = 0; pathIndex < this.drawing_path_list.length; ++pathIndex) {
                let path = this.drawing_path_list[pathIndex];
                if (path === undefined || path.length === 0) {
                    continue;
                }
                let p = path[0];
                ctx_temp.beginPath();
                ctx_temp.moveTo(p[0], p[1]);

                for (let i = 1; i < path.length; ++i) {
                    p = path[i];
                    ctx_temp.lineTo(p[0], p[1]);
                }
                ctx_temp.stroke();
            }

            for (let dotIndex = 0; dotIndex < this.drawing_dot_list.length; ++dotIndex) {
                let dotPos = this.drawing_dot_list[dotIndex];
                ctx_temp.beginPath();
                ctx_temp.arc(dotPos[0], dotPos[1], scaledDotWidth, 0, PI2);
                ctx_temp.fill();
            }

            drawnImageData = ctx_temp.getImageData(xmin, ymin, cropWidth, cropHeight);
        }

        for (let i = 0; i < drawnImageData.data.length; i += 4) {
            drawnImageData.data[i] = 255 - drawnImageData.data[i];
            drawnImageData.data[i + 1] = 255 - drawnImageData.data[i + 1];
            drawnImageData.data[i + 2] = 255 - drawnImageData.data[i + 2];
        }

        let canvas2 = document.createElement("canvas");
        canvas2.width = drawnImageData.width;
        canvas2.height = drawnImageData.height;
        let ctx2 = canvas2.getContext("2d");
        ctx2.mozImageSmoothingEnabled = false;
        ctx2.webkitImageSmoothingEnabled = false;
        ctx2.msImageSmoothingEnabled = false;
        ctx2.imageSmoothingEnabled = false;
        ctx2.putImageData(drawnImageData, 0, 0);

        let canvas = document.createElement("canvas");
        canvas.width = 24;
        canvas.height = 24;
        let ctx = canvas.getContext("2d");
        ctx.mozImageSmoothingEnabled = false;
        ctx.webkitImageSmoothingEnabled = false;
        ctx.msImageSmoothingEnabled = false;
        ctx.imageSmoothingEnabled = false;

        let xOffset = 0;
        let yOffset = 0;
        let xScale = 1;
        let yScale = 1;
        const padding = 1;

        if (canvas2.width > canvas2.height) {
            yOffset = ( canvas.width / ( canvas2.width + 2 * padding) ) * ( canvas2.width - canvas2.height ) / 2 + padding;
            yScale = canvas2.height / canvas2.width;

            xOffset = padding;

        }
        else if (canvas2.height > canvas2.width) {
            xOffset = ( canvas.height / canvas2.height ) * ( canvas2.height - canvas2.width ) / 2 + padding;
            xScale = canvas2.width / canvas2.height;

            yOffset = padding;
        }

        ctx.fillStyle = "black";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(canvas2, xOffset, yOffset, canvas.width * xScale - 2 * padding, canvas.height * yScale - 2 * padding);

        return ctx.getImageData(0, 0, 24, 24);
    };
    load_model_from_json = (model_json) => {
        this.model = new PureCnn();

        if (model_json.momentum !== undefined) this.model.set_momentum(model_json.momentum);
        if (model_json.l2 !== undefined) this.model.set_l2(model_json.l2);
        if (model_json.learning_rate !== undefined) this.model.set_learning_rate(model_json.learning_rate);

        for (let layer_index = 0; layer_index < model_json.layers.length; ++layer_index) {
            let layerDesc = model_json.layers[layer_index];
            this.model.add_layer(layerDesc);
        }

        for (let layerIndex = 0; layerIndex < model_json.layers.length; ++layerIndex) {
            let layerDesc = model_json.layers[layerIndex];

            switch (model_json.layers[layerIndex].type) {
                case LAYER_CONV:
                case LAYER_FULLY_CONNECTED: {
                    if (layerDesc.weight !== undefined && layerDesc.biases !== undefined) {
                        this.model.layers[layerIndex].set_params(layerDesc.weight, layerDesc.biases);
                    }
                    break;
                }
                default:
                    break;
            }
        }
    };
    button_click = (n) => {
        switch (n) {
            case 1: {
                this.drawing_path_list = [];
                this.drawing_path_index = -1;
                this.last_position = undefined;

                this.reset_draw();
                break;
            }

            default: {
                break
            }
        }
    };

    constructor(props) {
        super(props);
        this.draw_canvas = undefined;
        this.ctx_draw = undefined;
        this.drawing = false;

        this.last_position = undefined;
        this.lineWidth = 16;

        this.drawing_path_index = -1;
        this.drawing_dots_index = 0;
        this.drawing_path_list = [];
        this.drawing_dot_list = [];
        this.fresh_canvas = true;

        this.model = undefined;
    }

    componentDidMount() {
        this.draw_canvas = document.getElementById("drawCanvas");
        this.ctx_draw = this.draw_canvas.getContext("2d");
        this.ctx_draw.mozImageSmoothingEnabled = false;
        this.ctx_draw.webkitImageSmoothingEnabled = false;
        this.ctx_draw.msImageSmoothingEnabled = false;
        this.ctx_draw.imageSmoothingEnabled = false;

        this.reset_draw();

        window.addEventListener("mouseup", this.on_mouse_up, false);
        this.load_model_from_json(model_test);
    }

    render() {
        return (
            <div className="App">
                <div className="App-header">
                    <div className="App-title">CNN Testing</div>
                </div>
                <div id="recognizer">
                    <div className="inlineDiv">
                        <div id="drawingCanvasDiv">
                            <canvas id="drawCanvas"
                                    width="300"
                                    height="300"
                                    onMouseDown={this.on_mouse_down}
                                    onMouseMove={this.on_mouse_move}
                                    onMouseOut={this.on_mouse_out}
                                    onMouseOver={this.on_mouse_over}
                                    onMouseLeave={this.on_mouse_leave}/>
                        </div>
                        <button className="silverButton" onClick={() => this.button_click(1)} id="clearButton">Reset
                        </button>
                    </div>
                    <div className="inlineDiv">
                        <div id="guessNumberDiv"/>
                        <div>Confidence: <span id="confidence">0%</span></div>
                    </div>
                </div>
            </div>
        );
    }
}

export default Testing;
