import React, {Component} from 'react';
import ReactDOM from 'react-dom';
import './training.css';

import {ACTIVATION_SOFTMAX, LAYER_CONV, LAYER_FULLY_CONNECTED, LAYER_INPUT, LAYER_MAXPOOL} from './util/constant';
import {digit_labels} from './util/mnist_digit_labels';

import mnist_test from './resource/mnist_test.png';
import mnist_validation from './resource/mnist_validation.png';
import mnist_training_0 from './resource/mnist_training_0.png';
import mnist_training_1 from './resource/mnist_training_1.png';
import mnist_training_2 from './resource/mnist_training_2.png';
import mnist_training_3 from './resource/mnist_training_3.png';
import mnist_training_4 from './resource/mnist_training_4.png';

import PureCnn from './model/model';

class Training extends Component {
    constructor(props) {
        super(props);

        this.image_training_ori = [
            mnist_training_0,
            mnist_training_1,
            mnist_training_2,
            mnist_training_3,
            mnist_training_4
        ];

        this.digit_labels = digit_labels;

        this.train_num = 50000;
        this.validate_num = 10000;
        this.validate_offset = 50000;

        this.input_per_file_num = 10000;

        this.mini_batch_size = 20;

        this.iter = 0;
        this.epoch = 0;
        this.example_seen = 0;

        this.running = false;
        this.paused = false;
        this.testing = false;

        this.timeoutID = undefined;

        this.train_images = new Array(5);
        this.train_images_context = new Array(5);
        this.train_image_loaded = [false, false, false, false, false];

        this.validate_image = undefined;
        this.validate_image_loaded = false;
        this.validate_image_context = undefined;
        this.test_image = undefined;
        this.test_image_context = undefined;

        this.forward_time = 0;
        this.backward_time = 0;

        this.model = undefined;
    }

    componentDidMount() {
        this.init();
    }

    componentWillUnmount() {
        this.running = false;
        this.paused = false;
        this.testing = false;
    }

    init = () => {
        for (let i = 0; i < 5; ++i) {
            this.train_images[i] = new Image();
            this.train_images[i].src = this.image_training_ori[i];
            this.train_images[i].onload = this.on_train_image_load.bind(this);
        }

        this.validate_image = new Image();
        this.validate_image.src = mnist_validation;
        this.validate_image.onload = this.on_validate_image_load.bind(this);

        this.test_image = new Image();
        this.test_image.src = mnist_test;
        this.test_image.onload = this.on_test_image_load.bind(this);

        this.initialize_network();

    };

    check_image_load = () => {
        if (this.train_image_loaded[0] && this.validate_image_loaded) {
            const btn = document.getElementById("startButton");
            btn.innerHTML = "Start";
        }
    };

    initialize_network = () => {
        this.model = new PureCnn(this.mini_batch_size);
        this.model.add_layer({name: "image", type: LAYER_INPUT, width: 24, height: 24, depth: 1});
        this.model.add_layer({
            name: "conv1",
            type: LAYER_CONV,
            units: 10,
            kernel_width: 5,
            kernel_height: 5,
            pool_stride_x: 1,
            pool_stride_y: 1,
            padding: false
        });
        this.model.add_layer({
            name: "pool1",
            type: LAYER_MAXPOOL,
            pool_width: 2,
            pool_height: 2,
            pool_stride_x: 2,
            pool_stride_y: 2
        });
        this.model.add_layer({
            name: "conv2",
            type: LAYER_CONV,
            units: 20,
            kernel_width: 5,
            kernel_height: 5,
            pool_stride_x: 1,
            pool_stride_y: 1,
            padding: false
        });
        this.model.add_layer({
            name: "pool2",
            type: LAYER_MAXPOOL,
            pool_width: 2,
            pool_height: 2,
            pool_stride_x: 2,
            pool_stride_y: 2
        });
        this.model.add_layer({name: "out", type: LAYER_FULLY_CONNECTED, units: 10, activation: ACTIVATION_SOFTMAX});

        this.model.set_learning_rate(0.01);
        this.model.set_momentum(0.9);
        this.model.set_l2(0.0);
    };

    train = () => {
        if (this.model === undefined) {
            return;
        }

        if (!this.running) {
            return;
        }

        this.timeoutID = undefined;

        if (this.iter < this.train_num) {
            let train_image_batch = [];
            let train_label_batch = [];

            for (let i = 0; (i < this.mini_batch_size && this.iter < this.train_num); ++i, ++this.iter) {
                train_image_batch[i] = this.get_train_image_data(this.iter);
                train_label_batch[i] = this.digit_labels[this.iter];
            }

            this.example_seen += this.mini_batch_size;

            this.model.train(train_image_batch, train_label_batch);

            this.forward_time = String(Math.floor(10 * this.model.forward_time) / 10.0);
            this.backward_time = String(Math.floor(10 * this.model.backward_time) / 10.0);

            if (this.forward_time.indexOf(".") === -1) {
                this.forward_time += ".0";
            }
            if (this.backward_time.indexOf(".") === -1) {
                this.backward_time += ".0";
            }
            let accuracy = this.validate_accuracy();

            document.getElementById("example_seen").innerHTML = String(this.example_seen);
            document.getElementById("forward_time").innerHTML = this.forward_time + " ms";
            document.getElementById("backward_time").innerHTML = this.backward_time + " ms";
            document.getElementById("mini_batch_loss").innerHTML = (Math.floor(1000.0 * this.model.training_error) / 1000.0) + "";
            document.getElementById("train_accuracy").innerHTML = (Math.floor(1000.0 * accuracy) / 10.0) + "%";

            this.epoch++;

            if (!this.paused) {
                this.timeoutID = setTimeout(this.train, 0);
            }
            else {
                console.log("Pausing after iteration " + this.iter);
            }
        }
        else {
            this.running = false;
            this.paused = false;
            this.iter = 0;
            let btn = document.getElementById("startButton");
            btn.innerHTML = "Keep Training";
            let accuracy = this.validate_accuracy();
            document.getElementById("train_accuracy").innerHTML = (Math.floor(1000.0 * accuracy) / 10.0) + "%";
        }
    };

    validate_accuracy = () => {
        let correct = 0;
        let image_data_list = [];
        let image_label_list = [];

        for (let i = 0; i < 100; i += 10) {
            for (let j = 0; j < 10; j++) {
                let validate_image_index = Math.floor(Math.random() * this.validate_num);
                let validate_image_label = digit_labels[validate_image_index + this.validate_offset];

                for (let rand = 0; rand < 1; rand++) {
                    image_data_list[j + rand] = this.get_validate_image_data(validate_image_index);
                    image_label_list[j + rand] = validate_image_label;
                }
            }

            let results = this.model.predict(image_data_list);

            for (let m = 0; m < 10; m++) {
                let guess = 0;
                let max_value = 0;

                for (let c = 0; c < results[m].shape.depth; c++) {
                    let c_sum = 0;
                    for (let rand = 0; rand < 1; rand++) {
                        c_sum += results[m + rand].get_value_by_coordinate(0, 0, c);
                    }

                    if (c_sum > max_value) {
                        max_value = c_sum;
                        guess = c;
                    }
                }
                if (guess === image_label_list[m]) {
                    correct++;
                }
            }
        }
        return correct / 100;
    };

    get_train_image_data = (n) => {
        let image_file_index = Math.floor(n / this.input_per_file_num);
        let image_index = n % this.input_per_file_num;

        let y = 28 * Math.floor(image_index / 100) + Math.floor(Math.random() * 5);
        let x = 28 * (image_index % 100) + Math.floor(Math.random() * 5);

        return this.train_images_context[image_file_index].getImageData(x, y, 24, 24);
    };

    get_validate_image_data = (n) => {
        let y = 28 * Math.floor(n / 100) + Math.floor(Math.random() * 5);
        let x = 28 * (n % 100) + Math.floor(Math.random() * 5);

        return this.validate_image_context.getImageData(x, y, 24, 24);
    };

    button_click = (n) => {
        let btn = ReactDOM.findDOMNode(this.refs.btn);
        switch (n) {
            case 1: {
                if (this.train_image_loaded[0] && this.validate_image_loaded && !this.running && !this.testing && this.model !== undefined) {
                    this.running = true;
                    this.paused = false;

                    btn.innerHTML = "Pause";

                    this.train();
                }
                else if (this.running && !this.paused) {
                    this.paused = true;
                    btn.innerHTML = "Resume";
                }
                else if (this.running && this.paused && !this.testing) {
                    this.paused = false;
                    btn.innerHTML = "Pause";

                    if (this.timeoutID === undefined) {
                        this.train();
                    }
                }
                break;
            }
            default:
                break;
        }
    };

    on_validate_image_load = (event) => {
        let canvas = document.createElement("canvas");
        canvas.width = 2800;
        canvas.height = 2800;
        this.validate_image_context = canvas.getContext("2d");
        this.validate_image_context.drawImage(this.validate_image, 0, 0);
        this.validate_image_loaded = true;
        this.check_image_load()
    };

    on_test_image_load = (event) => {
        let canvas = document.createElement("canvas");
        canvas.width = 2800;
        canvas.height = 2800;
        this.test_image_context = canvas.getContext("2d");
        this.test_image_context.drawImage(this.test_image, 0, 0);
        this.check_image_load()
    };

    on_train_image_load = (event) => {
        for (let i = 0; i < 5; ++i) {
            if (this.train_images[i] === event.target) {
                let canvas = document.createElement("canvas");
                canvas.width = 2800;
                canvas.height = 2800;
                this.train_images_context[i] = canvas.getContext("2d");
                this.train_images_context[i].drawImage(event.target, 0, 0);
                this.train_image_loaded[i] = true;
                break;
            }
        }
        this.check_image_load()
    };

    render() {
        return (
            <div className="App">
                <div className="App-header">
                    <div className="App-title">CNN Training</div>
                </div>
                <div className="controlsDiv">
                    <div className="buttonContainer">
                        <button ref="btn"
                                className="silverButton" onClick={() => this.button_click(1)}
                                id="startButton">Loading
                        </button>
                    </div>

                    <br/>

                    Train Num: <span id="example_seen"/><br/>
                    Forward Duration: <span id="forward_time"/><br/>
                    Backward Duration: <span id="backward_time"/><br/>
                    Loss: <span id="mini_batch_loss"/><br/>
                    Accuracy: <span id="train_accuracy"/><br/>
                </div>
            </div>
        );
    }
}

export default Training;