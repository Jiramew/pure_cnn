import React, {Component} from 'react';
import './App.css';
import PureCnnPredicting from 'react_mnist';

class App extends Component {
    constructor(props) {
        super(props);
        this.state = {show_result: true, base64: ""};
    }

    valueChange(e) {
        console.log(e.target.value);
        this.setState({base64: e.target.value});
    }

    predict() {
        this.refs.pred.predict_do();
    }

    render() {
        console.log(this.state);
        return (
            <div className="App">
                Base64:<input onChange={(e) => this.valueChange(e)}/>
                <button onClick={() => this.predict()}>Predict</button>
                <PureCnnPredicting
                    ref="pred"
                    show_result={this.state.show_result}
                    base64={this.state.base64}/>
            </div>
        );
    }
}

export default App;

// examples:
// 9: data:image/png;base64,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

// 3: data:image/png;base64,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
