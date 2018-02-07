import React, {Component} from 'react';
import {
    BrowserRouter as Router,
    Route,
    Link
} from 'react-router-dom'

import Training from './training'
import Testing from './testing'

class App extends Component {
    componentDidMount() {

    }

    render() {
        const Home = () => (
            <div>
                <h2>Home</h2>
            </div>
        );
        return (
            <Router>
                <div>
                    <ul>
                        <li><Link to="/">Home</Link></li>
                        <li><Link to="/training">Training</Link></li>
                        <li><Link to="/testing">Testing</Link></li>
                    </ul>
                    <hr/>
                    <Route exact path="/" component={Home}/>
                    <Route path="/training" component={Training}/>
                    <Route path="/testing" component={Testing}/>
                </div>
            </Router>
        );
    }
}

export default App;
