export default class Shape {
    constructor(width, height, depth) {
        this.width = Math.floor(width);
        this.height = Math.floor(height);
        this.depth = Math.floor(depth);
    }

    get_size() {
        return this.width * this.height * this.depth;
    }
}