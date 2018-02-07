export default class Randn {
    constructor() {
        this.next_value = undefined;
    }

    getNextRandom() {
        if (this.next_value !== undefined) {
            const result = this.next_value;
            this.next_value = undefined;
            return result;
        }

        let a, b, s = 0;
        while (s > 1 || s === 0) {
            a = Math.random() * 2 - 1;
            b = Math.random() * 2 - 1;
            s = Math.pow(a, 2) + Math.pow(b, 2)
        }

        const multiple = Math.sqrt(-2 * Math.log(s) / s);
        this.next_value = a * multiple;
        return b * multiple;
    }
}
