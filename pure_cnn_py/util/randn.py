import math
import random


class Randn(object):
    def __init__(self):
        self.next_value = None

    def get_randn(self):
        if self.next_value is not None:
            result = self.next_value
            self.next_value = None
            return result

        a = 0
        b = 0
        s = 0

        while s > 1 or s == 0:
            a = random.random() * 2 - 1
            b = random.random() * 2 - 1
            s = math.pow(a, 2) + math.pow(b, 2)

        multiple = math.sqrt(-2 * math.log(s, math.e) / s)

        self.next_value = b * multiple

        return a * multiple


if __name__ == '__main__':
    rn = Randn()
    for i in range(0, 10):
        print(rn.get_randn())
