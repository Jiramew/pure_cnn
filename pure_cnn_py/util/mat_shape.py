class MatShape(object):
    def __init__(self, width, height, depth):
        self.width = int(width)
        self.height = int(height)
        self.depth = int(depth)

    def get_size(self):
        return int(self.width * self.height * self.depth)
