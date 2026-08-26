class tl:
    @staticmethod
    def load(pointer, mask):
        return pointer

    @staticmethod
    def store(pointer, value, mask):
        return None


def kernel(scores, output, token, expert, e):
    row = tl.load(scores, mask=expert < e)
    tl.store(output, row, mask=expert < e)
