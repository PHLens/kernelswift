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
    kernel[(1,)](scores, output, token, expert, e, num_warps=2, num_stages=2)
