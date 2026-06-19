
class Debugger():
    def __init__(self):
        self.is_debug = False

    def debug(self, *x):
        if self.is_debug:
            print(*x)
