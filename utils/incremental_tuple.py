import torch as t


class IncrementalTouple:
    def __init__(self, val=None):
        if val == None or t.isnan(val[0]).any():
            self.val = t.tensor([0, 0])
        else:
            self.val = val

    def reciprocal(self):
        return IncrementalTouple(t.tensor([self.val[1] - self.val[0], self.val[1]]))

    def __add__(self, x):
        return IncrementalTouple(self.val + (x.val if not t.isnan(x).any() else 0))

    def __iadd__(self, x):
        if not t.isnan(x.val).any():
            self.val += x.val
        else:
            import ipdb

            ipdb.set_trace()
        return self

    def item(self):
        return (self.val[0] / self.val[1]).item()

    def __str__(self):
        return f"{self.item()}"

    def __format__(self, x):
        return self.item().__format__(x)
