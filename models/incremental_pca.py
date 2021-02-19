from sklearn import decomposition

class IncrementalPCA():
    def __init__(self):
        self.ipca = decomposition.IncrementalPCA()
        return

    def partial_fit(self, batch_in):
        self.ipca.partial_fit(batch_in)

    def transform(self, batch_in):
        self.ipca.transform(batch_in)

