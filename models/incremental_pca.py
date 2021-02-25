from sklearn import decomposition

class IncrementalPCA():
    def __init__(self, n_components: int):
        self.ipca = decomposition.IncrementalPCA(n_components)
        return

    def partial_fit(self, batch_in):
        self.ipca.partial_fit(batch_in)

    def transform(self, batch_in):
        return self.ipca.transform(batch_in)

    def inverse_transform(self, batch_in):
        return self.ipca.inverse_transform(batch_in)

    @property
    def components_(self):
        return self.ipca.components_

    @property
    def explained_variance_(self):
        return self.ipca.explained_variance_

    @property
    def explained_variance_ratio_(self):
        return self.ipca.explained_variance_ratio_


    @property
    def meanvar_(self):
        return self.ipca.mean_, self.ipca.var_

