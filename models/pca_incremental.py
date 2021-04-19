from sklearn import decomposition


class IncrementalPCA():
    def __init__(self, n_components: int):
        self._ipca = decomposition.IncrementalPCA(n_components)

    def partial_fit(self, batch_in):
        self._ipca.partial_fit(batch_in)

    def transform(self, batch_in):
        return self._ipca.transform(batch_in)

    def inverse_transform(self, batch_in):
        return self._ipca.inverse_transform(batch_in)

    @property
    def components_(self):
        return self._ipca.components_

    @property
    def explained_variance_(self):
        return self._ipca.explained_variance_

    @property
    def explained_variance_ratio_(self):
        return self._ipca.explained_variance_ratio_

    @property
    def mean_(self):
        return self._ipca.mean_
    
    @property
    def var_(self):
        return self._ipca.var_

