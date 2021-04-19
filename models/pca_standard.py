from sklearn import decomposition
from utils.dtypes import *


class StandardPCA:
    def __init__(self, n_components: Optional[int]):
        self._pca = decomposition.PCA(n_components=n_components)

    def fit(self, x: np.ndarray):
        return self._pca.fit(x)

    def transform(self, x: np.ndarray):
        return self._pca.transform(x)

    def inverse_transform(self, x_reduced):
        return self._pca.inverse_transform(x_reduced)