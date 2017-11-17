"""Model config file."""
import numpy as np
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

NTREES = 100

class SKModel:
    def __init__(self, metadata):
        self.imp = Imputer(missing_values="NaN", strategy="mean", axis=0,
                           verbose=0, copy=True)

        psize = (2 * metadata.halfwidth + 1)**2
        n_values = [k for k in metadata.ncategories for _ in range(psize)]
        self.enc = OneHotEncoder(n_values=n_values, categorical_features="all",
                                 dtype=np.float32, sparse=False)

        self.est = RandomForestRegressor(n_estimators=NTREES)

    def fit(self, Xo: np.ndarray, Xc: np.ndarray, Y: np.array):
        X_onehot = self.enc.fit_transform(Xc)
        X_imputed = self.imp.fit_transform(Xo)
        X = np.concatenate([X_onehot, X_imputed], axis=1)
        self.est.fit(X, Y)


    def predict(self, Xo: np.ma.MaskedArray, Xc: np.ma.MaskedArray):
        X_onehot = self.enc.transform(Xc)
        X_imputed = self.imp.transform(Xo)
        X = np.concatenate([X_onehot, X_imputed], axis=1)
        Ey = self.est.predict(X)
        quantiles = []
        return Ey, quantiles



