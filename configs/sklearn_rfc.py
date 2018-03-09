"""Model config file."""
import numpy as np
from sklearn.preprocessing import Imputer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

NTREES = 100


class SKModel:
    def __init__(self, metadata):
        self.imp = Imputer(missing_values="NaN", strategy="mean", axis=0,
                           verbose=0, copy=True)

        psize = (2 * metadata.halfwidth + 1)**2
        if metadata.ncategories:
            n_values = [k for k in metadata.ncategories for _ in range(psize)]
            self.enc = OneHotEncoder(n_values=n_values,
                                     categorical_features="all",
                                     dtype=np.float32, sparse=False)

        self.est = RandomForestClassifier(n_estimators=NTREES)

    def fit(self, Xo: np.ndarray, Xc: np.ndarray, Y: np.array):
        X_list = []
        if Xc is not None:
            X_onehot = self.enc.fit_transform(Xc)
            X_list.append(X_onehot)
        if Xo is not None:
            X_imputed = self.imp.fit_transform(Xo)
            X_list.append(X_imputed)
        X = np.concatenate(X_list, axis=1)
        self.est.fit(X, Y)


    def predict(self, Xo: np.ma.MaskedArray, Xc: np.ma.MaskedArray):
        X_list = []
        if Xc is not None:
            X_onehot = self.enc.transform(Xc)
            X_list.append(X_onehot)
        if Xo is not None:
            X_imputed = self.imp.transform(Xo)
            X_list.append(X_imputed)
        X = np.concatenate(X_list, axis=1)
        Py = self.est.predict_proba(X)
        Ey = np.argmax(Py, axis=1)
        return Ey, Py
