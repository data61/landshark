"""Model config file."""
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

NTREES = 100

class SKModel:
    def __init__(self, metadata):
        self.imp = Imputer(missing_values="NaN", strategy="mean", axis=0,
                           verbose=0, copy=True)
        self.est = RandomForestClassifier(n_estimators=NTREES)
        self.ncats = len(metadata.target_map[0])

    def fit(self, Xo: np.ndarray, Xc: np.ndarray, Y: np.array):
        X = self.imp.fit_transform(Xo)
        self.est.fit(X, Y)


    def predict(self, Xo: np.ma.MaskedArray, Xc: np.ma.MaskedArray):
        X = self.imp.transform(Xo)
        Py = self.est.predict_proba(X)
        p = np.zeros((Py.shape[0], self.ncats), dtype=np.float32)
        for idx, ps in zip(self.est.classes_, Py.T):
            p[:, idx] = ps
        Ey = np.argmax(p, axis=1).astype(np.int32)
        return Ey, p
