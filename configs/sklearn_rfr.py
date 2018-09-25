"""Model config file."""
from typing import Optional, Tuple, Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer, OneHotEncoder

NTREES = 100


class SKModel:
    def __init__(self, metadata, random_seed) -> None:
        self.ord_imp = Imputer(missing_values="NaN", strategy="mean", axis=0,
                               verbose=0, copy=True)
        self.cat_imp = Imputer(missing_values=-1,
                               strategy="most_frequent",
                               axis=0, verbose=0, copy=True)
        psize = (2 * metadata.halfwidth + 1)**2
        if metadata.features.categorical:
            n_values = [k for k in metadata.features.categorical.ncategories
                        for _ in range(psize)]
            self.enc = OneHotEncoder(n_values=n_values,
                                     categorical_features="all",
                                     dtype=np.float32, sparse=False)

        self.est = RandomForestRegressor(n_estimators=NTREES,
                                         random_state=random_seed)

    def train(self, Xo: np.ndarray, Xc: np.ndarray, Y: np.array) -> None:


        X_list = []
        if Xc is not None:
            Xc = Xc.reshape((Xc.shape[0], -1))
            Xc.data[Xc.mask] = -1
            X_cat_imp = self.cat_imp.fit_transform(Xc.data)
            X_onehot = self.enc.fit_transform(X_cat_imp)
            X_list.append(X_onehot)
        if Xo is not None:
            Xo = Xo.reshape((Xo.shape[0], -1))
            Xo.data[Xo.mask] = np.nan
            X_imputed = self.ord_imp.fit_transform(Xo.data)
            X_list.append(X_imputed)
        X = np.concatenate(X_list, axis=1)
        self.est.fit(X, Y)

    def test(self, Y: np.array, predictions: Dict[str, np.array]) \
            -> Dict[str, np.ndarray]:
        return {"mse": 1.0}


    def predict(self, Xo: np.ma.MaskedArray, Xc: np.ma.MaskedArray) \
            -> Dict[str, np.ndarray]:
        X_list = []
        if Xc is not None:
            Xc = Xc.reshape((Xc.shape[0], -1))
            Xc.data[Xc.mask] = -1
            X_cat_imp = self.cat_imp.transform(Xc.data)
            X_onehot = self.enc.transform(X_cat_imp)
            X_list.append(X_onehot)
        if Xo is not None:
            Xo = Xo.reshape((Xo.shape[0], -1))
            Xo.data[Xo.mask] = np.nan
            X_imputed = self.ord_imp.transform(Xo)
            X_list.append(X_imputed)
        X = np.concatenate(X_list, axis=1)
        Ey = self.est.predict(X)
        # not doing quantiles
        predictions = {"predictions": Ey}
        return predictions
