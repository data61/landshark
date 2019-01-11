"""Model config file."""
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

NTREES = 100


class SKModel:
    def __init__(self, metadata, random_seed) -> None:
        self.con_imp = SimpleImputer(strategy="mean",
                               verbose=0, copy=True)
        self.cat_imp = SimpleImputer(missing_values=-1,
                               strategy="most_frequent",
                               verbose=0, copy=True)
        psize = (2 * metadata.features.halfwidth + 1)**2
        self.label = metadata.targets.labels[0]
        if metadata.features.categorical:
            n_values = np.array([k.nvalues.flatten()[0] for k in \
                        metadata.features.categorical.columns.values()])
            self.enc = OneHotEncoder(categories=[range(k) for k in n_values],
                                     dtype=np.float32, sparse=False)

        self.est = RandomForestClassifier(n_estimators=NTREES,
                                          random_state=random_seed)

    def train(self, X_con, X_cat, indices, coords, Y: np.array) -> None:

        # single task classification
        Y = Y[:, 0]

        X_list = []
        if X_cat is not None:
            X_cat = np.ma.concatenate(list(X_cat.values()), axis=1)
            X_cat = X_cat.reshape((X_cat.shape[0], -1))
            X_cat.data[X_cat.mask] = -1
            X_cat_imp = self.cat_imp.fit_transform(X_cat.data)
            X_onehot = self.enc.fit_transform(X_cat_imp)
            X_list.append(X_onehot)
        if X_con is not None:
            X_con = np.ma.concatenate(list(X_con.values()), axis=1)
            X_con = X_con.reshape((X_con.shape[0], -1))
            X_con.data[X_con.mask] = np.nan
            X_imputed = self.con_imp.fit_transform(X_con.data)
            X_list.append(X_imputed)
        X = np.concatenate(X_list, axis=1)
        self.est.fit(X, Y)

    def test(self, Y: np.array, predictions: Dict[str, np.array]) \
            -> Dict[str, np.ndarray]:
        Y = Y[:, 0]
        acc = accuracy_score(Y, predictions["predictions_" + self.label])
        return {"accuracy": acc}


    def predict(self, X_con, X_cat, indices, coords) \
            -> Dict[str, np.ndarray]:
        X_list = []
        if X_cat is not None:
            X_cat = np.ma.concatenate(list(X_cat.values()), axis=1)
            X_cat = X_cat.reshape((X_cat.shape[0], -1))
            X_cat.data[X_cat.mask] = -1
            X_cat_imp = self.cat_imp.transform(X_cat.data)
            X_onehot = self.enc.transform(X_cat_imp)
            X_list.append(X_onehot)
        if X_con is not None:
            X_con = np.ma.concatenate(list(X_con.values()), axis=1)
            X_con = X_con.reshape((X_con.shape[0], -1))
            X_con.data[X_con.mask] = np.nan
            X_imputed = self.con_imp.transform(X_con)
            X_list.append(X_imputed)
        X = np.concatenate(X_list, axis=1)
        Ey = self.est.predict(X)
        predictions = {'predictions_' + self.label: Ey}
        return predictions
