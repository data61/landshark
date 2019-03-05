"""Model config file."""

# Copyright 2019 CSIRO (Data61)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from landshark.metadata import Training

NTREES = 100


class SKModel:
    def __init__(self, metadata: Training, random_seed: int) -> None:
        self.con_imp = SimpleImputer(strategy="mean", verbose=0, copy=True)
        self.cat_imp = SimpleImputer(missing_values=-1,
                                     strategy="most_frequent",
                                     verbose=0, copy=True)
        self.label = metadata.targets.labels[0]
        if metadata.features.categorical:
            n_values = np.array([
                k.nvalues.flatten()[0] for k in
                metadata.features.categorical.columns.values()
            ])
            self.enc = OneHotEncoder(categories=[range(k) for k in n_values],
                                     dtype=np.float32, sparse=False)

        self.est = RandomForestClassifier(n_estimators=NTREES,
                                          random_state=random_seed)

    def train(self,
              X_con: Optional[Dict[str, np.ma.MaskedArray]],
              X_cat: Optional[Dict[str, np.ma.MaskedArray]],
              indices: np.ndarray,
              coords: np.ndarray,
              Y: np.ndarray
              ) -> None:
        # single task classification
        Y = Y[:, 0]

        X_list = []
        if X_cat is not None:
            X_cat_m = np.ma.concatenate(list(X_cat.values()), axis=1)
            X_cat_m = X_cat_m.reshape((X_cat_m.shape[0], -1))
            X_cat_m.data[X_cat_m.mask] = -1
            X_cat_imp = self.cat_imp.fit_transform(X_cat_m.data)
            X_onehot = self.enc.fit_transform(X_cat_imp)
            X_list.append(X_onehot)
        if X_con is not None:
            X_con_m = np.ma.concatenate(list(X_con.values()), axis=1)
            X_con_m = X_con_m.reshape((X_con_m.shape[0], -1))
            X_con_m.data[X_con_m.mask] = np.nan
            X_imputed = self.con_imp.fit_transform(X_con_m.data)
            X_list.append(X_imputed)
        X = np.concatenate(X_list, axis=1)
        self.est.fit(X, Y)

    def test(self, Y: np.array,
             predictions: Dict[str, np.ndarray]
             ) -> Dict[str, np.ndarray]:
        Y = Y[:, 0]
        acc = accuracy_score(Y, predictions["predictions_" + self.label])
        return {"accuracy": acc}

    def predict(self,
                X_con: Optional[Dict[str, np.ma.MaskedArray]],
                X_cat: Optional[Dict[str, np.ma.MaskedArray]],
                indices: np.ndarray,
                coords: np.ndarray,
                ) -> Dict[str, np.ndarray]:
        X_list = []
        if X_cat is not None:
            X_cat_m = np.ma.concatenate(list(X_cat.values()), axis=1)
            X_cat_m = X_cat_m.reshape((X_cat_m.shape[0], -1))
            X_cat_m.data[X_cat_m.mask] = -1
            X_cat_imp = self.cat_imp.transform(X_cat_m.data)
            X_onehot = self.enc.transform(X_cat_imp)
            X_list.append(X_onehot)
        if X_con is not None:
            X_con_m = np.ma.concatenate(list(X_con.values()), axis=1)
            X_con_m = X_con_m.reshape((X_con_m.shape[0], -1))
            X_con_m.data[X_con_m.mask] = np.nan
            X_con_imp = self.con_imp.transform(X_con_m)
            X_list.append(X_con_imp)
        X = np.concatenate(X_list, axis=1)
        Ey = self.est.predict(X)
        predictions = {"predictions_" + self.label: Ey}
        return predictions
