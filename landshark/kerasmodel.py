"""Train/test/predict with keras model."""

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

import logging
import signal
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from timeit import default_timer as timer
from typing import (
    Any,
    Dict,
    Generator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import tensorflow as tf

from landshark.metadata import CategoricalTarget, Target, Training
from landshark.model import QueryConfig, TrainingConfig
from landshark.tfread import dataset_fn

log = logging.getLogger(__name__)
signal.signal(signal.SIGINT, signal.default_int_handler)


@dataclass(frozen=True)
class FeatInput:
    """Feature input data."""

    data: tf.keras.Input
    mask: tf.keras.Input

    @property
    def data_mask(self) -> Tuple[tf.keras.Input, tf.keras.Input]:
        """Return data and mask together."""
        return self.data, self.mask


@dataclass(frozen=True)
class NumFeatInput(FeatInput):
    """Numerical data inputs."""

    pass


@dataclass(frozen=True)
class CatFeatInput(FeatInput):
    """Categorical data inputs."""

    n_classes: int


def _impute_const_fn(x: Sequence[tf.Tensor], value: int = 0) -> tf.Tensor:
    """Set masked elements within tensor to constant `value`."""
    data, mask = x
    tmask = tf.cast(mask, dtype=data.dtype)
    fmask = tf.cast(tf.logical_not(mask), dtype=data.dtype)
    data_imputed = data * fmask + value * tmask
    return data_imputed


def impute_const_layer(
    data: tf.keras.Input, mask: Optional[tf.keras.Input], value: int = 0
) -> tf.keras.layers.Layer:
    """Create an imputation Layer."""
    if mask is not None:
        name = f"impute_{value}_{data.name.split(':')[0]}"
        layer = tf.keras.layers.Lambda(_impute_const_fn, value, name=name)((data, mask))
    else:
        layer = tf.keras.layers.InputLayer(data)
    return layer


def impute_embed_concat_layer(
    num_feats: List[NumFeatInput],
    cat_feats: List[CatFeatInput],
    num_impute_val: int = 0,
    cat_embed_dims: int = 3,
) -> tf.keras.layers.Layer:
    """Impute missing data, embed categorical, and concat inputs together."""

    # inpute with constant value
    num_imputed = [
        impute_const_layer(x.data, x.mask, num_impute_val) for x in num_feats
    ]

    # impute/embed categorical
    def _impute_embed_cat(f: CatFeatInput) -> tf.keras.layers.Layer:
        f_imp = impute_const_layer(f.data, f.mask, f.n_classes)
        name = f"embed_cat_{f.n_classes}_{f_imp.name.split('/')[0]}"
        embedding = tf.keras.layers.Embedding(f.n_classes, cat_embed_dims, name=name)
        f_emb = embedding(tf.squeeze(f_imp, 3))
        return f_emb

    cat_embedded = [_impute_embed_cat(f) for f in cat_feats]

    # concatenate layer
    layer = tf.keras.layers.Concatenate(axis=3)(num_imputed + cat_embedded)
    return layer


def get_feat_input_list(
    num_feats: List[FeatInput], cat_feats: List[FeatInput]
) -> List[tf.keras.Input]:
    """Concatenate feature inputs."""
    i_list = [i for fs in (num_feats, cat_feats) for f in fs for i in f.data_mask]
    return i_list


class KerasInputs(NamedTuple):
    """Inputs required for the keras model configuration function."""

    num_feats: List[NumFeatInput]
    cat_feats: List[CatFeatInput]
    indices: tf.keras.Input
    coords: tf.keras.Input


def gen_keras_inputs(
    dataset: tf.data.TFRecordDataset,
    metadata: Training,
    x_only: bool = False,
) -> KerasInputs:
    """Generate keras.Inputs for each covariate in the dataset."""
    xs = dataset.element_spec if x_only else dataset.element_spec[0]

    def gen_keras_input(data: tf.TensorSpec, name: str) -> tf.keras.Input:
        """TensorSpec to keras input."""
        return tf.keras.Input(name=name, shape=data.shape[1:], dtype=data.dtype)

    def gen_data_mask_inputs(
        data: tf.TensorSpec, mask: tf.TensorSpec, name: str
    ) -> Tuple[tf.keras.Input, tf.keras.Input]:
        """Create keras inputs for data and mask."""
        return gen_keras_input(data, name), gen_keras_input(mask, f"{name}_mask")

    num_feats = []
    if "con" in xs:
        assert "con_mask" in xs
        for k in xs["con"]:
            data, mask = gen_data_mask_inputs(xs["con"][k], xs["con_mask"][k], k)
            num_feats.append(NumFeatInput(data, mask))

    cat_feats = []
    if "cat" in xs:
        assert "cat_mask" in xs and metadata.features.categorical
        for k in xs["cat"]:
            data, mask = gen_data_mask_inputs(xs["cat"][k], xs["cat_mask"][k], k)
            n_classes = metadata.features.categorical.columns[k].mapping.shape[0]
            cat_feats.append(CatFeatInput(data, mask, n_classes))

    feats = KerasInputs(
        num_feats=num_feats,
        cat_feats=cat_feats,
        indices=tf.keras.Input(name="indices", shape=xs["indices"].shape),
        coords=tf.keras.Input(name="coords", shape=xs["coords"].shape),
    )
    return feats


def flatten_dataset_x(x: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten tf dataset dictionary with x data only."""
    return flatten_dataset(x)[0]


def flatten_dataset(
    x: Dict[str, Any], y: Optional[tf.Tensor] = None
) -> Tuple[Dict[str, Any], Optional[Union[tf.Tensor, Tuple[tf.Tensor]]]]:
    """Flatten tf dataset dictionary."""

    def _flat_mask(x_: Dict[str, Any], key: str) -> Dict[str, Any]:
        x_flat = {
            **x_.get(key, {}),
            **{f"{k}_mask": v for k, v in x_.get(f"{key}_mask", {}).items()},
        }
        return x_flat

    x_flat = {**_flat_mask(x, "con"), **_flat_mask(x, "cat")}

    if y is not None and y.shape[1] > 1:
        y = tuple(tf.split(y, [1] * y.shape[1], 1))

    return x_flat, y


class UpdateCallback(tf.keras.callbacks.Callback):
    """Callback for printing loss and training/validation metrics."""

    remove_strs = ("tf_op_layer_",)

    def __init__(self, epochs: int = 1, iterations: Optional[int] = None):
        self.epochs = epochs
        self.total_epochs = epochs * iterations if iterations else None
        self.epoch_count = 0

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Start timer at beginning of each iteration."""
        self.epoch_count += self.epochs
        self.starttime = timer()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Print loss/metrics at the end of each iteration."""
        epoch_str = f"Epoch {self.epoch_count}"
        if self.total_epochs is not None:
            epoch_str += f"/{self.total_epochs}"

        time_str = f"{round(timer() - self.starttime)}s"

        def get_value_str(name: str, val_dict: Dict) -> str:
            name_ = reduce(lambda n, s: "".join(n.split(s)), self.remove_strs, name)
            value_str = f"{name_}: {val_dict[name]:.4f}"
            if f"val_{name}" in val_dict:
                value_str += f" / {val_dict[f'val_{name}']:.4f}"
            return value_str

        if logs is not None:
            metrics = [m for m in logs if not m.startswith("val_") and m != "loss"]
            if "loss" in logs:
                metrics = ["loss"] + metrics
            metrics_str = " - ".join([get_value_str(m, logs) for m in metrics])
        else:
            metrics_str = "No logged data."
        print(" - ".join([epoch_str, time_str, metrics_str]))


class TargetData(NamedTuple):
    """Target data."""

    label: str
    is_categorical: bool
    n_classes: Optional[int] = None


def get_target_data(target: Target) -> List[TargetData]:
    """Create list of target data."""
    if isinstance(target, CategoricalTarget):
        ts = [TargetData(l, True, n) for l, n in zip(target.labels, target.nvalues)]
    else:
        ts = [TargetData(l, False) for l in target.labels]
    return ts


def train_test(
    records_train: List[str],
    records_test: List[str],
    metadata: Training,
    directory: str,
    cf: Any,  # Module type
    params: TrainingConfig,
    iterations: Optional[int],
) -> None:
    """Model training and periodic hold-out testing."""
    xtrain = dataset_fn(
        records_train,
        params.batchsize,
        metadata.features,
        metadata.targets,
        params.epochs,
        shuffle=True,
    )()
    xtest = dataset_fn(
        records_test, params.test_batchsize, metadata.features, metadata.targets
    )()

    inputs = gen_keras_inputs(xtrain, metadata)
    targets = get_target_data(metadata.targets)

    model = cf.model(*inputs, targets)

    weights_file = Path(directory) / "checkpoint_weights.h5"
    if weights_file.exists():
        model.load_weights(str(weights_file))

    xtrain = xtrain.map(flatten_dataset)
    xtest = xtest.map(flatten_dataset)

    # create callbacks for tensorboard, model saving, and early stopping
    callbacks = [
        tf.keras.callbacks.TensorBoard(directory),
        tf.keras.callbacks.ModelCheckpoint(str(weights_file), save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=50),
        UpdateCallback(params.epochs, iterations),
    ]

    try:
        while True:
            model.fit(
                x=xtrain,
                epochs=iterations or 1_000_000,
                verbose=0,
                callbacks=callbacks,
                validation_data=xtest,
                shuffle=True,
                validation_freq=1,
                use_multiprocessing=False,
            )
            if iterations is not None:
                break

    except KeyboardInterrupt:
        print("Training interrupted.")

    return


def predict(
    checkpoint_dir: str,
    cf: Any,  # Module type
    metadata: Training,
    records: List[str],
    params: QueryConfig,
) -> Generator:
    """Load a model and predict results for record inputs."""
    x = dataset_fn(records, params.batchsize, metadata.features)()
    inputs = gen_keras_inputs(x, metadata, x_only=True)
    targets = get_target_data(metadata.targets)
    x = x.map(flatten_dataset_x)

    model = cf.model(*inputs, targets)

    weights_file = Path(checkpoint_dir) / "checkpoint_weights.h5"
    if weights_file.exists():
        model.load_weights(str(weights_file))

    for x_it in x:
        y_it = model.predict(
            x=x_it,
            batch_size=None,
            verbose=0,
            steps=1,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )
        if not isinstance(y_it, list):
            y_it = [y_it]

        predictions = dict(
            p for p in zip(model.output_names, y_it) if p[0].startswith("predictions")
        )
        yield predictions
