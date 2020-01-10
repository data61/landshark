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

from landshark.metadata import Training
from landshark.model import (
    QueryConfig,
    TrainingConfig,
    predict_data,
    test_data,
    train_data,
)

log = logging.getLogger(__name__)
signal.signal(signal.SIGINT, signal.default_int_handler)


class FeatInput(NamedTuple):
    """Paired Data/Mask inputs."""

    data: tf.keras.Input
    mask: Optional[tf.keras.Input]


def _impute_const_fn(x: Sequence[tf.Tensor], value: int = 0) -> tf.Tensor:
    """Set masked elements within tensor to constant `value`."""
    data, mask = x
    tmask = tf.cast(mask, dtype=data.dtype)
    fmask = tf.cast(tf.logical_not(mask), dtype=data.dtype)
    data_imputed = data * fmask + value * tmask
    return data_imputed


def impute_const_layer(feat: FeatInput, value: int = 0) -> tf.keras.layers.Layer:
    """Create an imputation Layer."""
    if feat.mask is not None:
        layer = tf.keras.layers.Lambda(_impute_const_fn, value)(feat)
    else:
        layer = tf.keras.layers.InputLayer(feat.data)
    return layer


def get_feat_input_list(
    num_feats: List[FeatInput], cat_feats: List[Tuple[FeatInput, int]]
) -> List[tf.keras.Input]:
    """Concatenate feature inputs."""
    num = [x for f in num_feats for x in f]
    cat = [x for f, _ in cat_feats for x in f if x is not None]
    return num + cat


class KerasInputs(NamedTuple):
    """Inputs required for the keras model configuration function."""

    num_feats: List[FeatInput]
    cat_feats: List[Tuple[FeatInput, int]]
    indices: tf.keras.Input
    coords: tf.keras.Input


def gen_keras_inputs(
    dataset: tf.data.TFRecordDataset, metadata: Training, x_only: bool = False,
) -> KerasInputs:
    """Generate keras.Inputs for each covariate in the dataset."""
    xs = dataset.element_spec if x_only else dataset.element_spec[0]

    def gen_feat_input(
        data: tf.TensorSpec, mask: tf.TensorSpec, name: str
    ) -> FeatInput:
        feat = FeatInput(
            data=tf.keras.Input(name=name, shape=data.shape[1:], dtype=data.dtype),
            mask=tf.keras.Input(
                name=f"{name}_mask", shape=mask.shape[1:], dtype=mask.dtype
            ),
        )
        return feat

    num_feats = []
    if "con" in xs:
        assert "con_mask" in xs
        for k in xs["con"]:
            num_feats.append(gen_feat_input(xs["con"][k], xs["con_mask"][k], k))

    cat_feats = []
    if "cat" in xs:
        assert "cat_mask" in xs and metadata.features.categorical
        for k in xs["cat"]:
            cat_feats.append(
                (
                    gen_feat_input(xs["cat"][k], xs["cat_mask"][k], k),
                    metadata.features.categorical.columns[k].mapping.shape[0],
                )
            )

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
    xtrain = train_data(records_train, metadata, params.batchsize, params.epochs)()
    xtest = test_data(records_test, metadata, params.test_batchsize)()
    inputs = gen_keras_inputs(xtrain, metadata)

    model = cf.model(*inputs, metadata.targets.labels)

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
    x = predict_data(records, metadata, params.batchsize)()
    inputs = gen_keras_inputs(x, metadata, x_only=True)
    x = x.map(flatten_dataset_x)

    model = cf.model(*inputs, metadata.targets.labels)

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
        if not isinstance(y_it, tuple):
            y_it = (y_it,)

        yield dict(zip(model.output_names, y_it))
