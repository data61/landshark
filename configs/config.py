"""Model config file."""
from typing import Tuple
import numpy as np
import tensorflow as tf
import aboleth as ab

from landshark.model import patch_slices
from landshark.importers.metadata import TrainingMetadata

sess_config = tf.ConfigProto(device_count={"GPU": 1})


def model(Xo: tf.Tensor, Xom: tf.Tensor, Xc: tf.Tensor, Xcm: tf.Tensor,
          Y: tf.Tensor, metadata: TrainingMetadata) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    ab.set_hyperseed(666)
    nsamps = 1  # Number of posterior samples
    noise = tf.Variable(1.0 * np.ones(metadata.ntargets, dtype=np.float32))
    slices = patch_slices(metadata)

    # Categorical features
    embed_layers = [ab.EmbedMAP(3, k, l1_reg=1e-3, l2_reg=0.)
                    for k in metadata.ncategories]

    cat_net = (
        ab.InputLayer(name="Xc", n_samples=nsamps) >>
        ab.PerFeature(*embed_layers, slices=slices)
        )

    # Continuous features
    data_input = ab.InputLayer(name="Xo", n_samples=nsamps)  # Data input
    mask_input = ab.MaskInputLayer(name="Mo")  # Missing data mask input

    con_net = (
        ab.LearnedScalarImpute(data_input, mask_input) >>
        ab.DenseMAP(output_dim=200, l1_reg=0., l2_reg=1e-3)
        )

    # Combined net
    net = (
        ab.Concat(con_net, cat_net) >>
        ab.Activation(tf.nn.relu) >>
        ab.DenseMAP(output_dim=100, l1_reg=0., l2_reg=1e-3) >>
        ab.Activation(tf.nn.relu) >>
        ab.DenseMAP(output_dim=20, l1_reg=0., l2_reg=1e-3) >>
        ab.Activation(tf.nn.relu) >>
        ab.DenseMAP(output_dim=3, l1_reg=0., l2_reg=1e-3)
        )

    phi, reg = net(Xo=Xo, Mo=Xom, Xc=Xc)
    lkhood = tf.distributions.StudentT(df=5., loc=phi, scale=ab.pos(noise))
    loss = ab.max_posterior(lkhood, Y, reg)

    return phi, lkhood, loss
