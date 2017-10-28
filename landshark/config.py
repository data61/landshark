"""Model config file."""
from typing import Tuple
import tensorflow as tf
import aboleth as ab

from landshark.importers.metadata import TrainingMetadata

batch_size = 10  # Learning batch size
psamps = 30  # Number of times to samples the network for prediction
epochs = 20  # epochs between tests

ab.set_hyperseed(666)
train_config = tf.ConfigProto(device_count={"GPU": 1})
predict_config = tf.ConfigProto(device_count={"GPU": 1})


def model(Xo: tf.Tensor, Xom: tf.Tensor, Xc: tf.Tensor, Xcm: tf.Tensor,
          Y: tf.Tensor, metadata: TrainingMetadata) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    nsamps = 3  # Number of posterior samples
    ls = 10.
    lenscale = tf.Variable(ls)
    noise = tf.Variable(1.0)
    slices = _patch_slices(metadata)

    # Categorical features
    embed_layers = [ab.EmbedVariational(3, k) for k in metadata.ncategories]

    cat_net = (
        ab.InputLayer(name="Xc", n_samples=nsamps) >>
        ab.PerFeature(*embed_layers, slices=slices) >>
        ab.Activation(tf.tanh)
        # ab.Activation(tf.nn.relu) >>
        # ab.DenseVariational(output_dim=10, std=1., full=True) >>
        # ab.Activation(tf.nn.relu)
        )

    # Continuous features
    kern = ab.RBF(lenscale=ab.pos(lenscale))
    # kern = ab.RBFVariational(lenscale=ab.pos(lenscale), lenscale_posterior=ls)

    data_input = ab.InputLayer(name="Xo", n_samples=nsamps)  # Data input
    mask_input = ab.MaskInputLayer(name="Mo")  # Missing data mask input

    con_net = (
        ab.LearnedScalarImpute(data_input, mask_input) >>
        ab.RandomFourier(n_features=50, kernel=kern)
        )

    # Combined net
    net = (
        ab.Concat(con_net, cat_net) >>
        ab.DenseVariational(output_dim=1, std=1., full=True)
        )

    phi, kl = net(Xo=Xo, Mo=Xom, Xc=Xc)
    lkhood = tf.distributions.StudentT(df=5., loc=phi, scale=ab.pos(noise))
    loss = ab.elbo(lkhood, Y, metadata.N, kl)

    return phi, lkhood, loss


def _patch_slices(metadata):
    npatch = (metadata.halfwidth * 2 + 1) ** 2
    dim = npatch * metadata.x_cat
    begin = range(0, dim, npatch)
    end = range(npatch, dim + npatch, npatch)
    slices = [slice(b, e) for b, e in zip(begin, end)]
    return slices
