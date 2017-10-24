from typing import Tuple
import tensorflow as tf
import aboleth as ab

from landshark.export import RecordShape

batch_size = 10  # Learning batch size
psamps = 20  # Number of times to samples the network for prediction
epochs = 20  # epochs between tests

ab.set_hyperseed(666)
train_config = tf.ConfigProto(device_count={"GPU": 1})
predict_config = tf.ConfigProto(device_count={"GPU": 1})


def model(Xo: tf.Tensor, Xom: tf.Tensor, Xc: tf.Tensor, Xcm: tf.Tensor,
          Y: tf.Tensor, metadata: RecordShape) \
        -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    nsamps = 5  # Number of posterior samples
    ls = 10.
    lenscale = tf.Variable(ls)
    noise = tf.Variable(1.0)
    ncats = _patch_ncategories(metadata)

    # Categorical features
    # FIXME we're going to have to rethink this... we've patched indices...
    # data_input = ab.InputLayer(name="Xc", n_samples=nsamps)  # Data input
    # mask_input = ab.MaskInputLayer(name="Mc")  # Missing data mask input

    # cat_net = (
    #     ab.MeanImpute(data_input, mask_input) >>
    #     ab.DenseVariational(output_dim=20, std=1., full=False) >>
    #     ab.Activation(tf.tanh)
    #     )

    # Continuous features
    # kern = ab.RBF(lenscale=ab.pos(lenscale))
    kern = ab.RBFVariational(lenscale=ab.pos(lenscale), lenscale_posterior=ls)

    data_input = ab.InputLayer(name="Xo", n_samples=nsamps)  # Data input
    mask_input = ab.MaskInputLayer(name="Mo")  # Missing data mask input

    con_net = (
        ab.LearnedScalarImpute(data_input, mask_input) >>
        ab.RandomFourier(n_features=50, kernel=kern)
        )

    # Combined net
    net = (
        # ab.Concat(con_net, cat_net) >>
        con_net >>
        ab.DenseVariational(output_dim=1, std=1., full=True)
        )

    phi, kl = net(Xo=Xo, Mo=Xom)  # , Xc=Xc, Mc=Xcm)
    lkhood = tf.distributions.StudentT(df=5., loc=phi, scale=ab.pos(noise))
    loss = ab.elbo(lkhood, Y, metadata.N, kl)

    return phi, lkhood, loss


def _patch_ncategories(metadata):
    npatch = (metadata.halfwidth * 2 + 1) ** 2
    ncats = [n for n in metadata.ncategories for _ in range(npatch)]
    return ncats


def _hash_categories(Xc, ncats):
    pass
