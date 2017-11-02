"""Model config file."""
import numpy as np
import tensorflow as tf
import aboleth as ab

from landshark.model import patch_slices

sess_config = tf.ConfigProto(device_count={"GPU": 1})

ab.set_hyperseed(666)
nsamps = 1  # Number of posterior samples
noise0 = 1.
embed_dim = 3


def model(Xo, Xom, Xc, Xcm, Y, metadata):
    noise = tf.Variable(noise0 * np.ones(metadata.ntargets, dtype=np.float32))
    slices = patch_slices(metadata)

    # Categorical features
    embed_layers = [ab.EmbedMAP(embed_dim, k, l1_reg=1e-3, l2_reg=0.)
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
        ab.DenseMAP(output_dim=20, l1_reg=0., l2_reg=1e-5) >>
        ab.Activation(tf.nn.relu) >>
        ab.DenseMAP(output_dim=3, l1_reg=0., l2_reg=1e-7)
        )

    F, reg = net(Xo=Xo, Mo=Xom, Xc=Xc)
    lkhood = tf.distributions.StudentT(df=5., loc=F, scale=ab.pos(noise))
    loss = ab.max_posterior(lkhood, Y, reg)

    return F, lkhood, loss
