"""Generic regression config file."""
import tensorflow as tf
import aboleth as ab

from landshark.model import patch_slices

ab.set_hyperseed(666)
embed_dim = 3


def model(Xo, Xom, Xc, Xcm, Y, samples, metadata):
    slices = patch_slices(metadata)
    target_ncats = len(metadata.target_map[0])
    # Categorical features
    embed_layers = [ab.EmbedMAP(embed_dim, k, l1_reg=1e-5, l2_reg=0.)
                    for k in metadata.ncategories]

    cat_net = (
        ab.InputLayer(name="Xc", n_samples=samples) >>
        ab.PerFeature(*embed_layers, slices=slices)
        )

    # Continuous features
    data_input = ab.InputLayer(name="Xo", n_samples=samples)  # Data input
    mask_input = ab.MaskInputLayer(name="Mo")  # Missing data mask input

    con_net = ab.LearnedScalarImpute(data_input, mask_input)

    # Combined net
    net = (
        ab.Concat(con_net, cat_net) >>
        ab.Activation(tf.nn.elu) >>
        ab.DenseMAP(output_dim=target_ncats, l1_reg=0., l2_reg=1e-5)
        )

    F, reg = net(Xo=Xo, Mo=Xom, Xc=Xc)
    lkhood = tf.distributions.Categorical(logits=F)
    loss = ab.max_posterior(lkhood, Y[:, 0], reg)
    return F, lkhood, loss, Y
