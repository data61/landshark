"""Generic regression config file."""
import numpy as np
import tensorflow as tf
import aboleth as ab

from landshark.model import patch_slices

ab.set_hyperseed(666)
noise0 = 0.1
embed_dim = 3


def model(Xo, Xom, Xc, Xcm, Y, samples, metadata):
    noise = tf.Variable(noise0 * np.ones(metadata.ntargets, dtype=np.float32))
    slices = patch_slices(metadata)

    arg_dict = {}
    layer_list = []
    # Categorical features
    if Xc.shape[1] != 0:

        input_layer = ab.ExtraCategoryImpute(
            ab.InputLayer(name="Xc", n_samples=samples),
            ab.MaskInputLayer(name="Xcm"), metadata.ncategories_patched)

        # Note the +1 because of the extra category imputation
        embed_layers = [ab.EmbedMAP(embed_dim, k + 1, l1_reg=1e-5, l2_reg=0.)
                        for k in metadata.ncategories]

        cat_net = input_layer >> ab.PerFeature(*embed_layers, slices=slices)
        layer_list.append(cat_net)
        arg_dict["Xc"] = Xc
        arg_dict["Xcm"] = Xcm

    # Continuous features
    if Xo.shape[1] != 0:
        data_input = ab.InputLayer(name="Xo", n_samples=samples)  # Data input
        mask_input = ab.MaskInputLayer(name="Xom")  # Missing data mask input
        con_net = ab.LearnedScalarImpute(data_input, mask_input)
        layer_list.append(con_net)
        arg_dict["Xo"] = Xo
        arg_dict["Xom"] = Xom

    if len(layer_list) == 2:
        input_layer = ab.Concat(*layer_list)
    elif len(layer_list) == 1:
        input_layer = layer_list[0]
    else:
        raise ValueError("Model has no ordinal or categorical inputs.")

    # Combined net
    net = (
        input_layer >>
        ab.Activation(tf.nn.elu) >>
        ab.DenseMAP(output_dim=1, l1_reg=0., l2_reg=1e-5)
        )

    F, reg = net(**arg_dict)
    lkhood = tf.distributions.StudentT(df=5., loc=F, scale=ab.pos(noise))
    loss = ab.max_posterior(lkhood.log_prob(Y), reg)

    return F, lkhood, loss, Y
