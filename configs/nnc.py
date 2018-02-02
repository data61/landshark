"""Generic regression config file."""
import tensorflow as tf
import aboleth as ab

from landshark.model import patch_slices

ab.set_hyperseed(666)
embed_dim = 3


def model(Xo, Xom, Xc, Xcm, Y, samples, metadata):
    slices = patch_slices(metadata)
    target_ncats = len(metadata.target_map[0])
    arg_dict = {}
    layer_list = []
    # Categorical features
    if Xc.shape[1] != 0:
        embed_layers = [ab.EmbedMAP(embed_dim, k, l1_reg=1e-5, l2_reg=0.)
                        for k in metadata.ncategories]
        cat_net = (
            ab.InputLayer(name="Xc", n_samples=samples) >>
            ab.PerFeature(*embed_layers, slices=slices)
            )
        layer_list.append(cat_net)
        arg_dict["Xc"] = Xc

    # Continuous features
    if Xo.shape[1] != 0:
        data_input = ab.InputLayer(name="Xo", n_samples=samples)  # Data input
        mask_input = ab.MaskInputLayer(name="Mo")  # Missing data mask input
        con_net = ab.LearnedScalarImpute(data_input, mask_input)
        layer_list.append(con_net)
        arg_dict["Xo"] = Xo
        arg_dict["Mo"] = Xom

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
        ab.DenseMAP(output_dim=target_ncats, l1_reg=0., l2_reg=1e-5)
        )

    F, reg = net(**arg_dict)
    lkhood = tf.distributions.Categorical(logits=F)
    loss = ab.max_posterior(lkhood, Y[:, 0], reg)
    return F, lkhood, loss, Y

