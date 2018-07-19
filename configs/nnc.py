"""Generic classification config file."""
import aboleth as ab
import tensorflow as tf

from landshark.model import patch_categories, patch_slices

ab.set_hyperseed(666)
embed_dim = 3


def model(Xo, Xom, Xc, Xcm, Y, samples, metadata):
    target_ncats = metadata.targets.ncategories[0]
    arg_dict = {}
    layer_list = []
    # Categorical features
    if Xc.shape[1] != 0:

        input_layer = ab.ExtraCategoryImpute(
            ab.InputLayer(name="Xc", n_samples=samples),
            ab.MaskInputLayer(name="Xcm"), patch_categories(metadata))

        # Note the +1 because of the extra category imputation
        embed_layers = [ab.EmbedMAP(embed_dim, k + 1, l1_reg=1e-5, l2_reg=0.)
                        for k in metadata.features.categorical.ncategories]


        slices = patch_slices(metadata)
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
        ab.DenseMAP(output_dim=target_ncats, l1_reg=0., l2_reg=1e-5)
        )

    F, reg = net(**arg_dict)
    lkhood = tf.distributions.Categorical(logits=F)
    loss = ab.max_posterior(lkhood.log_prob(Y[:, 0]), reg)
    return F, lkhood, loss, Y
