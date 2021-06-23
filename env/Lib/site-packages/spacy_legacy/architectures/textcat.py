from typing import Optional
from thinc.api import Model
from spacy.attrs import ID, ORTH, PREFIX, SUFFIX, SHAPE, LOWER
from spacy.util import registry


def TextCatEnsemble_v1(
    width: int,
    embed_size: int,
    pretrained_vectors: Optional[bool],
    exclusive_classes: bool,
    ngram_size: int,
    window_size: int,
    conv_depth: int,
    dropout: Optional[float],
    nO: Optional[int] = None,
) -> Model:
    # Don't document this yet, I'm not sure it's right.
    HashEmbed = registry.get("layers", "HashEmbed.v1")
    FeatureExtractor = registry.get("layers", "spacy.FeatureExtractor.v1")
    Maxout = registry.get("layers", "Maxout.v1")
    StaticVectors = registry.get("layers", "spacy.StaticVectors.v1")
    Softmax = registry.get("layers", "Softmax.v1")
    Linear = registry.get("layers", "Linear.v1")
    ParametricAttention = registry.get("layers", "ParametricAttention.v1")
    Dropout = registry.get("layers", "Dropout.v1")
    Logistic = registry.get("layers", "Logistic.v1")
    build_bow_text_classifier = registry.get("architectures", "spacy.TextCatBOW.v1")
    list2ragged = registry.get("layers", "list2ragged.v1")
    chain = registry.get("layers", "chain.v1")
    concatenate = registry.get("layers", "concatenate.v1")
    clone = registry.get("layers", "clone.v1")
    reduce_sum = registry.get("layers", "reduce_sum.v1")
    with_array = registry.get("layers", "with_array.v1")
    uniqued = registry.get("layers", "uniqued.v1")
    residual = registry.get("layers", "residual.v1")
    expand_window = registry.get("layers", "expand_window.v1")

    cols = [ORTH, LOWER, PREFIX, SUFFIX, SHAPE, ID]
    with Model.define_operators({">>": chain, "|": concatenate, "**": clone}):
        lower = HashEmbed(
            nO=width, nV=embed_size, column=cols.index(LOWER), dropout=dropout, seed=10
        )
        prefix = HashEmbed(
            nO=width // 2,
            nV=embed_size,
            column=cols.index(PREFIX),
            dropout=dropout,
            seed=11,
        )
        suffix = HashEmbed(
            nO=width // 2,
            nV=embed_size,
            column=cols.index(SUFFIX),
            dropout=dropout,
            seed=12,
        )
        shape = HashEmbed(
            nO=width // 2,
            nV=embed_size,
            column=cols.index(SHAPE),
            dropout=dropout,
            seed=13,
        )
        width_nI = sum(layer.get_dim("nO") for layer in [lower, prefix, suffix, shape])
        trained_vectors = FeatureExtractor(cols) >> with_array(
            uniqued(
                (lower | prefix | suffix | shape)
                >> Maxout(nO=width, nI=width_nI, normalize=True),
                column=cols.index(ORTH),
            )
        )
        if pretrained_vectors:
            static_vectors = StaticVectors(width)
            vector_layer = trained_vectors | static_vectors
            vectors_width = width * 2
        else:
            vector_layer = trained_vectors
            vectors_width = width
        tok2vec = vector_layer >> with_array(
            Maxout(width, vectors_width, normalize=True)
            >> residual(
                (
                    expand_window(window_size=window_size)
                    >> Maxout(
                        nO=width, nI=width * ((window_size * 2) + 1), normalize=True
                    )
                )
            )
            ** conv_depth,
            pad=conv_depth,
        )
        cnn_model = (
            tok2vec
            >> list2ragged()
            >> ParametricAttention(width)
            >> reduce_sum()
            >> residual(Maxout(nO=width, nI=width))
            >> Linear(nO=nO, nI=width)
            >> Dropout(0.0)
        )

        linear_model = build_bow_text_classifier(
            nO=nO,
            ngram_size=ngram_size,
            exclusive_classes=exclusive_classes,
            no_output_layer=False,
        )
        nO_double = nO * 2 if nO else None
        if exclusive_classes:
            output_layer = Softmax(nO=nO, nI=nO_double)
        else:
            output_layer = Linear(nO=nO, nI=nO_double) >> Dropout(0.0) >> Logistic()
        model = (linear_model | cnn_model) >> output_layer
        model.set_ref("tok2vec", tok2vec)
    if model.has_dim("nO") is not False:
        model.set_dim("nO", nO)
    model.set_ref("output_layer", linear_model.get_ref("output_layer"))
    model.attrs["multi_label"] = not exclusive_classes
    return model
