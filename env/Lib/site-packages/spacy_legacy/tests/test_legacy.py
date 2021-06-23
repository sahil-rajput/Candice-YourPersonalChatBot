import pytest
from spacy import registry

PACKAGES = ["spacy", "spacy-legacy"]
FUNCTIONS = [
    ("architectures", "Tok2Vec.v1"),
    ("architectures", "MaxoutWindowEncoder.v1"),
    ("architectures", "MishWindowEncoder.v1"),
    ("architectures", "TextCatEnsemble.v1"),
    ("architectures", "HashEmbedCNN.v1"),
    ("architectures", "MultiHashEmbed.v1"),
    ("architectures", "CharacterEmbed.v1"),
    ("loggers", "WandbLogger.v1"),
    ("layers", "StaticVectors.v1")
]

@pytest.mark.parametrize("package", PACKAGES)
@pytest.mark.parametrize("reg_name,name", FUNCTIONS)
def test_registry(package, reg_name, name):
    func_name = f"{package}.{name}"
    assert registry.has(reg_name, func_name)
    assert registry.get(reg_name, func_name)
