import pytest

from data import get_dataset


@pytest.mark.parametrize('embed_type', [None, 'codebert', 'word2vec'])
def test_embedding_sanity(embed_type):
    get_dataset(embed_type=embed_type)
