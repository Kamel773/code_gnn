import abc
import os

import gensim
import networkx as nx
import torch

from codebert.api import CodeBERTAPI


class EmbeddingGetter(abc.ABC):
    @abc.abstractmethod
    def get_embedding(self, file, cpg):
        raise NotImplementedError()


class CodeBERTEmbeddingGetter(EmbeddingGetter):
    def __init__(self):
        self.codebert_api = CodeBERTAPI()

    def get_embedding(self, file, cpg):
        # TODO: standardize the text whitespace so that there is just one space in between each token.
        #  Do the same for cpg so that the offsets match.
        with open(file) as f:
            text = f.read()
        examples = [
            {
                "func": text,
                "idx": 0,
                "target": 0,
            }
        ]
        token_embeddings, features = self.codebert_api.get_token_embeddings(examples, return_features=True)
        # Map cpg nodes to features given the offsets in features[0].encoded.offsets
        offsets = features[0].encoded.offsets
        node_location = nx.get_node_attributes(cpg, 'location')
        node_code = nx.get_node_attributes(cpg, 'code')
        node_type = nx.get_node_attributes(cpg, 'type')
        node_tokens = []
        node_embeddings = []
        for e in cpg.nodes:
            token_text = None
            embed = None
            loc = node_location[e]
            if loc:
                _, _, n_begin, n_end = map(int, loc.split(':'))
                absolute_t_begin_i = None
                absolute_t_end_i = None
                for t_i, (t_begin, t_end) in enumerate(offsets):
                    if n_begin >= t_begin:
                        absolute_t_begin_i = t_i
                    if n_end <= t_end:
                        absolute_t_end_i = t_i
                        break
                if absolute_t_begin_i is None or absolute_t_end_i is None:
                    continue
                tokens = features[0].encoded.tokens[absolute_t_begin_i:absolute_t_end_i + 1]
                token_text = ' '.join(tokens)
                embeddings = token_embeddings[absolute_t_begin_i:absolute_t_end_i + 1]
                embed = torch.mean(torch.stack(embeddings, dim=0), dim=0).squeeze(dim=0)
            node_tokens.append(token_text)
            node_embeddings.append(embed)

        # Debug outputs
        print('Mismatches:')
        for i, token_text in enumerate(node_tokens):
            if token_text is None:
                continue
            elif token_text.replace('Ġ', '') != node_code[i]:
                print(i, token_text, node_code[i], node_type[i], node_location[i], sep='|')

        print(len([n for n in node_embeddings if n is not None]), 'embeddings out of', len(cpg.nodes), 'nodes:')
        for i, (t, e) in enumerate(zip(node_tokens, node_embeddings)):
            if i in node_code:
                code = node_code[i]
            else:
                code = '<no code>'
            if i in node_location:
                loc = node_location[i]
            else:
                loc = '<no location>'
            if e is not None:
                embed_shape = e.shape
            else:
                embed_shape = '<no embedding>'
            print(i, code, loc, t, embed_shape, sep='|')
        print("graphviz:")
        nx.set_node_attributes(cpg, {i: 1 if n is not None else 2 for i, n in zip(cpg.nodes, node_embeddings)}, "color")
        nx.set_node_attributes(cpg, 'pastel28', "colorscheme")
        nx.set_node_attributes(cpg, 'filled', "style")
        graphviz_cpg = nx.nx_pydot.to_pydot(cpg)
        print(graphviz_cpg)

        default_init = torch.zeros
        node_embeddings = [n if n is not None else default_init(self.codebert_api.num_features) for n in
                           node_embeddings]
        embedding = torch.stack(node_embeddings, dim=0)

        return embedding


class Word2VecEmbeddingGetter(EmbeddingGetter):
    def __init__(self):
        corpus_pretrained = 'word2vec/devign.wv'  # TODO: initialize this for the dataset we're actually training on
        assert os.path.exists(corpus_pretrained), f"Train Word2Vec and save to {corpus_pretrained}! You're on your own bud."
        self.wv_model = gensim.models.Word2Vec.load(corpus_pretrained)

    def get_embedding(self, file, cpg):
        node_code = nx.get_node_attributes(cpg, 'code')

        def get_node_embedding(node):
            code = node_code[node]
            tokens = code.strip().split()
            token_embeddings = [torch.tensor(self.wv_model.wv[tok]) for tok in tokens if tok in self.wv_model.wv]
            if len(token_embeddings) == 0:
                node_embedding = torch.zeros(self.wv_model.vector_size)
            else:
                node_embedding = torch.stack(token_embeddings, dim=0).mean(dim=0)
            return node_embedding

        node_embeddings = [get_node_embedding(node) for node in cpg.nodes]
        embedding = torch.stack(node_embeddings, dim=0)

        return embedding


class RandomEmbeddingGetter(EmbeddingGetter):
    def get_embedding(self, file, cpg):
        return torch.randn((len(cpg.nodes), 128))