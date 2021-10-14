import torch

from codebert.run import parse_args, get_model, convert_examples_to_features


class CodeBERTAPI:
    def __init__(self):
        raw_args = """--output_dir=./saved_models 
--model_type=roberta --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base 
--train_data_file=../CodeXGLUE-Defect-detection/dataset/train.jsonl --eval_data_file=../CodeXGLUE-Defect-detection/dataset/valid.jsonl --test_data_file=../CodeXGLUE-Defect-detection/dataset/test.jsonl 
--epoch 5 --train_batch_size 32 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0
--seed 123456""".split()
        no_cuda = False  # TODO: This is a dev/test setting because CUDA doesn't work in WSL
        if no_cuda:
            raw_args.append('--no_cuda')
        self.args = parse_args(raw_args)
        self.model, self.tokenizer = get_model(self.args)

    @property
    def num_features(self):
        return self.model.config.hidden_size

    def get_token_embeddings(self, examples, return_features=False):
        features = []
        for example in examples:
            feature = convert_examples_to_features(example, self.tokenizer, self.args)
            features.append(feature)
        inputs = torch.stack([torch.tensor(f.input_ids) for f in features], dim=0)
        # print(f'{inputs.shape=}')
        logits, hidden_states = self.model(inputs, output_hidden_states=True)
        token_embeddings = torch.stack(hidden_states, dim=0).squeeze(dim=1).permute(1, 0, 2)
        # print(f'{token_embeddings.shape=}')

        assert inputs.shape[1] == token_embeddings.shape[0], f'Wrong shape: {token_embeddings.shape=}'  # Number of tokens
        assert self.model.config.num_hidden_layers+1 == token_embeddings.shape[1], f'Wrong shape: {token_embeddings.shape=}'  # Number of layers
        assert self.model.config.hidden_size == token_embeddings.shape[2], f'Wrong shape: {token_embeddings.shape=}'  # Number of layers

        token_vecs = []
        for token in token_embeddings:
            token_vecs.append(token[-2])  # Get state of second to last hidden layer
        if return_features:
            return token_vecs, features
        else:
            return token_vecs


def main():
    api = CodeBERTAPI()
    examples = [
        {
            "func": """int main()
{
    int x = 0;
    for (int i = 0; i < 10; i ++)
    {
        x ++;
    }
    return x;
}
""",
            "idx": 0,
            "target": 0,
        }
    ]
    token_embeddings = api.get_token_embeddings(examples)
    print(f'{len(token_embeddings)=}')
    print(f'{token_embeddings[0].shape=}')


if __name__ == '__main__':
    main()
