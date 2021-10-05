# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Source: https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, input_ids=None, labels=None, output_hidden_states=False):
        encoder_outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1),
                                       output_hidden_states=output_hidden_states)
        hidden_states = None
        if output_hidden_states:
            outputs = encoder_outputs.logits
            hidden_states = encoder_outputs.hidden_states
        else:
            outputs = encoder_outputs[0]
        logits = outputs
        prob = F.sigmoid(logits)

        ret = []
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            ret.append(loss)
        ret.append(prob)
        if output_hidden_states:
            ret.append(hidden_states)
        return tuple(ret)
