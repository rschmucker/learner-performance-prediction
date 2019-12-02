import torch
import torch.nn as nn


class RBP(nn.Module):
    def __init__(self, num_items, num_skills, hid_size, embed_size, num_hid_layers, drop_prob):
        """Recurrent behavior policy.

        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            hid_size (int): hidden layer dimension
            embed_size (int): query embedding dimension
            num_hid_layers (int): number of hidden layers
            drop_prob (float): dropout probability
        """
        super(RBP, self).__init__()
        self.embed_size = embed_size
        self.num_items = num_items

        self.item_embeds = nn.Embedding(num_items + 1, embed_size // 2, padding_idx=0)
        self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)

        self.lstm = nn.LSTM(2 * embed_size, hid_size, num_hid_layers, batch_first=True)
        self.dropout = nn.Dropout(p=drop_prob)
        self.lin = nn.Linear(hid_size, num_items)

    def forward(self, item_inputs, skill_inputs, label_inputs, hidden=None):
        inputs = self.get_inputs(item_inputs, skill_inputs, label_inputs)
        x, hidden = self.lstm(inputs, hx=hidden)
        output = (self.lin(self.dropout(x)))
        return output, hidden

    def get_inputs(self, item_inputs, skill_inputs, label_inputs):
        item_inputs = self.item_embeds(item_inputs)
        skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()

        inputs = torch.cat([item_inputs, skill_inputs, item_inputs, skill_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs
        inputs[..., self.embed_size:] *= 1 - label_inputs
        return inputs

    def repackage_hidden(self, hidden):
        # Return detached hidden for TBPTT
        return tuple((v.detach() for v in hidden))
