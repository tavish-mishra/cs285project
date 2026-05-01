import torch
import torch.nn as nn

#the transformer encoder, it can take in either actions or actions + states
#since we aren't really modeling a sequence here, attention operates across the features of the input
#bottleneck, same rank, rank increase are just dependent on what you set out_dim to be

class TransformerEncoder(nn.Module):
    def __init__(self, action_dim, obs_dim, d_model, out_dim, num_layers=2, num_heads=2):
        super(TransformerEncoder, self).__init__()
        self.input_dim = action_dim #+obs_dim, if we wanna go that route
        self.input_proj = nn.Linear(1, d_model)  # each input feature becomes a token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, out_dim)

    def forward(self, x):
        #so basically, you attend to each feature of the input and then aggregate via
        # mean to obtain a representation of the state-action pairs or of the actions
        x = x.unsqueeze(-1)        # (batch, input_dim, 1)
        x = self.input_proj(x)     # (batch, input_dim, d_model)
        x = self.transformer(x)    # (batch, input_dim, d_model)
        x = x.mean(dim=1)          # (batch, d_model)
        return self.output_proj(x)
