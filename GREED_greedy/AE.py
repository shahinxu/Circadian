import torch
import torch.nn as nn


class PhaseAutoEncoder(nn.Module):
    def __init__(self, input_dim, dropout=0.2, nhead=5):
        """
        Transformer autoencoder.
        Only used for training - greedy ordering happens on Transformer output AFTER training.
        """
        super(PhaseAutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.nhead = nhead

        # Calculate transformer_dim to be divisible by nhead
        if input_dim % nhead == 0:
            transformer_dim = input_dim
        else:
            transformer_dim = input_dim + (nhead - (input_dim % nhead))
        self.transformer_dim = transformer_dim

        # Input projection if needed
        if transformer_dim != input_dim:
            self.input_proj = nn.Linear(input_dim, transformer_dim)
        else:
            self.input_proj = None

        # Transformer encoder
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=nhead,
            dim_feedforward=transformer_dim * 4,
            dropout=dropout,
            batch_first=False,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)
        
        # Decoder from transformer output back to original dimension
        self.decoder = nn.Linear(transformer_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for training.
        Returns dummy phase values (not used) and reconstruction.
        """
        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
        
        # Transformer processing
        seq = x.unsqueeze(1)
        trans_out = self.transformer(seq)
        trans_out = trans_out.squeeze(1)
        
        # Decode back to original dimension
        reconstructed = self.decoder(trans_out)
        
        # Return dummy phase values (not used in new approach)
        dummy_phase_coords = torch.zeros(x.shape[0], 2, device=x.device)
        dummy_phase_angles = torch.zeros(x.shape[0], device=x.device)
        
        return dummy_phase_coords, dummy_phase_angles, reconstructed