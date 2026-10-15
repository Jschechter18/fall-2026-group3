import torch
from torch import nn, Tensor
from torch.nn import functional as F


class SparseAutoencoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, latent_dim: int,
        sparsity_mode: str = "l1", top_k: int | None = None, tied_initialization: bool = True,
    ):
        super().__init__()
        if sparsity_mode not in {"l1", "topk"}:
            raise ValueError("sparsity_mode must be 'l1' or 'topk'.")
        if sparsity_mode == "topk" and (
            not isinstance(top_k, int) or isinstance(top_k, bool)
            or not 1 <= top_k <= latent_dim
        ):
            raise ValueError("TopK requires an integer top_k between 1 and latent_dim.")
        self.sparsity_mode = sparsity_mode
        self.top_k = top_k
        
        self.input_dim = input_dim
        self.output_dim = input_dim # added this for clarity
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.input_scale: Tensor
        self.register_buffer("input_scale", torch.tensor(1.0))
        
        self.encoder_layer = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder_layer = nn.Sequential(
            nn.Linear(latent_dim, self.output_dim)
        )

        self.normalize_decoder_weights()

        if tied_initialization:
            encoder_linear = self.encoder_layer[0]
            decoder_linear = self.decoder_layer[0]
            assert isinstance(encoder_linear, nn.Linear)
            assert isinstance(decoder_linear, nn.Linear)
            with torch.no_grad():
                encoder_linear.weight.copy_(decoder_linear.weight.T)
        
    def encode_pre_topk(self, activation: Tensor) -> Tensor:
        """Compute nonnegative encoder activations before TopK selection.

        Parameters
        ----------
        activation : Tensor
            Input activation vector, divided by the stored input scale before encoding.

        Returns
        -------
        Tensor
            Nonnegative feature vector before sparsification, attached to the autograd graph.
        """
        return self.encoder_layer(activation / self.input_scale)

    @staticmethod
    def select_topk(features: Tensor, k: int) -> Tensor:
        """Keep the largest k values on the last axis, preserving gradients.

        Accepts k=0 for an empty auxiliary selection. Callers handle eligibility
        masking and cap k to the number of eligible features.

        Parameters
        ----------
        features : Tensor
            Feature values to select from along the last dimension.
        k : int
            Number of values to retain, between zero and the feature dimension.

        Returns
        -------
        Tensor
            Feature vector with unselected values set to zero and the input shape preserved.

        Raises
        ------
        ValueError
            If k is not an integer within the supported range.
        """
        if type(k) is not int or not 0 <= k <= features.shape[-1]:
            raise ValueError("k must be an integer between 0 and the feature dimension.")
        values, indices = torch.topk(features, k=k, dim=-1)
        return torch.zeros_like(features).scatter(-1, indices, values)

    def _sparsify(self, features: Tensor) -> Tensor:
        """Apply the sparsity mode configured for the model.

        Parameters
        ----------
        features : Tensor
            Nonnegative encoder activations before sparsification.

        Returns
        -------
        Tensor
            TopK feature vector in TopK mode, or the unchanged features in L1 mode.
        """
        if self.sparsity_mode == "topk":
            assert self.top_k is not None
            return self.select_topk(features, self.top_k)
        return features

    def encoder(self, activation: Tensor) -> Tensor:
        """Encoder layer of model. Used to convert activation to sparse feature vector.

        Parameters
        ----------
        activation : Tensor
            Activation output is the input of the encoder.

        Returns
        -------
        Tensor
            Sparse feature vector. This is a sparse representation of the activation input.
        """
        return self._sparsify(self.encode_pre_topk(activation))
        
    
    def decoder(self, sparse_features: Tensor) -> Tensor:
        """Decoder layer of model. Used to project sparse feature vector back to original activation vector representation.

        Parameters
        ----------
        sparse_features : Tensor
            Sparse feature vector, output of the encoder.

        Returns
        -------
        Tensor
            Reconstructed activation vector. This is the reconstructed version of the original activation vector from the sparse feature representation.
        """
        return self.decoder_layer(sparse_features) * self.input_scale
    
    def decode_without_bias(self, sparse_features: Tensor) -> Tensor:
        """Decode features without decoder bias or input-scale restoration.

        Parameters
        ----------
        sparse_features : Tensor
            Sparse feature vector to project using the shared decoder weights.

        Returns
        -------
        Tensor
            Decoded vector in normalized input units, suitable for residual reconstruction.
        """
        decoder_linear = self.decoder_layer[0]
        assert isinstance(decoder_linear, nn.Linear)
        return F.linear(sparse_features, decoder_linear.weight)

    def forward_with_pre_topk(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return features, reconstruction, and pre-TopK activations.

        Computes the encoder once and keeps the pre-TopK tensor attached to its
        autograd graph for auxiliary training objectives. Nothing is cached.

        Parameters
        ----------
        x : Tensor
            Input activation vector.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Tuple containing:
            - Sparse feature vector (output of the encoder)
            - Reconstructed activation vector in original input units
            - Nonnegative encoder activations before TopK selection
        """
        pre_topk = self.encode_pre_topk(x)
        sparse_features = self._sparsify(pre_topk)
        return sparse_features, self.decoder(sparse_features), pre_topk

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """Normalize each decoder column to prevent sparsity-penalty scaling."""
        decoder_weight = self.decoder_layer[0].weight
        norms = torch.linalg.vector_norm(decoder_weight, dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder_layer[0].weight.data = decoder_weight / norms
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Full forward pass through model.

        Parameters
        ----------
        x : Tensor
            Input activation vector.

        Returns
        -------
        tuple[Tensor, Tensor]
            Tuple containing:
            - Sparse feature vector (output of the encoder)
            - Reconstructed activation vector (output of the decoder)
        """
        sparse_features = self.encoder(x)
        reconstructed_activation = self.decoder(sparse_features)
        
        return sparse_features, reconstructed_activation
