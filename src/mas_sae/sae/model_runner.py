import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from mas_sae.sae.sparse_autoencoder import SparseAutoencoder as SAE

import torch.optim as optim


class ModelRunner:
    def __init__(
        self, model: SAE, sparsity_coefficient: float, optimizer: optim.Optimizer,
        training_mean: torch.Tensor | None = None,
        inactivity_window_examples: int = 10000,
        aux_coefficient: float = 0.0,
        aux_k: int = 512,
        aux_dead_window_examples: int = 10000,
    ):
        if inactivity_window_examples < 1:
            raise ValueError("inactivity_window_examples must be positive.")
        if (type(aux_coefficient) not in (int, float)
                or not math.isfinite(aux_coefficient) or aux_coefficient < 0):
            raise ValueError("aux_coefficient must be finite and nonnegative.")
        if type(aux_dead_window_examples) is not int or aux_dead_window_examples < 1:
            raise ValueError("aux_dead_window_examples must be a positive integer.")
        if type(aux_k) is not int or aux_k < 1:
            raise ValueError("aux_k must be a positive integer.")
        if aux_coefficient > 0:
            if model.sparsity_mode != "topk":
                raise ValueError("AuxK requires topk sparsity mode.")
            if aux_k > model.latent_dim:
                raise ValueError("aux_k must not exceed latent_dim when AuxK is enabled.")
        self.aux_coefficient = aux_coefficient
        self.aux_k = aux_k
        self.aux_dead_window_examples = aux_dead_window_examples
        self.model = model
        self.sparsity_coefficient = sparsity_coefficient
        self.training_mean = training_mean.detach().clone() if training_mean is not None else None
        self.inactivity_window_examples = inactivity_window_examples
        self.training_examples_seen = 0
        self.last_fired_example: torch.Tensor | None = None
        self.last_feature_frequencies: torch.Tensor | None = None
        
        self.optimizer = optimizer

    def _track_training_activity(self, active: torch.Tensor):
        """Record the last training-example position at which each feature fired."""
        if self.last_fired_example is None:
            self.last_fired_example = torch.zeros(self.model.latent_dim, dtype=torch.long)
        positions = torch.arange(1, active.shape[0] + 1, device=active.device)[:, None]
        last_positions = (active * positions).amax(dim=0).cpu()
        fired = last_positions > 0
        self.last_fired_example[fired] = self.training_examples_seen + last_positions[fired]
        self.training_examples_seen += active.shape[0]
    
    
    def _loss_fn(self, reconstructed: torch.Tensor, batch: torch.Tensor, sparse_features: torch.Tensor):
        scale = self.model.input_scale
        rec_loss = F.mse_loss(reconstructed / scale, batch / scale)
        
        if self.model.sparsity_mode == "l1":
            l1_penalty = sparse_features.abs().mean()
            weighted_sparsity_loss = self.sparsity_coefficient * l1_penalty
        else:
            weighted_sparsity_loss = rec_loss.new_zeros(())

        loss = rec_loss + weighted_sparsity_loss
        
        return {
            "loss": loss,
            "rec_loss": rec_loss,
            "weighted_sparsity_loss": weighted_sparsity_loss
        }

    
    def _auxiliary_loss(
        self, activations: torch.Tensor, reconstructed: torch.Tensor,
        pre_topk: torch.Tensor,
    ) -> tuple[torch.Tensor, float, dict[str, float]]:
        """Reconstruct the detached residual using persistently inactive features.

        Parameters
        ----------
        activations : torch.Tensor
            Input batch in original activation units.
        reconstructed : torch.Tensor
            Main-path reconstruction in original activation units.
        pre_topk : torch.Tensor
            Nonnegative encoder activations before main TopK selection.

        Returns
        -------
        tuple[torch.Tensor, float, dict[str, float]]
            Auxiliary loss in normalized units, fraction of eligible features,
            and detached counts and squared sums for epoch diagnostics. When
            disabled, warming up, or no features qualify, returns zero loss,
            zero eligibility, and an empty dictionary.
        """
        if (self.aux_coefficient == 0 or self.last_fired_example is None
                or self.training_examples_seen < self.aux_dead_window_examples):
            return reconstructed.new_zeros(()), 0.0, {}
        eligible = (
            self.training_examples_seen - self.last_fired_example
            >= self.aux_dead_window_examples
        ).to(pre_topk.device)
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0:
            return reconstructed.new_zeros(()), 0.0, {}

        # Exclude ineligible features even when eligible activations are zero.
        masked = pre_topk.masked_fill(~eligible, -torch.inf)
        aux_features = self.model.select_topk(masked, min(self.aux_k, eligible_count))
        auxiliary_prediction = self.model.decode_without_bias(aux_features)
        residual = ((activations - reconstructed) / self.model.input_scale).detach()
        aux_loss = F.mse_loss(auxiliary_prediction, residual)
        with torch.no_grad():
            diagnostics = {
                "examples": float(activations.shape[0]),
                "elements": float(residual.numel()),
                "positive_features": float((aux_features > 0).sum().item()),
                "eligible_positive_activations": float(((pre_topk > 0) & eligible).sum().item()),
                "eligible_activation_pairs": float(eligible_count * activations.shape[0]),
                "prediction_sse": auxiliary_prediction.square().sum().item(),
                "residual_sse": residual.square().sum().item(),
                "error_sse": aux_loss.detach().item() * residual.numel(),
            }
        return aux_loss, eligible_count / self.model.latent_dim, diagnostics

    def _common(self, batch: torch.Tensor, training_mode: bool = False):
        """Run a single forward pass and compute the loss.

        Parameters
        ----------
        batch : torch.Tensor
            Single batch from dataloader of size `batch_size`. The shape is (batch_size, input_size).
        training_mode : bool, optional
            Whether to update main-path activity counters and compute AuxK, by default False.

        Returns
        -------
        dict
            Loss components, reconstruction, sparse features, and auxiliary eligibility.
        """
        device = next(self.model.parameters()).device
        activations = batch.to(device) # batch.shape = (batch_size, input_size)
        
        use_aux = training_mode and self.aux_coefficient > 0
        pre_topk: torch.Tensor | None = None
        if use_aux:
            sparse_features, reconstructed, pre_topk = self.model.forward_with_pre_topk(activations)
        else:
            sparse_features, reconstructed = self.model(activations)
        
        losses = self._loss_fn(reconstructed, activations, sparse_features)
        aux_loss = reconstructed.new_zeros(())
        eligible_fraction = 0.0
        aux_diagnostics = {}
        if training_mode:
            self._track_training_activity(sparse_features.detach() > 0)
        if use_aux:
            assert pre_topk is not None
            aux_loss, eligible_fraction, aux_diagnostics = self._auxiliary_loss(
                activations, reconstructed, pre_topk,
            )
        weighted_aux_loss = self.aux_coefficient * aux_loss

        return {'loss': losses["loss"] + weighted_aux_loss,
                'aux_loss': aux_loss,
                'weighted_aux_loss': weighted_aux_loss,
                'aux_eligible_feature_fraction': eligible_fraction,
                'aux_diagnostics': aux_diagnostics,
                'rec_loss': losses["rec_loss"],
                'weighted_sparsity_loss': losses['weighted_sparsity_loss'],
                'reconstructed': reconstructed,
                'sparse_features': sparse_features}
    
    def _run_epoch(self, dataloader: DataLoader, training_mode: bool = False):
        """Run a single epoch for any evaluation mode.

        Parameters
        ----------
        dataloader : DataLoader
            The constructed DataLoader for the specified split.
        training_mode : bool, optional
            Whether to run the epoch in training mode or eval mode., by default False

        Returns
        -------
        dict
            Average losses, activity statistics, and available reconstruction diagnostics.
        """
        running_loss = 0
        running_rec_loss = 0
        running_weighted_sparsity_loss = 0
        running_aux_loss = 0.0
        running_weighted_aux_loss = 0.0
        running_aux_eligible_fraction = 0.0
        auxiliary_sums: dict[str, float] = {}
        num_examples = 0
        running_baseline_loss = 0.0
        device = next(self.model.parameters()).device
        feature_counts = torch.zeros(self.model.latent_dim, dtype=torch.long, device=device)
        mean = self.training_mean.to(device) if self.training_mean is not None else None
        
        self.model.train() if training_mode else self.model.eval()
        
        description = "Training" if training_mode else "Evaluating"
        progress_bar = tqdm(dataloader, desc=description, leave=False)
        
        for _, batch in enumerate(progress_bar, start=1):
            with torch.set_grad_enabled(training_mode):
                outputs = self._common(batch, training_mode=training_mode)
                
                active = outputs["sparse_features"].detach() > 0
                feature_counts += active.sum(dim=0)
                if mean is not None:
                    residual = (batch.to(device) - mean) / self.model.input_scale
                    running_baseline_loss += residual.square().mean().item() * batch.shape[0]
            
            if training_mode:
                self.optimizer.zero_grad()
                outputs['loss'].backward()
                self.optimizer.step()
                self.model.normalize_decoder_weights()
                
            batch_size = batch.shape[0]
            running_loss += outputs['loss'].item() * batch_size
            running_rec_loss += outputs['rec_loss'].item() * batch_size
            running_weighted_sparsity_loss += outputs['weighted_sparsity_loss'].item() * batch_size
            running_aux_loss += outputs['aux_loss'].item() * batch_size
            running_weighted_aux_loss += outputs['weighted_aux_loss'].item() * batch_size
            running_aux_eligible_fraction += outputs['aux_eligible_feature_fraction'] * batch_size
            for key, value in outputs['aux_diagnostics'].items():
                auxiliary_sums[key] = auxiliary_sums.get(key, 0.0) + value
            num_examples += batch_size
            
        if num_examples == 0:
            raise ValueError("Cannot evaluate an empty dataset.")

        frequencies = feature_counts.float().cpu() / num_examples
        self.last_feature_frequencies = frequencies
        percentiles = torch.quantile(frequencies, torch.tensor([0.0, 0.5, 0.9, 0.99, 1.0]))
        metrics = {
            "loss": running_loss / num_examples,
            "rec_loss": running_rec_loss / num_examples,
            "weighted_sparsity_loss": running_weighted_sparsity_loss / num_examples,
            "mean_active_features": feature_counts.sum().item() / num_examples,
            "inactive_feature_fraction": (feature_counts == 0).float().mean().item(),
            "firing_frequency_min": percentiles[0].item(),
            "firing_frequency_p50": percentiles[1].item(),
            "firing_frequency_p90": percentiles[2].item(),
            "firing_frequency_p99": percentiles[3].item(),
            "firing_frequency_max": percentiles[4].item(),
        }
        if self.training_mean is not None:
            baseline = running_baseline_loss / num_examples
            metrics["mean_baseline_rec_loss"] = baseline
            metrics["variance_explained"] = 1 - metrics["rec_loss"] / baseline if baseline > 0 else None
        if training_mode:
            metrics["aux_loss"] = running_aux_loss / num_examples
            metrics["weighted_aux_loss"] = running_weighted_aux_loss / num_examples
            metrics["aux_eligible_feature_fraction"] = running_aux_eligible_fraction / num_examples
            if self.aux_coefficient > 0:
                measured_examples = auxiliary_sums.get("examples", 0.0)
                elements = auxiliary_sums.get("elements", 0.0)
                residual_sse = auxiliary_sums.get("residual_sse", 0.0)
                eligible_pairs = auxiliary_sums.get("eligible_activation_pairs", 0.0)
                metrics["auxiliary"] = {
                    "eligible_feature_fraction": metrics["aux_eligible_feature_fraction"],
                    "measured_examples": int(measured_examples),
                    "eligible_positive_activation_fraction": auxiliary_sums["eligible_positive_activations"] / eligible_pairs if eligible_pairs else None,
                    "mean_positive_features": auxiliary_sums["positive_features"] / measured_examples if measured_examples else None,
                    "prediction_rms": math.sqrt(auxiliary_sums["prediction_sse"] / elements) if elements else None,
                    "residual_rms": math.sqrt(residual_sse / elements) if elements else None,
                    "residual_error_reduction": 1 - auxiliary_sums["error_sse"] / residual_sse if residual_sse > 0 else None,
                }
            last_fired = self.last_fired_example
            if last_fired is None:
                raise RuntimeError("Training did not initialize feature activity counters.")
            metrics["persistent_inactive_feature_fraction"] = (
                (self.training_examples_seen - last_fired >= self.inactivity_window_examples)
                .float().mean().item()
                if self.training_examples_seen >= self.inactivity_window_examples else None
            )
            metrics["training_examples_seen"] = self.training_examples_seen
        return metrics
    
    def train_epoch(self, dataloader: DataLoader):
        """Runs one epoch under full training conditions.

        Parameters
        ----------
        dataloader : DataLoader
            The constructed DataLoader for the specified split.

        Returns
        -------
        dict
            Epoch losses and diagnostics, including training inactivity counters.
            Auxiliary loss, weighted auxiliary loss, and eligible feature fraction
            are averaged across batches with weighting by the number of examples.
            Activity statistics count only main-path features.
            When AuxK is enabled, auxiliary contains effectiveness diagnostics
            over batches with eligible features. RMS and error reduction use
            accumulated squared sums; undefined values are None. Its
            measured_examples reports coverage, excluding warmup and batches
            without eligible features.
            eligible_positive_activation_fraction counts positive encoder
            outputs before auxiliary selection across eligible feature/example pairs.
        """
        return self._run_epoch(dataloader, training_mode=True)
    
    def val_epoch(self, dataloader: DataLoader):
        """Runs one epoch under full validation conditions.

        Parameters
        ----------
        dataloader : DataLoader
            The constructed DataLoader for the specified split.

        Returns
        -------
        dict
            Main-path evaluation losses and diagnostics without auxiliary metrics
            or updates to training counters. For TopK, loss equals reconstruction
            loss and can be used directly for model selection.
        """
        return self._run_epoch(dataloader, training_mode=False)
    
    def test(self, dataloader: DataLoader):
        """Runs one full forward pass under full test conditions.

        Parameters
        ----------
        dataloader : DataLoader
            The constructed DataLoader for the specified split.

        Returns
        -------
        dict
            Main-path evaluation losses and diagnostics without auxiliary metrics
            or updates to training counters. For TopK, loss equals reconstruction loss.
        """
        return self._run_epoch(dataloader, training_mode=False)
