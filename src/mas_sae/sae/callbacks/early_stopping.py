"""Patience-based early stopping on validation loss for SAE training (stage 4)."""


class EarlyStoppingCallback:
    """Counts epochs since validation loss last fell by more than min_delta below best_loss.

    best_loss moves only on such an improvement, so smaller gains accumulate until they clear the
    threshold. best_loss and counter are part of the continuation state saved by training_state.
    """

    def __init__(self, patience: int, min_delta: float=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def on_validation_end(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered based on the validation loss.

        Early stopping is triggered if the validation loss does not improve for a number of consecutive epochs
        defined by the patience parameter.

        Parameters
        ----------
        val_loss : float
            The validation loss for the current epoch.

        Returns
        -------
        bool
            True if early stopping should be triggered, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience