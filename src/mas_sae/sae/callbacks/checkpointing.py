import torch

class CheckpointEvaluatorCallback:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
    
    def on_validation_end(self, train_loss, val_loss, best_loss, epoch, model, optimizer, scheduler, run_directory):
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_loss': best_loss,
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, run_directory / "checkpoints" / "best_checkpoint.pt")