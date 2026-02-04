import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from sklearn.manifold import TSNE

def plot_model_loss(training_losses, validation_losses):

    # Plot training and validation losses
    plt.figure(figsize=(8, 6))

    plt.plot(range(len(training_losses)), training_losses, label='Training Loss')
    plt.plot(range(len(validation_losses)), validation_losses, label='Validation Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Losses')

    plt.legend()
    plt.grid(True)
    plt.show()

def save_checkpoint(epoch, model, optimizer, avg_train_loss, avg_val_loss, train_losses, val_losses, path='checkpoint\checkpoint.pth'):
    """Save training checkpoint"""
    checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_loss_history': train_losses,
            'val_loss_history': val_losses,
            'model_architecture': str(model),  # Optional: for verification
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at epoch {epoch} in the file: {path}")

def resume_training(checkpoint_path, model, optimizer):
    """Resume training from checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path)
    
    # Load model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    start_epoch = checkpoint['epoch']
    train_losses = checkpoint.get('train_loss_history', [])
    val_losses = checkpoint.get('val_loss_history', [])
    
    print(f"Resumed training from epoch {start_epoch}")
    
    return start_epoch, train_losses, val_losses

class Encoder(torch.nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=6, dropout=0.2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x, lengths):
        # x: tensor of shape (batch_size, seq_length, input_size)
        # lengths: tensor of shape (batch_size), containing the lengths of each sequence in the batch

        # NOTE: Here we use the pytorch functions pack_padded_sequence and pad_packed_sequence, which
        # allow us to
        packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_x)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output, hidden

class Decoder(torch.nn.Module):
    def __init__(
        self, input_size=2, hidden_size=64, output_size=1, num_layers=6, dropout=0.2
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, lengths=None):
        if lengths is not None:
            # unpad the light curves so that our latent representations learn only from real data
            packed_x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_output, hidden = self.gru(packed_x, hidden)

            # re-pad the light curves so that they can be processed elsewhere
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, hidden = self.gru(x, hidden)
        prediction = self.fc(output)
        return prediction, hidden

