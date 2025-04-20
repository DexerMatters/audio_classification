import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from baseline import AudioTransformer
from loader import AudioDataset

# Assuming AudioDataset and AudioTransformer classes are defined elsewhere
# AudioDataset takes file_paths, labels, and audio processing parameters

if __name__ == "__main__":
    # Adjusted Hyperparameters
    sample_rate = 16000
    duration = 10  # seconds
    n_mels = 128
    n_fft = 400  # 25ms window at 16kHz
    hop_length = 160  # 10ms hop
    hidden_size = 64  # Reduced from 128
    num_layers = 1  # Reduced from 2
    num_heads = 4
    dropout = 0.3  # Increased from 0.1
    max_seq_len = 1000
    batch_size = 8  # Increased to improve generalization
    learning_rate = 3e-4
    weight_decay = 1e-5  # Added weight decay for regularization
    num_epochs = 100
    patience = 10  # Early stopping patience

    # Audio file extensions to look for
    audio_extensions = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    # Load file paths from directories
    speech_dir = "./data/speech"
    music_dir = "./data/music"

    # Check if directories exist
    if not os.path.exists(speech_dir):
        raise FileNotFoundError(f"Speech directory not found: {speech_dir}")
    if not os.path.exists(music_dir):
        raise FileNotFoundError(f"Music directory not found: {music_dir}")

    # Get all audio files from directories
    speech_files = []
    for filename in os.listdir(speech_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in audio_extensions:
            speech_files.append(os.path.join(speech_dir, filename))

    music_files = []
    for filename in os.listdir(music_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in audio_extensions:
            music_files.append(os.path.join(music_dir, filename))

    # Verify files were found
    if len(speech_files) == 0:
        raise ValueError(f"No audio files found in {speech_dir}")
    if len(music_files) == 0:
        raise ValueError(f"No audio files found in {music_dir}")

    print(f"Found {len(speech_files)} speech files and {len(music_files)} music files")

    # Assign labels: 0 for speech, 1 for music
    speech_labels = [0] * len(speech_files)
    music_labels = [1] * len(music_files)

    # Combine files and labels
    all_files = speech_files + music_files
    all_labels = speech_labels + music_labels

    # Split into training and validation sets with shuffling
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files,
        all_labels,
        test_size=0.2,  # 20% for validation
        stratify=all_labels,  # Maintain class proportion
        random_state=42,  # For reproducibility
    )

    # Initialize datasets with augmentation enabled
    train_dataset = AudioDataset(
        train_files,
        train_labels,
        sample_rate,
        duration,
        n_mels,
        n_fft,
        hop_length,
        augment=True,
    )
    val_dataset = AudioDataset(
        val_files,
        val_labels,
        sample_rate,
        duration,
        n_mels,
        n_fft,
        hop_length,
        augment=False,
    )

    # Create data loaders (shuffle=True for training, shuffle=False for validation)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model setup (assuming AudioTransformer is defined)
    model = AudioTransformer(
        n_mels, hidden_size, num_layers, num_heads, dropout, max_seq_len
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Add label smoothing to CrossEntropyLoss for regularization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Training loop
    best_val_loss = float("inf")
    best_accuracy = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        # Calculate average loss for the epoch
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        val_loss = 0.0
        val_batch_count = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_batch_count += 1

        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        accuracy = correct / total
        val_accuracies.append(accuracy)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Accuracy: {accuracy:.4f}"
        )

        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Plot and save training curves
        plt.figure(figsize=(15, 5))

        # Plot training and validation loss
        plt.subplot(1, 3, 1)
        plt.plot(range(1, epoch + 2), train_losses, "b-", label="Training Loss")
        plt.plot(range(1, epoch + 2), val_losses, "g-", label="Validation Loss")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        # Plot validation accuracy
        plt.subplot(1, 3, 2)
        plt.plot(range(1, epoch + 2), val_accuracies, "r-", label="Validation Accuracy")
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

        # Plot train vs val loss difference (overfitting indicator)
        plt.subplot(1, 3, 3)
        loss_diff = np.array(train_losses) - np.array(val_losses)
        plt.plot(
            range(1, epoch + 2),
            np.abs(loss_diff),
            "m-",
            label="Train-Val Loss Difference",
        )
        plt.title("Overfitting Indicator")
        plt.xlabel("Epoch")
        plt.ylabel("|Train Loss - Val Loss|")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig("./training_progress.png")
        plt.close()

    print(
        f"Training completed. Best validation loss: {best_val_loss:.4f}, Best accuracy: {best_accuracy:.4f}"
    )
