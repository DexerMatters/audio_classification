import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from baseline import AudioTransformer
from loader import AudioDataset

# Assuming AudioDataset and AudioTransformer classes are defined elsewhere
# AudioDataset takes file_paths, labels, and audio processing parameters

if __name__ == "__main__":
    # Hyperparameters
    sample_rate = 16000
    duration = 10  # seconds
    n_mels = 128
    n_fft = 400  # 25ms window at 16kHz
    hop_length = 160  # 10ms hop
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    max_seq_len = 1000  # Approx frames for 10s with 10ms hop
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 50

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

    # Initialize datasets
    train_dataset = AudioDataset(
        train_files, train_labels, sample_rate, duration, n_mels, n_fft, hop_length
    )
    val_dataset = AudioDataset(
        val_files, val_labels, sample_rate, duration, n_mels, n_fft, hop_length
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_accuracy = 0
    train_losses = []
    val_accuracies = []

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
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        # Calculate average loss for the epoch
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        val_accuracies.append(accuracy)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Accuracy: {accuracy:.4f}"
        )

        # Plot and save training curves
        plt.figure(figsize=(12, 5))

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epoch + 2), train_losses, "b-", label="Training Loss")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.legend()

        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epoch + 2), val_accuracies, "r-", label="Validation Accuracy")
        plt.title("Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig("./status.png")
        plt.close()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Training completed. Best validation accuracy: {best_accuracy:.4f}")
