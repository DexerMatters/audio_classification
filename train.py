import glob
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

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

    # Load file paths from directories
    speech_files = glob.glob("./data/speech/*.*")  # All audio files in speech directory
    music_files = glob.glob("./data/music/*.*")  # All audio files in music directory

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
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Training completed. Best validation accuracy: {best_accuracy:.4f}")
