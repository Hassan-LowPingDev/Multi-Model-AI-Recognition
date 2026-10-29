import os
import random
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DATASET_PATH = "ESC-50"
AUDIO_PATH = os.path.join(DATASET_PATH, "audio")
CSV_PATH = os.path.join(DATASET_PATH, "meta", "esc50.csv")

CLASSES = [
    "dog",
    "rain",
    "siren",
    "car_horn",
    "clock_alarm"
]

SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512

BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001

metadata = pd.read_csv(CSV_PATH)
metadata = metadata[
    metadata["category"].isin(CLASSES)
].copy()

metadata["label"] = metadata["category"].apply(
    lambda name: CLASSES.index(name)
)

print("\nNumber of selected recordings:")
print(metadata["category"].value_counts())
print("\nTotal recordings:", len(metadata))


# Use folds 1, 2, 3 for training,
# fold 4 for validation, fold 5 for testing.

train_rows = metadata[
    metadata["fold"].isin([1, 2, 3])
].reset_index(drop=True)

val_rows = metadata[
    metadata["fold"] == 4
].reset_index(drop=True)

test_rows = metadata[
    metadata["fold"] == 5
].reset_index(drop=True)

print("\nTraining recordings:", len(train_rows))
print("Validation recordings:", len(val_rows))
print("Test recordings:", len(test_rows))

def extract_features(file_path):
    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=DURATION
    )
    expected_length = SAMPLE_RATE * DURATION
    if len(audio) < expected_length:
        audio = np.pad(
            audio,
            (0, expected_length - len(audio))
        )
    else:
        audio = audio[:expected_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )
    mel_db = mel_db / 80.0
    return mel_db.astype(np.float32)


class SoundDataset(Dataset):
    def __init__(self, rows):
        self.features = []
        self.labels = []
        for _, row in rows.iterrows():
            file_path = os.path.join(
                AUDIO_PATH,
                row["filename"]
            )
            feature = extract_features(file_path)
            self.features.append(feature)
            self.labels.append(int(row["label"]))

        self.features = torch.tensor(
            np.array(self.features),
            dtype=torch.float32
        ).unsqueeze(1)
        self.labels = torch.tensor(
            self.labels,
            dtype=torch.long
        )
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return (self.features[index], self.labels[index])


print("\nPreparing training data...")
train_dataset = SoundDataset(train_rows)

print("Preparing validation data...")
val_dataset = SoundDataset(val_rows)

print("Preparing test data...")
test_dataset = SoundDataset(test_rows)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# -----------------------------
# 5. BUILD THE CNN
# -----------------------------

class SoundCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, len(CLASSES))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SoundCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (
                predictions == labels
            ).sum().item()
            total += labels.size(0)
    
    accuracy = 100 * correct / total
    average_loss = total_loss / total
    return average_loss, accuracy


best_val_accuracy = -1.0
best_state = None

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * labels.size(0)

    train_loss = running_loss / len(train_dataset)
    val_loss, val_accuracy = evaluate(val_loader)
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

if best_state is None:
    raise RuntimeError("Training did not produce a model.")
model.load_state_dict(best_state)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "sample_rate": SAMPLE_RATE,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "duration": DURATION
    },
    "sound_model.pth")

print("\nBest validation accuracy:", best_val_accuracy)
print("Saved model to sound_model.pth")

test_loss, test_accuracy = evaluate(test_loader)

print("\n========== FINAL TEST RESULTS ==========")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
