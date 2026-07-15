import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEQ_LENGTH = 10
INPUT_SIZE = 132
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PoseSequenceDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.sequences = torch.tensor(x_data, dtype=torch.float32)
        self.labels = torch.tensor(y_data, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class LSTMClassifier(nn.Module):
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out[:, -1, :])
        return logits

def main():
    raw_data = np.load("data_train.npy", allow_pickle=True)

    clean_data = [(x, y) for x, y in raw_data if isinstance(x, np.ndarray) and x.shape == (SEQ_LENGTH, INPUT_SIZE)]
    if len(clean_data) == 0:
        raise ValueError("No valid arrays matching target shape found.")

    X = np.array([item[0] for item in clean_data], dtype=np.float32)
    y = np.array([item[1] for item in clean_data], dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_dataset = PoseSequenceDataset(X_train, y_train)
    val_dataset = PoseSequenceDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = LSTMClassifier().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_seqs, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch_seqs = batch_seqs.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            predictions = model(batch_seqs)
            loss = loss_fn(predictions, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        correct_preds = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_seqs, batch_labels in val_loader:
                batch_seqs = batch_seqs.to(device)
                batch_labels = batch_labels.to(device)

                predictions = model(batch_seqs)
                preds_class = torch.argmax(predictions, dim=1)
                
                total_samples += batch_labels.size(0)
                correct_preds += (preds_class == batch_labels).sum().item()

        accuracy = 100.0 * correct_preds / total_samples
        print(f"Validation Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), "mp_lstm_faint_model.pth")

if __name__ == "__main__":
    main()
