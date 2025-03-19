

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------- DATASET DEFINITION --------------
class ZipDataset(Dataset):

    def __init__(self, file_path):
        self.samples = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                label = int(float(parts[0]))  # first value is the digit label
                pixels = [float(x) for x in parts[1:]]
                if len(pixels) != 256:
                    raise ValueError(f"Expected 256 pixel values, got {len(pixels)}.")
                # Reshape to 1x16x16 (1 channel)
                image_tensor = torch.tensor(pixels, dtype=torch.float32).view(1, 16, 16)
                label_tensor = torch.tensor(label, dtype=torch.long)
                self.samples.append((image_tensor, label_tensor))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

# -------------- LOCALLY CONNECTED LAYER --------------
class LocallyConnectedLayer(nn.Module):
    """
    A single locally connected layer (no weight sharing).
    each output spatial location has its own filter.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = None
        self.bias = None
        self.out_h = None
        self.out_w = None
        self.initialized = False

    def _initialize_params(self, in_height, in_width):
        kh, kw = self.kernel_size
        out_h = (in_height + 2 * self.padding - kh) // self.stride + 1
        out_w = (in_width  + 2 * self.padding - kw) // self.stride + 1
        self.out_h = out_h
        self.out_w = out_w
        in_features = self.in_channels * kh * kw
        # Weight shape: (out_channels, in_features, out_h*out_w)
        self.weight = nn.Parameter(torch.empty(self.out_channels, in_features, out_h * out_w))
        self.bias   = nn.Parameter(torch.empty(self.out_channels, out_h * out_w))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.bias)
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            _, _, H_in, W_in = x.shape
            self._initialize_params(H_in, W_in)
        kh, kw = self.kernel_size
        patches = F.unfold(x, kernel_size=(kh, kw), stride=self.stride, padding=self.padding)
        out_unfold = torch.einsum('cif,nif->ncf', self.weight, patches)
        out_unfold += self.bias.unsqueeze(0)
        out_unfold = out_unfold.contiguous()
        out = out_unfold.view(x.size(0), self.out_channels, self.out_h, self.out_w)
        return out

# -------------- COMPLETE NETWORK DEFINITION --------------
class LocallyConnectedNet(nn.Module):
    def __init__(self):
        super().__init__()
        # With a 16x16 input and kernel=3, stride=1, no padding:
        self.loc1 = LocallyConnectedLayer(1, 8, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(8)
        self.loc2 = LocallyConnectedLayer(8, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)
        self.loc3 = LocallyConnectedLayer(16, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc   = nn.Linear(32 * 10 * 10, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.loc1(x)))
        x = F.relu(self.bn2(self.loc2(x)))
        x = F.relu(self.bn3(self.loc3(x)))
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.fc(x)
        return out

# -------------- TRAINING & EVALUATION FUNCTIONS --------------
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
    return model

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)
    avg_test_loss = test_loss / test_total
    avg_test_acc = 100.0 * test_correct / test_total
    return avg_test_loss, avg_test_acc

def evaluate_ensemble(models, test_loader, criterion):
    ensemble_loss = 0.0
    ensemble_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Sum probabilities from each model
            ensemble_probs = None
            for model in models:
                model.eval()
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                if ensemble_probs is None:
                    ensemble_probs = probs
                else:
                    ensemble_probs += probs
            # Average the probabilities
            ensemble_probs /= len(models)
            loss = criterion(ensemble_probs.log(), labels)
            ensemble_loss += loss.item() * labels.size(0)
            preds = ensemble_probs.argmax(dim=1)
            ensemble_correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = ensemble_loss / total
    avg_acc = 100.0 * ensemble_correct / total
    return avg_loss, avg_acc

# -------------- MAIN: ENSEMBLE TRAINING & EVALUATION --------------
def main():
    # 1) Dummy test to verify the forward pass
    dummy_input = torch.randn(1, 1, 16, 16)
    model_test = LocallyConnectedNet()
    dummy_out = model_test(dummy_input)
    print("Dummy output shape:", dummy_out.shape)

    # 2) Load data
    train_dataset = ZipDataset("zip_train.txt")
    test_dataset  = ZipDataset("zip_test.txt")
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 3) Train ensemble models
    num_models = 3
    num_epochs = 10
    models = []
    for i in range(num_models):
        print(f"\nTraining model {i+1} of {num_models}...")
        model = LocallyConnectedNet()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model = train_model(model, train_loader, criterion, optimizer, num_epochs)
        models.append(model)

    # 4) Evaluate individual model (optional)
    print("\nEvaluating one model...")
    avg_test_loss, avg_test_acc = evaluate_model(models[0], test_loader, criterion)
    print(f"Single Model - Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.2f}%")

    # 5) Evaluate ensemble of models
    print("\nEvaluating ensemble of models...")
    ensemble_loss, ensemble_acc = evaluate_ensemble(models, test_loader, criterion)
    print(f"Ensemble - Test Loss: {ensemble_loss:.4f}, Test Accuracy: {ensemble_acc:.2f}%")

if __name__ == "__main__":
    main()
