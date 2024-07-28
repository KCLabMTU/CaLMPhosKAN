import sys
import math
import pyfiglet
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import (matthews_corrcoef, f1_score, precision_score, recall_score, average_precision_score)
from torch.cuda.amp import autocast, GradScaler
from KAN import KANLinear
from tabulate import tabulate

text = "CaLMPhosKAN"
logo = pyfiglet.figlet_format(text)
print(logo)

# Check proper input
if len(sys.argv) != 2:
    print("usage: python evaluate.py dataset\ndataset may be ST or Y")
    sys.exit()

if sys.argv[1].upper() == 'ST':
    model_path = 'models/ST_model.pth'
    dataset = 'data/ST_dataset.npy'
    test_labels = 'data/ST_labels.csv'
elif sys.argv[1].upper() == 'Y':
    model_path = 'models/Y_model.pth'
    dataset = 'data/Y_dataset.npy'
    test_labels = 'data/Y_labels.csv'
else: 
    print("ERROR: invalid dataset\ndataset may be ST or Y")
    sys.exit()

# Set device to cpu
device = torch.device('cpu')

# Load test data
print("Loading data...")
x_test_combined = np.load(dataset)
x_test_combined = np.array(x_test_combined, dtype=np.float32).squeeze()
y_test = pd.read_csv(test_labels).values.flatten()

# Convert to Pytorch tensors
x_test_torch = torch.from_numpy(x_test_combined).unsqueeze(1).to(device)
y_test_torch = torch.tensor(y_test, dtype=torch.float32).to(device)
print("Data successfully loaded!")

# Set some hyperparams
print("\nInitializing model...")
wavelet = 'dog'
drop = 0.3

# Define model
class ConvBiGRUKAN(nn.Module):
    def __init__(self, window_size, dim=x_test_combined.shape[2]):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5,5), padding=0, bias=False)
        self.dropout = nn.Dropout(drop)
        conv_output_height = window_size - 4
        conv_output_width = dim - 4
        self.gru_input_dim = 16 * conv_output_width
        self.gru = nn.GRU(self.gru_input_dim, 8, bidirectional=True, batch_first=True)

        # KAN Layers for ST
        if sys.argv[1].upper() == 'ST':
            self.fc1 = KANLinear(conv_output_height * 8 * 2, 128, wavelet_type=wavelet)
            self.fc2 = KANLinear(128, 32, wavelet_type=wavelet)
            self.fc3 = KANLinear(32, 1, wavelet_type=wavelet)
        # KAN Layers for Y
        elif sys.argv[1].upper() == 'Y':
            self.fc1 = KANLinear(conv_output_height * 8 * 2, 24, wavelet_type=wavelet)
            self.fc2 = KANLinear(24, 1, wavelet_type=wavelet)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        x, _ = self.gru(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        if sys.argv[1].upper() == 'ST':  # Third layer is only needed for ST
            x = self.dropout(x)
            x = self.fc3(x)
        return x

# Use dataloader
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test_torch, y_test_torch), batch_size=1024, shuffle=False)

# Evaluate model
model = ConvBiGRUKAN(9).to(device)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
print("Model successfully initialized!")
if sys.argv[1].upper() == 'ST':
    print("\nEvaluating phosphorylation prediction on serine and threonine...")
elif sys.argv[1].upper() == 'Y':
    print("\nEvaluating phosphorylation prediction on tyrosine...")
model.eval()
y_pred_prob = []
y_test = []
with torch.no_grad():
    for samples, labels in test_loader:
        samples = samples.to(device)
        labels = labels.to(device)
        outputs = model(samples)
        y_test.extend(labels.detach().cpu().numpy())
        y_pred_prob.extend(torch.sigmoid(outputs).detach().cpu().numpy())
y_pred_prob = np.array(y_pred_prob)
y_pred = (y_pred_prob > 0.5).astype(int)
y_test_np = np.array(y_test)
mcc = matthews_corrcoef(y_test_np, y_pred)
precision = precision_score(y_test_np, y_pred)
recall = recall_score(y_test_np, y_pred)
f1 = f1_score(y_test_np, y_pred)
f1weighted = f1_score(y_test_np, y_pred, average='weighted')
aupr = average_precision_score(y_test_np, y_pred_prob)
print("Evaluation complete! Printing Results...\n")
table = [['MCC', 'Precision', 'Recall', 'F1', 'F1weighted', 'AUPR'], [('%.2f' % round(mcc,2)), ('%.2f' % round(precisio$
print(tabulate(table))
