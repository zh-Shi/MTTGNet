"""
MTTGNet v2 Training Script
===========================
Per-sample temporal graphs, multi-step prediction, DataLoader batching.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import random
import matplotlib.pyplot as plt

import utils
import net

# ==================== Configuration ====================

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(233)

# Device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Hyperparameters
l_x = 60            # lookback window (60 steps = 15 days at 6-hourly)
l_y = 4             # prediction horizon (4 steps = 24 hours)
hidden_dim = 64
lr = 0.001
weight_decay = 0.001
epochs = 3000
batch_size = 128
ratio_train = 0.9
aggregator_type = 'softmax'  # ['softmax', 'linear', 'mean']
cut_co2 = False
start_year = 1980    # data starts from 1980
PERIOD = 365 * 4     # samples per year (6-hourly)

# ==================== Data Loading ====================

print("Loading data...")
x_0 = np.load('./data/ERA5_global/ERA5_2m_temperature_global_mean_anomaly.npy')
data = np.load('./data/ERA5_global/data_mixed_6hourly_std.npy')
x_0 = np.vstack((x_0, data))  # [8, N_total]
num_total = x_0.shape[-1]
num_train = int(ratio_train * num_total)
feature = x_0.shape[0]

# CO2 cutting (sensitivity analysis)
if cut_co2:
    x_0[6, :] = np.concatenate((
        x_0[6, :num_train - 1000],
        2 * x_0[6, num_train - 1000] - x_0[6, num_train - 1000:]
    ))
    print("CO2 cutting enabled.")

# Split
data_train, data_test = x_0[:, :num_train], x_0[:, num_train:num_total]
print(f"Total time steps: {num_total}, Train: {num_train}, Test: {num_total - num_train}")

# Create sequences
start_time = time.time()
x_train, y_train = utils.create_inout_sequences(data_train, l_x, l_y)
x_test, y_test = utils.create_inout_sequences(data_test, l_x, l_y)
print(f'Sequence creation time: {time.time() - start_time:.2f}s')
print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test:  {x_test.shape}, y_test:  {y_test.shape}")

# Create DOY indices
# The first time step of x_train[0] corresponds to t=0 in the raw data
# DOY of a time step = (global_index) % PERIOD
doy_train = utils.create_doy_indices(x_train.shape[0], l_x, start_doy=0, period=PERIOD)
doy_test = utils.create_doy_indices(x_test.shape[0], l_x, start_doy=num_train, period=PERIOD)
print(f"doy_train: {doy_train.shape}, doy_test: {doy_test.shape}")

# Convert to tensors
x_train_t = torch.from_numpy(x_train).float()
y_train_t = torch.from_numpy(y_train).float()
x_test_t = torch.from_numpy(x_test).float()
y_test_t = torch.from_numpy(y_test).float()
doy_train_t = torch.from_numpy(doy_train).long()
doy_test_t = torch.from_numpy(doy_test).long()

# DataLoaders
train_dataset = TensorDataset(x_train_t, y_train_t, doy_train_t)
test_dataset = TensorDataset(x_test_t, y_test_t, doy_test_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==================== Model ====================

model = net.MTTGNetv2(
    num_vars=feature,
    hidden_dim=hidden_dim,
    seq_len=l_x,
    pred_len=l_y,
    aggregator_type=aggregator_type,
).to(device)

criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: MTTGNetv2")
print(f"  Total params: {total_params:,}")
print(f"  Trainable:    {trainable_params:,}")
print(f"  Aggregator:   {aggregator_type}")
print(f"  Pred horizon: {l_y} steps ({l_y * 6} hours)")
print(f"  Batch size:   {batch_size}")

# ==================== Training ====================

# Tracking
para_trainloss = np.zeros(epochs)
para_testloss = np.zeros(epochs)
para_r2_train = np.zeros(epochs)
para_r2_test = np.zeros(epochs)
para_rmse = np.zeros(epochs)
best_rmse = float('inf')
best_epoch = 0

os.makedirs('result_new', exist_ok=True)

print("\nStarting training...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    all_train_preds, all_train_trues = [], []

    for x_batch, y_batch, doy_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        doy_batch = doy_batch.to(device)

        optimizer.zero_grad()
        output = model(x_batch, doy_batch)  # [B, pred_len]
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * x_batch.size(0)
        all_train_preds.append(output.detach().cpu().numpy())
        all_train_trues.append(y_batch.detach().cpu().numpy())

    avg_train_loss = total_train_loss / len(train_dataset)
    para_trainloss[epoch] = avg_train_loss

    # Evaluation on test set
    model.eval()
    total_test_loss = 0.0
    all_test_preds, all_test_trues = [], []

    with torch.no_grad():
        for x_batch, y_batch, doy_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            doy_batch = doy_batch.to(device)

            output = model(x_batch, doy_batch)
            loss = criterion(output, y_batch)

            total_test_loss += loss.item() * x_batch.size(0)
            all_test_preds.append(output.cpu().numpy())
            all_test_trues.append(y_batch.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_dataset)
    para_testloss[epoch] = avg_test_loss

    # Concatenate all batches
    train_preds = np.concatenate(all_train_preds, axis=0)  # [N_train, pred_len]
    train_trues = np.concatenate(all_train_trues, axis=0)
    test_preds = np.concatenate(all_test_preds, axis=0)    # [N_test, pred_len]
    test_trues = np.concatenate(all_test_trues, axis=0)

    # Metrics on first prediction step (closest to original single-step evaluation)
    r2_train = utils.get_r2_score(train_preds[:, 0], train_trues[:, 0], axis=0)
    r2_test = utils.get_r2_score(test_preds[:, 0], test_trues[:, 0], axis=0)
    rmse_train = utils.get_rmse(train_preds[:, 0], train_trues[:, 0])

    para_r2_train[epoch] = r2_train
    para_r2_test[epoch] = r2_test
    para_rmse[epoch] = rmse_train

    # Save best model (based on test RMSE of step 1)
    test_rmse_step1 = utils.get_rmse(test_preds[:, 0], test_trues[:, 0])
    if test_rmse_step1 < best_rmse:
        best_rmse = test_rmse_step1
        best_epoch = epoch
        torch.save(model.state_dict(), 'model_v2.pth')
        best_test_preds = test_preds.copy()
        best_test_trues = test_trues.copy()

    # Logging
    if (epoch + 1) % 100 == 0:
        epoch_time = time.time() - start_time
        print(f'Epoch {epoch+1:05d}/{epochs} | '
              f'Train Loss: {avg_train_loss:.6f} | '
              f'Test Loss: {avg_test_loss:.6f} | '
              f'R² Train: {r2_train:.6f} | '
              f'R² Test: {r2_test:.6f} | '
              f'RMSE Train: {rmse_train:.6f} | '
              f'Best RMSE Test: {best_rmse:.6f} @ epoch {best_epoch+1} | '
              f'Time: {epoch_time:.1f}s')

# ==================== Results ====================

total_time = time.time() - start_time
print(f"\n{'='*60}")
print(f"Training complete. Total time: {total_time:.1f}s")
print(f"Best model: epoch {best_epoch+1}")
print(f"Best Test RMSE (step 1): {best_rmse:.6f}")

# Per-step metrics
print(f"\nPer-step test RMSE:")
for s in range(l_y):
    step_rmse = utils.get_rmse(best_test_preds[:, s], best_test_trues[:, s])
    step_r2 = utils.get_r2_score(best_test_preds[:, s], best_test_trues[:, s], axis=0)
    print(f"  Step {s+1} ({6*(s+1)}h ahead): RMSE={step_rmse:.6f}, R²={step_r2:.6f}")

# Save results
prefix = 'co2_' if cut_co2 else ''
np.save(f'result_new/{prefix}test_predict_v2.npy', best_test_preds)
np.save(f'result_new/{prefix}test_true_v2.npy', best_test_trues)
np.save(f'result_new/{prefix}train_loss_v2.npy', para_trainloss)
np.save(f'result_new/{prefix}test_loss_v2.npy', para_testloss)
np.save(f'result_new/{prefix}r2_test_v2.npy', para_r2_test)
np.save(f'result_new/{prefix}rmse_v2.npy', para_rmse)

# ==================== Visualization ====================

# Plot predictions vs true (step 1)
l = len(best_test_preds)
plt.figure(figsize=(60, 10))
plt.plot(best_test_preds[:l, 0], c="orangered", label="MTTGNetv2 Predict", alpha=0.7)
plt.plot(best_test_trues[:l, 0], c="darkblue", label="True", alpha=0.6)
date_labels = ['2017_03', '2018_07', '2019_12', '2021_04', '2022_08', '2023_12']
plt.xticks(np.linspace(0, l-1, 6).astype(int), date_labels, fontsize=25)
plt.xlabel("Date", fontsize=50)
plt.ylabel("t2m anomaly (°C)", fontsize=50)
plt.legend(fontsize=30)
plt.title("MTTGNet v2 — GSTA Prediction (6h ahead)", fontsize=40)
plt.tight_layout()
plt.savefig(f'result_new/{prefix}predict_v2.png')
plt.show()

# Training curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0, 0].plot(para_trainloss, alpha=0.7)
axes[0, 0].set_title('Train Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 1].plot(para_testloss, alpha=0.7, c='orange')
axes[0, 1].set_title('Test Loss')
axes[0, 1].set_xlabel('Epoch')
axes[1, 0].plot(para_r2_test, alpha=0.7, c='green')
axes[1, 0].set_title('Test R²')
axes[1, 0].set_xlabel('Epoch')
axes[1, 1].plot(para_rmse, alpha=0.7, c='red')
axes[1, 1].set_title('Train RMSE')
axes[1, 1].set_xlabel('Epoch')
plt.tight_layout()
plt.savefig(f'result_new/{prefix}training_curves_v2.png')
plt.show()

print("\nDone!")
