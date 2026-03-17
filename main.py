import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score

from src.networks       import DSCA_HLAII
from src.datasets       import HLAIIDataset
from src.data_utils     import get_hla_name_seq, get_data_mock
from src.early_stopping import EarlyStopping

HLA_SEQ_FILE     = 'data/hla_dict/hla_full_seq_dict.txt'
TRAIN_FILE       = 'data/small/small_train.txt'
CHECKPOINT_PATH  = 'checkpoints/checkpoint.pt'

BATCH_SIZE   = 16
NUM_EPOCHS   = 1
LR           = 0.01
WEIGHT_DECAY = 1e-4
VALID_RATIO  = 0.1
PATIENCE     = 7

os.makedirs('checkpoints', exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')

hla_name_seq = get_hla_name_seq(HLA_SEQ_FILE)
data_list = get_data_mock(hla_name_seq, TRAIN_FILE)

dataset = HLAIIDataset(data_list)
n_valid = int(len(dataset) * VALID_RATIO)
n_train = len(dataset) - n_valid
train_dataset, valid_dataset = random_split(dataset, [n_train, n_valid], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = DSCA_HLAII().to(device)
optimizer = torch.optim.Adadelta(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.BCELoss()

start_epoch = 0
if os.path.exists(CHECKPOINT_PATH):
    print(f'发现 checkpoint，从 {CHECKPOINT_PATH} 恢复...')
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt['epoch'] + 1
    print(f'从 epoch {start_epoch} 继续训练，上次验证 loss: {ckpt["val_loss"]:.4f}')
else:
    print('未发现 checkpoint，从头开始训练')

early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, checkpoint_path=CHECKPOINT_PATH)


for epoch in range(start_epoch, NUM_EPOCHS):
    #训练
    model.train()
    train_loss = 0.0
    for (pep_oh, pep_esm, hla_oh, hla_esm), labels in train_loader:
        pep_oh  = pep_oh.to(device)
        pep_esm = pep_esm.to(device)
        hla_oh  = hla_oh.to(device)
        hla_esm = hla_esm.to(device)
        labels  = labels.unsqueeze(1).to(device)

        optimizer.zero_grad()
        preds = model(pep_oh, pep_esm, hla_oh, hla_esm)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(labels)

    train_loss /= n_train

    #验证
    model.eval()
    valid_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for (pep_oh, pep_esm, hla_oh, hla_esm), labels in valid_loader:
            pep_oh  = pep_oh.to(device)
            pep_esm = pep_esm.to(device)
            hla_oh  = hla_oh.to(device)
            hla_esm = hla_esm.to(device)
            labels  = labels.unsqueeze(1).to(device)

            preds = model(pep_oh, pep_esm, hla_oh, hla_esm)
            loss  = criterion(preds, labels)
            valid_loss += loss.item() * len(labels)
            all_preds.extend(preds.squeeze(1).cpu().numpy())
            all_labels.extend(labels.squeeze(1).cpu().numpy())

    valid_loss /= n_valid
    auroc = roc_auc_score(all_labels, all_preds)

    print(f'Epoch {epoch+1:3d}/{NUM_EPOCHS} | '
          f'train_loss: {train_loss:.4f} | '
          f'valid_loss: {valid_loss:.4f} | '
          f'AUROC: {auroc:.4f}')

    early_stopping(valid_loss, model, optimizer, epoch)
    if early_stopping.early_stop:
        print(f'Early stopping 触发，共训练 {epoch+1} 个 epoch')
        break

print('训练完成')
