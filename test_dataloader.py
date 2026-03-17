# test_dataloader.py

import torch
from torch.utils.data import DataLoader
from src.data_utils  import get_hla_name_seq, get_data_mock
from src.datasets    import HLAIIDataset

HLA_SEQ_FILE = "data/hla_dict/hla_full_seq_dict.txt"
TRAIN_FILE   = "data/small/small_train.txt"

# 加载数据
hla_name_seq = get_hla_name_seq(HLA_SEQ_FILE)
data_list    = get_data_mock(hla_name_seq, TRAIN_FILE)
dataset      = HLAIIDataset(data_list)
loader       = DataLoader(dataset, batch_size=32, shuffle=True)

# 取第一个 batch，打印所有 shape
(pep_oh, pep_esm, hla_oh, hla_esm), labels = next(iter(loader))

print(f"data_list 长度   : {len(data_list)}")
print(f"pep_one_hot shape: {pep_oh.shape}")    # 期望 (32, 32, 22)
print(f"pep_esm     shape: {pep_esm.shape}")   # 期望 (32, 32, 1152)
print(f"hla_one_hot shape: {hla_oh.shape}")    # 期望 (32, 200, 22)
print(f"hla_esm     shape: {hla_esm.shape}")   # 期望 (32, 2, 100, 1152)
print(f"labels      shape: {labels.shape}")    # 期望 (32,)
print(f"labels dtype     : {labels.dtype}")    # 期望 torch.float32