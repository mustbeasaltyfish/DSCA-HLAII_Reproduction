import os
# 启用 HuggingFace 下载进度条（在 tmux 中可见）
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig



from src.networks       import DSCA_HLAII
from src.datasets       import HLAIIDataset
from src.data_utils     import get_hla_name_seq, get_data_mock
from src.early_stopping import EarlyStopping

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
login(token=os.getenv('HF_TOKEN'))   

DATA_FILE = 'data/medium/medium_train_100k.txt'
HLA_SEQ_FILE = 'data/hla_dict/hla_full_seq_dict.txt'
PEP_OUT = "data/pep/medium_train_100k_pep_esm.npy"
HLA_OUT = "data/hla/medium_train_100k_hla_esm.npy"
PEP_LEN = 32
HLA_PART_LEN = 100

os.makedirs("data/pep", exist_ok=True)
os.makedirs("data/hla", exist_ok=True)


hla_dic = {}

with open(HLA_SEQ_FILE) as f:
    for line in f:
        parts = line.strip().split('\t',1)
        if len(parts) == 2:
            name,seq = parts
            hla_dic[name] = seq

peptides,hla_seqs = [],[]

with open(DATA_FILE) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        pep,alpha,beta,_ = parts
        pep = pep[:PEP_LEN].ljust(PEP_LEN, 'X')
        peptides.append(pep)
        hla_seq = hla_dic.get(alpha,'')+hla_dic.get(beta,'')
        hla_seqs.append(hla_seq)

N = len(peptides)
print(f"共{N}条样本")

print('加载 ESMC 600M...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
client = ESMC.from_pretrained('esmc_600m').to(device)
client.eval()

def get_esm_embedding(seq, max_len):
    seq = seq[:max_len]
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True)
    )
    emb = logits_output.embeddings[0,1:-1,:]
    if emb.shape[0] < max_len:
        pad = torch.zeros(max_len-emb.shape[0],emb.shape[1],device=emb.device)
        emb = torch.cat([emb,pad],dim = 0)
    return emb.cpu().numpy().astype(np.float32)


if os.path.exists(PEP_OUT):
    print(f"肽段嵌入已存在，跳过计算")
else:
    print("开始计算肽段 ESMC embedding...")
    pep_embs = np.zeros((N,PEP_LEN,1152),dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(N)):
            pep_embs[i] = get_esm_embedding(peptides[i], PEP_LEN)
    np.save(PEP_OUT, pep_embs)
    print(f'肽段嵌入已保存: {PEP_OUT}, shape={pep_embs.shape}')


if os.path.exists(HLA_OUT):
    print(f"HLA嵌入已存在，跳过计算")
else:
    print("开始计算HLA ESMC embedding...")
    hla_embs = np.zeros((N, 2, HLA_PART_LEN, 1152), dtype=np.float32)
    with torch.no_grad():
            for i in tqdm(range(N)):
                seq = hla_seqs[i]
                part1 = seq[:HLA_PART_LEN]
                part2 = seq[HLA_PART_LEN:HLA_PART_LEN*2]
                hla_embs[i, 0] = get_esm_embedding(part1, HLA_PART_LEN)
                hla_embs[i, 1] = get_esm_embedding(part2, HLA_PART_LEN)
    np.save(HLA_OUT, hla_embs)
    print(f'HLA 嵌入已保存: {HLA_OUT}, shape={hla_embs.shape}')


pep_gb = os.path.getsize(PEP_OUT) / 1e9 if os.path.exists(PEP_OUT) else 0
hla_gb = os.path.getsize(HLA_OUT) / 1e9 if os.path.exists(HLA_OUT) else 0
print(f'\n磁盘占用: 肽段 {pep_gb:.2f} GB + HLA {hla_gb:.2f} GB = {pep_gb+hla_gb:.2f} GB')