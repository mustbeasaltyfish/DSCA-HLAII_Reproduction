import os

# 启用 HuggingFace 下载进度条（在 tmux 中可见）
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"

import numpy as np
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

from src.data_utils import get_hla_name_seq

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

DATA_FILE = "data/medium/medium_train_100k.txt"
HLA_SEQ_FILE = "data/hla_dict/hla_full_seq_dict.txt"
PEP_LEN = 32
HLA_PART_LEN = 100
ESM_DIM = 1152

DATA_STEM = os.path.splitext(os.path.basename(DATA_FILE))[0]
PEP_OUT = f"data/pep/{DATA_STEM}_pep_esm.npy"
HLA_UNIQUE_OUT = f"data/hla/{DATA_STEM}_hla_unique_esm.npy"
HLA_UNIQUE_NAMES_OUT = f"data/hla/{DATA_STEM}_hla_unique_names.npy"

os.makedirs("data/pep", exist_ok=True)
os.makedirs("data/hla", exist_ok=True)


def temp_path(path):
    return f"{path}.tmp"


def remove_temp_file(path):
    tmp_path = temp_path(path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def save_npy_atomic(path, array):
    tmp_path = temp_path(path)
    remove_temp_file(path)
    with open(tmp_path, "wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(tmp_path, path)


def get_esm_embedding(client, seq, max_len):
    seq = seq[:max_len]
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True),
    )
    emb = logits_output.embeddings[0, 1:-1, :]
    if emb.shape[0] < max_len:
        pad = torch.zeros(max_len - emb.shape[0], emb.shape[1], device=emb.device)
        emb = torch.cat([emb, pad], dim=0)
    return emb.cpu().numpy().astype(np.float32)


def scan_dataset(data_file, hla_name_seq):
    num_samples = 0
    unique_hla_names = set()
    skipped_hla = 0

    with open(data_file) as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 4:
                continue

            _, alpha, beta, _ = parts
            hla_name = f"{alpha}-{beta}"
            num_samples += 1

            if hla_name in hla_name_seq:
                unique_hla_names.add(hla_name)
            else:
                skipped_hla += 1

    return num_samples, sorted(unique_hla_names), skipped_hla


def build_peptide_embeddings(client, data_file, num_samples, out_path):
    if os.path.exists(out_path):
        pep_embs = np.load(out_path, mmap_mode="r")
        print(f"肽段嵌入已存在，跳过计算: {out_path}, shape={pep_embs.shape}")
        return pep_embs.shape[0]
    tmp_path = temp_path(out_path)
    remove_temp_file(out_path)

    print("开始计算肽段 ESMC embedding...")
    pep_memmap = np.lib.format.open_memmap(
        tmp_path,
        mode="w+",
        dtype=np.float32,
        shape=(num_samples, PEP_LEN, ESM_DIM),
    )

    with torch.no_grad():
        idx = 0
        with open(data_file) as handle:
            for line in tqdm(handle, total=num_samples):
                parts = line.strip().split()
                if len(parts) != 4:
                    continue

                pep_seq = parts[0][:PEP_LEN].ljust(PEP_LEN, "X")
                pep_memmap[idx] = get_esm_embedding(client, pep_seq, PEP_LEN)
                idx += 1

    pep_memmap.flush()
    del pep_memmap
    os.replace(tmp_path, out_path)

    print(f"肽段嵌入已保存: {out_path}, shape=({num_samples}, {PEP_LEN}, {ESM_DIM})")
    return num_samples


def build_unique_hla_embeddings(client, unique_hla_names, hla_name_seq, out_path, names_out_path):
    if os.path.exists(out_path) and os.path.exists(names_out_path):
        hla_embs = np.load(out_path, mmap_mode="r")
        hla_names = np.load(names_out_path, allow_pickle=False)
        print(f"HLA 唯一嵌入已存在，跳过计算: {out_path}, shape={hla_embs.shape}")
        print(f"HLA 名称索引已存在: {names_out_path}, count={len(hla_names)}")
        return

    if not unique_hla_names:
        raise RuntimeError("未找到任何可用的 HLA 名称，无法生成唯一 HLA embedding")

    print("开始计算唯一 HLA ESMC embedding...")
    hla_embs = np.zeros((len(unique_hla_names), 2, HLA_PART_LEN, ESM_DIM), dtype=np.float32)

    with torch.no_grad():
        for idx, hla_name in enumerate(tqdm(unique_hla_names)):
            hla_seq = hla_name_seq[hla_name]
            part1 = hla_seq[:HLA_PART_LEN]
            part2 = hla_seq[HLA_PART_LEN:HLA_PART_LEN * 2]
            hla_embs[idx, 0] = get_esm_embedding(client, part1, HLA_PART_LEN)
            hla_embs[idx, 1] = get_esm_embedding(client, part2, HLA_PART_LEN)

    save_npy_atomic(out_path, hla_embs)
    save_npy_atomic(names_out_path, np.array(unique_hla_names))

    print(f"HLA 唯一嵌入已保存: {out_path}, shape={hla_embs.shape}")
    print(f"HLA 名称索引已保存: {names_out_path}, count={len(unique_hla_names)}")


def main():
    hla_name_seq = get_hla_name_seq(HLA_SEQ_FILE)
    num_samples, unique_hla_names, skipped_hla = scan_dataset(DATA_FILE, hla_name_seq)

    print(f"共 {num_samples} 条样本")
    print(f"唯一 HLA pair 数: {len(unique_hla_names)}")
    if skipped_hla:
        print(f"[警告] 有 {skipped_hla} 条样本找不到 HLA 序列，训练侧会跳过这些样本")

    print("加载 ESMC 600M...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    client = ESMC.from_pretrained("esmc_600m").to(device)
    client.eval()

    build_peptide_embeddings(client, DATA_FILE, num_samples, PEP_OUT)
    build_unique_hla_embeddings(client, unique_hla_names, hla_name_seq, HLA_UNIQUE_OUT, HLA_UNIQUE_NAMES_OUT)

    pep_gb = os.path.getsize(PEP_OUT) / 1e9 if os.path.exists(PEP_OUT) else 0
    hla_gb = os.path.getsize(HLA_UNIQUE_OUT) / 1e9 if os.path.exists(HLA_UNIQUE_OUT) else 0
    names_mb = os.path.getsize(HLA_UNIQUE_NAMES_OUT) / 1e6 if os.path.exists(HLA_UNIQUE_NAMES_OUT) else 0
    print(
        f"\n磁盘占用: 肽段 {pep_gb:.2f} GB + HLA {hla_gb:.4f} GB "
        f"+ 名称索引 {names_mb:.2f} MB = {pep_gb + hla_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()
