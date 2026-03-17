import numpy as np
from torch.utils.data import Dataset

ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'

PEP_LEN = 32
HLA_LEN = 200
ESM_DIM = 1152

class HLAIIDataset(Dataset):

    def __init__(self,data_list):
        #元素形状：（hla_name,peptide_seq,hla_seq,score,pep_esm,hla_esm）

        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self,idx):
        hla_name,pep_seq,hla_seq,score,pep_esm,hla_esm = self.data_list[idx]
        
        if len(pep_seq) > PEP_LEN:
            pep_seq = pep_seq[:PEP_LEN]
        else:
            pep_seq = pep_seq.ljust(PEP_LEN, '0')

        if len(hla_seq) > HLA_LEN:
            hla_seq = hla_seq[:HLA_LEN]
        else:
            hla_seq = hla_seq.ljust(HLA_LEN, '0')

        def to_one_hot(seq,length):
            indices = [ACIDS.index(c) if c in ACIDS else 1 for c in seq]
            oh = np.zeros((length,len(ACIDS)),dtype=np.float32)
            oh[np.arange(length),indices] = 1.0
            return oh
        
        pep_one_hot = to_one_hot(pep_seq,PEP_LEN)
        hla_one_hot = to_one_hot(hla_seq,HLA_LEN)

        return (
            pep_one_hot,
            pep_esm.astype(np.float32),
            hla_one_hot,
            hla_esm.astype(np.float32),
        ),np.float32(score)
