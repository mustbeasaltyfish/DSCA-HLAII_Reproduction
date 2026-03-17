import numpy as np

ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'
PEP_LEN = 32
ESM_DIM = 1152


def get_hla_name_seq(hla_seq_file):
    
    #两种方式查询实现:单链和双链

    hla_dict = {}
    with open(hla_seq_file) as f:
        for line in f:
            parts = line.strip().split('\t',1)
            if len(parts) ==2:
                name,seq = parts
                hla_dict[name] = seq
    
    class HLASeqGetter:
        def __getitem__(self,name):
            if "-" in name:
                alpha, beta = name.split('-',1)
                return hla_dict[alpha] + hla_dict[beta]
            return hla_dict[name]

        def __contains__(self,name):
            if '-' in name:
                alpha, beta = name.split('-', 1)
                return alpha in hla_dict and beta in hla_dict
            return name in hla_dict

    return HLASeqGetter()

def get_data_mock(hla_name_seq,data_file):
    """
    读取训练/验证数据文件，返回 data_list。
    ESMC 嵌入用零向量占位，确认数据流 shape 正确后再替换为真实嵌入。
    
    返回：list of tuples:
        (hla_name, pep_seq, hla_seq, score, pep_esm, hla_esm)
        
    shape 说明：
        pep_esm : np.zeros((PEP_LEN, ESM_DIM))        -> (32, 1152)
        hla_esm : np.zeros((2, 100, ESM_DIM))         -> (2, 100, 1152)
    """
    data_list = []
    skipped = 0

    with open(data_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue

            pep_seq,alpha,beta,score =parts
            hla_name = f"{alpha}-{beta}"


            if hla_name not in hla_name_seq:
                skipped += 1
                continue

            hla_seq = hla_name_seq[hla_name]

            pep_esm = np.zeros((PEP_LEN, ESM_DIM),  dtype=np.float32)
            hla_esm = np.zeros((2, 100, ESM_DIM),   dtype=np.float32)

            data_list.append((
                hla_name,
                pep_seq,
                hla_seq,
                float(score),
                pep_esm,
                hla_esm,
            ))

    if skipped:
         print(f"[警告] 跳过 {skipped} 条找不到 HLA 序列的样本")
    return data_list






