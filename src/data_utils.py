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


def get_data_real(hla_name_seq, data_file, pep_esm_file, hla_esm_file, hla_esm_names_file=None):
    """
    读取真实 ESMC 预计算嵌入，返回 data_list。
    
    pep_esm_file : .npy, shape (N, 32,  1152)
    hla_esm_file : .npy, shape (N, 2, 100, 1152) 或 (M, 2, 100, 1152)
    hla_esm_names_file : 可选，若提供则 hla_esm_file 按唯一 hla_name 存储
    """
    print(f'加载 ESMC 嵌入...')
    pep_esm_all = np.load(pep_esm_file, mmap_mode='r')  # mmap 避免一次性占用全部内存
    print(f'  pep_esm: {pep_esm_all.shape}')

    hla_esm_all = np.load(hla_esm_file, mmap_mode='r')
    if hla_esm_names_file is None:
        hla_esm_index = None
        print(f'  hla_esm(sample-wise): {hla_esm_all.shape}')
    else:
        hla_esm_names = np.load(hla_esm_names_file, allow_pickle=False)
        hla_esm_index = {str(name): idx for idx, name in enumerate(hla_esm_names.tolist())}
        print(f'  hla_esm(unique): {hla_esm_all.shape}')
        print(f'  hla_esm_names: {hla_esm_names.shape}')

    data_list = []
    skipped   = 0
    idx       = 0

    with open(data_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            pep_seq, alpha, beta, score = parts
            hla_name = f"{alpha}-{beta}"

            if hla_name not in hla_name_seq:
                skipped += 1
                idx += 1
                continue

            hla_seq = hla_name_seq[hla_name]
            if hla_esm_index is None:
                hla_esm = hla_esm_all[idx]
            else:
                if hla_name not in hla_esm_index:
                    raise KeyError(f'HLA 唯一嵌入缺少 {hla_name}，请重新生成 {hla_esm_file} / {hla_esm_names_file}')
                hla_esm = hla_esm_all[hla_esm_index[hla_name]]

            data_list.append((
                hla_name,
                pep_seq,
                hla_seq,
                float(score),
                pep_esm_all[idx],   # (32,  1152)
                hla_esm,            # (2, 100, 1152)
            ))
            idx += 1

    if skipped:
        print(f'[警告] 跳过 {skipped} 条找不到 HLA 序列的样本')

    print(f'加载完成，共 {len(data_list)} 条')
    return data_list


