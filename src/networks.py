import torch
import torch.nn as nn
import torch.nn.functional as F



ACIDS_DIM = 22
ESM_DIM = 1152
D_MODEL = 384

import math 


class Self_Attention(nn.Module):
    """
    """
    def __init__(self,input_dim,dim_k,dim_v):
        super().__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1/math.sqrt(dim_k)

    def forward(self,x):
        Q = self.q(x)
        K = self.k(x)
        V = self.v(x)
        att = Q @ K.transpose(-2,-1) * self._norm_fact   #可疑点，原仓库把缩放放在了softmax之后
        att = F.softmax(att,dim=-1)
        out = att @ V
        return out

class ResConv1D(nn.Module):
    # 与上游仓库对齐：使用无 padding 的局部卷积

    def __init__(self,dim,kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(dim,dim,kernel_size)

    def forward(self,x):
        out = self.conv(x.transpose(1,2))
        return out.transpose(1,2)

class TransformerEncoder(nn.Module):
    
    def __init__(self,dim,depth,num_heads,mlp_ratio=4.0,dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=int(dim*mlp_ratio),
                    dropout=dropout,
                    batch_first=True,
                ) 
                for _ in range(depth)
            ]
        )

    def forward(self,x):
        for block in self.blocks:
            x = block(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.q = nn.Linear(dim,dim)
        self.k = nn.Linear(dim,dim)
        self.v = nn.Linear(dim,dim)
        self._norm_fact = 1/math.sqrt(dim)
    def forward(self,x_global,x_local,y_local):
        """
        核心设计
        x_global: 全局特征
        x_local: 局部特征
        y_local: 局部特征
        输出：
            out: 输出特征
        """
        Q = self.q(x_global)
        K = self.k(x_local)
        V = self.v(y_local)
        att = Q @ K.transpose(-2,-1) * self._norm_fact
        att = F.softmax(att,dim=-1)
        out = torch.einsum('bij,bkv->biv',att,V)
        out = F.normalize(out,p=2,dim=-1)
        return out+x_global

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return torch.max(x,dim=1)[0]

class ResidueEmbedding(nn.Module):
    """
    Stage 1: Residue-level Embedding
    
    输入：
        x_pep   : (B, 32,  22)      肽段 one-hot
        esm_pep : (B, 32,  1152)    肽段 ESMC 嵌入
        x_hla   : (B, 200, 22)      HLA  one-hot
        esm_hla : (B, 2,100, 1152) HLA ESMC 嵌入（两段分开）

    输出：
        encode_peptide : (B, 32,  384)   肽段融合特征
        encode_hla     : (B, 200, 384)   HLA  融合特征
    """

    def __init__(self):
        super().__init__()
        self.embed_seq = nn.Linear(ACIDS_DIM, 128)
        self.dense_esm_pep = nn.Linear(ESM_DIM, 128)
        self.dense_esm_hla = nn.Linear(ESM_DIM, 128)
        self.att = Self_Attention(128,128,128)

    def forward(self,x_pep,esm_pep,x_hla,esm_hla):
        #合并alpha链和beta链的ESMC嵌入
        esm_hla = torch.cat([esm_hla[:,0],esm_hla[:,1]],dim=1)

        #one-hot序列嵌入
        pep_seq_emb = self.embed_seq(x_pep)
        hla_seq_emb = self.embed_seq(x_hla)

        #ESMC嵌入
        pep_esm_emb = self.dense_esm_pep(esm_pep)
        hla_esm_emb = self.dense_esm_hla(esm_hla)

        #one-hot self-attention
        peptide_att = self.att(pep_seq_emb)
        hla_att = self.att(hla_seq_emb)

        #合并one-hot和ESMC嵌入
        pep_feature = torch.cat([pep_seq_emb,pep_esm_emb,peptide_att],dim=-1)
        hla_feature = torch.cat([hla_seq_emb,hla_esm_emb,hla_att],dim=-1)

        return pep_feature,hla_feature

class RepresentationExtractor(nn.Module):
    """
    Stage 2: Representation Extraction
    
    输入：
        feature_pep : (B, 32,  384)   肽段融合特征
        feature_hla : (B, 200, 384)   HLA  融合特征

    输出：
        pep_global : (B, 32,  384)   肽段全局特征
        hla_global : (B, 200, 384)   HLA  全局特征
        pep_local  : (B, 24,  384)   肽段局部特征
        hla_local  : (B, 196, 384)   HLA  局部特征
    """
    def __init__(self):
        super().__init__()
        self.pep_transformer = TransformerEncoder(dim=D_MODEL,depth=3,num_heads=4)
        self.hla_transformer = TransformerEncoder(dim=D_MODEL,depth=3,num_heads=4)
        self.pep_conv = ResConv1D(dim=D_MODEL,kernel_size=9)
        self.hla_conv = ResConv1D(dim=D_MODEL,kernel_size=5)
    
    def forward(self,feature_pep,feature_hla):
        pep_global = self.pep_transformer(feature_pep)
        hla_global = self.hla_transformer(feature_hla)
        pep_local = self.pep_conv(feature_pep)
        hla_local = self.hla_conv(feature_hla)
        return pep_global,hla_global,pep_local,hla_local

class DualStreamCrossAttention(nn.Module):
    """
    Stage 3: Dual Stream Cross Attention
    
    输入：
        pep_global : (B, 32,  384)   肽段全局特征
        hla_global : (B, 200, 384)   HLA  全局特征
        pep_local  : (B, 24,  384)   肽段局部特征
        hla_local  : (B, 196, 384)   HLA  局部特征
    """
    def __init__(self):
        super().__init__()
        self.cross_attn_pep = CrossAttention(dim=D_MODEL)
        self.cross_attn_hla = CrossAttention(dim=D_MODEL)
        self.conv_pep = ResConv1D(D_MODEL, kernel_size=9)
        self.conv_hla = ResConv1D(D_MODEL, kernel_size=5)

    def forward(self,pep_global,hla_global,pep_local,hla_local):
        pep_feature = self.cross_attn_pep(pep_global,pep_local,hla_local)
        hla_feature = self.cross_attn_hla(hla_global,hla_local,pep_local)
        pep_feature = self.conv_pep(pep_feature)
        hla_feature = self.conv_hla(hla_feature)
        return pep_feature,hla_feature

class Predictor(nn.Module):
    """
    Stage 4: Predictor
    
    输入：
        pep_global : (B, 32,  384)   肽段全局特征
        hla_global : (B, 200, 384)   HLA  全局特征
    输出：
        score : (B, 1)   评分
    """
    def __init__(self):
        super().__init__()
        self.global_max_pool = GlobalMaxPool1d()
        self.mlp = nn.Sequential(
            nn.Linear(768,  1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512),
        )
        self.output = nn.Linear(512,1)

    def forward(self,pep_feature,hla_feature):
        pep_vec = self.global_max_pool(pep_feature)            
        hla_vec = self.global_max_pool(hla_feature)            
        interaction = torch.cat([pep_vec, hla_vec], dim=-1)    
        interaction = self.mlp(interaction)                     
        predictions = torch.sigmoid(self.output(interaction))  
        return predictions  

class DSCA_HLAII(nn.Module):
    """
    完整模型组装
    输入：
        pep_one_hot : (B, 32,  22)      肽段 one-hot
        pep_esm     : (B, 32,  1152)    肽段 ESMC 嵌入
        hla_one_hot : (B, 200, 22)      HLA  one-hot
        hla_esm     : (B, 2, 100, 1152) HLA ESMC 嵌入（两段分开）
    输出：
        score : (B, 1)   评分
    """
    def __init__(self):
        super().__init__()
        self.residue_embedding = ResidueEmbedding()
        self.representation_extractor = RepresentationExtractor()
        self.dual_stream_cross_attention = DualStreamCrossAttention()
        self.predictor = Predictor()

    def forward(self,pep_one_hot,pep_esm,hla_one_hot,hla_esm,core=False):
        pep_feature,hla_feature = self.residue_embedding(pep_one_hot,pep_esm,hla_one_hot,hla_esm)
        pep_global,hla_global,pep_local,hla_local = self.representation_extractor(pep_feature,hla_feature)
        pep_feature,hla_feature = self.dual_stream_cross_attention(pep_global,hla_global,pep_local,hla_local)
        if core:
            return torch.mean(pep_feature, dim=-1)
        score = self.predictor(pep_feature,hla_feature)
        return score

if __name__ == '__main__':
    B = 4
    model = DSCA_HLAII()

    out = model(
        torch.zeros(B, 32,  22),
        torch.zeros(B, 32,  1152),
        torch.zeros(B, 200, 22),
        torch.zeros(B, 2, 100, 1152),
    )

    print(f"output shape : {out.shape}")   
    print(f"dtype        : {out.dtype}")   
    print(f"值域         : [{out.min():.3f}, {out.max():.3f}]")


    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量     : {total:,}")
    print(f"可训练参数量 : {trainable:,}")
