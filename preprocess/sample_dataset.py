from collections import defaultdict
import random
import os 

random.seed(42)

INPUT = "data/benchmark.txt"
OUTPUT = "data/medium/medium_train_100k.txt"
MAX_LEN = 100000

os.makedirs("data/medium", exist_ok=True)

allele_pos = defaultdict(list)
allele_neg = defaultdict(list)

with open(INPUT) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 4:
            continue 
        pep,alpha,beta,score = parts
        allele = f"{alpha}-{beta}"
        if score =='1':
            allele_pos[allele].append(line)
        else:
            allele_neg[allele].append(line)
#首先做格式处理，获得所有正/负样本

valid_alleles = set(allele_pos.keys()) | set(allele_neg.keys())
sorted_alleles = sorted(valid_alleles, key=lambda x: len(allele_pos[x]), reverse=True)



selected_alleles = []
sampled = []
count = 0
for allele in sorted_alleles:
    pos = allele_pos[allele]
    neg = allele_neg[allele]
    sampled.extend(pos+neg)
    count += len(pos+neg)
    selected_alleles.append(allele)
    if count >= MAX_LEN:
        break

print("选中的等位基因：")
for allele in selected_alleles:
    print(f"{allele}: {len(allele_pos[allele])} pos, {len(allele_neg[allele])} neg")

random.shuffle(sampled)

#抽取前18000条数据，正负比例为1:5
with open(OUTPUT, "w") as f:
    f.writelines(sampled)

n_pos = sum(1 for l in sampled if l.split()[3] == '1')
n_neg= len(sampled) - n_pos
print(f"\n输出到{OUTPUT}，正样本{n_pos}条，负样本{n_neg}条，比例{n_pos/n_neg:.1f}")



