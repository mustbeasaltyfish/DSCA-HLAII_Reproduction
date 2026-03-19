from collections import defaultdict

cnt_pos = defaultdict(int)
cnt_neg = defaultdict(int)

with open("data/benchmark.txt") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        pep, alpha, beta, score = parts
        allele = f"{alpha}-{beta}"
        if score == '1':
            cnt_pos[allele] += 1
        else:
            cnt_neg[allele] += 1

for allele in sorted(cnt_pos.keys() | cnt_neg.keys()):
    p = cnt_pos[allele]
    n = cnt_neg[allele]
    if p == 0:
        continue
    print(allele, p, n, round(n / p, 3))