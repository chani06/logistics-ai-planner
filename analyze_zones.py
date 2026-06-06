import json, sys, re
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

with open('branch_data.json', encoding='utf-8') as f:
    data = json.load(f)
with open('app.py', encoding='utf-8') as f:
    src = f.read()

branches = list(data.values())
prov_dist = defaultdict(lambda: defaultdict(int))
for b in branches:
    p = b.get('จังหวัด','')
    d = b.get('อำเภอ','')
    if p and d:
        prov_dist[p][d] += 1

# หาจังหวัดที่มีโซนย่อยเฉพาะอยู่แล้ว
covered = set()
for line in src.split('\n'):
    if "'provinces':" in line:
        for p in re.findall(r"'([ก-๙a-zA-Z\s]+)'", line):
            if len(p) > 2:
                covered.add(p.strip())

print('=== จังหวัดที่ต้องเพิ่มโซนย่อย ===')
for prov, dists in sorted(prov_dist.items(), key=lambda x: -sum(x[1].values())):
    total = sum(dists.values())
    if total < 15:
        continue
    # เช็คว่ามีโซนย่อยเฉพาะสำหรับจังหวัดนี้หรือยัง
    zone_count = src.count(f"'{prov}'")
    has_subzone = any(f'_{prov}_' in z for z in re.findall(r"'ZONE_[^']+'", src))
    if not has_subzone:
        print(f'\n{prov} ({total} สาขา) [zone_refs={zone_count}]:')
        for d,c in sorted(dists.items(), key=lambda x: -x[1])[:10]:
            print(f'  {d}: {c}')
