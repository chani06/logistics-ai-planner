import re

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

suspects = []
row_vars = r'(farthest_row|_gc_r|_sr|_sfp_r2|_tr_i|_nb_r|_sfp_r|_r2|_fur|_eprow|_ff_row|_pms_row|_sfp_r2)'

for i, l in enumerate(lines[4200:5800], start=4201):
    s = l.strip()
    # Pattern: rowvar['col'] or ...
    if re.search(row_vars + r"\['[^']+'\]\s+or", s):
        suspects.append((i, s[:120]))
    # Pattern: float(rowvar['col'] or 0)
    if re.search(r'float\(' + row_vars + r"\['[^']+'\]\s+or", s):
        suspects.append((i, s[:120]))
    # Pattern: int(rowvar['col']
    if re.search(row_vars + r"\['[^']+'\](?!\s*\[)", s) and ' or ' in s:
        suspects.append((i, s[:120]))

# deduplicate
seen = set()
for ln, s in suspects:
    if ln not in seen:
        seen.add(ln)
        print(f'{ln}: {s}')

print(f'\nTotal: {len(seen)}')
