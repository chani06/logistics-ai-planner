import re
src = open('app.py', encoding='utf-8').read()
lines = src.splitlines()
patterns = [
    r"row\['[^']+'\]\s+or\b",
    r'row\["[^"]+"\]\s+or\b',
    r"float\(row\['[^']+'\]\s+or",
    r'float\(row\["[^"]+"\]\s+or',
    r"_ff_row\['[^']+'\]\s+or",
    r'_ff_row\["[^"]+"\]\s+or',
]
for i, line in enumerate(lines, 1):
    s = line.strip()
    for p in patterns:
        if re.search(p, s):
            print(f'{i}: {s}')
            break
