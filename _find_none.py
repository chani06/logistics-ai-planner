with open(r'c:\Users\chani\app\app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

results = []
for i, line in enumerate(lines[:500]):
    s = line.rstrip()
    if not s:
        continue
    if s.startswith(' ') or s.startswith('\t'):
        continue
    skip = ['#', 'import ', 'from ', 'def ', 'class ', 'if ', 'try:', 'except', 'else:', 'elif ', '@', 'with ', 'for ', 'while ']
    if any(s.startswith(k) for k in skip):
        continue
    results.append(f'L{i+1}: {s[:120]}')

print('\n'.join(results))
