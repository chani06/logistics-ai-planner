with open('test_final.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find trips containing Lopburi
print("=== Searching for Lopburi (ลพบุรี) ===")
for i, line in enumerate(lines):
    if 'ลพบุรี' in line:
        # Find the trip header above this line
        for j in range(i-1, max(0, i-15), -1):
            if 'ทริป' in lines[j] and '-' in lines[j]:
                print(f"Found in: {lines[j].strip()}")
                print(f"  Line {i}: {line.strip()}")
                break

print("\n=== Searching for Nakhon Ratchasima (นครราชสีมา) ===")
for i, line in enumerate(lines):
    if 'นครราชสีมา' in line:
        # Find the trip header above this line
        for j in range(i-1, max(0, i-15), -1):
            if 'ทริป' in lines[j] and '-' in lines[j]:
                print(f"Found in: {lines[j].strip()}")
                print(f"  Line {i}: {line.strip()}")
                break
