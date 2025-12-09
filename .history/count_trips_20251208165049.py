with open('test_final.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

in_summary = False
trip_count = 0
for line in lines:
    if 'SUMMARY' in line and '===' in line:
        in_summary = True
    elif in_summary and line.strip() and line.strip()[0].isdigit():
        trip_count += 1

print(f"Total trips: {trip_count}")

# Show trips 60-70
print("\nTrips 60-70 from summary:")
in_summary = False
count = 0
for line in lines:
    if 'SUMMARY' in line and '===' in line:
        in_summary = True
    elif in_summary and line.strip() and line.strip()[0].isdigit():
        count += 1
        if 60 <= count <= 70:
            print(line.strip())
