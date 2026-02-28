"""สร้าง distance_cache.json ใหม่จาก branch_clusters.json"""
import json

print("Creating new distance_cache.json from branch_clusters.json...")

# โหลด branch_clusters
with open('branch_clusters.json', 'r', encoding='utf-8') as f:
    clusters = json.load(f)

branch_info = clusters.get('branch_info', {})
nearby_branches = clusters.get('nearby_branches', {})

print(f"Branch info: {len(branch_info)}")
print(f"Nearby branches: {len(nearby_branches)}")

# สร้าง distance cache ใหม่
distance_cache = {}

# DC coordinates
DC_LAT, DC_LON = 14.179394, 100.648149

# เพิ่มระยะทาง DC -> สาขา
for code, info in branch_info.items():
    if 'distance_from_dc' in info and 'lat' in info and 'lon' in info:
        key = f"{DC_LAT:.4f},{DC_LON:.4f}_{info['lat']:.4f},{info['lon']:.4f}"
        distance_cache[key] = info['distance_from_dc']

print(f"Added {len(distance_cache)} DC->branch distances")

# บันทึกเป็น JSON
with open('distance_cache.json', 'w', encoding='utf-8') as f:
    json.dump(distance_cache, f, ensure_ascii=False, indent=2)

print(f"Saved distance_cache.json with {len(distance_cache)} items")
