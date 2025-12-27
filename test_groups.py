import json

with open('branch_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'✅ โหลดข้อมูล: {len(data)} สาขา')

# จัดกลุ่มตามพิกัด (ทศนิยม 2 ตำแหน่ง = ~1.1 กม. หรือใช้ haversine)
# ใช้ระยะ 200 เมตร เพื่อรวมสาขาในห้างเดียวกัน
import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # เมตร
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# เก็บข้อมูลสาขาทั้งหมด
branches = []
for code, branch in data.items():
    try:
        lat = float(branch.get('ละ', 0))
        lon = float(branch.get('ลอง', 0))
        if lat == 0 or lon == 0: continue
        branches.append({
            'code': code,
            'name': branch.get('สาขา', ''),
            'province': branch.get('จังหวัด', ''),
            'district': branch.get('อำเภอ', ''),
            'lat': lat,
            'lon': lon
        })
    except: continue

# จัดกลุ่มด้วยระยะ 200 เมตร
MAX_DISTANCE = 200  # เมตร
groups = {}
assigned = set()

for i, b1 in enumerate(branches):
    if b1['code'] in assigned:
        continue
    
    # สร้างกลุ่มใหม่
    key = f"{b1['lat']:.3f}_{b1['lon']:.3f}"
    group = [b1]
    assigned.add(b1['code'])
    
    # หาสาขาอื่นที่อยู่ใกล้
    for j, b2 in enumerate(branches):
        if b2['code'] in assigned:
            continue
        dist = haversine(b1['lat'], b1['lon'], b2['lat'], b2['lon'])
        if dist <= MAX_DISTANCE:
            group.append(b2)
            assigned.add(b2['code'])
    
    if len(group) > 1:
        groups[key] = group

multi_groups = groups
print(f'\n📊 พบ {len(multi_groups)} กลุ่มที่มีหลายสาขา (จุดส่งเดียวกัน ≤{MAX_DISTANCE} เมตร)')

for i, (key, branches_list) in enumerate(list(multi_groups.items())[:10]):
    print(f'\n🔗 กลุ่ม {i+1}:')
    for b in branches_list:
        print(f"   - {b['code']}: {b['name']} ({b['district']}, {b['province']})")

# บันทึกผลลัพธ์
group_to_branches = {}  # group_id -> [codes]
branch_to_group = {}    # code -> group_id

group_id = 1
for key, branches_list in multi_groups.items():
    codes = [b['code'] for b in branches_list]
    gid = f"G{group_id:04d}"
    group_to_branches[gid] = codes
    for c in codes:
        branch_to_group[c] = gid
    group_id += 1

# บันทึกไฟล์
with open('branch_groups.json', 'w', encoding='utf-8') as f:
    json.dump({
        'groups': group_to_branches,
        'branch_to_group': branch_to_group,
        'total_groups': len(group_to_branches),
        'total_branches_in_groups': len(branch_to_group),
        'max_distance_meters': MAX_DISTANCE
    }, f, ensure_ascii=False, indent=2)
    
print(f'\n✅ บันทึก branch_groups.json เรียบร้อย')
print(f'   - {len(group_to_branches)} กลุ่ม')
print(f'   - {len(branch_to_group)} สาขาที่อยู่ในกลุ่ม')
