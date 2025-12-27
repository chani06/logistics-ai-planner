import json

with open('branch_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# ค้นหาสาขาที่มีคำว่า ฟิวเจอร์รังสิต
print('🔍 ค้นหาสาขา "ฟิวเจอร์รังสิต":')
print('='*60)

found = []
for code, branch in data.items():
    name = branch.get('สาขา', '')
    if 'ฟิวเจอร์' in name and 'รังสิต' in name:
        lat = branch.get('ละ', '')
        lon = branch.get('ลอง', '')
        province = branch.get('จังหวัด', '')
        district = branch.get('อำเภอ', '')
        found.append({
            'code': code,
            'name': name,
            'lat': lat,
            'lon': lon,
            'province': province,
            'district': district
        })

for b in found:
    print(f"รหัส: {b['code']}")
    print(f"ชื่อ: {b['name']}")
    print(f"พิกัด: {b['lat']}, {b['lon']}")
    print(f"ที่อยู่: {b['district']}, {b['province']}")
    print('-'*40)

# ดูว่าอยู่กลุ่มเดียวกันหรือไม่
if found:
    print(f'\n📊 สรุป: พบ {len(found)} สาขา')
    coords = set()
    for b in found:
        try:
            key = f"{float(b['lat']):.4f}_{float(b['lon']):.4f}"
            coords.add(key)
        except:
            pass
    if len(coords) == 1:
        print('✅ ทุกสาขาอยู่จุดส่งเดียวกัน (พิกัดเดียวกัน)')
    else:
        print(f'⚠️ มี {len(coords)} จุดส่งที่แตกต่างกัน')
        for c in coords:
            print(f'   - พิกัด: {c}')
