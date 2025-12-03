"""
ตรวจสอบข้อมูลจังหวัดของสาขาที่ระบุ
"""

import pickle

# โหลด model
with open('models/decision_tree_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

branch_info = model_data['branch_info']

# สาขาที่ต้องการเช็ค
check_branches = [
    'CW Future Rangsit Triangle 1 Fl.',
    'CW Future Rangsit Triangle 2 Fl',
    'FC470',
    'FC ฟิวเจอร์ปาร์ครังสิต',
]

print('='*80)
print('ตรวจสอบข้อมูลจังหวัดของสาขา')
print('='*80)

# ค้นหาทุก code ที่มีคำว่า "Future", "Rangsit", "รังสิต", "FC"
keywords = ['FUTURE', 'RANGSIT', 'รังสิต', 'FC470', 'FC ']

found_branches = []
for code, info in branch_info.items():
    name = info.get('name', '').upper()
    code_upper = code.upper()
    
    # เช็คว่ามี keyword ไหนใน name หรือ code
    if any(kw.upper() in name or kw.upper() in code_upper for kw in keywords):
        found_branches.append({
            'code': code,
            'name': info.get('name', ''),
            'province': info.get('province', 'ไม่มีข้อมูล'),
            'lat': info.get('latitude', 0),
            'lon': info.get('longitude', 0),
        })

if found_branches:
    print(f'\nพบ {len(found_branches)} สาขาที่เกี่ยวข้อง:\n')
    
    # จัดกลุ่มตามจังหวัด
    by_province = {}
    for b in found_branches:
        prov = b['province']
        if prov not in by_province:
            by_province[prov] = []
        by_province[prov].append(b)
    
    for prov, branches in sorted(by_province.items()):
        print(f'\nจังหวัด: {prov} ({len(branches)} สาขา)')
        print('-'*80)
        for b in sorted(branches, key=lambda x: x['name']):
            print(f"  Code: {b['code']}")
            print(f"  Name: {b['name']}")
            print(f"  Lat/Lon: {b['lat']:.6f}, {b['lon']:.6f}")
            print()
else:
    print('\n❌ ไม่พบสาขาที่ตรงกับคำค้นหา')

# เช็คว่ามีสาขาเหล่านี้จับคู่กันในประวัติไหม
print('\n' + '='*80)
print('ตรวจสอบการจับคู่ในประวัติ')
print('='*80)

trip_pairs = model_data['trip_pairs']
found_codes = [b['code'] for b in found_branches]

pairs_found = []
for code1 in found_codes:
    for code2 in found_codes:
        if code1 < code2:  # เช็คแค่ครั้งเดียว
            pair = tuple(sorted([code1, code2]))
            if pair in trip_pairs:
                pairs_found.append((code1, code2))

if pairs_found:
    print(f'\nพบ {len(pairs_found)} คู่ที่เคยไปด้วยกันในประวัติ:\n')
    for code1, code2 in pairs_found:
        info1 = branch_info.get(code1, {})
        info2 = branch_info.get(code2, {})
        prov1 = info1.get('province', 'ไม่มี')
        prov2 = info2.get('province', 'ไม่มี')
        name1 = info1.get('name', '')
        name2 = info2.get('name', '')
        
        same_prov = '✅' if prov1 == prov2 else '⚠️ '
        print(f'{same_prov} {code1} ({name1[:30]}) จ.{prov1}')
        print(f'     <-> {code2} ({name2[:30]}) จ.{prov2}')
        print()
else:
    print('\n✅ ไม่พบคู่ที่เคยไปด้วยกันในประวัติ')
