"""
ตรวจสอบกลุ่มที่ model จับได้ว่ามีสาขาข้ามโซนหรือไม่
"""

import pickle
import pandas as pd
import sys
import io

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# โหลด model
with open('models/decision_tree_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

pairs = model_data['trip_pairs']
branch_info = model_data['branch_info']

print('=' * 80)
print('ตรวจสอบกลุ่มที่ model จับได้ว่ามีสาขาข้ามโซนหรือไม่')
print('=' * 80)

# สร้าง dict เก็บจังหวัดของแต่ละสาขา
branch_provinces = {}
for code, info in branch_info.items():
    if 'province' in info:
        branch_provinces[code] = info['province']

print(f'\nสาขาที่มีข้อมูลจังหวัด: {len(branch_provinces)} สาขา')

# สร้างกลุ่มจาก pairs
groups = {}
for (code1, code2) in pairs:
    # เพิ่ม code1, code2 เข้ากลุ่ม
    found = False
    for group_id, members in groups.items():
        if code1 in members or code2 in members:
            members.add(code1)
            members.add(code2)
            found = True
            break
    
    if not found:
        group_id = len(groups)
        groups[group_id] = {code1, code2}

# Merge กลุ่มที่เชื่อมกัน
merged = True
while merged:
    merged = False
    group_ids = list(groups.keys())
    for i in range(len(group_ids)):
        for j in range(i+1, len(group_ids)):
            if group_ids[i] not in groups or group_ids[j] not in groups:
                continue
            
            if groups[group_ids[i]] & groups[group_ids[j]]:
                groups[group_ids[i]] = groups[group_ids[i]] | groups[group_ids[j]]
                del groups[group_ids[j]]
                merged = True
                break
        if merged:
            break

print(f'\n📊 จำนวนกลุ่มทั้งหมด: {len(groups)} กลุ่ม')
print(f'📦 จำนวน pairs ทั้งหมด: {len(pairs)} pairs')

# ตรวจสอบแต่ละกลุ่ม
cross_zone_groups = []
same_zone_groups = []

for group_id, members in groups.items():
    provinces = set()
    for code in members:
        if code in branch_provinces:
            provinces.add(branch_provinces[code])
    
    if len(provinces) > 1:
        cross_zone_groups.append({
            'group_id': group_id,
            'members': members,
            'provinces': provinces
        })
    elif len(provinces) == 1:
        same_zone_groups.append({
            'group_id': group_id,
            'members': members,
            'provinces': provinces
        })

print(f'\n✅ กลุ่มที่อยู่โซนเดียวกัน: {len(same_zone_groups)} กลุ่ม')
print(f'⚠️  กลุ่มที่มีสาขาข้ามโซน: {len(cross_zone_groups)} กลุ่ม')

if cross_zone_groups:
    print('\n' + '=' * 80)
    print('รายละเอียดกลุ่มที่มีสาขาข้ามโซน:')
    print('=' * 80)
    
    for idx, group in enumerate(cross_zone_groups[:30], 1):
        provinces_list = sorted(group['provinces'])
        print(f'\nกลุ่มที่ {idx}:')
        print(f'  จังหวัด: {", ".join(provinces_list)}')
        print(f'  จำนวนสาขา: {len(group["members"])} สาขา')
        
        # แสดงสาขาแต่ละจังหวัด
        for prov in provinces_list:
            branches_in_prov = [code for code in group['members'] if branch_provinces.get(code) == prov]
            print(f'\n  จ.{prov} ({len(branches_in_prov)} สาขา):')
            for code in sorted(branches_in_prov)[:5]:
                name = branch_info.get(code, {}).get('name', '')
                print(f'    - {code} ({name})')
            if len(branches_in_prov) > 5:
                print(f'    ... และอีก {len(branches_in_prov) - 5} สาขา')
else:
    print('\n✅ ไม่พบกลุ่มที่มีสาขาข้ามโซน - ระบบทำงานถูกต้อง!')

# แสดงตัวอย่างกลุ่มที่ถูกต้อง
if same_zone_groups:
    print('\n' + '=' * 80)
    print('ตัวอย่างกลุ่มที่ถูกต้อง (โซนเดียวกัน) - 10 กลุ่มแรก:')
    print('=' * 80)
    
    for idx, group in enumerate(same_zone_groups[:10], 1):
        prov = list(group['provinces'])[0]
        print(f'\nกลุ่มที่ {idx} - จ.{prov} ({len(group["members"])} สาขา):')
        for code in sorted(group['members'])[:8]:
            name = branch_info.get(code, {}).get('name', '')
            print(f'  - {code} ({name})')
        if len(group['members']) > 8:
            print(f'  ... และอีก {len(group["members"]) - 8} สาขา')
