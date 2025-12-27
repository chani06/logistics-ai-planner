import json

with open('branch_groups.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

btg = data['branch_to_group']
groups = data['groups']

# สาขาฟิวเจอร์รังสิต
codes = ['11005995', 'G017', 'G015', 'N022', 'G013', 'N200']

print('🔍 ตรวจสอบสาขาฟิวเจอร์รังสิต:')
print('='*60)

group_ids = set()
for c in codes:
    g = btg.get(c, 'ไม่อยู่ในกลุ่ม')
    print(f'{c}: {g}')
    if g != 'ไม่อยู่ในกลุ่ม':
        group_ids.add(g)

print(f'\n📊 สรุป:')
print(f'   - พบ {len(group_ids)} กลุ่ม: {list(group_ids)}')

if len(group_ids) == 1:
    print('   ✅ ทุกสาขาอยู่กลุ่มเดียวกัน!')
else:
    print('   ⚠️ สาขาอยู่คนละกลุ่ม')
    for gid in group_ids:
        print(f'\n   กลุ่ม {gid}:')
        for c in groups.get(gid, []):
            print(f'      - {c}')
