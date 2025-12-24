# -*- coding: utf-8 -*-
import json

# อ่านไฟล์ JSON
with open('branch_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"📊 จำนวนสาขาทั้งหมด: {len(data):,}")
print(f"🏢 มี DC วังน้อย (8nvDC011): {'✅' if '8nvDC011' in data else '❌'}")

# แสดง 5 สาขาแรก
print(f"\n📋 ตัวอย่าง 5 สาขาแรก:")
for i, code in enumerate(list(data.keys())[:5], 1):
    branch = data[code]
    name = branch.get('สาขา', branch.get('Plan Code', code))
    print(f"   {i}. {code}: {name}")
