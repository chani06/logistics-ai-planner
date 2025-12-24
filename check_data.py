"""ตรวจสอบข้อมูลจริงจาก Google Sheets"""
import json

with open('branch_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"✅ จำนวนสาขา: {len(data):,}")
print(f"✅ DC วังน้อย: {'✅ มี' if '8nvDC011' in data else '❌ ไม่มี'}")

# ดู 10 สาขาแรก
sample_keys = list(data.keys())[:10]
print(f"\n📋 ตัวอย่าง 10 สาขาแรก:")
for i, code in enumerate(sample_keys, 1):
    branch = data[code]
    name = branch.get('สาขา', branch.get('Branch Name', ''))
    province = branch.get('จังหวัด', branch.get('Province', ''))
    print(f"   {i}. {code}: {name} - {province}")

# ตรวจสอบ DC วังน้อย
if '8nvDC011' in data:
    dc = data['8nvDC011']
    print(f"\n🏢 DC วังน้อย:")
    print(f"   ชื่อ: {dc.get('สาขา', '')}")
    print(f"   จังหวัด: {dc.get('จังหวัด', '')}")
    print(f"   พิกัด: {dc.get('ละ', '')} , {dc.get('ลอง', '')}")

# เช็คว่าเป็นข้อมูล Sample หรือจริง
sample_count = sum(1 for k in data.keys() if k.startswith('BR'))
real_count = len(data) - sample_count

print(f"\n🔍 วิเคราะห์:")
print(f"   Sample data (BR00XX): {sample_count}")
print(f"   ข้อมูลจริง: {real_count}")
print(f"   {'✅ ข้อมูลจริงจาก Google Sheets' if sample_count == 0 else '⚠️ ยังมี Sample data'}")
