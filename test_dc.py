"""ทดสอบว่า DC วังน้อยถูกเพิ่มในฐานข้อมูล"""
import json
from app import sync_branch_data_from_sheets

# Sync ข้อมูล
print("🔄 กำลัง Sync ข้อมูลจาก Google Sheets...")
df = sync_branch_data_from_sheets()

# อ่านข้อมูลจาก JSON
with open('branch_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"\n✅ รวมทั้งหมด: {len(data)} สาขา")

# ตรวจสอบ DC วังน้อย
if '8nvDC011' in data:
    print("\n✅ พบ DC วังน้อย (8nvDC011) ในฐานข้อมูล:")
    dc_info = data['8nvDC011']
    print(f"   สาขา: {dc_info.get('สาขา', 'N/A')}")
    print(f"   จังหวัด: {dc_info.get('จังหวัด', 'N/A')}")
    print(f"   อำเภอ: {dc_info.get('อำเภอ', 'N/A')}")
    print(f"   ตำบล: {dc_info.get('ตำบล', 'N/A')}")
    print(f"   ละ: {dc_info.get('ละ', 'N/A')}")
    print(f"   ลอง: {dc_info.get('ลอง', 'N/A')}")
else:
    print("\n❌ ไม่พบ DC วังน้อย (8nvDC011) ในฐานข้อมูล")

# แสดง 5 สาขาแรก
print("\n📋 ตัวอย่าง 5 สาขาแรก:")
for i, code in enumerate(list(data.keys())[:5]):
    branch_name = data[code].get('สาขา', data[code].get('Plan Code', code))
    print(f"   {i+1}. {code}: {branch_name}")
