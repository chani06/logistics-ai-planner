"""
ตรวจสอบแหล่งข้อมูลที่ระบบใช้
"""
import os

print("=" * 70)
print("ตรวจสอบแหล่งข้อมูลในระบบ")
print("=" * 70)

# ตรวจสอบไฟล์ที่มีอยู่
files_to_check = {
    'branch_data.json': 'ข้อมูลสาขาจาก Google Sheets (Primary)',
    'Dc/สถานที่ส่ง.xlsx': 'Master สถานที่ส่ง (ตำแหน่ง GPS)',
    'Dc/Auto planning (1).xlsx': 'ข้อมูลข้อจำกัดรถ',
    'Dc/ประวัติงานจัดส่ง DC วังน้อย.xlsx': 'ประวัติการจัดส่ง (Booking History)',
    'Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx': 'แผนงาน Punthai',
    'Dc/Master Dist.xlsx': 'Master ระยะทาง',
}

print("\nสถานะไฟล์:")
print("-" * 70)
for file_path, description in files_to_check.items():
    exists = os.path.exists(file_path)
    status = "✅ พบ" if exists else "❌ ไม่พบ"
    
    size = ""
    if exists:
        file_size = os.path.getsize(file_path)
        if file_size < 1024:
            size = f"({file_size} B)"
        elif file_size < 1024 * 1024:
            size = f"({file_size / 1024:.1f} KB)"
        else:
            size = f"({file_size / (1024 * 1024):.1f} MB)"
    
    print(f"{status} {file_path} {size}")
    print(f"    → {description}")

# อ่านโค้ดเพื่อดูว่าใช้ไฟล์ไหน
print("\n" + "=" * 70)
print("การใช้งานข้อมูลในระบบ")
print("=" * 70)

usage_info = """
1. ข้อมูลสาขา (Branch Data):
   ✅ Google Sheets → branch_data.json
   - ดึงข้อมูลจาก Google Sheets ทุกครั้งที่โหลดเว็บ
   - บันทึกลง branch_data.json
   - รวม DC วังน้อย, MaxTruckType, พิกัด GPS

2. ตำแหน่ง GPS (Location):
   ⚠️  Dc/สถานที่ส่ง.xlsx (EXCEL - ยังใช้อยู่)
   - ใช้เป็น fallback ถ้า branch_data.json ไม่มีพิกัด
   - ควรใช้จาก Google Sheets แทน

3. ข้อจำกัดรถ (Vehicle Restrictions):
   ⚠️  Dc/Auto planning (1).xlsx (EXCEL - ยังใช้อยู่)
   - โหลดข้อจำกัดรถจาก Excel
   - ควรย้ายมาอยู่ใน Google Sheets

4. Booking History:
   ⚠️  Dc/ประวัติงานจัดส่ง DC วังน้อย.xlsx (EXCEL - ยังใช้อยู่)
   - ประวัติการใช้รถของแต่ละสาขา
   - ควรย้ายมาอยู่ใน Google Sheets

5. แผนงาน Punthai:
   ⚠️  Dc/แผนงาน Punthai... .xlsx (EXCEL - ยังใช้อยู่)
   - ข้อมูล Punthai planning
   - ควรย้ายมาอยู่ใน Google Sheets

สรุป:
✅ ข้อมูลสาขาหลัก: ใช้จาก Google Sheets แล้ว (branch_data.json)
⚠️  ข้อมูลอื่นๆ: ยังใช้ Excel อยู่ (ควรย้ายไป Sheets)
"""

print(usage_info)

# ตรวจสอบว่า sync เมื่อไหร่
if os.path.exists('branch_data.json'):
    import time
    mtime = os.path.getmtime('branch_data.json')
    import datetime
    last_sync = datetime.datetime.fromtimestamp(mtime)
    now = datetime.datetime.now()
    diff = now - last_sync
    
    print("\n" + "=" * 70)
    print("สถานะ branch_data.json")
    print("=" * 70)
    print(f"Sync ล่าสุด: {last_sync.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"ผ่านมา: {diff.seconds // 60} นาที {diff.seconds % 60} วินาที")
    
    # นับจำนวนสาขา
    import json
    with open('branch_data.json', encoding='utf-8') as f:
        data = json.load(f)
    print(f"จำนวนสาขา: {len(data):,}")
    
    # เช็ค DC วังน้อย
    if '8nvDC011' in data:
        print("✅ มี DC วังน้อย (8nvDC011)")
    else:
        print("❌ ไม่มี DC วังน้อย")
