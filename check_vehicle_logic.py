"""
ตรวจสอบการเลือกรถ และ DC วังน้อย
"""
import json

# โหลดข้อมูล
with open('branch_data.json', encoding='utf-8') as f:
    data = json.load(f)

# ตรวจสอบ DC วังน้อย
dc_code = '8NVDC011'
if dc_code in data:
    dc = data[dc_code]
    print("=" * 60)
    print(f"✅ พบ DC วังน้อย ({dc_code})")
    print("=" * 60)
    print(f"สาขา: {dc.get('สาขา', 'N/A')}")
    print(f"MaxTruckType: {dc.get('MaxTruckType', 'N/A')}")
    print(f"จังหวัด: {dc.get('จังหวัด', 'N/A')}")
    print(f"ละติจูด: {dc.get('ละ', dc.get('ละติจูด', 'N/A'))}")
    print(f"ลองติจูด: {dc.get('ลอง', dc.get('ลองติจูด', 'N/A'))}")
    print(f"รายละเอียด: {dc.get('รายละเอียด', 'N/A')}")
else:
    print(f"❌ ไม่พบ DC วังน้อย ({dc_code}) ใน JSON")

# สรุปประเภทรถ
print("\n" + "=" * 60)
print("สรุปข้อจำกัดรถในฐานข้อมูล")
print("=" * 60)

truck_types = {}
for code, branch in data.items():
    max_truck = branch.get('MaxTruckType', 'N/A')
    if max_truck not in truck_types:
        truck_types[max_truck] = []
    truck_types[max_truck].append(code)

for truck_type in sorted(truck_types.keys()):
    count = len(truck_types[truck_type])
    print(f"{truck_type}: {count:,} สาขา")
    
    # แสดงตัวอย่าง
    if count <= 5:
        for code in truck_types[truck_type]:
            branch_name = data[code].get('สาขา', 'N/A')
            print(f"  - {code}: {branch_name}")

# ตรวจสอบ logic การเลือกรถ
print("\n" + "=" * 60)
print("Logic การเลือกรถในระบบ")
print("=" * 60)
print("""
1. ระบบตรวจสอบ MaxTruckType ของแต่ละสาขาจาก Google Sheets
2. ถ้าสาขามี MaxTruckType = '6W' → สามารถใช้รถทุกประเภทได้ (4W, JB, 6W)
3. ถ้าสาขามี MaxTruckType = 'JB' → ใช้ได้แค่ 4W, JB
4. ถ้าสาขามี MaxTruckType = '4W' → ใช้ได้แค่ 4W

สำหรับ DC วังน้อย:
- MaxTruckType: {0}
- ความหมาย: {1}
""".format(
    dc.get('MaxTruckType', 'N/A') if dc_code in data else 'N/A',
    'ใช้ได้ทุกประเภทรถ (4W, JB, 6W)' if data.get(dc_code, {}).get('MaxTruckType') == '6W' 
    else 'ใช้ได้แค่ 4W และ JB' if data.get(dc_code, {}).get('MaxTruckType') == 'JB'
    else 'ใช้ได้แค่ 4W' if data.get(dc_code, {}).get('MaxTruckType') == '4W'
    else 'ไม่พบข้อมูล'
))

print("\nฟังก์ชันสำคัญที่ใช้ในการเลือกรถ:")
print("1. get_max_vehicle_for_branch(branch_code) → คืนรถที่ใหญ่ที่สุดที่สาขารองรับ")
print("2. get_max_vehicle_for_trip(trip_codes) → คืนรถที่เล็กที่สุดจากทุกสาขาในทริป")
print("3. suggest_truck(weight, cube, max_allowed) → แนะนำรถที่เหมาะสมตามน้ำหนัก/ปริมาตร")
