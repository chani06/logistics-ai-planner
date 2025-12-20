"""
ทดสอบหาปัญหาที่ 6W ยังผ่านไปได้ใน nearby provinces
"""
import pandas as pd

# จำลอง data จาก user's input
test_trips = [
    # ทริปที่มีปัญหา - 6W ใน nearby
    (40, '6W002', 'กรุงเทพมหานคร', 'ลาดกระบัง'),  # MG87
    (43, '6W003', 'กรุงเทพมหานคร', 'สะพานสูง'),   # หลายสาขา
    (43, '6W003', 'สมุทรปราการ', 'บางเสาธง'),     # มี nearby
    (52, '6W005', 'นครปฐม', 'สามพราน'),           # nearby
    (52, '6W005', 'กรุงเทพมหานคร', 'จตุจักร'),    # nearby
    (68, '6W006', 'นครปฐม', 'เมืองนครปฐม'),       # nearby
    (72, '6W013', 'กรุงเทพมหานคร', 'เฉลิมพระเกียรติ'), # mixed!
    (72, '6W013', 'นครราชสีมา', 'เมืองนครราชสีมา'),    # far แต่ใช้ 6W ผิด
    (78, '6W017', 'สมุทรปราการ', 'บางพลี'),       # nearby
]

# Import function
from app import get_region_type

print("=" * 70)
print("🔍 ทดสอบ get_region_type() สำหรับทริปที่มีปัญหา")
print("=" * 70)

problem_trips = {}

for trip_num, vehicle, province, district in test_trips:
    region = get_region_type(province)
    is_nearby = region == 'nearby'
    is_6w = '6W' in vehicle
    
    # ถ้า 6W ใน nearby = ปัญหา!
    if is_6w and is_nearby:
        status = "❌ BUG - 6W ใน nearby!"
        if trip_num not in problem_trips:
            problem_trips[trip_num] = {'vehicle': vehicle, 'provinces': []}
        problem_trips[trip_num]['provinces'].append(province)
    elif is_6w and region == 'far':
        status = "✅ OK - 6W ใน far"
    elif is_6w:
        status = f"⚠️ 6W ใน {region}"
    else:
        status = "✅ OK"
    
    print(f"Trip {trip_num:2} {vehicle}: {province:20} ({district:15}) → {region:10} {status}")

print("\n" + "=" * 70)
print("📊 สรุปทริปที่มีปัญหา (6W ใน nearby)")
print("=" * 70)

for trip_num, info in problem_trips.items():
    print(f"  ทริป {trip_num} ({info['vehicle']}): {', '.join(info['provinces'])}")

print("\n" + "=" * 70)
print("🔬 ทดสอบ any() logic")
print("=" * 70)

# ทดสอบ any() vs all() สำหรับ mixed trip 72
trip_72_provinces = ['กรุงเทพมหานคร', 'นครราชสีมา']
regions = [get_region_type(p) for p in trip_72_provinces]

any_nearby = any(r == 'nearby' for r in regions)
all_nearby = all(r == 'nearby' for r in regions)

print(f"ทริป 72: {trip_72_provinces}")
print(f"  regions = {regions}")
print(f"  any(nearby) = {any_nearby} {'✅ ควร BAN 6W' if any_nearby else ''}")
print(f"  all(nearby) = {all_nearby}")
print()

# ถ้า any() = True → ควร ban 6W แต่ทำไมไม่ ban?
if any_nearby:
    print("🔍 ปัญหา: any() = True แต่ 6W ยังถูกใช้!")
    print("   สาเหตุที่เป็นไปได้:")
    print("   1. get_province() return 'UNKNOWN' สำหรับบางสาขา")
    print("   2. provinces set ว่างเปล่า → any() = False by default")
    print("   3. มี code path อื่นที่ override การ ban")
else:
    print("🔍 any() = False → 6W ถูกอนุญาต (ปกติ)")

print("\n" + "=" * 70)
print("✅ การทดสอบเสร็จสิ้น")
print("=" * 70)
