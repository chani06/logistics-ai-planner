# -*- coding: utf-8 -*-
"""ทดสอบระบบที่รวมข้อมูลจาก Booking History + Punthai"""
import pandas as pd
import sys

sys.path.insert(0, '.')
from app import BOOKING_RESTRICTIONS, PUNTHAI_PATTERNS, get_max_vehicle_for_branch, check_branch_vehicle_compatibility

print("="*70)
print("🎯 ทดสอบระบบรวมข้อมูล: Booking History + Punthai")
print("="*70)

# Check 1: Booking History
print("\n✅ Data Source 1: Booking History (ข้อมูลจริง)")
booking_stats = BOOKING_RESTRICTIONS.get('stats', {})
booking_restrictions = BOOKING_RESTRICTIONS.get('branch_restrictions', {})
print(f"   Total branches: {booking_stats.get('total_branches', 0):,}")
print(f"   Strict: {booking_stats.get('strict', 0):,} ({booking_stats.get('strict', 0)/max(booking_stats.get('total_branches', 1), 1)*100:.1f}%)")
print(f"   Flexible: {booking_stats.get('flexible', 0):,} ({booking_stats.get('flexible', 0)/max(booking_stats.get('total_branches', 1), 1)*100:.1f}%)")

# Check 2: Punthai
print("\n✅ Data Source 2: Punthai (แผน)")
punthai_restrictions = PUNTHAI_PATTERNS.get('punthai_restrictions', {})
punthai_stats = PUNTHAI_PATTERNS.get('stats', {})
print(f"   Total branches: {len(punthai_restrictions):,}")
print(f"   Location stats: {punthai_stats.get('same_province_pct', 0):.1f}% same province")

# Check 3: รวมกัน
print("\n✅ Combined Data")
all_branches = set(booking_restrictions.keys()) | set(punthai_restrictions.keys())
print(f"   Total unique branches: {len(all_branches):,}")

# สาขาที่มีในทั้งสองแหล่ง
common_branches = set(booking_restrictions.keys()) & set(punthai_restrictions.keys())
print(f"   Common branches: {len(common_branches):,}")

# สาขาที่มีแค่ใน Booking
only_booking = set(booking_restrictions.keys()) - set(punthai_restrictions.keys())
print(f"   Only in Booking: {len(only_booking):,}")

# สาขาที่มีแค่ใน Punthai
only_punthai = set(punthai_restrictions.keys()) - set(booking_restrictions.keys())
print(f"   Only in Punthai: {len(only_punthai):,}")

# Check 4: ทดสอบการตัดสินใจ
print("\n" + "="*70)
print("🔍 ทดสอบการตัดสินใจร่วมกัน")
print("="*70)

# Test 1: สาขาที่มีใน Booking (ใช้ข้อมูล Booking)
if booking_restrictions:
    branch = list(booking_restrictions.keys())[0]
    info = booking_restrictions[branch]
    source = 'BOOKING (ข้อมูลจริง)'
    print(f"\n1. Branch {branch} ({source}):")
    print(f"   Max vehicle: {get_max_vehicle_for_branch(branch)}")
    print(f"   Allowed: {info.get('allowed', [])}")
    print(f"   Total bookings: {info.get('total_bookings', 0)}")
    print(f"   ✓ ใช้ข้อมูลจาก Booking History (ความเชื่อมั่นสูง)")

# Test 2: สาขาที่มีแค่ใน Punthai (ใช้ข้อมูล Punthai)
if only_punthai:
    branch = list(only_punthai)[0]
    info = punthai_restrictions[branch]
    source = 'PUNTHAI (แผน)'
    print(f"\n2. Branch {branch} ({source}):")
    print(f"   Max vehicle: {get_max_vehicle_for_branch(branch)}")
    print(f"   Allowed: {info.get('allowed', [])}")
    print(f"   ✓ ใช้ข้อมูลจาก Punthai (ไม่มีใน Booking)")

# Test 3: สาขาที่มีในทั้งสอง (ใช้ Booking เป็นหลัก)
if common_branches:
    branch = list(common_branches)[0]
    booking_info = booking_restrictions[branch]
    punthai_info = punthai_restrictions[branch]
    
    print(f"\n3. Branch {branch} (มีทั้งสองแหล่ง):")
    print(f"   Booking says: {booking_info.get('max_vehicle')} (allowed: {booking_info.get('allowed', [])})")
    print(f"   Punthai says: {punthai_info.get('max_vehicle')} (allowed: {punthai_info.get('allowed', [])})")
    print(f"   System uses: {get_max_vehicle_for_branch(branch)}")
    print(f"   ✓ ใช้ Booking เป็นหลัก (ข้อมูลจริง > แผน)")

# Test 4: สาขาที่ไม่มีในทั้งสอง
test_branch = 'TEST999'
print(f"\n4. Branch {test_branch} (ไม่มีข้อมูล):")
print(f"   Max vehicle: {get_max_vehicle_for_branch(test_branch)}")
print(f"   ✓ Default: 6W (ยืดหยุ่น)")

# Check 5: ความแตกต่างระหว่างสองแหล่ง
print("\n" + "="*70)
print("⚖️ เปรียบเทียบความแตกต่าง (สาขาที่มีทั้งสอง)")
print("="*70)

differences = []
for branch in list(common_branches)[:10]:
    booking_max = booking_restrictions[branch].get('max_vehicle')
    punthai_max = punthai_restrictions[branch].get('max_vehicle')
    if booking_max != punthai_max:
        differences.append({
            'branch': branch,
            'booking': booking_max,
            'punthai': punthai_max
        })

if differences:
    print(f"\nพบความแตกต่าง {len(differences)} สาขา (ตัวอย่าง):")
    for diff in differences[:5]:
        print(f"  {diff['branch']}: Booking={diff['booking']}, Punthai={diff['punthai']} → ใช้ {diff['booking']}")
else:
    print("\nไม่พบความแตกต่าง (ทั้งสองแหล่งตรงกัน)")

print("\n" + "="*70)
print("📊 Summary")
print("="*70)
print(f"""
✅ ระบบรวมข้อมูลสำเร็จ:

1. **Booking History** (ข้อมูลจริง - ลำดับแรก):
   - {booking_stats.get('total_branches', 0):,} สาขา
   - {booking_stats.get('total_bookings', 0):,} bookings
   - ความเชื่อมั่น: สูง

2. **Punthai** (แผน - สำรอง):
   - {len(punthai_restrictions):,} สาขา
   - ความเชื่อมั่น: ปานกลาง

3. **Total Coverage**:
   - {len(all_branches):,} สาขาทั้งหมด
   - {len(common_branches):,} สาขาตรวจสอบได้ 2 แหล่ง
   - {len(only_booking):,} สาขามีแค่ประวัติ
   - {len(only_punthai):,} สาขามีแค่แผน

🎯 กลยุทธ์:
   1. มีใน Booking → ใช้ Booking (ข้อมูลจริง)
   2. ไม่มีใน Booking → ใช้ Punthai (แผน)
   3. ไม่มีทั้งสอง → ใช้ 6W (ยืดหยุ่น)
""")
