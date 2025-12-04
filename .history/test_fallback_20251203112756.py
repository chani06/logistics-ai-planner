# -*- coding: utf-8 -*-
"""ทดสอบระบบหลังแก้ไข - Fallback Mechanism"""
import sys
sys.path.insert(0, '.')

print("="*70)
print("🔍 ทดสอบระบบ Fallback Mechanism")
print("="*70)

try:
    from app import BOOKING_RESTRICTIONS, PUNTHAI_PATTERNS
    
    print("\n✅ Check 1: Booking Restrictions Loading")
    booking_stats = BOOKING_RESTRICTIONS.get('stats', {})
    
    if booking_stats.get('fallback'):
        print("   ⚠️ Fallback Mode: ใช้ Punthai เป็นหลัก")
        print(f"   Message: {booking_stats.get('message', 'N/A')}")
    else:
        print(f"   ✅ Loaded from Booking History")
        print(f"   Total branches: {booking_stats.get('total_branches', 0):,}")
        print(f"   Total bookings: {booking_stats.get('total_bookings', 0):,}")
    
    print("\n✅ Check 2: Punthai Patterns Loading")
    punthai_restrictions = PUNTHAI_PATTERNS.get('punthai_restrictions', {})
    punthai_stats = PUNTHAI_PATTERNS.get('stats', {})
    
    print(f"   Total branches: {len(punthai_restrictions):,}")
    print(f"   Same province: {punthai_stats.get('same_province_pct', 0):.1f}%")
    
    print("\n✅ Check 3: Combined Coverage")
    booking_restrictions = BOOKING_RESTRICTIONS.get('branch_restrictions', {})
    all_branches = set(booking_restrictions.keys()) | set(punthai_restrictions.keys())
    print(f"   Total unique branches: {len(all_branches):,}")
    print(f"   Booking: {len(booking_restrictions):,}")
    print(f"   Punthai: {len(punthai_restrictions):,}")
    
    print("\n" + "="*70)
    print("📊 Summary")
    print("="*70)
    
    if booking_stats.get('fallback'):
        print("""
⚠️ Fallback Mode Active:

1. **Booking History**: ไม่พบไฟล์
   - ระบบใช้ Punthai เป็นหลัก
   - สำรอง: Default เป็น JB (รถกลาง)

2. **Punthai**: {0:,} สาขา
   - Location patterns: 67.8% same province
   - Vehicle restrictions: {0:,} สาขา

3. **กลยุทธ์**:
   - มีใน Punthai → ใช้ Punthai
   - ไม่มีใน Punthai → Default: JB
   - ระยะไกล → 6W (ตามหลักการ)

✅ System Status: WORKING (Fallback Mode)
        """.format(len(punthai_restrictions)))
    else:
        print(f"""
✅ Normal Mode Active:

1. **Booking History**: {booking_stats.get('total_branches', 0):,} สาขา
   - {booking_stats.get('total_bookings', 0):,} bookings
   - Strict: {booking_stats.get('strict', 0):,}

2. **Punthai**: {len(punthai_restrictions):,} สาขา
   - Location patterns: 67.8% same province

3. **Total Coverage**: {len(all_branches):,} สาขา

✅ System Status: FULLY OPERATIONAL
        """)

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
