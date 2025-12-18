"""
ทดสอบระบบ Highway-Based Logistics Zones
ตรวจสอบว่า:
1. Zone แยกถูกต้องตามถนนหลัก
2. No Cross-Zone ทำงาน (ห้ามข้ามเขา)
3. LIFO ordering (ไกลสุดก่อน)
4. Daisy Chain (ร้อยพวงถูกต้อง)
"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from app.py
from app import (
    LOGISTICS_ZONES, 
    HIGHWAY_ROUTES,
    NO_CROSS_ZONE_PAIRS,
    DISTRICT_TO_ZONE,
    PROVINCE_TO_ZONE,
    get_logistics_zone,
    is_cross_zone_violation,
    get_zone_priority,
    get_zone_distance_from_dc,
    get_highway_for_zone,
    can_combine_zones_by_highway,
    get_daisy_chain_order,
    DC_WANG_NOI_LAT,
    DC_WANG_NOI_LON
)

def test_zone_configuration():
    """ทดสอบว่า LOGISTICS_ZONES ครบถ้วน"""
    print("\n" + "="*60)
    print("🔍 ทดสอบ LOGISTICS_ZONES Configuration")
    print("="*60)
    
    print(f"\n📊 จำนวน Zones: {len(LOGISTICS_ZONES)}")
    
    # แสดงทุก zone
    for zone_name, zone_info in LOGISTICS_ZONES.items():
        priority = zone_info.get('priority', '?')
        distance = zone_info.get('distance_from_dc_km', '?')
        highway = zone_info.get('highway', '?')
        provinces = zone_info.get('provinces', [])
        
        print(f"  {priority:>2}. {zone_name[:30]:<30} | สาย {highway:<8} | {distance:>3}km | {', '.join(provinces[:3])}")
    
    return True

def test_zone_lookup():
    """ทดสอบการหา Zone จากจังหวัด/อำเภอ"""
    print("\n" + "="*60)
    print("🔍 ทดสอบ Zone Lookup")
    print("="*60)
    
    test_cases = [
        # (จังหวัด, อำเภอ, expected zone contains)
        ('พะเยา', 'เมืองพะเยา', 'ZONE_A'),
        ('น่าน', 'เมืองน่าน', 'ZONE_B'),
        ('แพร่', 'สูงเม่น', 'ZONE_C'),
        ('แพร่', 'เด่นชัย', 'ZONE_C'),
        ('อุตรดิตถ์', 'เมืองอุตรดิตถ์', 'ZONE_D'),
        ('พิษณุโลก', 'เมืองพิษณุโลก', 'ZONE_E'),
        ('พิจิตร', 'เมืองพิจิตร', 'ZONE_F'),
        ('นครสวรรค์', 'เมืองนครสวรรค์', 'ZONE_G'),
        ('นครราชสีมา', None, 'ZONE_H'),
        ('ขอนแก่น', None, 'ZONE_I'),
        ('ชลบุรี', None, 'ZONE_L'),
        ('ภูเก็ต', None, 'ZONE_P'),
        ('กรุงเทพมหานคร', None, 'NEARBY'),
    ]
    
    passed = 0
    failed = 0
    
    for prov, dist, expected in test_cases:
        zone = get_logistics_zone(prov, dist)
        if zone and expected in zone:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1
        
        print(f"  {status} {prov}/{dist or '-':<15} → {zone or 'None'}")
    
    print(f"\n📊 ผลลัพธ์: {passed}/{passed+failed} passed")
    return failed == 0

def test_no_cross_zone():
    """ทดสอบ No Cross-Zone Rules"""
    print("\n" + "="*60)
    print("🔍 ทดสอบ No Cross-Zone Rules (ห้ามข้ามเขา)")
    print("="*60)
    
    print(f"\n📋 กฎห้ามข้าม: {len(NO_CROSS_ZONE_PAIRS)} คู่")
    
    test_cases = [
        # (จังหวัด1, จังหวัด2, should_violate)
        ('เพชรบูรณ์', 'ชัยภูมิ', True),
        ('น่าน', 'พะเยา', True),
        ('แพร่', 'อุตรดิตถ์', True),
        ('กระบี่', 'สุราษฎร์ธานี', True),
        ('พิษณุโลก', 'พิจิตร', False),  # ถนนสายเดียวกัน
        ('ขอนแก่น', 'อุดรธานี', False),  # ถนนสายเดียวกัน
    ]
    
    passed = 0
    for prov1, prov2, should_violate in test_cases:
        result = is_cross_zone_violation(prov1, prov2)
        if result == should_violate:
            status = "✅"
            passed += 1
        else:
            status = "❌"
        
        action = "ห้ามรวม" if result else "รวมได้"
        print(f"  {status} {prov1} + {prov2} → {action}")
    
    print(f"\n📊 ผลลัพธ์: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_highway_merge():
    """ทดสอบการรวม Zone ตามถนนสายเดียวกัน"""
    print("\n" + "="*60)
    print("🔍 ทดสอบ Highway-Based Zone Merging")
    print("="*60)
    
    test_cases = [
        # (zone1, zone2, should_merge)
        ('ZONE_C_แพร่', 'ZONE_D_อุตรดิตถ์', True),  # สาย 11
        ('ZONE_D_อุตรดิตถ์', 'ZONE_E1_พิษณุโลก_ในเมือง', True),  # สาย 11
        ('ZONE_F1_พิจิตร_สายหลัก', 'ZONE_E1_พิษณุโลก_ในเมือง', True),  # สาย 11
        ('ZONE_H_โคราช', 'ZONE_I_ขอนแก่น', True),  # สาย 2
        ('ZONE_L_ชลบุรีระยอง', 'ZONE_M_จันทบุรีตราด', True),  # สาย 3
        ('ZONE_A_พะเยา', 'ZONE_H_โคราช', False),  # คนละสาย
        ('ZONE_O_ใต้อ่าวไทย', 'ZONE_P_ใต้อันดามัน', False),  # คนละฝั่ง
    ]
    
    passed = 0
    for zone1, zone2, should_merge in test_cases:
        hw1 = get_highway_for_zone(zone1)
        hw2 = get_highway_for_zone(zone2)
        result = can_combine_zones_by_highway(zone1, zone2)
        
        if result == should_merge:
            status = "✅"
            passed += 1
        else:
            status = "❌"
        
        action = "รวมได้" if result else "แยก"
        print(f"  {status} {zone1[:20]:<20} (สาย {hw1}) + {zone2[:20]:<20} (สาย {hw2}) → {action}")
    
    print(f"\n📊 ผลลัพธ์: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_lifo_ordering():
    """ทดสอบ LIFO ordering (ไกลสุดก่อน)"""
    print("\n" + "="*60)
    print("🔍 ทดสอบ LIFO Ordering (ไกลสุดโหลดก่อน)")
    print("="*60)
    
    # ทดสอบ priority
    zones_by_priority = []
    for zone_name, zone_info in LOGISTICS_ZONES.items():
        priority = zone_info.get('priority', 99)
        distance = zone_info.get('distance_from_dc_km', 0)
        zones_by_priority.append((zone_name, priority, distance))
    
    # เรียงตาม priority
    zones_by_priority.sort(key=lambda x: x[1])
    
    print("\n📋 ลำดับการโหลด (LIFO):")
    print("-" * 70)
    print(f"{'ลำดับ':>5} {'Zone':<35} {'Priority':>8} {'Distance':>10}")
    print("-" * 70)
    
    for i, (zone, priority, distance) in enumerate(zones_by_priority[:15], 1):
        print(f"{i:>5} {zone[:35]:<35} {priority:>8} {distance:>10}km")
    
    # ทดสอบ get_daisy_chain_order
    test_zones = ['ZONE_G_นครสวรรค์', 'ZONE_A_พะเยา', 'ZONE_E1_พิษณุโลก_ในเมือง', 'ZONE_C_แพร่']
    ordered = get_daisy_chain_order(test_zones)
    
    print(f"\n📦 ทดสอบ Daisy Chain:")
    print(f"  Input:  {test_zones}")
    print(f"  Output: {ordered}")
    
    # ตรวจว่า A (priority 1) อยู่ก่อน G (priority 11)
    if ordered[0] == 'ZONE_A_พะเยา' and ordered[-1] == 'ZONE_G_นครสวรรค์':
        print("  ✅ ลำดับถูกต้อง (ไกลสุดก่อน → ใกล้สุดท้าย)")
        return True
    else:
        print("  ❌ ลำดับไม่ถูกต้อง")
        return False

def test_with_real_branches():
    """ทดสอบกับข้อมูลสาขาจริง"""
    print("\n" + "="*60)
    print("🔍 ทดสอบกับสาขาจริง")
    print("="*60)
    
    # สาขาตัวอย่าง
    sample_branches = [
        {'code': 'PE00', 'name': 'สูงเม่น', 'province': 'แพร่', 'district': 'สูงเม่น'},
        {'code': 'PE01', 'name': 'เด่นชัย', 'province': 'แพร่', 'district': 'เด่นชัย'},
        {'code': 'UT00', 'name': 'เมืองอุตรดิตถ์', 'province': 'อุตรดิตถ์', 'district': 'เมืองอุตรดิตถ์'},
        {'code': 'PY00', 'name': 'เมืองพะเยา', 'province': 'พะเยา', 'district': 'เมืองพะเยา'},
        {'code': 'NN00', 'name': 'เมืองน่าน', 'province': 'น่าน', 'district': 'เมืองน่าน'},
        {'code': 'PL00', 'name': 'เมืองพิษณุโลก', 'province': 'พิษณุโลก', 'district': 'เมืองพิษณุโลก'},
        {'code': 'KK00', 'name': 'เมืองขอนแก่น', 'province': 'ขอนแก่น', 'district': 'เมืองขอนแก่น'},
    ]
    
    print("\n📋 Zone Assignment:")
    print("-" * 80)
    
    zone_groups = {}
    for branch in sample_branches:
        zone = get_logistics_zone(branch['province'], branch['district'])
        highway = get_highway_for_zone(zone) if zone else '-'
        priority = get_zone_priority(zone) if zone else 99
        
        print(f"  {branch['code']:<6} {branch['name']:<15} {branch['province']:<10} → {zone or 'None':<30} สาย {highway:<5} P={priority}")
        
        if zone:
            if zone not in zone_groups:
                zone_groups[zone] = []
            zone_groups[zone].append(branch['code'])
    
    print("\n📦 จัดกลุ่มตาม Zone:")
    for zone, codes in zone_groups.items():
        print(f"  {zone}: {codes}")
    
    # ตรวจว่า PE (แพร่) ไม่รวมกับ UT (อุตรดิตถ์)
    prae_zone = get_logistics_zone('แพร่', 'สูงเม่น')
    utt_zone = get_logistics_zone('อุตรดิตถ์', 'เมืองอุตรดิตถ์')
    
    print(f"\n🔒 ตรวจสอบ:")
    print(f"  แพร่ Zone: {prae_zone}")
    print(f"  อุตรดิตถ์ Zone: {utt_zone}")
    
    if prae_zone != utt_zone:
        print("  ✅ แพร่ และ อุตรดิตถ์ อยู่คนละ Zone (ถูกต้อง!)")
        return True
    else:
        print("  ❌ แพร่ และ อุตรดิตถ์ อยู่ Zone เดียวกัน (ผิด!)")
        return False

def main():
    """รันทุก test"""
    print("="*60)
    print("🧪 Highway-Based Logistics Zones Test Suite")
    print("="*60)
    
    results = []
    
    results.append(("Zone Configuration", test_zone_configuration()))
    results.append(("Zone Lookup", test_zone_lookup()))
    results.append(("No Cross-Zone", test_no_cross_zone()))
    results.append(("Highway Merge", test_highway_merge()))
    results.append(("LIFO Ordering", test_lifo_ordering()))
    results.append(("Real Branches", test_with_real_branches()))
    
    print("\n" + "="*60)
    print("📊 สรุปผลทดสอบ")
    print("="*60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {name}")
        if result:
            passed += 1
    
    print(f"\n🎯 รวม: {passed}/{len(results)} passed")
    
    if passed == len(results):
        print("\n✅ ทุก Test ผ่าน!")
    else:
        print("\n⚠️ มี Test ไม่ผ่าน")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
