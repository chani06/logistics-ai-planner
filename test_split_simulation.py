"""
ทดสอบด้วยข้อมูลจำลอง - ตรวจสอบว่า:
1. ไม่มีสาขาตกหล่น
2. รถทุกคันเต็มอย่างน้อย 95% (หรือเป็นรถสุดท้าย)
"""

import pandas as pd
import numpy as np

# สร้างข้อมูลจำลอง
np.random.seed(42)

# LIMITS
LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5},
    'JB': {'max_w': 3500, 'max_c': 7},
    '6W': {'max_w': 5500, 'max_c': 20}
}

print("=" * 80)
print("🔍 ทดสอบการตัดแบ่งสาขาด้วยข้อมูลจำลอง")
print("=" * 80)

# สร้างสาขาจำลอง - มีข้อมูลหลากหลาย
test_cases = [
    {
        'name': 'กรณี 1: รถ 6W เต็ม 200%+',
        'branches': [
            {'Code': f'B{i:03d}', 'Weight': 300, 'Cube': 2.0} for i in range(20)  # รวม 6000kg, 40m³ = 200%
        ]
    },
    {
        'name': 'กรณี 2: รถ 6W เต็ม 150%',
        'branches': [
            {'Code': f'C{i:03d}', 'Weight': 250, 'Cube': 1.5} for i in range(20)  # รวม 5000kg, 30m³ = 150%
        ]
    },
    {
        'name': 'กรณี 3: สาขาขนาดเล็กหลายสาขา',
        'branches': [
            {'Code': f'D{i:03d}', 'Weight': 100, 'Cube': 0.5} for i in range(30)  # รวม 3000kg, 15m³
        ]
    },
    {
        'name': 'กรณี 4: สาขาขนาดใหญ่เกินรถเล็ก',
        'branches': [
            {'Code': f'E{i:03d}', 'Weight': 1500, 'Cube': 3.0} for i in range(5)  # รวม 7500kg, 15m³
        ]
    }
]

def simulate_split(branches, target_vehicle='6W', min_util=95, max_util=120):
    """จำลองการตัดแบ่งสาขาใส่รถ"""
    target_w = LIMITS[target_vehicle]['max_w']
    target_c = LIMITS[target_vehicle]['max_c']
    
    # เรียงตามน้ำหนัก
    sorted_branches = sorted(branches, key=lambda x: x['Weight'], reverse=True)
    
    trips = []
    current_group = []
    current_w = 0
    current_c = 0
    
    for branch in sorted_branches:
        w = branch['Weight']
        c = branch['Cube']
        
        test_w = current_w + w
        test_c = current_c + c
        test_util = max((test_w / target_w) * 100, (test_c / target_c) * 100)
        
        # ถ้าใส่ได้ (≤120%) หรือเป็นสาขาแรก
        if test_util <= max_util or len(current_group) == 0:
            current_group.append(branch['Code'])
            current_w = test_w
            current_c = test_c
        else:
            # เต็มแล้ว → สร้างทริปใหม่
            current_util = max((current_w / target_w) * 100, (current_c / target_c) * 100)
            
            if current_util >= min_util:
                trips.append({
                    'codes': current_group.copy(),
                    'weight': current_w,
                    'cube': current_c,
                    'util': current_util
                })
                current_group = [branch['Code']]
                current_w = w
                current_c = c
            else:
                # ยังไม่เต็มพอ → ใส่ต่อ
                current_group.append(branch['Code'])
                current_w = test_w
                current_c = test_c
    
    # เพิ่มกลุ่มสุดท้าย (สำคัญมาก!)
    if current_group:
        final_util = max((current_w / target_w) * 100, (current_c / target_c) * 100)
        
        # ถ้ากลุ่มสุดท้ายน้อยเกินไป → รวมกับกลุ่มก่อนหน้า
        if final_util < min_util and len(trips) > 0:
            last_trip = trips[-1]
            combined_w = current_w + last_trip['weight']
            combined_c = current_c + last_trip['cube']
            combined_util = max((combined_w / target_w) * 100, (combined_c / target_c) * 100)
            
            if combined_util <= 140:
                # รวมได้
                trips[-1]['codes'].extend(current_group)
                trips[-1]['weight'] = combined_w
                trips[-1]['cube'] = combined_c
                trips[-1]['util'] = combined_util
            else:
                # รวมไม่ได้ → สร้างทริปใหม่แม้น้อย
                trips.append({
                    'codes': current_group,
                    'weight': current_w,
                    'cube': current_c,
                    'util': final_util
                })
        else:
            trips.append({
                'codes': current_group,
                'weight': current_w,
                'cube': current_c,
                'util': final_util
            })
    
    return trips

# ทดสอบแต่ละกรณี
all_passed = True

for case in test_cases:
    print(f"\n📋 {case['name']}")
    print("-" * 60)
    
    branches = case['branches']
    total_branches = len(branches)
    total_weight = sum(b['Weight'] for b in branches)
    total_cube = sum(b['Cube'] for b in branches)
    
    print(f"   สาขาทั้งหมด: {total_branches}")
    print(f"   น้ำหนักรวม: {total_weight:.0f} kg")
    print(f"   คิวรวม: {total_cube:.1f} m³")
    
    # จำลองการแบ่ง
    trips = simulate_split(branches)
    
    # ตรวจสอบผลลัพธ์
    assigned_branches = sum(len(t['codes']) for t in trips)
    
    # 1. ตรวจสอบสาขาตกหล่น
    if assigned_branches == total_branches:
        print(f"   ✅ สาขาครบ: {assigned_branches}/{total_branches}")
    else:
        print(f"   ❌ สาขาหาย: {assigned_branches}/{total_branches} (หาย {total_branches - assigned_branches})")
        all_passed = False
    
    # 2. ตรวจสอบ utilization
    for i, trip in enumerate(trips):
        is_last = (i == len(trips) - 1)
        status = "✅" if trip['util'] >= 95 or is_last else "❌"
        
        if trip['util'] < 95 and not is_last:
            all_passed = False
        
        print(f"   {status} ทริป {i+1}: {len(trip['codes'])} สาขา, Util {trip['util']:.1f}%" + 
              (" (รถสุดท้าย)" if is_last else ""))

# สรุปผล
print("\n" + "=" * 80)
print("📊 สรุปผล")
print("=" * 80)

if all_passed:
    print("✅ ทุกกรณีผ่าน - ไม่มีสาขาตกหล่น และรถเต็มตามเกณฑ์!")
else:
    print("❌ มีบางกรณีไม่ผ่าน - ต้องตรวจสอบโค้ดเพิ่มเติม")

print("\n" + "=" * 80)
print("✅ การทดสอบเสร็จสิ้น")
print("=" * 80)
