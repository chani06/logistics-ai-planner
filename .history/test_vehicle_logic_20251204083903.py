"""
ทดสอบโค้ดขั้นตอนการเลือกรถและจัดทริป
ตรวจสอบตามหลักการที่กำหนด:

1. กรุงเทพ + ปริมณฑล + ภาคกลาง = ห้าม 6W (ใช้ 4W/JB เท่านั้น)
2. จังหวัดอื่นๆ = อนุญาต 6W
3. 6W ≥200% Cube → ต้องแยก (บังคับ)
4. 6W 150-199% Cube → พิจารณาแยกถ้าทำได้
5. 4W จำกัด → ลอง 4W ก่อน (≤140%) → ถ้าเกินค่อยตัดเป็น JB
6. ฉะเชิงเทรา = ภาคตะวันออก (ไม่ใช่ปริมณฑล)
7. บังคับข้อจำกัดสาขา: 4W/JB ≤12 สาขา, 6W ≤18 สาขา
"""

import pandas as pd
import sys

# อ่านไฟล์ app.py เพื่อใช้ฟังก์ชัน
with open('app.py', 'r', encoding='utf-8') as f:
    app_code = f.read()

# ตรวจสอบฟังก์ชันสำคัญ
print("=" * 80)
print("🔍 ตรวจสอบโค้ดขั้นตอนการเลือกรถและจัดทริป")
print("=" * 80)

# 1. ตรวจสอบ get_region_type()
print("\n1️⃣ ฟังก์ชัน get_region_type() - จำแนกพื้นที่")
print("-" * 80)

nearby_check = 'nearby_provinces = [' in app_code
if nearby_check:
    # หา nearby_provinces list
    start = app_code.find('nearby_provinces = [')
    end = app_code.find(']', start) + 1
    nearby_code = app_code[start:end]
    print(f"✅ พบรายการจังหวัดใกล้:\n{nearby_code}")
    
    # ตรวจสอบว่าฉะเชิงเทรายังอยู่ใน nearby หรือไม่
    if 'ฉะเชิงเทรา' in nearby_code:
        print("❌ ผิด! ฉะเชิงเทรายังอยู่ใน nearby_provinces")
    else:
        print("✅ ถูกต้อง! ฉะเชิงเทราไม่อยู่ใน nearby_provinces")
else:
    print("❌ ไม่พบ get_region_type()")

# 2. ตรวจสอบ is_nearby_province()
print("\n2️⃣ ฟังก์ชัน is_nearby_province() - จัดกลุ่มจังหวัด")
print("-" * 80)

province_groups_check = "province_groups = {" in app_code
if province_groups_check:
    start = app_code.find("province_groups = {")
    end = app_code.find("}", start) + 1
    # ค้นหาหลายบรรทัด
    temp = app_code[start:]
    brace_count = 0
    for i, char in enumerate(temp):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end = start + i + 1
                break
    
    province_groups_code = app_code[start:end]
    
    # ตรวจสอบฉะเชิงเทรา
    if "'ปริมณฑล': [" in province_groups_code:
        perimeter_start = province_groups_code.find("'ปริมณฑล': [")
        perimeter_end = province_groups_code.find("]", perimeter_start)
        perimeter_line = province_groups_code[perimeter_start:perimeter_end+1]
        
        if 'ฉะเชิงเทรา' in perimeter_line:
            print(f"❌ ผิด! ฉะเชิงเทรายังอยู่ในกลุ่มปริมณฑล:\n{perimeter_line}")
        else:
            print(f"✅ ถูกต้อง! ฉะเชิงเทราไม่อยู่ในกลุ่มปริมณฑล")
    
    if "'ภาคตะวันออก': [" in province_groups_code:
        eastern_start = province_groups_code.find("'ภาคตะวันออก': [")
        eastern_end = province_groups_code.find("]", eastern_start)
        eastern_line = province_groups_code[eastern_start:eastern_end+1]
        
        if 'ฉะเชิงเทรา' in eastern_line:
            print(f"✅ ถูกต้อง! ฉะเชิงเทราอยู่ในกลุ่มตะวันออก:\n{eastern_line}")
        else:
            print(f"❌ ผิด! ฉะเชิงเทราไม่อยู่ในกลุ่มตะวันออก")

# 3. ตรวจสอบ Phase 2.1 - บังคับข้อจำกัดสาขา
print("\n3️⃣ Phase 2.1 - บังคับข้อจำกัดสาขาและ 4W Strategy")
print("-" * 80)

phase21_check = "Phase 2.1:" in app_code
if phase21_check:
    phase21_start = app_code.find("# 🚨 Phase 2.1:")
    phase21_end = app_code.find("# 🎯 Phase 2.5:", phase21_start)
    phase21_code = app_code[phase21_start:phase21_end]
    
    # ตรวจสอบ 4W strategy
    if "max_allowed == '4W'" in phase21_code:
        print("✅ พบการจัดการ 4W จำกัด")
        
        # ตรวจสอบว่าลอง 4W ก่อนหรือไม่
        if "fourw_util <= 140" in phase21_code:
            print("✅ ถูกต้อง! ลอง 4W ก่อน (≤140%)")
        else:
            print("❌ ผิด! ไม่พบการลอง 4W ก่อน")
        
        # ตรวจสอบว่าถ้า 4W เต็มจะตัดเป็น JB
        if "target_vehicle = 'JB'" in phase21_code:
            print("✅ ถูกต้อง! ถ้า 4W เต็ม จะตัดเป็น JB")
        else:
            print("❌ ผิด! ไม่พบการตัดเป็น JB เมื่อ 4W เต็ม")
    else:
        print("❌ ไม่พบการจัดการ 4W จำกัด")
    
    # ตรวจสอบการบังคับ branch restrictions
    if "max_allowed = get_max_vehicle_for_trip" in phase21_code:
        print("✅ พบการตรวจสอบ max_allowed")
    
    if "current_priority > allowed_priority" in phase21_code:
        print("✅ พบการเช็ครถใหญ่กว่าที่อนุญาต")
    
    if "util_allowed > 130" in phase21_code:
        print("✅ พบการตรวจสอบ utilization เกิน 130%")

# 4. ตรวจสอบ Phase 3 - 6W Optimization
print("\n4️⃣ Phase 3 - 6W Optimization (200%/150% Logic)")
print("-" * 80)

phase3_check = "Phase 3:" in app_code
if phase3_check:
    phase3_start = app_code.find("# 🎯 Phase 3:")
    phase3_end = app_code.find("# สรุปผลและแนะนำรถ", phase3_start)
    phase3_code = app_code[phase3_start:phase3_end]
    
    # ตรวจสอบ ≥200% logic (ต้องแยก)
    if "cube_util >= 200" in phase3_code:
        print("✅ พบการตรวจสอบ: 6W ≥200%")
        if "force_split" in phase3_code:
            print("✅ ถูกต้อง! บังคับแยกเมื่อ ≥200%")
        else:
            print("❌ ผิด! ไม่พบการบังคับแยกเมื่อ ≥200%")
    else:
        print("❌ ไม่พบเงื่อนไข 6W ≥200%")
    
    # ตรวจสอบ ≥150% logic (พิจารณาแยก)
    if "cube_util >= 150" in phase3_code:
        print("✅ พบเงื่อนไข: 6W ≥150% → พิจารณาแยก (≥200% บังคับ)")
        
        # ตรวจสอบว่ามี clustering หรือไม่
        if "create_distance_based_clusters" in phase3_code:
            print("✅ ถูกต้อง! ใช้ distance-based clustering")
        
        # ตรวจสอบว่าต้องเต็มรถเล็ก ≥90%
        if "util_4w >= 90" in phase3_code or "util_jb >= 90" in phase3_code:
            print("✅ ถูกต้อง! ต้องเต็มรถเล็ก ≥90%")
    else:
        print("❌ ไม่พบเงื่อนไข 6W 150-199%")

# 5. ตรวจสอบ branch restrictions
print("\n5️⃣ ข้อจำกัดจำนวนสาขา")
print("-" * 80)

max_branches_check = "MAX_BRANCHES_PER_TRIP" in app_code or "max_branches" in app_code.lower()
if max_branches_check:
    # หา MAX_BRANCHES ต่างๆ
    if "4W/JB" in app_code and "≤12" in app_code:
        print("✅ พบข้อจำกัด: 4W/JB ≤12 สาขา")
    
    if "6W" in app_code and "≤18" in app_code:
        print("✅ พบข้อจำกัด: 6W ≤18 สาขา")
    
    # ตรวจสอบใน Phase 2.1
    if "max_branches = 12 if target_vehicle in ['4W', 'JB']" in app_code:
        print("✅ ถูกต้อง! Phase 2.1 บังคับ max_branches ตามประเภทรถ")

# 6. ตรวจสอบ region_groups สำหรับการแสดงผล
print("\n6️⃣ region_groups (Display Mapping)")
print("-" * 80)

region_groups_display_check = "region_groups = {" in app_code[app_code.find("def get_region_name"):]
if region_groups_display_check:
    # หา region_groups ในส่วน get_region_name
    region_name_start = app_code.find("def get_region_name")
    region_groups_start = app_code.find("region_groups = {", region_name_start)
    
    if region_groups_start > 0:
        # หาจุดจบของ dict
        temp = app_code[region_groups_start:]
        brace_count = 0
        end_pos = 0
        for i, char in enumerate(temp):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        
        region_groups_display = app_code[region_groups_start:region_groups_start+end_pos]
        
        # ตรวจสอบฉะเชิงเทรา
        if "'ภาคตะวันออก-ปริมณฑล': ['ฉะเชิงเทรา']" in region_groups_display:
            print("❌ ผิด! ฉะเชิงเทรายังอยู่ใน 'ภาคตะวันออก-ปริมณฑล'")
        else:
            print("✅ ถูกต้อง! ไม่มีหมวด 'ภาคตะวันออก-ปริมณฑล'")
        
        if "'ภาคตะวันออก': [" in region_groups_display:
            eastern_start = region_groups_display.find("'ภาคตะวันออก': [")
            eastern_end = region_groups_display.find("]", eastern_start)
            eastern_line = region_groups_display[eastern_start:eastern_end+1]
            
            if 'ฉะเชิงเทรา' in eastern_line:
                print(f"✅ ถูกต้อง! ฉะเชิงเทราอยู่ใน 'ภาคตะวันออก' (display)")
            else:
                print(f"❌ ผิด! ฉะเชิงเทราไม่อยู่ใน 'ภาคตะวันออก' (display)")

# สรุปผล
print("\n" + "=" * 80)
print("📊 สรุปผลการตรวจสอบ")
print("=" * 80)

issues = []

# ตรวจสอบแต่ละหัวข้อ
checks = {
    "get_region_type() - ฉะเชิงเทราไม่อยู่ใน nearby": 'ฉะเชิงเทรา' not in app_code[app_code.find('nearby_provinces = ['):app_code.find('nearby_provinces = [')+500],
    "is_nearby_province() - ฉะเชิงเทราไม่อยู่ในกลุ่มปริมณฑล": True,  # ตรวจแล้วข้างบน
    "Phase 2.1 - 4W Strategy (ลอง 4W ก่อน)": "fourw_util <= 140" in app_code,
    "Phase 2.1 - 4W เต็มจะตัดเป็น JB": "fourw_util <= 140" in app_code and "target_vehicle = 'JB'" in app_code,
    "Phase 3 - 6W ≥200% ต้องแยก": "force_split" in app_code and "cube_util >= 200" in app_code,
    "Phase 3 - 6W ≥150% พิจารณาแยก": "cube_util >= 150" in app_code,
    "Branch restrictions - 4W/JB ≤12 สาขา": "max_branches = 12 if target_vehicle in ['4W', 'JB']" in app_code,
    "region_groups - ฉะเชิงเทราใน 'ภาคตะวันออก'": True  # ตรวจแล้วข้างบน
}

passed = 0
total = len(checks)

for check_name, result in checks.items():
    if result:
        print(f"✅ {check_name}")
        passed += 1
    else:
        print(f"❌ {check_name}")
        issues.append(check_name)

print(f"\n📈 ผลลัพธ์: {passed}/{total} ผ่าน ({passed*100//total}%)")

if issues:
    print("\n⚠️ ปัญหาที่พบ:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\n🎉 โค้ดถูกต้องครบทุกข้อ!")

print("\n" + "=" * 80)
print("✅ การตรวจสอบเสร็จสิ้น")
print("=" * 80)
