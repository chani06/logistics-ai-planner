"""
ทดสอบหลังจาก cleanup phases:
1. ทดสอบกับไฟล์ข้อมูลจริง
2. ตรวจสอบ logic 6W ban
3. ตรวจสอบการ sort ทริป
"""
import pandas as pd
import sys
import os

# Import functions from app.py
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 60)
print("🧪 ทดสอบหลัง Cleanup Phases")
print("=" * 60)

# ======================================
# 1. ทดสอบ Logic 6W Ban
# ======================================
print("\n📌 Test 1: ตรวจสอบ Logic 6W Ban")
print("-" * 40)

from app import get_region_type

# nearby provinces (hardcoded for test)
NEARBY_PROVINCES = ['กรุงเทพมหานคร', 'นนทบุรี', 'ปทุมธานี', 'สมุทรปราการ', 'นครปฐม', 'สมุทรสาคร']

# ทดสอบ NEARBY_PROVINCES
print(f"NEARBY_PROVINCES: {NEARBY_PROVINCES}")

# ทดสอบ get_region_type
test_provinces = [
    'กรุงเทพมหานคร',
    'นนทบุรี', 
    'ปทุมธานี',
    'สมุทรปราการ',
    'นครปฐม',
    'สมุทรสาคร',
    'พระนครศรีอยุธยา',
    'สระบุรี',
    'ชลบุรี',
    'ระยอง',
    'เชียงใหม่',
    'ภูเก็ต',
    'นครราชสีมา',
]

print("\n🔍 ทดสอบ get_region_type():")
for prov in test_provinces:
    region = get_region_type(prov)
    ban_6w = "❌ BAN 6W" if region == 'nearby' else "✅ OK 6W"
    print(f"  {prov:25} → {region:10} {ban_6w}")

# ทดสอบ any() vs all() logic
print("\n🔍 ทดสอบ any() vs all() สำหรับ 6W ban:")
test_cases = [
    (['กรุงเทพมหานคร', 'นนทบุรี'], "ทั้งหมด nearby"),
    (['กรุงเทพมหานคร', 'เชียงใหม่'], "ผสม nearby+far"),
    (['เชียงใหม่', 'ภูเก็ต'], "ทั้งหมด far"),
]

for provinces, desc in test_cases:
    regions = [get_region_type(p) for p in provinces]
    
    # any() = ถ้ามีแม้แต่ 1 nearby → ban 6W
    any_nearby = any(r == 'nearby' for r in regions)
    # all() = ทุกตัวต้อง nearby ถึง ban (ผิด!)
    all_nearby = all(r == 'nearby' for r in regions)
    
    correct = "✅ ถูกต้อง" if any_nearby else ""
    print(f"  {desc:25} provinces={provinces}")
    print(f"    any()={any_nearby} (ใช้อันนี้ → BAN 6W ถ้า True)")
    print(f"    all()={all_nearby} (อันนี้ผิด!)")
    print()

# ======================================
# 2. ทดสอบกับไฟล์ข้อมูลจริง
# ======================================
print("\n📌 Test 2: ทดสอบกับไฟล์ข้อมูลจริง")
print("-" * 40)

# โหลดข้อมูลสถานที่
try:
    from app import LOCATION_INFO, LOCATION_COORDS, get_province_from_df
    print(f"✅ โหลด LOCATION_INFO: {len(LOCATION_INFO)} รายการ")
    print(f"✅ โหลด LOCATION_COORDS: {len(LOCATION_COORDS)} รายการ")
    
    # แสดงตัวอย่าง
    sample_codes = list(LOCATION_INFO.keys())[:5]
    print(f"\n📋 ตัวอย่างข้อมูลสถานที่:")
    for code in sample_codes:
        info = LOCATION_INFO.get(code, {})
        print(f"  {code}: {info.get('province', 'N/A')} / {info.get('district', 'N/A')}")
        
except Exception as e:
    print(f"❌ Error loading location data: {e}")

# ทดสอบไฟล์ test.xlsx
test_file = "Dc/test.xlsx"
if os.path.exists(test_file):
    print(f"\n📊 ทดสอบกับ {test_file}:")
    try:
        df = pd.read_excel(test_file)
        print(f"  จำนวนแถว: {len(df)}")
        print(f"  คอลัมน์: {list(df.columns)}")
        
        if 'Code' in df.columns:
            codes = df['Code'].unique()
            print(f"  จำนวน Code: {len(codes)}")
            
            # ตรวจสอบว่าหา province ได้
            found_provinces = 0
            for code in codes[:10]:
                prov = get_province_from_df(df, code) if 'get_province_from_df' in dir() else None
                if not prov:
                    info = LOCATION_INFO.get(code, {})
                    prov = info.get('province', '')
                if prov:
                    found_provinces += 1
            print(f"  พบจังหวัด: {found_provinces}/10 codes แรก")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
else:
    print(f"⚠️ ไม่พบไฟล์ {test_file}")

# ======================================
# 3. ตรวจสอบการ Sort ทริป
# ======================================
print("\n📌 Test 3: ตรวจสอบการ Sort ทริป")
print("-" * 40)

# ทดสอบ region_order และ zone_order
from app import LOGISTICS_ZONES

print(f"📋 LOGISTICS_ZONES มี {len(LOGISTICS_ZONES)} zones:")
for zone, data in list(LOGISTICS_ZONES.items())[:5]:
    print(f"  {zone}: priority={data.get('priority', 'N/A')}, provinces={data.get('provinces', [])[:3]}...")

# ทดสอบ sorting key
region_order = {'south': 1, 'north': 2, 'far': 3, 'nearby': 4, 'other': 5}
print(f"\n📋 region_order (ใต้→เหนือ→ไกล→ใกล้):")
for region, order in sorted(region_order.items(), key=lambda x: x[1]):
    print(f"  {order}. {region}")

# ======================================
# 4. ตรวจสอบว่า any() ถูกใช้ใน code จริง
# ======================================
print("\n📌 Test 4: ตรวจสอบ any() ใน code")
print("-" * 40)

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()
    
# นับจำนวน any() และ all() สำหรับ get_region_type
any_count = content.count("any(get_region_type")
all_count = content.count("all(get_region_type")

print(f"  any(get_region_type...) พบ: {any_count} ครั้ง ✅")
print(f"  all(get_region_type...) พบ: {all_count} ครั้ง {'❌ ควรเป็น 0!' if all_count > 0 else '✅'}")

# หา is_nearby_trip
if "is_nearby_trip = any(" in content:
    print(f"  is_nearby_trip = any(...) ✅ ถูกต้อง")
elif "is_nearby_trip = all(" in content:
    print(f"  is_nearby_trip = all(...) ❌ ผิด!")
else:
    print(f"  ไม่พบ is_nearby_trip pattern")

# หา all_nearby
if "all_nearby = any(" in content:
    print(f"  all_nearby = any(...) ✅ ถูกต้อง")
elif "all_nearby = all(" in content:
    print(f"  all_nearby = all(...) ❌ ผิด!")

print("\n" + "=" * 60)
print("✅ การทดสอบเสร็จสิ้น")
print("=" * 60)
