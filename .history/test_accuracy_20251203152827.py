"""
Script สำหรับทดสอบความถูกต้องของระบบจัดทริป
เปรียบเทียบผลลัพธ์กับแผน Punthai
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys

print("=" * 80)
print("🧪 ระบบทดสอบความถูกต้องการจัดทริป")
print("=" * 80)
print()

# โหลดข้อมูล Test
print("📂 โหลดข้อมูล Test...")
print("   💡 ใช้ไฟล์ Excel ที่อัปโหลดผ่าน Streamlit")
print("   📝 ขั้นตอน:")
print("      1. เปิด Streamlit: streamlit run app.py")
print("      2. อัปโหลดไฟล์ Excel")
print("      3. Export ผลลัพธ์")
print("      4. รัน: python compare_results.py")

# โหลดข้อมูล Punthai (แผนจริง)
print("📂 โหลดข้อมูล Punthai (แผนจริง)...")
try:
    punthai_file = 'Dc/Punthai_reference.xlsx'
    df_punthai = pd.read_excel(punthai_file)
    print(f"   ✅ โหลดสำเร็จ: {len(df_punthai)} สาขา, {df_punthai['Trip'].nunique()} ทริป")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

print()
print("-" * 80)
print("📊 วิเคราะห์ข้อมูล")
print("-" * 80)

# วิเคราะห์ Punthai
punthai_stats = {
    'total_branches': len(df_punthai),
    'total_trips': df_punthai['Trip'].nunique(),
    'avg_branches_per_trip': len(df_punthai) / df_punthai['Trip'].nunique(),
}

print(f"\n📋 Punthai (แผนจริง):")
print(f"   - จำนวนสาขา: {punthai_stats['total_branches']}")
print(f"   - จำนวนทริป: {punthai_stats['total_trips']}")
print(f"   - เฉลี่ยสาขา/ทริป: {punthai_stats['avg_branches_per_trip']:.1f}")

# วิเคราะห์การกระจายรถใน Punthai
if 'Vehicle_Type' in df_punthai.columns:
    print(f"\n🚛 การใช้รถใน Punthai:")
    vehicle_counts = df_punthai.groupby('Vehicle_Type').size()
    trip_counts = df_punthai.groupby(['Trip', 'Vehicle_Type']).size().reset_index()
    trip_vehicle_counts = trip_counts.groupby('Vehicle_Type').size()
    
    for vehicle, count in vehicle_counts.items():
        trips = trip_vehicle_counts.get(vehicle, 0)
        print(f"   - {vehicle}: {count} สาขา ({trips} ทริป)")

print()
print("-" * 80)
print("🔍 เปรียบเทียบผลลัพธ์")
print("-" * 80)

# จำลองการจัดทริปด้วย AI (ใช้ app.py logic)
print("\n⚙️  กำลังจัดทริปด้วย AI...")
print("   (ต้องรันผ่าน app.py เพื่อได้ผลลัพธ์)")
print("   ใช้คำสั่ง: python app.py --test-mode")

print()
print("-" * 80)
print("📈 สูตรคำนวณความถูกต้อง")
print("-" * 80)
print("""
1. Trip Matching Accuracy:
   - ตรวจสอบว่าสาขาที่อยู่ทริปเดียวกันใน Punthai
   - อยู่ทริปเดียวกันใน AI ด้วยหรือไม่
   - Accuracy = (Correct Pairs) / (Total Pairs)

2. Vehicle Assignment Accuracy:
   - เปรียบเทียบประเภทรถที่ใช้
   - Accuracy = (Correct Vehicle) / (Total Trips)

3. Branch Count per Trip:
   - เปรียบเทียบจำนวนสาขาต่อทริป
   - MAE = Mean Absolute Error

4. Overall Score:
   - คะแนนรวมจากทุกตัวชี้วัด
""")

print()
print("-" * 80)
print("💡 วิธีใช้งาน")
print("-" * 80)
print("""
1. รันสคริปต์นี้: python test_accuracy.py
2. อัปโหลดไฟล์ Test ไปที่เว็บ Streamlit
3. Export ผลลัพธ์เป็น Excel
4. รัน: python compare_results.py
   - จะเปรียบเทียบผลลัพธ์กับ Punthai
   - แสดงค่า Accuracy
""")

print()
print("=" * 80)
print("✅ การวิเคราะห์เบื้องต้นเสร็จสิ้น")
print("=" * 80)
