"""
ทดสอบ Simple Trip Planner
"""

import pandas as pd
import sys
from simple_trip_planner import simple_plan_trips, export_to_excel_with_colors

print("=" * 80)
print("ทดสอบการจัดทริปแบบใหม่ (Simple)")
print("=" * 80)

# โหลดไฟล์
input_file = "Dc/test.xlsx"
master_file = "Dc/Master สถานที่ส่ง.xlsx"

print(f"\n📂 โหลดไฟล์: {input_file}")
df = pd.read_excel(input_file)
print(f"✅ โหลดสำเร็จ: {len(df)} รายการ")

# แก้ไขชื่อคอลัมน์
column_mapping = {
    'สาขา': 'Code',
    'ชื่อสาขา': 'Name',
    'TOTALWGT': 'Weight',
    'TOTALCUBE': 'Cube'
}

for old_col, new_col in column_mapping.items():
    if old_col in df.columns and new_col not in df.columns:
        df.rename(columns={old_col: new_col}, inplace=True)

print(f"📋 คอลัมน์: {df.columns.tolist()}")

# โหลด Master
print(f"\n📖 โหลด Master: {master_file}")
master_df = pd.read_excel(master_file)
print(f"✅ โหลดสำเร็จ: {len(master_df)} รายการ")

# จัดทริป
print("\n🔄 กำลังจัดทริป...")
result_df, summary_df = simple_plan_trips(df, master_df)

print("\n" + "=" * 80)
print("✅ จัดทริปเสร็จสิ้น!")
print("=" * 80)

# แสดงสรุป
print(f"\n📊 สรุปผลการจัดทริป:")
print(f"- จำนวนทริป: {len(summary_df)}")
print(f"- จำนวนสาขา: {len(result_df)}")
print(f"- เฉลี่ยสาขา/ทริป: {len(result_df)/len(summary_df):.1f}")

print("\n📋 สรุปแต่ละทริป (10 ทริปแรก):")
print(summary_df.head(10).to_string(index=False))

# แสดงตัวอย่างทริป
print("\n🔍 ตัวอย่างทริป 1:")
trip1 = result_df[result_df['Trip'] == 1][['Code', 'Name', 'Cube', 'Province', 'District', 'Subdistrict', 'Is_Punthai']]
print(trip1.to_string(index=False))

# ตรวจสอบสาขาฟิวเจอร์รังสิต
print("\n🔍 ตรวจสอบสาขาฟิวเจอร์รังสิต:")
future_branches = result_df[result_df['Base_Name'].str.contains('ฟิวเจอร์', na=False)]
if len(future_branches) > 0:
    print(future_branches[['Trip', 'Code', 'Name', 'Province', 'District', 'Subdistrict']].to_string(index=False))
    
    # เช็คว่าอยู่ทริปเดียวกันหรือไม่
    trips = future_branches['Trip'].unique()
    if len(trips) == 1:
        print(f"✅ ฟิวเจอร์รังสิตอยู่ทริปเดียวกัน (Trip {trips[0]})")
    else:
        print(f"⚠️ ฟิวเจอร์รังสิตแยกกัน {len(trips)} ทริป: {trips}")

# Export
output_file = "Dc/test_output_simple.xlsx"
print(f"\n💾 บันทึกผล: {output_file}")
try:
    export_to_excel_with_colors(result_df, output_file, input_file)
except Exception as e:
    print(f"⚠️ ไม่สามารถ export แบบมีสี: {e}")
    print("💾 บันทึกแบบธรรมดา...")
    result_df.to_excel(output_file, index=False)
    print("✅ บันทึกเสร็จสิ้น")

print("\n" + "=" * 80)
print("เสร็จสิ้น!")
print("=" * 80)
