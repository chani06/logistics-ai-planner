"""
ทดสอบการจัดทริปผ่าน Backend (ไม่ต้องใช้ Streamlit UI)
"""
import pandas as pd
import sys
import time

# Import functions จาก app.py
print("📦 กำลังโหลด modules...")
start_load = time.time()

# Suppress streamlit warnings
import warnings
warnings.filterwarnings('ignore')

# Mock streamlit functions
class MockStreamlit:
    def cache_data(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def warning(self, msg):
        print(f"⚠️ {msg}")
    
    def info(self, msg):
        print(f"ℹ️ {msg}")
    
    def error(self, msg):
        print(f"❌ {msg}")
    
    def success(self, msg):
        print(f"✅ {msg}")

sys.modules['streamlit'] = MockStreamlit()
import streamlit as st
st.cache_data = MockStreamlit().cache_data

# Now import app
from app import (
    predict_trips, 
    load_master_data, 
    load_booking_history_restrictions,
    load_punthai_reference,
    LIMITS, MIN_UTIL, BUFFER, MAX_DISTANCE_IN_TRIP
)

print(f"✅ โหลด modules เสร็จ ({time.time() - start_load:.2f} วินาที)")

# แสดง config
print("\n" + "="*60)
print("📋 CONFIG ปัจจุบัน:")
print("="*60)
print(f"  LIMITS:")
for vehicle, limits in LIMITS.items():
    print(f"    {vehicle}: max_w={limits['max_w']}, max_c={limits['max_c']}")
print(f"  MIN_UTIL: {MIN_UTIL}")
print(f"  BUFFER: {BUFFER}")
print(f"  MAX_DISTANCE_IN_TRIP: {MAX_DISTANCE_IN_TRIP} km")

# โหลดไฟล์ทดสอบ
print("\n" + "="*60)
print("📂 กำลังโหลดไฟล์ Punthai...")
print("="*60)

try:
    file_path = 'Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx'
    df = pd.read_excel(file_path, sheet_name='2.Punthai', header=1)
    print(f"✅ โหลดไฟล์สำเร็จ: {len(df)} แถว")
    
    # ทำความสะอาดข้อมูล
    df = df[df['Trip'].notna()].copy()
    
    # Rename columns
    rename_map = {}
    for col in df.columns:
        col_upper = str(col).upper().strip()
        if 'BRANCHCODE' in col_upper or col == 'รหัสสาขา':
            rename_map[col] = 'Code'
        elif 'BRANCH NAME' in col_upper or 'ชื่อสาขา' in col_upper or col == 'สาขา':
            rename_map[col] = 'Name'
        elif 'CUBE' in col_upper or 'คิว' in col_upper:
            rename_map[col] = 'Cube'
        elif 'WEIGHT' in col_upper or 'WGT' in col_upper or 'น้ำหนัก' in col_upper:
            rename_map[col] = 'Weight'
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # ตรวจสอบ columns
    print(f"📋 Columns: {list(df.columns[:15])}")
    
    # กรอง DC ออก
    if 'Code' in df.columns:
        df = df[~df['Code'].isin(['DC011', 'PTDC', 'PTG Distribution Center'])]
    
    # แปลงเป็น numeric
    if 'Weight' in df.columns:
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce').fillna(0)
    if 'Cube' in df.columns:
        df['Cube'] = pd.to_numeric(df['Cube'], errors='coerce').fillna(0)
    
    print(f"📊 หลังกรอง: {len(df)} สาขา")
    print(f"   Total Weight: {df['Weight'].sum():.2f} kg")
    print(f"   Total Cube: {df['Cube'].sum():.2f} m³")
    
except Exception as e:
    print(f"❌ Error loading file: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# สร้าง model_data (mock)
print("\n" + "="*60)
print("🧠 กำลังสร้าง Model Data...")
print("="*60)

model_data = {
    'model': None,
    'trip_pairs': set(),
    'branch_info': {},
    'trip_vehicles': {},
    'branch_vehicles': {}
}

# เพิ่ม branch_info
for code in df['Code'].unique():
    code_data = df[df['Code'] == code]
    model_data['branch_info'][code] = {
        'avg_weight': code_data['Weight'].mean(),
        'avg_cube': code_data['Cube'].mean(),
        'total_trips': 1,
        'province': 'UNKNOWN',
        'latitude': 0.0,
        'longitude': 0.0
    }

print(f"✅ สร้าง branch_info: {len(model_data['branch_info'])} สาขา")

# ทดสอบการจัดทริป
print("\n" + "="*60)
print("🚚 กำลังจัดทริป...")
print("="*60)

start_time = time.time()

try:
    result_df, summary = predict_trips(df.copy(), model_data)
    elapsed = time.time() - start_time
    
    print(f"\n✅ จัดทริปเสร็จ! ใช้เวลา {elapsed:.2f} วินาที")
    print(f"   จำนวนทริป: {len(summary)}")
    print(f"   จำนวนสาขา: {len(result_df)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# แสดงผลลัพธ์
print("\n" + "="*60)
print("📊 สรุปผลการจัดทริป")
print("="*60)

# นับประเภทรถ
vehicle_counts = {}
for _, row in summary.iterrows():
    truck = row['Truck'].split()[0] if row['Truck'] else 'Unknown'
    vehicle_counts[truck] = vehicle_counts.get(truck, 0) + 1

print(f"\n🚛 จำนวนรถแต่ละประเภท:")
for vehicle, count in sorted(vehicle_counts.items()):
    print(f"   {vehicle}: {count} คัน")

# แสดง status
print(f"\n📋 สถานะทริป:")
if 'Status' in summary.columns:
    status_counts = summary['Status'].value_counts()
    for status, count in status_counts.items():
        print(f"   {status}: {count} ทริป")
else:
    print("   (ไม่มีคอลัมน์ Status)")

# แสดงทริปที่ไม่ผ่าน
print(f"\n⚠️ ทริปที่ไม่ผ่านเกณฑ์:")
if 'Status' in summary.columns:
    failed = summary[summary['Status'] != '✅ ผ่าน']
    if len(failed) > 0:
        for _, row in failed.head(10).iterrows():
            print(f"   Trip {row['Trip']}: {row['Truck'].split()[0]} - W:{row['Weight_Use%']:.1f}% C:{row['Cube_Use%']:.1f}% - {row['Status']}")
        if len(failed) > 10:
            print(f"   ... และอีก {len(failed) - 10} ทริป")
    else:
        print("   ✅ ทุกทริปผ่านเกณฑ์!")
else:
    # ตรวจสอบเอง
    for _, row in summary.iterrows():
        truck = row['Truck'].split()[0] if row['Truck'] else '4W'
        w_util = row.get('Weight_Use%', 0)
        c_util = row.get('Cube_Use%', 0)
        
        if w_util > 100 or c_util > 100:
            print(f"   Trip {row['Trip']}: {truck} - W:{w_util:.1f}% C:{c_util:.1f}% - 🚫 เกิน100%")

# แสดง utilization เฉลี่ย
print(f"\n📈 Utilization เฉลี่ย:")
print(f"   Weight: {summary['Weight_Use%'].mean():.1f}%")
print(f"   Cube: {summary['Cube_Use%'].mean():.1f}%")
if 'Max_Util%' in summary.columns:
    print(f"   Max: {summary['Max_Util%'].mean():.1f}%")

# แสดงตัวอย่าง 10 ทริปแรก
print(f"\n📋 ตัวอย่าง 10 ทริปแรก:")
print("-" * 80)
cols_to_show = ['Trip', 'Branches', 'Weight', 'Cube', 'Truck', 'Weight_Use%', 'Cube_Use%']
if 'Status' in summary.columns:
    cols_to_show.append('Status')
cols_to_show = [c for c in cols_to_show if c in summary.columns]

for _, row in summary.head(10).iterrows():
    truck = row['Truck'].split()[0] if row['Truck'] else 'Unknown'
    status = row.get('Status', '')
    print(f"  Trip {row['Trip']:3.0f}: {truck:3s} | {row['Branches']:2.0f} สาขา | W:{row['Weight']:7.1f}kg ({row['Weight_Use%']:5.1f}%) | C:{row['Cube']:5.2f}m³ ({row['Cube_Use%']:5.1f}%) | {status}")

print("\n" + "="*60)
print("🏁 ทดสอบเสร็จสิ้น!")
print("="*60)
