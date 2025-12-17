"""
🚛 ทดสอบจัดทริปจากไฟล์จริงในหลังบ้าน
"""
import pandas as pd
import sys
import os

# Import functions จาก app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ไฟล์ที่จะใช้
DATA_FILE = r"Dc\แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx"
MASTER_FILE = r"Dc\Master สถานที่ส่ง.xlsx"
AUTO_PLAN_FILE = r"Dc\Auto planning (1).xlsx"

print("=" * 70)
print("🚛 ทดสอบจัดทริปจากไฟล์ Punthai")
print("=" * 70)

# 1. อ่านไฟล์ข้อมูล Punthai (header อยู่แถว 2)
print("\n📂 1. อ่านไฟล์ข้อมูล Punthai...")
try:
    # อ่านด้วย header=1 (แถว 2)
    df = pd.read_excel(DATA_FILE, sheet_name='2.Punthai', header=1)
    print(f"   ✅ อ่านชีต '2.Punthai' สำเร็จ: {len(df)} แถว")
    print(f"   คอลัมน์: {list(df.columns)[:10]}")
    
    # กรองแถวที่มีข้อมูล
    if 'BranchCode' in df.columns:
        df_valid = df[df['BranchCode'].notna()].copy()
        print(f"   ✅ แถวที่มีข้อมูล: {len(df_valid)} สาขา")
        
        # แสดงสถิติ
        total_cube = df_valid['TOTALCUBE'].sum() if 'TOTALCUBE' in df_valid.columns else 0
        total_weight = df_valid['TOTALWGT'].sum() if 'TOTALWGT' in df_valid.columns else 0
        trip_count = df_valid['Trip'].nunique() if 'Trip' in df_valid.columns else 0
        
        print(f"\n   📈 สถิติ:")
        print(f"      Total Cube: {total_cube:.2f}")
        print(f"      Total Weight: {total_weight:.2f}")
        print(f"      จำนวนทริป (เดิม): {trip_count}")
        
        # แสดงทริปที่มี
        if 'Trip no' in df_valid.columns:
            trip_nos = df_valid['Trip no'].value_counts()
            print(f"\n   📊 Trip no distribution (Top 10):")
            for trip_no, count in trip_nos.head(10).items():
                print(f"      {trip_no}: {count} สาขา")
                
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# 2. อ่าน Auto Plan
print("\n📂 2. อ่านไฟล์ Auto Plan (MaxTruckType)...")
try:
    xls_auto = pd.ExcelFile(AUTO_PLAN_FILE)
    
    # หาชีต Info
    info_sheet = None
    for s in xls_auto.sheet_names:
        if 'info' in s.lower():
            info_sheet = s
            break
    
    if info_sheet:
        df_info = pd.read_excel(xls_auto, sheet_name=info_sheet)
        
        # หาคอลัมน์ MaxTruckType
        truck_col = None
        for col in df_info.columns:
            if 'maxtruck' in str(col).lower():
                truck_col = col
                break
        
        if truck_col:
            truck_dist = df_info[truck_col].value_counts()
            print(f"   ✅ พบ MaxTruckType: {len(df_info)} สาขา")
            print(f"   📊 Distribution:")
            for tt, count in truck_dist.head(10).items():
                print(f"      {tt}: {count} สาขา")
                
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. อ่าน Master
print("\n📂 3. อ่านไฟล์ Master (พิกัด, จังหวัด)...")
try:
    df_master = pd.read_excel(MASTER_FILE)
    print(f"   ✅ อ่านสำเร็จ: {len(df_master)} สาขา")
    
    # แสดงจังหวัดที่มี
    if 'จังหวัด' in df_master.columns:
        provinces = df_master['จังหวัด'].value_counts()
        print(f"   📊 จังหวัด Top 10:")
        for prov, count in provinces.head(10).items():
            print(f"      {prov}: {count} สาขา")
            
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("✅ ทดสอบอ่านไฟล์เสร็จสิ้น - พร้อมจัดทริป!")
print("=" * 70)
