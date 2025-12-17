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

# 1. อ่านไฟล์ข้อมูล Punthai
print("\n📂 1. อ่านไฟล์ข้อมูล Punthai...")
try:
    xls = pd.ExcelFile(DATA_FILE)
    print(f"   ชีตที่มี: {xls.sheet_names}")
    
    # หาชีต 2.Punthai
    target_sheet = None
    for s in xls.sheet_names:
        if 'punthai' in s.lower() or '2.' in s.lower():
            target_sheet = s
            break
    
    if not target_sheet and len(xls.sheet_names) > 0:
        target_sheet = xls.sheet_names[0]
    
    if target_sheet:
        df = pd.read_excel(xls, sheet_name=target_sheet)
        print(f"   ✅ อ่านชีต '{target_sheet}' สำเร็จ: {len(df)} แถว")
        print(f"   คอลัมน์: {list(df.columns)[:8]}")
        
        # หาคอลัมน์สำคัญ
        code_col = None
        cube_col = None
        weight_col = None
        
        for col in df.columns:
            col_str = str(col).lower()
            if 'code' in col_str or 'รหัส' in col_str:
                if not code_col:
                    code_col = col
            elif 'cube' in col_str or 'คิว' in col_str:
                cube_col = col
            elif 'weight' in col_str or 'น้ำหนัก' in col_str or 'wgt' in col_str:
                weight_col = col
        
        print(f"   📊 คอลัมน์ Code: {code_col}")
        print(f"   📊 คอลัมน์ Cube: {cube_col}")
        print(f"   📊 คอลัมน์ Weight: {weight_col}")
        
        # แสดงสถิติ
        if cube_col:
            total_cube = df[cube_col].sum()
            print(f"\n   📈 Total Cube: {total_cube:.2f}")
        if weight_col:
            total_weight = df[weight_col].sum()
            print(f"   📈 Total Weight: {total_weight:.2f}")
            
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
