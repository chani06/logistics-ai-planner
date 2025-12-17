"""
ทดสอบจัดทริปจากไฟล์ Master สถานที่ส่ง
"""
import pandas as pd
import sys
import os

# เพิ่ม path สำหรับ import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# อ่านไฟล์ Master
master_path = r"Dc\Master สถานที่ส่ง.xlsx"
auto_plan_path = r"Dc\Auto planning (1).xlsx"

print("=" * 60)
print("🚛 ทดสอบจัดทริปจากไฟล์ Master")
print("=" * 60)

# อ่าน Master
print("\n📂 อ่านไฟล์ Master...")
try:
    xls = pd.ExcelFile(master_path)
    print(f"   ชีตที่มี: {xls.sheet_names}")
    
    # อ่านชีต 2.Punthai หรือ Sheet1
    target_sheet = None
    for sheet in ['2.Punthai', 'Sheet1', 'Punthai']:
        if sheet in xls.sheet_names:
            target_sheet = sheet
            break
    
    if target_sheet:
        df = pd.read_excel(xls, sheet_name=target_sheet)
        print(f"   ✅ อ่านชีต {target_sheet} สำเร็จ: {len(df)} แถว")
        print(f"   คอลัมน์: {list(df.columns)[:10]}...")
    else:
        print("   ❌ ไม่พบชีตที่ต้องการ")
        df = None
except Exception as e:
    print(f"   ❌ Error: {e}")
    df = None

# อ่าน Auto Plan
print("\n📂 อ่านไฟล์ Auto Plan...")
try:
    xls_auto = pd.ExcelFile(auto_plan_path)
    print(f"   ชีตที่มี: {xls_auto.sheet_names}")
    
    # อ่านชีต info หรือ Info
    info_sheet = None
    for sheet in ['info', 'Info', 'INFO']:
        if sheet in xls_auto.sheet_names:
            info_sheet = sheet
            break
    
    if info_sheet:
        df_info = pd.read_excel(xls_auto, sheet_name=info_sheet)
        print(f"   ✅ อ่านชีต {info_sheet} สำเร็จ: {len(df_info)} แถว")
        print(f"   คอลัมน์: {list(df_info.columns)}")
        
        # หา MaxTruckType
        max_truck_col = None
        for col in df_info.columns:
            if 'maxtruck' in str(col).lower() or 'truck' in str(col).lower():
                max_truck_col = col
                break
        
        if max_truck_col:
            truck_types = df_info[max_truck_col].value_counts()
            print(f"\n   📊 {max_truck_col} distribution:")
            for tt, count in truck_types.items():
                print(f"      {tt}: {count} สาขา")
        else:
            print(f"\n   ⚠️ ไม่พบคอลัมน์ MaxTruckType")
    else:
        print("   ❌ ไม่พบชีต info/Info")
        df_info = None
except Exception as e:
    print(f"   ❌ Error: {e}")
    df_info = None

# แสดงตัวอย่างข้อมูล
if df is not None and len(df) > 0:
    print("\n📊 ตัวอย่างข้อมูลจาก Master:")
    print(df.head(10).to_string())

print("\n" + "=" * 60)
print("✅ ทดสอบอ่านไฟล์เสร็จสิ้น")
print("=" * 60)
