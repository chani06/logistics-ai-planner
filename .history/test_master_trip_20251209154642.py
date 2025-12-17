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
    
    # อ่านชีต 2.Punthai
    if '2.Punthai' in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name='2.Punthai')
        print(f"   ✅ อ่านชีต 2.Punthai สำเร็จ: {len(df)} แถว")
        print(f"   คอลัมน์: {list(df.columns)[:10]}...")
    else:
        print("   ❌ ไม่พบชีต 2.Punthai")
        df = None
except Exception as e:
    print(f"   ❌ Error: {e}")
    df = None

# อ่าน Auto Plan
print("\n📂 อ่านไฟล์ Auto Plan...")
try:
    xls_auto = pd.ExcelFile(auto_plan_path)
    print(f"   ชีตที่มี: {xls_auto.sheet_names}")
    
    # อ่านชีต info
    if 'info' in xls_auto.sheet_names:
        df_info = pd.read_excel(xls_auto, sheet_name='info')
        print(f"   ✅ อ่านชีต info สำเร็จ: {len(df_info)} แถว")
        print(f"   คอลัมน์: {list(df_info.columns)}")
        
        # หา MaxTruckType
        if 'MaxTruckType' in df_info.columns:
            truck_types = df_info['MaxTruckType'].value_counts()
            print(f"\n   📊 MaxTruckType distribution:")
            for tt, count in truck_types.items():
                print(f"      {tt}: {count} สาขา")
    else:
        print("   ❌ ไม่พบชีต info")
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
