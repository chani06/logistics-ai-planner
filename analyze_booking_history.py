# -*- coding: utf-8 -*-
"""วิเคราะห์ไฟล์ประวัติงานจัดส่ง DC วังน้อย - เรียนรู้ความสัมพันธ์สาขา-รถ"""
import pandas as pd
import sys

print("="*70)
print("📋 วิเคราะห์ประวัติการจัดส่ง DC วังน้อย")
print("="*70)

try:
    # โหลดไฟล์ประวัติ
    file_path = 'Dc/ประวัติงานจัดส่ง DC วังน้อย(1).xlsx'
    print(f"\nโหลดไฟล์: {file_path}")
    
    # อ่าน sheet ทั้งหมด
    excel_file = pd.ExcelFile(file_path)
    print(f"Sheet ทั้งหมด: {excel_file.sheet_names}")
    
    # อ่าน sheet แรก
    df = pd.read_excel(file_path, sheet_name=0)
    print(f"\n✅ โหลดข้อมูลสำเร็จ: {len(df)} แถว")
    
    # แสดงโครงสร้าง
    print("\n" + "="*70)
    print("📊 โครงสร้างข้อมูล")
    print("="*70)
    print(f"จำนวนคอลัมน์: {len(df.columns)}")
    print("\nรายชื่อคอลัมน์:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # แสดงตัวอย่างข้อมูล 3 แถวแรก
    print("\n" + "="*70)
    print("📝 ตัวอย่างข้อมูล (3 แถวแรก)")
    print("="*70)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.head(3))
    
    # วิเคราะห์คอลัมน์สำคัญ
    print("\n" + "="*70)
    print("🔍 วิเคราะห์คอลัมน์สำคัญ")
    print("="*70)
    
    # หาคอลัมน์ที่เกี่ยวข้องกับบุ๊คกิ้ง
    booking_cols = [col for col in df.columns if 'book' in col.lower() or 'บุ๊ค' in col.lower()]
    if booking_cols:
        print(f"\nคอลัมน์บุ๊คกิ้ง: {booking_cols}")
        for col in booking_cols:
            print(f"  {col}: {df[col].nunique()} unique values")
            print(f"  ตัวอย่าง: {df[col].dropna().head(3).tolist()}")
    
    # หาคอลัมน์สาขา
    branch_cols = [col for col in df.columns if 'branch' in col.lower() or 'สาขา' in col.lower() or 'code' in col.lower()]
    if branch_cols:
        print(f"\nคอลัมน์สาขา: {branch_cols}")
        for col in branch_cols:
            print(f"  {col}: {df[col].nunique()} unique values")
    
    # หาคอลัมน์รถ
    vehicle_cols = [col for col in df.columns if 'truck' in col.lower() or 'vehicle' in col.lower() or 'รถ' in col.lower() or 'trip' in col.lower()]
    if vehicle_cols:
        print(f"\nคอลัมน์รถ/เที่ยว: {vehicle_cols}")
        for col in vehicle_cols:
            print(f"  {col}: {df[col].nunique()} unique values")
            print(f"  ตัวอย่าง: {df[col].dropna().head(5).tolist()}")
    
    # หาคอลัมน์น้ำหนัก/ลูกบาศก์
    weight_cols = [col for col in df.columns if any(x in col.lower() for x in ['weight', 'wgt', 'น้ำหนัก', 'kg'])]
    if weight_cols:
        print(f"\nคอลัมน์น้ำหนัก: {weight_cols}")
    
    cube_cols = [col for col in df.columns if any(x in col.lower() for x in ['cube', 'cbm', 'ลูกบาศก์', 'm3'])]
    if cube_cols:
        print(f"\nคอลัมน์ลูกบาศก์: {cube_cols}")
    
    print("\n" + "="*70)
    print("💡 คำแนะนำ")
    print("="*70)
    print("กรุณาระบุคอลัมน์ที่ถูกต้อง:")
    print("  - เลขบุ๊คกิ้ง (Booking Number)")
    print("  - รหัสสาขา (Branch Code)")
    print("  - ประเภทรถ (Vehicle Type: 4W/JB/6W)")
    print("  - น้ำหนัก (Weight)")
    print("  - ลูกบาศก์ (Cube)")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
