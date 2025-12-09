"""
ทดสอบ Simple Trip Planner V2
"""

import pandas as pd
import sys
import os
import io

# แก้ encoding สำหรับ Windows/PowerShell
if sys.platform == 'win32':
    # ใช้ UTF-8 สำหรับ stdout
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from simple_trip_planner_v2 import plan_trips_v2, export_with_colors

def test_planner_v2():
    """ทดสอบการจัดทริป"""
    
    # อ่านไฟล์
    file_path = r"Dc\แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx"
    sheet_name = "2.Punthai"
    master_file = r"Dc\Master สถานที่ส่ง.xlsx"
    
    print(f"📖 อ่าน: {file_path} sheet: {sheet_name}")
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
        master_df = pd.read_excel(master_file, header=0)
    except Exception as e:
        print(f"❌ อ่านไฟล์ไม่ได้: {e}")
        return
    
    print(f"📋 จำนวนคอลัมน์: {len(df.columns)}")
    
    # ใช้ index แทนชื่อคอลัมน์
    df_work = pd.DataFrame()
    df_work['BU'] = df.iloc[:, 1]
    df_work['Code'] = df.iloc[:, 2]
    df_work['Name'] = df.iloc[:, 4]
    df_work['Cube'] = pd.to_numeric(df.iloc[:, 5], errors='coerce')
    df_work['Weight'] = pd.to_numeric(df.iloc[:, 6], errors='coerce')
    
    df = df_work
    
    print(f"📊 จำนวนสาขา: {len(df)}")
    print(f"📦 Cube รวม: {df['Cube'].sum():.2f}")
    print(f"⚖️  Weight รวม: {df['Weight'].sum():.2f} kg")
    print()
    
    # จัดทริป
    result_df, summary_df = plan_trips_v2(df, master_df)
    
    # แสดงผล
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print()
    
    print("=" * 80)
    print("สาขาแต่ละทริป (แสดง 10 ทริปแรก)")
    print("=" * 80)
    
    for trip in sorted(result_df['Trip'].unique())[:10]:
        if trip == 0:
            continue
        
        trip_data = result_df[result_df['Trip'] == trip].copy()
        
        # เรียงลำดับ: DC011 อยู่ท้ายสุด (Distance_DC = -1), ที่เหลือเรียงจากไกล→ใกล้
        trip_data = trip_data.sort_values(by='Distance_DC', ascending=False)
        
        print(f"\n🚛 ทริป {int(trip)} - รถ {trip_data['Truck'].iloc[0]} "
              f"({len(trip_data)} สาขา, {trip_data['Cube'].sum():.2f} คิว)")
        
        for _, row in trip_data.iterrows():
            dist = row.get('Distance_DC', 0)
            code = str(row.get('Code', ''))
            name = str(row.get('Branch_Name', row.get('Name', '')))
            province = str(row.get('Province', ''))
            district = str(row.get('District', ''))
            subdistrict = str(row.get('Subdistrict', ''))
            cube = row.get('Cube', 0)
            
            # DC011 แสดงพิเศษ
            if code.upper() == 'DC011':
                if dist == -1:
                    print(f"  [{code:10s}] ↩️  กลับ DC วังน้อย")
                else:
                    print(f"  [{code:10s}] 🏭 DC วังน้อย พระนครศรีอยุธยา")
            else:
                # แสดงข้อมูลเต็ม: ระยะทาง → ชื่อ → จังหวัด → อำเภอ → ตำบล
                location = f"{province}"
                if district and district != 'nan':
                    location += f" › {district}"
                if subdistrict and subdistrict != 'nan':
                    location += f" › {subdistrict}"
                
                print(f"  [{code:10s}] {dist:6.1f} กม. | {cube:5.2f} คิว | {name[:35]:35s} | {location}")
    
    # Export
    output_file = file_path  # บันทึกกลับไฟล์เดิม
    export_with_colors(result_df, output_file, file_path, sheet_name)
    
    print(f"\n✅ เสร็จสิ้น - บันทึกกลับ: {output_file}")

if __name__ == "__main__":
    test_planner_v2()
