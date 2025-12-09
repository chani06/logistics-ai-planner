"""
ทดสอบ Simple Trip Planner V2
"""

import pandas as pd
import sys
import io

# แก้ encoding สำหรับ Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from simple_trip_planner_v2 import plan_trips_v2, export_with_colors

def test_planner_v2():
    """ทดสอบการจัดทริป"""
    
    # อ่านไฟล์
    file_path = r"Dc\test_temp.xlsx"
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
        
        trip_data = result_df[result_df['Trip'] == trip]
        print(f"\n🚛 Trip {int(trip)} - {trip_data['Truck'].iloc[0]} "
              f"({len(trip_data)} สาขา, {trip_data['Cube'].sum():.2f} คิว)")
        
        for _, row in trip_data.iterrows():
            dist = row['Distance_DC']
            code = str(row['Code'])
            name = str(row['Name'])
            print(f"  [{code:8s}] {name:40s} "
                  f"Cube:{row['Cube']:5.2f} Dist:{dist:6.1f}km")
    
    # Export
    output_file = file_path  # บันทึกกลับไฟล์เดิม
    export_with_colors(result_df, output_file, file_path, sheet_name)
    
    print(f"\n✅ เสร็จสิ้น - บันทึกกลับ: {output_file}")

if __name__ == "__main__":
    test_planner_v2()
