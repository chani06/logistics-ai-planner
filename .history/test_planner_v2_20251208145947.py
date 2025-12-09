"""
ทดสอบ Simple Trip Planner V2
"""

import pandas as pd
import sys
from simple_trip_planner_v2 import plan_trips_v2, export_with_colors

def test_planner_v2():
    """ทดสอบการจัดทริป"""
    
    # อ่านไฟล์
    file_path = r"Dc\test.xlsx"
    sheet_name = "2.Punthai"
    master_file = r"Dc\Master data.xlsx"
    
    print(f"📖 อ่าน: {file_path} sheet: {sheet_name}")
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=1)
        master_df = pd.read_excel(master_file, header=0)
    except Exception as e:
        print(f"❌ อ่านไฟล์ไม่ได้: {e}")
        return
    
    # แปลงชื่อคอลัมน์
    df.columns = ['No', 'BU', 'Code', 'Name_Thai', 'Name', 'Cube', 'Weight', 'Drop', 'Trip', 
                  'ชื่อเต็ม', 'ชื่อย่อ', 'LatLong', 'Province', 'District', 'Subdistrict']
    
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
            print(f"  [{row['Code']:8s}] {row['Name']:40s} "
                  f"Cube:{row['Cube']:5.2f} Dist:{dist:6.1f}km")
    
    # Export
    output_file = file_path  # บันทึกกลับไฟล์เดิม
    export_with_colors(result_df, output_file, file_path, sheet_name)
    
    print(f"\n✅ เสร็จสิ้น - บันทึกกลับ: {output_file}")

if __name__ == "__main__":
    test_planner_v2()
