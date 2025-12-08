"""
ทดสอบการจัดทริปด้วยไฟล์จากหลังบ้าน (Dc/test.xlsx)
ตรวจสอบ:
1. ทุกสาขามีทริปครบหรือไม่
2. จังหวัดถูกต้องหรือไม่ (เช็คจากชื่อสาขา)
3. ตำบล/อำเภอดึงได้หรือไม่
4. สาขาชื่อเดียวกันอยู่ทริปเดียวกันหรือใกล้กัน
"""

import pandas as pd
import sys
import os

# เพิ่ม path เพื่อ import app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("🧪 ทดสอบการจัดทริปด้วยไฟล์หลังบ้าน")
print("=" * 80)

# โหลดไฟล์ทดสอบ
test_file = "Dc/test.xlsx"
print(f"\n📂 โหลดไฟล์: {test_file}")

try:
    # ลอง header=0 ก่อน ถ้าไม่ได้ลอง header=1
    df = pd.read_excel(test_file, header=1)  # เปลี่ยนจาก 0 เป็น 1
    
    # ตัด whitespace ในชื่อคอลัมน์
    df.columns = df.columns.str.strip()
    
    print(f"✅ โหลดสำเร็จ: {len(df)} รายการ")
    print(f"📋 คอลัมน์: {list(df.columns)}")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาด: {e}")
    sys.exit(1)

# แสดงตัวอย่างข้อมูล
print("\n📊 ตัวอย่างข้อมูล 5 แถวแรก:")
print(df.head())

print("\n📈 สรุปข้อมูล:")
print(f"- จำนวนสาขา: {df['Code'].nunique() if 'Code' in df.columns else 'N/A'}")
print(f"- น้ำหนักรวม: {df['Weight'].sum():.2f} kg" if 'Weight' in df.columns else "- น้ำหนัก: N/A")
print(f"- คิวรวม: {df['Cube'].sum():.2f} m³" if 'Cube' in df.columns else "- คิว: N/A")

# เช็คคอลัมน์ที่จำเป็น
required_cols = ['Code', 'Name', 'Weight', 'Cube']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    print(f"\n⚠️ คอลัมน์ที่ขาดหายไป: {missing_cols}")
    print("ลองแก้ไขชื่อคอลัมน์...")
    
    # แก้ไขชื่อคอลัมน์ที่อาจจะต่างกัน
    col_mapping = {
        'รหัสสาขา': 'Code',
        'รหัส': 'Code',
        'ชื่อสาขา': 'Name',
        'ชื่อ': 'Name',
        'น้ำหนัก': 'Weight',
        'คิว': 'Cube',
        'Cubic': 'Cube',
        'จังหวัด': 'Province'
    }
    
    df.rename(columns=col_mapping, inplace=True)
    print(f"✅ แก้ไขแล้ว: {list(df.columns)}")

# เช็คว่ามี Province หรือไม่
if 'Province' not in df.columns:
    print("\n⚠️ ไม่มีคอลัมน์ Province - จะดึงจากชื่อสาขาและ Master Data")
    df['Province'] = 'UNKNOWN'

print("\n" + "=" * 80)
print("🚀 เริ่มจัดทริป...")
print("=" * 80)

# Import ฟังก์ชันจาก app.py
try:
    from app import predict_trips, load_master_data, load_booking_history
    
    # โหลด Master Data
    print("\n📖 โหลด Master Data...")
    load_master_data()
    
    # โหลด Booking History
    print("📚 โหลด Booking History...")
    model_data = load_booking_history()
    
    if not model_data:
        print("⚠️ ไม่มี Booking History - สร้าง model_data เปล่า")
        model_data = {
            'model': None,
            'trip_pairs': set(),
            'branch_info': {},
            'trip_vehicles': {},
            'branch_vehicles': {}
        }
    
    # จัดทริป
    print("\n🔄 กำลังจัดทริป...")
    result_df, summary_df = predict_trips(df.copy(), model_data)
    
    print("\n" + "=" * 80)
    print("✅ จัดทริปเสร็จสิ้น!")
    print("=" * 80)
    
    # วิเคราะห์ผลลัพธ์
    print("\n📊 สรุปผลการจัดทริป:")
    print(f"- จำนวนทริป: {len(summary_df)}")
    print(f"- จำนวนสาขาทั้งหมด: {len(result_df)}")
    print(f"- เฉลี่ยสาขา/ทริป: {len(result_df)/len(summary_df):.1f}")
    
    # เช็คสาขาที่ไม่มีทริป
    unassigned = result_df[result_df['Trip'].isna()]
    if len(unassigned) > 0:
        print(f"\n❌ สาขาที่ไม่มีทริป: {len(unassigned)} สาขา")
        print(unassigned[['Code', 'Name', 'Weight', 'Cube']])
    else:
        print("\n✅ ทุกสาขามีทริปครบทั้งหมด!")
    
    # เช็คจังหวัด
    print("\n🗺️ ตรวจสอบจังหวัด:")
    if 'Province' in result_df.columns:
        unknown_provinces = result_df[result_df['Province'] == 'UNKNOWN']
        if len(unknown_provinces) > 0:
            print(f"⚠️ สาขาที่ไม่มีจังหวัด: {len(unknown_provinces)} สาขา")
            print(unknown_provinces[['Code', 'Name']])
        else:
            print("✅ ทุกสาขามีจังหวัด")
            
            # แสดงจังหวัดแต่ละจังหวัดมีกี่สาขา
            province_counts = result_df['Province'].value_counts()
            print("\n📍 จำนวนสาขาแต่ละจังหวัด:")
            for prov, count in province_counts.head(10).items():
                print(f"  - {prov}: {count} สาขา")
    
    # เช็คตำบล/อำเภอ
    if 'Subdistrict' in result_df.columns and 'District' in result_df.columns:
        has_subdistrict = result_df[result_df['Subdistrict'].notna() & (result_df['Subdistrict'] != '')]
        has_district = result_df[result_df['District'].notna() & (result_df['District'] != '')]
        
        print(f"\n🏘️ ข้อมูลตำบล/อำเภอ:")
        print(f"  - มีตำบล: {len(has_subdistrict)}/{len(result_df)} สาขา ({len(has_subdistrict)/len(result_df)*100:.1f}%)")
        print(f"  - มีอำเภอ: {len(has_district)}/{len(result_df)} สาขา ({len(has_district)/len(result_df)*100:.1f}%)")
    
    # เช็คสาขาชื่อเดียวกัน
    print("\n👥 ตรวจสอบสาขาชื่อเดียวกัน:")
    if 'Name' in result_df.columns:
        # หาชื่อที่มีหลายสาขา (เช่น พิษณุโลก1, พิษณุโลก2)
        def get_base_name(name):
            import re
            if not name:
                return ""
            base = re.sub(r'\s*\d+\s*$', '', str(name).strip())
            base = re.sub(r'^สาขา\s*', '', base)
            return base.strip()
        
        result_df['BaseName'] = result_df['Name'].apply(get_base_name)
        
        # หากลุ่มที่มีชื่อเดียวกัน
        name_groups = result_df.groupby('BaseName').size()
        multi_branch_names = name_groups[name_groups > 1]
        
        if len(multi_branch_names) > 0:
            print(f"  พบ {len(multi_branch_names)} ชื่อที่มีหลายสาขา:")
            
            for base_name in multi_branch_names.head(5).index:
                branches = result_df[result_df['BaseName'] == base_name]
                trips = branches['Trip'].unique()
                print(f"\n  📌 {base_name} ({len(branches)} สาขา):")
                print(f"     - ทริป: {sorted(trips)}")
                print(f"     - จำนวนทริป: {len(trips)}")
                
                if len(trips) == 1:
                    print(f"     ✅ อยู่ทริปเดียวกัน")
                elif len(trips) <= 3:
                    print(f"     ⚠️ แยกเป็น {len(trips)} ทริป (ยอมรับได้)")
                else:
                    print(f"     ❌ แยกเป็น {len(trips)} ทริป (มากเกินไป)")
    
    # แสดงสรุปแต่ละทริป (5 ทริปแรก)
    print("\n" + "=" * 80)
    print("📋 รายละเอียด 5 ทริปแรก:")
    print("=" * 80)
    
    for _, trip in summary_df.head(5).iterrows():
        trip_num = trip['Trip']
        trip_branches = result_df[result_df['Trip'] == trip_num]
        
        print(f"\n🚛 Trip {int(trip_num)}: {trip['Truck']}")
        print(f"   - สาขา: {trip['Branches']} สาขา")
        print(f"   - น้ำหนัก: {trip['Weight']:.2f} kg ({trip['Weight_Use%']:.1f}%)")
        print(f"   - คิว: {trip['Cube']:.2f} m³ ({trip['Cube_Use%']:.1f}%)")
        
        if 'Province' in trip_branches.columns:
            provinces = trip_branches['Province'].unique()
            print(f"   - จังหวัด: {', '.join([str(p) for p in provinces])}")
        
        print(f"   - สาขา: {', '.join(trip_branches['Code'].tolist())}")
    
    # บันทึกผลลัพธ์
    output_file = "test_result_backoffice.xlsx"
    print(f"\n💾 บันทึกผลลัพธ์: {output_file}")
    
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        result_df.to_excel(writer, sheet_name='รายละเอียด', index=False)
        summary_df.to_excel(writer, sheet_name='สรุป', index=False)
    
    print(f"✅ บันทึกสำเร็จ!")
    
    print("\n" + "=" * 80)
    print("🎉 ทดสอบเสร็จสิ้น!")
    print("=" * 80)

except ImportError as e:
    print(f"❌ ไม่สามารถ import จาก app.py: {e}")
    print("กรุณาตรวจสอบว่า app.py อยู่ในโฟลเดอร์เดียวกัน")
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาด: {e}")
    import traceback
    traceback.print_exc()
