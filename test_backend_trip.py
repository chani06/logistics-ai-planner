"""
ทดสอบการจัดทริปผ่าน backend โดยตรง (ไม่ผ่าน Streamlit UI)
"""
import pandas as pd
import sys
import json

print("=" * 80)
print("🚀 ทดสอบการจัดทริปด้วยไฟล์ test.xlsx")
print("=" * 80)

# 1. อ่านไฟล์ test.xlsx
print("\n📁 กำลังอ่านไฟล์...")
df = pd.read_excel('Dc/test.xlsx', sheet_name='2.Punthai', header=1)

# เตรียมข้อมูลให้ตรงกับ format ที่ process_dataframe() ต้องการ
# ตามที่กำหนดไว้: ตำแหน่งคอลัมน์
print(f"✅ อ่านข้อมูล: {len(df)} แถว, {len(df.columns)} คอลัมน์")

# แสดงตัวอย่างข้อมูล
print("\n📊 ตัวอย่างข้อมูล 3 แถวแรก:")
sample_cols = ['BU', 'BranchCode', 'Branch', 'TOTALWGT', 'TOTALCUBE']
print(df[sample_cols].head(3).to_string(index=False))

# 2. Import ฟังก์ชันจาก app.py
print("\n⚙️ กำลัง import ฟังก์ชันจาก app.py...")
try:
    # Import ฟังก์ชันที่ต้องการ
    import importlib.util
    spec = importlib.util.spec_from_file_location("app", "app.py")
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    
    # ดึงฟังก์ชันที่ต้องการ
    process_dataframe = app_module.process_dataframe
    predict_trips = app_module.predict_trips
    
    print("✅ Import สำเร็จ")
except Exception as e:
    print(f"❌ Error importing: {e}")
    sys.exit(1)

# 3. Process dataframe
print("\n🔄 กำลัง process dataframe...")
try:
    processed_df = process_dataframe(df)
    if processed_df is None or processed_df.empty:
        print("❌ Process dataframe ล้มเหลว")
        sys.exit(1)
    print(f"✅ Process สำเร็จ: {len(processed_df)} แถว")
    
    # ตรวจสอบคอลัมน์ที่จำเป็น
    required_cols = ['Code', 'Name', 'Weight', 'Cube', 'BU']
    missing = [c for c in required_cols if c not in processed_df.columns]
    if missing:
        print(f"❌ ขาดคอลัมน์: {missing}")
        print(f"คอลัมน์ที่มี: {list(processed_df.columns)[:10]}")
        sys.exit(1)
    
    print(f"\n📋 ข้อมูลหลัง process:")
    print(f"  - สาขา: {processed_df['Code'].nunique()}")
    print(f"  - น้ำหนักรวม: {processed_df['Weight'].sum():.2f} kg")
    print(f"  - คิวรวม: {processed_df['Cube'].sum():.2f} m³")
    
except Exception as e:
    print(f"❌ Error processing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. เรียก predict_trips
print("\n" + "=" * 80)
print("🚛 กำลังจัดทริป...")
print("=" * 80)

try:
    # เตรียม model_data (ต้องมี MASTER_DATA)
    model_data = app_module.MASTER_DATA
    
    # เรียกฟังก์ชัน predict_trips
    result_df, summary_df = predict_trips(
        test_df=processed_df,
        model_data=model_data,
        punthai_buffer=1.0,
        maxmart_buffer=1.10
    )
    
    if result_df is None or result_df.empty:
        print("❌ จัดทริปล้มเหลว")
        sys.exit(1)
    
    print(f"✅ จัดทริปสำเร็จ!")
    
except Exception as e:
    print(f"❌ Error จัดทริป: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. แสดงผลสรุป
print("\n" + "=" * 80)
print("📊 สรุปผลการจัดทริป")
print("=" * 80)

total_trips = result_df['Trip'].max()
print(f"\n🚛 จำนวนทริปทั้งหมด: {total_trips}")

# สรุปรถแต่ละประเภท
if 'Vehicle' in result_df.columns:
    vehicle_counts = result_df['Vehicle'].value_counts()
    print(f"\n🚗 รถแต่ละประเภท:")
    for vehicle, count in vehicle_counts.items():
        print(f"  - {vehicle}: {count} ทริป")

# แสดงทริปแรก 5 ทริป
print(f"\n📋 ตัวอย่าง 5 ทริปแรก:")
for trip_num in range(1, min(6, total_trips + 1)):
    trip_df = result_df[result_df['Trip'] == trip_num]
    total_w = trip_df['Weight'].sum()
    total_c = trip_df['Cube'].sum()
    drops = len(trip_df)
    vehicle = trip_df['Vehicle'].iloc[0] if 'Vehicle' in trip_df.columns else 'N/A'
    codes = ', '.join(trip_df['Code'].head(3).tolist())
    if len(trip_df) > 3:
        codes += f" ... (+{len(trip_df)-3})"
    
    print(f"\nTrip {trip_num}: {vehicle} | {drops} จุด | {total_w:.1f}kg | {total_c:.2f}m³")
    print(f"  สาขา: {codes}")

# ตรวจสอบความถูกต้อง
print("\n" + "=" * 80)
print("✅ การตรวจสอบความถูกต้อง")
print("=" * 80)

# เช็คว่าทุกสาขาได้รับการจัดทริป
unassigned = result_df[result_df['Trip'] == 0]
if len(unassigned) > 0:
    print(f"⚠️ มีสาขาที่ยังไม่ได้จัดทริป: {len(unassigned)} สาขา")
    print(f"  สาขา: {unassigned['Code'].tolist()[:5]}")
else:
    print("✅ ทุกสาขาได้รับการจัดทริปแล้ว")

# เช็คน้ำหนัก/คิวเกิน
print("\n🔍 ตรวจสอบขีดจำกัด:")
LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5},
    'JB': {'max_w': 3500, 'max_c': 7},
    '6W': {'max_w': 6000, 'max_c': 20}
}

over_limit = []
for trip_num in range(1, total_trips + 1):
    trip_df = result_df[result_df['Trip'] == trip_num]
    if trip_df.empty:
        continue
    
    vehicle = trip_df['Vehicle'].iloc[0] if 'Vehicle' in trip_df.columns else '6W'
    total_w = trip_df['Weight'].sum()
    total_c = trip_df['Cube'].sum()
    
    # เช็ค buffer (Punthai 100%, Maxmart 110%)
    bu = trip_df['BU'].iloc[0] if 'BU' in trip_df.columns else 'PUNTHAI'
    buffer = 1.0 if str(bu).upper() in ['PUNTHAI', 'GFA', '211'] else 1.10
    
    limit = LIMITS.get(vehicle, LIMITS['6W'])
    if total_w > limit['max_w'] * buffer or total_c > limit['max_c'] * buffer:
        over_limit.append({
            'trip': trip_num,
            'vehicle': vehicle,
            'weight': total_w,
            'cube': total_c,
            'limit_w': limit['max_w'] * buffer,
            'limit_c': limit['max_c'] * buffer
        })

if over_limit:
    print(f"❌ มีทริปเกินขีดจำกัด: {len(over_limit)} ทริป")
    for item in over_limit[:5]:
        print(f"  Trip {item['trip']}: {item['vehicle']} | {item['weight']:.0f}/{item['limit_w']:.0f}kg | {item['cube']:.2f}/{item['limit_c']:.2f}m³")
else:
    print("✅ ทุกทริปไม่เกินขีดจำกัด")

print("\n" + "=" * 80)
print("✅ การทดสอบเสร็จสิ้น")
print("=" * 80)
