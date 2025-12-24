"""
ตรวจสอบคุณภาพการจัดทริป: ตำบล → อำเภอ → จังหวัด + ระยะทาง + buffer
"""
import pandas as pd
import sys
import io

# Fix encoding for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 80)
print("🔍 ตรวจสอบคุณภาพการจัดทริป")
print("=" * 80)

# อ่านไฟล์และ import
df = pd.read_excel('Dc/test.xlsx', sheet_name='2.Punthai', header=1)

import importlib.util
spec = importlib.util.spec_from_file_location("app", "app.py")
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)

process_dataframe = app_module.process_dataframe
predict_trips = app_module.predict_trips
model_data = app_module.MASTER_DATA

# Process และจัดทริป
processed_df = process_dataframe(df)
result_df, summary_df = predict_trips(processed_df, model_data, punthai_buffer=1.0, maxmart_buffer=1.10)

print(f"✅ จัดทริปสำเร็จ: {result_df['Trip'].max()} ทริป\n")

# ตรวจสอบข้อมูลตำบล/อำเภอ/จังหวัด
if '_province' not in result_df.columns or '_district' not in result_df.columns:
    print("⚠️ ไม่มีข้อมูลตำบล/อำเภอ/จังหวัด ในผลลัพธ์")
    print(f"คอลัมน์ที่มี: {[c for c in result_df.columns if c.startswith('_')]}")
    sys.exit(1)

# กำหนดขีดจำกัด
LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5},
    'JB': {'max_w': 3500, 'max_c': 7},
    '6W': {'max_w': 6000, 'max_c': 20}
}

print("=" * 80)
print("📊 ตรวจสอบรายทริป")
print("=" * 80)

issues = []
for trip_num in range(1, min(result_df['Trip'].max() + 1, 11)):  # ตรวจ 10 ทริปแรก
    trip_df = result_df[result_df['Trip'] == trip_num]
    if trip_df.empty:
        continue
    
    total_w = trip_df['Weight'].sum()
    total_c = trip_df['Cube'].sum()
    drops = len(trip_df)
    
    # ตรวจสอบ BU และ buffer
    bu_counts = trip_df['BU'].value_counts()
    main_bu = bu_counts.index[0] if len(bu_counts) > 0 else 'PUNTHAI'
    is_punthai = str(main_bu).upper() in ['PUNTHAI', 'GFA', '211']
    buffer = 1.0 if is_punthai else 1.10
    
    # ตรวจสอบจังหวัด/อำเภอ/ตำบล
    provinces = trip_df['_province'].unique()
    districts = trip_df['_district'].unique()
    subdistricts = trip_df['_subdistrict'].unique() if '_subdistrict' in trip_df.columns else ['N/A']
    
    # ตรวจสอบ vehicle priority
    if '_max_vehicle' in trip_df.columns:
        max_vehicles = trip_df['_max_vehicle'].value_counts()
        has_4w = '4W' in max_vehicles.index
        has_jb = 'JB' in max_vehicles.index
        has_6w = '6W' in max_vehicles.index
    else:
        has_4w = has_jb = has_6w = False
    
    # หารถที่เหมาะสม (เริ่มจากใหญ่ไปเล็ก)
    suitable_vehicle = None
    for v in ['6W', 'JB', '4W']:
        lim = LIMITS[v]
        if total_w <= lim['max_w'] * buffer and total_c <= lim['max_c'] * buffer and drops <= lim.get('max_drops', 12):
            suitable_vehicle = v
            break
    
    # สถานะ
    status = "✅"
    problem = []
    
    # เช็คข้อจำกัด
    if suitable_vehicle is None:
        status = "❌"
        problem.append("เกินขีดจำกัดทุกรถ")
    
    # เช็คการกระจายจังหวัด
    if len(provinces) > 2:
        status = "⚠️"
        problem.append(f"กระจาย {len(provinces)} จังหวัด")
    
    # เช็ค vehicle priority mixing
    vehicle_mix = []
    if has_4w:
        vehicle_mix.append("4W")
    if has_jb:
        vehicle_mix.append("JB")
    if has_6w:
        vehicle_mix.append("6W")
    
    if len(vehicle_mix) > 1:
        status = "⚠️"
        problem.append(f"ผสม {'+'.join(vehicle_mix)}")
    
    # รายงาน
    print(f"\n{status} Trip {trip_num}:")
    print(f"  โหลด: {total_w:.0f}kg / {total_c:.2f}m³ / {drops}จุด")
    print(f"  รถที่เหมาะสม: {suitable_vehicle or 'ไม่มี'}")
    print(f"  BU: {main_bu} (buffer {buffer*100:.0f}%)")
    print(f"  จังหวัด: {list(provinces)[:3]}")
    print(f"  อำเภอ: {list(districts)[:3]}")
    print(f"  ตำบล: {list(subdistricts)[:3]}")
    if vehicle_mix:
        print(f"  ข้อจำกัดรถ: {', '.join(vehicle_mix)}")
    if problem:
        print(f"  ⚠️ ปัญหา: {', '.join(problem)}")
        issues.append((trip_num, problem))

print("\n" + "=" * 80)
print("📈 สรุปการตรวจสอบ")
print("=" * 80)

if issues:
    print(f"\n❌ พบปัญหา {len(issues)} ทริป:")
    for trip_num, probs in issues[:5]:
        print(f"  - Trip {trip_num}: {', '.join(probs)}")
else:
    print("\n✅ ไม่พบปัญหา")

# ตรวจสอบการ split
print("\n🔧 ข้อเสนอแนะ:")
print("1. ทริปที่เกินขีดจำกัด → ต้อง split ให้เล็กลง")
print("2. ทริปที่กระจายหลายจังหวัด → ตรวจสอบ province_remaining logic")
print("3. ทริปที่ผสม vehicle priority → ตรวจสอบ vehicle sorting")

print("\n" + "=" * 80)
