"""
ตรวจสอบความแม่นยำของระบบ AI กับแผน Punthai
"""

import pandas as pd
import sys
import os

# Import functions from app.py
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 80)
print("🎯 ตรวจสอบความแม่นยำของระบบ AI vs แผน Punthai")
print("=" * 80)

# =============================================
# 1. โหลดไฟล์แผน Punthai (ต้นฉบับ)
# =============================================
print("\n📂 Step 1: โหลดแผน Punthai (Ground Truth)")
print("-" * 80)

try:
    punthai_file = 'Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx'
    df_punthai = pd.read_excel(punthai_file, sheet_name='2.Punthai', header=1)
    
    # ทำความสะอาด
    df_punthai = df_punthai[df_punthai['Trip'].notna()].copy()
    df_punthai = df_punthai[~df_punthai['BranchCode'].isin(['DC011', 'PTDC', 'PTG Distribution Center'])].copy()
    
    # Extract vehicle type
    df_punthai['Vehicle_Type'] = df_punthai['Trip no'].apply(
        lambda x: str(x)[:2] if pd.notna(x) else 'Unknown'
    )
    
    print(f"✅ โหลดแผน Punthai สำเร็จ")
    print(f"   📊 จำนวนทริป: {df_punthai['Trip'].nunique()}")
    print(f"   📊 จำนวนสาขา: {df_punthai['BranchCode'].nunique()}")
    print(f"   📊 จำนวนแถว: {len(df_punthai)}")
    
    # สรุปรถแต่ละประเภท
    vehicle_summary = df_punthai.groupby('Vehicle_Type')['Trip'].nunique()
    print(f"\n   🚛 สรุปรถตามแผน Punthai:")
    for vehicle, count in vehicle_summary.items():
        print(f"      - {vehicle}: {count} ทริป")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# =============================================
# 2. สร้างข้อมูล Test (สาขา + น้ำหนัก/คิว)
# =============================================
print("\n📂 Step 2: สร้างข้อมูล Test จากแผน Punthai")
print("-" * 80)

# สร้าง test data โดยใช้ข้อมูลจาก Punthai แต่ไม่รวม Trip และ Vehicle
test_data = []
for _, row in df_punthai.iterrows():
    test_data.append({
        'Code': row['BranchCode'],
        'Name': row.get('BranchName', ''),
        'Province': row.get('Province', ''),
        'Weight': row.get('Weight (kg)', 0),
        'Cube': row.get('CBM', 0)
    })

df_test = pd.DataFrame(test_data)

# รวมสาขาซ้ำ (บางสาขาอาจมีหลายแถว)
df_test = df_test.groupby('Code', as_index=False).agg({
    'Name': 'first',
    'Province': 'first',
    'Weight': 'sum',
    'Cube': 'sum'
})

print(f"✅ สร้างข้อมูล Test สำเร็จ")
print(f"   📊 จำนวนสาขา: {len(df_test)}")
print(f"   📊 น้ำหนักรวม: {df_test['Weight'].sum():,.0f} kg")
print(f"   📊 คิวรวม: {df_test['Cube'].sum():.2f} CBM")

# บันทึกเป็นไฟล์ Excel
test_file = 'punthai_test_data.xlsx'
df_test.to_excel(test_file, index=False)
print(f"   💾 บันทึกไฟล์: {test_file}")

# =============================================
# 3. รันระบบ AI จัดทริป
# =============================================
print("\n🤖 Step 3: รันระบบ AI จัดทริป")
print("-" * 80)
print("⚠️ ต้องรันผ่าน Streamlit Web App")
print("   คำแนะนำ:")
print("   1. เปิด Streamlit: streamlit run app.py")
print("   2. อัปโหลดไฟล์: punthai_test_data.xlsx")
print("   3. กดปุ่ม 'จัดทริปอัตโนมัติ'")
print("   4. ดาวน์โหลดผลลัพธ์และบันทึกเป็น: ai_result.xlsx")

# =============================================
# 4. เปรียบเทียบผลลัพธ์
# =============================================
print("\n📊 Step 4: เปรียบเทียบผลลัพธ์ AI vs Punthai")
print("-" * 80)

ai_result_file = 'ai_result.xlsx'
if not os.path.exists(ai_result_file):
    print(f"⚠️ ไม่พบไฟล์ {ai_result_file}")
    print(f"   กรุณารันระบบ AI และบันทึกผลลัพธ์ก่อน")
    print(f"\n📝 หลังจากได้ไฟล์ {ai_result_file} แล้ว รัน script นี้อีกครั้ง")
    sys.exit(0)

# โหลดผลลัพธ์จาก AI
try:
    df_ai = pd.read_excel(ai_result_file, sheet_name='รายละเอียดทริป')
    print(f"✅ โหลดผลลัพธ์ AI สำเร็จ")
    print(f"   📊 จำนวนทริป: {df_ai['Trip'].nunique()}")
    print(f"   📊 จำนวนสาขา: {df_ai['Code'].nunique()}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# =============================================
# 5. คำนวณความแม่นยำ
# =============================================
print("\n🎯 Step 5: คำนวณความแม่นยำ")
print("=" * 80)

# 5.1 เปรียบเทียบการจับคู่สาขา (Branch Pairing)
print("\n1️⃣ การจับคู่สาขา (Branch Pairing Accuracy)")
print("-" * 80)

# สร้าง pairs จาก Punthai
punthai_pairs = set()
for trip in df_punthai['Trip'].unique():
    trip_branches = df_punthai[df_punthai['Trip'] == trip]['BranchCode'].tolist()
    if len(trip_branches) > 1:
        # สร้างคู่ทุกคู่ในทริป
        for i in range(len(trip_branches)):
            for j in range(i + 1, len(trip_branches)):
                pair = tuple(sorted([trip_branches[i], trip_branches[j]]))
                punthai_pairs.add(pair)

# สร้าง pairs จาก AI
ai_pairs = set()
for trip in df_ai['Trip'].unique():
    trip_branches = df_ai[df_ai['Trip'] == trip]['Code'].tolist()
    if len(trip_branches) > 1:
        for i in range(len(trip_branches)):
            for j in range(i + 1, len(trip_branches)):
                pair = tuple(sorted([trip_branches[i], trip_branches[j]]))
                ai_pairs.add(pair)

# คำนวณ accuracy
correct_pairs = punthai_pairs & ai_pairs
total_pairs_punthai = len(punthai_pairs)
total_pairs_ai = len(ai_pairs)
correct_count = len(correct_pairs)

if total_pairs_punthai > 0:
    precision = (correct_count / total_pairs_ai * 100) if total_pairs_ai > 0 else 0
    recall = (correct_count / total_pairs_punthai * 100)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    print(f"   📊 Punthai Pairs: {total_pairs_punthai:,} คู่")
    print(f"   📊 AI Pairs: {total_pairs_ai:,} คู่")
    print(f"   ✅ ตรงกัน: {correct_count:,} คู่")
    print(f"   🎯 Precision: {precision:.2f}%")
    print(f"   🎯 Recall: {recall:.2f}%")
    print(f"   🎯 F1-Score: {f1:.2f}%")

# 5.2 เปรียบเทียบการเลือกรถ (Vehicle Selection)
print("\n2️⃣ การเลือกรถ (Vehicle Accuracy)")
print("-" * 80)

# สร้าง mapping: สาขา -> รถที่ใช้
punthai_vehicle = {}
for _, row in df_punthai.iterrows():
    code = row['BranchCode']
    vehicle = row['Vehicle_Type']
    if code not in punthai_vehicle:
        punthai_vehicle[code] = []
    punthai_vehicle[code].append(vehicle)

# หารถที่ใช้บ่อยที่สุดสำหรับแต่ละสาขา
punthai_vehicle_most = {}
for code, vehicles in punthai_vehicle.items():
    punthai_vehicle_most[code] = max(set(vehicles), key=vehicles.count)

# AI vehicle
ai_vehicle = {}
for _, row in df_ai.iterrows():
    code = row['Code']
    truck = row['Truck']
    # Extract vehicle type from "6W 📜 ประวัติ"
    vehicle = truck.split()[0] if pd.notna(truck) else 'Unknown'
    ai_vehicle[code] = vehicle

# เปรียบเทียบ
common_branches = set(punthai_vehicle_most.keys()) & set(ai_vehicle.keys())
correct_vehicle = 0
for code in common_branches:
    if punthai_vehicle_most[code] == ai_vehicle[code]:
        correct_vehicle += 1

vehicle_accuracy = (correct_vehicle / len(common_branches) * 100) if len(common_branches) > 0 else 0

print(f"   📊 สาขาที่เปรียบเทียบได้: {len(common_branches):,}")
print(f"   ✅ เลือกรถถูกต้อง: {correct_vehicle:,} สาขา")
print(f"   ❌ เลือกรถผิด: {len(common_branches) - correct_vehicle:,} สาขา")
print(f"   🎯 Accuracy: {vehicle_accuracy:.2f}%")

# แสดงตัวอย่างที่ผิด
if correct_vehicle < len(common_branches):
    print(f"\n   📋 ตัวอย่างที่เลือกรถผิด (5 อันดับแรก):")
    wrong_count = 0
    for code in common_branches:
        if punthai_vehicle_most[code] != ai_vehicle[code] and wrong_count < 5:
            print(f"      - {code}: Punthai={punthai_vehicle_most[code]}, AI={ai_vehicle[code]}")
            wrong_count += 1

# 5.3 เปรียบเทียบจำนวนทริป
print("\n3️⃣ จำนวนทริป (Trip Count)")
print("-" * 80)

punthai_trips = df_punthai['Trip'].nunique()
ai_trips = df_ai['Trip'].nunique()
trip_diff = ai_trips - punthai_trips
trip_diff_pct = (trip_diff / punthai_trips * 100) if punthai_trips > 0 else 0

print(f"   📊 Punthai: {punthai_trips} ทริป")
print(f"   📊 AI: {ai_trips} ทริป")
print(f"   📊 ส่วนต่าง: {trip_diff:+d} ทริป ({trip_diff_pct:+.1f}%)")

if ai_trips < punthai_trips:
    print(f"   ✅ AI ใช้รถน้อยกว่า (ดีกว่า)")
elif ai_trips > punthai_trips:
    print(f"   ⚠️ AI ใช้รถมากกว่า")
else:
    print(f"   ✅ จำนวนทริปเท่ากัน")

# 5.4 เปรียบเทียบ Utilization
print("\n4️⃣ การใช้ประโยชน์รถ (Utilization)")
print("-" * 80)

if 'Utilization' in df_ai.columns:
    ai_avg_util = df_ai.groupby('Trip')['Utilization'].first().mean()
    print(f"   📊 AI Average Utilization: {ai_avg_util:.1f}%")
    
    if ai_avg_util >= 90:
        print(f"   ✅ ใช้รถได้มีประสิทธิภาพ (≥90%)")
    elif ai_avg_util >= 80:
        print(f"   ⚠️ ใช้รถได้ปานกลาง (80-90%)")
    else:
        print(f"   ❌ ใช้รถได้น้อย (<80%)")

# =============================================
# 6. สรุปผลรวม
# =============================================
print("\n" + "=" * 80)
print("📊 สรุปผลการประเมิน")
print("=" * 80)

print(f"\n✅ การจับคู่สาขา (Branch Pairing):")
print(f"   • Precision: {precision:.2f}%")
print(f"   • Recall: {recall:.2f}%")
print(f"   • F1-Score: {f1:.2f}%")

print(f"\n✅ การเลือกรถ (Vehicle Selection):")
print(f"   • Accuracy: {vehicle_accuracy:.2f}%")

print(f"\n✅ จำนวนทริป:")
print(f"   • Punthai: {punthai_trips} ทริป")
print(f"   • AI: {ai_trips} ทริป ({trip_diff_pct:+.1f}%)")

print(f"\n✅ คะแนนรวม:")
overall_score = (f1 + vehicle_accuracy) / 2
print(f"   • Overall Score: {overall_score:.2f}%")

if overall_score >= 80:
    print(f"   🌟 ระบบทำงานได้ดีมาก!")
elif overall_score >= 70:
    print(f"   ✅ ระบบทำงานได้ดี")
elif overall_score >= 60:
    print(f"   ⚠️ ระบบต้องปรับปรุง")
else:
    print(f"   ❌ ระบบต้องพัฒนาเพิ่มเติม")

print("\n" + "=" * 80)
print("✅ การประเมินเสร็จสิ้น!")
print("=" * 80)
