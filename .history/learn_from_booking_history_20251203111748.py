# -*- coding: utf-8 -*-
"""เรียนรู้ความสัมพันธ์สาขา-รถจากประวัติการจัดส่ง (Booking History)"""
import pandas as pd
import sys

print("="*70)
print("🎓 เรียนรู้ความสัมพันธ์สาขา-รถจากประวัติ Booking")
print("="*70)

# โหลดไฟล์ประวัติ
file_path = 'Dc/ประวัติงานจัดส่ง DC วังน้อย(1).xlsx'
print(f"\nโหลดไฟล์: {file_path}")
df = pd.read_excel(file_path)
print(f"✅ โหลดข้อมูลสำเร็จ: {len(df):,} แถว")

# แปลงประเภทรถ
print("\n" + "="*70)
print("🔄 แปลงประเภทรถ")
print("="*70)
vehicle_mapping = {
    '4 ล้อ จัมโบ้ ตู้ทึบ': 'JB',
    '6 ล้อ ตู้ทึบ': '6W',
    '4 ล้อ ตู้ทึบ': '4W'
}

df['Vehicle_Type'] = df['ประเภทรถ'].map(vehicle_mapping)
print("ประเภทรถในไฟล์:")
for original, mapped in vehicle_mapping.items():
    count = len(df[df['ประเภทรถ'] == original])
    pct = (count / len(df)) * 100
    print(f"  {original:25s} → {mapped:3s} ({count:,} แถว, {pct:.1f}%)")

# วิเคราะห์ความสัมพันธ์สาขา-รถ ตาม Booking
print("\n" + "="*70)
print("📊 วิเคราะห์ความสัมพันธ์สาขา-รถ (จาก Booking)")
print("="*70)

# กลุ่มตาม Booking No + สาขา
branch_vehicle_history = {}
booking_groups = df.groupby('Booking No')

print(f"\nจำนวน Booking: {len(booking_groups):,}")

for booking_no, booking_data in booking_groups:
    # แต่ละ booking มีสาขาอะไรบ้าง ใช้รถอะไร
    vehicle_types = booking_data['Vehicle_Type'].dropna().unique()
    
    if len(vehicle_types) > 0:
        # สมมติว่า 1 booking ใช้รถเดียว (ถ้าหลายคัน เอาที่พบบ่อยสุด)
        vehicle = booking_data['Vehicle_Type'].mode()[0] if len(booking_data['Vehicle_Type'].mode()) > 0 else vehicle_types[0]
        
        # บันทึกประวัติแต่ละสาขาใน booking นี้
        for branch_code in booking_data['รหัสสาขา'].dropna().unique():
            if branch_code not in branch_vehicle_history:
                branch_vehicle_history[branch_code] = []
            branch_vehicle_history[branch_code].append(vehicle)

print(f"สาขาที่มีประวัติ: {len(branch_vehicle_history):,}")

# วิเคราะห์ประวัติแต่ละสาขา
print("\n" + "="*70)
print("🔍 วิเคราะห์ข้อจำกัดรถของแต่ละสาขา")
print("="*70)

branch_restrictions = {}
vehicle_sizes = {'4W': 1, 'JB': 2, '6W': 3}

strict_4w = []
strict_jb = []
strict_6w = []
flexible = []

for branch_code, vehicle_list in branch_vehicle_history.items():
    vehicles_used = set(vehicle_list)
    vehicle_counts = pd.Series(vehicle_list).value_counts().to_dict()
    
    if len(vehicles_used) == 1:
        # ใช้รถเดียว = ข้อจำกัดเข้มงวด
        vehicle = list(vehicles_used)[0]
        branch_restrictions[str(branch_code)] = {
            'max_vehicle': vehicle,
            'allowed': [vehicle],
            'history': vehicle_counts,
            'total_bookings': len(vehicle_list),
            'restriction_type': 'STRICT'
        }
        
        if vehicle == '4W':
            strict_4w.append(branch_code)
        elif vehicle == 'JB':
            strict_jb.append(branch_code)
        elif vehicle == '6W':
            strict_6w.append(branch_code)
            
    else:
        # ใช้หลายประเภท = ยืดหยุ่น (แต่มีข้อจำกัด = ใช้ได้ถึงรถที่ใหญ่ที่สุดที่เคยใช้)
        max_vehicle = max(vehicles_used, key=lambda v: vehicle_sizes.get(v, 0))
        branch_restrictions[str(branch_code)] = {
            'max_vehicle': max_vehicle,
            'allowed': list(vehicles_used),
            'history': vehicle_counts,
            'total_bookings': len(vehicle_list),
            'restriction_type': 'FLEXIBLE'
        }
        flexible.append(branch_code)

print(f"สาขาที่มีข้อจำกัดเข้มงวด: {len(strict_4w) + len(strict_jb) + len(strict_6w):,} สาขา")
print(f"  - 4W เท่านั้น: {len(strict_4w):,} สาขา ({len(strict_4w)/len(branch_restrictions)*100:.1f}%)")
print(f"  - JB เท่านั้น: {len(strict_jb):,} สาขา ({len(strict_jb)/len(branch_restrictions)*100:.1f}%)")
print(f"  - 6W เท่านั้น: {len(strict_6w):,} สาขา ({len(strict_6w)/len(branch_restrictions)*100:.1f}%)")
print(f"สาขาที่ยืดหยุ่น: {len(flexible):,} สาขา ({len(flexible)/len(branch_restrictions)*100:.1f}%)")

# แสดงตัวอย่าง
print("\n" + "="*70)
print("📝 ตัวอย่างข้อจำกัดสาขา")
print("="*70)

print("\n4W เท่านั้น (10 สาขาแรก):")
for branch in strict_4w[:10]:
    info = branch_restrictions[str(branch)]
    print(f"  {branch}: ใช้ 4W {info['total_bookings']} ครั้ง")

print("\nJB เท่านั้น (10 สาขาแรก):")
for branch in strict_jb[:10]:
    info = branch_restrictions[str(branch)]
    print(f"  {branch}: ใช้ JB {info['total_bookings']} ครั้ง")

print("\n6W เท่านั้น (10 สาขาแรก):")
for branch in strict_6w[:10]:
    info = branch_restrictions[str(branch)]
    print(f"  {branch}: ใช้ 6W {info['total_bookings']} ครั้ง")

print("\nยืดหยุ่น (10 สาขาแรก):")
for branch in flexible[:10]:
    info = branch_restrictions[str(branch)]
    print(f"  {branch}: ใช้ {list(info['history'].keys())} (max: {info['max_vehicle']})")
    print(f"    → {info['history']}")

# หลักการสำคัญ
print("\n" + "="*70)
print("🎯 หลักการที่เรียนรู้")
print("="*70)
print("""
✅ หลักการจากประวัติ Booking:

1. **สาขาที่ใช้รถเดียว (STRICT)** = มีข้อจำกัดเข้มงวด
   - ถ้าประวัติไม่เคยใช้รถใหญ่ → ห้ามใช้รถใหญ่
   - เช่น: ใช้แค่ 4W → รถใหญ่เข้าไม่ได้

2. **สาขาที่ใช้หลายประเภท (FLEXIBLE)** = มีความยืดหยุ่น
   - ใช้ได้ถึงรถที่ใหญ่ที่สุดที่เคยใช้
   - เช่น: เคยใช้ 4W+JB → ใช้ได้ถึง JB (ไม่ได้ 6W)

3. **ลบกฎระยะทาง 100 กม.**
   - ไม่มีกฎตายตัว
   - ใช้ประวัติจริงเป็นหลัก
   - สาขาใกล้อาจต้องใช้รถใหญ่ (น้ำหนักมาก)
   - สาขาไกลอาจใช้รถเล็กได้ (รถใหญ่เข้าไม่ได้)

4. **ระดับความเชื่อมั่น**
   - Booking มาก = เชื่อมั่นสูง
   - Booking น้อย = ระวัง อาจมีข้อยกเว้น
""")

# บันทึกผลลัพธ์
print("\n" + "="*70)
print("💾 บันทึกผลลัพธ์")
print("="*70)

output_data = []
for branch_code, info in branch_restrictions.items():
    output_data.append({
        'Branch_Code': branch_code,
        'Max_Vehicle': info['max_vehicle'],
        'Allowed_Vehicles': ', '.join(info['allowed']),
        'Restriction_Type': info['restriction_type'],
        'Total_Bookings': info['total_bookings'],
        'History': str(info['history'])
    })

output_df = pd.DataFrame(output_data)
output_df = output_df.sort_values('Total_Bookings', ascending=False)
output_file = 'branch_vehicle_restrictions_from_booking.xlsx'
output_df.to_excel(output_file, index=False)
print(f"✅ บันทึกไฟล์: {output_file}")
print(f"   สาขาทั้งหมด: {len(output_df):,}")

print("\n🎉 วิเคราะห์เสร็จสิ้น!")
