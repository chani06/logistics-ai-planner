import pandas as pd
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

df = pd.read_excel('Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx', 
                   sheet_name='2.Punthai')

print('=' * 80)
print('ตรวจสอบคอลัมน์ Trip')
print('=' * 80)

print(f'\n📊 จำนวนแถวทั้งหมด: {len(df)}')
print(f'✅ จำนวนที่มี Trip: {df["Trip"].notna().sum()}')
print(f'❌ จำนวนที่ไม่มี Trip: {df["Trip"].isna().sum()}')

# ดึง Trip ที่ไม่ซ้ำ
trips = df[df["Trip"].notna()]["Trip"].unique()
trip_nums = sorted([int(t) for t in trips if pd.notna(t)])

print(f'\n🚛 จำนวน Trip ทั้งหมด: {len(trip_nums)}')
print(f'📝 Trip เลขที่: {trip_nums[:30]}')
if len(trip_nums) > 30:
    print(f'   ... และอีก {len(trip_nums) - 30} ทริป')

# ตรวจสอบความต่อเนื่อง
print(f'\n🔍 ตรวจสอบความต่อเนื่อง:')
print(f'   Trip เริ่มต้น: {min(trip_nums)}')
print(f'   Trip สุดท้าย: {max(trip_nums)}')

missing = [i for i in range(min(trip_nums), max(trip_nums)+1) if i not in trip_nums]
if missing:
    print(f'⚠️  Trip ที่ขาดหาย: {missing[:30]}')
    if len(missing) > 30:
        print(f'   ... และอีก {len(missing) - 30} ทริป')
else:
    print(f'✅ ไม่มี Trip ขาดหาย - ต่อเนื่องทุกหมายเลข')

# แสดงตัวอย่างข้อมูล
print(f'\n📋 ตัวอย่างข้อมูล Trip (10 แถวแรก):')
sample = df[df["Trip"].notna()][['รหัส', 'ชื่อสาขา', 'จังหวัด', 'Trip', 'Truck_Type']].head(10)
print(sample.to_string(index=False))

# นับจำนวนสาขาในแต่ละ Trip
print(f'\n🔢 จำนวนสาขาในแต่ละ Trip (10 ทริปแรก):')
trip_counts = df[df["Trip"].notna()].groupby('Trip').size().sort_index()
for trip, count in list(trip_counts.items())[:10]:
    print(f'   Trip {int(trip):3d}: {count:2d} สาขา')
