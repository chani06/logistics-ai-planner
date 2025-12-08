"""
ตรวจสอบโครงสร้างไฟล์ punthai_test_data.xlsx
"""
import pandas as pd

# โหลดไฟล์
df = pd.read_excel('punthai_test_data.xlsx')

print("="*80)
print("📄 โครงสร้างไฟล์ punthai_test_data.xlsx")
print("="*80)

print(f"\n📊 จำนวนแถว: {len(df)}")
print(f"📊 จำนวนคอลัมน์: {len(df.columns)}")

print(f"\n📋 รายชื่อคอลัมน์:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\n📋 ตัวอย่างข้อมูล 10 แถวแรก:")
print(df.head(10).to_string())

print(f"\n📋 สาขาที่ไม่ซ้ำกัน: {df['Code'].nunique()} สาขา")
print(f"📋 ชื่อสาขาที่ไม่ซ้ำกัน: {df['Name'].nunique() if 'Name' in df.columns else 'N/A'}")

# เช็คคอลัมน์จังหวัด
if 'Province' in df.columns:
    print(f"\n📍 จังหวัดที่พบ:")
    print(df['Province'].value_counts().head(10))
else:
    print(f"\n⚠️ ไม่มีคอลัมน์ Province")

# เช็คคอลัมน์ Name
if 'Name' in df.columns:
    print(f"\n📝 ชื่อสาขาที่พบ (10 อันดับแรก):")
    print(df['Name'].value_counts().head(10))
