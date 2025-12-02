"""
ตรวจสอบโครงสร้างไฟล์ Excel ทั้งหมด
"""
import pandas as pd
import os

files = [
    'Dc/ประวัติงานจัดส่ง DC วังน้อย(1).xlsx',
    'Dc/ปริมาณงานต่อสาขา.xlsx',
    'Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx'
]

for file_path in files:
    if not os.path.exists(file_path):
        print(f"\n❌ ไม่พบไฟล์: {file_path}")
        continue
    
    print(f"\n{'='*80}")
    print(f"File: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    try:
        xls = pd.ExcelFile(file_path)
        print(f"\nSheets: {xls.sheet_names}")
        
        for sheet_name in xls.sheet_names:
            print(f"\n  Sheet: {sheet_name}")
            
            # ลองอ่านหลายแบบ
            for header_row in [0, 1, 2]:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row, nrows=5)
                    
                    if df is not None and len(df) > 0:
                        print(f"\n    Header Row {header_row}:")
                        print(f"    Columns ({len(df.columns)}): {list(df.columns)}")
                        print(f"\n    Sample Data (3 rows):")
                        print(df.head(3).to_string())
                        break
                except:
                    continue
    
    except Exception as e:
        print(f"❌ Error: {e}")

print("\n" + "="*80)
print("✅ เสร็จสิ้น")
print("="*80)
