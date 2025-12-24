"""
วิเคราะห์ความต่างระหว่าง Google Sheets และ JSON
"""
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from collections import Counter

print("=" * 70)
print("วิเคราะห์ข้อมูล Google Sheets vs JSON")
print("=" * 70)

# เชื่อมต่อ Google Sheets
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
client = gspread.authorize(creds)

spreadsheet_id = '12DmIfECwVpsWfl8rl2r1A_LB4_5XMrmnmwlPUHKNU-o'
sh = client.open_by_key(spreadsheet_id)
worksheet = sh.get_worksheet_by_id(876257177)

print(f"\nชื่อ Sheet: {sh.title}")
print(f"ชื่อ Worksheet: {worksheet.title}")

# ดึงข้อมูล
data = worksheet.get_all_records()
df = pd.DataFrame(data)

print(f"\nจำนวนแถวทั้งหมดใน Sheets: {len(df):,}")

# หาคอลัมน์ Plan Code
code_col = None
for col in df.columns:
    if 'Code' in col or 'code' in col or 'รหัส' in col:
        code_col = col
        break

if not code_col:
    print("ไม่พบคอลัมน์รหัสสาขา")
    exit()

print(f"ใช้คอลัมน์: {code_col}")

# วิเคราะห์ข้อมูล
all_codes = [str(row).strip().upper() for row in df[code_col] if row and str(row).strip()]
code_counter = Counter(all_codes)

print(f"\n{'=' * 70}")
print("สรุปข้อมูล")
print(f"{'=' * 70}")
print(f"แถวทั้งหมด:           {len(df):,}")
print(f"Plan Code ไม่ซ้ำ:     {len(code_counter):,}")
print(f"Plan Code ว่าง/ไม่มี: {len(df) - len(all_codes):,}")
print(f"Plan Code ซ้ำ:        {len(all_codes) - len(code_counter):,}")

# แสดงข้อมูลซ้ำ
duplicates = {code: count for code, count in code_counter.items() if count > 1}
if duplicates:
    print(f"\n{'=' * 70}")
    print(f"พบ Plan Code ซ้ำ: {len(duplicates):,} รายการ")
    print(f"{'=' * 70}")
    
    # เรียงตามจำนวนที่ซ้ำมากที่สุด
    sorted_dups = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 Plan Code ที่ซ้ำมากที่สุด:")
    print(f"{'Plan Code':<15} {'จำนวน':>8}  {'สาขา'}")
    print("-" * 70)
    
    for code, count in sorted_dups[:20]:
        # หาชื่อสาขา
        branch_rows = df[df[code_col].astype(str).str.strip().str.upper() == code]
        if not branch_rows.empty:
            branch_name = branch_rows.iloc[0].get('สาขา', 'N/A')
            print(f"{code:<15} {count:>8}  {branch_name}")
    
    # สรุปสถิติการซ้ำ
    print(f"\n{'=' * 70}")
    print("สถิติการซ้ำ")
    print(f"{'=' * 70}")
    
    dup_counts = Counter(duplicates.values())
    for num_dups in sorted(dup_counts.keys(), reverse=True)[:10]:
        count = dup_counts[num_dups]
        print(f"ซ้ำ {num_dups} ครั้ง: {count:,} รายการ")

# เหตุผล
print(f"\n{'=' * 70}")
print("สรุป")
print(f"{'=' * 70}")
print(f"""
Google Sheets มี:    {len(df):,} แถว
JSON เก็บได้:        {len(code_counter):,} สาขา (ไม่ซ้ำ)
ความแตกต่าง:        {len(df) - len(code_counter):,} แถว

เหตุผล:
1. ข้อมูลซ้ำ ({len(all_codes) - len(code_counter):,} แถว) - ระบบเก็บแค่ค่าล่าสุดของแต่ละ Plan Code
2. แถวว่าง ({len(df) - len(all_codes):,} แถว) - ไม่มี Plan Code หรือเป็นค่าว่าง

ระบบใช้ Plan Code เป็น Primary Key ดังนั้นแต่ละ Plan Code จะมีได้แค่ 1 รายการ
ถ้ามีข้อมูลซ้ำ ระบบจะเก็บข้อมูลล่าสุดที่พบใน Sheet
""")
