"""
ตรวจหาสาขาที่ขาดข้อมูล จังหวัด/อำเภอ/ตำบล
"""
import pandas as pd
import sys
import io
import gspread
from oauth2client.service_account import ServiceAccountCredentials

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# โหลด MASTER_DATA จาก Google Sheets โดยตรง
print("🔗 เชื่อมต่อ Google Sheets...")
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
gc = gspread.authorize(creds)
SPREADSHEET_ID = '12DmIfECwVpsWfl8rl2r1A_LB4_5XMrmnmwlPUHKNU-o'
sh = gc.open_by_key(SPREADSHEET_ID)

# ดึงข้อมูลจาก worksheet GID: 876257177
worksheet = None
for ws in sh.worksheets():
    if ws.id == 876257177:
        worksheet = ws
        break

if worksheet is None:
    worksheet = sh.get_worksheet(0)

data = worksheet.get_all_values()
headers = data[0]
MASTER_DATA = pd.DataFrame(data[1:], columns=headers)
print(f"✅ โหลดข้อมูลจาก Google Sheets สำเร็จ ({len(MASTER_DATA)} แถว)")
print(f"📋 คอลัมน์ที่มี: {list(MASTER_DATA.columns)}\n")

print("=" * 80)
print("🔍 ตรวจหาสาขาที่ขาดข้อมูล")
print("=" * 80)

if isinstance(MASTER_DATA, pd.DataFrame):
    # เช็คว่าสาขาไหนไม่มีจังหวัด/อำเภอ/ตำบล
    missing_province = MASTER_DATA[MASTER_DATA['จังหวัด'].isna() | (MASTER_DATA['จังหวัด'] == '')]
    missing_district = MASTER_DATA[MASTER_DATA['อำเภอ'].isna() | (MASTER_DATA['อำเภอ'] == '')]
    missing_subdistrict = MASTER_DATA[MASTER_DATA['ตำบล'].isna() | (MASTER_DATA['ตำบล'] == '')]
    
    print(f"\n📊 สาขาที่ไม่มีจังหวัด: {len(missing_province)} สาขา")
    if len(missing_province) > 0:
        cols = ['Plan Code'] if 'Plan Code' in MASTER_DATA.columns else []
        if 'Branch Name' in MASTER_DATA.columns:
            cols.append('Branch Name')
        elif 'สถานที่ส่ง' in MASTER_DATA.columns:
            cols.append('สถานที่ส่ง')
        print(missing_province[cols].head(20) if cols else missing_province.head(20))
    
    print(f"\n📊 สาขาที่ไม่มีอำเภอ: {len(missing_district)} สาขา")
    if len(missing_district) > 0:
        cols = ['Plan Code'] if 'Plan Code' in MASTER_DATA.columns else []
        if 'Branch Name' in MASTER_DATA.columns:
            cols.append('Branch Name')
        elif 'สถานที่ส่ง' in MASTER_DATA.columns:
            cols.append('สถานที่ส่ง')
        print(missing_district[cols].head(20) if cols else missing_district.head(20))
    
    print(f"\n📊 สาขาที่ไม่มีตำบล: {len(missing_subdistrict)} สาขา")
    if len(missing_subdistrict) > 0:
        cols = []
        if 'Branch Name' in MASTER_DATA.columns:
            cols.append('Branch Name')
        elif 'สถานที่ส่ง' in MASTER_DATA.columns:
            cols.append('สถานที่ส่ง')
        if 'Plan Code' in MASTER_DATA.columns:
            cols.append('Plan Code')
        if 'จังหวัด' in MASTER_DATA.columns:
            cols.append('จังหวัด')
        if 'อำเภอ' in MASTER_DATA.columns:
            cols.append('อำเภอ')
        print(missing_subdistrict[cols] if cols else missing_subdistrict)
    
    # เช็คใน test.xlsx
    print("\n" + "=" * 80)
    print("🔍 ตรวจสาขาใน test.xlsx ที่อาจมีปัญหา")
    print("=" * 80)
    
    df = pd.read_excel('Dc/test.xlsx', sheet_name='2.Punthai', header=1)
    test_codes = df['Code'].unique()
    
    problem_codes = []
    for code in test_codes:
        if code in MASTER_DATA['Plan Code'].values:
            row = MASTER_DATA[MASTER_DATA['Plan Code'] == code].iloc[0]
            province = str(row.get('จังหวัด', '')).strip()
            district = str(row.get('อำเภอ', '')).strip()
            subdistrict = str(row.get('ตำบล', '')).strip()
            name_col = 'Branch Name' if 'Branch Name' in MASTER_DATA.columns else 'สถานที่ส่ง'
            
            if not province or not district or not subdistrict:
                problem_codes.append({
                    'Code': code,
                    'Name': row.get(name_col, ''),
                    'Province': province or '❌ ไม่มี',
                    'District': district or '❌ ไม่มี',
                    'Subdistrict': subdistrict or '❌ ไม่มี'
                })
    
    if problem_codes:
        print(f"\n⚠️ พบ {len(problem_codes)} สาขาที่ข้อมูลไม่ครบ:")
        problem_df = pd.DataFrame(problem_codes)
        print(problem_df.to_string(index=False))
    else:
        print("\n✅ ทุกสาขาในไฟล์ test.xlsx มีข้อมูลครบ!")
else:
    print("❌ MASTER_DATA ไม่ใช่ DataFrame")
