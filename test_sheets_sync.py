#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ทดสอบการดึงข้อมูลจาก Google Sheets
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import os

print('=' * 60)
print('📊 ทดสอบดึงข้อมูลจาก Google Sheets')
print('=' * 60)

# 1. ตรวจสอบไฟล์ credentials
if not os.path.exists('credentials.json'):
    print('❌ ไม่พบไฟล์ credentials.json')
    exit(1)
else:
    print('✅ พบไฟล์ credentials.json')

# 2. เชื่อมต่อ Google Sheets
scope = ['https://spreadsheets.google.com/feeds', 
         'https://www.googleapis.com/auth/drive']

try:
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    gc = gspread.authorize(creds)
    print('✅ เชื่อมต่อ Google Sheets API สำเร็จ\n')
except Exception as e:
    print(f'❌ ไม่สามารถเชื่อมต่อได้: {e}')
    exit(1)

# 3. เปิด Spreadsheet
SPREADSHEET_ID = '12DmIfECwVpsWfl8rl2r1A_LB4_5XMrmnmwlPUHKNU-o'
WORKSHEET_GID = 876257177

try:
    sh = gc.open_by_key(SPREADSHEET_ID)
    print(f'📄 Spreadsheet Title: {sh.title}')
    print(f'📑 URL: {sh.url}')
    print(f'📊 จำนวน Worksheets: {len(sh.worksheets())}\n')
except Exception as e:
    print(f'❌ ไม่สามารถเปิด Spreadsheet ได้: {e}')
    exit(1)

# 4. แสดงรายการ Worksheets
print('📋 รายการ Worksheets:')
for i, ws in enumerate(sh.worksheets(), 1):
    print(f'  {i}. {ws.title:<30} (ID: {ws.id}, Rows: {ws.row_count}, Cols: {ws.col_count})')

# 5. หา Worksheet ที่ต้องการ
worksheet = None
for ws in sh.worksheets():
    if ws.id == WORKSHEET_GID:
        worksheet = ws
        break

if worksheet is None:
    print(f'\n❌ ไม่พบ Worksheet GID {WORKSHEET_GID}')
    print('💡 ใช้ Worksheet แรกแทน...')
    worksheet = sh.get_worksheet(0)

print(f'\n✅ ใช้ Worksheet: "{worksheet.title}"')
print(f'   - ID: {worksheet.id}')
print(f'   - ขนาด: {worksheet.row_count} rows × {worksheet.col_count} columns')

# 6. ดึงข้อมูล
try:
    print('\n⏳ กำลังดึงข้อมูล...')
    data = worksheet.get_all_values()
    print(f'✅ ดึงข้อมูลสำเร็จ: {len(data)} แถว')
except Exception as e:
    print(f'❌ ไม่สามารถดึงข้อมูลได้: {e}')
    exit(1)

# 7. แสดง Header
if len(data) >= 1:
    print(f'\n🔤 Header (บรรทัดที่ 1):')
    headers = data[0]
    for i, h in enumerate(headers[:15], 1):  # แสดง 15 คอลัมน์แรก
        print(f'   {i:2d}. {h}')
    if len(headers) > 15:
        print(f'   ... และอีก {len(headers)-15} คอลัมน์')

# 8. แสดงตัวอย่างข้อมูล
if len(data) >= 2:
    print(f'\n📋 ตัวอย่างข้อมูล (3 แถวแรก):')
    for i, row in enumerate(data[1:4], 1):
        print(f'\n   แถว {i}:')
        for j, (header, value) in enumerate(zip(headers[:10], row[:10])):
            print(f'      {header}: {value}')

# 9. สร้าง DataFrame
try:
    df = pd.DataFrame(data[1:], columns=data[0])
    print(f'\n✅ สร้าง DataFrame สำเร็จ')
    print(f'   - Shape: {df.shape[0]} rows × {df.shape[1]} columns')
    print(f'   - Columns: {len(df.columns)} คอลัมน์')
    
    # ตรวจสอบคอลัมน์สำคัญ
    important_cols = ['Code', 'Plan Code', 'รหัสสาขา', 'สาขา', 'จังหวัด', 'อำเภอ', 'ตำบล']
    print(f'\n🔍 ตรวจสอบคอลัมน์สำคัญ:')
    for col in important_cols:
        if col in df.columns:
            non_empty = df[col].notna().sum()
            print(f'   ✅ {col:<15} - มีข้อมูล {non_empty} แถว')
        else:
            print(f'   ❌ {col:<15} - ไม่พบคอลัมน์นี้')
    
    # หารหัสสาขา
    code_col = None
    for col in ['Code', 'Plan Code', 'รหัสสาขา', 'สาขา']:
        if col in df.columns:
            code_col = col
            break
    
    if code_col:
        print(f'\n✅ ใช้คอลัมน์ "{code_col}" เป็นรหัสสาขา')
        unique_codes = df[code_col].nunique()
        print(f'   จำนวนสาขาไม่ซ้ำ: {unique_codes} สาขา')
        
        # แสดงตัวอย่างรหัสสาขา
        print(f'\n📍 ตัวอย่างรหัสสาขา 10 ตัวแรก:')
        sample_codes = df[code_col].dropna().head(10).tolist()
        for i, code in enumerate(sample_codes, 1):
            print(f'   {i:2d}. {code}')
    else:
        print(f'\n⚠️  ไม่พบคอลัมน์รหัสสาขา')
    
except Exception as e:
    print(f'❌ ไม่สามารถสร้าง DataFrame ได้: {e}')

print('\n' + '=' * 60)
print('✅ การทดสอบเสร็จสิ้น')
print('=' * 60)
