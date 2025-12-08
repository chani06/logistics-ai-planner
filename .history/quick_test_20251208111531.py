import pandas as pd

# อ่านไฟล์ทดสอบ
df = pd.read_excel('punthai_test_data.xlsx')
print('=== ไฟล์ทดสอบ ===')
print(f'จำนวนสาขา: {len(df)}')
print(f'Columns: {df.columns.tolist()}')

# อ่าน Master Data
master = pd.read_excel('Dc/Master สถานที่ส่ง.xlsx')
print(f'\n=== Master Data ===')
print(f'จำนวน: {len(master)}')

# ทดสอบ match
codes = df['Code'].unique()
matched = 0
not_matched = []
for code in codes[:20]:
    m = master[master['Plan Code'] == code]
    if len(m) > 0:
        matched += 1
        row = m.iloc[0]
        print(f'{code}: {row.get("ตำบล", "")} / {row.get("อำเภอ", "")} / {row.get("จังหวัด", "")}')
    else:
        not_matched.append(code)
        print(f'{code}: ไม่พบใน Master')

print(f'\nMatched: {matched}/20')
print(f'Not matched: {not_matched}')
