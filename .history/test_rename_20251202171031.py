import pandas as pd

df = pd.read_excel('Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx', sheet_name='2.Punthai', header=1)

print("Original columns:")
for col in df.columns:
    print(f"  '{col}'")

rename_map = {}
for col in df.columns:
    col_clean = str(col).strip()
    col_upper = col_clean.upper().replace(' ', '').replace('_', '')
    
    if col_clean == 'BranchCode':
        rename_map[col] = 'Code'
    elif col_clean == 'Branch':
        rename_map[col] = 'Name'
    elif col_upper in ['TRIPNO', 'TRIP_NO'] or col_clean == 'Trip no':
        rename_map[col] = 'TripNo'
        print(f"Found Trip no: '{col}' -> TripNo")
    elif col_upper == 'TRIP':
        rename_map[col] = 'Trip'
        print(f"Found Trip: '{col}' -> Trip")

print("\nRename map:", rename_map)

df = df.rename(columns=rename_map)
print("\nColumns after rename:")
for col in df.columns:
    print(f"  '{col}'")
print("\nHas TripNo:", 'TripNo' in df.columns)
print("Has Trip:", 'Trip' in df.columns)

# ดูข้อมูลสาขาพระเทพ
if 'Name' in df.columns and 'TripNo' in df.columns:
    phra = df[df['Name'].str.contains('พระเทพ', na=False)]
    print("\nสาขาพระเทพ:")
    print(phra[['Code', 'Name', 'Trip', 'TripNo']].to_string())
