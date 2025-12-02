import pandas as pd

df = pd.read_excel('Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx', sheet_name='2.Punthai', header=1)

# กรอง DC011 ออก
df_filtered = df[~df['BranchCode'].astype(str).str.contains('DC011', na=False)]

# สรุปต่อทริป
summary = df_filtered.groupby(['Trip', 'Trip no']).agg({
    'BranchCode': 'count',
    'TOTALWGT': 'sum',
    'TOTALCUBE': 'sum'
}).reset_index()
summary.columns = ['Trip', 'รถ', 'จำนวนสาขา', 'น้ำหนัก', 'คิว']

# คำนวณ % การใช้รถ (4W=2500kg/5m3, JB=3500kg/8m3, 6W=5800kg/22m3)
limits = {'4W': (2500, 5), 'JB': (3500, 8), '6W': (5800, 22)}

def calc_percent(row):
    truck = str(row['รถ'])[:2]
    if truck in limits:
        w_max, c_max = limits[truck]
        w_pct = row['น้ำหนัก'] / w_max * 100
        c_pct = row['คิว'] / c_max * 100
        return max(w_pct, c_pct)
    return 0

summary['%ใช้รถ'] = summary.apply(calc_percent, axis=1)
print(summary.to_string())
print()
print(f"เฉลี่ย %ใช้รถ: {summary['%ใช้รถ'].mean():.1f}%")
print(f"ต่ำสุด: {summary['%ใช้รถ'].min():.1f}%")
print(f"สูงสุด: {summary['%ใช้รถ'].max():.1f}%")
