"""
ทดสอบ logic รวม Code เดียวกันให้ไปทริปเดียวกัน
"""
import pandas as pd

# ทดสอบ logic รวม Code เดียวกัน
data = {
    'Code': ['CD1731', 'CD1731', 'CD129', 'CD129', 'CD6734', 'CD6734', 'CD100'],
    'Name': ['PTC แก่งคอย 2', 'แก่งคอย2', 'สระบุรี5', 'FC สระบุรี5', 'หนองแค1', 'หนองแค2', 'อื่น'],
    'Cube': [2.89, 6.14, 7.59, 1.11, 1.74, 4.36, 1.0],
    'Weight': [800, 1400, 2400, 400, 500, 1200, 200],
    'Trip': [84, 85, 45, 74, 46, 62, 50]  # จำลองว่า Code เดียวกันแยกทริป
}
df = pd.DataFrame(data)

print('=' * 60)
print('Before merge:')
print('=' * 60)
print(df)
print()

# Logic รวม Code เดียวกัน
BUFFER = 1.10
LIMITS = {
    '4W': {'max_c': 5, 'max_w': 1200},
    'JB': {'max_c': 7, 'max_w': 2500},
    '6W': {'max_c': 12, 'max_w': 4500}
}

# หา Code ที่มีหลายทริป
code_trips = df.groupby('Code')['Trip'].apply(lambda x: x.unique().tolist()).to_dict()
codes_with_multiple_trips = {code: trips for code, trips in code_trips.items() if len(trips) > 1}

print(f'Codes with multiple trips: {codes_with_multiple_trips}')
print()

for code, trips in codes_with_multiple_trips.items():
    total_cube = df[df['Code'] == code]['Cube'].sum()
    total_weight = df[df['Code'] == code]['Weight'].sum()
    
    # สระบุรี nearby → ใช้ JB สูงสุด
    max_vehicle = 'JB'  # จำลอง
    limit = LIMITS[max_vehicle]
    max_cube = limit['max_c'] * BUFFER
    max_weight = limit['max_w'] * BUFFER
    fits_in_one = total_cube <= max_cube and total_weight <= max_weight
    
    print(f'📍 {code}: Cube={total_cube:.2f}, Weight={total_weight}')
    print(f'   JB limit: Cube={max_cube:.1f}, Weight={max_weight:.0f}')
    print(f'   Fits in one JB: {fits_in_one}')
    
    if fits_in_one:
        target_trip = min(trips)
        for t in trips:
            if t != target_trip:
                df.loc[(df['Code'] == code) & (df['Trip'] == t), 'Trip'] = target_trip
        print(f'   ✅ Merged to trip {target_trip}')
    else:
        print(f'   ❌ Keep separate (exceeds JB capacity)')
    print()

print('=' * 60)
print('After merge:')
print('=' * 60)
print(df)
print()

# สรุป
print('=' * 60)
print('📊 สรุป:')
print('=' * 60)
for code in codes_with_multiple_trips.keys():
    final_trips = df[df['Code'] == code]['Trip'].unique()
    print(f'  {code}: ทริป {list(final_trips)}')
