import pandas as pd
from collections import defaultdict

print('=' * 70)
print('ANALYZING LOCATION PATTERNS FROM PUNTHAI FILE')
print('=' * 70)

# Read Punthai file
df = pd.read_excel('Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx', 
                   sheet_name='2.Punthai', header=1)

# Filter out DC and distribution centers
exclude_codes = ['DC011', 'PTDC', 'PTG Distribution Center']
df_clean = df[df['Trip'].notna()].copy()
df_clean = df_clean[~df_clean['BranchCode'].isin(exclude_codes)].copy()

# Load Master data for location info
try:
    df_master = pd.read_excel('Dc/Master สถานที่ส่ง.xlsx')
    print(f'\n✅ Loaded Master file: {len(df_master)} branches')
    
    # Merge with Punthai data
    df_merged = df_clean.merge(
        df_master[['Plan Code', 'ตำบล', 'อำเภอ', 'จังหวัด']],
        left_on='BranchCode',
        right_on='Plan Code',
        how='left'
    )
    
    print(f'✅ Merged data: {len(df_merged)} records')
    
except Exception as e:
    print(f'❌ Error loading Master: {e}')
    df_merged = df_clean.copy()

print('\n' + '=' * 70)
print('🗺️  LOCATION-BASED GROUPING ANALYSIS')
print('=' * 70)

# Analyze trip patterns by location
trip_location_patterns = defaultdict(lambda: {
    'trips': [],
    'subdistricts': set(),
    'districts': set(), 
    'provinces': set(),
    'branch_count': 0
})

for trip_num in sorted(df_merged['Trip'].unique()):
    trip_data = df_merged[df_merged['Trip'] == trip_num]
    
    # Get locations
    subdistricts = set(trip_data['ตำบล'].dropna().tolist()) if 'ตำบล' in trip_data.columns else set()
    districts = set(trip_data['อำเภอ'].dropna().tolist()) if 'อำเภอ' in trip_data.columns else set()
    provinces = set(trip_data['จังหวัด'].dropna().tolist()) if 'จังหวัด' in trip_data.columns else set()
    
    # Skip if no location data
    if not provinces:
        continue
    
    # Create location key
    if len(provinces) == 1:
        prov = list(provinces)[0]
        if len(districts) == 1:
            dist = list(districts)[0]
            if len(subdistricts) == 1:
                loc_key = f'{prov}/{dist}/{list(subdistricts)[0]}'
            else:
                loc_key = f'{prov}/{dist}'
        else:
            loc_key = prov
    else:
        loc_key = 'MIXED_PROVINCE'
    
    trip_location_patterns[loc_key]['trips'].append(int(trip_num))
    trip_location_patterns[loc_key]['subdistricts'].update(subdistricts)
    trip_location_patterns[loc_key]['districts'].update(districts)
    trip_location_patterns[loc_key]['provinces'].update(provinces)
    trip_location_patterns[loc_key]['branch_count'] += len(trip_data)

print('\n📍 LOCATION GROUPING PATTERNS:')
print(f'Total location groups: {len(trip_location_patterns)}')
print()

# Sort by number of trips
sorted_patterns = sorted(trip_location_patterns.items(), 
                         key=lambda x: len(x[1]['trips']), 
                         reverse=True)

print('Top 20 location groups:')
for i, (loc_key, data) in enumerate(sorted_patterns[:20], 1):
    trips = data['trips']
    branches = data['branch_count']
    avg_branches = branches / len(trips)
    
    print(f'{i:2d}. {loc_key[:50]:50s} | {len(trips):2d} trips | {branches:3d} branches | avg {avg_branches:.1f}')

print('\n' + '=' * 70)
print('🔍 DETAILED TRIP EXAMPLES BY LOCATION')
print('=' * 70)

# Show examples for top 5 location groups
for loc_key, data in sorted_patterns[:5]:
    print(f'\n📍 {loc_key}')
    print(f'   Trips: {data["trips"][:10]}{"..." if len(data["trips"]) > 10 else ""}')
    print(f'   Provinces: {", ".join(data["provinces"])}')
    if data['districts']:
        print(f'   Districts: {", ".join(list(data["districts"])[:5])}{"..." if len(data["districts"]) > 5 else ""}')
    if data['subdistricts']:
        print(f'   Subdistricts: {", ".join(list(data["subdistricts"])[:5])}{"..." if len(data["subdistricts"]) > 5 else ""}')

print('\n' + '=' * 70)
print('📊 SAME-PROVINCE TRIP ANALYSIS')
print('=' * 70)

same_province_count = 0
mixed_province_count = 0

for trip_num in sorted(df_merged['Trip'].unique()):
    trip_data = df_merged[df_merged['Trip'] == trip_num]
    provinces = set(trip_data['จังหวัด'].dropna().tolist()) if 'จังหวัด' in trip_data.columns else set()
    
    if len(provinces) == 1:
        same_province_count += 1
    elif len(provinces) > 1:
        mixed_province_count += 1

total_trips = same_province_count + mixed_province_count
print(f'Same province trips: {same_province_count} ({same_province_count/total_trips*100:.1f}%)')
print(f'Mixed province trips: {mixed_province_count} ({mixed_province_count/total_trips*100:.1f}%)')

print('\n' + '=' * 70)
print('✅ ANALYSIS COMPLETE')
print('=' * 70)
