import pandas as pd
from collections import defaultdict

print('=' * 80)
print('ANALYZING VEHICLE SELECTION RULES FROM PUNTHAI FILE')
print('=' * 80)

# Read Punthai file
df = pd.read_excel('Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx', 
                   sheet_name='2.Punthai', header=1)

# Filter out DC
exclude_codes = ['DC011', 'PTDC', 'PTG Distribution Center']
df_clean = df[df['Trip'].notna()].copy()
df_clean = df_clean[~df_clean['BranchCode'].isin(exclude_codes)].copy()

# Load Master data
try:
    df_master = pd.read_excel('Dc/Master สถานที่ส่ง.xlsx')
    df_merged = df_clean.merge(
        df_master[['Plan Code', 'ละติจูด', 'ลองติจูด']],
        left_on='BranchCode',
        right_on='Plan Code',
        how='left'
    )
    print(f'✅ Merged with Master: {len(df_merged)} records')
except:
    df_merged = df_clean.copy()
    print('❌ Could not merge with Master')

# DC coordinates
DC_LAT, DC_LON = 14.179394, 100.648149

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance using Haversine formula"""
    import math
    if lat1 == 0 or lon1 == 0 or lat2 == 0 or lon2 == 0:
        return 0
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c

# Calculate distance from DC for each branch
df_merged['Distance_from_DC'] = df_merged.apply(
    lambda row: calculate_distance(DC_LAT, DC_LON, row.get('ละติจูด', 0), row.get('ลองติจูด', 0))
    if 'ละติจูด' in df_merged.columns else 0,
    axis=1
)

# Extract vehicle type from Trip no
df_merged['Vehicle_Type'] = df_merged['Trip no'].apply(
    lambda x: str(x)[:2] if pd.notna(x) else 'Unknown'
)

print('\n' + '=' * 80)
print('🚛 VEHICLE USAGE BY DISTANCE')
print('=' * 80)

# Analyze vehicle usage by distance
vehicle_distance_stats = defaultdict(list)

for _, row in df_merged.iterrows():
    vehicle = row['Vehicle_Type']
    distance = row['Distance_from_DC']
    if distance > 0 and vehicle in ['4W', 'JB', '6W']:
        vehicle_distance_stats[vehicle].append(distance)

print('\nDistance statistics by vehicle type:')
for vehicle in ['4W', 'JB', '6W']:
    if vehicle in vehicle_distance_stats:
        distances = vehicle_distance_stats[vehicle]
        print(f'\n{vehicle}:')
        print(f'  Count: {len(distances)} branches')
        print(f'  Min distance: {min(distances):.1f} km')
        print(f'  Max distance: {max(distances):.1f} km')
        print(f'  Avg distance: {sum(distances)/len(distances):.1f} km')
        print(f'  Median distance: {sorted(distances)[len(distances)//2]:.1f} km')

print('\n' + '=' * 80)
print('📊 BRANCH-SPECIFIC VEHICLE RESTRICTIONS')
print('=' * 80)

# Analyze which branches use which vehicles
branch_vehicle_usage = defaultdict(set)

for _, row in df_merged.iterrows():
    branch = row['BranchCode']
    vehicle = row['Vehicle_Type']
    if vehicle in ['4W', 'JB', '6W']:
        branch_vehicle_usage[branch].add(vehicle)

# Find branches that ONLY use specific vehicles
only_4w = [b for b, v in branch_vehicle_usage.items() if v == {'4W'}]
only_jb = [b for b, v in branch_vehicle_usage.items() if v == {'JB'}]
only_6w = [b for b, v in branch_vehicle_usage.items() if v == {'6W'}]
mixed = [b for b, v in branch_vehicle_usage.items() if len(v) > 1]

print(f'\nBranches by vehicle restriction:')
print(f'  Only 4W: {len(only_4w)} branches (รถเล็กเท่านั้น)')
print(f'  Only JB: {len(only_jb)} branches')
print(f'  Only 6W: {len(only_6w)} branches (ต้องรถใหญ่)')
print(f'  Mixed: {len(mixed)} branches (ใช้ได้หลายประเภท)')

# Sample branches with restrictions
if only_4w:
    print(f'\n  Sample 4W-only branches: {only_4w[:10]}')
if only_6w:
    print(f'\n  Sample 6W-only branches: {only_6w[:10]}')

print('\n' + '=' * 80)
print('🎯 DISTANCE vs VEHICLE RULES')
print('=' * 80)

# Analyze distance ranges for each vehicle
print('\nDistance ranges where each vehicle is used:')
for vehicle in ['4W', 'JB', '6W']:
    if vehicle in vehicle_distance_stats:
        distances = sorted(vehicle_distance_stats[vehicle])
        print(f'\n{vehicle}: {distances[0]:.1f} - {distances[-1]:.1f} km')
        
        # Count by distance ranges
        ranges = {
            '0-20km': sum(1 for d in distances if d < 20),
            '20-50km': sum(1 for d in distances if 20 <= d < 50),
            '50-100km': sum(1 for d in distances if 50 <= d < 100),
            '>100km': sum(1 for d in distances if d >= 100)
        }
        
        for range_name, count in ranges.items():
            pct = count / len(distances) * 100
            print(f'  {range_name}: {count:3d} branches ({pct:5.1f}%)')

print('\n' + '=' * 80)
print('🔍 DETAILED TRIP ANALYSIS')
print('=' * 80)

# Analyze trips with distance + vehicle info
trip_analysis = []
for trip_num in sorted(df_merged['Trip'].unique())[:10]:
    trip_data = df_merged[df_merged['Trip'] == trip_num]
    
    vehicle_type = trip_data['Vehicle_Type'].mode()[0] if len(trip_data) > 0 else 'Unknown'
    max_distance = trip_data['Distance_from_DC'].max()
    avg_distance = trip_data['Distance_from_DC'].mean()
    branches = len(trip_data)
    weight = trip_data['TOTALWGT'].sum() if 'TOTALWGT' in trip_data.columns else 0
    cube = trip_data['TOTALCUBE'].sum() if 'TOTALCUBE' in trip_data.columns else 0
    
    trip_analysis.append({
        'Trip': int(trip_num),
        'Vehicle': vehicle_type,
        'Branches': branches,
        'Max_Dist': max_distance,
        'Avg_Dist': avg_distance,
        'Weight': weight,
        'Cube': cube
    })

df_trips = pd.DataFrame(trip_analysis)
print('\nFirst 10 trips with distance and vehicle info:')
print(df_trips.to_string(index=False))

print('\n' + '=' * 80)
print('✅ ANALYSIS COMPLETE')
print('=' * 80)
