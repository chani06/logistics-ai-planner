"""
ทดสอบ Phase 8: รวม Code เดียวกันให้ไปทริปเดียวกัน
"""
import pandas as pd

# จำลองข้อมูล
data = {
    'Code': ['CD1731', 'CD1731', 'CD129', 'CD129', 'CD6734', 'CD6734', 'CD100'],
    'Name': ['PTC แก่งคอย 2', 'แก่งคอย2', 'สระบุรี5', 'FC สระบุรี5', 'หนองแค1', 'หนองแค2', 'อื่น'],
    'Cube': [2.89, 6.14, 7.59, 1.11, 1.74, 4.36, 1.0],
    'Weight': [800, 1400, 2400, 400, 500, 1200, 200],
    'Trip': [84, 85, 45, 74, 46, 62, 50]
}
test_df = pd.DataFrame(data)

print("=" * 70)
print("📊 ก่อน Phase 8:")
print("=" * 70)
print(test_df)
print()

# จำลอง constants
BUFFER = 1.10
LIMITS = {
    '4W': {'max_c': 5, 'max_w': 1200},
    'JB': {'max_c': 7, 'max_w': 2500},
    '6W': {'max_c': 12, 'max_w': 4500}
}
trip_recommended_vehicles = {}

def get_max_vehicle_for_trip(codes):
    return 'JB'  # สระบุรี nearby = JB สูงสุด

def get_province(code):
    return 'สระบุรี'

def get_region_type(prov):
    return 'nearby'

# ===============================================
# Phase 8: รวม Code เดียวกันให้ไปทริปเดียวกัน
# ===============================================
same_code_merged = 0

# หา Code ที่มีหลายทริป
code_trips = test_df.groupby('Code')['Trip'].apply(lambda x: x.unique().tolist()).to_dict()
codes_with_multiple_trips = {code: trips for code, trips in code_trips.items() if len(trips) > 1}

print(f"📍 Codes with multiple trips: {codes_with_multiple_trips}")
print()

trips_to_recheck = set()

for code, trips in codes_with_multiple_trips.items():
    if len(trips) <= 1:
        continue
    
    # รวมทุกกรณี - ย้ายทุก row ไปทริปแรก
    target_trip = min(trips)
    for t in trips:
        if t != target_trip:
            test_df.loc[(test_df['Code'] == code) & (test_df['Trip'] == t), 'Trip'] = target_trip
    same_code_merged += 1
    trips_to_recheck.add(target_trip)
    print(f"   ✅ รวม {code}: ทริป {trips} → ทริป {target_trip}")

print()
print("=" * 70)
print("📊 หลังรวม (ก่อน re-split):")
print("=" * 70)
print(test_df)
print()

# Phase 8.5: Re-split ทริปที่เกิน capacity
print("=" * 70)
print("🔧 Phase 8.5: Re-split ทริปที่เกิน capacity")
print("=" * 70)

for trip_num in trips_to_recheck:
    trip_data = test_df[test_df['Trip'] == trip_num]
    if len(trip_data) == 0:
        continue
    
    trip_codes = list(trip_data['Code'].values)
    trip_w = trip_data['Weight'].sum()
    trip_c = trip_data['Cube'].sum()
    
    max_allowed = 'JB'  # nearby
    limits = LIMITS.get(max_allowed, LIMITS['JB'])
    util = max((trip_w / limits['max_w']) * 100, (trip_c / limits['max_c']) * 100)
    
    print(f"\n📍 ทริป {trip_num}:")
    print(f"   Cube: {trip_c:.2f}, Weight: {trip_w}")
    print(f"   JB limit: Cube {limits['max_c'] * BUFFER:.1f}, Weight {limits['max_w'] * BUFFER:.0f}")
    print(f"   Utilization: {util:.1f}%")
    
    if util > 100:
        num_vehicles = int(util / 100) + 1
        print(f"   ⚠️ เกิน capacity → แยกเป็น {num_vehicles} คัน")
        
        sorted_data = trip_data.sort_values('Cube', ascending=False)
        max_trip = test_df['Trip'].max()
        new_trips = [trip_num] + [max_trip + i + 1 for i in range(num_vehicles - 1)]
        
        vehicle_loads = [0] * num_vehicles
        
        for idx, row in sorted_data.iterrows():
            min_load_idx = vehicle_loads.index(min(vehicle_loads))
            test_df.at[idx, 'Trip'] = new_trips[min_load_idx]
            vehicle_loads[min_load_idx] += row['Cube']
            print(f"      {row['Code']} ({row['Cube']:.2f} cube) → ทริป {new_trips[min_load_idx]}")
        
        for new_trip in new_trips:
            trip_recommended_vehicles[new_trip] = max_allowed
        
        print(f"   ✅ แยกเป็นทริป: {new_trips}")
        print(f"   📦 Load per vehicle: {[f'{l:.2f}' for l in vehicle_loads]}")
    else:
        print(f"   ✅ พอใส่รถเดียว")

print()
print("=" * 70)
print("📊 ผลลัพธ์สุดท้าย:")
print("=" * 70)
print(test_df)
print()

# สรุป
print("=" * 70)
print("📊 สรุปผลลัพธ์:")
print("=" * 70)
for code in codes_with_multiple_trips.keys():
    final_trips = test_df[test_df['Code'] == code]['Trip'].unique()
    total_cube = test_df[test_df['Code'] == code]['Cube'].sum()
    print(f"   {code}: Cube รวม {total_cube:.2f} → ทริป {list(final_trips)}")
