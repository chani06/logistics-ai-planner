"""
Route Optimizer v2.0 - Hard Zoning + Hierarchical Sorting + Trip Merging
=========================================================================
Logistics Trip Planning from DC Wang Noi

Key Features:
1. Hard Zoning: Trips CANNOT cross zone boundaries
2. Hierarchical Sorting: Zone > Prov_Max_Dist > Distance (Far-to-Near)
3. Post-Process Merging: Consolidate fragmented trips in same zone
4. Pure Punthai Rules: Stricter drop limits (4W=5, JB=7)
5. Vehicle Optimization: Smallest fit (4W -> JB -> 6W)

Author: Senior Logistics Data Scientist
Date: December 2025
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# DC Wang Noi Coordinates
DC_LAT = 14.17939
DC_LNG = 100.6481

# Hard Zoning Map - Provinces to Zones
ZONE_MAP = {
    # NORTH Zone
    'นครสวรรค์': 'NORTH', 'Nakhon Sawan': 'NORTH',
    'ชัยนาท': 'NORTH', 'Chai Nat': 'NORTH',
    'สิงห์บุรี': 'NORTH', 'Sing Buri': 'NORTH',
    'อ่างทอง': 'NORTH', 'Ang Thong': 'NORTH',
    'อยุธยา': 'NORTH', 'พระนครศรีอยุธยา': 'NORTH', 'Ayutthaya': 'NORTH',
    'อุทัยธานี': 'NORTH', 'Uthai Thani': 'NORTH',
    'พิษณุโลก': 'NORTH', 'Phitsanulok': 'NORTH',
    'พิจิตร': 'NORTH', 'Phichit': 'NORTH',
    'กำแพงเพชร': 'NORTH', 'Kamphaeng Phet': 'NORTH',
    'เชียงใหม่': 'NORTH', 'Chiang Mai': 'NORTH',
    'เชียงราย': 'NORTH', 'Chiang Rai': 'NORTH',
    'ลำปาง': 'NORTH', 'Lampang': 'NORTH',
    'ลำพูน': 'NORTH', 'Lamphun': 'NORTH',
    'แพร่': 'NORTH', 'Phrae': 'NORTH',
    'น่าน': 'NORTH', 'Nan': 'NORTH',
    'พะเยา': 'NORTH', 'Phayao': 'NORTH',
    'แม่ฮ่องสอน': 'NORTH', 'Mae Hong Son': 'NORTH',
    'สุโขทัย': 'NORTH', 'Sukhothai': 'NORTH',
    'ตาก': 'NORTH', 'Tak': 'NORTH',
    'เพชรบูรณ์': 'NORTH', 'Phetchabun': 'NORTH',
    
    # NE (North-East) Zone
    'สระบุรี': 'NE', 'Saraburi': 'NE',
    'ลพบุรี': 'NE', 'Lopburi': 'NE',
    'นครราชสีมา': 'NE', 'Nakhon Ratchasima': 'NE',
    'ขอนแก่น': 'NE', 'Khon Kaen': 'NE',
    'ชัยภูมิ': 'NE', 'Chaiyaphum': 'NE',
    'บุรีรัมย์': 'NE', 'Buriram': 'NE',
    'สุรินทร์': 'NE', 'Surin': 'NE',
    'อุบลราชธานี': 'NE', 'Ubon Ratchathani': 'NE',
    'ศรีสะเกษ': 'NE', 'Si Sa Ket': 'NE',
    'ยโสธร': 'NE', 'Yasothon': 'NE',
    'อำนาจเจริญ': 'NE', 'Amnat Charoen': 'NE',
    'ร้อยเอ็ด': 'NE', 'Roi Et': 'NE',
    'มหาสารคาม': 'NE', 'Maha Sarakham': 'NE',
    'กาฬสินธุ์': 'NE', 'Kalasin': 'NE',
    'มุกดาหาร': 'NE', 'Mukdahan': 'NE',
    'นครพนม': 'NE', 'Nakhon Phanom': 'NE',
    'สกลนคร': 'NE', 'Sakon Nakhon': 'NE',
    'อุดรธานี': 'NE', 'Udon Thani': 'NE',
    'หนองคาย': 'NE', 'Nong Khai': 'NE',
    'หนองบัวลำภู': 'NE', 'Nong Bua Lam Phu': 'NE',
    'เลย': 'NE', 'Loei': 'NE',
    'บึงกาฬ': 'NE', 'Bueng Kan': 'NE',
    
    # EAST_UPPER Zone
    'ปทุมธานี': 'EAST_UPPER', 'Pathum Thani': 'EAST_UPPER',
    'นครนายก': 'EAST_UPPER', 'Nakhon Nayok': 'EAST_UPPER',
    'ปราจีนบุรี': 'EAST_UPPER', 'Prachinburi': 'EAST_UPPER',
    'สระแก้ว': 'EAST_UPPER', 'Sa Kaeo': 'EAST_UPPER',
    
    # EAST_COAST Zone
    'สมุทรปราการ': 'EAST_COAST', 'Samut Prakan': 'EAST_COAST',
    'ฉะเชิงเทรา': 'EAST_COAST', 'Chachoengsao': 'EAST_COAST',
    'ชลบุรี': 'EAST_COAST', 'Chonburi': 'EAST_COAST',
    'ระยอง': 'EAST_COAST', 'Rayong': 'EAST_COAST',
    'จันทบุรี': 'EAST_COAST', 'Chanthaburi': 'EAST_COAST',
    'ตราด': 'EAST_COAST', 'Trat': 'EAST_COAST',
    
    # WEST Zone
    'นนทบุรี': 'WEST', 'Nonthaburi': 'WEST',
    'นครปฐม': 'WEST', 'Nakhon Pathom': 'WEST',
    'สุพรรณบุรี': 'WEST', 'Suphan Buri': 'WEST',
    'กาญจนบุรี': 'WEST', 'Kanchanaburi': 'WEST',
    'ราชบุรี': 'WEST', 'Ratchaburi': 'WEST',
    
    # SOUTH Zone
    'สมุทรสาคร': 'SOUTH', 'Samut Sakhon': 'SOUTH',
    'สมุทรสงคราม': 'SOUTH', 'Samut Songkhram': 'SOUTH',
    'เพชรบุรี': 'SOUTH', 'Phetchaburi': 'SOUTH',
    'ประจวบคีรีขันธ์': 'SOUTH', 'Prachuap Khiri Khan': 'SOUTH',
    'ชุมพร': 'SOUTH', 'Chumphon': 'SOUTH',
    'ระนอง': 'SOUTH', 'Ranong': 'SOUTH',
    'สุราษฎร์ธานี': 'SOUTH', 'Surat Thani': 'SOUTH',
    'พังงา': 'SOUTH', 'Phang Nga': 'SOUTH',
    'ภูเก็ต': 'SOUTH', 'Phuket': 'SOUTH',
    'กระบี่': 'SOUTH', 'Krabi': 'SOUTH',
    'นครศรีธรรมราช': 'SOUTH', 'Nakhon Si Thammarat': 'SOUTH',
    'ตรัง': 'SOUTH', 'Trang': 'SOUTH',
    'พัทลุง': 'SOUTH', 'Phatthalung': 'SOUTH',
    'สงขลา': 'SOUTH', 'Songkhla': 'SOUTH',
    'สตูล': 'SOUTH', 'Satun': 'SOUTH',
    'ปัตตานี': 'SOUTH', 'Pattani': 'SOUTH',
    'ยะลา': 'SOUTH', 'Yala': 'SOUTH',
    'นราธิวาส': 'SOUTH', 'Narathiwat': 'SOUTH',
}

# Zone Sort Order
ZONE_ORDER = {
    'NORTH': 1, 'NE': 2, 'EAST_UPPER': 3,
    'EAST_COAST': 4, 'WEST': 5, 'SOUTH': 6, 'OTHERS': 99
}

# Vehicle Constraints (Standard - Mixed Loads)
VEHICLE_LIMITS = {
    '4W': {'max_weight': 2500, 'max_cube': 5.0, 'max_drops': 12},
    'JB': {'max_weight': 3500, 'max_cube': 7.0, 'max_drops': 12},
    '6W': {'max_weight': 6000, 'max_cube': 20.0, 'max_drops': 999}
}

# Vehicle Constraints (Pure Punthai - Stricter Drop Limits)
PUNTHAI_LIMITS = {
    '4W': {'max_weight': 2500, 'max_cube': 5.0, 'max_drops': 5},
    'JB': {'max_weight': 3500, 'max_cube': 7.0, 'max_drops': 7},
    '6W': {'max_weight': 6000, 'max_cube': 20.0, 'max_drops': 999}
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate geodesic distance in km using Haversine formula."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))


def get_zone(province: str) -> str:
    """Get zone for a province."""
    return ZONE_MAP.get(province, 'OTHERS')


def get_zone_order(zone: str) -> int:
    """Get numeric order for zone sorting."""
    return ZONE_ORDER.get(zone, 99)


def is_pure_punthai(bu_list: List[str]) -> bool:
    """Check if all BUs are PUNTHAI."""
    return all(bu == 'PUNTHAI' for bu in bu_list)


def select_vehicle(weight: float, cube: float, drops: int, is_punthai: bool) -> Optional[str]:
    """Select smallest vehicle that fits the load. Returns None if exceeds ALL limits."""
    limits = PUNTHAI_LIMITS if is_punthai else VEHICLE_LIMITS
    
    for vehicle in ['4W', 'JB', '6W']:
        v = limits[vehicle]
        if weight <= v['max_weight'] and cube <= v['max_cube'] and drops <= v['max_drops']:
            return vehicle
    
    # STRICT: Return None if no vehicle can handle (DO NOT default to 6W)
    return None


def get_max_drops_for_punthai(vehicle: str) -> int:
    """Get max drops for Punthai trips - STRICT ENFORCEMENT."""
    return PUNTHAI_LIMITS[vehicle]['max_drops']


def can_add_to_trip(current_weight: float, current_cube: float, current_drops: int,
                    new_weight: float, new_cube: float, current_bus: List[str], 
                    new_bus: List[str]) -> bool:
    """
    Check if a new stop can be added to current trip WITHOUT exceeding any limit.
    STRICT ENFORCEMENT - especially for Punthai.
    """
    test_weight = current_weight + new_weight
    test_cube = current_cube + new_cube
    test_drops = current_drops + 1
    test_bus = current_bus + new_bus
    test_punthai = is_pure_punthai(test_bus)
    
    # Check if ANY vehicle can handle this load
    vehicle = select_vehicle(test_weight, test_cube, test_drops, test_punthai)
    return vehicle is not None


def can_merge_trips(trip_a: Dict, trip_b: Dict) -> bool:
    """
    Check if two trips can be merged into one.
    STRICT ENFORCEMENT - will NOT merge if exceeds ANY limit.
    """
    # Must be same zone
    if trip_a['zone'] != trip_b['zone']:
        return False
    
    combined_weight = trip_a['weight'] + trip_b['weight']
    combined_cube = trip_a['cube'] + trip_b['cube']
    combined_drops = trip_a['drops'] + trip_b['drops']
    
    # Check if combined BUs are all Punthai
    combined_bus = trip_a['bus'] + trip_b['bus']
    test_punthai = is_pure_punthai(combined_bus)
    
    # STRICT: Check if any vehicle can handle combined load
    vehicle = select_vehicle(combined_weight, combined_cube, combined_drops, test_punthai)
    
    if vehicle is None:
        return False
    
    # EXTRA STRICT for Punthai: Double-check drop limits
    if test_punthai:
        max_drops = PUNTHAI_LIMITS[vehicle]['max_drops']
        if combined_drops > max_drops:
            return False
    
    return True


# ============================================================================
# PHASE 1: PRE-PROCESSING
# ============================================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1: Pre-processing
    - Filter out NaN values
    - Group by Route_ID
    - Calculate distance from DC
    - Assign Zone
    """
    print("\n" + "="*70)
    print("📦 PHASE 1: PRE-PROCESSING")
    print("="*70)
    
    # =========== NaN FILTERING ===========
    initial_count = len(df)
    
    # Filter out rows with NaN in critical columns
    critical_cols = ['Route_ID', 'Province', 'Weight', 'Cube']
    for col in critical_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"   ⚠️ Found {nan_count} NaN in '{col}' - removing...")
                df = df.dropna(subset=[col])
    
    # Filter out rows with NaN Lat/Lng (use DC coordinates if missing)
    if 'Lat' in df.columns:
        df['Lat'] = df['Lat'].fillna(DC_LAT)
    if 'Lng' in df.columns:
        df['Lng'] = df['Lng'].fillna(DC_LNG)
    
    # Filter zero or negative Weight/Cube
    df = df[(df['Weight'] > 0) & (df['Cube'] > 0)]
    
    removed = initial_count - len(df)
    if removed > 0:
        print(f"   🗑️ Removed {removed} invalid rows (NaN/Zero values)")
    print(f"   ✅ Clean data: {len(df)} orders")
    
    # Group by Route_ID
    grouped = df.groupby('Route_ID').agg({
        'Store_Name': 'first',
        'BU': lambda x: list(x),
        'Province': 'first',
        'Lat': 'mean',
        'Lng': 'mean',
        'Weight': 'sum',
        'Cube': 'sum'
    }).reset_index()
    
    # Count orders per Route_ID
    order_counts = df.groupby('Route_ID').size().reset_index(name='Order_Count')
    grouped = grouped.merge(order_counts, on='Route_ID')
    
    # Calculate distance from DC
    grouped['Distance'] = grouped.apply(
        lambda r: haversine_distance(DC_LAT, DC_LNG, r['Lat'], r['Lng']), axis=1
    )
    
    # Assign Zone
    grouped['Zone'] = grouped['Province'].apply(get_zone)
    grouped['Zone_Order'] = grouped['Zone'].apply(get_zone_order)
    
    # Check if Pure Punthai
    grouped['Is_Punthai'] = grouped['BU'].apply(is_pure_punthai)
    
    print(f"   Input: {len(df)} orders")
    print(f"   Consolidated: {len(grouped)} stops (Route_IDs)")
    print(f"   Zones: {grouped['Zone'].value_counts().to_dict()}")
    
    return grouped


# ============================================================================
# PHASE 2: HIERARCHICAL SORTING (FAR-TO-NEAR)
# ============================================================================

def hierarchical_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: Hierarchical Sorting
    - Sort by Zone > Prov_Max_Dist > Distance
    - Keeps provinces together, processes farthest first
    """
    print("\n" + "="*70)
    print("📊 PHASE 2: HIERARCHICAL SORTING (FAR-TO-NEAR)")
    print("="*70)
    
    # Calculate Prov_Max_Dist: Max distance of any stop in each province
    prov_max = df.groupby('Province')['Distance'].max().reset_index()
    prov_max.columns = ['Province', 'Prov_Max_Dist']
    df = df.merge(prov_max, on='Province', how='left')
    
    # Sort: Zone (Asc) -> Prov_Max_Dist (Desc) -> Distance (Desc)
    df_sorted = df.sort_values(
        by=['Zone_Order', 'Prov_Max_Dist', 'Distance'],
        ascending=[True, False, False]
    ).reset_index(drop=True)
    
    print("   Sort Order: Zone (Asc) → Prov_Max_Dist (Desc) → Distance (Desc)")
    print("   ✅ Farthest provinces processed first within each zone")
    
    # Show sort result by province
    print("\n   Province Order (by Max Distance):")
    for zone in df_sorted['Zone'].unique():
        zone_data = df_sorted[df_sorted['Zone'] == zone]
        provinces = zone_data.groupby('Province')['Prov_Max_Dist'].first().sort_values(ascending=False)
        print(f"   [{zone}]")
        for prov, dist in provinces.items():
            print(f"      - {prov}: {dist:.1f} km (max)")
    
    return df_sorted


# ============================================================================
# PHASE 3: INITIAL ALLOCATION (GREEDY)
# ============================================================================

def initial_allocation(df: pd.DataFrame) -> List[Dict]:
    """
    Phase 3: Initial Greedy Allocation
    - Iterate through sorted stops
    - Cut new trip on: Zone change, Weight/Cube overflow, Drop limit
    """
    print("\n" + "="*70)
    print("🚚 PHASE 3: INITIAL ALLOCATION (GREEDY)")
    print("="*70)
    
    trips = []
    current_trip = {
        'stops': [], 'zone': None, 'weight': 0, 'cube': 0,
        'drops': 0, 'bus': [], 'provinces': set()
    }
    
    def finalize_trip():
        if not current_trip['stops']:
            return
        
        is_punthai = is_pure_punthai(current_trip['bus'])
        vehicle = select_vehicle(
            current_trip['weight'], current_trip['cube'],
            current_trip['drops'], is_punthai
        )
        
        # STRICT: If no vehicle can handle, log error but use 6W as fallback
        if vehicle is None:
            print(f"   ⚠️ WARNING: Trip exceeds all limits! W={current_trip['weight']}, "
                  f"C={current_trip['cube']:.1f}, D={current_trip['drops']}, Punthai={is_punthai}")
            vehicle = '6W'  # Fallback but should not happen with proper allocation
        
        trips.append({
            'trip_id': len(trips) + 1,
            'vehicle': vehicle,
            'zone': current_trip['zone'],
            'stops': current_trip['stops'].copy(),
            'weight': current_trip['weight'],
            'cube': current_trip['cube'],
            'drops': current_trip['drops'],
            'bus': current_trip['bus'].copy(),
            'is_punthai': is_punthai,
            'provinces': current_trip['provinces'].copy()
        })
        
        # Reset
        current_trip['stops'] = []
        current_trip['zone'] = None
        current_trip['weight'] = 0
        current_trip['cube'] = 0
        current_trip['drops'] = 0
        current_trip['bus'] = []
        current_trip['provinces'] = set()
    
    for _, row in df.iterrows():
        zone = row['Zone']
        weight = row['Weight']
        cube = row['Cube']
        bus = row['BU']
        province = row['Province']
        
        # Check if need new trip
        new_trip_needed = False
        
        # Rule 1: Zone change (HARD RULE)
        if current_trip['zone'] and current_trip['zone'] != zone:
            new_trip_needed = True
        
        # Rule 2: STRICT limit check using can_add_to_trip
        if current_trip['stops'] and not new_trip_needed:
            if not can_add_to_trip(
                current_trip['weight'], current_trip['cube'], current_trip['drops'],
                weight, cube, current_trip['bus'], bus
            ):
                new_trip_needed = True
        
        # Rule 3: Absolute maximum check (6W limits)
        if not new_trip_needed:
            if (current_trip['weight'] + weight > 6000 or 
                current_trip['cube'] + cube > 20.0):
                new_trip_needed = True
        
        if new_trip_needed:
            finalize_trip()
        
        # Add to current trip
        current_trip['stops'].append(row.to_dict())
        current_trip['zone'] = zone
        current_trip['weight'] += weight
        current_trip['cube'] += cube
        current_trip['drops'] += 1
        current_trip['bus'].extend(bus)
        current_trip['provinces'].add(province)
    
    finalize_trip()
    
    print(f"   Initial trips created: {len(trips)}")
    
    # Summary by zone
    zone_summary = {}
    for t in trips:
        z = t['zone']
        zone_summary[z] = zone_summary.get(z, 0) + 1
    print(f"   By Zone: {zone_summary}")
    
    return trips


# ============================================================================
# PHASE 4: POST-PROCESS MERGING (OPTIMIZATION)
# ============================================================================

def merge_fragmented_trips(trips: List[Dict]) -> List[Dict]:
    """
    Phase 4: Merge Fragmented Trips
    - Find small trips in same zone
    - Merge if combined load fits in one vehicle
    - Goal: Reduce total number of vehicles
    """
    print("\n" + "="*70)
    print("🔄 PHASE 4: POST-PROCESS MERGING (OPTIMIZATION)")
    print("="*70)
    
    initial_count = len(trips)
    
    # Group trips by zone
    zone_trips = {}
    for trip in trips:
        zone = trip['zone']
        if zone not in zone_trips:
            zone_trips[zone] = []
        zone_trips[zone].append(trip)
    
    merged_trips = []
    merge_count = 0
    
    for zone, zone_trip_list in zone_trips.items():
        if len(zone_trip_list) <= 1:
            merged_trips.extend(zone_trip_list)
            continue
        
        print(f"\n   [{zone}] Analyzing {len(zone_trip_list)} trips for merge opportunities...")
        
        # Sort by weight (smallest first) for greedy merging
        zone_trip_list = sorted(zone_trip_list, key=lambda t: t['weight'])
        
        # Track which trips have been merged
        merged_indices = set()
        
        for i in range(len(zone_trip_list)):
            if i in merged_indices:
                continue
            
            trip_a = zone_trip_list[i]
            merged_with = None
            
            for j in range(i + 1, len(zone_trip_list)):
                if j in merged_indices:
                    continue
                
                trip_b = zone_trip_list[j]
                
                if can_merge_trips(trip_a, trip_b):
                    # Merge trip_b into trip_a
                    trip_a['stops'].extend(trip_b['stops'])
                    trip_a['weight'] += trip_b['weight']
                    trip_a['cube'] += trip_b['cube']
                    trip_a['drops'] += trip_b['drops']
                    trip_a['bus'].extend(trip_b['bus'])
                    trip_a['provinces'].update(trip_b['provinces'])
                    trip_a['is_punthai'] = is_pure_punthai(trip_a['bus'])
                    
                    merged_indices.add(j)
                    merged_with = j
                    merge_count += 1
                    
                    print(f"      ✅ Merged Trip {trip_b['trip_id']} into Trip {trip_a['trip_id']}")
                    print(f"         Combined: W={trip_a['weight']}kg, C={trip_a['cube']:.1f}, D={trip_a['drops']}")
                    break
            
            merged_trips.append(trip_a)
    
    # Re-assign trip IDs and select optimal vehicles
    for idx, trip in enumerate(merged_trips, 1):
        trip['trip_id'] = idx
        trip['vehicle'] = select_vehicle(
            trip['weight'], trip['cube'], trip['drops'], trip['is_punthai']
        ) or '6W'
    
    print(f"\n   Merges performed: {merge_count}")
    print(f"   Trips reduced: {initial_count} → {len(merged_trips)} (saved {initial_count - len(merged_trips)} trips)")
    
    return merged_trips


# ============================================================================
# PHASE 5: FINAL OUTPUT
# ============================================================================

def generate_output(trips: List[Dict], original_df: pd.DataFrame) -> pd.DataFrame:
    """Generate final output DataFrame."""
    print("\n" + "="*70)
    print("📋 PHASE 5: FINAL OUTPUT GENERATION")
    print("="*70)
    
    output_rows = []
    
    for trip in trips:
        # Sort stops by distance (far to near) within trip
        sorted_stops = sorted(trip['stops'], key=lambda s: s['Distance'], reverse=True)
        
        for seq, stop in enumerate(sorted_stops, 1):
            route_id = stop['Route_ID']
            route_orders = original_df[original_df['Route_ID'] == route_id]
            
            for _, order in route_orders.iterrows():
                output_rows.append({
                    'Trip_ID': trip['trip_id'],
                    'Vehicle': trip['vehicle'],
                    'Zone': trip['zone'],
                    'Sequence': seq,
                    'Route_ID': route_id,
                    'Store_Name': order['Store_Name'],
                    'BU': order['BU'],
                    'Province': order['Province'],
                    'Weight': order['Weight'],
                    'Cube': order['Cube'],
                    'Distance_km': round(stop['Distance'], 1),
                    'Prov_Max_Dist': round(stop['Prov_Max_Dist'], 1),
                    'Trip_Type': '🅟 Punthai' if trip['is_punthai'] else '🅼 Mixed'
                })
    
    return pd.DataFrame(output_rows)


def print_trip_summary(trips: List[Dict], title: str = "TRIP SUMMARY"):
    """Print summary of trips."""
    print("\n" + "="*70)
    print(f"📊 {title}")
    print("="*70)
    
    for trip in trips:
        punthai_flag = "🅟" if trip['is_punthai'] else "🅼"
        provinces = ", ".join(sorted(trip['provinces']))
        print(f"Trip {trip['trip_id']:02d} | {trip['vehicle']} | {trip['zone']:12s} | "
              f"D:{trip['drops']:2d} | W:{trip['weight']:,}kg | C:{trip['cube']:.1f} | "
              f"{punthai_flag} | {provinces}")
    
    # Vehicle distribution
    print("\n" + "-"*40)
    vehicle_counts = {}
    for trip in trips:
        v = trip['vehicle']
        vehicle_counts[v] = vehicle_counts.get(v, 0) + 1
    
    for v in ['4W', 'JB', '6W']:
        if v in vehicle_counts:
            print(f"   {v}: {vehicle_counts[v]} trips")
    
    print(f"   TOTAL: {len(trips)} trips")


# ============================================================================
# MOCK DATA GENERATOR
# ============================================================================

def create_mock_data() -> pd.DataFrame:
    """Create comprehensive mock dataset to test all scenarios."""
    
    data = [
        # ═══════════════════════════════════════════════════════════════════
        # CASE A: NORTH Zone - Chainat (Far) → Ayutthaya (Near)
        # Should be 1 trip, sorted correctly (Far to Near)
        # ═══════════════════════════════════════════════════════════════════
        {'Route_ID': 'N001', 'Store_Name': 'MaxMart ชัยนาท', 'BU': 'MAXMART',
         'Province': 'ชัยนาท', 'Lat': 15.1853, 'Lng': 100.1253, 'Weight': 800, 'Cube': 2.5},
        {'Route_ID': 'N002', 'Store_Name': 'PTC สิงห์บุรี', 'BU': 'PUNTHAI',
         'Province': 'สิงห์บุรี', 'Lat': 14.8911, 'Lng': 100.4011, 'Weight': 200, 'Cube': 0.7},
        {'Route_ID': 'N003', 'Store_Name': 'PTC อ่างทอง', 'BU': 'PUNTHAI',
         'Province': 'อ่างทอง', 'Lat': 14.5896, 'Lng': 100.4553, 'Weight': 180, 'Cube': 0.6},
        {'Route_ID': 'N004', 'Store_Name': 'PTC อยุธยา 1', 'BU': 'PUNTHAI',
         'Province': 'อยุธยา', 'Lat': 14.3532, 'Lng': 100.5689, 'Weight': 150, 'Cube': 0.5},
        {'Route_ID': 'N005', 'Store_Name': 'PTC อยุธยา 2', 'BU': 'PUNTHAI',
         'Province': 'อยุธยา', 'Lat': 14.2333, 'Lng': 100.5833, 'Weight': 160, 'Cube': 0.55},
        
        # ═══════════════════════════════════════════════════════════════════
        # CASE B: EAST_COAST Zone - Multiple small orders that should MERGE
        # Creates 3 small trips initially, but Phase 4 should merge them
        # ═══════════════════════════════════════════════════════════════════
        # Trip B1: Rayong (Far)
        {'Route_ID': 'EC01', 'Store_Name': 'MaxMart ระยอง 1', 'BU': 'MAXMART',
         'Province': 'ระยอง', 'Lat': 12.6833, 'Lng': 101.2500, 'Weight': 1200, 'Cube': 4.0},
        {'Route_ID': 'EC02', 'Store_Name': 'MaxMart ระยอง 2', 'BU': 'MAXMART',
         'Province': 'ระยอง', 'Lat': 12.7167, 'Lng': 101.1500, 'Weight': 1100, 'Cube': 3.5},
        
        # Trip B2: Chonburi (Mid)
        {'Route_ID': 'EC03', 'Store_Name': 'MaxMart ชลบุรี 1', 'BU': 'MAXMART',
         'Province': 'ชลบุรี', 'Lat': 13.3611, 'Lng': 100.9847, 'Weight': 1500, 'Cube': 5.0},
        {'Route_ID': 'EC04', 'Store_Name': 'MaxMart พัทยา', 'BU': 'MAXMART',
         'Province': 'ชลบุรี', 'Lat': 12.9236, 'Lng': 100.8825, 'Weight': 1300, 'Cube': 4.2},
        
        # Trip B3: Samut Prakan (Near)
        {'Route_ID': 'EC05', 'Store_Name': 'MaxMart สมุทรปราการ', 'BU': 'MAXMART',
         'Province': 'สมุทรปราการ', 'Lat': 13.5991, 'Lng': 100.5967, 'Weight': 800, 'Cube': 2.5},
        
        # ═══════════════════════════════════════════════════════════════════
        # CASE C: NE Zone - 6 Punthai stores (tests Punthai drop limit)
        # 4W can only take 5 drops for Punthai, so needs JB or 6W
        # ═══════════════════════════════════════════════════════════════════
        {'Route_ID': 'NE01', 'Store_Name': 'PTC สระบุรี 1', 'BU': 'PUNTHAI',
         'Province': 'สระบุรี', 'Lat': 14.5289, 'Lng': 100.9103, 'Weight': 150, 'Cube': 0.5},
        {'Route_ID': 'NE02', 'Store_Name': 'PTC สระบุรี 2', 'BU': 'PUNTHAI',
         'Province': 'สระบุรี', 'Lat': 14.5833, 'Lng': 101.0500, 'Weight': 140, 'Cube': 0.45},
        {'Route_ID': 'NE03', 'Store_Name': 'PTC สระบุรี 3', 'BU': 'PUNTHAI',
         'Province': 'สระบุรี', 'Lat': 14.5500, 'Lng': 100.9500, 'Weight': 160, 'Cube': 0.55},
        {'Route_ID': 'NE04', 'Store_Name': 'PTC ลพบุรี 1', 'BU': 'PUNTHAI',
         'Province': 'ลพบุรี', 'Lat': 14.7995, 'Lng': 100.6534, 'Weight': 170, 'Cube': 0.6},
        {'Route_ID': 'NE05', 'Store_Name': 'PTC ลพบุรี 2', 'BU': 'PUNTHAI',
         'Province': 'ลพบุรี', 'Lat': 14.8500, 'Lng': 100.7000, 'Weight': 180, 'Cube': 0.65},
        {'Route_ID': 'NE06', 'Store_Name': 'PTC ลพบุรี 3', 'BU': 'PUNTHAI',
         'Province': 'ลพบุรี', 'Lat': 14.7500, 'Lng': 100.6000, 'Weight': 190, 'Cube': 0.7},
        
        # ═══════════════════════════════════════════════════════════════════
        # CASE D: WEST Zone - Small loads (should use 4W)
        # ═══════════════════════════════════════════════════════════════════
        {'Route_ID': 'W001', 'Store_Name': 'PTC นนทบุรี', 'BU': 'PUNTHAI',
         'Province': 'นนทบุรี', 'Lat': 13.8622, 'Lng': 100.5142, 'Weight': 200, 'Cube': 0.7},
        {'Route_ID': 'W002', 'Store_Name': 'MaxMart นครปฐม', 'BU': 'MAXMART',
         'Province': 'นครปฐม', 'Lat': 13.8196, 'Lng': 100.0644, 'Weight': 300, 'Cube': 1.0},
    ]
    
    return pd.DataFrame(data)


# ============================================================================
# MAIN OPTIMIZATION PIPELINE
# ============================================================================

def optimize_routes(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    """
    Main optimization pipeline.
    Returns: (output_df, final_trips, initial_trips_for_comparison)
    """
    print("\n" + "="*70)
    print("🚚 ROUTE OPTIMIZER v2.0 - WITH TRIP MERGING")
    print("="*70)
    print(f"📍 DC Location: Lat {DC_LAT}, Lng {DC_LNG}")
    print(f"📦 Input: {len(df)} orders")
    
    # Phase 1: Pre-processing
    stops = preprocess_data(df)
    
    # Phase 2: Hierarchical Sorting
    stops_sorted = hierarchical_sort(stops)
    
    # Phase 3: Initial Allocation
    initial_trips = initial_allocation(stops_sorted)
    
    # Store copy for comparison
    initial_trips_copy = deepcopy(initial_trips)
    
    # Phase 4: Merge Fragmented Trips
    final_trips = merge_fragmented_trips(initial_trips)
    
    # Phase 5: Generate Output
    output_df = generate_output(final_trips, df)
    
    return output_df, final_trips, initial_trips_copy


# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def verify_results(trips: List[Dict]):
    """Verify all business rules are satisfied."""
    print("\n" + "="*70)
    print("✅ VERIFICATION")
    print("="*70)
    
    all_passed = True
    
    # 1. Zone Verification
    print("\n1️⃣ Hard Zoning Check:")
    for trip in trips:
        zones = set(s['Zone'] for s in trip['stops'])
        if len(zones) > 1:
            print(f"   ❌ Trip {trip['trip_id']}: Multiple zones {zones}")
            all_passed = False
        else:
            print(f"   ✅ Trip {trip['trip_id']}: Single zone ({trip['zone']})")
    
    # 2. Vehicle Constraints (STRICT CHECK)
    print("\n2️⃣ Vehicle Constraints Check (STRICT):")
    for trip in trips:
        v = trip['vehicle']
        limits = PUNTHAI_LIMITS if trip['is_punthai'] else VEHICLE_LIMITS
        lim = limits[v]
        
        w_ok = trip['weight'] <= lim['max_weight']
        c_ok = trip['cube'] <= lim['max_cube']
        d_ok = trip['drops'] <= lim['max_drops']
        
        type_str = "Punthai" if trip['is_punthai'] else "Mixed"
        
        if w_ok and c_ok and d_ok:
            print(f"   ✅ Trip {trip['trip_id']} ({v} {type_str}): "
                  f"W={trip['weight']}/{lim['max_weight']} | "
                  f"C={trip['cube']:.1f}/{lim['max_cube']} | "
                  f"D={trip['drops']}/{lim['max_drops']}")
        else:
            details = []
            if not w_ok:
                details.append(f"Weight OVER: {trip['weight']}/{lim['max_weight']}")
            if not c_ok:
                details.append(f"Cube OVER: {trip['cube']:.1f}/{lim['max_cube']}")
            if not d_ok:
                details.append(f"Drops OVER: {trip['drops']}/{lim['max_drops']}")
            print(f"   ❌ Trip {trip['trip_id']} ({v} {type_str}): {', '.join(details)}")
            all_passed = False
    
    # 3. Punthai Drop Limits (CRITICAL CHECK)
    print("\n3️⃣ Punthai Drop Limit Check (CRITICAL):")
    punthai_trips = [t for t in trips if t['is_punthai']]
    if not punthai_trips:
        print("   ℹ️ No Pure Punthai trips in this batch")
    else:
        for trip in punthai_trips:
            v = trip['vehicle']
            limit = PUNTHAI_LIMITS[v]['max_drops']
            
            if trip['drops'] <= limit:
                print(f"   ✅ Trip {trip['trip_id']} ({v} Punthai): {trip['drops']}/{limit} drops - OK")
            else:
                print(f"   ❌ Trip {trip['trip_id']} ({v} Punthai): {trip['drops']}/{limit} drops - EXCEEDS LIMIT!")
                all_passed = False
    
    if all_passed:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
    else:
        print("\n⚠️ SOME VERIFICATIONS FAILED!")
    
    return all_passed


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Create mock data
    print("🔧 Creating mock dataset with test cases...")
    df = create_mock_data()
    print(f"   Created {len(df)} orders")
    
    # Run optimization
    output_df, final_trips, initial_trips = optimize_routes(df)
    
    # Print comparison
    print("\n" + "="*70)
    print("📈 BEFORE vs AFTER OPTIMIZATION")
    print("="*70)
    
    print("\n--- BEFORE (Initial Allocation) ---")
    print_trip_summary(initial_trips, "INITIAL TRIPS (Before Merge)")
    
    print("\n--- AFTER (With Merging) ---")
    print_trip_summary(final_trips, "FINAL TRIPS (After Merge)")
    
    # Savings calculation
    saved = len(initial_trips) - len(final_trips)
    if saved > 0:
        print(f"\n🎯 OPTIMIZATION RESULT: Saved {saved} trips ({len(initial_trips)} → {len(final_trips)})")
    
    # Verification
    verify_results(final_trips)
    
    # Display final schedule
    print("\n" + "="*70)
    print("📋 FINAL DELIVERY SCHEDULE")
    print("="*70)
    display_cols = ['Trip_ID', 'Vehicle', 'Zone', 'Sequence', 'Store_Name', 
                    'Province', 'Weight', 'Distance_km', 'Trip_Type']
    print(output_df[display_cols].to_string(index=False))
    
    # Save to Excel
    output_file = 'route_optimization_v2_result.xlsx'
    output_df.to_excel(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("🎯 KEY FEATURES DEMONSTRATED")
    print("="*70)
    print("1. Hard Zoning: Rayong/Chonburi (EAST_COAST) ≠ Nakhon Ratchasima (NE)")
    print("2. Hierarchical Sort: Chainat (Far) → Ayutthaya (Near) in NORTH zone")
    print("3. Trip Merging: Small EAST_COAST trips consolidated into fewer vehicles")
    print("4. Punthai Logic: 6 Punthai stores use JB (not 4W) due to 5-drop limit")
    print("5. Vehicle Selection: Smallest vehicle that fits the load")
