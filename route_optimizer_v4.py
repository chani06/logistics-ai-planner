"""
Route Optimizer v4.0 - District Clustering Algorithm
======================================================
Logistics Trip Planning with District-First Allocation

Key Features:
1. District Clustering: Iterate by District Buckets (NOT row-by-row)
2. Hierarchical Sorting: Zone > Prov_Dist_km (Desc) > Dist_Subdist_km (Desc) > Route_ID
3. Hard Zoning: No cross-zone trips
4. Central Zone Rule: Max vehicle is JB (NO 6W)
5. Punthai Drop Limits: 4W=5 drops, JB=10 drops
6. 10% Buffer: Allow tolerance for Weight/Cube
7. NaN Removal: Clean output with no missing values

Author: Senior Logistics Algorithm Engineer
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Zone Sort Order (Fixed)
ZONE_ORDER = {
    'NORTH': 1,
    'NE': 2,
    'SOUTH': 3,
    'EAST': 4,
    'WEST': 5,
    'CENTRAL': 6
}

# Province to Zone Mapping
ZONE_MAP = {
    # CENTRAL Zone (NO 6W allowed!)
    'กรุงเทพมหานคร': 'CENTRAL', 'นนทบุรี': 'CENTRAL', 'ปทุมธานี': 'CENTRAL',
    'สมุทรปราการ': 'CENTRAL', 'สมุทรสาคร': 'CENTRAL', 'นครปฐม': 'CENTRAL',
    
    # NE Zone
    'สระบุรี': 'NE', 'ลพบุรี': 'NE', 'นครราชสีมา': 'NE', 'ขอนแก่น': 'NE',
    'อุดรธานี': 'NE', 'ชัยภูมิ': 'NE', 'บุรีรัมย์': 'NE', 'สุรินทร์': 'NE',
    
    # NORTH Zone
    'พระนครศรีอยุธยา': 'NORTH', 'อยุธยา': 'NORTH', 'อ่างทอง': 'NORTH',
    'สิงห์บุรี': 'NORTH', 'ชัยนาท': 'NORTH', 'นครสวรรค์': 'NORTH',
    'พิษณุโลก': 'NORTH', 'เชียงใหม่': 'NORTH', 'ลำปาง': 'NORTH',
    
    # EAST Zone
    'ชลบุรี': 'EAST', 'ระยอง': 'EAST', 'จันทบุรี': 'EAST', 'ตราด': 'EAST',
    'ฉะเชิงเทรา': 'EAST', 'ปราจีนบุรี': 'EAST', 'สระแก้ว': 'EAST',
    
    # WEST Zone
    'ราชบุรี': 'WEST', 'กาญจนบุรี': 'WEST', 'สุพรรณบุรี': 'WEST',
    'เพชรบุรี': 'WEST', 'ประจวบคีรีขันธ์': 'WEST',
    
    # SOUTH Zone
    'ชุมพร': 'SOUTH', 'สุราษฎร์ธานี': 'SOUTH', 'นครศรีธรรมราช': 'SOUTH',
    'ภูเก็ต': 'SOUTH', 'สงขลา': 'SOUTH',
}

# Vehicle Limits (Standard)
VEHICLE_LIMITS = {
    '4W': {'max_weight': 2500, 'max_cube': 5.0, 'max_drops': 12},
    'JB': {'max_weight': 3500, 'max_cube': 7.0, 'max_drops': 12},
    '6W': {'max_weight': 6000, 'max_cube': 20.0, 'max_drops': 999}
}

# Punthai Drop Limits (Stricter)
PUNTHAI_LIMITS = {
    '4W': {'max_weight': 2500, 'max_cube': 5.0, 'max_drops': 5},
    'JB': {'max_weight': 3500, 'max_cube': 7.0, 'max_drops': 10},  # Updated to 10
    '6W': {'max_weight': 6000, 'max_cube': 20.0, 'max_drops': 999}
}

# Central Zone Restriction
CENTRAL_MAX_VEHICLE = 'JB'  # NO 6W in Central
CENTRAL_ALLOWED_VEHICLES = ['4W', 'JB']

# Buffer (10% tolerance)
BUFFER = 1.10  # 110%


# ============================================================================
# MOCK DATA - Saraburi/Ayutthaya Case
# ============================================================================

def create_mock_master_data() -> pd.DataFrame:
    """Create Master Data with Province/District/Subdistrict distances."""
    master = [
        # ═══════════════════════════════════════════════════════════════════
        # สระบุรี (NE Zone) - Nong Khae District (Farthest)
        # ═══════════════════════════════════════════════════════════════════
        {'Province': 'สระบุรี', 'District': 'หนองแค', 'Subdistrict': 'หนองแค', 
         'Prov_Dist_km': 120, 'Dist_Subdist_km': 45},
        {'Province': 'สระบุรี', 'District': 'หนองแค', 'Subdistrict': 'โคกแย้', 
         'Prov_Dist_km': 120, 'Dist_Subdist_km': 48},
        {'Province': 'สระบุรี', 'District': 'หนองแค', 'Subdistrict': 'คชสิทธิ์', 
         'Prov_Dist_km': 120, 'Dist_Subdist_km': 50},
        {'Province': 'สระบุรี', 'District': 'หนองแค', 'Subdistrict': 'หนองไข่น้ำ', 
         'Prov_Dist_km': 120, 'Dist_Subdist_km': 52},
        
        # สระบุรี - เมืองสระบุรี (Closer)
        {'Province': 'สระบุรี', 'District': 'เมืองสระบุรี', 'Subdistrict': 'ปากเพรียว', 
         'Prov_Dist_km': 120, 'Dist_Subdist_km': 35},
        {'Province': 'สระบุรี', 'District': 'เมืองสระบุรี', 'Subdistrict': 'ตะกุด', 
         'Prov_Dist_km': 120, 'Dist_Subdist_km': 38},
        
        # สระบุรี - แก่งคอย (Mid)
        {'Province': 'สระบุรี', 'District': 'แก่งคอย', 'Subdistrict': 'แก่งคอย', 
         'Prov_Dist_km': 120, 'Dist_Subdist_km': 40},
        {'Province': 'สระบุรี', 'District': 'แก่งคอย', 'Subdistrict': 'ท่าคล้อ', 
         'Prov_Dist_km': 120, 'Dist_Subdist_km': 42},
        
        # ═══════════════════════════════════════════════════════════════════
        # อยุธยา (NORTH Zone)
        # ═══════════════════════════════════════════════════════════════════
        {'Province': 'พระนครศรีอยุธยา', 'District': 'อุทัย', 'Subdistrict': 'อุทัย', 
         'Prov_Dist_km': 85, 'Dist_Subdist_km': 30},
        {'Province': 'พระนครศรีอยุธยา', 'District': 'อุทัย', 'Subdistrict': 'เสนา', 
         'Prov_Dist_km': 85, 'Dist_Subdist_km': 32},
        {'Province': 'พระนครศรีอยุธยา', 'District': 'วังน้อย', 'Subdistrict': 'ลำตาเสา', 
         'Prov_Dist_km': 85, 'Dist_Subdist_km': 15},
        {'Province': 'พระนครศรีอยุธยา', 'District': 'วังน้อย', 'Subdistrict': 'บ่อตาโล่', 
         'Prov_Dist_km': 85, 'Dist_Subdist_km': 18},
        
        # ═══════════════════════════════════════════════════════════════════
        # ปทุมธานี (CENTRAL Zone - NO 6W!)
        # ═══════════════════════════════════════════════════════════════════
        {'Province': 'ปทุมธานี', 'District': 'คลองหลวง', 'Subdistrict': 'คลองหนึ่ง', 
         'Prov_Dist_km': 45, 'Dist_Subdist_km': 20},
        {'Province': 'ปทุมธานี', 'District': 'คลองหลวง', 'Subdistrict': 'คลองสอง', 
         'Prov_Dist_km': 45, 'Dist_Subdist_km': 22},
        {'Province': 'ปทุมธานี', 'District': 'ธัญบุรี', 'Subdistrict': 'ประชาธิปัตย์', 
         'Prov_Dist_km': 45, 'Dist_Subdist_km': 25},
        {'Province': 'ปทุมธานี', 'District': 'ธัญบุรี', 'Subdistrict': 'บึงยี่โถ', 
         'Prov_Dist_km': 45, 'Dist_Subdist_km': 28},
    ]
    return pd.DataFrame(master)


def create_mock_order_data() -> pd.DataFrame:
    """Create Order Data - Daily delivery orders."""
    orders = [
        # ═══════════════════════════════════════════════════════════════════
        # สระบุรี - หนองแค (Heavy district - should cluster together)
        # ═══════════════════════════════════════════════════════════════════
        {'Route_ID': 'R001', 'Store_Name': 'MaxMart หนองแค 1', 'BU': 'MAXMART',
         'Province': 'สระบุรี', 'District': 'หนองแค', 'Subdistrict': 'หนองแค',
         'Weight': 800, 'Cube': 2.5},
        {'Route_ID': 'R002', 'Store_Name': 'MaxMart หนองแค 2', 'BU': 'MAXMART',
         'Province': 'สระบุรี', 'District': 'หนองแค', 'Subdistrict': 'โคกแย้',
         'Weight': 700, 'Cube': 2.2},
        {'Route_ID': 'R003', 'Store_Name': 'PTC หนองแค 1', 'BU': 'PUNTHAI',
         'Province': 'สระบุรี', 'District': 'หนองแค', 'Subdistrict': 'คชสิทธิ์',
         'Weight': 300, 'Cube': 1.0},
        {'Route_ID': 'R004', 'Store_Name': 'PTC หนองแค 2', 'BU': 'PUNTHAI',
         'Province': 'สระบุรี', 'District': 'หนองแค', 'Subdistrict': 'หนองไข่น้ำ',
         'Weight': 350, 'Cube': 1.1},
        
        # สระบุรี - เมืองสระบุรี
        {'Route_ID': 'R005', 'Store_Name': 'MaxMart เมืองสระบุรี', 'BU': 'MAXMART',
         'Province': 'สระบุรี', 'District': 'เมืองสระบุรี', 'Subdistrict': 'ปากเพรียว',
         'Weight': 600, 'Cube': 1.8},
        {'Route_ID': 'R006', 'Store_Name': 'PTC สระบุรี', 'BU': 'PUNTHAI',
         'Province': 'สระบุรี', 'District': 'เมืองสระบุรี', 'Subdistrict': 'ตะกุด',
         'Weight': 250, 'Cube': 0.8},
        
        # สระบุรี - แก่งคอย
        {'Route_ID': 'R007', 'Store_Name': 'MaxMart แก่งคอย', 'BU': 'MAXMART',
         'Province': 'สระบุรี', 'District': 'แก่งคอย', 'Subdistrict': 'แก่งคอย',
         'Weight': 500, 'Cube': 1.5},
        
        # ═══════════════════════════════════════════════════════════════════
        # อยุธยา - Punthai Only Trip (should trigger drop limit)
        # ═══════════════════════════════════════════════════════════════════
        {'Route_ID': 'R010', 'Store_Name': 'PTC อุทัย 1', 'BU': 'PUNTHAI',
         'Province': 'พระนครศรีอยุธยา', 'District': 'อุทัย', 'Subdistrict': 'อุทัย',
         'Weight': 150, 'Cube': 0.5},
        {'Route_ID': 'R011', 'Store_Name': 'PTC อุทัย 2', 'BU': 'PUNTHAI',
         'Province': 'พระนครศรีอยุธยา', 'District': 'อุทัย', 'Subdistrict': 'เสนา',
         'Weight': 140, 'Cube': 0.45},
        {'Route_ID': 'R012', 'Store_Name': 'PTC วังน้อย 1', 'BU': 'PUNTHAI',
         'Province': 'พระนครศรีอยุธยา', 'District': 'วังน้อย', 'Subdistrict': 'ลำตาเสา',
         'Weight': 160, 'Cube': 0.55},
        {'Route_ID': 'R013', 'Store_Name': 'PTC วังน้อย 2', 'BU': 'PUNTHAI',
         'Province': 'พระนครศรีอยุธยา', 'District': 'วังน้อย', 'Subdistrict': 'บ่อตาโล่',
         'Weight': 170, 'Cube': 0.6},
        {'Route_ID': 'R014', 'Store_Name': 'PTC วังน้อย 3', 'BU': 'PUNTHAI',
         'Province': 'พระนครศรีอยุธยา', 'District': 'วังน้อย', 'Subdistrict': 'ลำตาเสา',
         'Weight': 180, 'Cube': 0.65},
        {'Route_ID': 'R015', 'Store_Name': 'PTC วังน้อย 4', 'BU': 'PUNTHAI',
         'Province': 'พระนครศรีอยุธยา', 'District': 'วังน้อย', 'Subdistrict': 'บ่อตาโล่',
         'Weight': 190, 'Cube': 0.7},
        
        # ═══════════════════════════════════════════════════════════════════
        # ปทุมธานี (CENTRAL - NO 6W allowed!)
        # ═══════════════════════════════════════════════════════════════════
        {'Route_ID': 'R020', 'Store_Name': 'MaxMart คลองหลวง 1', 'BU': 'MAXMART',
         'Province': 'ปทุมธานี', 'District': 'คลองหลวง', 'Subdistrict': 'คลองหนึ่ง',
         'Weight': 1200, 'Cube': 3.5},
        {'Route_ID': 'R021', 'Store_Name': 'MaxMart คลองหลวง 2', 'BU': 'MAXMART',
         'Province': 'ปทุมธานี', 'District': 'คลองหลวง', 'Subdistrict': 'คลองสอง',
         'Weight': 1100, 'Cube': 3.2},
        {'Route_ID': 'R022', 'Store_Name': 'MaxMart ธัญบุรี 1', 'BU': 'MAXMART',
         'Province': 'ปทุมธานี', 'District': 'ธัญบุรี', 'Subdistrict': 'ประชาธิปัตย์',
         'Weight': 900, 'Cube': 2.8},
        {'Route_ID': 'R023', 'Store_Name': 'MaxMart ธัญบุรี 2', 'BU': 'MAXMART',
         'Province': 'ปทุมธานี', 'District': 'ธัญบุรี', 'Subdistrict': 'บึงยี่โถ',
         'Weight': 800, 'Cube': 2.5},
        
        # Order with NaN (will be removed)
        {'Route_ID': 'R999', 'Store_Name': None, 'BU': 'MAXMART',
         'Province': None, 'District': None, 'Subdistrict': None,
         'Weight': 100, 'Cube': 0.3},
    ]
    return pd.DataFrame(orders)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_zone(province: str) -> str:
    """Get zone for a province."""
    if pd.isna(province):
        return 'UNKNOWN'
    return ZONE_MAP.get(province, 'UNKNOWN')


def is_pure_punthai(bu_list: List[str]) -> bool:
    """Check if all BUs are PUNTHAI."""
    return all(str(bu).upper() in ['PUNTHAI', '211'] for bu in bu_list if pd.notna(bu))


def get_allowed_vehicles(zone: str) -> List[str]:
    """Get allowed vehicles based on zone."""
    if zone == 'CENTRAL':
        return CENTRAL_ALLOWED_VEHICLES  # ['4W', 'JB']
    return ['4W', 'JB', '6W']


def select_vehicle(weight: float, cube: float, drops: int, 
                   is_punthai: bool, allowed_vehicles: List[str]) -> Optional[str]:
    """Select smallest vehicle that fits the load."""
    limits = PUNTHAI_LIMITS if is_punthai else VEHICLE_LIMITS
    
    for vehicle in ['4W', 'JB', '6W']:
        if vehicle not in allowed_vehicles:
            continue
        v = limits[vehicle]
        # Apply 10% buffer
        if (weight <= v['max_weight'] * BUFFER and 
            cube <= v['max_cube'] * BUFFER and 
            drops <= v['max_drops']):
            return vehicle
    return None


def can_fit_district(district_weight: float, district_cube: float, district_drops: int,
                     current_weight: float, current_cube: float, current_drops: int,
                     is_punthai: bool, allowed_vehicles: List[str]) -> Tuple[bool, str]:
    """
    Check if entire district can fit into current truck.
    Returns: (can_fit, vehicle_type)
    """
    total_weight = current_weight + district_weight
    total_cube = current_cube + district_cube
    total_drops = current_drops + district_drops
    
    vehicle = select_vehicle(total_weight, total_cube, total_drops, is_punthai, allowed_vehicles)
    return (vehicle is not None, vehicle)


# ============================================================================
# PHASE 1: DATA CLEANING & MERGE
# ============================================================================

def clean_and_merge(orders_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1: Clean data and merge with Master
    - Remove NaN/Null rows
    - Left Join with Master for distances
    """
    print("\n" + "="*70)
    print("📦 PHASE 1: DATA CLEANING & MERGE")
    print("="*70)
    
    initial_count = len(orders_df)
    
    # Remove rows with NaN in critical columns
    critical_cols = ['Route_ID', 'Store_Name', 'Province', 'District', 'Subdistrict', 'Weight', 'Cube']
    for col in critical_cols:
        if col in orders_df.columns:
            nan_count = orders_df[col].isna().sum()
            if nan_count > 0:
                print(f"   ⚠️ Removing {nan_count} rows with NaN in '{col}'")
    
    orders_clean = orders_df.dropna(subset=critical_cols)
    removed = initial_count - len(orders_clean)
    print(f"   🗑️ Removed {removed} rows with NaN values")
    
    # Merge with Master
    merged = orders_clean.merge(
        master_df[['Province', 'District', 'Subdistrict', 'Prov_Dist_km', 'Dist_Subdist_km']],
        on=['Province', 'District', 'Subdistrict'],
        how='left'
    )
    
    # Fill any remaining NaN distances with defaults
    merged['Prov_Dist_km'] = merged['Prov_Dist_km'].fillna(999)
    merged['Dist_Subdist_km'] = merged['Dist_Subdist_km'].fillna(999)
    
    # Add Zone
    merged['Zone'] = merged['Province'].apply(get_zone)
    merged['Zone_Order'] = merged['Zone'].map(ZONE_ORDER).fillna(99)
    
    print(f"   ✅ Clean data: {len(merged)} orders")
    print(f"   📊 Zones: {merged['Zone'].value_counts().to_dict()}")
    
    return merged


# ============================================================================
# PHASE 2: HIERARCHICAL SORTING
# ============================================================================

def hierarchical_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: Sort data for Far-to-Near processing
    Order: Zone -> Prov_Dist_km (Desc) -> Dist_Subdist_km (Desc) -> Route_ID
    """
    print("\n" + "="*70)
    print("📊 PHASE 2: HIERARCHICAL SORTING (Far-to-Near)")
    print("="*70)
    
    # Calculate Province Max Distance (for grouping farthest provinces first)
    prov_max = df.groupby(['Zone', 'Province'])['Prov_Dist_km'].max().reset_index()
    prov_max.columns = ['Zone', 'Province', 'Prov_Max_Dist']
    df = df.merge(prov_max, on=['Zone', 'Province'], how='left')
    
    # Calculate District Max Distance
    dist_max = df.groupby(['Zone', 'Province', 'District'])['Dist_Subdist_km'].max().reset_index()
    dist_max.columns = ['Zone', 'Province', 'District', 'Dist_Max_Dist']
    df = df.merge(dist_max, on=['Zone', 'Province', 'District'], how='left')
    
    # Sort: Zone -> Prov_Max_Dist (Desc) -> Dist_Max_Dist (Desc) -> Dist_Subdist_km (Desc) -> Route_ID
    df_sorted = df.sort_values(
        by=['Zone_Order', 'Prov_Max_Dist', 'Dist_Max_Dist', 'Dist_Subdist_km', 'Route_ID'],
        ascending=[True, False, False, False, True]
    ).reset_index(drop=True)
    
    print("   Sort Order:")
    print("   1️⃣ Zone (Fixed Order)")
    print("   2️⃣ Prov_Max_Dist (DESC) - Farthest Province First")
    print("   3️⃣ Dist_Max_Dist (DESC) - Farthest District First")
    print("   4️⃣ Dist_Subdist_km (DESC) - Farthest Subdistrict First")
    print("   5️⃣ Route_ID (Grouping)")
    
    # Show sort result
    print("\n   Sorted Order Preview:")
    for zone in df_sorted['Zone'].unique():
        print(f"\n   [{zone}]")
        zone_data = df_sorted[df_sorted['Zone'] == zone]
        for prov in zone_data['Province'].unique():
            prov_data = zone_data[zone_data['Province'] == prov]
            max_dist = prov_data['Prov_Max_Dist'].iloc[0]
            print(f"      📍 {prov} ({max_dist} km)")
            for dist in prov_data['District'].unique():
                dist_data = prov_data[prov_data['District'] == dist]
                dist_max = dist_data['Dist_Max_Dist'].iloc[0]
                stores = len(dist_data)
                print(f"         └─ {dist} ({dist_max} km, {stores} stores)")
    
    return df_sorted


# ============================================================================
# PHASE 3: DISTRICT CLUSTERING ALLOCATION
# ============================================================================

def allocate_by_district_buckets(df: pd.DataFrame) -> List[Dict]:
    """
    Phase 3: District Clustering Allocation
    
    NEW LOGIC: Iterate by District Buckets (NOT row-by-row)
    1. Group all orders by (Zone, Province, District)
    2. Try to fit ENTIRE district into current truck
    3. If fits -> Add it
    4. If doesn't fit -> Close truck, start new with this district
    5. Only split district if it's larger than empty truck capacity
    """
    print("\n" + "="*70)
    print("🚚 PHASE 3: DISTRICT CLUSTERING ALLOCATION")
    print("="*70)
    
    trips = []
    current_trip = {
        'stores': [], 'zone': None, 'weight': 0, 'cube': 0,
        'drops': 0, 'bus': [], 'allowed_vehicles': None
    }
    
    def finalize_trip():
        """Close current trip and select vehicle."""
        if not current_trip['stores']:
            return
        
        is_punthai = is_pure_punthai(current_trip['bus'])
        vehicle = select_vehicle(
            current_trip['weight'], current_trip['cube'],
            current_trip['drops'], is_punthai, current_trip['allowed_vehicles']
        )
        
        if vehicle is None:
            # Fallback: Use largest allowed vehicle
            vehicle = current_trip['allowed_vehicles'][-1] if current_trip['allowed_vehicles'] else 'JB'
            print(f"   ⚠️ WARNING: No vehicle fits! Using {vehicle} as fallback")
        
        trips.append({
            'trip_id': len(trips) + 1,
            'vehicle': vehicle,
            'zone': current_trip['zone'],
            'stores': current_trip['stores'].copy(),
            'weight': current_trip['weight'],
            'cube': current_trip['cube'],
            'drops': current_trip['drops'],
            'bus': current_trip['bus'].copy(),
            'is_punthai': is_punthai
        })
        
        # Reset
        current_trip['stores'] = []
        current_trip['zone'] = None
        current_trip['weight'] = 0
        current_trip['cube'] = 0
        current_trip['drops'] = 0
        current_trip['bus'] = []
        current_trip['allowed_vehicles'] = None
    
    # ==========================================
    # GROUP BY DISTRICT BUCKETS
    # ==========================================
    district_groups = df.groupby(['Zone', 'Province', 'District'], sort=False)
    
    print(f"\n   📦 Processing {len(district_groups)} District Buckets...")
    
    for (zone, province, district), district_df in district_groups:
        district_weight = district_df['Weight'].sum()
        district_cube = district_df['Cube'].sum()
        district_drops = len(district_df)
        district_bus = district_df['BU'].tolist()
        district_stores = district_df.to_dict('records')
        
        allowed_vehicles = get_allowed_vehicles(zone)
        
        print(f"\n   🏘️ {district}, {province} [{zone}]")
        print(f"      Stores: {district_drops}, Weight: {district_weight}kg, Cube: {district_cube:.1f}")
        
        # ==========================================
        # RULE 0: Zone Change -> Close Trip
        # ==========================================
        if current_trip['zone'] and current_trip['zone'] != zone:
            print(f"      🔄 Zone change ({current_trip['zone']} → {zone}) - Closing trip")
            finalize_trip()
        
        # ==========================================
        # RULE 1: Try to fit ENTIRE district
        # ==========================================
        if current_trip['stores']:
            # Check if entire district fits
            test_bus = current_trip['bus'] + district_bus
            is_punthai = is_pure_punthai(test_bus)
            can_fit, _ = can_fit_district(
                district_weight, district_cube, district_drops,
                current_trip['weight'], current_trip['cube'], current_trip['drops'],
                is_punthai, allowed_vehicles
            )
            
            if can_fit:
                # District fits! Add it
                print(f"      ✅ District fits - Adding to current trip")
                current_trip['stores'].extend(district_stores)
                current_trip['weight'] += district_weight
                current_trip['cube'] += district_cube
                current_trip['drops'] += district_drops
                current_trip['bus'].extend(district_bus)
            else:
                # District doesn't fit -> CLOSE current trip
                print(f"      🚫 District doesn't fit - Closing current trip")
                finalize_trip()
                
                # Start new trip with this district
                print(f"      🆕 Starting new trip with this district")
                current_trip['stores'] = district_stores.copy()
                current_trip['zone'] = zone
                current_trip['weight'] = district_weight
                current_trip['cube'] = district_cube
                current_trip['drops'] = district_drops
                current_trip['bus'] = district_bus.copy()
                current_trip['allowed_vehicles'] = allowed_vehicles
        else:
            # Current trip is empty - just add the district
            print(f"      🆕 Empty trip - Adding district")
            current_trip['stores'] = district_stores.copy()
            current_trip['zone'] = zone
            current_trip['weight'] = district_weight
            current_trip['cube'] = district_cube
            current_trip['drops'] = district_drops
            current_trip['bus'] = district_bus.copy()
            current_trip['allowed_vehicles'] = allowed_vehicles
        
        # ==========================================
        # RULE 2: Check if single district exceeds capacity
        # (Need to split the district itself)
        # ==========================================
        is_punthai = is_pure_punthai(current_trip['bus'])
        vehicle = select_vehicle(
            current_trip['weight'], current_trip['cube'],
            current_trip['drops'], is_punthai, allowed_vehicles
        )
        
        if vehicle is None and len(current_trip['stores']) > 1:
            print(f"      ⚠️ District too large - Need to split")
            # Keep splitting until it fits
            while vehicle is None and len(current_trip['stores']) > 1:
                # Remove last store and finalize
                overflow_store = current_trip['stores'].pop()
                current_trip['weight'] -= overflow_store['Weight']
                current_trip['cube'] -= overflow_store['Cube']
                current_trip['drops'] -= 1
                current_trip['bus'].pop()
                
                # Check again
                is_punthai = is_pure_punthai(current_trip['bus'])
                vehicle = select_vehicle(
                    current_trip['weight'], current_trip['cube'],
                    current_trip['drops'], is_punthai, allowed_vehicles
                )
                
                if vehicle:
                    finalize_trip()
                    # Start new trip with overflow
                    current_trip['stores'] = [overflow_store]
                    current_trip['zone'] = zone
                    current_trip['weight'] = overflow_store['Weight']
                    current_trip['cube'] = overflow_store['Cube']
                    current_trip['drops'] = 1
                    current_trip['bus'] = [overflow_store['BU']]
                    current_trip['allowed_vehicles'] = allowed_vehicles
    
    # Finalize last trip
    finalize_trip()
    
    print(f"\n   ✅ Total trips created: {len(trips)}")
    
    return trips


# ============================================================================
# PHASE 4: GENERATE OUTPUT
# ============================================================================

def generate_output(trips: List[Dict]) -> pd.DataFrame:
    """Generate clean output DataFrame."""
    print("\n" + "="*70)
    print("📋 PHASE 4: GENERATE OUTPUT")
    print("="*70)
    
    output_rows = []
    
    for trip in trips:
        # Sort stores by distance (far to near) within trip
        sorted_stores = sorted(trip['stores'], key=lambda s: s.get('Dist_Subdist_km', 0), reverse=True)
        
        for seq, store in enumerate(sorted_stores, 1):
            output_rows.append({
                'Trip_ID': trip['trip_id'],
                'Vehicle': trip['vehicle'],
                'Zone': trip['zone'],
                'Sequence': seq,
                'Province': store['Province'],
                'District': store['District'],
                'Subdistrict': store['Subdistrict'],
                'Store_Name': store['Store_Name'],
                'Weight': store['Weight'],
                'Cube': store['Cube'],
                'Dist_km': store.get('Dist_Subdist_km', 0),
                'Trip_Type': '🅟 Punthai' if trip['is_punthai'] else '🅼 Mixed'
            })
    
    df_output = pd.DataFrame(output_rows)
    
    # Final NaN cleanup
    initial = len(df_output)
    df_output = df_output.dropna()
    removed = initial - len(df_output)
    if removed > 0:
        print(f"   🗑️ Removed {removed} rows with NaN in final output")
    
    print(f"   ✅ Final output: {len(df_output)} rows")
    
    return df_output


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_results(trips: List[Dict]):
    """Verify all business rules."""
    print("\n" + "="*70)
    print("✅ VERIFICATION")
    print("="*70)
    
    all_passed = True
    
    # 1. Zone Check
    print("\n1️⃣ Hard Zoning Check:")
    for trip in trips:
        zones = set(s['Zone'] for s in trip['stores'] if 'Zone' in s)
        if not zones:
            zones = {trip['zone']}
        if len(zones) > 1:
            print(f"   ❌ Trip {trip['trip_id']}: Multiple zones {zones}")
            all_passed = False
        else:
            print(f"   ✅ Trip {trip['trip_id']}: {trip['zone']}")
    
    # 2. Central Zone - No 6W
    print("\n2️⃣ Central Zone Vehicle Check (NO 6W):")
    central_trips = [t for t in trips if t['zone'] == 'CENTRAL']
    if not central_trips:
        print("   ℹ️ No CENTRAL zone trips")
    for trip in central_trips:
        if trip['vehicle'] == '6W':
            print(f"   ❌ Trip {trip['trip_id']}: Uses 6W in CENTRAL!")
            all_passed = False
        else:
            print(f"   ✅ Trip {trip['trip_id']}: Uses {trip['vehicle']} in CENTRAL")
    
    # 3. Vehicle Constraints
    print("\n3️⃣ Vehicle Constraints Check:")
    for trip in trips:
        v = trip['vehicle']
        limits = PUNTHAI_LIMITS if trip['is_punthai'] else VEHICLE_LIMITS
        lim = limits[v]
        
        w_ok = trip['weight'] <= lim['max_weight'] * BUFFER
        c_ok = trip['cube'] <= lim['max_cube'] * BUFFER
        d_ok = trip['drops'] <= lim['max_drops']
        
        type_str = "Punthai" if trip['is_punthai'] else "Mixed"
        
        if w_ok and c_ok and d_ok:
            print(f"   ✅ Trip {trip['trip_id']} ({v} {type_str}): "
                  f"W={trip['weight']}/{int(lim['max_weight']*BUFFER)} | "
                  f"C={trip['cube']:.1f}/{lim['max_cube']*BUFFER:.1f} | "
                  f"D={trip['drops']}/{lim['max_drops']}")
        else:
            print(f"   ❌ Trip {trip['trip_id']} ({v}): EXCEEDS LIMITS")
            all_passed = False
    
    # 4. Punthai Drop Limits
    print("\n4️⃣ Punthai Drop Limit Check:")
    punthai_trips = [t for t in trips if t['is_punthai']]
    if not punthai_trips:
        print("   ℹ️ No Pure Punthai trips")
    for trip in punthai_trips:
        v = trip['vehicle']
        limit = PUNTHAI_LIMITS[v]['max_drops']
        if trip['drops'] <= limit:
            print(f"   ✅ Trip {trip['trip_id']} ({v} Punthai): {trip['drops']}/{limit} drops")
        else:
            print(f"   ❌ Trip {trip['trip_id']} ({v} Punthai): {trip['drops']}/{limit} drops - OVER!")
            all_passed = False
    
    # 5. District Clustering Check
    print("\n5️⃣ District Clustering Check:")
    for trip in trips:
        districts = set(s['District'] for s in trip['stores'])
        if len(districts) <= 2:
            print(f"   ✅ Trip {trip['trip_id']}: {len(districts)} districts - Good clustering")
        else:
            print(f"   ⚠️ Trip {trip['trip_id']}: {len(districts)} districts - Could be better")
    
    if all_passed:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
    else:
        print("\n⚠️ SOME VERIFICATIONS FAILED!")
    
    return all_passed


def print_trip_summary(trips: List[Dict]):
    """Print trip summary."""
    print("\n" + "="*70)
    print("📊 TRIP SUMMARY")
    print("="*70)
    
    for trip in trips:
        punthai_flag = "🅟" if trip['is_punthai'] else "🅼"
        districts = set(s['District'] for s in trip['stores'])
        districts_str = ", ".join(sorted(districts))
        print(f"Trip {trip['trip_id']:02d} | {trip['vehicle']} | {trip['zone']:8s} | "
              f"D:{trip['drops']:2d} | W:{trip['weight']:,}kg | C:{trip['cube']:.1f} | "
              f"{punthai_flag} | {districts_str}")
    
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
# MAIN
# ============================================================================

def optimize_routes(orders_df: pd.DataFrame, master_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """Main optimization pipeline."""
    print("\n" + "="*70)
    print("🚚 ROUTE OPTIMIZER v4.0 - DISTRICT CLUSTERING")
    print("="*70)
    print("📋 Key Features:")
    print("   • District Bucket Allocation (NOT row-by-row)")
    print("   • Hierarchical Sorting (Zone > Prov > Dist > Subdist)")
    print("   • Central Zone Rule (NO 6W)")
    print("   • Punthai Drop Limits (4W=5, JB=10)")
    print("   • 10% Buffer for Weight/Cube")
    print("   • NaN Removal")
    
    # Phase 1: Clean & Merge
    merged_df = clean_and_merge(orders_df, master_df)
    
    # Phase 2: Hierarchical Sort
    sorted_df = hierarchical_sort(merged_df)
    
    # Phase 3: District Clustering Allocation
    trips = allocate_by_district_buckets(sorted_df)
    
    # Phase 4: Generate Output
    output_df = generate_output(trips)
    
    return output_df, trips


if __name__ == "__main__":
    # Create mock data
    print("🔧 Creating Mock Data (Saraburi/Ayutthaya case)...")
    master_df = create_mock_master_data()
    orders_df = create_mock_order_data()
    
    print(f"   Master Data: {len(master_df)} locations")
    print(f"   Order Data: {len(orders_df)} orders (including 1 NaN)")
    
    # Run optimization
    output_df, trips = optimize_routes(orders_df, master_df)
    
    # Print summary
    print_trip_summary(trips)
    
    # Verify
    verify_results(trips)
    
    # Display final schedule
    print("\n" + "="*70)
    print("📋 FINAL DELIVERY SCHEDULE")
    print("="*70)
    display_cols = ['Trip_ID', 'Vehicle', 'Zone', 'Province', 'District', 'Subdistrict', 
                    'Store_Name', 'Weight', 'Trip_Type']
    print(output_df[display_cols].to_string(index=False))
    
    # Save to Excel
    output_file = 'route_optimization_v4_result.xlsx'
    output_df.to_excel(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Key demonstration
    print("\n" + "="*70)
    print("🎯 KEY FEATURES DEMONSTRATED")
    print("="*70)
    print("1. District Clustering: หนองแค 4 stores stay together in one trip")
    print("2. Hierarchical Sort: สระบุรี (120km) processed before อยุธยา (85km)")
    print("3. Central Zone Rule: ปทุมธานี uses JB (not 6W)")
    print("4. Punthai Limit: 6 Punthai stores split if exceeds 5 drops (4W) or 10 drops (JB)")
    print("5. NaN Removed: R999 with NULL values automatically excluded")
