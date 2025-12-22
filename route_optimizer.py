"""
Route Optimizer - Hard Zoning + Hierarchical Far-to-Near Sorting
================================================================
Logistics Trip Planning from DC Wang Noi

Key Features:
1. Hard Zoning: Trips CANNOT cross zone boundaries
2. Hierarchical Sorting: Zone > Prov_Score > Dist_Score > Distance
3. Site Constraints (V_Limit): 4W_Only, Not_6W, 6W_Only, All
4. Pure Punthai Rules: Stricter drop limits
5. Vehicle Optimization: Smallest fit (4W -> JB -> 6W)

Author: Senior Logistics Data Scientist
Date: December 2025
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2, degrees
from typing import Dict, List, Tuple, Optional

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# DC Wang Noi Coordinates
DC_LAT = 14.17939
DC_LNG = 100.6481

# Hard Zoning Map - Provinces to Zones (CRITICAL: Trips cannot cross zones)
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
    
    # NE (North-East) Zone
    'สระบุรี': 'NE', 'Saraburi': 'NE',
    'ลพบุรี': 'NE', 'Lopburi': 'NE',
    'นครราชสีมา': 'NE', 'Nakhon Ratchasima': 'NE',
    'ขอนแก่น': 'NE', 'Khon Kaen': 'NE',
    'ชัยภูมิ': 'NE', 'Chaiyaphum': 'NE',
    'บุรีรัมย์': 'NE', 'Buriram': 'NE',
    'สุรินทร์': 'NE', 'Surin': 'NE',
    
    # EAST_UPPER Zone
    'ปทุมธานี': 'EAST_UPPER', 'Pathum Thani': 'EAST_UPPER',
    'นครนายก': 'EAST_UPPER', 'Nakhon Nayok': 'EAST_UPPER',
    'ปราจีนบุรี': 'EAST_UPPER', 'Prachinburi': 'EAST_UPPER',
    'สระแก้ว': 'EAST_UPPER', 'Sa Kaeo': 'EAST_UPPER',
    
    # EAST_COAST Zone (Rayong, Chonburi, etc.)
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
}

# Zone Sort Order (for consistent sorting)
ZONE_ORDER = {
    'NORTH': 1,
    'NE': 2,
    'EAST_UPPER': 3,
    'EAST_COAST': 4,
    'WEST': 5,
    'SOUTH': 6,
    'OTHERS': 99
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

# Site Constraint Mapping (V_Limit)
V_LIMIT_ALLOWED = {
    '4W_Only': ['4W'],
    'Not_6W': ['4W', 'JB'],
    '6W_Only': ['6W'],
    'All': ['4W', 'JB', '6W']
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate geodesic distance in km between two points using Haversine formula."""
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate bearing (0-360 degrees) from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    
    bearing = degrees(atan2(x, y))
    return (bearing + 360) % 360


def get_zone(province: str) -> str:
    """Get zone for a province. Returns 'OTHERS' if not mapped."""
    return ZONE_MAP.get(province, 'OTHERS')


def get_zone_order(zone: str) -> int:
    """Get numeric order for zone sorting."""
    return ZONE_ORDER.get(zone, 99)


def intersect_v_limits(v_limits: List[str]) -> List[str]:
    """
    Find intersection of allowed vehicles across multiple V_Limit constraints.
    If one stop requires '4W_Only', the entire trip must use 4W.
    """
    if not v_limits:
        return ['4W', 'JB', '6W']
    
    allowed = set(['4W', 'JB', '6W'])
    for v_limit in v_limits:
        constraint_allowed = set(V_LIMIT_ALLOWED.get(v_limit, ['4W', 'JB', '6W']))
        allowed = allowed.intersection(constraint_allowed)
    
    return list(allowed)


def select_smallest_vehicle(
    weight: float, 
    cube: float, 
    drops: int,
    is_punthai: bool,
    allowed_vehicles: List[str]
) -> Optional[str]:
    """
    Select the smallest vehicle that satisfies all constraints.
    Order: 4W -> JB -> 6W (prefer smaller)
    """
    limits = PUNTHAI_LIMITS if is_punthai else VEHICLE_LIMITS
    
    for vehicle in ['4W', 'JB', '6W']:
        if vehicle not in allowed_vehicles:
            continue
            
        v_limits = limits[vehicle]
        if (weight <= v_limits['max_weight'] and 
            cube <= v_limits['max_cube'] and 
            drops <= v_limits['max_drops']):
            return vehicle
    
    return None


def can_add_to_trip(
    current_weight: float,
    current_cube: float,
    current_drops: int,
    add_weight: float,
    add_cube: float,
    is_punthai: bool,
    allowed_vehicles: List[str]
) -> bool:
    """Check if adding a stop would still fit in any allowed vehicle."""
    new_weight = current_weight + add_weight
    new_cube = current_cube + add_cube
    new_drops = current_drops + 1
    
    return select_smallest_vehicle(
        new_weight, new_cube, new_drops, is_punthai, allowed_vehicles
    ) is not None


# ============================================================================
# STEP 1: CONSOLIDATE STOPS BY ROUTE_ID
# ============================================================================

def consolidate_stops(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group data by Route_ID to create consolidated stops.
    - Sum Weight and Cube
    - Mean Lat/Lng
    - Intersect V_Limit constraints (most restrictive wins)
    - Determine Zone from Province
    """
    print("Step 1: Consolidating stops by Route_ID...")
    
    consolidated = df.groupby('Route_ID').agg({
        'Store_Name': 'first',
        'BU': lambda x: list(x),  # Keep all BUs for Punthai check
        'Province': 'first',
        'District': 'first',
        'Lat': 'mean',
        'Lng': 'mean',
        'Weight': 'sum',
        'Cube': 'sum',
        'V_Limit': lambda x: list(x)  # Keep all for intersection
    }).reset_index()
    
    # Count orders per Route_ID (for reference)
    order_counts = df.groupby('Route_ID').size().reset_index(name='Order_Count')
    consolidated = consolidated.merge(order_counts, on='Route_ID')
    
    # Intersect V_Limit constraints
    consolidated['Allowed_Vehicles'] = consolidated['V_Limit'].apply(intersect_v_limits)
    
    # Determine effective V_Limit (most restrictive)
    def get_effective_vlimit(v_limits):
        allowed = intersect_v_limits(v_limits)
        if allowed == ['4W']:
            return '4W_Only'
        elif set(allowed) == {'4W', 'JB'}:
            return 'Not_6W'
        elif allowed == ['6W']:
            return '6W_Only'
        return 'All'
    
    consolidated['Effective_V_Limit'] = consolidated['V_Limit'].apply(get_effective_vlimit)
    
    # Check if stop is Pure Punthai
    consolidated['Is_Punthai'] = consolidated['BU'].apply(
        lambda bus: all(bu == 'PUNTHAI' for bu in bus)
    )
    
    # Determine Zone
    consolidated['Zone'] = consolidated['Province'].apply(get_zone)
    consolidated['Zone_Order'] = consolidated['Zone'].apply(get_zone_order)
    
    print(f"   → {len(consolidated)} unique stops (drops) from {len(df)} orders")
    
    return consolidated


# ============================================================================
# STEP 2: CALCULATE METRICS (Distance, Bearing, Sector)
# ============================================================================

def calculate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate distance, bearing, and sector from DC for each stop."""
    print("Step 2: Calculating distance, bearing, and sector from DC...")
    
    df['Distance'] = df.apply(
        lambda row: haversine_distance(DC_LAT, DC_LNG, row['Lat'], row['Lng']),
        axis=1
    )
    
    df['Bearing'] = df.apply(
        lambda row: calculate_bearing(DC_LAT, DC_LNG, row['Lat'], row['Lng']),
        axis=1
    )
    
    # Sector: 10-degree slices (0-35)
    df['Sector'] = (df['Bearing'] // 10).astype(int)
    
    return df


# ============================================================================
# STEP 3: HIERARCHICAL SCORING (Prov_Score, Dist_Score)
# ============================================================================

def calculate_area_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate hierarchical area scores for Far-to-Near sorting.
    - Prov_Score: Max distance of all stops in same Province
    - Dist_Score: Max distance of all stops in same District
    """
    print("Step 3: Calculating hierarchical area scores (Prov_Score, Dist_Score)...")
    
    # Province Score: Max distance within each province
    prov_scores = df.groupby('Province')['Distance'].max().reset_index()
    prov_scores.columns = ['Province', 'Prov_Score']
    df = df.merge(prov_scores, on='Province', how='left')
    
    # District Score: Max distance within each district
    dist_scores = df.groupby(['Province', 'District'])['Distance'].max().reset_index()
    dist_scores.columns = ['Province', 'District', 'Dist_Score']
    df = df.merge(dist_scores, on=['Province', 'District'], how='left')
    
    return df


# ============================================================================
# STEP 4: ADVANCED HIERARCHICAL SORTING
# ============================================================================

def hierarchical_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort stops with strict hierarchy:
    1. Zone_Order (Ascending) - Hard separation between regions
    2. Prov_Score (Descending) - Process farthest province first
    3. Dist_Score (Descending) - Process farthest district within province
    4. Distance (Descending) - Process farthest stop within district
    """
    print("Step 4: Applying hierarchical sorting (Zone > Prov > District > Stop)...")
    
    df_sorted = df.sort_values(
        by=['Zone_Order', 'Prov_Score', 'Dist_Score', 'Distance'],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)
    
    return df_sorted


# ============================================================================
# STEP 5: TRIP ALLOCATION WITH VEHICLE OPTIMIZATION
# ============================================================================

def allocate_trips(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Allocate stops to trips with Hard Zoning and Vehicle Optimization.
    
    Trigger New Trip If:
    1. Zone Mismatch (Hard Zoning - CRITICAL)
    2. Weight/Cube exceeds max vehicle limit
    3. Drop count exceeds limit (considering Punthai rules)
    4. No compatible vehicle for all stops in trip
    """
    print("Step 5: Allocating trips with vehicle optimization...")
    
    trips = []
    current_trip = {
        'stops': [],
        'zone': None,
        'weight': 0,
        'cube': 0,
        'drops': 0,
        'is_punthai': True,  # Assume Punthai until proven otherwise
        'allowed_vehicles': ['4W', 'JB', '6W']
    }
    
    trip_id = 1
    
    def finalize_trip():
        """Finalize current trip and select optimal vehicle."""
        nonlocal trip_id, current_trip
        
        if not current_trip['stops']:
            return
        
        # Select smallest vehicle that fits
        vehicle = select_smallest_vehicle(
            current_trip['weight'],
            current_trip['cube'],
            current_trip['drops'],
            current_trip['is_punthai'],
            current_trip['allowed_vehicles']
        )
        
        if vehicle is None:
            # Fallback to 6W if no vehicle fits (shouldn't happen)
            vehicle = '6W'
            print(f"   ⚠️ Warning: Trip {trip_id} exceeds all limits, forcing 6W")
        
        trips.append({
            'trip_id': trip_id,
            'vehicle': vehicle,
            'zone': current_trip['zone'],
            'stops': current_trip['stops'].copy(),
            'weight': current_trip['weight'],
            'cube': current_trip['cube'],
            'drops': current_trip['drops'],
            'is_punthai': current_trip['is_punthai']
        })
        
        trip_id += 1
        
        # Reset current trip
        current_trip['stops'] = []
        current_trip['zone'] = None
        current_trip['weight'] = 0
        current_trip['cube'] = 0
        current_trip['drops'] = 0
        current_trip['is_punthai'] = True
        current_trip['allowed_vehicles'] = ['4W', 'JB', '6W']
    
    for idx, row in df.iterrows():
        stop_zone = row['Zone']
        stop_weight = row['Weight']
        stop_cube = row['Cube']
        stop_is_punthai = row['Is_Punthai']
        stop_allowed = row['Allowed_Vehicles']
        
        # Check if we need to start a new trip
        start_new_trip = False
        reason = ""
        
        # Reason 1: Zone Mismatch (HARD ZONING - Most Critical)
        if current_trip['zone'] is not None and current_trip['zone'] != stop_zone:
            start_new_trip = True
            reason = f"Zone change ({current_trip['zone']} → {stop_zone})"
        
        # Reason 2: No compatible vehicle after adding stop
        elif current_trip['stops']:
            new_allowed = list(set(current_trip['allowed_vehicles']).intersection(set(stop_allowed)))
            if not new_allowed:
                start_new_trip = True
                reason = "No compatible vehicle (V_Limit conflict)"
            else:
                # Check if adding stop would exceed all vehicle limits
                new_is_punthai = current_trip['is_punthai'] and stop_is_punthai
                if not can_add_to_trip(
                    current_trip['weight'],
                    current_trip['cube'],
                    current_trip['drops'],
                    stop_weight,
                    stop_cube,
                    new_is_punthai,
                    new_allowed
                ):
                    start_new_trip = True
                    reason = "Capacity exceeded (Weight/Cube/Drops)"
        
        if start_new_trip:
            finalize_trip()
        
        # Add stop to current trip
        current_trip['stops'].append(row.to_dict())
        current_trip['zone'] = stop_zone
        current_trip['weight'] += stop_weight
        current_trip['cube'] += stop_cube
        current_trip['drops'] += 1
        current_trip['is_punthai'] = current_trip['is_punthai'] and stop_is_punthai
        current_trip['allowed_vehicles'] = list(
            set(current_trip['allowed_vehicles']).intersection(set(stop_allowed))
        )
    
    # Finalize last trip
    finalize_trip()
    
    print(f"   → {len(trips)} trips created")
    
    return df, trips


# ============================================================================
# STEP 6: GENERATE OUTPUT
# ============================================================================

def generate_output(trips: List[Dict], original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate final output DataFrame with trip assignments.
    Explode stops back to individual orders.
    """
    print("Step 6: Generating output with trip assignments...")
    
    output_rows = []
    
    for trip in trips:
        trip_id = trip['trip_id']
        vehicle = trip['vehicle']
        zone = trip['zone']
        is_punthai = trip['is_punthai']
        
        for seq, stop in enumerate(trip['stops'], 1):
            # Get original orders for this Route_ID
            route_orders = original_df[original_df['Route_ID'] == stop['Route_ID']]
            
            for _, order in route_orders.iterrows():
                output_rows.append({
                    'Trip_ID': trip_id,
                    'Vehicle': vehicle,
                    'Zone': zone,
                    'Sequence': seq,
                    'Route_ID': stop['Route_ID'],
                    'Store_Name': order['Store_Name'],
                    'BU': order['BU'],
                    'Province': order['Province'],
                    'District': order['District'],
                    'Weight': order['Weight'],
                    'Cube': order['Cube'],
                    'V_Limit': order['V_Limit'],
                    'Distance_km': round(stop['Distance'], 2),
                    'Prov_Score': round(stop['Prov_Score'], 2),
                    'Is_Punthai_Trip': '🅟 Punthai' if is_punthai else '🅼 Mixed'
                })
    
    output_df = pd.DataFrame(output_rows)
    
    return output_df


# ============================================================================
# MOCK DATA GENERATOR
# ============================================================================

def create_mock_data() -> pd.DataFrame:
    """
    Create comprehensive mock dataset with edge cases:
    - Stores in Rayong (EAST_COAST) and Nakhon Ratchasima (NE) for zone testing
    - Mixed V_Limit constraints
    - Multiple orders per Route_ID
    - Pure Punthai and Mixed BU
    """
    
    data = [
        # EAST_COAST Zone - Rayong & Chonburi (should NOT mix with NE)
        {'Route_ID': 'EC001', 'Store_Name': 'MaxMart ระยอง 1', 'BU': 'MAXMART', 
         'Province': 'ระยอง', 'District': 'เมืองระยอง', 'Lat': 12.6833, 'Lng': 101.2500,
         'Weight': 1200, 'Cube': 4.0, 'V_Limit': 'All'},
        {'Route_ID': 'EC001', 'Store_Name': 'MaxMart ระยอง 1', 'BU': 'MAXMART', 
         'Province': 'ระยอง', 'District': 'เมืองระยอง', 'Lat': 12.6833, 'Lng': 101.2500,
         'Weight': 800, 'Cube': 2.5, 'V_Limit': 'All'},
        {'Route_ID': 'EC002', 'Store_Name': 'PTC มาบตาพุด', 'BU': 'PUNTHAI', 
         'Province': 'ระยอง', 'District': 'มาบตาพุด', 'Lat': 12.7167, 'Lng': 101.1500,
         'Weight': 300, 'Cube': 1.0, 'V_Limit': '4W_Only'},
        {'Route_ID': 'EC003', 'Store_Name': 'MaxMart ชลบุรี', 'BU': 'MAXMART', 
         'Province': 'ชลบุรี', 'District': 'เมืองชลบุรี', 'Lat': 13.3611, 'Lng': 100.9847,
         'Weight': 2000, 'Cube': 6.0, 'V_Limit': 'Not_6W'},
        {'Route_ID': 'EC004', 'Store_Name': 'PTC ศรีราชา', 'BU': 'PUNTHAI', 
         'Province': 'ชลบุรี', 'District': 'ศรีราชา', 'Lat': 13.1667, 'Lng': 100.9333,
         'Weight': 250, 'Cube': 0.8, 'V_Limit': 'All'},
        {'Route_ID': 'EC005', 'Store_Name': 'MaxMart พัทยา', 'BU': 'MAXMART', 
         'Province': 'ชลบุรี', 'District': 'บางละมุง', 'Lat': 12.9236, 'Lng': 100.8825,
         'Weight': 1500, 'Cube': 5.0, 'V_Limit': 'All'},
        
        # NE Zone - Nakhon Ratchasima & Saraburi (should NOT mix with EAST_COAST)
        {'Route_ID': 'NE001', 'Store_Name': 'MaxMart โคราช 1', 'BU': 'MAXMART', 
         'Province': 'นครราชสีมา', 'District': 'เมืองนครราชสีมา', 'Lat': 14.9799, 'Lng': 102.0978,
         'Weight': 1800, 'Cube': 5.5, 'V_Limit': 'All'},
        {'Route_ID': 'NE002', 'Store_Name': 'PTC ปากช่อง', 'BU': 'PUNTHAI', 
         'Province': 'นครราชสีมา', 'District': 'ปากช่อง', 'Lat': 14.7167, 'Lng': 101.4167,
         'Weight': 200, 'Cube': 0.7, 'V_Limit': 'All'},
        {'Route_ID': 'NE003', 'Store_Name': 'PTC สระบุรี 1', 'BU': 'PUNTHAI', 
         'Province': 'สระบุรี', 'District': 'เมืองสระบุรี', 'Lat': 14.5289, 'Lng': 100.9103,
         'Weight': 180, 'Cube': 0.6, 'V_Limit': 'All'},
        {'Route_ID': 'NE004', 'Store_Name': 'PTC แก่งคอย', 'BU': 'PUNTHAI', 
         'Province': 'สระบุรี', 'District': 'แก่งคอย', 'Lat': 14.5833, 'Lng': 101.0500,
         'Weight': 150, 'Cube': 0.5, 'V_Limit': 'All'},
        {'Route_ID': 'NE005', 'Store_Name': 'PTC ลพบุรี', 'BU': 'PUNTHAI', 
         'Province': 'ลพบุรี', 'District': 'เมืองลพบุรี', 'Lat': 14.7995, 'Lng': 100.6534,
         'Weight': 170, 'Cube': 0.55, 'V_Limit': '4W_Only'},
        
        # NORTH Zone - Ayutthaya, Ang Thong, Sing Buri
        {'Route_ID': 'N001', 'Store_Name': 'PTC อยุธยา 1', 'BU': 'PUNTHAI', 
         'Province': 'อยุธยา', 'District': 'พระนครศรีอยุธยา', 'Lat': 14.3532, 'Lng': 100.5689,
         'Weight': 200, 'Cube': 0.7, 'V_Limit': 'All'},
        {'Route_ID': 'N002', 'Store_Name': 'PTC อยุธยา 2', 'BU': 'PUNTHAI', 
         'Province': 'อยุธยา', 'District': 'บางปะอิน', 'Lat': 14.2333, 'Lng': 100.5833,
         'Weight': 180, 'Cube': 0.6, 'V_Limit': 'All'},
        {'Route_ID': 'N003', 'Store_Name': 'PTC อ่างทอง', 'BU': 'PUNTHAI', 
         'Province': 'อ่างทอง', 'District': 'เมืองอ่างทอง', 'Lat': 14.5896, 'Lng': 100.4553,
         'Weight': 190, 'Cube': 0.65, 'V_Limit': 'All'},
        {'Route_ID': 'N004', 'Store_Name': 'PTC สิงห์บุรี', 'BU': 'PUNTHAI', 
         'Province': 'สิงห์บุรี', 'District': 'เมืองสิงห์บุรี', 'Lat': 14.8911, 'Lng': 100.4011,
         'Weight': 160, 'Cube': 0.5, 'V_Limit': 'All'},
        {'Route_ID': 'N005', 'Store_Name': 'MaxMart ชัยนาท', 'BU': 'MAXMART', 
         'Province': 'ชัยนาท', 'District': 'เมืองชัยนาท', 'Lat': 15.1853, 'Lng': 100.1253,
         'Weight': 1400, 'Cube': 4.5, 'V_Limit': 'All'},
        
        # EAST_UPPER Zone - Pathum Thani, Nakhon Nayok
        {'Route_ID': 'EU001', 'Store_Name': 'PTC รังสิต 1', 'BU': 'PUNTHAI', 
         'Province': 'ปทุมธานี', 'District': 'ธัญบุรี', 'Lat': 14.0364, 'Lng': 100.7439,
         'Weight': 120, 'Cube': 0.4, 'V_Limit': 'All'},
        {'Route_ID': 'EU002', 'Store_Name': 'PTC รังสิต 2', 'BU': 'PUNTHAI', 
         'Province': 'ปทุมธานี', 'District': 'ลำลูกกา', 'Lat': 14.0167, 'Lng': 100.7500,
         'Weight': 130, 'Cube': 0.45, 'V_Limit': 'All'},
        {'Route_ID': 'EU003', 'Store_Name': 'PTC นครนายก', 'BU': 'PUNTHAI', 
         'Province': 'นครนายก', 'District': 'เมืองนครนายก', 'Lat': 14.2069, 'Lng': 101.2131,
         'Weight': 220, 'Cube': 0.75, 'V_Limit': 'All'},
        {'Route_ID': 'EU004', 'Store_Name': 'MaxMart ปราจีนบุรี', 'BU': 'MAXMART', 
         'Province': 'ปราจีนบุรี', 'District': 'เมืองปราจีนบุรี', 'Lat': 14.0500, 'Lng': 101.3667,
         'Weight': 1600, 'Cube': 5.2, 'V_Limit': '6W_Only'},
        
        # WEST Zone - Nonthaburi, Nakhon Pathom
        {'Route_ID': 'W001', 'Store_Name': 'MaxMart นนทบุรี', 'BU': 'MAXMART', 
         'Province': 'นนทบุรี', 'District': 'เมืองนนทบุรี', 'Lat': 13.8622, 'Lng': 100.5142,
         'Weight': 1100, 'Cube': 3.5, 'V_Limit': 'All'},
        {'Route_ID': 'W002', 'Store_Name': 'PTC นครปฐม', 'BU': 'PUNTHAI', 
         'Province': 'นครปฐม', 'District': 'เมืองนครปฐม', 'Lat': 13.8196, 'Lng': 100.0644,
         'Weight': 240, 'Cube': 0.8, 'V_Limit': 'All'},
        
        # Heavy load test - should trigger 6W
        {'Route_ID': 'EC006', 'Store_Name': 'MaxMart บางแสน', 'BU': 'MAXMART', 
         'Province': 'ชลบุรี', 'District': 'เมืองชลบุรี', 'Lat': 13.2833, 'Lng': 100.9167,
         'Weight': 4500, 'Cube': 15.0, 'V_Limit': 'All'},
    ]
    
    return pd.DataFrame(data)


# ============================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================

def optimize_routes(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Main route optimization function.
    
    Steps:
    1. Consolidate stops by Route_ID
    2. Calculate metrics (Distance, Bearing, Sector)
    3. Calculate area scores (Prov_Score, Dist_Score)
    4. Apply hierarchical sorting
    5. Allocate trips with vehicle optimization
    6. Generate output
    """
    print()
    print("=" * 70)
    print("🚚 ROUTE OPTIMIZER - Hard Zoning + Hierarchical Far-to-Near")
    print("=" * 70)
    print(f"📍 DC Location: Lat {DC_LAT}, Lng {DC_LNG}")
    print(f"📦 Input: {len(df)} orders")
    print()
    
    # Step 1: Consolidate
    stops = consolidate_stops(df)
    
    # Step 2: Calculate metrics
    stops = calculate_metrics(stops)
    
    # Step 3: Calculate area scores
    stops = calculate_area_scores(stops)
    
    # Step 4: Hierarchical sort
    stops_sorted = hierarchical_sort(stops)
    
    # Step 5: Allocate trips
    _, trips = allocate_trips(stops_sorted)
    
    # Step 6: Generate output
    output_df = generate_output(trips, df)
    
    return output_df, trips


def print_trip_summary(trips: List[Dict]):
    """Print summary of all trips."""
    print()
    print("=" * 70)
    print("📊 TRIP SUMMARY")
    print("=" * 70)
    
    for trip in trips:
        punthai_flag = "🅟 Punthai" if trip['is_punthai'] else "🅼 Mixed"
        print(f"Trip {trip['trip_id']:02d} | {trip['vehicle']} | Zone: {trip['zone']:12s} | "
              f"Drops: {trip['drops']:2d} | W: {trip['weight']:,} kg | "
              f"C: {trip['cube']:.2f} | {punthai_flag}")
    
    # Vehicle distribution
    print()
    print("=" * 70)
    print("📈 VEHICLE DISTRIBUTION")
    print("=" * 70)
    vehicle_counts = {}
    for trip in trips:
        v = trip['vehicle']
        vehicle_counts[v] = vehicle_counts.get(v, 0) + 1
    for v, count in sorted(vehicle_counts.items()):
        print(f"   {v}: {count} trips")
    
    # Zone distribution
    print()
    print("=" * 70)
    print("🗺️  ZONE DISTRIBUTION")
    print("=" * 70)
    zone_counts = {}
    for trip in trips:
        z = trip['zone']
        zone_counts[z] = zone_counts.get(z, 0) + 1
    for z, count in sorted(zone_counts.items(), key=lambda x: ZONE_ORDER.get(x[0], 99)):
        print(f"   {z}: {count} trips")


def verify_hard_zoning(trips: List[Dict]):
    """Verify that Hard Zoning is enforced - no trips cross zone boundaries."""
    print()
    print("=" * 70)
    print("✅ HARD ZONING VERIFICATION")
    print("=" * 70)
    
    all_passed = True
    for trip in trips:
        zones_in_trip = set()
        for stop in trip['stops']:
            zones_in_trip.add(stop['Zone'])
        
        if len(zones_in_trip) > 1:
            print(f"❌ Trip {trip['trip_id']}: FAILED - Multiple zones: {zones_in_trip}")
            all_passed = False
        else:
            print(f"✅ Trip {trip['trip_id']}: OK - Single zone: {trip['zone']}")
    
    if all_passed:
        print()
        print("🎉 All trips passed Hard Zoning verification!")
        print("   Rayong (EAST_COAST) orders are NOT mixed with Nakhon Ratchasima (NE) orders.")


def verify_punthai_rules(trips: List[Dict]):
    """Verify that Pure Punthai drop limits are respected."""
    print()
    print("=" * 70)
    print("✅ PUNTHAI DROP LIMIT VERIFICATION")
    print("=" * 70)
    
    punthai_trips = [t for t in trips if t['is_punthai']]
    if not punthai_trips:
        print("   ไม่มี Pure Punthai trips")
        return
    
    for trip in punthai_trips:
        vehicle = trip['vehicle']
        drops = trip['drops']
        limit = PUNTHAI_LIMITS[vehicle]['max_drops']
        status = "✅ OK" if drops <= limit else "❌ EXCEEDS"
        print(f"Trip {trip['trip_id']:02d} | {vehicle} Punthai | "
              f"Drops: {drops} / Max: {limit} | {status}")


def verify_site_constraints(trips: List[Dict], original_df: pd.DataFrame):
    """
    Verify that Vehicle Type matches V_Limit (Site Constraints) for all stops.
    
    V_Limit Rules:
    - '4W_Only': Only 4W allowed
    - 'Not_6W': Only 4W and JB allowed  
    - '6W_Only': Only 6W allowed
    - 'All': Any vehicle allowed
    """
    print()
    print("=" * 70)
    print("✅ SITE CONSTRAINTS (V_Limit) VERIFICATION")
    print("=" * 70)
    
    all_passed = True
    violations = []
    
    for trip in trips:
        trip_id = trip['trip_id']
        vehicle = trip['vehicle']
        
        for stop in trip['stops']:
            route_id = stop['Route_ID']
            
            # Get all V_Limits for this Route_ID from original data
            route_vlimits = original_df[original_df['Route_ID'] == route_id]['V_Limit'].unique()
            
            for v_limit in route_vlimits:
                allowed = V_LIMIT_ALLOWED.get(v_limit, ['4W', 'JB', '6W'])
                
                if vehicle not in allowed:
                    all_passed = False
                    violations.append({
                        'trip_id': trip_id,
                        'route_id': route_id,
                        'vehicle': vehicle,
                        'v_limit': v_limit,
                        'allowed': allowed
                    })
    
    if all_passed:
        print("🎉 All trips passed Site Constraints verification!")
        print()
        # Show summary of V_Limit constraints used
        print("📋 V_Limit Summary per Trip:")
        for trip in trips:
            trip_id = trip['trip_id']
            vehicle = trip['vehicle']
            vlimit_set = set()
            for stop in trip['stops']:
                route_vlimits = original_df[original_df['Route_ID'] == stop['Route_ID']]['V_Limit'].unique()
                vlimit_set.update(route_vlimits)
            
            # Determine effective constraint
            if '4W_Only' in vlimit_set:
                effective = '4W_Only → Must use 4W'
            elif '6W_Only' in vlimit_set:
                effective = '6W_Only → Must use 6W'
            elif 'Not_6W' in vlimit_set:
                effective = 'Not_6W → 4W or JB only'
            else:
                effective = 'All → Any vehicle'
            
            print(f"   Trip {trip_id:02d} | {vehicle} | Constraints: {vlimit_set} | {effective} | ✅")
    else:
        print("❌ VIOLATIONS FOUND:")
        for v in violations:
            print(f"   Trip {v['trip_id']:02d} | Route {v['route_id']} | "
                  f"Used: {v['vehicle']} | V_Limit: {v['v_limit']} | "
                  f"Allowed: {v['allowed']} | ❌ INVALID")
    
    return all_passed


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🔧 Creating mock dataset with edge cases...")
    df = create_mock_data()
    print(f"   Created {len(df)} orders with {df['Route_ID'].nunique()} unique routes")
    
    # Show zone distribution in input
    print()
    print("📍 Input Province/Zone Distribution:")
    for prov in df['Province'].unique():
        zone = get_zone(prov)
        count = len(df[df['Province'] == prov])
        print(f"   {prov} ({zone}): {count} orders")
    
    # Run optimization
    output_df, trips = optimize_routes(df)
    
    # Print summaries
    print_trip_summary(trips)
    
    # Verify Hard Zoning
    verify_hard_zoning(trips)
    
    # Verify Site Constraints (V_Limit)
    verify_site_constraints(trips, df)
    
    # Verify Punthai Rules
    verify_punthai_rules(trips)
    
    # Display output
    print()
    print("=" * 70)
    print("📋 FINAL OUTPUT (Sample)")
    print("=" * 70)
    display_cols = ['Trip_ID', 'Vehicle', 'Zone', 'Sequence', 'Route_ID', 
                    'Store_Name', 'Province', 'District', 'Is_Punthai_Trip']
    print(output_df[display_cols].head(25).to_string(index=False))
    
    # Save to Excel
    output_file = 'route_optimization_result.xlsx'
    output_df.to_excel(output_file, index=False)
    print()
    print(f"💾 Results saved to: {output_file}")
    
    # Final verification message
    print()
    print("=" * 70)
    print("🎯 KEY VERIFICATION POINTS")
    print("=" * 70)
    print("1. Hard Zoning: ระยอง (EAST_COAST) และ นครราชสีมา (NE) อยู่คนละ Trip")
    print("2. Hierarchical Sort: Zone > Prov_Score > Dist_Score > Distance")
    print("3. Vehicle Selection: เลือกรถเล็กสุดที่รับได้ (4W → JB → 6W)")
    print("4. Site Constraints: V_Limit intersection ถูกคำนวณ")
    print("5. Punthai Rules: Pure Punthai trips มี drop limit เข้มงวดขึ้น")
