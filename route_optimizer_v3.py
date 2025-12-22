"""
Route Optimizer v3.0 - Master Data Merge + Hierarchical Sorting
================================================================
Logistics Trip Planning from DC Wang Noi

Key Features:
1. Master Data Merge: Left Join Order Data with Master for Distance_KM & Region
2. Hierarchical Sorting: Region > Province (Max Dist) > District (Max Dist) > Subdistrict (Dist)
3. Central Region Rule: NO 6W trucks allowed in Central
4. Punthai Logic: Stricter drop limits (4W=5, JB=7)
5. Route_ID Grouping: Same Route_ID stays together
6. NaN Removal: Clean output with no missing values

Author: Senior Logistics Data Scientist
Date: December 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Region Sort Order (Fixed)
REGION_ORDER = {
    'NORTH': 1,
    'NE': 2,
    'SOUTH': 3,
    'EAST': 4,
    'CENTRAL': 5
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

# Central Region Vehicle Restriction
CENTRAL_ALLOWED_VEHICLES = ['4W', 'JB']  # NO 6W in Central


# ============================================================================
# MOCK DATA GENERATORS
# ============================================================================

def create_master_data() -> pd.DataFrame:
    """
    Create Master Data with standard hierarchy and distance for every location.
    Columns: [Province, District, Subdistrict, Region, Distance_KM]
    """
    master_data = [
        # ═══════════════════════════════════════════════════════════════════
        # NORTH Region
        # ═══════════════════════════════════════════════════════════════════
        {'Province': 'เชียงใหม่', 'District': 'เมืองเชียงใหม่', 'Subdistrict': 'ช้างเผือก', 'Region': 'NORTH', 'Distance_KM': 685},
        {'Province': 'เชียงใหม่', 'District': 'เมืองเชียงใหม่', 'Subdistrict': 'ศรีภูมิ', 'Region': 'NORTH', 'Distance_KM': 680},
        {'Province': 'เชียงใหม่', 'District': 'หางดง', 'Subdistrict': 'หางดง', 'Region': 'NORTH', 'Distance_KM': 700},
        {'Province': 'เชียงใหม่', 'District': 'หางดง', 'Subdistrict': 'หนองแก๋ว', 'Region': 'NORTH', 'Distance_KM': 695},
        {'Province': 'ลำปาง', 'District': 'เมืองลำปาง', 'Subdistrict': 'หัวเวียง', 'Region': 'NORTH', 'Distance_KM': 600},
        {'Province': 'ลำปาง', 'District': 'เมืองลำปาง', 'Subdistrict': 'สบตุ๋ย', 'Region': 'NORTH', 'Distance_KM': 595},
        {'Province': 'พิษณุโลก', 'District': 'เมืองพิษณุโลก', 'Subdistrict': 'ในเมือง', 'Region': 'NORTH', 'Distance_KM': 380},
        {'Province': 'พิษณุโลก', 'District': 'เมืองพิษณุโลก', 'Subdistrict': 'อรัญญิก', 'Region': 'NORTH', 'Distance_KM': 375},
        {'Province': 'นครสวรรค์', 'District': 'เมืองนครสวรรค์', 'Subdistrict': 'ปากน้ำโพ', 'Region': 'NORTH', 'Distance_KM': 240},
        {'Province': 'นครสวรรค์', 'District': 'เมืองนครสวรรค์', 'Subdistrict': 'นครสวรรค์ตก', 'Region': 'NORTH', 'Distance_KM': 235},
        
        # ═══════════════════════════════════════════════════════════════════
        # NE (North-East) Region
        # ═══════════════════════════════════════════════════════════════════
        {'Province': 'ขอนแก่น', 'District': 'เมืองขอนแก่น', 'Subdistrict': 'ในเมือง', 'Region': 'NE', 'Distance_KM': 450},
        {'Province': 'ขอนแก่น', 'District': 'เมืองขอนแก่น', 'Subdistrict': 'ศิลา', 'Region': 'NE', 'Distance_KM': 445},
        {'Province': 'ขอนแก่น', 'District': 'บ้านไผ่', 'Subdistrict': 'บ้านไผ่', 'Region': 'NE', 'Distance_KM': 420},
        {'Province': 'นครราชสีมา', 'District': 'เมืองนครราชสีมา', 'Subdistrict': 'ในเมือง', 'Region': 'NE', 'Distance_KM': 260},
        {'Province': 'นครราชสีมา', 'District': 'เมืองนครราชสีมา', 'Subdistrict': 'โพธิ์กลาง', 'Region': 'NE', 'Distance_KM': 255},
        {'Province': 'นครราชสีมา', 'District': 'ปากช่อง', 'Subdistrict': 'ปากช่อง', 'Region': 'NE', 'Distance_KM': 180},
        {'Province': 'อุดรธานี', 'District': 'เมืองอุดรธานี', 'Subdistrict': 'หมากแข้ง', 'Region': 'NE', 'Distance_KM': 560},
        {'Province': 'อุดรธานี', 'District': 'เมืองอุดรธานี', 'Subdistrict': 'บ้านตาด', 'Region': 'NE', 'Distance_KM': 555},
        
        # ═══════════════════════════════════════════════════════════════════
        # SOUTH Region
        # ═══════════════════════════════════════════════════════════════════
        {'Province': 'สุราษฎร์ธานี', 'District': 'เมืองสุราษฎร์ธานี', 'Subdistrict': 'ตลาด', 'Region': 'SOUTH', 'Distance_KM': 645},
        {'Province': 'สุราษฎร์ธานี', 'District': 'เมืองสุราษฎร์ธานี', 'Subdistrict': 'มะขามเตี้ย', 'Region': 'SOUTH', 'Distance_KM': 640},
        {'Province': 'ภูเก็ต', 'District': 'เมืองภูเก็ต', 'Subdistrict': 'ตลาดใหญ่', 'Region': 'SOUTH', 'Distance_KM': 860},
        {'Province': 'ภูเก็ต', 'District': 'เมืองภูเก็ต', 'Subdistrict': 'ราไวย์', 'Region': 'SOUTH', 'Distance_KM': 870},
        {'Province': 'นครศรีธรรมราช', 'District': 'เมืองนครศรีธรรมราช', 'Subdistrict': 'ในเมือง', 'Region': 'SOUTH', 'Distance_KM': 780},
        {'Province': 'นครศรีธรรมราช', 'District': 'เมืองนครศรีธรรมราช', 'Subdistrict': 'ท่าวัง', 'Region': 'SOUTH', 'Distance_KM': 775},
        
        # ═══════════════════════════════════════════════════════════════════
        # EAST Region
        # ═══════════════════════════════════════════════════════════════════
        {'Province': 'ชลบุรี', 'District': 'เมืองชลบุรี', 'Subdistrict': 'บางปลาสร้อย', 'Region': 'EAST', 'Distance_KM': 80},
        {'Province': 'ชลบุรี', 'District': 'เมืองชลบุรี', 'Subdistrict': 'บ้านสวน', 'Region': 'EAST', 'Distance_KM': 85},
        {'Province': 'ชลบุรี', 'District': 'พัทยา', 'Subdistrict': 'นาเกลือ', 'Region': 'EAST', 'Distance_KM': 145},
        {'Province': 'ชลบุรี', 'District': 'พัทยา', 'Subdistrict': 'หนองปรือ', 'Region': 'EAST', 'Distance_KM': 150},
        {'Province': 'ระยอง', 'District': 'เมืองระยอง', 'Subdistrict': 'ท่าประดู่', 'Region': 'EAST', 'Distance_KM': 180},
        {'Province': 'ระยอง', 'District': 'เมืองระยอง', 'Subdistrict': 'เชิงเนิน', 'Region': 'EAST', 'Distance_KM': 175},
        {'Province': 'ระยอง', 'District': 'บ้านฉาง', 'Subdistrict': 'บ้านฉาง', 'Region': 'EAST', 'Distance_KM': 195},
        {'Province': 'จันทบุรี', 'District': 'เมืองจันทบุรี', 'Subdistrict': 'ตลาด', 'Region': 'EAST', 'Distance_KM': 245},
        {'Province': 'จันทบุรี', 'District': 'เมืองจันทบุรี', 'Subdistrict': 'วัดใหม่', 'Region': 'EAST', 'Distance_KM': 240},
        
        # ═══════════════════════════════════════════════════════════════════
        # CENTRAL Region (NO 6W allowed here!)
        # ═══════════════════════════════════════════════════════════════════
        {'Province': 'กรุงเทพมหานคร', 'District': 'บางรัก', 'Subdistrict': 'สีลม', 'Region': 'CENTRAL', 'Distance_KM': 35},
        {'Province': 'กรุงเทพมหานคร', 'District': 'บางรัก', 'Subdistrict': 'สุริยวงศ์', 'Region': 'CENTRAL', 'Distance_KM': 33},
        {'Province': 'กรุงเทพมหานคร', 'District': 'ปทุมวัน', 'Subdistrict': 'ลุมพินี', 'Region': 'CENTRAL', 'Distance_KM': 40},
        {'Province': 'กรุงเทพมหานคร', 'District': 'ปทุมวัน', 'Subdistrict': 'ปทุมวัน', 'Region': 'CENTRAL', 'Distance_KM': 38},
        {'Province': 'กรุงเทพมหานคร', 'District': 'จตุจักร', 'Subdistrict': 'จตุจักร', 'Region': 'CENTRAL', 'Distance_KM': 25},
        {'Province': 'กรุงเทพมหานคร', 'District': 'จตุจักร', 'Subdistrict': 'ลาดยาว', 'Region': 'CENTRAL', 'Distance_KM': 28},
        {'Province': 'นนทบุรี', 'District': 'เมืองนนทบุรี', 'Subdistrict': 'บางกระสอ', 'Region': 'CENTRAL', 'Distance_KM': 30},
        {'Province': 'นนทบุรี', 'District': 'เมืองนนทบุรี', 'Subdistrict': 'ตลาดขวัญ', 'Region': 'CENTRAL', 'Distance_KM': 32},
        {'Province': 'ปทุมธานี', 'District': 'เมืองปทุมธานี', 'Subdistrict': 'บางปรอก', 'Region': 'CENTRAL', 'Distance_KM': 20},
        {'Province': 'ปทุมธานี', 'District': 'เมืองปทุมธานี', 'Subdistrict': 'บ้านใหม่', 'Region': 'CENTRAL', 'Distance_KM': 22},
        {'Province': 'สมุทรปราการ', 'District': 'เมืองสมุทรปราการ', 'Subdistrict': 'ปากน้ำ', 'Region': 'CENTRAL', 'Distance_KM': 50},
        {'Province': 'สมุทรปราการ', 'District': 'เมืองสมุทรปราการ', 'Subdistrict': 'บางเมือง', 'Region': 'CENTRAL', 'Distance_KM': 48},
    ]
    
    return pd.DataFrame(master_data)


def create_order_data() -> pd.DataFrame:
    """
    Create Order Data (Daily delivery orders).
    Columns: [Route_ID, Store_Name, BU, Province, District, Subdistrict, Weight, Cube, V_Limit]
    
    V_Limit values:
    - 'All': Can use any vehicle
    - '4W_Only': Must use 4W
    - 'Not_6W': Cannot use 6W (4W or JB only)
    - '6W_Only': Must use 6W
    """
    order_data = [
        # ═══════════════════════════════════════════════════════════════════
        # NORTH Region Orders
        # ═══════════════════════════════════════════════════════════════════
        {'Route_ID': 'R001', 'Store_Name': 'MaxMart เชียงใหม่ 1', 'BU': 'MAXMART',
         'Province': 'เชียงใหม่', 'District': 'เมืองเชียงใหม่', 'Subdistrict': 'ช้างเผือก',
         'Weight': 800, 'Cube': 2.5, 'V_Limit': 'All'},
        {'Route_ID': 'R001', 'Store_Name': 'MaxMart เชียงใหม่ 2', 'BU': 'MAXMART',
         'Province': 'เชียงใหม่', 'District': 'หางดง', 'Subdistrict': 'หางดง',
         'Weight': 600, 'Cube': 2.0, 'V_Limit': 'All'},
        {'Route_ID': 'R002', 'Store_Name': 'PTC ลำปาง', 'BU': 'PUNTHAI',
         'Province': 'ลำปาง', 'District': 'เมืองลำปาง', 'Subdistrict': 'หัวเวียง',
         'Weight': 200, 'Cube': 0.7, 'V_Limit': 'All'},
        
        # ═══════════════════════════════════════════════════════════════════
        # NE Region Orders - Pure Punthai Trip (should trigger drop limit)
        # ═══════════════════════════════════════════════════════════════════
        {'Route_ID': 'R003', 'Store_Name': 'PTC ขอนแก่น 1', 'BU': 'PUNTHAI',
         'Province': 'ขอนแก่น', 'District': 'เมืองขอนแก่น', 'Subdistrict': 'ในเมือง',
         'Weight': 150, 'Cube': 0.5, 'V_Limit': 'All'},
        {'Route_ID': 'R004', 'Store_Name': 'PTC ขอนแก่น 2', 'BU': 'PUNTHAI',
         'Province': 'ขอนแก่น', 'District': 'บ้านไผ่', 'Subdistrict': 'บ้านไผ่',
         'Weight': 140, 'Cube': 0.45, 'V_Limit': 'All'},
        {'Route_ID': 'R005', 'Store_Name': 'PTC โคราช 1', 'BU': 'PUNTHAI',
         'Province': 'นครราชสีมา', 'District': 'เมืองนครราชสีมา', 'Subdistrict': 'ในเมือง',
         'Weight': 160, 'Cube': 0.55, 'V_Limit': 'All'},
        {'Route_ID': 'R006', 'Store_Name': 'PTC โคราช 2', 'BU': 'PUNTHAI',
         'Province': 'นครราชสีมา', 'District': 'ปากช่อง', 'Subdistrict': 'ปากช่อง',
         'Weight': 170, 'Cube': 0.6, 'V_Limit': 'All'},
        {'Route_ID': 'R007', 'Store_Name': 'PTC อุดร 1', 'BU': 'PUNTHAI',
         'Province': 'อุดรธานี', 'District': 'เมืองอุดรธานี', 'Subdistrict': 'หมากแข้ง',
         'Weight': 180, 'Cube': 0.65, 'V_Limit': 'All'},
        {'Route_ID': 'R008', 'Store_Name': 'PTC อุดร 2', 'BU': 'PUNTHAI',
         'Province': 'อุดรธานี', 'District': 'เมืองอุดรธานี', 'Subdistrict': 'บ้านตาด',
         'Weight': 190, 'Cube': 0.7, 'V_Limit': 'All'},
        
        # ═══════════════════════════════════════════════════════════════════
        # EAST Region Orders - Heavy Load (needs 6W)
        # ═══════════════════════════════════════════════════════════════════
        {'Route_ID': 'R009', 'Store_Name': 'MaxMart ระยอง 1', 'BU': 'MAXMART',
         'Province': 'ระยอง', 'District': 'บ้านฉาง', 'Subdistrict': 'บ้านฉาง',
         'Weight': 2000, 'Cube': 6.0, 'V_Limit': 'All'},
        {'Route_ID': 'R009', 'Store_Name': 'MaxMart ระยอง 2', 'BU': 'MAXMART',
         'Province': 'ระยอง', 'District': 'เมืองระยอง', 'Subdistrict': 'ท่าประดู่',
         'Weight': 1800, 'Cube': 5.5, 'V_Limit': 'All'},
        {'Route_ID': 'R010', 'Store_Name': 'MaxMart ชลบุรี', 'BU': 'MAXMART',
         'Province': 'ชลบุรี', 'District': 'พัทยา', 'Subdistrict': 'หนองปรือ',
         'Weight': 1500, 'Cube': 4.5, 'V_Limit': 'All'},
        
        # ═══════════════════════════════════════════════════════════════════
        # CENTRAL Region Orders (NO 6W allowed!)
        # ═══════════════════════════════════════════════════════════════════
        {'Route_ID': 'R011', 'Store_Name': 'MaxMart สีลม', 'BU': 'MAXMART',
         'Province': 'กรุงเทพมหานคร', 'District': 'บางรัก', 'Subdistrict': 'สีลม',
         'Weight': 800, 'Cube': 2.5, 'V_Limit': 'Not_6W'},
        {'Route_ID': 'R011', 'Store_Name': 'MaxMart ลุมพินี', 'BU': 'MAXMART',
         'Province': 'กรุงเทพมหานคร', 'District': 'ปทุมวัน', 'Subdistrict': 'ลุมพินี',
         'Weight': 700, 'Cube': 2.2, 'V_Limit': 'Not_6W'},
        {'Route_ID': 'R012', 'Store_Name': 'PTC จตุจักร', 'BU': 'PUNTHAI',
         'Province': 'กรุงเทพมหานคร', 'District': 'จตุจักร', 'Subdistrict': 'จตุจักร',
         'Weight': 300, 'Cube': 1.0, 'V_Limit': 'All'},
        {'Route_ID': 'R013', 'Store_Name': 'MaxMart นนทบุรี', 'BU': 'MAXMART',
         'Province': 'นนทบุรี', 'District': 'เมืองนนทบุรี', 'Subdistrict': 'บางกระสอ',
         'Weight': 500, 'Cube': 1.5, 'V_Limit': 'All'},
        {'Route_ID': 'R014', 'Store_Name': 'PTC ปทุมธานี', 'BU': 'PUNTHAI',
         'Province': 'ปทุมธานี', 'District': 'เมืองปทุมธานี', 'Subdistrict': 'บางปรอก',
         'Weight': 250, 'Cube': 0.8, 'V_Limit': '4W_Only'},
        
        # ═══════════════════════════════════════════════════════════════════
        # SOUTH Region Orders
        # ═══════════════════════════════════════════════════════════════════
        {'Route_ID': 'R015', 'Store_Name': 'MaxMart ภูเก็ต', 'BU': 'MAXMART',
         'Province': 'ภูเก็ต', 'District': 'เมืองภูเก็ต', 'Subdistrict': 'ราไวย์',
         'Weight': 1200, 'Cube': 4.0, 'V_Limit': 'All'},
        {'Route_ID': 'R016', 'Store_Name': 'PTC สุราษฎร์', 'BU': 'PUNTHAI',
         'Province': 'สุราษฎร์ธานี', 'District': 'เมืองสุราษฎร์ธานี', 'Subdistrict': 'ตลาด',
         'Weight': 200, 'Cube': 0.7, 'V_Limit': 'All'},
        
        # ═══════════════════════════════════════════════════════════════════
        # Order with intentional NaN (will be removed)
        # ═══════════════════════════════════════════════════════════════════
        {'Route_ID': 'R999', 'Store_Name': 'Unknown Store', 'BU': 'MAXMART',
         'Province': 'ไม่ระบุ', 'District': 'ไม่ระบุ', 'Subdistrict': 'ไม่ระบุ',
         'Weight': 100, 'Cube': 0.3, 'V_Limit': 'All'},
    ]
    
    return pd.DataFrame(order_data)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_pure_punthai(bu_list: List[str]) -> bool:
    """Check if all BUs in the list are PUNTHAI."""
    return all(str(bu).upper() in ['PUNTHAI', '211'] for bu in bu_list)


def get_allowed_vehicles(region: str, v_limit: str) -> List[str]:
    """
    Get allowed vehicles based on region and V_Limit.
    Central Region: NO 6W allowed.
    """
    # Start with all vehicles
    all_vehicles = ['4W', 'JB', '6W']
    
    # Apply V_Limit constraint
    if v_limit == '4W_Only':
        allowed = ['4W']
    elif v_limit == 'Not_6W':
        allowed = ['4W', 'JB']
    elif v_limit == '6W_Only':
        allowed = ['6W']
    else:  # 'All'
        allowed = all_vehicles.copy()
    
    # Apply Central Region constraint (NO 6W)
    if region == 'CENTRAL' and '6W' in allowed:
        allowed.remove('6W')
    
    return allowed


def select_vehicle(weight: float, cube: float, drops: int, 
                   is_punthai: bool, allowed_vehicles: List[str]) -> Optional[str]:
    """
    Select smallest vehicle that fits the load from allowed vehicles.
    Returns None if no vehicle can handle the load.
    """
    limits = PUNTHAI_LIMITS if is_punthai else VEHICLE_LIMITS
    
    for vehicle in ['4W', 'JB', '6W']:
        if vehicle not in allowed_vehicles:
            continue
        
        v = limits[vehicle]
        if weight <= v['max_weight'] and cube <= v['max_cube'] and drops <= v['max_drops']:
            return vehicle
    
    return None


def can_add_to_trip(current_weight: float, current_cube: float, current_drops: int,
                    new_weight: float, new_cube: float, 
                    current_bus: List[str], new_bu: str,
                    allowed_vehicles: List[str]) -> bool:
    """
    Check if a new stop can be added to current trip.
    STRICT: Must fit within at least one allowed vehicle.
    """
    test_weight = current_weight + new_weight
    test_cube = current_cube + new_cube
    test_drops = current_drops + 1
    test_bus = current_bus + [new_bu]
    test_punthai = is_pure_punthai(test_bus)
    
    vehicle = select_vehicle(test_weight, test_cube, test_drops, test_punthai, allowed_vehicles)
    return vehicle is not None


# ============================================================================
# PHASE 1: MERGE DATA
# ============================================================================

def merge_data(orders_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1: Merge Order Data with Master Data (Left Join)
    Join on: Province, District, Subdistrict
    Retrieve: Distance_KM, Region
    """
    print("\n" + "="*70)
    print("📦 PHASE 1: MERGE DATA (Left Join)")
    print("="*70)
    
    initial_count = len(orders_df)
    print(f"   Orders: {initial_count}")
    print(f"   Master: {len(master_df)}")
    
    # Left Join
    merged = orders_df.merge(
        master_df[['Province', 'District', 'Subdistrict', 'Region', 'Distance_KM']],
        on=['Province', 'District', 'Subdistrict'],
        how='left'
    )
    
    # Check for unmatched (NaN)
    nan_count = merged['Distance_KM'].isna().sum()
    if nan_count > 0:
        print(f"   ⚠️ Found {nan_count} unmatched orders (will be removed)")
        unmatched = merged[merged['Distance_KM'].isna()][['Route_ID', 'Store_Name', 'Province', 'District', 'Subdistrict']]
        print("   Unmatched locations:")
        for _, row in unmatched.iterrows():
            print(f"      - {row['Store_Name']}: {row['Province']}/{row['District']}/{row['Subdistrict']}")
    
    # Remove NaN rows
    merged_clean = merged.dropna(subset=['Distance_KM', 'Region'])
    removed = len(merged) - len(merged_clean)
    
    print(f"   ✅ Clean data: {len(merged_clean)} orders (removed {removed} with NaN)")
    
    return merged_clean


# ============================================================================
# PHASE 2: HIERARCHICAL SORTING
# ============================================================================

def hierarchical_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: Hierarchical Sorting (Far-to-Near)
    
    Step 1: Sort Region (Fixed Order: NORTH -> NE -> SOUTH -> EAST -> CENTRAL)
    Step 2: Within Region, sort Province by Max Distance (Farthest first)
    Step 3: Within Province, sort District by Max Distance (Farthest first)
    Step 4: Within District, sort Subdistrict by Distance_KM (Descending)
    """
    print("\n" + "="*70)
    print("📊 PHASE 2: HIERARCHICAL SORTING (FAR-TO-NEAR)")
    print("="*70)
    
    # Add Region Order
    df['Region_Order'] = df['Region'].map(REGION_ORDER).fillna(99)
    
    # Calculate Province Max Distance
    prov_max = df.groupby(['Region', 'Province'])['Distance_KM'].max().reset_index()
    prov_max.columns = ['Region', 'Province', 'Prov_Max_Dist']
    df = df.merge(prov_max, on=['Region', 'Province'], how='left')
    
    # Calculate District Max Distance
    dist_max = df.groupby(['Region', 'Province', 'District'])['Distance_KM'].max().reset_index()
    dist_max.columns = ['Region', 'Province', 'District', 'Dist_Max_Dist']
    df = df.merge(dist_max, on=['Region', 'Province', 'District'], how='left')
    
    # Sort: Region_Order (Asc) -> Prov_Max_Dist (Desc) -> Dist_Max_Dist (Desc) -> Distance_KM (Desc)
    df_sorted = df.sort_values(
        by=['Region_Order', 'Prov_Max_Dist', 'Dist_Max_Dist', 'Distance_KM'],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)
    
    print("   Sort Order:")
    print("   1️⃣ Region (NORTH → NE → SOUTH → EAST → CENTRAL)")
    print("   2️⃣ Province (Max Distance DESC)")
    print("   3️⃣ District (Max Distance DESC)")
    print("   4️⃣ Subdistrict (Distance_KM DESC)")
    
    # Show hierarchy
    print("\n   Sorted Hierarchy:")
    for region in df_sorted['Region'].unique():
        print(f"\n   [{region}]")
        region_data = df_sorted[df_sorted['Region'] == region]
        for prov in region_data['Province'].unique():
            prov_data = region_data[region_data['Province'] == prov]
            max_dist = prov_data['Prov_Max_Dist'].iloc[0]
            print(f"      📍 {prov} (Max: {max_dist} km)")
            for dist in prov_data['District'].unique():
                dist_data = prov_data[prov_data['District'] == dist]
                dist_max = dist_data['Dist_Max_Dist'].iloc[0]
                print(f"         └─ {dist} (Max: {dist_max} km)")
    
    return df_sorted


# ============================================================================
# PHASE 3: CONSOLIDATE BY ROUTE_ID
# ============================================================================

def consolidate_routes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 3: Consolidate orders by Route_ID
    Orders with same Route_ID must stay together.
    """
    print("\n" + "="*70)
    print("🔗 PHASE 3: CONSOLIDATE BY ROUTE_ID")
    print("="*70)
    
    # Group by Route_ID
    grouped = df.groupby('Route_ID').agg({
        'Store_Name': lambda x: ' | '.join(x),
        'BU': lambda x: list(x),
        'Province': 'first',
        'District': 'first',
        'Subdistrict': 'first',
        'Region': 'first',
        'Weight': 'sum',
        'Cube': 'sum',
        'V_Limit': lambda x: x.iloc[0],  # Take first V_Limit
        'Distance_KM': 'max',
        'Region_Order': 'first',
        'Prov_Max_Dist': 'first',
        'Dist_Max_Dist': 'first'
    }).reset_index()
    
    # Count orders per Route_ID
    order_counts = df.groupby('Route_ID').size().reset_index(name='Order_Count')
    grouped = grouped.merge(order_counts, on='Route_ID')
    
    print(f"   Input: {len(df)} orders")
    print(f"   Consolidated: {len(grouped)} routes (unique Route_IDs)")
    
    # Re-sort after consolidation
    grouped = grouped.sort_values(
        by=['Region_Order', 'Prov_Max_Dist', 'Dist_Max_Dist', 'Distance_KM'],
        ascending=[True, False, False, False]
    ).reset_index(drop=True)
    
    return grouped


# ============================================================================
# PHASE 4: TRIP ALLOCATION
# ============================================================================

def allocate_trips(df: pd.DataFrame) -> List[Dict]:
    """
    Phase 4: Allocate routes to trips
    - Same Region only
    - Respect vehicle constraints
    - Apply Central Region rule (NO 6W)
    - Apply Punthai drop limits
    """
    print("\n" + "="*70)
    print("🚚 PHASE 4: TRIP ALLOCATION")
    print("="*70)
    
    trips = []
    current_trip = {
        'routes': [], 'region': None, 'weight': 0, 'cube': 0,
        'drops': 0, 'bus': [], 'allowed_vehicles': None
    }
    
    def finalize_trip():
        if not current_trip['routes']:
            return
        
        is_punthai = is_pure_punthai(current_trip['bus'])
        vehicle = select_vehicle(
            current_trip['weight'], current_trip['cube'],
            current_trip['drops'], is_punthai, current_trip['allowed_vehicles']
        )
        
        if vehicle is None:
            print(f"   ⚠️ WARNING: No vehicle can handle trip! Using JB as fallback.")
            vehicle = 'JB'
        
        trips.append({
            'trip_id': len(trips) + 1,
            'vehicle': vehicle,
            'region': current_trip['region'],
            'routes': current_trip['routes'].copy(),
            'weight': current_trip['weight'],
            'cube': current_trip['cube'],
            'drops': current_trip['drops'],
            'bus': current_trip['bus'].copy(),
            'is_punthai': is_punthai,
            'allowed_vehicles': current_trip['allowed_vehicles']
        })
        
        # Reset
        current_trip['routes'] = []
        current_trip['region'] = None
        current_trip['weight'] = 0
        current_trip['cube'] = 0
        current_trip['drops'] = 0
        current_trip['bus'] = []
        current_trip['allowed_vehicles'] = None
    
    for _, row in df.iterrows():
        region = row['Region']
        weight = row['Weight']
        cube = row['Cube']
        bus = row['BU']
        v_limit = row['V_Limit']
        
        # Get allowed vehicles for this route
        route_allowed = get_allowed_vehicles(region, v_limit)
        
        # Check if need new trip
        new_trip_needed = False
        
        # Rule 1: Region change
        if current_trip['region'] and current_trip['region'] != region:
            new_trip_needed = True
        
        # Rule 2: Capacity check
        if current_trip['routes'] and not new_trip_needed:
            # Use intersection of allowed vehicles
            combined_allowed = list(set(current_trip['allowed_vehicles']) & set(route_allowed))
            if not combined_allowed:
                new_trip_needed = True
            elif not can_add_to_trip(
                current_trip['weight'], current_trip['cube'], current_trip['drops'],
                weight, cube, current_trip['bus'], bus[0] if isinstance(bus, list) else bus,
                combined_allowed
            ):
                new_trip_needed = True
        
        if new_trip_needed:
            finalize_trip()
        
        # Update allowed vehicles (intersection)
        if current_trip['allowed_vehicles'] is None:
            current_trip['allowed_vehicles'] = route_allowed.copy()
        else:
            current_trip['allowed_vehicles'] = list(set(current_trip['allowed_vehicles']) & set(route_allowed))
        
        # Add to current trip
        current_trip['routes'].append(row.to_dict())
        current_trip['region'] = region
        current_trip['weight'] += weight
        current_trip['cube'] += cube
        current_trip['drops'] += 1
        if isinstance(bus, list):
            current_trip['bus'].extend(bus)
        else:
            current_trip['bus'].append(bus)
    
    finalize_trip()
    
    print(f"   Total trips created: {len(trips)}")
    
    # Summary by region
    region_summary = {}
    for t in trips:
        r = t['region']
        region_summary[r] = region_summary.get(r, 0) + 1
    print(f"   By Region: {region_summary}")
    
    return trips


# ============================================================================
# PHASE 5: GENERATE OUTPUT
# ============================================================================

def generate_output(trips: List[Dict], original_df: pd.DataFrame) -> pd.DataFrame:
    """Generate final output DataFrame with clean formatting."""
    print("\n" + "="*70)
    print("📋 PHASE 5: GENERATE OUTPUT")
    print("="*70)
    
    output_rows = []
    
    for trip in trips:
        # Sort routes by distance (far to near) within trip
        sorted_routes = sorted(trip['routes'], key=lambda r: r['Distance_KM'], reverse=True)
        
        for seq, route in enumerate(sorted_routes, 1):
            route_id = route['Route_ID']
            # Get original orders for this Route_ID
            route_orders = original_df[original_df['Route_ID'] == route_id]
            
            for _, order in route_orders.iterrows():
                output_rows.append({
                    'Trip_ID': trip['trip_id'],
                    'Vehicle': trip['vehicle'],
                    'Region': trip['region'],
                    'Sequence': seq,
                    'Route_ID': route_id,
                    'Store_Name': order['Store_Name'],
                    'BU': order['BU'],
                    'Province': order['Province'],
                    'District': order['District'],
                    'Subdistrict': order['Subdistrict'],
                    'Weight': order['Weight'],
                    'Cube': order['Cube'],
                    'Distance_KM': order['Distance_KM'],
                    'V_Limit': order['V_Limit'],
                    'Trip_Type': '🅟 Punthai' if trip['is_punthai'] else '🅼 Mixed'
                })
    
    df_output = pd.DataFrame(output_rows)
    
    # Clean: Remove any remaining NaN
    initial = len(df_output)
    df_output = df_output.dropna()
    removed = initial - len(df_output)
    if removed > 0:
        print(f"   Removed {removed} rows with NaN in final output")
    
    print(f"   ✅ Final output: {len(df_output)} rows")
    
    return df_output


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_results(trips: List[Dict]):
    """Verify all business rules are satisfied."""
    print("\n" + "="*70)
    print("✅ VERIFICATION")
    print("="*70)
    
    all_passed = True
    
    # 1. Region Check
    print("\n1️⃣ Same Region Check:")
    for trip in trips:
        regions = set(r['Region'] for r in trip['routes'])
        if len(regions) > 1:
            print(f"   ❌ Trip {trip['trip_id']}: Multiple regions {regions}")
            all_passed = False
        else:
            print(f"   ✅ Trip {trip['trip_id']}: {trip['region']}")
    
    # 2. Central Region - No 6W Check
    print("\n2️⃣ Central Region Vehicle Check (NO 6W):")
    central_trips = [t for t in trips if t['region'] == 'CENTRAL']
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
            print(f"   ❌ Trip {trip['trip_id']} ({v}): EXCEEDS LIMITS")
            all_passed = False
    
    # 4. Punthai Drop Limits
    print("\n4️⃣ Punthai Drop Limit Check:")
    punthai_trips = [t for t in trips if t['is_punthai']]
    if not punthai_trips:
        print("   ℹ️ No Pure Punthai trips")
    else:
        for trip in punthai_trips:
            v = trip['vehicle']
            limit = PUNTHAI_LIMITS[v]['max_drops']
            if trip['drops'] <= limit:
                print(f"   ✅ Trip {trip['trip_id']} ({v} Punthai): {trip['drops']}/{limit} drops")
            else:
                print(f"   ❌ Trip {trip['trip_id']} ({v} Punthai): {trip['drops']}/{limit} drops - OVER!")
                all_passed = False
    
    if all_passed:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
    else:
        print("\n⚠️ SOME VERIFICATIONS FAILED!")
    
    return all_passed


def print_trip_summary(trips: List[Dict]):
    """Print summary of all trips."""
    print("\n" + "="*70)
    print("📊 TRIP SUMMARY")
    print("="*70)
    
    for trip in trips:
        punthai_flag = "🅟" if trip['is_punthai'] else "🅼"
        routes = ", ".join([r['Route_ID'] for r in trip['routes']])
        print(f"Trip {trip['trip_id']:02d} | {trip['vehicle']} | {trip['region']:8s} | "
              f"D:{trip['drops']:2d} | W:{trip['weight']:,}kg | C:{trip['cube']:.1f} | "
              f"{punthai_flag} | Routes: {routes}")
    
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
# MAIN EXECUTION
# ============================================================================

def optimize_routes(orders_df: pd.DataFrame, master_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
    """Main optimization pipeline."""
    print("\n" + "="*70)
    print("🚚 ROUTE OPTIMIZER v3.0 - MASTER DATA MERGE")
    print("="*70)
    print("📋 Features:")
    print("   • Master Data Merge (Left Join)")
    print("   • Hierarchical Sorting (Region > Province > District > Subdistrict)")
    print("   • Central Region Rule (NO 6W)")
    print("   • Punthai Drop Limits (4W=5, JB=7)")
    print("   • NaN Removal")
    
    # Phase 1: Merge
    merged_df = merge_data(orders_df, master_df)
    
    # Phase 2: Hierarchical Sort
    sorted_df = hierarchical_sort(merged_df)
    
    # Phase 3: Consolidate by Route_ID
    consolidated_df = consolidate_routes(sorted_df)
    
    # Phase 4: Trip Allocation
    trips = allocate_trips(consolidated_df)
    
    # Phase 5: Generate Output
    output_df = generate_output(trips, sorted_df)
    
    return output_df, trips


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Create mock data
    print("🔧 Creating Mock Data...")
    master_df = create_master_data()
    orders_df = create_order_data()
    
    print(f"   Master Data: {len(master_df)} locations")
    print(f"   Order Data: {len(orders_df)} orders")
    
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
    display_cols = ['Trip_ID', 'Vehicle', 'Region', 'Sequence', 'Route_ID', 
                    'Store_Name', 'Province', 'District', 'Distance_KM', 'Trip_Type']
    print(output_df[display_cols].to_string(index=False))
    
    # Save to Excel
    output_file = 'route_optimization_v3_result.xlsx'
    output_df.to_excel(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Final notes
    print("\n" + "="*70)
    print("🎯 KEY FEATURES DEMONSTRATED")
    print("="*70)
    print("1. Master Data Merge: Distance_KM & Region from Master")
    print("2. Hierarchical Sort: Region → Province (Max Dist) → District (Max Dist) → Subdistrict")
    print("3. Central Region: Only 4W/JB allowed (NO 6W)")
    print("4. Punthai Trips: 6 drops need JB (4W max 5 drops)")
    print("5. NaN Removal: 'ไม่ระบุ' location removed automatically")
    print("6. Route_ID Grouping: R009 (2 orders) stay together")
