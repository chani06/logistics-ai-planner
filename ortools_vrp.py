"""
🤖 Google OR-Tools Optimization for Trip Planning
ใช้ CP-SAT Solver สำหรับ Multi-Dimensional Bin Packing Problem

หลักการ:
- แต่ละสาขา = Item ที่มี weight, cube, drops
- แต่ละทริป = Bin ที่มีขีดจำกัด (ขึ้นกับประเภทรถ)
- เป้าหมาย: ลดจำนวนทริป + รักษามาตรฐานคุณภาพ + เคารพข้อจำกัดภูมิศาสตร์
"""

from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

# Import vehicle logic
try:
    from vehicle_logic import (
        get_buffer_for_trip,
        get_punthai_drop_limit,
        get_max_vehicle_for_branch,
        get_max_vehicle_for_trip,
        check_branch_vehicle_compatibility,
        filter_vehicles_by_region,
        suggest_truck,
        calculate_utilization,
        is_central_region
    )
    VEHICLE_LOGIC_AVAILABLE = True
except ImportError:
    VEHICLE_LOGIC_AVAILABLE = False

# Vehicle Limits (ต้องซิงค์กับ app.py)
LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5.0, 'max_drops': 12},
    'JB': {'max_w': 3500, 'max_c': 7.0, 'max_drops': 12},
    '6W': {'max_w': 6000, 'max_c': 20.0, 'max_drops': 20}
}

PUNTHAI_LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5.0, 'max_drops': 12},
    'JB': {'max_w': 3500, 'max_c': 7.0, 'max_drops': 12},
    '6W': {'max_w': 6000, 'max_c': 20.0, 'max_drops': 20}
}

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculate bearing (0-360°) from point 1 to point 2"""
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    
    initial_bearing = math.atan2(x, y)
    bearing = (math.degrees(initial_bearing) + 360) % 360
    return bearing


def is_opposite_direction(bearing1, bearing2, threshold=100):
    """Check if two bearings are opposite (>threshold degrees apart)"""
    diff = abs(bearing1 - bearing2)
    if diff > 180:
        diff = 360 - diff
    return diff >= threshold


class TripOptimizer:
    """OR-Tools CP-SAT based trip optimizer"""
    
    def __init__(self, df, buffer_punthai=1.0, buffer_maxmart=1.10, 
                 dc_lat=14.2378, dc_lon=100.7319,
                 master_data=None, global_limiting_factor='weight'):
        """
        Initialize optimizer
        
        Args:
            df: DataFrame with columns [Code, Weight, Cube, Route, จังหวัด, อำเภอ, ตำบล, ละติจูด, ลองจิจูด]
            buffer_punthai: Capacity multiplier for Punthai branches
            buffer_maxmart: Capacity multiplier for Maxmart branches
            dc_lat, dc_lon: DC location coordinates
            master_data: Master data DataFrame for coordinate lookup
            global_limiting_factor: 'weight' or 'cube' - which metric to prioritize
        """
        self.df = df.copy()
        self.buffer_punthai = buffer_punthai
        self.buffer_maxmart = buffer_maxmart
        self.dc_lat = dc_lat
        self.dc_lon = dc_lon
        self.master_data = master_data
        self.global_limiting_factor = global_limiting_factor
        
        # Prepare branch data
        self._prepare_branches()
        
    def _prepare_branches(self):
        """Prepare branch data with coordinates and metadata"""
        self.branches = []
        
        for idx, row in self.df.iterrows():
            code = str(row.get('Code', '')).strip().upper()
            weight = float(row.get('Weight', 0))
            cube = float(row.get('Cube', 0))
            drops = int(row.get('Drops', 1))
            
            # Determine if Punthai
            branch_name = str(row.get('สาขา', '')).upper()
            is_punthai = 'PUNTHAI' in branch_name and 'MAXMART' not in branch_name
            
            # Get coordinates
            lat = row.get('ละติจูด', None)
            lon = row.get('ลองจิจูด', None)
            
            if pd.isna(lat) or pd.isna(lon):
                # Try to get from master data
                if self.master_data is not None and 'Plan Code' in self.master_data.columns:
                    match = self.master_data[self.master_data['Plan Code'] == code]
                    if not match.empty:
                        lat = match.iloc[0].get('ละติจูด', None)
                        lon = match.iloc[0].get('ลองจิจูด', None)
            
            # Calculate distance and bearing from DC
            distance_from_dc = 0
            bearing_from_dc = 0
            if lat and lon and not pd.isna(lat) and not pd.isna(lon):
                distance_from_dc = haversine(self.dc_lat, self.dc_lon, float(lat), float(lon))
                bearing_from_dc = calculate_bearing(self.dc_lat, self.dc_lon, float(lat), float(lon))
            
            # Geographic info
            province = str(row.get('จังหวัด', '')).strip()
            district = str(row.get('อำเภอ', '')).strip()
            subdistrict = str(row.get('ตำบล', '')).strip()
            
            self.branches.append({
                'idx': idx,
                'code': code,
                'weight': weight,
                'cube': cube,
                'drops': drops,
                'is_punthai': is_punthai,
                'lat': lat if not pd.isna(lat) else None,
                'lon': lon if not pd.isna(lon) else None,
                'distance_from_dc': distance_from_dc,
                'bearing_from_dc': bearing_from_dc,
                'province': province,
                'district': district,
                'subdistrict': subdistrict,
                'row': row
            })
        
        print(f"📦 Prepared {len(self.branches)} branches for optimization")
    
    def _get_max_vehicle(self, code):
        """Determine maximum allowed vehicle for a branch"""
        # Simplified - can be enhanced with actual branch_vehicles data
        return '6W'
    
    def _get_vehicle_limits(self, vehicle_type, is_punthai):
        """Get capacity limits for vehicle type"""
        buffer = self.buffer_punthai if is_punthai else self.buffer_maxmart
        limits_dict = PUNTHAI_LIMITS if is_punthai else LIMITS
        
        if vehicle_type not in limits_dict:
            return None
        
        lim = limits_dict[vehicle_type]
        return {
            'max_w': int(lim['max_w'] * buffer),
            'max_c': int(lim['max_c'] * buffer * 100),  # Convert to integer (×100 for precision)
            'max_drops': lim.get('max_drops', 12)
        }
    
    def optimize(self, max_trips=100, time_limit_seconds=120):
        """
        Run OR-Tools optimization
        
        Args:
            max_trips: Maximum number of trips to consider
            time_limit_seconds: Solver time limit
        
        Returns:
            (result_df, summary_dict)
        """
        print(f"🤖 Starting OR-Tools optimization...")
        print(f"   Global Limiting Factor: {self.global_limiting_factor.upper()}")
        print(f"   Branches: {len(self.branches)}")
        print(f"   Max Trips: {max_trips}")
        print(f"   Time Limit: {time_limit_seconds}s")
        
        # Create model
        model = cp_model.CpModel()
        
        n_branches = len(self.branches)
        n_trips = max_trips
        
        # Decision variables: branch_in_trip[i, t] = 1 if branch i is in trip t
        branch_in_trip = {}
        for i in range(n_branches):
            for t in range(n_trips):
                branch_in_trip[(i, t)] = model.NewBoolVar(f'branch_{i}_in_trip_{t}')
        
        # Trip active: trip_active[t] = 1 if trip t is used
        trip_active = {}
        for t in range(n_trips):
            trip_active[t] = model.NewBoolVar(f'trip_{t}_active')
        
        # Vehicle type for each trip: vehicle_type[t] in {0:4W, 1:JB, 2:6W}
        vehicle_types = ['4W', 'JB', '6W']
        trip_vehicle = {}
        for t in range(n_trips):
            trip_vehicle[t] = model.NewIntVar(0, 2, f'trip_{t}_vehicle')
        
        # 🔥 Constraint 1: Each branch assigned to exactly one trip
        for i in range(n_branches):
            model.Add(sum(branch_in_trip[(i, t)] for t in range(n_trips)) == 1)
        
        # 🔥 Constraint 2: Trip is active if any branch assigned to it
        for t in range(n_trips):
            model.Add(sum(branch_in_trip[(i, t)] for i in range(n_branches)) >= 1).OnlyEnforceIf(trip_active[t])
            model.Add(sum(branch_in_trip[(i, t)] for i in range(n_branches)) == 0).OnlyEnforceIf(trip_active[t].Not())
        
        # 🔥 Constraint 3: Capacity constraints for each trip
        # We need to handle multiple vehicle types, so we'll use conditional constraints
        for t in range(n_trips):
            # Calculate total weight, cube, drops for this trip
            total_weight = sum(int(self.branches[i]['weight']) * branch_in_trip[(i, t)] 
                             for i in range(n_branches))
            total_cube = sum(int(self.branches[i]['cube'] * 100) * branch_in_trip[(i, t)] 
                           for i in range(n_branches))
            total_drops = sum(self.branches[i]['drops'] * branch_in_trip[(i, t)] 
                            for i in range(n_branches))
            
            # For each vehicle type, check if capacity is respected
            for v_idx, v_type in enumerate(vehicle_types):
                # Assume mixed punthai (use maxmart buffer as more conservative)
                limits = self._get_vehicle_limits(v_type, False)
                
                # If this trip uses this vehicle type, enforce limits
                is_this_vehicle = model.NewBoolVar(f'trip_{t}_is_{v_type}')
                model.Add(trip_vehicle[t] == v_idx).OnlyEnforceIf(is_this_vehicle)
                model.Add(trip_vehicle[t] != v_idx).OnlyEnforceIf(is_this_vehicle.Not())
                
                # Enforce capacity only if this vehicle type is selected
                model.Add(total_weight <= limits['max_w']).OnlyEnforceIf([trip_active[t], is_this_vehicle])
                model.Add(total_cube <= limits['max_c']).OnlyEnforceIf([trip_active[t], is_this_vehicle])
                model.Add(total_drops <= limits['max_drops']).OnlyEnforceIf([trip_active[t], is_this_vehicle])
        
        # 🔥 Constraint 4: Province clustering (try to keep same province in same trip)
        # Group branches by province
        province_groups = {}
        for i, branch in enumerate(self.branches):
            prov = branch['province']
            if prov not in province_groups:
                province_groups[prov] = []
            province_groups[prov].append(i)
        
        # Soft constraint: branches in same province should be in nearby trips
        # (This is simplified - full implementation would use distance-based penalties)
        
        # 🔥 Constraint 5: Minimum standard (70% of limiting factor)
        for t in range(n_trips):
            total_weight = sum(int(self.branches[i]['weight']) * branch_in_trip[(i, t)] 
                             for i in range(n_branches))
            total_cube = sum(int(self.branches[i]['cube'] * 100) * branch_in_trip[(i, t)] 
                           for i in range(n_branches))
            
            # Get minimum capacity (smallest vehicle = 4W)
            limits_4w = self._get_vehicle_limits('4W', False)
            min_threshold_weight = int(limits_4w['max_w'] * 0.7)
            min_threshold_cube = int(limits_4w['max_c'] * 0.7)
            
            # Enforce minimum based on global limiting factor
            if self.global_limiting_factor == 'weight':
                model.Add(total_weight >= min_threshold_weight).OnlyEnforceIf(trip_active[t])
            else:  # cube
                model.Add(total_cube >= min_threshold_cube).OnlyEnforceIf(trip_active[t])
        
        # 🎯 Objective: Minimize number of trips (primary) + maximize utilization (secondary)
        # We use weighted sum: heavily penalize active trips, slightly reward high utilization
        trip_penalty = 10000  # Heavy penalty for each trip
        
        # Calculate utilization bonus for each trip
        utilization_bonus = []
        for t in range(n_trips):
            total_weight = sum(int(self.branches[i]['weight']) * branch_in_trip[(i, t)] 
                             for i in range(n_branches))
            utilization_bonus.append(total_weight)  # Simple proxy for utilization
        
        # Objective = minimize (trip_count * penalty - total_utilization)
        model.Minimize(
            trip_penalty * sum(trip_active[t] for t in range(n_trips)) - 
            sum(utilization_bonus[t] for t in range(n_trips))
        )
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit_seconds
        solver.parameters.log_search_progress = False  # ปิด log เพื่อเร็วขึ้น
        
        # ⚡ ปรับพารามิเตอร์เพื่อความเร็ว
        solver.parameters.num_search_workers = 8  # ใช้ CPU หลาย core
        solver.parameters.cp_model_presolve = True
        solver.parameters.linearization_level = 2
        solver.parameters.cp_model_probing_level = 0  # ลด probing เพื่อความเร็ว
        
        import time
        start_time = time.time()
        
        print(f"⚙️ Solving with CP-SAT...")
        status = solver.Solve(model)
        
        elapsed_time = time.time() - start_time
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print(f"✅ Solution found: {solver.StatusName(status)} (⏱️ {elapsed_time:.1f}s)")
            return self._extract_solution(solver, branch_in_trip, trip_active, trip_vehicle, n_trips, elapsed_time)
        else:
            print(f"❌ No solution found: {solver.StatusName(status)}")
            return None, {}
    
    def _extract_solution(self, solver, branch_in_trip, trip_active, trip_vehicle, n_trips, elapsed_time=0):
        """Extract solution from solver"""
        vehicle_types = ['4W', 'JB', '6W']
        
        # Group branches by trip
        trips = {}
        for t in range(n_trips):
            if solver.Value(trip_active[t]) == 1:
                trip_branches = []
                for i in range(len(self.branches)):
                    if solver.Value(branch_in_trip[(i, t)]) == 1:
                        trip_branches.append(i)
                
                if trip_branches:
                    vehicle_idx = solver.Value(trip_vehicle[t])
                    trips[t] = {
                        'branches': trip_branches,
                        'vehicle': vehicle_types[vehicle_idx]
                    }
        
        print(f"📊 Solution: {len(trips)} trips (⏱️ {elapsed_time:.1f}s)")
        
        # Create result DataFrame
        result_df = self.df.copy()
        result_df['Trip'] = 0
        result_df['Vehicle'] = ''
        
        for trip_num, (t, trip_data) in enumerate(trips.items(), 1):
            for branch_idx in trip_data['branches']:
                df_idx = self.branches[branch_idx]['idx']
                result_df.loc[df_idx, 'Trip'] = trip_num
                result_df.loc[df_idx, 'Vehicle'] = trip_data['vehicle']
        
        # Calculate summary
        summary = self._calculate_summary(result_df, trips)
        
        return result_df, summary
    
    def _calculate_summary(self, result_df, trips):
        """Calculate trip summary statistics"""
        summary = {}
        
        # Detect province column name (support both Thai and English)
        province_col = None
        for col in ['Province', 'จังหวัด']:
            if col in result_df.columns:
                province_col = col
                break
        
        for trip_num, (t, trip_data) in enumerate(trips.items(), 1):
            trip_df = result_df[result_df['Trip'] == trip_num]
            
            total_weight = trip_df['Weight'].sum()
            total_cube = trip_df['Cube'].sum()
            total_drops = len(trip_df)
            vehicle = trip_data['vehicle']
            
            # Get limits
            limits = self._get_vehicle_limits(vehicle, False)
            
            provinces_count = 0
            if province_col:
                provinces_count = trip_df[province_col].nunique()
            
            summary[trip_num] = {
                'Vehicle': vehicle,
                'Weight': total_weight,
                'Cube': total_cube,
                'Drops': total_drops,
                'Weight%': (total_weight / limits['max_w']) * 100 if limits else 0,
                'Cube%': (total_cube / (limits['max_c']/100)) * 100 if limits else 0,
                'Branches': len(trip_df),
                'Provinces': provinces_count
            }
        
        return summary


def predict_trips_ortools(test_df, buffer_punthai=1.0, buffer_maxmart=1.10, 
                          dc_lat=14.2378, dc_lon=100.7319,
                          master_data=None, max_trips=80, time_limit=50,
                          restrictions=None):
    """
    Main entry point for OR-Tools optimization
    
    Args:
        test_df: Input DataFrame
        buffer_punthai: Buffer for Punthai branches
        buffer_maxmart: Buffer for Maxmart branches
        dc_lat, dc_lon: DC coordinates
        master_data: Master data for coordinate lookup
        max_trips: Maximum number of trips (default: 80)
        time_limit: Solver time limit in seconds (default: 50)
        restrictions: Vehicle restrictions dict from vehicle_logic
    
    Returns:
        (result_df, summary_dict)
    """
    # Determine global limiting factor
    global_limiting_factor = determine_global_limiting_factor(test_df)
    
    # Create optimizer
    optimizer = TripOptimizer(
        test_df, 
        buffer_punthai=buffer_punthai,
        buffer_maxmart=buffer_maxmart,
        dc_lat=dc_lat,
        dc_lon=dc_lon,
        master_data=master_data,
        global_limiting_factor=global_limiting_factor
    )
    
    # Run optimization
    result_df, summary = optimizer.optimize(
        max_trips=max_trips,
        time_limit_seconds=time_limit
    )
    
    return result_df, summary


def determine_global_limiting_factor(df):
    """
    Analyze data to determine if weight or cube is the primary constraint
    (Same logic as in app.py)
    """
    if df.empty:
        return 'weight'
    
    # Simplified analysis (use weight if avg weight > avg cube ratio-wise)
    total_weight = df['Weight'].sum()
    total_cube = df['Cube'].sum()
    
    # Compare ratios against 4W vehicle (baseline)
    limits_4w = LIMITS['4W']
    weight_ratio = (total_weight / df.shape[0]) / limits_4w['max_w'] if df.shape[0] > 0 else 0
    cube_ratio = (total_cube / df.shape[0]) / limits_4w['max_c'] if df.shape[0] > 0 else 0
    
    if weight_ratio >= cube_ratio:
        print(f"📊 Global Limiting Factor: WEIGHT (ratio {weight_ratio:.2f} vs cube {cube_ratio:.2f})")
        return 'weight'
    else:
        print(f"📊 Global Limiting Factor: CUBE (ratio {cube_ratio:.2f} vs weight {weight_ratio:.2f})")
        return 'cube'


if __name__ == "__main__":
    print("🤖 OR-Tools Trip Optimizer Module")
    print("   Import this module to use: from ortools_vrp import predict_trips_ortools")
