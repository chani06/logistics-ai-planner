# =============================
# Vehicle Logic & Restrictions
# รวม logic ทั้งหมดจากโค้ดเดิม
# =============================

import pandas as pd
import os

# ==========================================
# CONSTANTS
# ==========================================

# ขีดจำกัดรถแต่ละประเภท (มาตรฐาน)
LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5.0, 'max_drops': 12},
    'JB': {'max_w': 3500, 'max_c': 7.0, 'max_drops': 12},
    '6W': {'max_w': 6000, 'max_c': 20.0, 'max_drops': 999}
}

# ขีดจำกัดสำหรับ Punthai ล้วน (ห้ามเกิน 100%)
PUNTHAI_LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5.0, 'max_drops': 5},
    'JB': {'max_w': 3500, 'max_c': 7.0, 'max_drops': 7},
    '6W': {'max_w': 6000, 'max_c': 20.0, 'max_drops': 999}
}

# Buffer
PUNTHAI_BUFFER = 1.0   # 100% - ห้ามเกิน
MAXMART_BUFFER = 1.10  # 110% - เกินได้ 10%
DEFAULT_BUFFER = 1.0

# Central Region List (ห้าม 6W)
CENTRAL_REGIONS = [
    'ภาคกลาง-กรุงเทพชั้นใน',
    'ภาคกลาง-กรุงเทพชั้นกลาง',
    'ภาคกลาง-กรุงเทพชั้นนอก',
    'ภาคกลาง-ปริมณฑล'
]

# ==========================================
# BU DETECTION & BUFFER
# ==========================================

def is_punthai_only(trip_data):
    """
    ตรวจสอบว่าทริปนี้เป็น Punthai ล้วน, Maxmart ล้วน หรือผสม
    
    Returns:
        'punthai_only': ถ้าทั้งหมดเป็น Punthai (BU = 211 หรือชื่อมี PUNTHAI)
        'maxmart_only': ถ้าทั้งหมดเป็น Maxmart (BU = 200 หรือชื่อมี MAXMART)
        'mixed': ถ้ามีทั้ง Punthai และ Maxmart
        'other': ถ้าไม่มีข้อมูล BU
    """
    if trip_data is None or len(trip_data) == 0:
        return 'other'
    
    punthai_count = 0
    maxmart_count = 0
    total_count = len(trip_data)
    
    for _, row in trip_data.iterrows():
        bu = row.get('BU', None)
        name = str(row.get('Name', '')).upper()
        
        # เช็ค Punthai: BU = 211 หรือชื่อมี PUNTHAI
        if bu == 211 or bu == '211' or 'PUNTHAI' in name or 'PUN-' in name:
            punthai_count += 1
        # เช็ค Maxmart: BU = 200 หรือชื่อมี MAXMART/MAX MART
        elif bu == 200 or bu == '200' or 'MAXMART' in name or 'MAX MART' in name:
            maxmart_count += 1
    
    if punthai_count == total_count:
        return 'punthai_only'
    elif maxmart_count == total_count:
        return 'maxmart_only'
    elif punthai_count > 0 or maxmart_count > 0:
        return 'mixed'
    else:
        return 'other'


def get_buffer_for_trip(trip_data):
    """
    ดึง Buffer ที่เหมาะสมตาม BU ของทริป
    
    Rules:
    - Punthai ล้วน: BUFFER = 1.0 (ห้ามเกิน 100%)
    - Maxmart ล้วน/ผสม: BUFFER = 1.10 (เกินได้ 10%)
    
    Returns:
        float: buffer multiplier (1.0 หรือ 1.10)
    """
    trip_type = is_punthai_only(trip_data)
    
    if trip_type == 'punthai_only':
        return PUNTHAI_BUFFER  # 1.0
    elif trip_type in ['maxmart_only', 'mixed']:
        return MAXMART_BUFFER  # 1.10
    else:
        return DEFAULT_BUFFER  # 1.0


def get_punthai_drop_limit(trip_data, vehicle_type):
    """
    ดึงจำกัดจำนวน Drop สำหรับ Punthai ล้วน
    
    Rules:
    - Punthai ล้วน + 4W: สูงสุด 5 สาขา
    - Punthai ล้วน + JB: สูงสุด 7 drop
    - Punthai ล้วน + 6W: ไม่จำกัด
    - อื่นๆ: ตามมาตรฐาน (4W/JB: 12, 6W: 999)
    
    Returns:
        int: max drops allowed
    """
    trip_type = is_punthai_only(trip_data)
    
    if trip_type == 'punthai_only':
        return PUNTHAI_LIMITS.get(vehicle_type, {}).get('max_drops', 999)
    else:
        return LIMITS.get(vehicle_type, {}).get('max_drops', 999)


def get_vehicle_limits(vehicle_type, trip_data=None):
    """
    ดึงขีดจำกัดรถตามประเภทและ BU
    
    Args:
        vehicle_type: '4W', 'JB', '6W'
        trip_data: DataFrame ของทริป (ถ้ามี)
    
    Returns:
        dict: {'max_w': float, 'max_c': float, 'max_drops': int}
    """
    if trip_data is not None and len(trip_data) > 0:
        trip_type = is_punthai_only(trip_data)
        if trip_type == 'punthai_only':
            return PUNTHAI_LIMITS.get(vehicle_type, LIMITS[vehicle_type])
    
    return LIMITS.get(vehicle_type, LIMITS['6W'])

# ==========================================
# VEHICLE RESTRICTIONS
# ==========================================

def load_vehicle_restrictions_from_excel(filepath='Dc/Auto planning (1).xlsx', sheet='Info'):
    """
    โหลดข้อจำกัดรถจากไฟล์ Excel (Auto planning)
    
    Returns:
        dict: {branch_code: [allowed_vehicles]}
    """
    try:
        if not os.path.exists(filepath):
            return {}
        
        df = pd.read_excel(filepath, sheet_name=sheet)
        
        # หา column ที่ตรงกับ LocationNumber และ MaxTruckType
        def find_col(cols, keywords):
            for c in cols:
                c_norm = c.replace(' ', '').lower()
                if all(word.lower() in c_norm for word in keywords):
                    return c
            return None
        
        code_col = find_col(df.columns, ['location', 'code'])
        truck_col = find_col(df.columns, ['truck', 'type'])
        
        if not code_col or not truck_col:
            return {}
        
        restrictions = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            max_truck = str(row[truck_col]).strip().upper()
            
            # Mapping ตามความหมายจริง
            if max_truck == '4W':
                allowed = ['4W']
            elif max_truck in ['JB', '4WJB', 'JB4W']:
                allowed = ['4W', 'JB']
            elif max_truck == '6W':
                allowed = ['4W', 'JB', '6W']
            else:
                allowed = ['4W', 'JB', '6W']  # default
            
            restrictions[code] = allowed
        
        return restrictions
    except Exception as e:
        print(f"⚠️ ไม่สามารถโหลด vehicle restrictions: {e}")
        return {}


def check_branch_vehicle_compatibility(branch_code, vehicle_type, restrictions=None):
    """
    ตรวจสอบว่าสาขานี้ใช้รถประเภทนี้ได้ไหม
    
    Args:
        branch_code: รหัสสาขา
        vehicle_type: '4W', 'JB', '6W'
        restrictions: dict จาก load_vehicle_restrictions_from_excel
    
    Returns:
        bool: True ถ้าใช้ได้, False ถ้าใช้ไม่ได้
    """
    if restrictions is None or len(restrictions) == 0:
        return True  # ไม่มีข้อมูล = ใช้ได้ทุกประเภท
    
    branch_code_str = str(branch_code).strip()
    
    if branch_code_str not in restrictions:
        return True  # ไม่มีข้อมูลสาขานี้ = ใช้ได้
    
    allowed = restrictions[branch_code_str]
    return vehicle_type in allowed


def get_max_vehicle_for_branch(branch_code, restrictions=None):
    """
    ดึงรถใหญ่สุดที่สาขานี้รองรับ
    
    Args:
        branch_code: รหัสสาขา
        restrictions: dict จาก load_vehicle_restrictions_from_excel
    
    Returns:
        str: '4W', 'JB', หรือ '6W'
    """
    if restrictions is None or len(restrictions) == 0:
        return '6W'  # ไม่มีข้อมูล = ใช้รถใหญ่ได้
    
    branch_code_str = str(branch_code).strip()
    
    if branch_code_str not in restrictions:
        return '6W'  # ไม่มีข้อมูลสาขานี้ = ใช้รถใหญ่ได้
    
    allowed = restrictions[branch_code_str]
    
    # เลือกรถใหญ่สุดที่อนุญาต
    if '6W' in allowed:
        return '6W'
    elif 'JB' in allowed:
        return 'JB'
    elif '4W' in allowed:
        return '4W'
    else:
        return '6W'  # fallback


def get_max_vehicle_for_trip(trip_codes, restrictions=None):
    """
    หารถใหญ่สุดที่ทริปนี้ใช้ได้ (เช็คข้อจำกัดของทุกสาขาในทริป)
    
    Args:
        trip_codes: list/set ของ branch codes ในทริป
        restrictions: dict จาก load_vehicle_restrictions_from_excel
    
    Returns:
        str: '4W', 'JB', หรือ '6W'
    """
    vehicle_priority = {'4W': 1, 'JB': 2, '6W': 3}
    max_allowed = '6W'  # เริ่มจากใหญ่สุด
    min_priority = 3
    
    for code in trip_codes:
        branch_max = get_max_vehicle_for_branch(code, restrictions)
        priority = vehicle_priority.get(branch_max, 3)
        
        # เลือกรถที่เล็กที่สุด (ข้อจำกัดมากที่สุด) จากทุกสาขา
        if priority < min_priority:
            min_priority = priority
            max_allowed = branch_max
    
    return max_allowed

# ==========================================
# CENTRAL REGION RULE
# ==========================================

def is_central_region(region):
    """เช็คว่าเป็นภาคกลางหรือไม่ (ห้าม 6W)"""
    if not region:
        return False
    return region in CENTRAL_REGIONS


def filter_vehicles_by_region(allowed_vehicles, region):
    """
    กรองรถตามภูมิภาค
    - ภาคกลาง: ห้าม 6W
    - ภาคอื่น: ใช้ได้ทุกประเภท
    
    Args:
        allowed_vehicles: list ของรถที่อนุญาต
        region: ชื่อภาค
    
    Returns:
        list: รถที่กรองแล้ว
    """
    if is_central_region(region):
        # ภาคกลาง - ห้าม 6W
        return [v for v in allowed_vehicles if v != '6W']
    else:
        return allowed_vehicles

# ==========================================
# VEHICLE SELECTION
# ==========================================

def can_fit_truck(total_weight, total_cube, truck_type, buffer=1.0):
    """
    เช็คว่าน้ำหนัก/คิวใส่รถได้หรือไม่
    
    Args:
        total_weight: น้ำหนักรวม (kg)
        total_cube: คิวรวม (m³)
        truck_type: '4W', 'JB', '6W'
        buffer: buffer multiplier (1.0 หรือ 1.10)
    
    Returns:
        bool: True ถ้าใส่ได้
    """
    limits = LIMITS[truck_type]
    max_w = limits['max_w'] * buffer
    max_c = limits['max_c'] * buffer
    return total_weight <= max_w and total_cube <= max_c


def suggest_truck(total_weight, total_cube, max_allowed='6W', buffer=1.0):
    """
    แนะนำรถที่เหมาะสม โดยเลือกรถที่:
    1. ใส่ของได้พอดี (ไม่เกินขีดจำกัด)
    2. ใช้งานได้ใกล้ 100% มากที่สุด (เป้าหมาย: 90-100%)
    
    Args:
        total_weight: น้ำหนักรวม (kg)
        total_cube: คิวรวม (m³)
        max_allowed: รถใหญ่สุดที่ใช้ได้
        buffer: buffer multiplier
    
    Returns:
        str: '4W', 'JB', หรือ '6W'
    """
    vehicle_sizes = {'4W': 1, 'JB': 2, '6W': 3}
    max_size = vehicle_sizes.get(max_allowed, 3)
    
    # ลองตามลำดับจากเล็กไปใหญ่
    for vehicle in ['4W', 'JB', '6W']:
        if vehicle_sizes[vehicle] > max_size:
            continue
        
        if can_fit_truck(total_weight, total_cube, vehicle, buffer):
            return vehicle
    
    # ถ้าไม่มีรถไหนพอ ให้ใช้รถใหญ่สุด
    return max_allowed


def calculate_utilization(weight, cube, vehicle_type):
    """
    คำนวณ utilization % ของรถ
    
    Returns:
        dict: {'weight_util': float, 'cube_util': float, 'limiting_factor': str}
    """
    limits = LIMITS[vehicle_type]
    weight_util = (weight / limits['max_w']) * 100
    cube_util = (cube / limits['max_c']) * 100
    
    limiting_factor = 'Weight' if weight_util > cube_util else 'Cube'
    
    return {
        'weight_util': weight_util,
        'cube_util': cube_util,
        'limiting_factor': limiting_factor
    }
