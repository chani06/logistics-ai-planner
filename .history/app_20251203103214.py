"""
Logistics Planner 
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import glob
from datetime import datetime
import io

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = 'models/decision_tree_model.pkl'

# ขีดจำกัดรถแต่ละประเภท
LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5.0},
    '6W': {'max_w': 5800, 'max_c': 22.0},
    'JB': {'max_w': 3500, 'max_c': 8.0}
}

# เผื่อการใช้รถได้เกิน 5%
BUFFER = 1.05

# จำกัดจำนวนสาขาต่อทริป
MAX_BRANCHES_PER_TRIP = 12  # สูงสุด 12 สาขาต่อทริป
TARGET_BRANCHES_PER_TRIP = 10  # เป้าหมาย 10 สาขาต่อทริป

# รายการสาขาที่ไม่ต้องการจัดส่ง (ตัดออก)
EXCLUDE_BRANCHES = ['DC011', 'PTDC', 'PTG DISTRIBUTION CENTER']

# รายชื่อที่ต้องตัดออก (ใช้ตรวจสอบชื่อ)
EXCLUDE_NAMES = ['Distribution Center', 'PTG Distribution', 'บ.พีทีจี เอ็นเนอยี']

# พิกัด DC วังน้อย (จุดกลาง)
DC_WANG_NOI_LAT = 14.179394
DC_WANG_NOI_LON = 100.648149

# ระยะทางที่ต้องใช้รถ 6W (กม.)
DISTANCE_REQUIRE_6W = 100  # ถ้าห่างจาก DC เกิน 100 กม. ต้องใช้ 6W

# ==========================================
# LOAD MASTER DATA
# ==========================================
@st.cache_data
def load_master_data():
    """โหลดไฟล์ Master สถานที่ส่ง"""
    try:
        df_master = pd.read_excel('Dc/Master สถานที่ส่ง.xlsx')
        # ทำความสะอาด Plan Code
        if 'Plan Code' in df_master.columns:
            df_master['Plan Code'] = df_master['Plan Code'].apply(lambda x: str(x).strip().upper() if pd.notna(x) else '')
        return df_master
    except FileNotFoundError:
        # ไม่มีไฟล์ Master - ใช้งานได้ปกติแต่ไม่มีข้อมูลตำบล/อำเภอ
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"ไม่สามารถโหลดไฟล์ Master: {e} (จะใช้ข้อมูลจากไฟล์อัปโหลดแทน)")
        return pd.DataFrame()

# โหลด Master Data
MASTER_DATA = load_master_data()

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def normalize(val):
    """ทำให้รหัสสาขาเป็นมาตรฐาน"""
    return str(val).strip().upper().replace(" ", "").replace(".0", "")

def calculate_distance_from_dc(lat, lon):
    """คำนวณระยะทางจาก DC วังน้อย (กม.)"""
    if lat == 0 or lon == 0:
        return 0
    import math
    lat1, lon1 = math.radians(DC_WANG_NOI_LAT), math.radians(DC_WANG_NOI_LON)
    lat2, lon2 = math.radians(lat), math.radians(lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return 6371 * c

def get_required_vehicle_by_distance(branch_code):
    """ตรวจสอบว่าสาขาต้องใช้รถอะไรตามระยะทางจาก DC"""
    # ดึงพิกัดจาก Master
    if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
        master_row = MASTER_DATA[MASTER_DATA['Plan Code'] == branch_code]
        if len(master_row) > 0:
            lat = master_row.iloc[0].get('ละติจูด', 0)
            lon = master_row.iloc[0].get('ลองติจูด', 0)
            distance = calculate_distance_from_dc(lat, lon)
            
            # ถ้าห่างจาก DC เกินกำหนด → ต้องใช้ 6W
            if distance > DISTANCE_REQUIRE_6W:
                return '6W', distance
    
    return None, 0

def can_fit_truck(total_weight, total_cube, truck_type):
    """เช็คว่าน้ำหนัก/คิวใส่รถได้หรือไม่"""
    limits = LIMITS[truck_type]
    max_w = limits['max_w'] * BUFFER
    max_c = limits['max_c'] * BUFFER
    return total_weight <= max_w and total_cube <= max_c

def suggest_truck(total_weight, total_cube, max_allowed='6W', trip_codes=None):
    """
    แนะนำรถที่เหมาะสม โดยเลือกรถที่:
    1. เช็คระยะทางจาก DC - ถ้าไกล → บังคับ 6W
    2. ใส่ของได้พอดี (ไม่เกินขีดจำกัด)
    3. ใช้งานได้มากที่สุด (ใกล้ 100% ที่สุด)
    """
    # เช็คว่ามีสาขาที่ต้องใช้ 6W เพราะอยู่ไกลหรือไม่
    if trip_codes:
        for code in trip_codes:
            required_vehicle, distance = get_required_vehicle_by_distance(code)
            if required_vehicle == '6W':
                return '6W'  # บังคับ 6W เพราะมีสาขาที่อยู่ไกล
    vehicle_sizes = {'4W': 1, 'JB': 2, '6W': 3}
    max_size = vehicle_sizes.get(max_allowed, 3)
    
    best_truck = None
    best_utilization = 0
    
    for truck in ['4W', 'JB', '6W']:
        truck_size = vehicle_sizes.get(truck, 0)
        # ถ้ารถใหญ่กว่าที่อนุญาต ข้ามไป
        if truck_size > max_size:
            continue
        if can_fit_truck(total_weight, total_cube, truck):
            # คำนวณ % การใช้รถ
            limits = LIMITS[truck]
            w_util = (total_weight / limits['max_w']) * 100
            c_util = (total_cube / limits['max_c']) * 100
            utilization = max(w_util, c_util)
            
            # เลือกรถที่ใช้งานได้มากที่สุด (ใกล้ 100%)
            if utilization > best_utilization:
                best_utilization = utilization
                best_truck = truck
    
    if best_truck:
        return best_truck
    
    # ถ้าไม่มีรถที่เหมาะสม ใช้รถใหญ่สุดที่อนุญาต
    return max_allowed if max_allowed in LIMITS else '6W+'

def can_branch_use_vehicle(code, vehicle_type, branch_vehicles):
    """
    เช็คว่าสาขาสามารถใช้รถประเภทนี้ได้หรือไม่
    - ถ้าไม่มีประวัติ = ใช้ได้ทุกประเภท
    - ถ้ามีประวัติใช้รถใหญ่ = ใช้รถเล็กกว่าได้
    - ถ้ามีประวัติใช้แค่รถเล็ก (เช่น 4W) = ใช้รถใหญ่ไม่ได้ (รถใหญ่เข้าไม่ได้)
    """
    if not branch_vehicles or code not in branch_vehicles:
        return True  # ไม่มีประวัติ = ใช้ได้ทุกประเภท
    
    vehicle_history = branch_vehicles[code]
    if not vehicle_history:
        return True  # ไม่มีข้อมูลรถ = ใช้ได้ทุกประเภท
    
    # ถ้าเคยใช้รถประเภทนี้ = ใช้ได้
    if vehicle_type in vehicle_history:
        return True
    
    # เช็คขนาดรถ (6W > JB > 4W)
    vehicle_sizes = {'4W': 1, 'JB': 2, '6W': 3}
    requested_size = vehicle_sizes.get(vehicle_type, 0)
    
    # หารถที่ใหญ่ที่สุดที่สาขาเคยใช้
    max_used_size = max(vehicle_sizes.get(v, 0) for v in vehicle_history)
    
    # ถ้าขอใช้รถเล็กกว่าหรือเท่ากับที่เคยใช้ = ใช้ได้
    # ถ้าขอใช้รถใหญ่กว่าที่เคยใช้ = ใช้ไม่ได้ (รถใหญ่อาจเข้าไม่ได้)
    return requested_size <= max_used_size

def get_max_vehicle_for_branch(code, branch_vehicles):
    """ดึงประเภทรถที่ใหญ่ที่สุดที่สาขาเคยใช้ (จำกัดไม่ให้ใช้รถใหญ่กว่านี้)"""
    if not branch_vehicles or code not in branch_vehicles:
        return '6W'  # ไม่มีประวัติ = ใช้ได้ถึง 6W
    
    vehicle_history = branch_vehicles[code]
    if not vehicle_history:
        return '6W'
    
    vehicle_sizes = {'4W': 1, 'JB': 2, '6W': 3}
    max_vehicle = max(vehicle_history.keys(), key=lambda v: vehicle_sizes.get(v, 0))
    return max_vehicle

def get_most_used_vehicle_for_branch(code, branch_vehicles):
    """ดึงประเภทรถที่สาขาใช้บ่อยที่สุด"""
    if not branch_vehicles or code not in branch_vehicles:
        return None
    
    vehicle_history = branch_vehicles[code]
    if not vehicle_history:
        return None
    
    return max(vehicle_history, key=vehicle_history.get)

def is_similar_name(name1, name2):
    """เช็คว่าชื่อสาขาคล้ายกันหรือไม่ - รองรับทั้งไทยและอังกฤษ"""
    def clean_name(name):
        if pd.isna(name) or name is None:
            return "", ""
        s = str(name).strip().upper()
        
        # ลบ prefix/suffix ที่พบบ่อย
        prefixes = ['PTC-MRT-', 'PTC-', 'PUN-', 'MAXMART', 'FUTURE', 'ฟิวเจอร์', 'CW', 'FC']
        for prefix in prefixes:
            s = s.replace(prefix, '')
        
        # แยกภาษาไทยและอังกฤษ
        thai_chars = ''.join([c for c in s if '\u0e01' <= c <= '\u0e5b'])
        eng_chars = ''.join([c for c in s if c.isalpha() and c.isascii()])
        
        return thai_chars, eng_chars
    
    thai1, eng1 = clean_name(name1)
    thai2, eng2 = clean_name(name2)
    
    # ต้องมีความยาวพอสมควร
    if len(thai1) < 3 and len(eng1) < 3:
        return False
    if len(thai2) < 3 and len(eng2) < 3:
        return False
    
    # เช็คภาษาไทย
    if thai1 and thai2:
        shorter_thai = min(thai1, thai2, key=len)
        longer_thai = max(thai1, thai2, key=len)
        if len(shorter_thai) >= 3 and shorter_thai in longer_thai:
            return True
        # ความคล้าย 80%+
        if len(shorter_thai) >= 5:
            common = sum(1 for c in shorter_thai if c in longer_thai)
            if common / len(shorter_thai) >= 0.8:
                return True
    
    # เช็คภาษาอังกฤษ
    if eng1 and eng2:
        shorter_eng = min(eng1, eng2, key=len)
        longer_eng = max(eng1, eng2, key=len)
        if len(shorter_eng) >= 3 and shorter_eng in longer_eng:
            return True
        # ความคล้าย 80%+
        if len(shorter_eng) >= 5:
            common = sum(1 for c in shorter_eng if c in longer_eng)
            if common / len(shorter_eng) >= 0.8:
                return True
    
    # เช็คคำสำคัญระหว่างไทย-อังกฤษ (เช่น Future = ฟิวเจอร์, Rangsit = รังสิต)
    thai_eng_map = {
        'RANGSIT': 'รังสิต',
        'FUTURE': 'ฟิวเจอร',
        'PARK': 'ปารค',
        'TRIANGLE': 'ไตรแองเกิล',
    }
    
    for eng_word, thai_word in thai_eng_map.items():
        # ตรวจสอบว่ามีคำนี้ในชื่อทั้งสองฝั่ง (ไทย-อังกฤษ หรือ อังกฤษ-อังกฤษ หรือ ไทย-ไทย)
        has_eng_in_1 = eng_word in eng1
        has_eng_in_2 = eng_word in eng2
        has_thai_in_1 = thai_word in thai1
        has_thai_in_2 = thai_word in thai2
        
        # ถ้าทั้งสองมีคำเดียวกัน (ไม่ว่าจะไทยหรืออังกฤษ) = คล้ายกัน
        if (has_eng_in_1 and has_eng_in_2) or (has_thai_in_1 and has_thai_in_2):
            return True
        # ถ้าข้ามภาษา (อังกฤษ-ไทย)
        if (has_eng_in_1 and has_thai_in_2) or (has_eng_in_2 and has_thai_in_1):
            return True
    
    return False

def is_nearby_province(prov1, prov2):
    """เช็คว่าจังหวัดใกล้เคียงกันหรือไม่ (จากไฟล์ประวัติ)"""
    if pd.isna(prov1) or pd.isna(prov2):
        return False
    
    if prov1 == prov2:
        return True
    
    # จัดกลุ่มจังหวัดตามภาคย่อย (จากไฟล์ประวัติ)
    province_groups = {
        'กรุงเทพ': ['กรุงเทพมหานคร', 'กรุงเทพ'],
        'ปริมณฑล': ['นครปฐม', 'นนทบุรี', 'ปทุมธานี', 'สมุทรปราการ', 'สมุทรสาคร', 'ฉะเชิงเทรา'],
        'กลางตอนบน': ['ชัยนาท', 'พระนครศรีอยุธยา', 'ลพบุรี', 'สระบุรี', 'สิงห์บุรี', 'อ่างทอง', 'อยุธยา'],
        'กลางตอนล่าง': ['สมุทรสงคราม', 'สุพรรณบุรี'],
        'ภาคตะวันตก': ['กาญจนบุรี', 'ประจวบคีรีขันธ์', 'ราชบุรี', 'เพชรบุรี'],
        'ภาคตะวันออก': ['จันทบุรี', 'ชลบุรี', 'ตราด', 'นครนายก', 'ปราจีนบุรี', 'ระยอง', 'สระแก้ว'],
        'อีสานเหนือ': ['นครพนม', 'บึงกาฬ', 'มุกดาหาร', 'สกลนคร', 'หนองคาย', 'หนองบัวลำภู', 'อุดรธานี', 'เลย'],
        'อีสานกลาง': ['กาฬสินธุ์', 'ขอนแก่น', 'ชัยภูมิ', 'มหาสารคาม', 'ร้อยเอ็ด'],
        'อีสานใต้': ['นครราชสีมา', 'โคราช', 'บุรีรัมย์', 'ยโสธร', 'ศรีสะเกษ', 'สุรินทร์', 'อำนาจเจริญ', 'อุบลราชธานี'],
        'เหนือตอนบน': ['น่าน', 'พะเยา', 'ลำปาง', 'ลำพูน', 'เชียงราย', 'เชียงใหม่', 'แพร่', 'แม่ฮ่องสอน'],
        'เหนือตอนล่าง': ['กำแพงเพชร', 'ตาก', 'นครสวรรค์', 'พิจิตร', 'พิษณุโลก', 'สุโขทัย', 'อุตรดิตถ์', 'อุทัยธานี', 'เพชรบูรณ์'],
        'ใต้ฝั่งอันดามัน': ['กระบี่', 'ตรัง', 'พังงา', 'ภูเก็ต', 'ระนอง', 'สตูล'],
        'ใต้ฝั่งอ่าวไทย': ['ชุมพร', 'นครศรีธรรมราช', 'พัทลุง', 'ยะลา', 'สงขลา', 'สุราษฎร์ธานี', 'ปัตตานี', 'นราธิวาส']
    }
    
    # หาว่าจังหวัดทั้ง 2 อยู่กลุ่มเดียวกันหรือไม่
    for group, provinces in province_groups.items():
        in_group_1 = any(p in str(prov1) for p in provinces)
        in_group_2 = any(p in str(prov2) for p in provinces)
        
        if in_group_1 and in_group_2:
            return True
    
    return False

def load_model():
    """โหลดโมเดลที่เทรนไว้"""
    if not os.path.exists(MODEL_PATH):
        return None
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def create_pair_features(code1, code2, branch_info):
    """สร้าง features สำหรับคู่สาขา"""
    import math
    
    info1 = branch_info[code1]
    info2 = branch_info[code2]
    
    # คำนวณความต่างของน้ำหนักและคิว
    weight_diff = abs(info1['avg_weight'] - info2['avg_weight'])
    cube_diff = abs(info1['avg_cube'] - info2['avg_cube'])
    weight_sum = info1['avg_weight'] + info2['avg_weight']
    cube_sum = info1['avg_cube'] + info2['avg_cube']
    
    # จังหวัดเดียวกันหรือไม่
    same_province = 1 if info1['province'] == info2['province'] else 0
    
    # คำนวณระยะทางจากพิกัด
    distance_km = 0.0
    if info1['latitude'] != 0 and info2['latitude'] != 0:
        lat1, lon1 = math.radians(info1['latitude']), math.radians(info1['longitude'])
        lat2, lon2 = math.radians(info2['latitude']), math.radians(info2['longitude'])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance_km = 6371 * c
    
    # ความถี่
    freq_product = info1['total_trips'] * info2['total_trips']
    freq_diff = abs(info1['total_trips'] - info2['total_trips'])
    
    # Ratio
    weight_ratio = (info1['avg_weight'] / info2['avg_weight']) if info2['avg_weight'] > 0 else 0
    cube_ratio = (info1['avg_cube'] / info2['avg_cube']) if info2['avg_cube'] > 0 else 0
    
    # ข้อจำกัดรถ
    over_4w = 1 if (weight_sum > 2500 or cube_sum > 5.0) else 0
    over_jb = 1 if (weight_sum > 3500 or cube_sum > 8.0) else 0
    over_6w = 1 if (weight_sum > 5800 or cube_sum > 22.0) else 0
    
    return {
        'weight_sum': weight_sum,
        'cube_sum': cube_sum,
        'weight_diff': weight_diff,
        'cube_diff': cube_diff,
        'same_province': same_province,
        'distance_km': distance_km,
        'avg_weight_1': info1['avg_weight'],
        'avg_weight_2': info2['avg_weight'],
        'avg_cube_1': info1['avg_cube'],
        'avg_cube_2': info2['avg_cube'],
        'freq_product': freq_product,
        'freq_diff': freq_diff,
        'weight_ratio': weight_ratio,
        'cube_ratio': cube_ratio,
        'over_4w': over_4w,
        'over_jb': over_jb,
        'over_6w': over_6w
    }

def load_excel(file_content, sheet_name=None):
    """โหลด Excel"""
    try:
        xls = pd.ExcelFile(io.BytesIO(file_content))
        
        target_sheet = None
        if sheet_name and sheet_name in xls.sheet_names:
            target_sheet = sheet_name
        else:
            for s in xls.sheet_names:
                if 'punthai' in s.lower() or '2.' in s.lower():
                    target_sheet = s
                    break
        
        if not target_sheet:
            target_sheet = xls.sheet_names[0]
        
        # หา header row
        df_temp = pd.read_excel(xls, sheet_name=target_sheet, header=None)
        header_row = 0
        
        for i in range(min(10, len(df_temp))):
            row_values = df_temp.iloc[i].astype(str).str.upper()
            match_count = sum([
                'BRANCH' in ' '.join(row_values),
                'TRIP' in ' '.join(row_values),
                'รหัสสาขา' in ' '.join(df_temp.iloc[i].astype(str))
            ])
            if match_count >= 2:
                header_row = i
                break
        
        df = pd.read_excel(xls, sheet_name=target_sheet, header=header_row)
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def process_dataframe(df):
    """แปลงคอลัมน์เป็นรูปแบบมาตรฐาน"""
    if df is None:
        return None
    
    rename_map = {}
    
    # ถ้ามีคอลัมน์น้อยกว่า 15 = ใช้ลำดับคอลัมน์
    # ลำดับมาตรฐาน: Sep, BU, รหัสสาขา, รหัส WMS, สาขา, Total Cube, Total Wgt, จำนวนชิ้น, Trip, Trip no, ...
    if len(df.columns) >= 8:
        col_list = list(df.columns)
        # ลำดับ 2 = รหัสสาขา
        if len(col_list) > 2:
            rename_map[col_list[2]] = 'Code'
        # ลำดับ 4 = สาขา/ชื่อ
        if len(col_list) > 4:
            rename_map[col_list[4]] = 'Name'
        # ลำดับ 5 = Total Cube
        if len(col_list) > 5:
            rename_map[col_list[5]] = 'Cube'
        # ลำดับ 6 = Total Wgt
        if len(col_list) > 6:
            rename_map[col_list[6]] = 'Weight'
        # ลำดับ 8 = Trip
        if len(col_list) > 8:
            rename_map[col_list[8]] = 'Trip'
        # ลำดับ 9 = Trip no
        if len(col_list) > 9:
            rename_map[col_list[9]] = 'TripNo'
    
    # ตรวจสอบเพิ่มเติมจากชื่อคอลัมน์
    for col in df.columns:
        if col in rename_map:
            continue
        col_clean = str(col).strip()
        col_upper = col_clean.upper().replace(' ', '').replace('_', '')
        
        if col_clean == 'BranchCode' or 'รหัสสาขา' in col_clean or col_clean == 'รหัส WMS' or 'BRANCH_CODE' in col_upper:
            rename_map[col] = 'Code'
        elif col_clean == 'Branch' or 'ชื่อสาขา' in col_clean or col_clean == 'สาขา' or 'BRANCH' in col_upper:
            rename_map[col] = 'Name'
        elif 'TOTALWGT' in col_upper or 'น้ำหนัก' in col_clean or 'WGT' in col_upper or 'WEIGHT' in col_upper:
            rename_map[col] = 'Weight'
        elif 'TOTALCUBE' in col_upper or 'คิว' in col_clean or 'CUBE' in col_upper:
            rename_map[col] = 'Cube'
        elif 'latitude' in col_clean.lower() or col_clean == 'ละติจูด' or 'LAT' in col_upper:
            rename_map[col] = 'Latitude'
        elif 'longitude' in col_clean.lower() or col_clean == 'ลองติจูด' or 'LONG' in col_upper or 'LNG' in col_upper:
            rename_map[col] = 'Longitude'
        elif 'จังหวัด' in col_clean or 'PROVINCE' in col_upper:
            rename_map[col] = 'Province'
        elif col_upper in ['TRIPNO', 'TRIP_NO'] or col_clean == 'Trip no':
            rename_map[col] = 'TripNo'
        elif col_upper == 'TRIP' or 'ทริป' in col_clean or 'เที่ยว' in col_clean:
            rename_map[col] = 'Trip'
        elif 'BOOKING' in col_upper:
            rename_map[col] = 'Booking'
    
    df = df.rename(columns=rename_map)
    
    # ลบคอลัมน์ซ้ำ
    df = df.loc[:, ~df.columns.duplicated()]
    
    if 'Code' in df.columns:
        df['Code'] = df['Code'].apply(normalize)
        
        # ตัดสาขาที่ไม่ต้องการออก (รหัส)
        df = df[~df['Code'].isin(EXCLUDE_BRANCHES)]
        
        # ตัดสาขาที่ชื่อมี keyword ที่ไม่ต้องการ
        if 'Name' in df.columns:
            exclude_pattern = '|'.join(EXCLUDE_NAMES)
            df = df[~df['Name'].str.contains(exclude_pattern, case=False, na=False)]
    
    for col in ['Weight', 'Cube']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # เพิ่มจังหวัดจาก Master ถ้ายังไม่มี
    if 'Province' not in df.columns or df['Province'].isna().all():
        if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns and 'Code' in df.columns:
            # สร้าง mapping จาก Master
            province_map = {}
            for _, row in MASTER_DATA.iterrows():
                code = row.get('Plan Code', '')
                province = row.get('จังหวัด', '')
                if code and province:
                    province_map[code] = province
            
            # ฟังก์ชันค้นหาจังหวัดจากชื่อสาขา
            def find_province_by_name(code, name):
                # ลองหาจาก code ก่อน
                if code in province_map:
                    return province_map[code]
                
                # ถ้าไม่เจอ ลองค้นหาจากชื่อสาขา
                if not name or pd.isna(name):
                    return None
                
                # แยกคำสำคัญจากชื่อ (เอาคำแรกที่ไม่ใช่ prefix)
                keywords = str(name).replace('MAX MART-', '').replace('PUNTHAI-', '').replace('LUBE', '').strip()
                if not keywords:
                    return None
                
                # ค้นหาในชื่อสาขาของ Master
                for _, master_row in MASTER_DATA.iterrows():
                    master_name = str(master_row.get('สาขา', ''))
                    # ถ้าชื่อคล้ายกัน (มีคำสำคัญเหมือนกัน)
                    if keywords[:10] in master_name or master_name[:10] in keywords:
                        province = master_row.get('จังหวัด', '')
                        if province:
                            return province
                
                return None
            
            # ใส่จังหวัดให้แต่ละสาขา
            if 'Name' in df.columns:
                df['Province'] = df.apply(lambda row: find_province_by_name(row['Code'], row.get('Name', '')), axis=1)
            else:
                df['Province'] = df['Code'].map(province_map)
    
    return df.reset_index(drop=True)

def predict_trips(test_df, model_data):
    """
    จัดทริปด้วยระบบอัจฉริยะ โดยใช้กฎลำดับความสำคัญ:
    
    เงื่อนไขบังคับ (ต้องผ่านก่อน):
    0. ✅ จำกัดจำนวนสาขาต่อทริป (สูงสุด 12 สาขา)
    0. ✅ เช็คจังหวัดใกล้เคียงกัน (ต้องอยู่กลุ่มเดียวกัน)
    0. ✅ ถ้าเกิน 10 สาขา ต้องเป็นจังหวัดเดียวกันเท่านั้น
    0. ✅ ตรวจสอบว่าสาขาสามารถใช้รถประเภทนี้ได้หรือไม่จากประวัติ
    
    กฎการจับคู่ (หลังผ่านเงื่อนไขบังคับ):
    1. ✅ เคยไปด้วยกันในประวัติ (trip_pairs) + ใช้รถแบบเดิม
    2. ✅ ชื่อสาขาคล้ายกัน (เช่น นครราชสีมา1, นครราชสีมา2)
    3. ✅ AI ทำนายจาก Decision Tree Model
    4. ✅ เช็คน้ำหนัก/คิว ไม่เกินขีดจำกัดรถ
    """
    model = model_data['model']
    trip_pairs = model_data['trip_pairs'].copy()  # คัดลอกเพื่อไม่ให้กระทบต้นฉบับ
    branch_info = model_data['branch_info']
    trip_vehicles = model_data.get('trip_vehicles', {}).copy()
    branch_vehicles = model_data.get('branch_vehicles', {})
    
    # ★ ถ้าไฟล์อัปโหลดมีคอลัมน์ Trip ให้ใช้เป็นข้อมูลอ้างอิงหลัก
    # เพราะเป็นแผนงานที่ใช้จริงมาแล้ว
    file_trip_vehicles = {}  # เก็บประเภทรถจากไฟล์แผนงาน
    use_file_trips = False  # ใช้ทริปจากไฟล์โดยตรง
    
    if 'Trip' in test_df.columns and test_df['Trip'].notna().any():
        use_file_trips = True
        st.info(f"📋 พบคอลัมน์ทริปในไฟล์ - ใช้การจัดทริปจากไฟล์โดยตรง")
        
        # สร้าง trip_pairs จากไฟล์แผนงาน
        for trip_id, group in test_df.groupby('Trip'):
            if pd.isna(trip_id):
                continue
            codes = group['Code'].unique().tolist()
            
            # ดึงประเภทรถจาก TripNo (เช่น 4W009 -> 4W, JB014 -> JB)
            if 'TripNo' in group.columns:
                trip_no = group['TripNo'].iloc[0]
                if pd.notna(trip_no):
                    trip_no_str = str(trip_no).strip()
                    if trip_no_str.startswith('4W'):
                        vehicle_type = '4W'
                    elif trip_no_str.startswith('JB'):
                        vehicle_type = 'JB'
                    elif trip_no_str.startswith('6W'):
                        vehicle_type = '6W'
                    else:
                        vehicle_type = None
                    
                    # เก็บประเภทรถสำหรับแต่ละคู่สาขา
                    if vehicle_type:
                        for i in range(len(codes)):
                            for j in range(i+1, len(codes)):
                                pair = tuple(sorted([codes[i], codes[j]]))
                                file_trip_vehicles[pair] = vehicle_type
            
            # สร้างคู่สาขาในทริปเดียวกัน
            for i in range(len(codes)):
                for j in range(i+1, len(codes)):
                    pair = tuple(sorted([codes[i], codes[j]]))
                    trip_pairs.add(pair)  # เพิ่มเข้า trip_pairs
    
    # รวม file_trip_vehicles เข้ากับ trip_vehicles (ให้ไฟล์มีความสำคัญกว่า)
    for pair, vehicle in file_trip_vehicles.items():
        trip_vehicles[pair] = {'most_used': vehicle, 'vehicle': vehicle}
    
    # เพิ่มสาขาใหม่
    for code in test_df['Code'].unique():
        if code not in branch_info:
            code_data = test_df[test_df['Code'] == code]
            branch_info[code] = {
                'avg_weight': code_data['Weight'].mean(),
                'avg_cube': code_data['Cube'].mean(),
                'total_trips': 1,
                'province': code_data['Province'].iloc[0] if 'Province' in code_data.columns else 'UNKNOWN',
                'latitude': 0.0,
                'longitude': 0.0
            }
    
    # ★★★ ถ้ามีคอลัมน์ Trip ในไฟล์ ใช้โดยตรงเลย ★★★
    if use_file_trips:
        # ใช้ Trip จากไฟล์โดยตรง
        test_df_result = test_df.copy()
        
        # ดึงประเภทรถจาก TripNo
        trip_truck_map_file = {}
        if 'TripNo' in test_df.columns:
            for trip_id in test_df['Trip'].dropna().unique():
                trip_data = test_df[test_df['Trip'] == trip_id]
                if 'TripNo' in trip_data.columns and len(trip_data) > 0:
                    trip_no = trip_data['TripNo'].iloc[0]
                    if pd.notna(trip_no):
                        trip_no_str = str(trip_no).strip()
                        if trip_no_str.startswith('4W'):
                            trip_truck_map_file[trip_id] = '4W'
                        elif trip_no_str.startswith('JB'):
                            trip_truck_map_file[trip_id] = 'JB'
                        elif trip_no_str.startswith('6W'):
                            trip_truck_map_file[trip_id] = '6W'
        
        # สร้าง summary
        summary_data = []
        for trip_num in sorted(test_df_result['Trip'].dropna().unique()):
            trip_data = test_df_result[test_df_result['Trip'] == trip_num]
            total_w = trip_data['Weight'].sum()
            total_c = trip_data['Cube'].sum()
            
            # ใช้รถจากไฟล์
            if trip_num in trip_truck_map_file:
                suggested = trip_truck_map_file[trip_num]
                source = "📋 ไฟล์"
            else:
                trip_codes = trip_data['Code'].unique()
                suggested = suggest_truck(total_w, total_c, '6W', trip_codes)
                source = "🤖 AI"
            
            # คำนวณ % การใช้รถ
            if suggested in LIMITS:
                w_util = (total_w / LIMITS[suggested]['max_w']) * 100
                c_util = (total_c / LIMITS[suggested]['max_c']) * 100
            else:
                w_util = c_util = 0
            
            summary_data.append({
                'Trip': int(trip_num),
                'Branches': len(trip_data['Code'].unique()),
                'Weight': total_w,
                'Cube': total_c,
                'Truck': f"{suggested} {source}",
                'Weight_Use%': w_util,
                'Cube_Use%': c_util
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # เพิ่มคอลัมน์รถ
        trip_truck_display = {}
        for _, row in summary_df.iterrows():
            trip_truck_display[row['Trip']] = row['Truck']
        
        test_df_result['Truck'] = test_df_result['Trip'].map(trip_truck_display)
        test_df_result['VehicleCheck'] = "✅ ใช้ตามไฟล์"
        
        return test_df_result, summary_df
    
    # ถ้าไม่มีคอลัมน์ Trip ให้จัดทริปใหม่
    all_codes = test_df['Code'].unique().tolist()
    assigned_trips = {}
    trip_counter = 1
    trip_recommended_vehicles = {}  # เก็บรถที่แนะนำสำหรับแต่ละทริป
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_codes = len(all_codes)
    processed = 0
    
    while all_codes:
        seed_code = all_codes.pop(0)
        current_trip = [seed_code]
        assigned_trips[seed_code] = trip_counter
        
        processed += 1
        progress_bar.progress(processed / total_codes)
        status_text.text(f"กำลังจัดทริป {trip_counter}... ({processed}/{total_codes} สาขา)")
        
        remaining = all_codes[:]
        recommended_vehicle = None  # รถที่แนะนำสำหรับทริปนี้
        
        # ฟังก์ชันดึงจังหวัดจากหลายแหล่ง (Master → ไฟล์อัปโหลด → ประวัติ)
        def get_province(branch_code):
            # 1. ลองดึงจาก Master ก่อน (ข้อมูลแม่นที่สุด)
            if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
                master_row = MASTER_DATA[MASTER_DATA['Plan Code'] == branch_code]
                if len(master_row) > 0:
                    prov = master_row.iloc[0].get('จังหวัด', '')
                    if prov and str(prov).strip() and prov != 'UNKNOWN':
                        return str(prov).strip()
            
            # 2. ลองดึงจากไฟล์อัปโหลด
            if 'Province' in test_df.columns:
                prov = test_df[test_df['Code'] == branch_code]['Province'].iloc[0] if len(test_df[test_df['Code'] == branch_code]) > 0 else None
                if prov and prov != 'UNKNOWN' and str(prov).strip():
                    return prov
            
            # 3. ถ้าไม่มี ลองดึงจาก branch_info (ประวัติการเทรน)
            if branch_code in branch_info:
                prov = branch_info[branch_code].get('province', 'UNKNOWN')
                if prov and prov != 'UNKNOWN' and str(prov).strip():
                    return prov
            
            return 'UNKNOWN'
        
        # ข้อมูลจังหวัดของ seed
        seed_province = get_province(seed_code)
        seed_name = test_df[test_df['Code'] == seed_code]['Name'].iloc[0] if 'Name' in test_df.columns else ''
        
        # จัดเรียง remaining ตามลำดับ: ชื่อคล้ายกัน แล้วตามลำดับในไฟล์
        code_to_index = {row['Code']: idx for idx, row in test_df.iterrows()} if 'Code' in test_df.columns else {}
        
        def get_priority(code):
            code_name = test_df[test_df['Code'] == code]['Name'].iloc[0] if 'Name' in test_df.columns else ''
            code_index = code_to_index.get(code, 999999)
            seed_index = code_to_index.get(seed_code, 0)
            
            if is_similar_name(seed_name, code_name):
                # ลำดับแรก - ชื่อคล้ายกัน + เรียงตามลำดับในไฟล์
                return (0, abs(code_index - seed_index))
            pair = tuple(sorted([seed_code, code]))
            if pair in trip_pairs:
                # ลำดับสอง - เคยไปด้วยกัน
                return (1, code_index)
            # ลำดับสุดท้าย - อื่นๆ
            return (2, code_index)
        
        remaining_sorted = sorted(remaining, key=get_priority)
        
        for code in remaining_sorted:
            pair = tuple(sorted([seed_code, code]))
            code_province = get_province(code)
            
            # เช็คจำนวนสาขาก่อน - ถ้าเกิน MAX แล้วไม่เพิ่ม
            if len(current_trip) >= MAX_BRANCHES_PER_TRIP:
                continue  # เกินจำนวนสูงสุดแล้ว
            
            # ดึงชื่อสาขาก่อนเพื่อใช้เช็คชื่อคล้ายกัน
            seed_name = test_df[test_df['Code'] == seed_code]['Name'].iloc[0] if 'Name' in test_df.columns else ''
            code_name = test_df[test_df['Code'] == code]['Name'].iloc[0] if 'Name' in test_df.columns else ''
            names_are_similar = is_similar_name(seed_name, code_name)
            
            # กฎการเช็คจังหวัด/พื้นที่ - เช็คจังหวัดก่อนเสมอ:
            # 1. เช็คจังหวัดก่อน - ต้องเป็นจังหวัดเดียวกัน (ไม่มีข้อยกเว้น)
            # 2. ถ้าชื่อคล้ายกัน → อนุญาต (ไม่เช็คระยะทาง)
            # 3. ถ้ามีประวัติร่วมกัน → อนุญาต (ไม่เช็คระยะทาง)
            # 4. อื่นๆ → เช็คระยะทาง (ต้องอยู่ใกล้กันภายใน 20 กม.)
            
            has_history = pair in trip_pairs
            
            # เช็คจังหวัดก่อนเสมอ (ไม่มีข้อยกเว้น)
            if seed_province == 'UNKNOWN' or code_province == 'UNKNOWN':
                # ไม่มีข้อมูลจังหวัด - อนุญาตเฉพาะชื่อคล้ายกัน
                if not names_are_similar:
                    continue
            elif seed_province != code_province:
                # ต่างจังหวัด = ห้ามจับคู่เด็ดขาด (แม้มีประวัติ)
                continue
            
            # ผ่านเช็คจังหวัดแล้ว - ตรวจสอบความเหมาะสมในการรวมกลุ่ม
            # ลำดับความสำคัญ: 1. ชื่อคล้ายกัน  2. ประวัติ  3. ตำบล/อำเภอเดียวกัน  4. ระยะทาง
            
            can_pair = False
            
            # 1. ชื่อคล้ายกัน → รวมได้ทันที
            if names_are_similar:
                can_pair = True
            # 2. มีประวัติร่วมกัน → รวมได้ทันที
            elif has_history:
                can_pair = True
            # 3. เช็คตำบล/อำเภอจาก Master
            elif not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
                seed_master = MASTER_DATA[MASTER_DATA['Plan Code'] == seed_code]
                code_master = MASTER_DATA[MASTER_DATA['Plan Code'] == code]
                
                if len(seed_master) > 0 and len(code_master) > 0:
                    seed_m = seed_master.iloc[0]
                    code_m = code_master.iloc[0]
                    
                    # เช็คตำบลเดียวกัน
                    if seed_m.get('ตำบล', '') and code_m.get('ตำบล', ''):
                        if seed_m.get('ตำบล', '') == code_m.get('ตำบล', ''):
                            can_pair = True
                    
                    # เช็คอำเภอเดียวกัน (และจังหวัดเดียวกัน)
                    if not can_pair and seed_m.get('อำเภอ', '') and code_m.get('อำเภอ', ''):
                        if (seed_m.get('อำเภอ', '') == code_m.get('อำเภอ', '') and 
                            seed_m.get('จังหวัด', '') == code_m.get('จังหวัด', '')):
                            can_pair = True
            
            # 4. ถ้ายังไม่ผ่าน → เช็คระยะทาง (ภายใน 20 กม.) - ใช้พิกัดจาก Master
            if not can_pair:
                # ดึงพิกัดจาก Master
                seed_lat, seed_lon = 0, 0
                code_lat, code_lon = 0, 0
                
                if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
                    seed_master = MASTER_DATA[MASTER_DATA['Plan Code'] == seed_code]
                    code_master = MASTER_DATA[MASTER_DATA['Plan Code'] == code]
                    
                    if len(seed_master) > 0:
                        seed_lat = seed_master.iloc[0].get('ละติจูด', 0)
                        seed_lon = seed_master.iloc[0].get('ลองติจูด', 0)
                    
                    if len(code_master) > 0:
                        code_lat = code_master.iloc[0].get('ละติจูด', 0)
                        code_lon = code_master.iloc[0].get('ลองติจูด', 0)
                
                # ถ้าไม่มีใน Master ลองดึงจากไฟล์อัปโหลด
                if seed_lat == 0 and seed_lon == 0 and 'Latitude' in test_df.columns:
                    seed_lat = test_df[test_df['Code'] == seed_code]['Latitude'].iloc[0] if len(test_df[test_df['Code'] == seed_code]) > 0 else 0
                    seed_lon = test_df[test_df['Code'] == seed_code]['Longitude'].iloc[0] if len(test_df[test_df['Code'] == seed_code]) > 0 else 0
                
                if code_lat == 0 and code_lon == 0 and 'Latitude' in test_df.columns:
                    code_lat = test_df[test_df['Code'] == code]['Latitude'].iloc[0] if len(test_df[test_df['Code'] == code]) > 0 else 0
                    code_lon = test_df[test_df['Code'] == code]['Longitude'].iloc[0] if len(test_df[test_df['Code'] == code]) > 0 else 0
                    
                    # คำนวณระยะทาง (haversine formula)
                    if seed_lat != 0 and seed_lon != 0 and code_lat != 0 and code_lon != 0:
                        import math
                        lat1, lon1 = math.radians(seed_lat), math.radians(seed_lon)
                        lat2, lon2 = math.radians(code_lat), math.radians(code_lon)
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                        c = 2 * math.asin(math.sqrt(a))
                        distance_km = 6371 * c  # รัศมีโลก
                        
                        # ถ้าห่างเกิน 20 กม. = ต่างพื้นที่ → ข้าม
                        if distance_km <= 20:
                            can_pair = True
            
            # ถ้าไม่ผ่านเงื่อนไขใดๆ → ข้ามสาขานี้
            if not can_pair:
                continue
            
            # กฎ 1: ถ้าเคยไปด้วยกันในประวัติ = จัดเข้าทริปเดียวกัน + ใช้รถแบบเดิม
            if pair in trip_pairs:
                should_pair = True
                # ดึงข้อมูลรถจากประวัติ
                if pair in trip_vehicles and recommended_vehicle is None:
                    vehicle_info = trip_vehicles[pair]
                    hist_vehicle = vehicle_info.get('most_used') or vehicle_info.get('vehicle', '6W')
                    
                    # ตรวจสอบว่าสาขาทั้งสองสามารถใช้รถประเภทนี้ได้หรือไม่
                    seed_can_use = can_branch_use_vehicle(seed_code, hist_vehicle, branch_vehicles)
                    code_can_use = can_branch_use_vehicle(code, hist_vehicle, branch_vehicles)
                    
                    if seed_can_use and code_can_use:
                        recommended_vehicle = hist_vehicle
                    else:
                        # ถ้าสาขาใดไม่เคยใช้รถประเภทนี้ ให้หารถที่ทั้งคู่เคยใช้ร่วมกัน
                        seed_most_used = get_most_used_vehicle_for_branch(seed_code, branch_vehicles)
                        code_most_used = get_most_used_vehicle_for_branch(code, branch_vehicles)
                        
                        # เลือกรถที่ใหญ่กว่าที่ทั้งสองเคยใช้
                        vehicle_sizes = {'4W': 1, 'JB': 2, '6W': 3}
                        if seed_most_used and code_most_used:
                            if vehicle_sizes.get(seed_most_used, 0) >= vehicle_sizes.get(code_most_used, 0):
                                recommended_vehicle = seed_most_used
                            else:
                                recommended_vehicle = code_most_used
            else:
                # กฎ 2: เช็คชื่อสาขาคล้ายกัน (เช่น นครราชสีมา1, นครราชสีมา2)
                # (ใช้ names_are_similar ที่คำนวณไว้ข้างบนแล้ว)
                if names_are_similar:
                    should_pair = True
                else:
                    # กฎ 3: ใช้โมเดล AI ทำนาย (เฉพาะกรณีที่มีข้อมูลจังหวัด)
                    if seed_province != 'UNKNOWN' and code_province != 'UNKNOWN':
                        features = create_pair_features(seed_code, code, branch_info)
                        X = pd.DataFrame([features])
                        should_pair = model.predict(X)[0] == 1
                    else:
                        should_pair = False  # ไม่ใช้ AI ถ้าไม่มีข้อมูลจังหวัด
            
            if should_pair:
                # คำนวณน้ำหนัก/คิวหลังเพิ่มสาขานี้
                trip_weight = test_df[test_df['Code'].isin(current_trip + [code])]['Weight'].sum()
                trip_cube = test_df[test_df['Code'].isin(current_trip + [code])]['Cube'].sum()
                
                # ถ้ามีรถแนะนำจากประวัติ ใช้ขีดจำกัดของรถนั้น
                if recommended_vehicle and recommended_vehicle in LIMITS:
                    max_w = LIMITS[recommended_vehicle]['max_w'] * BUFFER
                    max_c = LIMITS[recommended_vehicle]['max_c'] * BUFFER
                    vehicle_type = recommended_vehicle
                else:
                    # ถ้าไม่มี ใช้รถ 6W เป็นค่าเริ่มต้น
                    max_w = LIMITS['6W']['max_w'] * BUFFER
                    max_c = LIMITS['6W']['max_c'] * BUFFER
                    vehicle_type = '6W'
                
                # ฟังก์ชันเช็คว่าสาขาอยู่ใกล้กันหรือไม่ (ตำบล/อำเภอเดียวกัน)
                def branches_are_close(code1, code2):
                    """เช็คว่าสาขาอยู่ใกล้กันหรือไม่ (ตำบล/อำเภอเดียวกัน)"""
                    # ถ้าไม่มี Master Data ให้ถือว่าใกล้กัน (ใช้จังหวัดเดียวกันแทน)
                    if MASTER_DATA.empty or 'Plan Code' not in MASTER_DATA.columns:
                        return True
                    
                    # ดึงข้อมูลจาก Master
                    master1 = MASTER_DATA[MASTER_DATA['Plan Code'] == code1]
                    master2 = MASTER_DATA[MASTER_DATA['Plan Code'] == code2]
                    
                    if len(master1) > 0 and len(master2) > 0:
                        m1 = master1.iloc[0]
                        m2 = master2.iloc[0]
                        
                        # เช็คตำบลก่อน
                        if m1.get('ตำบล', '') and m2.get('ตำบล', '') and m1.get('ตำบล', '') == m2.get('ตำบล', ''):
                            return True
                        
                        # เช็คอำเภอ
                        if (m1.get('อำเภอ', '') and m2.get('อำเภอ', '') and 
                            m1.get('อำเภอ', '') == m2.get('อำเภอ', '') and
                            m1.get('จังหวัด', '') == m2.get('จังหวัด', '')):
                            return True
                    
                    return False
                
                # เช็คว่าเกินขีดจำกัดหรือไม่
                can_fit = trip_weight <= max_w and trip_cube <= max_c
                
                # ถ้าเกิน → เช็คว่าเกินนิดหน่อยและอยู่ใกล้กันไหม
                if not can_fit:
                    # เกินเล็กน้อย = เกินไม่เกิน 10%
                    weight_exceed = (trip_weight - max_w) / max_w if max_w > 0 else 0
                    cube_exceed = (trip_cube - max_c) / max_c if max_c > 0 else 0
                    
                    slightly_exceed = weight_exceed <= 0.10 or cube_exceed <= 0.10
                    
                    if slightly_exceed:
                        # เช็คว่าสาขาอยู่ใกล้กันหรือไม่
                        all_branches_close = True
                        for existing_code in current_trip:
                            if not branches_are_close(existing_code, code):
                                all_branches_close = False
                                break
                        
                        if all_branches_close:
                            # คำนวณว่าถ้าแยก สาขาที่แยกออกไปจะใช้รถเล็กหรือไม่เต็ม
                            code_weight = test_df[test_df['Code'] == code]['Weight'].sum()
                            code_cube = test_df[test_df['Code'] == code]['Cube'].sum()
                            
                            # ลองดูว่าถ้าแยกออกไป รถเล็กจะไม่เต็มหรือไม่
                            # ใช้รถเล็กสุด (4W) เป็นตัวอ้างอิง
                            small_vehicle_fill = max(
                                code_weight / LIMITS['4W']['max_w'],
                                code_cube / LIMITS['4W']['max_c']
                            ) if vehicle_type != '4W' else 0
                            
                            # ถ้ารถเล็กไม่เต็ม 50% = สิ้นเปลือง → ยอมรับให้รวมกันแม้เกิน
                            if small_vehicle_fill < 0.5:
                                can_fit = True  # ยอมรับเกินเพื่อประหยัดรถ
                
                if can_fit:
                    current_trip.append(code)
                    assigned_trips[code] = trip_counter
                    all_codes.remove(code)
        
        # บันทึกรถที่แนะนำสำหรับทริปนี้
        if recommended_vehicle:
            trip_recommended_vehicles[trip_counter] = recommended_vehicle
        
        trip_counter += 1
    
    progress_bar.empty()
    status_text.empty()
    
    test_df['Trip'] = test_df['Code'].map(assigned_trips)
    
    # ===============================================
    # Post-processing: รวมทริปที่มีสาขาน้อยและใช้รถต่ำ
    # ===============================================
    st.text("กำลังปรับปรุงการจัดทริป...")
    
    # หาทริปที่มีปัญหา (1-2 สาขา หรือ ใช้รถต่ำกว่า 50%)
    problem_trips = []
    for trip_num in test_df['Trip'].unique():
        trip_data = test_df[test_df['Trip'] == trip_num]
        branch_count = len(trip_data)
        total_w = trip_data['Weight'].sum()
        total_c = trip_data['Cube'].sum()
        
        # คำนวณ % การใช้รถ 4W
        w_util = (total_w / LIMITS['4W']['max_w']) * 100
        c_util = (total_c / LIMITS['4W']['max_c']) * 100
        max_util = max(w_util, c_util)
        
        # ถ้ามี 1-2 สาขา หรือ ใช้รถต่ำกว่า 40% → ต้องพยายามรวม
        if branch_count <= 2 or max_util < 40:
            problem_trips.append({
                'trip': trip_num,
                'count': branch_count,
                'util': max_util,
                'weight': total_w,
                'cube': total_c,
                'codes': set(trip_data['Code'].values)
            })
    
    # พยายามรวมทริปที่มีปัญหา
    merged = True
    while merged and len(problem_trips) > 0:
        merged = False
        for i, prob1 in enumerate(problem_trips):
            if prob1 is None:
                continue
            
            # หาทริปอื่นที่จังหวัดเดียวกัน
            prob1_provinces = set()
            for code in prob1['codes']:
                prov = get_province(code)
                if prov != 'UNKNOWN':
                    prob1_provinces.add(prov)
            
            # ลองรวมกับทริปอื่น
            for j, prob2 in enumerate(problem_trips[i+1:], start=i+1):
                if prob2 is None:
                    continue
                
                # เช็คจังหวัด
                prob2_provinces = set()
                for code in prob2['codes']:
                    prov = get_province(code)
                    if prov != 'UNKNOWN':
                        prob2_provinces.add(prov)
                
                # ถ้าจังหวัดเดียวกัน → ลองรวม
                if prob1_provinces & prob2_provinces:
                    combined_w = prob1['weight'] + prob2['weight']
                    combined_c = prob1['cube'] + prob2['cube']
                    combined_count = prob1['count'] + prob2['count']
                    
                    # เช็คว่ารวมแล้วใส่รถได้ไหม (ใช้ 6W)
                    if (combined_w <= LIMITS['6W']['max_w'] * BUFFER and 
                        combined_c <= LIMITS['6W']['max_c'] * BUFFER and
                        combined_count <= MAX_BRANCHES_PER_TRIP):
                        
                        # รวมทริป
                        for code in prob2['codes']:
                            test_df.loc[test_df['Code'] == code, 'Trip'] = prob1['trip']
                        
                        # อัปเดตข้อมูล prob1
                        prob1['weight'] = combined_w
                        prob1['cube'] = combined_c
                        prob1['count'] = combined_count
                        prob1['codes'] |= prob2['codes']
                        
                        # ลบ prob2 ออก
                        problem_trips[j] = None
                        merged = True
                        break
            
            if merged:
                break
        
        # ลบ None ออก
        problem_trips = [p for p in problem_trips if p is not None]
    
    # สรุปผลและแนะนำรถ
    summary_data = []
    for trip_num in sorted(test_df['Trip'].unique()):
        trip_data = test_df[test_df['Trip'] == trip_num]
        total_w = trip_data['Weight'].sum()
        total_c = trip_data['Cube'].sum()
        
        # หารถที่ใหญ่สุดที่ทุกสาขาในทริปสามารถใช้ได้
        trip_codes = trip_data['Code'].unique()
        max_vehicles = [get_max_vehicle_for_branch(c, branch_vehicles) for c in trip_codes]
        vehicle_sizes = {'4W': 1, 'JB': 2, '6W': 3}
        min_max_size = min(vehicle_sizes.get(v, 3) for v in max_vehicles)
        max_allowed_vehicle = {1: '4W', 2: 'JB', 3: '6W'}.get(min_max_size, '6W')
        
        # เช็คว่ามีสาขาที่ต้องใช้ 6W เพราะอยู่ไกลหรือไม่
        must_use_6w = False
        for code in trip_codes:
            required_vehicle, distance = get_required_vehicle_by_distance(code)
            if required_vehicle == '6W':
                must_use_6w = True
                break
        
        # เลือกรถ: ถ้ามีในประวัติใช้ตามประวัติ ไม่มีก็ auto-suggest
        if must_use_6w:
            suggested = '6W'
            source = "📍 ระยะไกล"
        elif trip_num in trip_recommended_vehicles:
            suggested = trip_recommended_vehicles[trip_num]
            # ตรวจสอบว่ารถที่แนะนำไม่เกินข้อจำกัด
            if vehicle_sizes.get(suggested, 0) > min_max_size:
                suggested = max_allowed_vehicle
            # ตรวจสอบว่าต้องใช้ 6W หรือไม่
            if must_use_6w and suggested != '6W':
                suggested = '6W'
            source = "📜 ประวัติ"
        else:
            suggested = suggest_truck(total_w, total_c, max_allowed_vehicle, trip_codes)
            source = "🤖 AI"
        
        # คำนวณ % การใช้รถ
        if suggested in LIMITS:
            w_util = (total_w / LIMITS[suggested]['max_w']) * 100
            c_util = (total_c / LIMITS[suggested]['max_c']) * 100
        else:
            w_util = c_util = 0
        
        # คำนวณระยะทางรวมของทริป
        total_distance = 0
        for code in trip_codes:
            _, distance = get_required_vehicle_by_distance(code)
            total_distance += distance
        
        summary_data.append({
            'Trip': trip_num,
            'Branches': len(trip_data),
            'Weight': total_w,
            'Cube': total_c,
            'Truck': f"{suggested} {source}",
            'Weight_Use%': w_util,
            'Cube_Use%': c_util,
            'Total_Distance': total_distance
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # เพิ่มคอลัมน์รถที่จัดส่งในรายละเอียดรายสาขา
    trip_truck_map = {}
    trip_truck_type_map = {}  # เก็บเฉพาะประเภทรถ (ไม่รวม source)
    for _, row in summary_df.iterrows():
        trip_truck_map[row['Trip']] = row['Truck']
        # ดึงประเภทรถ (ตัด emoji และ source ออก)
        truck_type = row['Truck'].split()[0] if row['Truck'] else '6W'
        trip_truck_type_map[row['Trip']] = truck_type
    
    test_df['Truck'] = test_df['Trip'].map(trip_truck_map)
    
    # เพิ่มคอลัมน์ระยะทางจาก DC และเรียงลำดับ
    def add_distance_and_sort(df):
        # เพิ่มคอลัมน์ระยะทาง
        distances = []
        for _, row in df.iterrows():
            _, distance = get_required_vehicle_by_distance(row['Code'])
            distances.append(distance)
        df['Distance_from_DC'] = distances
        
        # เรียงลำดับภายในแต่ละทริป: Trip → Distance
        df = df.sort_values(['Trip', 'Distance_from_DC'], ascending=[True, True])
        return df
    
    test_df = add_distance_and_sort(test_df)
    
    # เพิ่มคอลัมน์เช็คว่าสาขาเคยใช้รถประเภทนี้หรือไม่
    def check_vehicle_history(row):
        code = row['Code']
        trip = row['Trip']
        truck_type = trip_truck_type_map.get(trip, '6W')
        
        if code not in branch_vehicles:
            return "✅ ใช้ได้ (สาขาใหม่)"
        
        vehicle_history = branch_vehicles.get(code, {})
        if not vehicle_history:
            return "✅ ใช้ได้ (ไม่มีประวัติ)"
        
        vehicle_sizes = {'4W': 1, 'JB': 2, '6W': 3}
        requested_size = vehicle_sizes.get(truck_type, 0)
        
        # หารถใหญ่สุดที่สาขาเคยใช้
        max_used_size = max(vehicle_sizes.get(v, 0) for v in vehicle_history)
        max_used_vehicle = {1: '4W', 2: 'JB', 3: '6W'}.get(max_used_size, '6W')
        
        # ถ้าเคยใช้รถประเภทนี้
        if truck_type in vehicle_history:
            count = vehicle_history[truck_type]
            return f"✅ เคยใช้ ({count} ครั้ง)"
        
        # ถ้าขอใช้รถเล็กกว่าที่เคยใช้ = ใช้ได้
        if requested_size < max_used_size:
            return f"✅ ใช้ได้ (เคยใช้ {max_used_vehicle})"
        
        # ถ้าขอใช้รถใหญ่กว่าที่เคยใช้ = อาจเข้าไม่ได้
        history_str = ", ".join([f"{v}:{c}" for v, c in vehicle_history.items()])
        return f"🚫 จำกัด {max_used_vehicle} ({history_str})"
    
    test_df['VehicleCheck'] = test_df.apply(check_vehicle_history, axis=1)
    
    return test_df, summary_df

# ==========================================
# STREAMLIT UI
# ==========================================
def main():
    st.set_page_config(
        page_title="ระบบจัดเที่ยว",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚚 ระบบจัดเที่ยว")
    with col2:
        st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f69a.svg", width=100)
    
    st.markdown("---")
    
    # โหลดโมเดล
    model_data = load_model()
    
    if not model_data:
        st.error("❌ ไม่พบข้อมูลโมเดล กรุณาเทรนโมเดลก่อนใช้งาน")
        st.info("💡 รันคำสั่ง: `python test_model.py`")
        st.stop()
    
    # อัปโหลดไฟล์ครั้งเดียว
    st.markdown("### 📂 อัปโหลดไฟล์รายการออเดอร์")
    uploaded_file = st.file_uploader(
        "เลือกไฟล์ Excel (.xlsx)", 
        type=['xlsx'],
        help="อัปโหลดไฟล์ Excel ที่มีรายการสาขาและออเดอร์"
    )
    
    if uploaded_file:
        with st.spinner("⏳ กำลังอ่านข้อมูล..."):
            df = load_excel(uploaded_file.read())
            df = process_dataframe(df)
            
            if df is not None and 'Code' in df.columns:
                st.success(f"✅ อ่านข้อมูลสำเร็จ: **{len(df):,}** รายการ")
                
                # แสดงข้อมูลพื้นฐาน
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📍 จำนวนสาขา", f"{df['Code'].nunique():,}")
                with col2:
                    st.metric("⚖️ น้ำหนักรวม", f"{df['Weight'].sum():,.0f} kg")
                with col3:
                    st.metric("📦 คิวรวม", f"{df['Cube'].sum():.1f} m³")
                with col4:
                    provinces = df['Province'].nunique() if 'Province' in df.columns else 0
                    st.metric("🗺️ จังหวัด", f"{provinces}")
                
                # แสดงตัวอย่างข้อมูล
                with st.expander("🔍 ดูข้อมูลตัวอย่าง"):
                    st.dataframe(df.head(10), use_container_width=True)
                
                st.markdown("---")
                
                # แท็บหลัก
                tab1, tab2 = st.tabs(["📦 จัดเที่ยว (ตามน้ำหนัก)", "🗺️ จัดกลุ่มตามภาค (ไม่สนน้ำหนัก)"])
                    
                # ==========================================
                # แท็บ 1: จัดเที่ยว (ตามน้ำหนัก)
                # ==========================================
                with tab1:
                    # ปุ่มจัดทริป
                    if st.button("🚀 เริ่มจัดเที่ยว", type="primary", use_container_width=True):
                        with st.spinner("⏳ กำลังประมวลผล..."):
                            result_df, summary = predict_trips(df.copy(), model_data)
                            
                            st.balloons()
                            st.success(f"✅ **จัดทริปเสร็จสมบูรณ์!** รวม **{len(summary)}** ทริป")
                            
                            st.markdown("---")
                            
                            # สถิติโดยรวม
                            st.markdown("### 📊 สรุปผลการจัดทริป")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("🚚 จำนวนทริป", len(summary))
                            with col2:
                                st.metric("📍 จำนวนสาขา", len(result_df))
                            with col3:
                                avg_branches = len(result_df) / result_df['Trip'].nunique()
                                st.metric("📊 เฉลี่ยสาขา/ทริป", f"{avg_branches:.1f}")
                            with col4:
                                avg_util = summary['Cube_Use%'].mean()
                                st.metric("📈 การใช้รถเฉลี่ย", f"{avg_util:.0f}%")
                            
                            st.markdown("---")
                            
                            # ตารางสรุปแต่ละทริป
                            st.markdown("### 🚛 รายละเอียดแต่ละทริป")
                            st.dataframe(
                                summary.style.format({
                                    'Weight': '{:.2f}',
                                    'Cube': '{:.2f}',
                                    'Weight_Use%': '{:.1f}%',
                                    'Cube_Use%': '{:.1f}%',
                                    'Total_Distance': '{:.1f} km'
                                }).background_gradient(
                                    subset=['Weight_Use%', 'Cube_Use%'],
                                    cmap='RdYlGn',
                                    vmin=0,
                                    vmax=100
                                ),
                                use_container_width=True,
                                height=400
                            )
                            
                            # ตารางรายละเอียดทั้งหมด (มีคอลัมน์รถและระยะทาง)
                            with st.expander("📋 ดูรายละเอียดรายสาขา (เรียงตามระยะทาง)"):
                                # จัดเรียงคอลัมน์ที่สำคัญ
                                display_cols = ['Trip', 'Code', 'Name', 'Distance_from_DC', 'Weight', 'Cube', 'Truck', 'VehicleCheck']
                                if 'Province' in result_df.columns:
                                    display_cols.insert(3, 'Province')
                                
                                display_df = result_df[display_cols].copy()
                                if 'Province' not in result_df.columns:
                                    display_df.columns = ['ทริป', 'รหัส', 'ชื่อสาขา', 'ระยะทาง(km)', 'น้ำหนัก(kg)', 'คิว(m³)', 'รถ', 'ตรวจสอบรถ']
                                else:
                                    display_df.columns = ['ทริป', 'รหัส', 'ชื่อสาขา', 'จังหวัด', 'ระยะทาง(km)', 'น้ำหนัก(kg)', 'คิว(m³)', 'รถ', 'ตรวจสอบรถ']
                                
                                # จัดรูปแบบคอลัมน์ระยะทาง
                                st.dataframe(
                                    display_df.style.format({
                                        'ระยะทาง(km)': '{:.1f}',
                                        'น้ำหนัก(kg)': '{:.2f}',
                                        'คิว(m³)': '{:.2f}'
                                    }),
                                    use_container_width=True, 
                                    height=400
                                )
                            
                            # แสดงสาขาที่มีคำเตือน
                            warning_branches = result_df[result_df['VehicleCheck'].str.contains('⚠️', na=False)]
                            if len(warning_branches) > 0:
                                with st.expander(f"⚠️ สาขาที่ใช้รถต่างจากปกติ ({len(warning_branches)} สาขา)"):
                                    st.warning("สาขาเหล่านี้ปกติใช้รถประเภทอื่น แต่ถูกจัดให้ใช้รถประเภทที่ต่างออกไป")
                                    display_cols_warn = ['Trip', 'Code', 'Name', 'Truck', 'VehicleCheck']
                                    display_warn_df = warning_branches[display_cols_warn].copy()
                                    display_warn_df.columns = ['ทริป', 'รหัส', 'ชื่อสาขา', 'รถที่จัด', 'ประวัติการใช้รถ']
                                    st.dataframe(display_warn_df, use_container_width=True)
                            
                            st.markdown("---")
                            
                            # ดาวน์โหลด
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                result_df.to_excel(writer, sheet_name='รายละเอียดทริป', index=False)
                                summary.to_excel(writer, sheet_name='สรุปทริป', index=False)
                            
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                st.download_button(
                                    label="📥 ดาวน์โหลดผลลัพธ์ (Excel)",
                                    data=output.getvalue(),
                                    file_name=f"ผลจัดทริป_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                
                # ==========================================
                # แท็บ 2: จัดกลุ่มสาขาตามภาค (ไม่สนน้ำหนัก)
                # ==========================================
                with tab2:
                    df_region = df.copy()
                    
                    # จัดกลุ่มตามภาค
                    branch_info = model_data.get('branch_info', {})
                    trip_pairs = model_data.get('trip_pairs', set())
                    
                    # สร้างข้อมูลภาคสำหรับแต่ละสาขา (จากไฟล์ประวัติ)
                    region_groups = {
                        'ภาคกลาง-กรุงเทพชั้นใน': ['กรุงเทพมหานคร'],
                        'ภาคกลาง-กรุงเทพชั้นกลาง': ['กรุงเทพมหานคร'],
                        'ภาคกลาง-กรุงเทพชั้นนอก': ['กรุงเทพมหานคร'],
                        'ภาคกลาง-ปริมณฑล': ['นครปฐม', 'นนทบุรี', 'ปทุมธานี', 'สมุทรปราการ', 'สมุทรสาคร'],
                        'ภาคกลาง-กลางตอนบน': ['ชัยนาท', 'พระนครศรีอยุธยา', 'ลพบุรี', 'สระบุรี', 'สิงห์บุรี', 'อ่างทอง', 'อยุธยา'],
                        'ภาคกลาง-กลางตอนล่าง': ['สมุทรสงคราม', 'สุพรรณบุรี'],
                        'ภาคตะวันตก': ['กาญจนบุรี', 'ประจวบคีรีขันธ์', 'ราชบุรี', 'เพชรบุรี'],
                        'ภาคตะวันออก-ปริมณฑล': ['ฉะเชิงเทรา'],
                        'ภาคตะวันออก': ['จันทบุรี', 'ชลบุรี', 'ตราด', 'นครนายก', 'ปราจีนบุรี', 'ระยอง', 'สระแก้ว'],
                        'ภาคอีสาน-อีสานเหนือ': ['นครพนม', 'บึงกาฬ', 'มุกดาหาร', 'สกลนคร', 'หนองคาย', 'หนองบัวลำภู', 'อุดรธานี', 'เลย'],
                        'ภาคอีสาน-อีสานกลาง': ['กาฬสินธุ์', 'ขอนแก่น', 'ชัยภูมิ', 'มหาสารคาม', 'ร้อยเอ็ด'],
                        'ภาคอีสาน-อีสานใต้': ['นครราชสีมา', 'โคราช', 'บุรีรัมย์', 'ยโสธร', 'ศรีสะเกษ', 'สุรินทร์', 'อำนาจเจริญ', 'อุบลราชธานี'],
                        'ภาคเหนือ-เหนือตอนบน': ['น่าน', 'พะเยา', 'ลำปาง', 'ลำพูน', 'เชียงราย', 'เชียงใหม่', 'แพร่', 'แม่ฮ่องสอน'],
                        'ภาคเหนือ-เหนือตอนล่าง': ['กำแพงเพชร', 'ตาก', 'นครสวรรค์', 'พิจิตร', 'พิษณุโลก', 'สุโขทัย', 'อุตรดิตถ์', 'อุทัยธานี', 'เพชรบูรณ์'],
                        'ภาคใต้-ใต้ฝั่งอันดามัน': ['กระบี่', 'ตรัง', 'พังงา', 'ภูเก็ต', 'ระนอง', 'สตูล'],
                        'ภาคใต้-ใต้ฝั่งอ่าวไทย': ['ชุมพร', 'นครศรีธรรมราช', 'พัทลุง', 'ยะลา', 'สงขลา', 'สุราษฎร์ธานี', 'ปัตตานี', 'นราธิวาส']
                    }
                    
                    def get_region(province):
                        if pd.isna(province) or not province or str(province).strip() in ['', 'nan', 'UNKNOWN']:
                            return 'ไม่ระบุ'
                        for region, provinces in region_groups.items():
                            if any(p in str(province) for p in provinces):
                                return region
                        return 'อื่นๆ'
                    
                    # เพิ่มคอลัมน์ภาค - ดึงจังหวัดจาก Master ถ้าไม่มี
                    if 'Province' not in df_region.columns or df_region['Province'].isna().any():
                        # ดึงจังหวัดจาก Master
                        if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
                            province_map = {}
                            for _, row in MASTER_DATA.iterrows():
                                code = row.get('Plan Code', '')
                                province = row.get('จังหวัด', '')
                                if code and province:
                                    province_map[code] = province
                            
                            # ใส่จังหวัดให้แต่ละสาขา
                            if 'Province' not in df_region.columns:
                                df_region['Province'] = df_region['Code'].map(province_map)
                            else:
                                # เติมเฉพาะที่เป็น NaN
                                df_region['Province'] = df_region.apply(
                                    lambda row: province_map.get(row['Code'], row.get('Province', 'UNKNOWN')) 
                                    if pd.isna(row.get('Province')) else row['Province'],
                                    axis=1
                                )
                    
                    df_region['Region'] = df_region['Province'].apply(get_region)
                    
                    # หากลุ่มสาขา (ใช้ Booking No. เป็นหลัก)
                    def find_paired_branches(code, code_province, df_data):
                        paired = set()
                        
                        # หา Booking No. ของสาขานี้
                        code_rows = df_data[df_data['Code'] == code]
                        if len(code_rows) == 0:
                            return paired
                        
                        # เช็คว่ามีคอลัมน์ Booking หรือไม่
                        if 'Booking' not in df_data.columns and 'Trip' not in df_data.columns:
                            return paired
                        
                        booking_col = 'Booking' if 'Booking' in df_data.columns else 'Trip'
                        code_bookings = set(code_rows[booking_col].dropna().astype(str))
                        
                        if not code_bookings:
                            return paired
                        
                        # หาสาขาอื่นที่อยู่ Booking เดียวกัน (ไม่สนจังหวัด)
                        for booking in code_bookings:
                            if booking == 'nan' or not booking.strip():
                                continue
                            
                            same_booking = df_data[df_data[booking_col].astype(str) == booking]
                            for _, other_row in same_booking.iterrows():
                                other_code = other_row['Code']
                                
                                # เงื่อนไข: Booking เดียวกัน = รวมกลุ่ม (ไม่สนจังหวัด)
                                if other_code != code:
                                    paired.add(other_code)
                        
                        return paired
                    
                    all_codes_set = set(df_region['Code'].unique())
                    
                    # สร้างกลุ่มสาขาแบบ Union-Find (ตามลำดับ: ตำบล → อำเภอ → จังหวัด)
                    # Step 1: เริ่มจากแต่ละสาขาเป็นกลุ่มๆ พร้อมข้อมูล Master
                    initial_groups = {}
                    for code in all_codes_set:
                        # ดึงข้อมูลจาก Master
                        location = {}
                        if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
                            master_row = MASTER_DATA[MASTER_DATA['Plan Code'] == code]
                            if len(master_row) > 0:
                                master_row = master_row.iloc[0]
                                location = {
                                    'subdistrict': master_row.get('ตำบล', ''),
                                    'district': master_row.get('อำเภอ', ''),
                                    'province': master_row.get('จังหวัด', 'UNKNOWN'),
                                    'lat': master_row.get('ละติจูด', 0),
                                    'lon': master_row.get('ลองติจูด', 0)
                                }
                        
                        # ถ้าไม่มีใน Master ลองดึงจากไฟล์อัปโหลด
                        if not location or location.get('province', 'UNKNOWN') == 'UNKNOWN':
                            c_row = df_region[df_region['Code'] == code].iloc[0] if len(df_region[df_region['Code'] == code]) > 0 else None
                            if c_row is not None:
                                location = {
                                    'subdistrict': '',
                                    'district': '',
                                    'province': c_row.get('Province', 'UNKNOWN'),
                                    'lat': 0,
                                    'lon': 0
                                }
                        
                        if location:
                            initial_groups[(code,)] = {code: location}
                    
                    # ใช้ initial_groups แทน booking_groups
                    booking_groups = initial_groups
                    
                    # Step 2: รวมกลุ่มตามลำดับ ตำบล → อำเภอ → จังหวัด
                    def groups_can_merge(locs1, locs2):
                        """เช็คว่า 2 กลุ่มควรรวมกันไหม (ตามลำดับความละเอียด)"""
                        # 1. เช็คตำบลเดียวกัน (ต้องมีข้อมูลตำบล)
                        subdistricts1 = set(loc.get('subdistrict', '') for loc in locs1.values() if loc.get('subdistrict', ''))
                        subdistricts2 = set(loc.get('subdistrict', '') for loc in locs2.values() if loc.get('subdistrict', ''))
                        if subdistricts1 and subdistricts2 and (subdistricts1 & subdistricts2):
                            return True, 'ตำบล'
                        
                        # 2. เช็คอำเภอเดียวกัน (ต้องมีข้อมูลอำเภอและจังหวัดเดียวกัน)
                        districts1 = {(loc.get('district', ''), loc.get('province', '')) for loc in locs1.values() if loc.get('district', '')}
                        districts2 = {(loc.get('district', ''), loc.get('province', '')) for loc in locs2.values() if loc.get('district', '')}
                        if districts1 and districts2:
                            # เช็คว่ามีอำเภอและจังหวัดตรงกัน
                            for d1, p1 in districts1:
                                for d2, p2 in districts2:
                                    if d1 == d2 and p1 == p2 and p1:
                                        return True, 'อำเภอ'
                        
                        # 3. เช็คจังหวัดเดียวกัน
                        provinces1 = set(loc.get('province', '') for loc in locs1.values() if loc.get('province', ''))
                        provinces2 = set(loc.get('province', '') for loc in locs2.values() if loc.get('province', ''))
                        if provinces1 & provinces2:
                            return True, 'จังหวัด'
                        
                        return False, None
                    
                    merged_groups = []
                    used_groups = set()
                    
                    for group1, locs1 in booking_groups.items():
                        if group1 in used_groups:
                            continue
                        
                        merged_codes = set(group1)
                        merged_locs = locs1.copy()
                        used_groups.add(group1)
                        
                        # หากลุ่มอื่นที่ใกล้เคียง
                        changed = True
                        while changed:
                            changed = False
                            for group2, locs2 in booking_groups.items():
                                if group2 in used_groups:
                                    continue
                                can_merge, level = groups_can_merge(merged_locs, locs2)
                                if can_merge:
                                    merged_codes |= set(group2)
                                    merged_locs.update(locs2)
                                    used_groups.add(group2)
                                    changed = True
                        
                        merged_groups.append({
                            'codes': merged_codes,
                            'locations': merged_locs
                        })
                    
                    # Step 3: แปลงเป็น groups format
                    groups = []
                    for mg in merged_groups:
                        rep_code = list(mg['codes'])[0]
                        rep_row = df_region[df_region['Code'] == rep_code].iloc[0]
                        # กรองเฉพาะจังหวัดที่ไม่ใช่ UNKNOWN และไม่เป็น NaN
                        provinces = set(
                            str(loc.get('province', '')).strip() 
                            for loc in mg['locations'].values() 
                            if loc.get('province') and str(loc.get('province', '')).strip() not in ['UNKNOWN', 'nan', '']
                        )
                        
                        # ถ้าไม่มีจังหวัดเลย ใส่ "ไม่ระบุ"
                        province_str = ', '.join(sorted(provinces)) if provinces else 'ไม่ระบุ'
                        
                        groups.append({
                            'codes': mg['codes'],
                            'region': rep_row.get('Region', 'ไม่ระบุ'),
                            'province': province_str
                        })
                    
                    # แสดงสถิติ
                    st.markdown("---")
                    st.markdown("### 📊 สรุปการจัดกลุ่ม")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📍 จำนวนสาขา", df_region['Code'].nunique())
                    with col2:
                        st.metric("🗂️ จำนวนกลุ่ม", len(groups))
                    with col3:
                        regions_count = df_region['Region'].nunique()
                        st.metric("🗺️ จำนวนภาค", regions_count)
                    
                    # แสดงตามภาค
                    st.markdown("---")
                    st.markdown("### 🗺️ สาขาแยกตามภาค")
                    
                    region_summary = df_region.groupby('Region').agg({
                        'Code': 'nunique',
                        'Weight': 'sum',
                        'Cube': 'sum'
                    }).reset_index()
                    region_summary.columns = ['ภาค', 'จำนวนสาขา', 'น้ำหนักรวม', 'คิวรวม']
                    st.dataframe(region_summary, use_container_width=True)
                    
                    # แสดงรายละเอียดแต่ละภาค
                    for region in sorted(df_region['Region'].unique()):
                        region_data = df_region[df_region['Region'] == region]
                        with st.expander(f"📍 {region} ({region_data['Code'].nunique()} สาขา)"):
                            display_cols = ['Code', 'Name', 'Province', 'Weight', 'Cube']
                            display_cols = [c for c in display_cols if c in region_data.columns]
                            
                            region_display = region_data[display_cols].drop_duplicates('Code')
                            col_names = {'Code': 'รหัส', 'Name': 'ชื่อสาขา', 'Province': 'จังหวัด', 'Weight': 'น้ำหนัก', 'Cube': 'คิว'}
                            region_display.columns = [col_names.get(c, c) for c in display_cols]
                            st.dataframe(region_display, use_container_width=True)
                    
                    # แสดงกลุ่มสาขาที่เคยไปด้วยกัน
                    st.markdown("---")
                    st.markdown("### 🔗 กลุ่มสาขาที่เคยไปด้วยกัน (จากประวัติ)")
                    
                    paired_groups = [g for g in groups if len(g['codes']) > 1]
                    if paired_groups:
                        for i, group in enumerate(paired_groups, 1):
                            codes_list = list(group['codes'])
                            names = []
                            for c in codes_list:
                                name_row = df_region[df_region['Code'] == c]
                                if len(name_row) > 0 and 'Name' in name_row.columns:
                                    names.append(f"{c} ({name_row['Name'].iloc[0]})")
                                else:
                                    names.append(c)
                            
                            st.write(f"**กลุ่ม {i}** - {group['region']}: {', '.join(names)}")
                    else:
                        st.info("ไม่พบกลุ่มสาขาที่เคยไปด้วยกันในรายการนี้")
                    
                    # ดาวน์โหลด
                    st.markdown("---")
                    output_region = io.BytesIO()
                    with pd.ExcelWriter(output_region, engine='xlsxwriter') as writer:
                        df_region.to_excel(writer, sheet_name='สาขาทั้งหมด', index=False)
                        region_summary.to_excel(writer, sheet_name='สรุปตามภาค', index=False)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="📥 ดาวน์โหลดข้อมูลจัดกลุ่ม (Excel)",
                            data=output_region.getvalue(),
                            file_name=f"จัดกลุ่มสาขา_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )

if __name__ == "__main__":
    main()
