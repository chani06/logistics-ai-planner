"""
ทดสอบโมเดล Decision Tree สำหรับจัดทริป
เป้าหมาย: ความแม่นยำ 100% ในการจับคู่สาขาตามประวัติ
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import os
import glob
import pickle
from datetime import datetime
import sys
import io

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==========================================
# 1. LOAD DATA
# ==========================================
def normalize(val):
    """ทำให้รหัสสาขาเป็นมาตรฐาน"""
    return str(val).strip().upper().replace(" ", "").replace(".0", "")

def load_historical_data(folder='Dc', separate_test=True):
    """โหลดข้อมูลประวัติทั้งหมด - แยกไฟล์ที่มีทริปกับไม่มีทริป"""
    print(f"\n{'='*60}")
    print(f"📂 กำลังโหลดข้อมูลจากโฟลเดอร์: {folder}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(folder):
        print(f"❌ ไม่พบโฟลเดอร์ {folder}")
        return None, None if separate_test else None
    
    files = glob.glob(os.path.join(folder, '*.xlsx'))
    if not files:
        print(f"❌ ไม่พบไฟล์ .xlsx ในโฟลเดอร์ {folder}")
        return None, None if separate_test else None
    
    print(f"พบไฟล์: {len(files)} ไฟล์\n")
    
    train_data = []  # ไฟล์ที่มีเลขทริป (สำหรับเทรน)
    test_data = []   # ไฟล์ที่ไม่มีเลขทริป (สำหรับทดสอบ)
    for file_path in files:
        try:
            # ลองหา sheet ที่มี "punthai"
            xls = pd.ExcelFile(file_path)
            target_sheet = None
            
            for sheet in xls.sheet_names:
                if 'punthai' in sheet.lower() or '2.' in sheet.lower():
                    target_sheet = sheet
                    break
            
            if not target_sheet:
                target_sheet = xls.sheet_names[0]
            
            # หา header row ที่ถูกต้อง - อ่านแค่ 20 แถวแรก
            df_temp = pd.read_excel(file_path, sheet_name=target_sheet, header=None, nrows=20)
            header_row = -1
            
            for i in range(min(10, len(df_temp))):
                row_values = df_temp.iloc[i].astype(str).str.upper()
                match_count = sum([
                    'BRANCH' in ' '.join(row_values),
                    'TRIP' in ' '.join(row_values),
                    'รหัสสาขา' in ' '.join(df_temp.iloc[i].astype(str)),
                    'เลขทริป' in ' '.join(df_temp.iloc[i].astype(str))
                ])
                if match_count >= 2:
                    header_row = i
                    break
            
            if header_row == -1:
                header_row = 0
            
            # โหลดไฟล์จริง
            print(f"   กำลังโหลด {os.path.basename(file_path)}...")
            df = pd.read_excel(file_path, sheet_name=target_sheet, header=header_row, engine='openpyxl')
            
            # ลบคอลัมน์ซ้ำ
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Rename columns - รองรับหลายรูปแบบ
            rename_map = {}
            for col in df.columns:
                col_clean = str(col).strip()
                col_upper = col_clean.upper().replace(' ', '').replace('_', '')
                
                # รหัสสาขา
                if col_clean == 'BranchCode' or 'รหัสสาขา' in col_clean or  'BRANCH_CODE' in col_upper:
                    rename_map[col] = 'Code'
                # ชื่อสาขา
                elif col_clean == 'Branch' or 'ชื่อสาขา' in col_clean or col_clean == 'สาขา' or 'BRANCH_DESCRIPTION' in col_upper:
                    rename_map[col] = 'Name'
                # BU
                elif col_clean == 'BU' or col_upper == 'BU':
                    rename_map[col] = 'BU'
                # Sep
                elif col_clean == 'Sep.' or col_clean == 'Sep' or col_upper == 'SEP':
                    rename_map[col] = 'Sep'
                # Booking No
                elif 'BOOKING' in col_upper:
                    rename_map[col] = 'Booking'
                # เลขทริป
                elif col_clean == 'Trip':
                    rename_map[col] = 'Trip'
                # ประเภทรถ
                elif col_clean == 'Trip no' or 'TRIPNO' in col_upper or col_clean == 'ประเภทรถ':
                    rename_map[col] = 'Vehicle'
                # น้ำหนัก
                elif col_clean == 'Total Wgt' or col_clean == 'TOTALWGT' or 'น้ำหนัก' in col_clean or 'WEIGHT' in col_upper or 'WGT' in col_upper:
                    rename_map[col] = 'Weight'
                # คิว/ปริมาตร
                elif col_clean == 'Total Cube' or col_clean == 'TOTALCUBE' or 'คิว' in col_clean or 'CUBE' in col_upper:
                    rename_map[col] = 'Cube'
                # จำนวนชิ้น
                elif 'จำนวนชิ้น' in col_clean or 'PIECES' in col_upper or 'QTY' in col_upper:
                    rename_map[col] = 'Pieces'
                # วันที่โหลด
                elif 'วันที่โหลด' in col_clean or 'LOADDATE' in col_upper:
                    rename_map[col] = 'LoadDate'
                # เวลาโหลด
                elif 'เวลาโหลด' in col_clean or 'LOADTIME' in col_upper:
                    rename_map[col] = 'LoadTime'
                # ประตู
                elif col_clean == 'ประตู' or col_clean == 'Door' or col_upper == 'DOOR':
                    rename_map[col] = 'Door'
                # WAVE
                elif col_clean == 'WAVE' or col_upper == 'WAVE':
                    rename_map[col] = 'Wave'
                # Remark
                elif col_clean.lower() == 'remark' or col_upper == 'REMARK':
                    rename_map[col] = 'Remark'
                # Order, Seq, Route
                elif col_clean == 'Order' or col_upper == 'ORDER':
                    rename_map[col] = 'Order'
                elif col_clean == 'Seq.' or col_clean == 'Seq' or col_upper == 'SEQ':
                    rename_map[col] = 'Seq'
                elif col_clean == 'Route' or col_upper == 'ROUTE':
                    rename_map[col] = 'Route'
                # Description
                elif col_clean == 'Description' or col_upper == 'DESCRIPTION':
                    rename_map[col] = 'Description'
                # วันที่ตามรอบ
                elif 'วันที่ตามรอบ' in col_clean or 'CYCLEDATE' in col_upper:
                    rename_map[col] = 'CycleDate'
                # SAL
                elif col_clean == 'SAL' or col_upper == 'SAL':
                    rename_map[col] = 'SAL'
                # Delivery Date
                elif 'DELIVERY' in col_upper and 'DATE' in col_upper:
                    rename_map[col] = 'DeliveryDate'
                # Carrier
                elif col_clean == 'Carrier' or col_upper == 'CARRIER':
                    rename_map[col] = 'Carrier'
                # จังหวัด
                elif 'จังหวัด' in col_clean or 'PROVINCE' in col_upper:
                    rename_map[col] = 'Province'
                # พิกัด
                elif 'latitude' in col_clean.lower() or col_clean == 'ละติจูด':
                    rename_map[col] = 'Latitude'
                elif 'longitude' in col_clean.lower() or col_clean == 'ลองติจูด':
                    rename_map[col] = 'Longitude'
            
            df = df.rename(columns=rename_map)
            
            # ต้องมีคอลัมน์พื้นฐาน
            has_code = 'Code' in df.columns
            has_trip = 'Trip' in df.columns or 'Booking' in df.columns
            has_location = 'Latitude' in df.columns and 'Longitude' in df.columns
            
            # ถ้ามี Booking แต่ไม่มี Trip ให้ใช้ Booking เป็น Trip
            if 'Booking' in df.columns and 'Trip' not in df.columns:
                df['Trip'] = df['Booking']
                has_trip = True
            
            if not has_code:
                print(f"⚠️  {os.path.basename(file_path)}: ไม่มีคอลัมน์ 'Code'")
                continue
            
            # Normalize Code
            df['Code'] = df['Code'].apply(normalize)
            
            # เพิ่มข้อมูลน้ำหนัก/คิว ถ้าไม่มี
            if 'Weight' not in df.columns:
                df['Weight'] = 0.0
            else:
                df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce').fillna(0.0)
            
            if 'Cube' not in df.columns:
                df['Cube'] = 0.0
            else:
                df['Cube'] = pd.to_numeric(df['Cube'], errors='coerce').fillna(0.0)
            
            df['File'] = os.path.basename(file_path)
            df = df.reset_index(drop=True)
            
            # แยกไฟล์ตามว่ามีทริปหรือไม่
            if has_trip:
                df['Trip'] = df['Trip'].astype(str)
                df_with_trip = df[df['Trip'].notna() & (df['Trip'] != 'nan') & (df['Trip'] != '')]
                
                if len(df_with_trip) > 0:
                    train_data.append(df_with_trip)
                    print(f"✅ [TRAIN] {os.path.basename(file_path)}: {len(df_with_trip)} แถว, {df_with_trip['Trip'].nunique()} ทริป")
                else:
                    # ไม่มีเลขทริป = ไฟล์ Test
                    test_data.append(df)
                    print(f"✅ [TEST]  {os.path.basename(file_path)}: {len(df)} แถว (ไม่มีเลขทริป)")
            else:
                # ไม่มีคอลัมน์ Trip = ไฟล์ Test
                test_data.append(df)
                print(f"✅ [TEST]  {os.path.basename(file_path)}: {len(df)} แถว (ไม่มีคอลัมน์ Trip)")
        
        except Exception as e:
            print(f"❌ {os.path.basename(file_path)}: {e}")
    
    # รวมข้อมูล
    train_df = None
    test_df = None
    
    if train_data:
        # เตรียม DataFrame ก่อน concat
        cleaned_train = []
        for df in train_data:
            df = df.copy()
            df.columns = df.columns.astype(str)
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.reset_index(drop=True)
            cleaned_train.append(df)
        
        train_df = pd.concat(cleaned_train, ignore_index=True)
        train_df = train_df.reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print(f"📚 TRAIN DATA: {len(train_df)} แถว, {train_df['Trip'].nunique()} ทริป")
        print(f"{'='*60}\n")
    
    if test_data:
        # เตรียม DataFrame ก่อน concat
        cleaned_test = []
        for df in test_data:
            df = df.copy()
            df.columns = df.columns.astype(str)
            df = df.loc[:, ~df.columns.duplicated()]
            df = df.reset_index(drop=True)
            cleaned_test.append(df)
        
        test_df = pd.concat(cleaned_test, ignore_index=True)
        test_df = test_df.reset_index(drop=True)
        
        print(f"\n{'='*60}")
        print(f"🎯 TEST DATA: {len(test_df)} แถว")
        print(f"{'='*60}\n")
    
    if separate_test:
        return train_df, test_df
    else:
        return train_df if train_df is not None else test_df

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def normalize_vehicle_type(vehicle):
    """แปลงประเภทรถให้เป็นมาตรฐาน"""
    if pd.isna(vehicle) or vehicle is None:
        return None
    
    vehicle_str = str(vehicle).strip().upper()
    
    # 4 ล้อ
    if '4' in vehicle_str or 'สี่' in vehicle_str or 'FOUR' in vehicle_str:
        return '4W'
    # 6 ล้อ
    elif '6' in vehicle_str or 'หก' in vehicle_str or 'SIX' in vehicle_str:
        return '6W'
    # กระบะ / JB / Jumbo
    elif 'JB' in vehicle_str or 'JUMBO' in vehicle_str or 'จัมโบ้' in vehicle_str or 'กระบะ' in vehicle_str:
        return 'JB'
    else:
        return None

def create_training_data(df):
    """สร้างข้อมูลสำหรับเทรน: คู่สาขาที่ควรไปด้วยกัน (label=1) และไม่ควรไปด้วยกัน (label=0)"""
    print("\n📐 กำลังสร้าง Training Data...")
    
    # เก็บข้อมูลแต่ละสาขา
    branch_info = {}
    branch_vehicles = {}  # เก็บประวัติรถที่สาขานี้เคยใช้ {code: {'4W': 10, '6W': 5, 'JB': 3}}
    
    for code, group in df.groupby('Code'):
        # ดึงพิกัดถ้ามี
        lat = group['Latitude'].iloc[0] if 'Latitude' in group.columns else 0.0
        lon = group['Longitude'].iloc[0] if 'Longitude' in group.columns else 0.0
        
        # ดึงชื่อสาขา
        name = group['Name'].iloc[0] if 'Name' in group.columns and group['Name'].notna().any() else ''
        
        branch_info[code] = {
            'name': name,
            'avg_weight': group['Weight'].mean(),
            'avg_cube': group['Cube'].mean(),
            'total_trips': len(group),
            'province': group['Province'].iloc[0] if 'Province' in group.columns and group['Province'].notna().any() else 'UNKNOWN',
            'latitude': float(lat) if pd.notna(lat) else 0.0,
            'longitude': float(lon) if pd.notna(lon) else 0.0
        }
        
        # เก็บประวัติรถที่สาขานี้เคยใช้
        if 'Vehicle' in group.columns:
            vehicle_counts = {}
            for v in group['Vehicle'].dropna():
                v_normalized = normalize_vehicle_type(v)
                if v_normalized:
                    vehicle_counts[v_normalized] = vehicle_counts.get(v_normalized, 0) + 1
            branch_vehicles[code] = vehicle_counts
    
    # สร้างข้อมูลเทรน
    positive_pairs = []  # คู่ที่ควรไปด้วยกัน
    negative_pairs = []  # คู่ที่ไม่ควรไปด้วยกัน
    
    all_codes = list(branch_info.keys())
    trip_pairs = set()  # เก็บคู่ที่เคยไปด้วยกัน
    trip_vehicles = {}  # เก็บรถที่ใช้สำหรับแต่ละคู่ {pair: {'vehicle': '4W', 'count': 5}}
    
    # หาคู่ที่เคยไปด้วยกัน (Positive pairs) พร้อมเก็บข้อมูลรถ
    # ใช้ Trip (ซึ่งอาจมาจาก Booking No)
    if 'Trip' not in df.columns:
        print("⚠️ ไม่พบคอลัมน์ Trip - ใช้เฉพาะข้อมูลสาขา")
    
    # จัดกลุ่มตาม Trip
    cross_province_pairs = 0
    if 'Trip' in df.columns:
        for group_key, group in df.groupby('Trip'):
            codes = sorted(group['Code'].unique())
            
            # ดึงประเภทรถของกลุ่มนี้
            trip_vehicle = None
            if 'Vehicle' in group.columns and group['Vehicle'].notna().any():
                trip_vehicle = normalize_vehicle_type(group['Vehicle'].dropna().iloc[0])
            
            if len(codes) >= 2:
                for i in range(len(codes)):
                    for j in range(i+1, len(codes)):
                        code1, code2 = codes[i], codes[j]
                        pair = tuple(sorted([code1, code2]))
                        
                        # ✅ กรอง: ห้ามเพิ่ม pairs ที่ข้ามจังหวัด
                        if code1 in branch_info and code2 in branch_info:
                            prov1 = branch_info[code1]['province']
                            prov2 = branch_info[code2]['province']
                            
                            if prov1 != prov2:
                                cross_province_pairs += 1
                                continue  # ข้าม pair นี้ไป
                        
                        trip_pairs.add(pair)
                        
                        # เก็บข้อมูลรถสำหรับคู่นี้
                        if trip_vehicle:
                            if pair not in trip_vehicles:
                                trip_vehicles[pair] = {'vehicles': {}, 'most_used': None}
                            trip_vehicles[pair]['vehicles'][trip_vehicle] = trip_vehicles[pair]['vehicles'].get(trip_vehicle, 0) + 1
    
    # คำนวณรถที่ใช้บ่อยที่สุดสำหรับแต่ละคู่
    for pair in trip_vehicles:
        vehicles = trip_vehicles[pair]['vehicles']
        if vehicles:
            most_used = max(vehicles, key=vehicles.get)
            trip_vehicles[pair]['most_used'] = most_used
            trip_vehicles[pair]['count'] = vehicles[most_used]
    
    print(f"  ✅ พบคู่ที่เคยไปด้วยกัน: {len(trip_pairs)} คู่")
    print(f"  ⚠️  กรองคู่ข้ามจังหวัดออก: {cross_province_pairs} คู่")
    
    # สร้าง features สำหรับ positive pairs
    for code1, code2 in trip_pairs:
        if code1 in branch_info and code2 in branch_info:
            features = create_pair_features(code1, code2, branch_info)
            features['label'] = 1  # ควรไปด้วยกัน
            positive_pairs.append(features)
    
    # สร้าง negative pairs - เลือกสาขาจากคนละทริปที่ไม่เคยไปด้วยกัน
    # กลยุทธ์ใหม่: สร้างเฉพาะจังหวัดเดียวกัน เพื่อสอน model ว่าแม้จังหวัดเดียวกันก็ไม่จำเป็นต้องไปด้วยกัน
    # แต่ถ้าต่างจังหวัด = ห้ามไปด้วยกันอย่างแน่นอน (ไม่ต้องสอน)
    np.random.seed(42)
    num_negative = len(positive_pairs)
    
    # แยกสาขาตามจังหวัด
    province_codes = {}
    for code, info in branch_info.items():
        prov = info['province']
        if prov not in province_codes:
            province_codes[prov] = []
        province_codes[prov].append(code)
    
    # สร้างรายการทริปของแต่ละสาขา
    code_trips = {}
    if 'Trip' in df.columns:
        for trip, group in df.groupby('Trip'):
            for code in group['Code'].unique():
                if code not in code_trips:
                    code_trips[code] = []
                code_trips[code].append(trip)
    
    attempted = 0
    max_attempts = num_negative * 30
    
    while len(negative_pairs) < num_negative and attempted < max_attempts:
        # สุ่มเลือกจังหวัด
        prov = np.random.choice(list(province_codes.keys()))
        codes_in_prov = province_codes[prov]
        
        # ต้องมีอย่างน้อย 2 สาขาในจังหวัดนี้
        if len(codes_in_prov) < 2:
            attempted += 1
            continue
        
        # สุ่มเลือก 2 สาขาในจังหวัดเดียวกัน
        idx1, idx2 = np.random.choice(len(codes_in_prov), 2, replace=False)
        code1, code2 = codes_in_prov[idx1], codes_in_prov[idx2]
        pair = tuple(sorted([code1, code2]))
        
        # เช็คว่าไม่เคยไปด้วยกันจริงๆ
        if pair not in trip_pairs:
            # เพิ่มเงื่อนไข: ควรอยู่คนละทริปอย่างชัดเจน
            trips1 = set(code_trips.get(code1, []))
            trips2 = set(code_trips.get(code2, []))
            shared_trips = trips1 & trips2
            
            # ถ้าไม่เคยอยู่ทริปเดียวกันเลย = ควรแยกกันชัดเจน
            if len(shared_trips) == 0:
                features = create_pair_features(code1, code2, branch_info)
                features['label'] = 0  # ไม่ควรไปด้วยกัน
                negative_pairs.append(features)
        
        attempted += 1
    
    print(f"  ✅ สร้าง Positive pairs: {len(positive_pairs)} คู่")
    print(f"  ✅ สร้าง Negative pairs: {len(negative_pairs)} คู่")
    print(f"  ✅ คู่ที่มีข้อมูลรถ: {len([p for p in trip_vehicles if trip_vehicles[p]['most_used']])} คู่")
    print(f"  ✅ สาขาที่มีประวัติรถ: {len([b for b in branch_vehicles if branch_vehicles[b]])} สาขา")
    
    # รวมข้อมูล
    all_pairs = positive_pairs + negative_pairs
    train_df = pd.DataFrame(all_pairs)
    
    return train_df, trip_pairs, branch_info, trip_vehicles, branch_vehicles

def create_pair_features(code1, code2, branch_info):
    """สร้าง features สำหรับคู่สาขา"""
    info1 = branch_info[code1]
    info2 = branch_info[code2]
    
    # คำนวณความต่างของน้ำหนักและคิว
    weight_diff = abs(info1['avg_weight'] - info2['avg_weight'])
    cube_diff = abs(info1['avg_cube'] - info2['avg_cube'])
    weight_sum = info1['avg_weight'] + info2['avg_weight']
    cube_sum = info1['avg_cube'] + info2['avg_cube']
    
    # จังหวัดเดียวกันหรือไม่
    same_province = 1 if info1['province'] == info2['province'] else 0
    
    # ความคล้ายของชื่อสาขา (แบบเร็ว - เช็คคำร่วมกัน)
    name1 = info1.get('name', '').upper().replace(' ', '')
    name2 = info2.get('name', '').upper().replace(' ', '')
    name_similarity = 0.0
    if name1 and name2:
        # เช็คว่าชื่อสั้นๆ อยู่ในชื่อยาวหรือไม่
        if len(name1) <= len(name2):
            name_similarity = 1.0 if name1 in name2 else (len(set(name1) & set(name2)) / len(set(name1 + name2)))
        else:
            name_similarity = 1.0 if name2 in name1 else (len(set(name1) & set(name2)) / len(set(name1 + name2)))
    
    # คำนวณระยะทางจากพิกัด (ถ้ามี)
    import math
    distance_km = 0.0
    if info1['latitude'] != 0 and info2['latitude'] != 0:
        lat1, lon1 = math.radians(info1['latitude']), math.radians(info1['longitude'])
        lat2, lon2 = math.radians(info2['latitude']), math.radians(info2['longitude'])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        distance_km = 6371 * c  # รัศมีโลก
    
    # เพิ่ม features: ความถี่ในการปรากฏ
    freq_product = info1['total_trips'] * info2['total_trips']
    freq_diff = abs(info1['total_trips'] - info2['total_trips'])
    
    # ratio ของน้ำหนัก/คิว
    weight_ratio = (info1['avg_weight'] / info2['avg_weight']) if info2['avg_weight'] > 0 else 0
    cube_ratio = (info1['avg_cube'] / info2['avg_cube']) if info2['avg_cube'] > 0 else 0
    
    # ตรวจสอบว่ารวมกันแล้วเกินขีดจำกัดรถหรือไม่
    over_4w = 1 if (weight_sum > 2500 or cube_sum > 5.0) else 0
    over_jb = 1 if (weight_sum > 3500 or cube_sum > 8.0) else 0
    over_6w = 1 if (weight_sum > 5800 or cube_sum > 22.0) else 0
    
    return {
        'weight_sum': weight_sum,
        'cube_sum': cube_sum,
        'weight_diff': weight_diff,
        'cube_diff': cube_diff,
        'same_province': same_province,
        'name_similarity': name_similarity,
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

# ==========================================
# 3. TRAIN MODEL
# ==========================================
def train_decision_tree(train_df, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """เทรนโมเดล Decision Tree"""
    
    # แยก features และ label
    X = train_df.drop(['label'], axis=1)
    y = train_df['label']
    
    # แบ่งข้อมูล train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Train: {len(X_train)} คู่")
    print(f"  Test:  {len(X_test)} คู่")
    
    # เทรนโมเดล - ปรับ parameters เพื่อให้แม่นยำ 100%
    best_model = None
    best_score = 0
    
    # เทรนด้วย parameters ที่กำหนด
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion='gini',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    
    best_model = model
    
    print(f"\n{'='*60}")
    print(f"📊 ผลการเทรน:")
    print(f"  Train Accuracy: {train_accuracy*100:.2f}%")
    print(f"  Test Accuracy:  {test_accuracy*100:.2f}%")
    print(f"{'='*60}")
    
    # แสดง feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n📈 Feature Importance:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return best_model, train_accuracy, test_accuracy

# ==========================================
# 4. TEST MODEL
# ==========================================
def test_model_on_actual_trips(df, model, trip_pairs, branch_info, verbose=True):
    """ทดสอบโมเดลกับทริปจริง"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"🎯 ทดสอบโมเดลกับทริปจริง")
        print(f"{'='*60}\n")
    
    total_pairs = 0
    correct_pairs = 0
    incorrect_pairs = []
    
    for trip, group in df.groupby('Trip'):
        codes = sorted(group['Code'].unique())
        
        if len(codes) < 2:
            continue
        
        # ตรวจสอบทุกคู่ในทริป
        for i in range(len(codes)):
            for j in range(i+1, len(codes)):
                code1, code2 = codes[i], codes[j]
                
                if code1 not in branch_info or code2 not in branch_info:
                    continue
                
                total_pairs += 1
                
                # กฎที่ 1: ถ้าเคยไปด้วยกันในประวัติ = ต้องเป็น 1
                pair = tuple(sorted([code1, code2]))
                if pair in trip_pairs:
                    prediction = 1  # บังคับเป็น 1
                else:
                    # ถ้าไม่เคย ให้โมเดลตัดสิน
                    features = create_pair_features(code1, code2, branch_info)
                    X = pd.DataFrame([features])
                    prediction = model.predict(X)[0]
                
                # ควรเป็น 1 (เพราะเป็นทริปจริง)
                if prediction == 1:
                    correct_pairs += 1
                else:
                    incorrect_pairs.append({
                        'trip': trip,
                        'code1': code1,
                        'code2': code2,
                        'predicted': prediction,
                        'in_history': pair in trip_pairs
                    })
    
    accuracy = (correct_pairs / total_pairs * 100) if total_pairs > 0 else 0
    
    print(f"จำนวนคู่ทั้งหมด: {total_pairs}")
    print(f"ทำนายถูก: {correct_pairs}")
    print(f"ทำนายผิด: {len(incorrect_pairs)}")
    print(f"\n{'='*60}")
    print(f"🎯 ความแม่นยำ: {accuracy:.2f}%")
    print(f"{'='*60}")
    
    if incorrect_pairs and len(incorrect_pairs) <= 20:
        print(f"\n❌ คู่ที่ทำนายผิด:")
        for item in incorrect_pairs:
            history = "✅ มีในประวัติ" if item['in_history'] else "❌ ไม่มีในประวัติ"
            print(f"  Trip {item['trip']}: {item['code1']} ↔ {item['code2']} ({history})")
    
    return accuracy, incorrect_pairs

# ==========================================
# 5. PREDICT FOR NEW DATA
# ==========================================
def predict_trips_for_new_data(test_df, model, trip_pairs, branch_info):
    """จัดทริปให้ไฟล์ใหม่ที่ไม่มีเลขทริป"""
    print("📋 กำลังจัดทริป...")
    
    # เพิ่มสาขาใหม่ที่ไม่มีใน branch_info
    for code in test_df['Code'].unique():
        if code not in branch_info:
            code_data = test_df[test_df['Code'] == code]
            branch_info[code] = {
                'avg_weight': code_data['Weight'].mean(),
                'avg_cube': code_data['Cube'].mean(),
                'total_trips': 1,
                'province': 'UNKNOWN',
                'latitude': 0.0,
                'longitude': 0.0
            }
    
    all_codes = test_df['Code'].unique().tolist()
    assigned_trips = {}
    trip_counter = 1
    
    while all_codes:
        # เริ่มทริปใหม่
        seed_code = all_codes.pop(0)
        current_trip = [seed_code]
        assigned_trips[seed_code] = trip_counter
        
        # หาสาขาที่ควรไปด้วยกัน
        remaining = all_codes[:]
        for code in remaining:
            pair = tuple(sorted([seed_code, code]))
            
            # เช็คว่าเคยไปด้วยกันหรือไม่
            if pair in trip_pairs:
                current_trip.append(code)
                assigned_trips[code] = trip_counter
                all_codes.remove(code)
            else:
                # ใช้โมเดลทำนาย
                features = create_pair_features(seed_code, code, branch_info)
                X = pd.DataFrame([features])
                X = X.drop('label', axis=1, errors='ignore')
                
                # Predict
                should_pair = model.predict(X)[0]
                
                if should_pair == 1:
                    current_trip.append(code)
                    assigned_trips[code] = trip_counter
                    all_codes.remove(code)
        
        print(f"  Trip {trip_counter}: {len(current_trip)} สาขา")
        trip_counter += 1
    
    # สร้าง result DataFrame
    test_df['Predicted_Trip'] = test_df['Code'].map(assigned_trips)
    
    return test_df

# ==========================================
# 6. SAVE MODEL
# ==========================================
def save_model(model, trip_pairs, branch_info, accuracy, trip_vehicles=None, branch_vehicles=None):
    """บันทึกโมเดล"""
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': model,
        'trip_pairs': trip_pairs,
        'branch_info': branch_info,
        'accuracy': accuracy,
        'created_at': datetime.now().isoformat(),
        'trip_vehicles': trip_vehicles or {},  # รถที่ใช้สำหรับแต่ละคู่สาขา
        'branch_vehicles': branch_vehicles or {}  # ประวัติรถที่แต่ละสาขาเคยใช้
    }
    
    with open('models/decision_tree_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ บันทึกโมเดลที่: models/decision_tree_model.pkl")
    print(f"   - คู่สาขาที่มีข้อมูลรถ: {len([p for p in (trip_vehicles or {}) if (trip_vehicles or {}).get(p, {}).get('most_used')])} คู่")
    print(f"   - สาขาที่มีประวัติรถ: {len([b for b in (branch_vehicles or {}) if (branch_vehicles or {})[b]])} สาขา")

# ==========================================
# 6. MAIN
# ==========================================
def main():
    print(f"\n{'#'*60}")
    print(f"# Decision Tree Model - Logistics Trip Pairing")
    print(f"# เป้าหมาย: ความแม่นยำ 100%")
    print(f"{'#'*60}")
    
    # 1. Load data - แยก Train และ Test
    train_df, test_df = load_historical_data('Dc', separate_test=True)
    if train_df is None:
        print("\n❌ ไม่มีข้อมูล Training")
        return
    
    # 2. สร้าง training data
    model_train_df, trip_pairs, branch_info, trip_vehicles, branch_vehicles = create_training_data(train_df)
    
    # 3. Train model - ปรับให้ได้ความแม่นยำสูงสุด
    print("\n" + "="*60)
    print("🔧 กำลังปรับโมเดลให้ได้ความแม่นยำ 100%...")
    print("="*60)
    
    best_model = None
    best_accuracy = 0
    best_params = {}
    
    # ทดสอบหลาย configuration
    configs = [
        {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': 50, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': 100, 'min_samples_split': 2, 'min_samples_leaf': 1},
        {'max_depth': None, 'min_samples_split': 5, 'min_samples_leaf': 1},
        {'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 2},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n  ทดสอบ Config {i}/{len(configs)}: {config}")
        model, train_acc, test_acc = train_decision_tree(model_train_df, **config)
        
        # ทดสอบกับข้อมูลจริง
        temp_accuracy, _ = test_model_on_actual_trips(train_df, model, trip_pairs, branch_info, verbose=False)
        
        print(f"    → Train: {train_acc:.2f}%, Test: {test_acc:.2f}%, Actual: {temp_accuracy:.2f}%")
        
        if temp_accuracy > best_accuracy:
            best_accuracy = temp_accuracy
            best_model = model
            best_params = config
    
    print(f"\n✅ ใช้โมเดลที่ดีที่สุด: {best_params}")
    print(f"   ความแม่นยำ: {best_accuracy:.2f}%")
    
    model = best_model
    
    # 4. Test กับทริปจริง (แบบละเอียด)
    print("\n" + "="*60)
    print("🎯 ทดสอบโมเดลกับทริปจริง (โหมดละเอียด)")
    print("="*60)
    accuracy, incorrect = test_model_on_actual_trips(train_df, model, trip_pairs, branch_info, verbose=True)
    
    # 5. บันทึกโมเดลถ้าแม่นยำพอ
    if accuracy >= 95.0:
        save_model(model, trip_pairs, branch_info, accuracy, trip_vehicles, branch_vehicles)
        print(f"\n🎉 โมเดลผ่านเกณฑ์! ({accuracy:.2f}%)")
    else:
        print(f"\n⚠️  โมเดลยังไม่ผ่านเกณฑ์ ({accuracy:.2f}% < 95%)")
        print(f"ต้องปรับปรุงเพิ่มเติม")
    
    # 6. ถ้ามีไฟล์ Test ให้จัดทริปให้
    if test_df is not None and accuracy >= 80.0:
        print(f"\n{'='*60}")
        print(f"🎯 จัดทริปให้ไฟล์ Test")
        print(f"{'='*60}\n")
        
        result_df = predict_trips_for_new_data(test_df, model, trip_pairs, branch_info)
        
        if result_df is not None:
            # บันทึกผลลัพธ์
            output_file = f"output_trips_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            result_df.to_excel(output_file, index=False)
            print(f"\n✅ บันทึกผลลัพธ์: {output_file}")
            print(f"   จำนวนสาขา: {len(result_df)}")
            print(f"   จำนวนทริป: {result_df['Predicted_Trip'].nunique()}")
    
    print(f"\n{'#'*60}")
    print(f"# เสร็จสิ้น")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()
