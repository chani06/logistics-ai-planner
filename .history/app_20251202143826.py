import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import glob
import networkx as nx
from sklearn.cluster import DBSCAN
import math
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIG
# ==========================================
LIMITS = {'4W': {'max_w': 2500, 'max_c': 5.0}, 'JB': {'max_w': 3500, 'max_c': 8.0}, '6W': {'max_w': 5800, 'max_c': 22.0}}
BUFFER = 1.05
MAX_KM_CLUSTER = 30.0
TARGET_DROPS = 10
MAX_DROPS_FLEX = 12
NEARBY_RADIUS = 5.0
MAX_ZONE_DISTANCE = 100.0
STRICT_ZONE_MODE = True  # เปิดการเช็คโซนภาค
HISTORICAL_ONLY_MODE = True  # เปิดโหมดเข้มงวด - จัดเฉพาะที่มีประวัติ ไม่คาดเดา

# Utilization thresholds for truck optimization - เน้นใช้รถให้เต็ม 100%
MIN_CUBE_UTILIZATION = 0.95  # อย่างต่ำ 95% ก่อนปิดรถ (เพิ่มจาก 90%)
TARGET_CUBE_UTILIZATION = 1.00  # เป้าหมาย 100%
FLEX_CUBE_UTILIZATION = 1.05  # ยอมเกินได้ถึง 105%

EXCLUDE = ['PTDC', 'Distribution Center', 'DCวังน้อย', 'DC011']

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def normalize(val):
    return str(val).strip().upper().replace(" ", "").replace(".0", "")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def is_similar_name(name1, name2):
    """
    เช็คว่าชื่อสาขาคล้ายกันหรือไม่ โดยไม่สนใจ prefix (PTC, PUN, FC, etc.)
    เน้นดูชื่อพื้นที่หลักเท่านั้น
    """
    def clean(n):
        # ลบ prefix ทั้งหมดออก (PTC-MRT-, PUN-, FC, etc.)
        s = str(n)
        # ลบ prefix ที่พบบ่อย
        prefixes = ['PTC-MRT-', 'PTC-', 'PUN-', 'FC', 'MaxMart', 'MAXMART']
        for p in prefixes:
            s = s.replace(p, '')
        # เอาแค่ตัวอักษร
        return ''.join([c for c in s if c.isalpha() or c.isdigit()])
    
    clean1 = clean(name1)
    clean2 = clean(name2)
    
    # ต้องมีความยาวพอสมควร และตรงกันมากกว่า 70%
    if len(clean1) < 3 or len(clean2) < 3:
        return False
    
    # เช็คว่ามีส่วนที่เหมือนกันหรือไม่
    shorter = min(clean1, clean2, key=len)
    longer = max(clean1, clean2, key=len)
    
    return shorter in longer or clean1 == clean2

def get_province_zone(province):
    if not province or pd.isna(province):
        return 'UNKNOWN'
    
    prov = str(province).strip()
    
    central = ['กรุงเทพ', 'นนทบุรี', 'ปทุมธานี', 'สมุทรปราการ', 'สมุทรสาคร', 'นครปฐม', 
               'สมุทรสงคราม', 'ราชบุรี', 'กาญจนบุรี', 'สุพรรณบุรี', 'ชัยนาท', 'สิงห์บุรี', 
               'อ่างทอง', 'ลพบุรี', 'สระบุรี', 'อยุธยา', 'พระนครศรีอยุธยา']
    
    northeast = ['นครราชสีมา', 'โคราช', 'บุรีรัมย์', 'สุรินทร์', 'ศีขรภูมิ', 'ขอนแก่น', 
                 'อุดรธานี', 'เลย', 'หนองคาย', 'มหาสารคาม', 'ร้อยเอ็ด', 'กาฬสินธุ์', 
                 'สกลนคร', 'นครพนม', 'มุกดาหาร', 'ยโสธร', 'อำนาจเจริญ', 'อุบลราชธานี', 
                 'ชัยภูมิ', 'บึงกาฬ']
    
    north = ['เชียงใหม่', 'เชียงราย', 'ลำพูน', 'ลำปาง', 'พะเยา', 'แพร่', 'น่าน', 
             'อุตรดิตถ์', 'ตาก', 'สุโขทัย', 'พิษณุโลก', 'พิจิตร', 'เพชรบูรณ์', 'กำแพงเพชร']
    
    south = ['ชุมพร', 'สุราษฎร์ธานี', 'ระนอง', 'พังงา', 'ภูเก็ต', 'กระบี่', 'นครศรีธรรมราช', 
             'ตรัง', 'พัทลุง', 'สงขลา', 'สตูล', 'ปัตตานี', 'ยะลา', 'นราธิวาส']
    
    east = ['ฉะเชิงเทรา', 'ชลบุรี', 'ระยอง', 'จันทบุรี', 'ตราด', 'ปราจีนบุรี', 'สระแก้ว']
    
    west = ['กาญจนบุรี', 'ตาก', 'ประจวบคีรีขันธ์', 'เพชรบุรี']
    
    for p in central:
        if p in prov: return 'CENTRAL'
    for p in northeast:
        if p in prov: return 'NORTHEAST'
    for p in north:
        if p in prov: return 'NORTH'
    for p in south:
        if p in prov: return 'SOUTH'
    for p in east:
        if p in prov: return 'EAST'
    for p in west:
        if p in prov: return 'WEST'
    
    return 'UNKNOWN'

def is_same_zone(code1, code2, zone_map, geo):
    """เช็คว่า 2 สาขาอยู่ zone เดียวกันหรือไม่ - เข้มงวดมาก"""
    if not STRICT_ZONE_MODE:
        return True
    
    # เช็คภูมิภาคก่อน - ต้องเหมือนกัน 100%
    zone1 = zone_map.get(code1, 'UNKNOWN')
    zone2 = zone_map.get(code2, 'UNKNOWN')
    
    # ถ้ารู้ภูมิภาคทั้ง 2 ต้องเหมือนกัน ไม่งั้นห้ามรวม
    if zone1 != 'UNKNOWN' and zone2 != 'UNKNOWN':
        if zone1 != zone2:
            return False
    
    # ถ้าไม่รู้ภูมิภาคอย่างใดอย่างหนึ่ง เช็คระยะทาง
    if code1 in geo and code2 in geo:
        lat1, lon1 = geo[code1]
        lat2, lon2 = geo[code2]
        if lat1 != 0 and lat2 != 0:
            dist = haversine(lat1, lon1, lat2, lon2)
            if dist > MAX_ZONE_DISTANCE:
                return False
    else:
        # ถ้าไม่มีพิกัดและไม่รู้ภูมิภาค ห้ามรวม
        if zone1 == 'UNKNOWN' or zone2 == 'UNKNOWN':
            return False
    
    return True

# ==========================================
# 3. LOADERS & PROCESSORS
# ==========================================
def load_excel(content, sheet_name=None):
    try:
        xls = pd.ExcelFile(io.BytesIO(content))
        target_sheet = None
        
        # ถ้าระบุชื่อชีตเฉพาะ
        if sheet_name:
            if sheet_name in xls.sheet_names:
                target_sheet = sheet_name
            else:
                # ลองหาชีตที่มีชื่อคล้ายกัน
                for s in xls.sheet_names:
                    if sheet_name.lower() in s.lower():
                        target_sheet = s
                        break
        
        # ถ้ายังไม่เจอ ใช้ลำดับความสำคัญ
        if not target_sheet:
            priority = ['2.punthai', '2. punthai', '2.', 'punthai', 'order', 'history', 'data', 'sheet']
            
            for p in priority:
                for s in xls.sheet_names:
                    if p in s.lower(): 
                        target_sheet = s
                        break
                if target_sheet: break
        
        if not target_sheet: target_sheet = xls.sheet_names[0]
        
        # ค้นหา header row โดยดูหลายๆ คีย์เวิร์ด
        df_tmp = pd.read_excel(xls, sheet_name=target_sheet, nrows=30, header=None)
        h_row = -1
        
        keywords = ['CODE', 'BRANCH', 'สาขา', 'WGT', 'CUBE', 'คิว', 'น้ำหนัก', 
                   'TRIP', 'BOOKING', 'รหัส', 'ทริป', 'LAT', 'LON', 'VEHICLE']
        
        for i, r in df_tmp.iterrows():
            row_str = r.astype(str).str.upper().tolist()
            # นับจำนวนคีย์เวิร์ดที่พบในแถว
            match_count = sum(1 for k in keywords if any(k in s for s in row_str))
            if match_count >= 3:  # ถ้าพบอย่างน้อย 3 คีย์เวิร์ด = header
                h_row = i
                break
        
        if h_row == -1: h_row = 0  # ถ้าหาไม่เจอ ใช้แถวแรก
        
        df = pd.read_excel(xls, sheet_name=target_sheet, header=h_row)
        return df
    except Exception as e:
        st.error(f"❌ Error loading Excel sheet '{sheet_name}': {str(e)}")
        return None

def process_dataframe(df):
    if df is None: return None
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    rename_map = {}
    for c in df.columns:
        c_stripped = c.strip()
        cu = c.upper().replace(' ','').replace('_','')
        
        # ตรวจสอบชื่อคอลัมน์แบบตรงตัว (exact match) ก่อน
        if c_stripped == 'BranchCode':
            rename_map[c] = 'Code'
        elif c_stripped == 'Branch':
            rename_map[c] = 'Name'
        elif c_stripped == 'TOTALWGT':
            rename_map[c] = 'Wgt'
        elif c_stripped == 'TOTALCUBE':
            rename_map[c] = 'Cube'
        elif c_stripped == 'latitude' or c_stripped == ' latitude ':
            rename_map[c] = 'Lat'
        elif c_stripped == 'longitude':
            rename_map[c] = 'Lon'
        elif c_stripped == 'Trip':
            rename_map[c] = 'Trip'
        elif c_stripped == 'Trip no':
            rename_map[c] = 'Vehicle'
        elif c_stripped == 'จังหวัด':
            rename_map[c] = 'Province'
        # ถ้าไม่ตรงแบบ exact ให้ใช้ partial match
        elif 'BRANCHCODE' in cu or 'รหัสสาขา' in cu:
            rename_map[c] = 'Code'
        elif 'WGT' in cu or 'น้ำหนัก' in cu:
            rename_map[c] = 'Wgt'
        elif 'CUBE' in cu or 'คิว' in cu:
            rename_map[c] = 'Cube'
        elif 'LAT' in cu:
            rename_map[c] = 'Lat'
        elif 'LON' in cu:
            rename_map[c] = 'Lon'
        elif 'BOOKING' in cu:
            rename_map[c] = 'Trip'
        elif 'VEHICLE' in cu or 'TRIPNO' in cu:
            rename_map[c] = 'Vehicle'
    
    df.rename(columns=rename_map, inplace=True)
    
    # รีเซ็ต index เพื่อป้องกัน duplicate labels
    df = df.reset_index(drop=True)
    
    if 'Code' not in df.columns:
        if 'Name' in df.columns: df['Code'] = df['Name']
        else: return None
        
    df['Code'] = df['Code'].apply(normalize)
    for c in ['Wgt','Cube','Lat','Lon']:
        if c not in df.columns: df[c] = 0.0
        else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
    
    # ใช้ numpy array เพื่อหลีกเลี่ยงปัญหา duplicate index
    import numpy as np
    mask_to_keep = ~df['Code'].isin(EXCLUDE).values
    
    if 'Name' in df.columns:
        # สร้าง mask จาก Name โดยใช้ numpy array
        name_str = df['Name'].astype(str).values
        for exclude_key in EXCLUDE:
            name_mask = np.array([exclude_key not in s for s in name_str])
            mask_to_keep = mask_to_keep & name_mask
    
    # กรองข้อมูลโดยใช้ boolean indexing
    df = df[mask_to_keep].reset_index(drop=True)
    
    return df.copy()

def process_geo(df):
    if df is None: return {}
    # ไม่ต้อง process_dataframe อีกครั้ง เพราะ df ที่ส่งเข้ามาถูก process แล้ว
    geo = {}
    if df is not None and 'Code' in df.columns and 'Lat' in df.columns and 'Lon' in df.columns:
        for _, r in df.iterrows():
            if pd.notna(r['Lat']) and r['Lat'] != 0 and pd.notna(r['Code']):
                code = normalize(str(r['Code']))
                geo[code] = (float(r['Lat']), float(r['Lon']))
    return geo

# ==========================================
# 4. AI CORE
# ==========================================
def train_ai(df_list):
    G = nx.Graph()
    req = {}
    zones = {}
    regions = {}
    trip_distances = {}  # เก็บระยะทางของแต่ละทริป
    trip_patterns = []   # เก็บรูปแบบการจัด
    
    for df in df_list:
        if df is None or 'Trip' not in df.columns: continue
        
        # สร้าง copy และลบคอลัมน์ซ้ำ
        df = df.copy()
        df = df.loc[:, ~df.columns.duplicated()]
        
        # แก้ไขปัญหา Trip เป็น DataFrame
        if isinstance(df['Trip'], pd.DataFrame):
            df['Trip'] = df['Trip'].iloc[:,0]
        
        # แปลง Trip เป็น string และกรองข้อมูล
        df['Trip'] = df['Trip'].astype(str)
        df = df[(df['Trip'].notna()) & (df['Trip'] != 'nan') & (df['Trip'] != '') & (df['Trip'] != 'None')]
        
        if len(df) == 0:
            continue
        
        # เก็บข้อมูลจังหวัดและภูมิภาค
        for idx, r in df.iterrows():
            if 'Province' in df.columns and pd.notna(r['Province']):
                prov = str(r['Province']).strip()
                zones[r['Code']] = prov
                regions[r['Code']] = get_province_zone(prov)
        
        # วิเคราะห์รูปแบบการจัดทริป
        for t, g in df.groupby('Trip'):
            codes = g['Code'].unique()
            veh = str(g['Vehicle'].iloc[0]).upper() if 'Vehicle' in g.columns else ''
            # ตรวจจับประเภทรถจาก Trip no (4W, 6W, JB)
            rank = 3 if '6W' in veh or '6ล้อ' in veh else (2 if 'JB' in veh or 'จัมโบ' in veh else 1)
            
            # Debug: แสดงข้อมูล Trip
            if len(trip_patterns) < 5:  # แสดงแค่ 5 ทริปแรก
                print(f"[DEBUG] Trip {t}: {len(codes)} สาขา = {list(codes[:3])}... Vehicle={veh}")
            
            # บันทึก requirement ของแต่ละสาขา
            for c in codes: 
                req[c] = max(req.get(c,1), rank)
            
            # คำนวณระยะทางรวมของทริป (ถ้ามีพิกัด)
            if 'Lat' in g.columns and 'Lon' in g.columns:
                total_dist = 0
                coords = g[['Lat', 'Lon']].values
                for i in range(len(coords)-1):
                    if coords[i][0] != 0 and coords[i+1][0] != 0:
                        total_dist += haversine(coords[i][0], coords[i][1], 
                                               coords[i+1][0], coords[i+1][1])
                
                if total_dist > 0:
                    trip_distances[t] = total_dist
            
            # บันทึกรูปแบบการจัดทริป
            trip_info = {
                'trip': t,
                'branches': len(codes),
                'vehicle': veh,
                'weight': g['Wgt'].sum() if 'Wgt' in g.columns else 0,
                'cube': g['Cube'].sum() if 'Cube' in g.columns else 0,
                'codes': list(codes),
                'region': regions.get(codes[0], 'UNKNOWN') if len(codes) > 0 else 'UNKNOWN'
            }
            trip_patterns.append(trip_info)
            
            # สร้างกราฟความสัมพันธ์ (สาขาที่เคยไปด้วยกัน)
            if len(codes)>1:
                for i in range(len(codes)):
                    for j in range(i+1, len(codes)): 
                        G.add_edge(codes[i], codes[j])
                # Debug: แสดง edges ที่สร้าง
                if len(trip_patterns) < 5:
                    print(f"[DEBUG] สร้าง edges: {codes[0]} <-> {codes[1]} (และอื่นๆ {len(codes)} สาขา)")
            elif len(codes)==1: 
                G.add_node(codes[0])
    
    # สร้างสถิติการเรียนรู้
    learning_stats = {
        'total_trips': len(trip_patterns),
        'total_branches': len(req),
        'avg_drops': sum(p['branches'] for p in trip_patterns) / len(trip_patterns) if trip_patterns else 0,
        'avg_distance': sum(trip_distances.values()) / len(trip_distances) if trip_distances else 0,
        'region_distribution': {},
        'vehicle_usage': {}
    }
    
    # นับการกระจายตามภูมิภาค
    for pattern in trip_patterns:
        region = pattern['region']
        learning_stats['region_distribution'][region] = learning_stats['region_distribution'].get(region, 0) + 1
        
        veh = pattern['vehicle']
        if '6' in veh:
            veh_type = '6W'
        elif 'J' in veh or 'จัมโบ' in veh:
            veh_type = '4W-JB'
        else:
            veh_type = '4W'
        learning_stats['vehicle_usage'][veh_type] = learning_stats['vehicle_usage'].get(veh_type, 0) + 1
    
    return G, req, regions, learning_stats

def select_truck(w, c, min_rank, avg_distance=0, cube_utilization=0):
    """
    เลือกรถตามน้ำหนัก คิว และระยะทาง โดยเน้นความคุ้มค่าสูงสุด
    
    กลยุทธ์ใหม่ (เน้นใช้รถเล็กก่อน):
    1. ถ้าน้ำหนัก/คิวต่ำ → ใช้ 4W หรือ 4W Jumbo
    2. ถ้าจำเป็นต้องใช้ 6W → ต้องมี cube อย่างน้อย 15 cbm (68%+)
    3. ลดการใช้ 6W สำหรับงานเล็กๆ
    """
    
    # ถ้า requirement จากประวัติบอกว่าต้องใช้ 6W
    if min_rank >= 3:
        return '6 ล้อ ตู้ทึบ'
    
    # 4W ธรรมดา: น้ำหนัก ≤ 2500 kg และ คิว ≤ 5.0
    if w <= LIMITS['4W']['max_w'] and c <= LIMITS['4W']['max_c']:
        return '4 ล้อ ตู้ทึบ'
    
    # 4W จัมโบ้: น้ำหนัก ≤ 3500 kg และ คิว ≤ 8.0
    if w <= LIMITS['JB']['max_w'] and c <= LIMITS['JB']['max_c']:
        return '4 ล้อ จัมโบ้ ตู้ทึบ'
    
    # 6W: เฉพาะเมื่อจำเป็นจริงๆ
    # เงื่อนไขเข้มงวด: ต้องมี cube อย่างน้อย 15 cbm (68%) หรือน้ำหนักมาก
    if c >= 15.0 or w >= 4500:  # อย่างน้อย 15 cbm หรือ 4.5 ตัน
        return '6 ล้อ ตู้ทึบ'
    
    # ถ้าน้ำหนักเกิน 4W Jumbo แต่ cube ยังไม่ถึง 15 → พยายามใช้ 4W Jumbo
    if w > LIMITS['JB']['max_w'] and w <= 4000:
        return '4 ล้อ จัมโบ้ ตู้ทึบ'  # บังคับใช้ 4W Jumbo
    
    # Default: 6W (สำหรับกรณีที่น้ำหนักหนักมาก)
    return '6 ล้อ ตู้ทึบ'

def merge_small_trips(df_result, geo, region_map):
    """รวมทริปเล็กๆ (1-3 จุด) ที่มีน้ำหนักน้อยเข้าด้วยกัน แบบก้าวร้าว"""
    
    # คำนวณสถิติของแต่ละทริป
    trip_stats = df_result.groupby('Booking No').agg({
        'รหัสสาขา': 'count',
        'TOTALWGT': 'sum',
        'TOTALCUBE': 'sum'
    }).rename(columns={'รหัสสาขา': 'drops'})
    
    # หาทริปเล็กที่สามารถรวมได้ (≤ 3 จุด, น้ำหนัก < 2000 kg, คิว < 5.0)
    small_trips = trip_stats[(trip_stats['drops'] <= 3) & 
                            (trip_stats['TOTALWGT'] < 2000) & 
                            (trip_stats['TOTALCUBE'] < 5.0)].index.tolist()
    
    if not small_trips:
        return df_result
    
    # จัดกลุ่มทริปเล็กตามภูมิภาค
    trip_by_region = {}
    for trip_id in small_trips:
        trip_data = df_result[df_result['Booking No'] == trip_id]
        first_code = trip_data.iloc[0]['รหัสสาขา']
        region = region_map.get(first_code, 'UNKNOWN')
        
        if region not in trip_by_region:
            trip_by_region[region] = []
        
        # เก็บข้อมูลทริป
        trip_info = {
            'trip_id': trip_id,
            'data': trip_data,
            'weight': trip_stats.loc[trip_id, 'TOTALWGT'],
            'cube': trip_stats.loc[trip_id, 'TOTALCUBE'],
            'drops': trip_stats.loc[trip_id, 'drops'],
            'codes': trip_data['รหัสสาขา'].tolist()
        }
        trip_by_region[region].append(trip_info)
    
    # รวมทริปในแต่ละภูมิภาค
    new_rows = []
    merged_trips = set()
    
    for region, trips in trip_by_region.items():
        if len(trips) <= 1 or region == 'UNKNOWN':
            continue
        
        # เรียงตามน้ำหนักจากมากไปน้อย
        trips.sort(key=lambda x: x['cube'], reverse=True)
        
        i = 0
        while i < len(trips):
            if trips[i]['trip_id'] in merged_trips:
                i += 1
                continue
            
            # เริ่มรถใหม่ด้วยทริปปัจจุบัน
            current_group = [trips[i]]
            merged_trips.add(trips[i]['trip_id'])
            
            curr_w = trips[i]['weight']
            curr_c = trips[i]['cube']
            curr_drops = trips[i]['drops']
            
            # พยายามรวมทริปอื่นๆ เข้ามา
            j = i + 1
            while j < len(trips):
                if trips[j]['trip_id'] in merged_trips:
                    j += 1
                    continue
                
                new_w = curr_w + trips[j]['weight']
                new_c = curr_c + trips[j]['cube']
                new_drops = curr_drops + trips[j]['drops']
                
                # เช็คว่ารวมได้ไหม (ไม่เกิน 4W Jumbo หรือ 12 จุด)
                if new_w <= 3500 and new_c <= 8.0 and new_drops <= MAX_DROPS_FLEX:
                    current_group.append(trips[j])
                    merged_trips.add(trips[j]['trip_id'])
                    curr_w = new_w
                    curr_c = new_c
                    curr_drops = new_drops
                elif new_w <= 5800 and new_c <= 22.0 and new_drops <= MAX_DROPS_FLEX:
                    # ถ้าเกิน 4W Jumbo แต่ใส่ 6W ได้
                    current_group.append(trips[j])
                    merged_trips.add(trips[j]['trip_id'])
                    curr_w = new_w
                    curr_c = new_c
                    curr_drops = new_drops
                
                j += 1
            
            # บันทึกกลุ่มนี้
            if len(current_group) > 1:
                # รวมทริป
                for trip_info in current_group:
                    for _, row in trip_info['data'].iterrows():
                        row_dict = row.to_dict()
                        row_dict['Booking No'] = f"MERGED-{region}-{len(new_rows)}"
                        row_dict['Remark'] = f"Drops:{curr_drops}"
                        new_rows.append(row_dict)
            else:
                # ไม่รวม - เก็บทริปเดิม
                for _, row in current_group[0]['data'].iterrows():
                    new_rows.append(row.to_dict())
            
            i += 1
    
    # เก็บทริปที่ไม่ได้รวม
    for _, row in df_result.iterrows():
        if row['Booking No'] not in merged_trips:
            new_rows.append(row.to_dict())
    
    # สร้าง DataFrame ใหม่
    if new_rows:
        df_merged = pd.DataFrame(new_rows)
        
        # เรียงลำดับ Booking No ใหม่
        unique_bookings = sorted(df_merged['Booking No'].unique())
        booking_map = {old: f"AI-{i+1:03d}" for i, old in enumerate(unique_bookings)}
        df_merged['Booking No'] = df_merged['Booking No'].map(booking_map)
        
        return df_merged
    
    return df_result

def run_prediction(df_test, G, geo, constraints, region_map):
    # Debug: แสดง Code ที่จะ Test
    print(f"\n[DEBUG TEST] จำนวนสาขาที่จะทดสอบ: {len(df_test)}")
    print(f"[DEBUG TEST] Code ตัวอย่าง: {list(df_test['Code'].unique()[:5])}")
    print(f"[DEBUG TEST] Nodes ใน Graph: {G.number_of_nodes()}")
    print(f"[DEBUG TEST] Edges ใน Graph: {G.number_of_edges()}")
    
    df_test['Lat'] = df_test.apply(lambda r: geo.get(r['Code'],(0,0))[0] if r['Lat']==0 else r['Lat'], axis=1)
    df_test['Lon'] = df_test.apply(lambda r: geo.get(r['Code'],(0,0))[1] if r['Lon']==0 else r['Lon'], axis=1)
    
    # ถ้ามีข้อมูล Province ในไฟล์ Test ให้ใช้ข้อมูลนั้น ถ้าไม่มีให้ใช้จาก region_map
    if 'Province' in df_test.columns:
        df_test['Region'] = df_test.apply(
            lambda r: get_province_zone(r['Province']) if pd.notna(r['Province']) 
            else region_map.get(r['Code'], 'UNKNOWN'), 
            axis=1
        )
    else:
        df_test['Region'] = df_test['Code'].map(lambda x: region_map.get(x, 'UNKNOWN'))
    
    # โหมดเข้มงวด: จัดกลุ่มเฉพาะตามประวัติ
    if HISTORICAL_ONLY_MODE:
        # ให้แต่ละสาขาเป็น cluster แยก จะรวมเฉพาะตอนสร้างทริปถ้ามีประวัติ
        df_test['Cluster'] = df_test['Code'].map(lambda x: f"SINGLE-{x}")
    else:
        # โหมดยืดหยุ่น: ใช้ connected components
        hist_map = {n:i for i,c in enumerate(nx.connected_components(G)) for n in c}
        df_test['Cluster'] = df_test['Code'].map(lambda x: f"H-{hist_map[x]}" if x in hist_map else "UNK")
        
        # เพิ่มโซนภาคเข้าไปใน Cluster เพื่อแยกกลุ่มตามภูมิภาค
        if STRICT_ZONE_MODE:
            new_clusters = []
            for idx, row in df_test.iterrows():
                # ถ้ารู้จักภูมิภาค ให้เพิ่มเข้าไปใน cluster
                if row['Region'] != 'UNKNOWN':
                    new_clusters.append(f"{row['Cluster']}-{row['Region']}")
                else:
                    new_clusters.append(row['Cluster'])
            df_test['Cluster'] = new_clusters
    
    # จัดกลุ่มสาขาที่ไม่รู้จัก (UNK) - เฉพาะโหมดยืดหยุ่น
    if not HISTORICAL_ONLY_MODE:
        mask_unk = df_test['Cluster']=="UNK"
        mask_geo = (df_test['Lat']!=0) & mask_unk
        if mask_geo.any():
            coords = np.radians(df_test.loc[mask_geo, ['Lat','Lon']].values)
            db = DBSCAN(eps=MAX_KM_CLUSTER/6371.0, min_samples=1).fit(coords)
            df_test.loc[mask_geo, 'Cluster'] = [f"G-{x}" if x!=-1 else "NOISE" for x in db.labels_]
        
        # สำหรับสาขาที่ไม่มีพิกัด ให้จัดกลุ่มตาม prefix ของ Code
        mask_no_geo = (df_test['Lat']==0) & mask_unk
        if mask_no_geo.any():
            def get_code_prefix(code):
                # ดึง prefix จากรหัสสาขา (เช่น ZS, N, M, P)
                code_str = str(code)
                if len(code_str) >= 2:
                    # ถ้าขึ้นต้นด้วยตัวอักษร 2-3 ตัว
                    prefix = ''.join([c for c in code_str[:3] if c.isalpha()])
                    return f"PREFIX-{prefix}" if prefix else f"CODE-{code_str[:2]}"
                return f"SINGLE-{code_str}"
            
            df_test.loc[mask_no_geo, 'Cluster'] = df_test.loc[mask_no_geo, 'Code'].apply(get_code_prefix)
    
        mask_fin = df_test['Cluster'].isin(["UNK","NOISE"])
        if mask_fin.any():
            df_test.loc[mask_fin, 'Cluster'] = df_test.loc[mask_fin, 'Code'].map(
                lambda x: f"Z-{region_map.get(x, 'NEW')}" if x in region_map else f"NEW-{x}"
            )
    
    final_rows = []
    trip_cnt = 1
    
    for cid, group in df_test.groupby('Cluster'):
        pool = []
        for code, sub in group.groupby('Code'):
            pool.append({
                'Code': code, 'Name': sub.iloc[0]['Name'],
                'Wgt': sub['Wgt'].sum(), 'Cube': sub['Cube'].sum(),
                'Lat': sub.iloc[0]['Lat'], 'Lon': sub.iloc[0]['Lon']
            })
            
        while pool:
            pool.sort(key=lambda x: x['Cube'], reverse=True)
            current_truck = []
            seed = pool.pop(0)
            current_truck.append(seed)
            
            curr_w = seed['Wgt']
            curr_c = seed['Cube']
            last_lat = seed['Lat']
            last_lon = seed['Lon']
            last_name = seed['Name']
            drops = 1
            max_req = constraints.get(seed['Code'], 1)
            
            while True:
                best_idx = -1
                best_score = float('inf')
                best_is_same_name = False
                
                for i, cand in enumerate(pool):
                    # โหมดเข้มงวด: ต้องมีประวัติก่อน
                    if HISTORICAL_ONLY_MODE:
                        has_history = False
                        for truck_item in current_truck:
                            if G.has_edge(truck_item['Code'], cand['Code']):
                                has_history = True
                                break
                        if not has_history:
                            # Debug: แสดงว่าทำไมไม่จับคู่
                            if trip_cnt <= 3 and len(current_truck) == 1:  # แสดงแค่ 3 ทริปแรก
                                print(f"[DEBUG MATCH] ❌ {current_truck[0]['Code']} ไม่มีประวัติกับ {cand['Code']}")
                            continue
                    
                    if STRICT_ZONE_MODE:
                        if last_lat != 0 and cand['Lat'] != 0:
                            zone_dist = haversine(last_lat, last_lon, cand['Lat'], cand['Lon'])
                            if zone_dist > MAX_ZONE_DISTANCE:
                                continue
                        
                        if not is_same_zone(seed['Code'], cand['Code'], region_map, geo):
                            continue
                    
                    new_w = curr_w + cand['Wgt']
                    new_c = curr_c + cand['Cube']
                    
                    if new_w > 5800: continue
                    if new_c > 22.0 * BUFFER: continue
                    
                    is_same_name = is_similar_name(last_name, cand['Name'])
                    dist = haversine(last_lat, last_lon, cand['Lat'], cand['Lon']) if last_lat!=0 and cand['Lat']!=0 else 999
                    is_nearby = (dist <= NEARBY_RADIUS)
                    
                    # ผ่อนปรนกฎการจำกัด drops ถ้า utilization ยังต่ำมาก
                    current_util = curr_c / 22.0  # คำนวณ utilization สำหรับ 6W
                    
                    if drops >= TARGET_DROPS:
                        # ถ้า utilization < 70% ยังพอรับได้ถึง 12 drops
                        if drops >= MAX_DROPS_FLEX:
                            continue
                        # ถ้า utilization < 50% ให้ผ่อนปรนมากขึ้น
                        if current_util >= 0.50 and not (is_same_name or is_nearby):
                            continue
                    
                    # คำนวณคะแนน โดยเน้นชื่อเหมือนกันเป็นหลัก
                    score = dist
                    if is_same_name:
                        score -= 10000  # ลดคะแนนมากๆ สำหรับชื่อเหมือนกัน
                    
                    # ชื่อเหมือนกัน = ลำดับแรกเสมอ
                    is_better = False
                    if is_same_name and not best_is_same_name:
                        is_better = True  # ชื่อเหมือนกันชนะทุกกรณี
                    elif best_is_same_name and not is_same_name:
                        is_better = False  # ถ้า best เป็นชื่อเหมือนแล้ว ไม่เปลี่ยน
                    elif is_same_name and best_is_same_name:
                        is_better = (score < best_score)  # ทั้งคู่ชื่อเหมือน เลือกใกล้กว่า
                    else:
                        is_better = (score < best_score)  # ทั้งคู่ไม่เหมือน เลือกใกล้กว่า
                    
                    if is_better:
                        best_score = score
                        best_idx = i
                        best_is_same_name = is_same_name
                        
                if best_idx != -1:
                    sel = pool.pop(best_idx)
                    current_truck.append(sel)
                    
                    curr_w += sel['Wgt']
                    curr_c += sel['Cube']
                    drops += 1
                    
                    if sel['Lat']!=0: 
                        last_lat = sel['Lat']; last_lon = sel['Lon']
                    last_name = sel['Name']
                    max_req = max(max_req, constraints.get(sel['Code'], 1))
                else:
                    break
            
            # คำนวณระยะทางรวมของเส้นทาง
            total_distance = 0
            for i in range(len(current_truck) - 1):
                if current_truck[i]['Lat'] != 0 and current_truck[i+1]['Lat'] != 0:
                    total_distance += haversine(
                        current_truck[i]['Lat'], current_truck[i]['Lon'],
                        current_truck[i+1]['Lat'], current_truck[i+1]['Lon']
                    )
            avg_distance = total_distance / max(1, len(current_truck) - 1)
            
            # คำนวณ cube utilization ปัจจุบัน
            # ใช้ตามประเภทรถที่คาดว่าจะใช้
            if curr_w <= LIMITS['4W']['max_w'] and curr_c <= LIMITS['4W']['max_c']:
                cube_util = curr_c / LIMITS['4W']['max_c']
            elif curr_w <= LIMITS['JB']['max_w'] and curr_c <= LIMITS['JB']['max_c']:
                cube_util = curr_c / LIMITS['JB']['max_c']
            else:
                cube_util = curr_c / LIMITS['6W']['max_c']
            
            # ถ้า utilization ยังต่ำ (<90%) และยังมี pool เหลือ → พยายามใส่เพิ่ม
            # แต่ไม่เกิน MAX_DROPS_FLEX (12 จุด)
            # จำกัดการสแกนไม่เกิน 20 รอบเพื่อประหยัดเวลา
            max_scan = min(20, len(pool))
            if cube_util < MIN_CUBE_UTILIZATION and len(pool) > 0 and drops < MAX_DROPS_FLEX:
                # สแกนหาสาขาที่ใส่เพิ่มได้โดยไม่เกินขีดจำกัด
                # ใช้ while loop และตรวจสอบ index ที่ถูกต้อง
                i = 0
                scan_count = 0
                while i < len(pool) and cube_util < MIN_CUBE_UTILIZATION and drops < MAX_DROPS_FLEX and scan_count < max_scan:
                    scan_count += 1
                    cand = pool[i]
                    
                    # เช็คว่าใส่เพิ่มได้ไหม
                    new_w_test = curr_w + cand['Wgt']
                    new_c_test = curr_c + cand['Cube']
                    
                    can_add = False
                    
                    # ถ้าใส่แล้วยังไม่เกิน 6W limit และอยู่ใน zone เดียวกัน
                    if new_w_test <= 5800 and new_c_test <= 22.0 * FLEX_CUBE_UTILIZATION:
                        can_add = True
                        
                        # โหมดเข้มงวด: ต้องมีประวัติก่อน
                        if HISTORICAL_ONLY_MODE and can_add:
                            has_history = False
                            for truck_item in current_truck:
                                if G.has_edge(truck_item['Code'], cand['Code']):
                                    has_history = True
                                    break
                            if not has_history:
                                can_add = False
                        
                        # เช็ค zone อย่างเข้มงวด
                        if STRICT_ZONE_MODE and can_add:
                            # เช็คระยะทางจากจุดล่าสุด
                            if last_lat != 0 and cand['Lat'] != 0:
                                zone_dist = haversine(last_lat, last_lon, cand['Lat'], cand['Lon'])
                                if zone_dist > MAX_ZONE_DISTANCE:
                                    can_add = False
                            
                            # เช็คว่าอยู่ zone เดียวกันกับ seed (จุดแรกของรถ)
                            if can_add:
                                # ต้องอยู่ region เดียวกันกับ seed
                                seed_region = region_map.get(seed['Code'], 'UNKNOWN')
                                cand_region = region_map.get(cand['Code'], 'UNKNOWN')
                                
                                if seed_region != 'UNKNOWN' and cand_region != 'UNKNOWN':
                                    if seed_region != cand_region:
                                        can_add = False
                                
                                # เช็คระยะทางจาก seed ด้วย
                                if can_add and seed['Lat'] != 0 and cand['Lat'] != 0:
                                    seed_dist = haversine(seed['Lat'], seed['Lon'], cand['Lat'], cand['Lon'])
                                    if seed_dist > MAX_ZONE_DISTANCE:
                                        can_add = False
                    
                    if can_add:
                        # ใส่เพิ่ม - pop จาก index ปัจจุบัน
                        sel_extra = pool.pop(i)
                        current_truck.append(sel_extra)
                        curr_w = new_w_test
                        curr_c = new_c_test
                        drops += 1
                        
                        if sel_extra['Lat'] != 0:
                            last_lat = sel_extra['Lat']
                            last_lon = sel_extra['Lon']
                        last_name = sel_extra['Name']
                        max_req = max(max_req, constraints.get(sel_extra['Code'], 1))
                        
                        # คำนวณ utilization ใหม่
                        if curr_w <= LIMITS['4W']['max_w'] and curr_c <= LIMITS['4W']['max_c']:
                            cube_util = curr_c / LIMITS['4W']['max_c']
                        elif curr_w <= LIMITS['JB']['max_w'] and curr_c <= LIMITS['JB']['max_c']:
                            cube_util = curr_c / LIMITS['JB']['max_c']
                        else:
                            cube_util = curr_c / LIMITS['6W']['max_c']
                        
                        # ไม่เพิ่ม i เพราะ pop แล้ว item ถัดไปจะมาที่ index เดิม
                    else:
                        # ถ้าใส่ไม่ได้ ให้ข้ามไปตัวถัดไป
                        i += 1
            
            v_type = select_truck(curr_w, curr_c, max_req, avg_distance, cube_util)
            tid = f"AI-{trip_cnt:03d}"
            
            # หาภาคของทริปนี้ (ใช้ภาคของสาขาแรก)
            trip_region = region_map.get(current_truck[0]['Code'], 'UNKNOWN')
            
            for item in current_truck:
                item_region = region_map.get(item['Code'], 'UNKNOWN')
                final_rows.append({
                    'Booking No': tid, 'ประเภทรถ': v_type,
                    'รหัสสาขา': item['Code'], 'สาขา': item['Name'],
                    'ภาค': item_region,  # เพิ่มคอลัมน์ภาค
                    'TOTALWGT': item['Wgt'], 'TOTALCUBE': item['Cube'],
                    'Remark': f"Drops:{drops}", 'Lat': item['Lat'], 'Lon': item['Lon']
                })
            trip_cnt += 1
            
    return pd.DataFrame(final_rows)

def analyze_branch_groups(df_result, G):
    """
    วิเคราะห์กลุ่มสาขาจากผลลัพธ์และเทียบกับประวัติ
    ระบุว่ากลุ่มใดเคยไปด้วยกันในประวัติ (บางส่วนก็พอ)
    """
    branch_groups = []
    
    for booking_no, group in df_result.groupby('Booking No'):
        codes = group['รหัสสาขา'].tolist()
        names = group['สาขา'].tolist()
        
        # เช็คว่าสาขาในกลุ่มนี้เคยไปด้วยกันในประวัติหรือไม่
        # ผ่อนปรน: มีประวัติบางส่วนก็ถือว่ามี
        historical_match = False
        if len(codes) > 1:
            # นับจำนวนคู่ที่มีประวัติ
            total_pairs = 0
            paired_count = 0
            for i in range(len(codes)):
                for j in range(i+1, len(codes)):
                    total_pairs += 1
                    if G.has_edge(codes[i], codes[j]):
                        paired_count += 1
            
            # ถ้ามีประวัติอย่างน้อย 30% ของคู่ทั้งหมด ถือว่ามีประวัติ
            if paired_count > 0 and (paired_count / total_pairs) >= 0.3:
                historical_match = True
        
        # ดึงข้อมูลภาค
        regions_in_group = group['ภาค'].unique() if 'ภาค' in group.columns else ['UNKNOWN']
        region_text = ', '.join(regions_in_group) if len(regions_in_group) <= 3 else f"{regions_in_group[0]} +{len(regions_in_group)-1}"
        
        # สร้างข้อมูลกลุ่ม (เน้นชื่อสาขา)
        group_info = {
            'Booking No': booking_no,
            'ภาค': region_text,  # เพิ่มคอลัมน์ภาค
            'ประเภทรถ': group['ประเภทรถ'].iloc[0],
            'จำนวนสาขา': len(codes),
            'รายชื่อสาขา': ', '.join(names),  # แสดงชื่อแทนรหัส
            'รายชื่อสาขาเต็ม': '\n'.join([f"{name} ({code})" for code, name in zip(group['รหัสสาขา'], group['สาขา'])]),  # ชื่อก่อน รหัสหลัง
            'น้ำหนักรวม': group['TOTALWGT'].sum(),
            'คิวรวม': group['TOTALCUBE'].sum(),
            'Drops': group['Remark'].iloc[0] if 'Remark' in group.columns else f"Drops:{len(codes)}",
            'เคยไปด้วยกันในประวัติ': 'ใช่ ✓' if historical_match else 'ไม่ ✗'
        }
        
        branch_groups.append(group_info)
    
    return pd.DataFrame(branch_groups)

def export_styled_excel(df, filename, df_groups=None):
    try:
        import xlsxwriter
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        
        # แท็บ 1: กลุ่มสาขา (Branch Groups)
        if df_groups is not None:
            df_groups.to_excel(writer, index=False, sheet_name='Branch Groups')
            wb = writer.book
            ws_groups = writer.sheets['Branch Groups']
            
            # Format headers
            fmt_h = wb.add_format({
                'bold': True, 
                'bg_color': '#4472C4', 
                'font_color': 'white', 
                'border': 1,
                'text_wrap': True,
                'align': 'center',
                'valign': 'vcenter'
            })
            
            # Format for historical match (Yes)
            fmt_yes = wb.add_format({
                'bg_color': '#C6EFCE', 
                'font_color': '#006100',
                'border': 1,
                'align': 'center'
            })
            
            # Format for historical match (No)
            fmt_no = wb.add_format({
                'bg_color': '#FFC7CE', 
                'font_color': '#9C0006',
                'border': 1,
                'align': 'center'
            })
            
            # Format for normal cells
            fmt_normal = wb.add_format({'border': 1, 'text_wrap': True, 'valign': 'top'})
            fmt_number = wb.add_format({'border': 1, 'num_format': '#,##0.00'})
            
            # Write headers
            for c, val in enumerate(df_groups.columns):
                ws_groups.write(0, c, val, fmt_h)
            
            # Write data with conditional formatting
            for r, row in df_groups.iterrows():
                for c, (col_name, val) in enumerate(row.items()):
                    if col_name == 'เคยไปด้วยกันในประวัติ':
                        if 'ใช่' in str(val):
                            ws_groups.write(r+1, c, val, fmt_yes)
                        else:
                            ws_groups.write(r+1, c, val, fmt_no)
                    elif col_name in ['น้ำหนักรวม', 'คิวรวม']:
                        ws_groups.write(r+1, c, val, fmt_number)
                    else:
                        ws_groups.write(r+1, c, val, fmt_normal)
            
            # Adjust column widths
            ws_groups.set_column('A:A', 15)  # Booking No
            ws_groups.set_column('B:B', 18)  # ประเภทรถ
            ws_groups.set_column('C:C', 12)  # จำนวนสาขา
            ws_groups.set_column('D:D', 30)  # รายชื่อสาขา
            ws_groups.set_column('E:E', 40)  # รายชื่อสาขาเต็ม
            ws_groups.set_column('F:F', 15)  # น้ำหนักรวม
            ws_groups.set_column('G:G', 12)  # คิวรวม
            ws_groups.set_column('H:H', 15)  # Drops
            ws_groups.set_column('I:I', 20)  # เคยไปด้วยกันในประวัติ
        
        # แท็บ 2: ผลการจัดทริปแบบเต็ม (Plan)
        df.to_excel(writer, index=False, sheet_name='Plan')
        wb = writer.book
        ws = writer.sheets['Plan']
        
        fmt_h = wb.add_format({'bold': True, 'bg_color': '#4472C4', 'font_color': 'white', 'border': 1})
        fmt_1 = wb.add_format({'bg_color': '#FFFFFF', 'border': 1})
        fmt_2 = wb.add_format({'bg_color': '#D9D9D9', 'border': 1})
        
        for c, val in enumerate(df.columns):
            ws.write(0, c, val, fmt_h)
        
        curr = None
        toggle = False
        for r, row in df.iterrows():
            if row['Booking No'] != curr:
                toggle = not toggle
                curr = row['Booking No']
            fmt = fmt_1 if toggle else fmt_2
            for c, val in enumerate(row):
                ws.write(r+1, c, val, fmt)
        
        writer.close()
    except Exception as e:
        st.error(f"❌ Error exporting Excel: {str(e)}")
        # Fallback to simple export
        df.to_excel(filename, index=False)

# ==========================================
# 5. STREAMLIT UI
# ==========================================
def main():
    st.set_page_config(page_title="AI Logistics Planner", page_icon="🚚", layout="wide")
    
    st.title("🚚 AI Logistics Planner")
    st.markdown("---")
    
    # ตรวจสอบโฟลเดอร์ DC (ซ่อนไว้)
    dc_folder = os.path.join(os.getcwd(), 'DC')
    dc_files_found = []
    if os.path.exists(dc_folder):
        dc_files_found = glob.glob(os.path.join(dc_folder, '*.xlsx')) + glob.glob(os.path.join(dc_folder, '*.xls'))
    
    # File uploader - เฉพาะ Test
    st.subheader("🎯 อัปโหลดไฟล์ออเดอร์ (Test)")
    
    # เก็บข้อมูลไฟล์เก่าใน session state (ใช้ทั้งชื่อและขนาดไฟล์)
    if 'last_uploaded_info' not in st.session_state:
        st.session_state.last_uploaded_info = None
    if 'result_ready' not in st.session_state:
        st.session_state.result_ready = False
    
    test_file = st.file_uploader("เลือกไฟล์ Test ที่ต้องการวางแผน", type=['xlsx', 'xls'], key='test')
    
    # ตรวจสอบว่ามีการอัปโหลดไฟล์หรือไม่ (ทุกครั้งที่อัปโหลด)
    if test_file is not None:
        # สร้าง signature ของไฟล์จากชื่อ + ขนาด + เวลาปัจจุบัน
        current_file_info = f"{test_file.name}_{test_file.size}_{test_file.tell()}"
        
        # เคลียร์ข้อมูลเก่าทุกครั้งที่มีการอัปโหลด (แม้จะเป็นไฟล์ชื่อเดิม)
        if not st.session_state.result_ready or st.session_state.last_uploaded_info != current_file_info:
            st.session_state.last_uploaded_info = current_file_info
            st.session_state.result_ready = False
            st.cache_data.clear()
            st.success(f"✅ โหลดไฟล์: {test_file.name}")
    elif test_file is None:
        # ถ้าลบไฟล์ออก ให้เคลียร์ session
        if st.session_state.last_uploaded_info is not None:
            st.session_state.last_uploaded_info = None
            st.session_state.result_ready = False
            st.cache_data.clear()
    
    st.markdown("---")
    
    if st.button("🚀 เริ่มวางแผน", type="primary", use_container_width=True):
        if not test_file:
            st.error("❌ กรุณาอัปโหลดไฟล์ Test")
            return
        
        with st.spinner("⏳ กำลังประมวลผล..."):
            # Load training data จากโฟลเดอร์ DC (เงียบๆ)
            tr_dfs = []
            
            if dc_files_found:
                for dc_file_path in dc_files_found:
                    try:
                        with open(dc_file_path, 'rb') as f:
                            file_content = f.read()
                            train_df = process_dataframe(load_excel(file_content))
                            if train_df is not None:
                                tr_dfs.append(train_df)
                    except:
                        pass
            
            # เช็คว่าไฟล์ประวัติมีข้อมูลจังหวัดหรือไม่
            has_province_in_history = False
            for df in tr_dfs:
                if df is not None and 'Province' in df.columns:
                    if df['Province'].notna().sum() > 0:
                        has_province_in_history = True
                        break
            
            if not has_province_in_history:
                st.warning("⚠️ **ไม่พบข้อมูลจังหวัดในไฟล์ประวัติ!** กรุณาเพิ่มคอลัมน์ 'จังหวัด' ในไฟล์ประวัติเพื่อให้ระบบจัดกลุ่มตามภูมิภาคได้อย่างถูกต้อง")
            
            # Train AI
            G, const, regions, learning_stats = train_ai(tr_dfs)
            
            # Debug: แสดงข้อมูลการเรียนรู้
            st.info(f"📚 โหลดไฟล์ประวัติ: {len(tr_dfs)} ไฟล์")
            st.info(f"🔗 จำนวน edges ใน Graph: {G.number_of_edges()}")
            st.info(f"🏪 จำนวนสาขาที่รู้จัก: {G.number_of_nodes()}")
            
            known_regions = len([k for k, v in regions.items() if v != 'UNKNOWN'])
            if known_regions > 0:
                st.success(f"✅ จำนวนสาขาที่รู้จักภูมิภาค: {known_regions} สาขา")
            else:
                st.error(f"❌ ไม่มีสาขาที่รู้จักภูมิภาค! ระบบจะไม่สามารถแยกกลุ่มตามภาคได้")
            
            # แสดงสถิติการเรียนรู้ (ซ่อนไว้)
            with st.expander("📊 ข้อมูลที่เรียนรู้จากประวัติ", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🚚 จำนวนทริปที่เรียนรู้", f"{learning_stats['total_trips']}")
                with col2:
                    st.metric("🏪 จำนวนสาขาทั้งหมด", f"{learning_stats['total_branches']}")
                with col3:
                    st.metric("📍 จุดส่งเฉลี่ย/ทริป", f"{learning_stats['avg_drops']:.1f}")
                with col4:
                    st.metric("🗺️ ระยะทางเฉลี่ย", f"{learning_stats['avg_distance']:.0f} km")
                
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**🌏 การกระจายตามภูมิภาค:**")
                    for region, count in sorted(learning_stats['region_distribution'].items(), key=lambda x: x[1], reverse=True):
                        if region != 'UNKNOWN':
                            st.write(f"- {region}: {count} ทริป")
                
                with col2:
                    st.write("**🚛 การใช้รถตามประเภท:**")
                    for veh, count in sorted(learning_stats['vehicle_usage'].items(), key=lambda x: x[1], reverse=True):
                        st.write(f"- {veh}: {count} ทริป")
                
                st.info(f"💡 ระบบจะใช้ข้อมูลเหล่านี้ในการจัดกลุ่มสาขาที่เคยไปด้วยกัน และเลือกรถตามประวัติ")
            
            # Load geo - ใช้ข้อมูลจาก training files
            geo = {}
            for df in tr_dfs:
                if df is not None:
                    temp_geo = process_geo(df)
                    geo.update(temp_geo)
            
            # Process test data
            test_content = test_file.read()
            df_test = process_dataframe(load_excel(test_content))
            if df_test is None:
                st.error("❌ เกิดข้อผิดพลาดในการอ่านไฟล์ Test")
                return
            
            # Debug: เช็คว่ามีข้อมูลจังหวัดในไฟล์ Test หรือไม่
            if 'Province' in df_test.columns:
                prov_count = df_test['Province'].notna().sum()
                st.info(f"✅ พบข้อมูลจังหวัดในไฟล์ Test: {prov_count}/{len(df_test)} สาขา")
            else:
                st.warning("⚠️ ไม่พบคอลัมน์จังหวัดในไฟล์ Test → จะใช้ข้อมูลจากประวัติ")
            
            # ดึงพิกัดจากชีต Location ในไฟล์ Test (ถ้ามี)
            test_file.seek(0)  # reset file pointer
            df_location = load_excel(test_file.read(), sheet_name='Location')
            if df_location is not None:
                df_location_processed = process_dataframe(df_location)
                if df_location_processed is not None:
                    location_geo = process_geo(df_location_processed)
                    if location_geo:
                        geo.update(location_geo)
            
            # Run prediction
            res = run_prediction(df_test, G, geo, const, regions)
            
            # ปิด Post-processing: merge_small_trips เพื่อความเร็ว
            # res = merge_small_trips(res, geo, regions)
            
            res = res.sort_values(by=['Booking No', 'Lat'])
            
            # บันทึกสถานะว่าได้ผลลัพธ์แล้ว
            st.session_state.result_ready = True
            
            # Display results
            total_trips = res['Booking No'].nunique()
            trip_summary = res.groupby('Booking No').agg({
                'รหัสสาขา': 'count',
                'TOTALWGT': 'sum',
                'TOTALCUBE': 'sum'
            }).rename(columns={'รหัสสาขา': 'Drops'})
            
            st.markdown("---")
            st.success("### ✅ วางแผนเสร็จสิ้น!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🚚 จำนวนเที่ยว", f"{total_trips} เที่ยว")
            with col2:
                st.metric("📍 จุดส่งเฉลี่ย", f"{trip_summary['Drops'].mean():.1f} จุด/เที่ยว")
            with col3:
                st.metric("⚖️ น้ำหนักเฉลี่ย", f"{trip_summary['TOTALWGT'].mean():.0f} kg/เที่ยว")
            with col4:
                st.metric("📦 คิวเฉลี่ย", f"{trip_summary['TOTALCUBE'].mean():.2f} cbm/เที่ยว")
            
            # วิเคราะห์กลุ่มสาขา
            df_groups = analyze_branch_groups(res, G)
            
            # Display tabs
            st.markdown("---")
            tab1, tab2 = st.tabs(["📊 กลุ่มสาขา (Branch Groups)", "📋 ผลการจัดทริป (Plan)"])
            
            with tab1:
                st.subheader("📊 สรุปกลุ่มสาขาที่เคยไปด้วยกันในประวัติ")
                st.markdown("""
                **คำอธิบาย:**
                - แสดงเฉพาะกลุ่มสาขาที่มีประวัติไปด้วยกันในไฟล์เก่า
                - ระบบจะจัดทริปโดยอิงจากประวัติเท่านั้น (Historical-Based Routing)
                """)
                
                # กรองเฉพาะกลุ่มที่เคยไปด้วยกันในประวัติ
                df_groups_historical = df_groups[df_groups['เคยไปด้วยกันในประวัติ'].str.contains('ใช่')].copy()
                
                # แสดงสถิติ
                total_groups = len(df_groups)
                historical_groups = len(df_groups_historical)
                new_groups = total_groups - historical_groups
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📦 จำนวนกลุ่มทั้งหมด", total_groups)
                with col2:
                    st.metric("✅ เคยไปด้วยกันในประวัติ", historical_groups)
                with col3:
                    st.metric("🆕 การจับคู่ใหม่", new_groups)
                
                # แสดงตาราง (เฉพาะกลุ่มที่เคยไปด้วยกัน)
                if len(df_groups_historical) > 0:
                    st.dataframe(
                        df_groups_historical,
                        use_container_width=True,
                        height=400,
                    column_config={
                        'Booking No': st.column_config.TextColumn('Booking No', width='small'),
                        'ภาค': st.column_config.TextColumn('ภาค', width='small'),
                        'ประเภทรถ': st.column_config.TextColumn('ประเภทรถ', width='medium'),
                        'จำนวนสาขา': st.column_config.NumberColumn('จำนวนสาขา', width='small'),
                        'รายชื่อสาขา': st.column_config.TextColumn('รายชื่อสาขา', width='large'),
                        'รายชื่อสาขาเต็ม': st.column_config.TextColumn('รายชื่อสาขา (เต็ม)', width='large'),
                        'น้ำหนักรวม': st.column_config.NumberColumn('น้ำหนักรวม (kg)', format='%.2f'),
                        'คิวรวม': st.column_config.NumberColumn('คิวรวม (cbm)', format='%.2f'),
                        'Drops': st.column_config.TextColumn('Drops', width='small'),
                        'เคยไปด้วยกันในประวัติ': st.column_config.TextColumn('เคยไปด้วยกันในประวัติ', width='medium')
                    }
                )
                
                else:
                    st.warning("⚠️ ไม่พบกลุ่มสาขาที่เคยไปด้วยกันในประวัติ")
                
                # แสดงข้อมูลสาขาเดี่ยว (ถ้ามี)
                if new_groups > 0:
                    st.info(f"ℹ️ มี {new_groups} ทริปที่เป็นสาขาเดี่ยวหรือไม่มีประวัติ (ดูรายละเอียดในแท็บ Plan)")
                    
                    new_group_list = df_groups[df_groups['เคยไปด้วยกันในประวัติ'].str.contains('ไม่')]
                    with st.expander("📋 ดูรายละเอียดสาขาเดี่ยว/ไม่มีประวัติ"):
                        for _, row in new_group_list.iterrows():
                            st.markdown(f"""
                            **{row['Booking No']}** ({row['ประเภทรถ']})
                            - สาขา: {row['รายชื่อสาขา']}
                            - น้ำหนัก: {row['น้ำหนักรวม']:.2f} kg
                            - คิว: {row['คิวรวม']:.2f} cbm
                            """)
            
            with tab2:
                st.subheader("📋 ผลการจัดทริปแบบเต็ม")
                st.dataframe(res, use_container_width=True, height=400)
            
            # Export
            output_filename = 'AI_Sticky_Routing_Plan.xlsx'
            export_styled_excel(res, output_filename, df_groups)
            
            with open(output_filename, 'rb') as f:
                st.download_button(
                    label="💾 ดาวน์โหลดไฟล์ Excel (2 แท็บ)",
                    data=f,
                    file_name=output_filename,
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )
            
            st.balloons()

if __name__ == "__main__":
    main()
