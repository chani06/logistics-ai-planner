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
STRICT_ZONE_MODE = True
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
    def clean(n):
        return ''.join([c for c in str(n) if c.isalpha()])
    return clean(name1) == clean(name2) and len(clean(name1)) > 3

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
    if not STRICT_ZONE_MODE:
        return True
    
    if code1 in geo and code2 in geo:
        lat1, lon1 = geo[code1]
        lat2, lon2 = geo[code2]
        if lat1 != 0 and lat2 != 0:
            dist = haversine(lat1, lon1, lat2, lon2)
            if dist > MAX_ZONE_DISTANCE:
                return False
    
    zone1 = zone_map.get(code1, 'UNKNOWN')
    zone2 = zone_map.get(code2, 'UNKNOWN')
    
    if zone1 != 'UNKNOWN' and zone2 != 'UNKNOWN':
        if zone1 != zone2:
            return False
    
    return True

# ==========================================
# 3. LOADERS & PROCESSORS
# ==========================================
def load_excel(content):
    try:
        xls = pd.ExcelFile(io.BytesIO(content))
        target_sheet = None
        
        # ลำดับความสำคัญในการเลือกชีต: 2.Punthai > 2. > punthai > order > history
        priority = ['2.punthai', '2.', 'punthai', 'order', 'history']
        
        for p in priority:
            for s in xls.sheet_names:
                if p in s.lower(): 
                    target_sheet = s
                    break
            if target_sheet: break
        
        if not target_sheet: target_sheet = xls.sheet_names[0]
        
        df_tmp = pd.read_excel(xls, sheet_name=target_sheet, nrows=20, header=None)
        h_row = -1
        for i, r in df_tmp.iterrows():
            row_str = r.astype(str).str.upper().tolist()
            if sum(1 for k in ['CODE','BRANCH','สาขา','WGT'] if any(k in s for s in row_str)) >= 2:
                h_row = i; break
        if h_row == -1: h_row = 1
        return pd.read_excel(xls, sheet_name=target_sheet, header=h_row)
    except: return None

def process_dataframe(df):
    if df is None: return None
    df.columns = df.columns.astype(str).str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    rename_map = {}
    for c in df.columns:
        cu = c.upper().replace(' ','').replace('_','')
        if 'BRANCHCODE' in cu or 'รหัสสาขา' in cu: rename_map[c] = 'Code'
        elif 'BRANCH' in cu or 'ชื่อสาขา' in cu or 'สาขา'==c: rename_map[c] = 'Name'
        elif 'WGT' in cu or 'น้ำหนัก' in cu: rename_map[c] = 'Wgt'
        elif 'CUBE' in cu or 'คิว' in cu: rename_map[c] = 'Cube'
        elif 'LAT' in cu: rename_map[c] = 'Lat'
        elif 'LON' in cu: rename_map[c] = 'Lon'
        elif 'TRIP' in cu or 'BOOKING' in cu: rename_map[c] = 'Trip'
        elif 'VEHICLE' in cu or 'TRIPNO' in cu: rename_map[c] = 'Vehicle'
        elif 'จังหวัด' in cu: rename_map[c] = 'Province'
    
    df.rename(columns=rename_map, inplace=True)
    if 'Code' not in df.columns:
        if 'Name' in df.columns: df['Code'] = df['Name']
        else: return None
        
    df['Code'] = df['Code'].apply(normalize)
    for c in ['Wgt','Cube','Lat','Lon']:
        if c not in df.columns: df[c] = 0.0
        else: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        
    mask_ex = df['Code'].isin(EXCLUDE)
    if 'Name' in df.columns: mask_ex |= df['Name'].apply(lambda x: any(k in str(x) for k in EXCLUDE))
    return df[~mask_ex].copy()

def process_geo(df):
    if df is None: return {}
    df = process_dataframe(df)
    geo = {}
    if df is not None:
        for _, r in df.iterrows():
            if r['Lat']!=0: geo[r['Code']] = (r['Lat'], r['Lon'])
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
            rank = 3 if '6' in veh else (2 if 'J' in veh else 1)
            
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

def select_truck(w, c, min_rank):
    s = min_rank
    if s >= 3: return '6 ล้อ ตู้ทึบ'
    if s <= 1 and c <= LIMITS['4W']['max_c']*BUFFER and w <= LIMITS['4W']['max_w']: return '4 ล้อ ตู้ทึบ'
    if s <= 2 and c <= LIMITS['JB']['max_c']*BUFFER and w <= LIMITS['JB']['max_w']: return '4 ล้อ จัมโบ้ ตู้ทึบ'
    return '6 ล้อ ตู้ทึบ'

def run_prediction(df_test, G, geo, constraints, region_map):
    df_test['Lat'] = df_test.apply(lambda r: geo.get(r['Code'],(0,0))[0] if r['Lat']==0 else r['Lat'], axis=1)
    df_test['Lon'] = df_test.apply(lambda r: geo.get(r['Code'],(0,0))[1] if r['Lon']==0 else r['Lon'], axis=1)
    df_test['Region'] = df_test['Code'].map(lambda x: region_map.get(x, 'UNKNOWN'))
    
    hist_map = {n:i for i,c in enumerate(nx.connected_components(G)) for n in c}
    df_test['Cluster'] = df_test['Code'].map(lambda x: f"H-{hist_map[x]}" if x in hist_map else "UNK")
    
    if STRICT_ZONE_MODE:
        new_clusters = []
        for idx, row in df_test.iterrows():
            if row['Cluster'] != 'UNK' and row['Region'] != 'UNKNOWN':
                new_clusters.append(f"{row['Cluster']}-{row['Region']}")
            else:
                new_clusters.append(row['Cluster'])
        df_test['Cluster'] = new_clusters
    
    mask_unk = df_test['Cluster']=="UNK"
    if mask_unk.any():
        mask_geo = (df_test['Lat']!=0) & mask_unk
        if mask_geo.any():
            coords = np.radians(df_test.loc[mask_geo, ['Lat','Lon']].values)
            db = DBSCAN(eps=MAX_KM_CLUSTER/6371.0, min_samples=1).fit(coords)
            df_test.loc[mask_geo, 'Cluster'] = [f"G-{x}" if x!=-1 else "NOISE" for x in db.labels_]
    
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
                    
                    if drops >= TARGET_DROPS:
                        if drops >= MAX_DROPS_FLEX: 
                            continue
                        if not (is_same_name or is_nearby): 
                            continue
                    
                    score = dist
                    if is_same_name:
                        score -= 1000
                    
                    is_better = (score < best_score)
                    if is_same_name and not best_is_same_name:
                        is_better = True
                    elif best_is_same_name and not is_same_name:
                        is_better = False
                    
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
            
            v_type = select_truck(curr_w, curr_c, max_req)
            tid = f"AI-{trip_cnt:03d}"
            
            for item in current_truck:
                final_rows.append({
                    'Booking No': tid, 'ประเภทรถ': v_type,
                    'รหัสสาขา': item['Code'], 'สาขา': item['Name'],
                    'TOTALWGT': item['Wgt'], 'TOTALCUBE': item['Cube'],
                    'Remark': f"Drops:{drops}", 'Lat': item['Lat'], 'Lon': item['Lon']
                })
            trip_cnt += 1
            
    return pd.DataFrame(final_rows)

def export_styled_excel(df, filename):
    try:
        import xlsxwriter
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Plan')
        wb = writer.book; ws = writer.sheets['Plan']
        fmt_h = wb.add_format({'bold': True, 'bg_color': '#4472C4', 'font_color': 'white', 'border': 1})
        fmt_1 = wb.add_format({'bg_color': '#FFFFFF', 'border': 1})
        fmt_2 = wb.add_format({'bg_color': '#D9D9D9', 'border': 1})
        for c, val in enumerate(df.columns): ws.write(0, c, val, fmt_h)
        curr = None; toggle = False
        for r, row in df.iterrows():
            if row['Booking No'] != curr: toggle = not toggle; curr = row['Booking No']
            fmt = fmt_1 if toggle else fmt_2
            for c, val in enumerate(row): ws.write(r+1, c, val, fmt)
        writer.close()
    except:
        df.to_excel(filename, index=False)

# ==========================================
# 5. STREAMLIT UI
# ==========================================
def main():
    st.set_page_config(page_title="AI Logistics Planner", page_icon="🚚", layout="wide")
    
    st.title("🚚 AI Logistics Planner: Sticky Routing Edition")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("✨ **Sticky Routing**: ชื่อเหมือนกันไปก่อน + ใกล้กันไปก่อน")
    with col2:
        st.info("📦 **Drop Rules**: 1-10 ✓ | 11-12 (ชื่อเหมือน/ใกล้≤5km) ✓ | 13+ ✗")
    with col3:
        st.info("🌏 **Zone Filter**: Geofence 100km + Province/Region Aware")
    
    st.markdown("---")
    
    # ตรวจสอบโฟลเดอร์ DC
    dc_folder = os.path.join(os.getcwd(), 'DC')
    dc_files_found = []
    if os.path.exists(dc_folder):
        dc_files_found = glob.glob(os.path.join(dc_folder, '*.xlsx')) + glob.glob(os.path.join(dc_folder, '*.xls'))
    
    # แสดงข้อมูลไฟล์ที่พบในโฟลเดอร์ DC
    if dc_files_found:
        with st.expander(f"📂 พบไฟล์ใน DC/ : {len(dc_files_found)} ไฟล์", expanded=True):
            for f in dc_files_found:
                st.text(f"✓ {os.path.basename(f)}")
    else:
        st.warning("⚠️ ไม่พบโฟลเดอร์ 'DC/' หรือไม่มีไฟล์ Excel ในโฟลเดอร์")
    
    st.markdown("---")
    
    # File uploader - เฉพาะ Test
    st.subheader("🎯 อัปโหลดไฟล์ออเดอร์ (Test)")
    
    # เก็บไฟล์เก่าใน session state
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    
    test_file = st.file_uploader("เลือกไฟล์ Test ที่ต้องการวางแผน", type=['xlsx', 'xls'], key='test')
    
    # ตรวจสอบว่ามีการอัปโหลดไฟล์ใหม่หรือไม่
    if test_file is not None:
        current_file_name = test_file.name
        if st.session_state.last_uploaded_file != current_file_name:
            # เคลียร์ข้อมูลเก่าเมื่ออัปโหลดไฟล์ใหม่
            st.session_state.last_uploaded_file = current_file_name
            st.cache_data.clear()
            st.info(f"📄 ไฟล์ใหม่: {current_file_name}")
    elif test_file is None and st.session_state.last_uploaded_file is not None:
        # ถ้าลบไฟล์ออก ให้เคลียร์ session
        st.session_state.last_uploaded_file = None
        st.cache_data.clear()
    
    st.markdown("---")
    
    if st.button("🚀 เริ่มวางแผน", type="primary", use_container_width=True):
        if not test_file:
            st.error("❌ กรุณาอัปโหลดไฟล์ Test")
            return
        
        if not dc_files_found:
            st.error("❌ ไม่พบไฟล์ในโฟลเดอร์ DC/ กรุณาสร้างโฟลเดอร์ DC และวางไฟล์ประวัติไว้ในนั้น")
            return
        
        with st.spinner("⏳ กำลังประมวลผล..."):
            # Load training data จากโฟลเดอร์ DC
            tr_dfs = []
            
            st.info(f"📂 กำลังโหลดไฟล์จากโฟลเดอร์ DC/ ({len(dc_files_found)} ไฟล์)")
            
            for dc_file_path in dc_files_found:
                try:
                    with open(dc_file_path, 'rb') as f:
                        file_content = f.read()
                        train_df = process_dataframe(load_excel(file_content))
                        if train_df is not None:
                            tr_dfs.append(train_df)
                            st.success(f"✅ {os.path.basename(dc_file_path)}: {len(train_df)} รายการ")
                        else:
                            st.warning(f"⚠️ {os.path.basename(dc_file_path)}: ไม่สามารถประมวลผลได้")
                except Exception as e:
                    st.error(f"❌ {os.path.basename(dc_file_path)}: {str(e)}")
            
            if not tr_dfs:
                st.error("❌ ไม่สามารถโหลดไฟล์ใดๆ จากโฟลเดอร์ DC ได้")
                return
            
            st.info(f"📚 รวมไฟล์เทรนทั้งหมด: {len(tr_dfs)} ไฟล์")
            
            # Train AI
            G, const, regions, learning_stats = train_ai(tr_dfs)
            
            # แสดงสถิติการเรียนรู้
            st.success(f"🧠 เทรน AI เสร็จสิ้น!")
            
            with st.expander("📊 ข้อมูลที่เรียนรู้จากประวัติ", expanded=True):
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
            
            if geo:
                st.success(f"📍 ดึงพิกัดจากไฟล์เทรน: {len(geo)} สาขา")
            else:
                st.info("📍 ไม่พบข้อมูลพิกัดในไฟล์เทรน")
            
            # Process test data
            df_test = process_dataframe(load_excel(test_file.read()))
            if df_test is None:
                st.error("❌ เกิดข้อผิดพลาดในการอ่านไฟล์ Test")
                return
            
            st.info(f"📦 ออเดอร์ทั้งหมด: {len(df_test)} รายการ | สาขาที่ต้องส่ง: {df_test['Code'].nunique()} สาขา")
            
            # Run prediction
            res = run_prediction(df_test, G, geo, const, regions)
            res = res.sort_values(by=['Booking No', 'Lat'])
            
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
            
            # Display dataframe
            st.subheader("📋 ผลลัพธ์")
            st.dataframe(res, use_container_width=True, height=400)
            
            # Export
            output_filename = 'AI_Sticky_Routing_Plan.xlsx'
            export_styled_excel(res, output_filename)
            
            with open(output_filename, 'rb') as f:
                st.download_button(
                    label="💾 ดาวน์โหลดไฟล์ Excel",
                    data=f,
                    file_name=output_filename,
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )
            
            st.balloons()

if __name__ == "__main__":
    main()
