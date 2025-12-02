# Install xlsxwriter
try:
    import xlsxwriter
except ImportError:
    pass

import pandas as pd
import numpy as np
import io
import os
import glob
import ipywidgets as widgets
from IPython.display import display, clear_output
import networkx as nx
from sklearn.cluster import DBSCAN
import math
import warnings
try:
    from google.colab import files
except ImportError:
    pass

warnings.filterwarnings('ignore')

# ==========================================
# 1. CONFIG
# ==========================================
LIMITS = {'4W': {'max_w': 2500, 'max_c': 5.0}, 'JB': {'max_w': 3500, 'max_c': 8.0}, '6W': {'max_w': 5800, 'max_c': 22.0}}
BUFFER = 1.05
MAX_KM_CLUSTER = 50.0 # รัศมีจับกลุ่มก้อนใหญ่

# กฎการจัดเส้นทาง (Routing Rules)
TARGET_DROPS = 10      # เป้าหมายคือ 10 จุด
MAX_DROPS_FLEX = 12    # อนุโลมได้ถึง 12 จุด
NEARBY_RADIUS = 5.0    # ระยะ "ใกล้มาก" (5 กม.) ที่อนุญาตให้เกิน 10 จุดได้

EXCLUDE = ['PTDC', 'Distribution Center', 'DCวังน้อย', 'DC011']

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def normalize(val):
    return str(val).strip().upper().replace(" ", "").replace(".0", "")

def haversine(lat1, lon1, lat2, lon2):
    """คำนวณระยะทางจริงบนโลก (กิโลเมตร)"""
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLon = math.radians(lon2 - lon1)
    a = math.sin(dLat/2) * math.sin(dLat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def is_similar_name(name1, name2):
    """เช็คว่าชื่อร้านคล้ายกันไหม (ตัดตัวเลขออกแล้วเทียบ)"""
    def clean(n):
        return ''.join([c for c in str(n) if c.isalpha()]) # เอาเฉพาะตัวหนังสือ
    return clean(name1) == clean(name2) and len(clean(name1)) > 3

# ==========================================
# 3. LOADERS & PROCESSORS
# ==========================================
def load_excel(content, file_type='Order'):
    try:
        xls = pd.ExcelFile(io.BytesIO(content))
        target_sheet = None
        priority = ['2.', 'punthai', 'order', 'history']
        for p in priority:
            for s in xls.sheet_names:
                if p in s.lower(): target_sheet = s; break
            if target_sheet: break
        if not target_sheet: target_sheet = xls.sheet_names[0]
        
        print(f"📖 Reading '{target_sheet}'...")
        # Auto Scan Header
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
    print(f"🧠 Training from {len(df_list)} files...")
    
    for df in df_list:
        if df is None or 'Trip' not in df.columns: continue
        if isinstance(df['Trip'], pd.DataFrame): df['Trip'] = df['Trip'].iloc[:,0]
        df = df.dropna(subset=['Trip'])
        
        for idx, r in df.iterrows():
            if 'Province' in r and pd.notna(r['Province']): zones[r['Code']] = str(r['Province']).strip()
            
        for t, g in df.groupby('Trip'):
            codes = g['Code'].unique()
            veh = str(g['Vehicle'].iloc[0]).upper() if 'Vehicle' in g.columns else ''
            rank = 3 if '6' in veh else (2 if 'J' in veh else 1)
            for c in codes: req[c] = max(req.get(c,1), rank)
            
            if len(codes)>1:
                for i in range(len(codes)):
                    for j in range(i+1, len(codes)): G.add_edge(codes[i], codes[j])
            elif len(codes)==1: G.add_node(codes[0])
            
    return G, req, zones

def select_truck(w, c, min_rank):
    s = min_rank
    if s >= 3: return '6 ล้อ ตู้ทึบ'
    if s <= 1 and c <= LIMITS['4W']['max_c']*BUFFER and w <= LIMITS['4W']['max_w']: return '4 ล้อ ตู้ทึบ'
    if s <= 2 and c <= LIMITS['JB']['max_c']*BUFFER and w <= LIMITS['JB']['max_w']: return '4 ล้อ จัมโบ้ ตู้ทึบ'
    return '6 ล้อ ตู้ทึบ'

# ==========================================
# 5. NEW ALGORITHM: STICKY ROUTING
# ==========================================
def run_prediction(df_test, G, geo, constraints, zone_mem):
    print("🚀 Predicting with Sticky Neighbor Logic (Close/SameName)...")
    
    df_test['Lat'] = df_test.apply(lambda r: geo.get(r['Code'],(0,0))[0] if r['Lat']==0 else r['Lat'], axis=1)
    df_test['Lon'] = df_test.apply(lambda r: geo.get(r['Code'],(0,0))[1] if r['Lon']==0 else r['Lon'], axis=1)
    
    # 1. Clustering
    hist_map = {n:i for i,c in enumerate(nx.connected_components(G)) for n in c}
    df_test['Cluster'] = df_test['Code'].map(lambda x: f"H-{hist_map[x]}" if x in hist_map else "UNK")
    
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
            lambda x: f"Z-{zone_mem.get(x, 'NEW')}" if x in zone_mem else f"NEW-{x}"
        )
    
    final_rows = []
    trip_cnt = 1
    
    # 2. Optimization Per Cluster
    for cid, group in df_test.groupby('Cluster'):
        
        # Merge Duplicates First
        pool = []
        for code, sub in group.groupby('Code'):
            pool.append({
                'Code': code, 'Name': sub.iloc[0]['Name'],
                'Wgt': sub['Wgt'].sum(), 'Cube': sub['Cube'].sum(),
                'Lat': sub.iloc[0]['Lat'], 'Lon': sub.iloc[0]['Lon']
            })
            
        # เริ่มจัดรถแบบ "หาเพื่อนใกล้ๆ"
        while pool:
            # 1. Start with largest remaining item
            pool.sort(key=lambda x: x['Cube'], reverse=True)
            current_truck = []
            
            # ดึงชิ้นแรก
            seed = pool.pop(0)
            current_truck.append(seed)
            
            curr_w = seed['Wgt']
            curr_c = seed['Cube']
            last_lat = seed['Lat']
            last_lon = seed['Lon']
            last_name = seed['Name']
            
            drops = 1
            max_req = constraints.get(seed['Code'], 1)
            
            # 2. Find neighbors loop
            while True:
                best_idx = -1
                best_score = float('inf') # Lower is better
                
                # Scan remaining pool
                for i, cand in enumerate(pool):
                    # Check Capacity
                    new_w = curr_w + cand['Wgt']
                    new_c = curr_c + cand['Cube']
                    
                    if new_w > 5800: continue
                    if new_c > 22.0 * BUFFER: continue
                    
                    # Check Drop Limit
                    is_nearby = False
                    is_same_name = is_similar_name(last_name, cand['Name'])
                    dist = haversine(last_lat, last_lon, cand['Lat'], cand['Lon']) if last_lat!=0 and cand['Lat']!=0 else 999
                    
                    if dist <= NEARBY_RADIUS: is_nearby = True
                    
                    # Logic 10-12 Drops
                    if drops >= TARGET_DROPS:
                        # เกิน 10 จุด -> รับเฉพาะ พวกชื่อเหมือน หรือ ใกล้มากๆ เท่านั้น
                        if drops >= MAX_DROPS_FLEX: continue # เกิน 12 ตัดทิ้งเลย
                        if not (is_same_name or is_nearby): continue
                    
                    # Score (ยิ่งน้อยยิ่งดี: ใกล้ 0)
                    # ให้ Priority: ชื่อเหมือน > ใกล้ > ไกล
                    score = dist 
                    if is_same_name: score -= 1000 # Bonus ชื่อเหมือน
                    
                    if score < best_score:
                        best_score = score
                        best_idx = i
                        
                if best_idx != -1:
                    # Add Item
                    sel = pool.pop(best_idx)
                    current_truck.append(sel)
                    
                    curr_w += sel['Wgt']
                    curr_c += sel['Cube']
                    drops += 1
                    
                    # Update Ref (ย้ายจุดอ้างอิงไปที่ล่าสุด เพื่อให้วิ่งเป็นเส้น)
                    if sel['Lat']!=0: 
                        last_lat = sel['Lat']; last_lon = sel['Lon']
                    last_name = sel['Name']
                    
                    # Update Constraint
                    max_req = max(max_req, constraints.get(sel['Code'], 1))
                else:
                    break # ไม่มีอะไรใส่เพิ่มได้แล้ว
            
            # Finalize Truck
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

# ==========================================
# 6. MAIN
# ==========================================
def main():
    print("🤖 AI Logistics: Nearest Neighbor + Same Name Grouping")
    print("   - Max 10 Drops (Soft Limit)")
    print("   - Up to 12 Drops if Nearby/Same Name")
    
    up_hist = widgets.FileUpload(description='1. ประวัติ')
    up_geo = widgets.FileUpload(description='2. พิกัด')
    up_train = widgets.FileUpload(description='3. Train')
    up_test = widgets.FileUpload(description='4. Test')
    btn = widgets.Button(description="Start", button_style='success')
    out = widgets.Output()
    
    display(up_hist, up_geo, up_train, up_test, btn, out)
    
    def run(b):
        with out:
            clear_output()
            # 1. Train
            tr_dfs = []
            if up_hist.value: tr_dfs.append(process_dataframe(load_excel(list(up_hist.value.values())[0]['content'])))
            if up_train.value: tr_dfs.append(process_dataframe(load_excel(list(up_train.value.values())[0]['content'])))
            
            G, const, zones = train_ai(tr_dfs)
            geo = {}
            if up_geo.value: geo = process_geo(load_excel(list(up_geo.value.values())[0]['content']))
            
            # 2. Predict
            if up_test.value:
                df_test = process_dataframe(load_excel(list(up_test.value.values())[0]['content']))
                if df_test is not None:
                    res = run_prediction(df_test, G, geo, const, zones)
                    res = res.sort_values(by=['Booking No', 'Lat'])
                    print(f"✅ Predicted {res['Booking No'].nunique()} Trips.")
                    export_styled_excel(res, 'AI_Smart_Drops.xlsx')
                    files.download('AI_Smart_Drops.xlsx')
                else: print("❌ Test Error")
            else: print("⚠️ No Test File")
            
    btn.on_click(run)

if __name__ == "__main__":
    main()