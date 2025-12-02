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

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def normalize(val):
    """ทำให้รหัสสาขาเป็นมาตรฐาน"""
    return str(val).strip().upper().replace(" ", "").replace(".0", "")

def can_fit_truck(total_weight, total_cube, truck_type):
    """เช็คว่าน้ำหนัก/คิวใส่รถได้หรือไม่"""
    limits = LIMITS[truck_type]
    max_w = limits['max_w'] * BUFFER
    max_c = limits['max_c'] * BUFFER
    return total_weight <= max_w and total_cube <= max_c

def suggest_truck(total_weight, total_cube):
    """แนะนำรถที่เหมาะสม"""
    for truck in ['4W', 'JB', '6W']:
        if can_fit_truck(total_weight, total_cube, truck):
            return truck
    return '6W+'  # เกินกำลัง 6W

def is_similar_name(name1, name2):
    """เช็คว่าชื่อสาขาคล้ายกันหรือไม่ (เช่น นครราชสีมา1, นครราชสีมา2)"""
    def clean_name(name):
        if pd.isna(name) or name is None:
            return ""
        s = str(name).strip().upper()
        # ลบ prefix ที่พบบ่อย
        prefixes = ['PTC-MRT-', 'PTC-', 'PUN-', 'FC', 'MAXMART', 'MaxMart']
        for prefix in prefixes:
            s = s.replace(prefix.upper(), '')
        # เอาเฉพาะตัวอักษรภาษาไทยและภาษาอังกฤษ (ไม่รวมตัวเลข)
        cleaned = ''.join([c for c in s if c.isalpha()])
        return cleaned
    
    clean1 = clean_name(name1)
    clean2 = clean_name(name2)
    
    # ต้องมีความยาวพอสมควร
    if len(clean1) < 3 or len(clean2) < 3:
        return False
    
    # เช็คว่ามีส่วนที่เหมือนกันหรือไม่
    shorter = min(clean1, clean2, key=len)
    longer = max(clean1, clean2, key=len)
    
    # ถ้าชื่อสั้นอยู่ในชื่อยาว หรือ ตรงกัน 80%+ = คล้ายกัน
    if shorter in longer:
        return True
    
    # นับตัวอักษรที่เหมือนกัน
    matches = sum(1 for a, b in zip(clean1, clean2) if a == b)
    similarity = matches / max(len(clean1), len(clean2))
    
    return similarity >= 0.8  # คล้ายกัน 80% ขึ้นไป

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
    
    # Code prefix เหมือนกันหรือไม่
    prefix1 = code1[:2] if len(code1) >= 2 else code1
    prefix2 = code2[:2] if len(code2) >= 2 else code2
    same_prefix = 1 if prefix1 == prefix2 else 0
    
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
        'same_prefix': same_prefix,
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
    for col in df.columns:
        col_clean = str(col).strip()
        col_upper = col_clean.upper().replace(' ', '').replace('_', '')
        
        if col_clean == 'BranchCode' or 'รหัสสาขา' in col_clean or col_clean == 'รหัส WMS':
            rename_map[col] = 'Code'
        elif col_clean == 'Branch' or 'ชื่อสาขา' in col_clean or col_clean == 'สาขา':
            rename_map[col] = 'Name'
        elif col_clean == 'TOTALWGT' or 'น้ำหนัก' in col_clean:
            rename_map[col] = 'Weight'
        elif col_clean == 'TOTALCUBE' or 'คิว' in col_clean:
            rename_map[col] = 'Cube'
        elif 'latitude' in col_clean.lower() or col_clean == 'ละติจูด':
            rename_map[col] = 'Latitude'
        elif 'longitude' in col_clean.lower() or col_clean == 'ลองติจูด':
            rename_map[col] = 'Longitude'
        elif 'จังหวัด' in col_clean:
            rename_map[col] = 'Province'
    
    df = df.rename(columns=rename_map)
    
    # ลบคอลัมน์ซ้ำ
    df = df.loc[:, ~df.columns.duplicated()]
    
    if 'Code' in df.columns:
        df['Code'] = df['Code'].apply(normalize)
    
    for col in ['Weight', 'Cube']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    return df.reset_index(drop=True)

def predict_trips(test_df, model_data):
    """
    จัดทริปด้วย AI โดยใช้กฎลำดับความสำคัญ:
    1. ✅ เคยไปด้วยกันในประวัติ (trip_pairs) - ความแม่นยำ 100% + ใช้รถแบบเดิม
    2. ✅ ชื่อสาขาคล้ายกัน (เช่น นครราชสีมา1, นครราชสีมา2)
    3. ✅ AI ทำนายจาก Decision Tree Model
    4. ✅ เลือกประเภทรถตามประวัติ หรือ Auto-suggest
    """
    model = model_data['model']
    trip_pairs = model_data['trip_pairs']
    branch_info = model_data['branch_info']
    trip_vehicles = model_data.get('trip_vehicles', {})  # ข้อมูลรถจากประวัติ (ถ้ามี)
    
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
        
        for code in remaining:
            pair = tuple(sorted([seed_code, code]))
            
            # กฎ 1: ถ้าเคยไปด้วยกันในประวัติ = จัดเข้าทริปเดียวกัน + ใช้รถแบบเดิม (สำคัญที่สุด)
            if pair in trip_pairs:
                should_pair = True
                # ดึงข้อมูลรถจากประวัติ
                if pair in trip_vehicles and recommended_vehicle is None:
                    vehicle_info = trip_vehicles[pair]
                    recommended_vehicle = vehicle_info.get('vehicle', '6W')
            else:
                # กฎ 2: เช็คชื่อสาขาคล้ายกัน (เช่น นครราชสีมา1, นครราชสีมา2)
                seed_name = test_df[test_df['Code'] == seed_code]['Name'].iloc[0] if 'Name' in test_df.columns else ''
                code_name = test_df[test_df['Code'] == code]['Name'].iloc[0] if 'Name' in test_df.columns else ''
                
                if is_similar_name(seed_name, code_name):
                    should_pair = True
                else:
                    # กฎ 3: ใช้โมเดล AI ทำนาย
                    features = create_pair_features(seed_code, code, branch_info)
                    X = pd.DataFrame([features])
                    should_pair = model.predict(X)[0] == 1
            
            if should_pair:
                # คำนวณน้ำหนัก/คิวหลังเพิ่มสาขานี้
                trip_weight = test_df[test_df['Code'].isin(current_trip + [code])]['Weight'].sum()
                trip_cube = test_df[test_df['Code'].isin(current_trip + [code])]['Cube'].sum()
                
                # ถ้ามีรถแนะนำจากประวัติ ใช้ขีดจำกัดของรถนั้น
                if recommended_vehicle and recommended_vehicle in LIMITS:
                    max_w = LIMITS[recommended_vehicle]['max_w'] * BUFFER
                    max_c = LIMITS[recommended_vehicle]['max_c'] * BUFFER
                else:
                    # ถ้าไม่มี ใช้รถ 6W เป็นค่าเริ่มต้น
                    max_w = LIMITS['6W']['max_w'] * BUFFER
                    max_c = LIMITS['6W']['max_c'] * BUFFER
                
                # เช็คว่าเกินขีดจำกัดหรือไม่
                if trip_weight <= max_w and trip_cube <= max_c:
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
    
    # สรุปผลและแนะนำรถ
    summary_data = []
    for trip_num in sorted(test_df['Trip'].unique()):
        trip_data = test_df[test_df['Trip'] == trip_num]
        total_w = trip_data['Weight'].sum()
        total_c = trip_data['Cube'].sum()
        
        # เลือกรถ: ถ้ามีในประวัติใช้ตามประวัติ ไม่มีก็ auto-suggest
        if trip_num in trip_recommended_vehicles:
            suggested = trip_recommended_vehicles[trip_num]
            source = "📜 ประวัติ"
        else:
            suggested = suggest_truck(total_w, total_c)
            source = "🤖 AI"
        
        # คำนวณ % การใช้รถ
        if suggested in LIMITS:
            w_util = (total_w / LIMITS[suggested]['max_w']) * 100
            c_util = (total_c / LIMITS[suggested]['max_c']) * 100
        else:
            w_util = c_util = 0
        
        summary_data.append({
            'Trip': trip_num,
            'Branches': len(trip_data),
            'Weight': total_w,
            'Cube': total_c,
            'Truck': f"{suggested} {source}",
            'Weight_Use%': w_util,
            'Cube_Use%': c_util
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    return test_df, summary_df

# ==========================================
# STREAMLIT UI
# ==========================================
def main():
    st.set_page_config(
        page_title="ระบบจัดทริปส่งของ",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🚚 ระบบจัดทริปส่งของอัจฉริยะ")
        st.caption("Smart Logistics Planner")
    with col2:
        st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/1f69a.svg", width=100)
    
    st.markdown("---")
    
    # โหลดโมเดล
    model_data = load_model()
    
    if not model_data:
        st.error("❌ ไม่พบข้อมูลโมเดล กรุณาเทรนโมเดลก่อนใช้งาน")
        st.info("💡 รันคำสั่ง: `python test_model.py`")
        st.stop()
    
    # แท็บหลัก
    tab1, tab2 = st.tabs(["📦 จัดทริปส่งของ", "📚 คู่มือการใช้งาน"])
    
    # ==========================================
    # แท็บ 1: จัดทริป
    # ==========================================
    with tab1:
        st.markdown("### 📂 อัปโหลดไฟล์รายการออเดอร์")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "เลือกไฟล์ Excel (.xlsx)", 
                type=['xlsx'],
                help="อัปโหลดไฟล์ Excel ที่มีรายการสาขาและออเดอร์"
            )
        with col2:
            st.info("""
            **รูปแบบไฟล์:**
            - รหัสสาขา
            - ชื่อสาขา
            - น้ำหนัก (kg)
            - คิว (m³)
            """)
    
    if uploaded_file:
        with st.spinner("⏳ กำลังโหลดข้อมูล..."):
            df = load_excel(uploaded_file.read())
            df = process_dataframe(df)
            
            if df is not None and 'Code' in df.columns:
                st.success(f"✅ โหลดข้อมูลสำเร็จ: {len(df)} แถว")
                
                # แสดงตัวอย่างข้อมูล
                with st.expander("🔍 ดูข้อมูลต้นฉบับ"):
                    st.dataframe(df.head(10))
                
                # ปุ่มจัดทริป
                if st.button("🚀 จัดทริปอัตโนมัติ", type="primary", use_container_width=True):
                    with st.spinner("⏳ กำลังจัดทริป (ระบบเลือกรถอัตโนมัติ)..."):
                        result_df, summary = predict_trips(df, model_data)
                        
                        st.success(f"✅ จัดทริปเสร็จสิ้น! จัดได้ {len(summary)} ทริป")
                        
                        # แสดงสรุป
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("จำนวนสาขา", len(result_df))
                        with col2:
                            st.metric("จำนวนทริป", result_df['Trip'].nunique())
                        with col3:
                            avg_branches = len(result_df) / result_df['Trip'].nunique()
                            st.metric("เฉลี่ยสาขา/ทริป", f"{avg_branches:.1f}")
                        
                        st.markdown("---")
                        
                        # แสดงสรุปแต่ละทริป
                        st.markdown("### 📊 สรุปแต่ละทริป")
                        st.dataframe(
                            summary.style.format({
                                'Weight': '{:.2f}',
                                'Cube': '{:.2f}',
                                'Weight_Use%': '{:.1f}%',
                                'Cube_Use%': '{:.1f}%'
                            }).background_gradient(
                                subset=['Weight_Use%', 'Cube_Use%'],
                                cmap='RdYlGn',
                                vmin=0,
                                vmax=100
                            ),
                            use_container_width=True
                        )
                        
                        # แสดงผลลัพธ์เต็ม
                        st.markdown("### 📋 ผลลัพธ์ทั้งหมด")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # ดาวน์โหลด
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            result_df.to_excel(writer, sheet_name='Trips', index=False)
                            summary.to_excel(writer, sheet_name='Summary', index=False)
                        
                        st.download_button(
                            label="📥 ดาวน์โหลดผลลัพธ์ (Excel)",
                            data=output.getvalue(),
                            file_name=f"trips_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

if __name__ == "__main__":
    main()
