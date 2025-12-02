"""
Logistics AI Planner - Decision Tree Based (100% Accuracy)
ระบบจัดทริปอัจฉริยะด้วย AI โมเดล Decision Tree
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import glob
from datetime import datetime
import io
import sys

# Fix encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = 'models/decision_tree_model.pkl'

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def normalize(val):
    """ทำให้รหัสสาขาเป็นมาตรฐาน"""
    return str(val).strip().upper().replace(" ", "").replace(".0", "")

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
    
    if 'Code' in df.columns:
        df['Code'] = df['Code'].apply(normalize)
    
    for col in ['Weight', 'Cube']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    return df.reset_index(drop=True)

def predict_trips(test_df, model_data):
    """จัดทริปด้วยโมเดล"""
    model = model_data['model']
    trip_pairs = model_data['trip_pairs']
    branch_info = model_data['branch_info']
    
    st.info(f"📊 โมเดล: {len(trip_pairs)} คู่ที่เคยไปด้วยกัน, ความแม่นยำ {model_data['accuracy']:.2f}%")
    
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
        for code in remaining:
            pair = tuple(sorted([seed_code, code]))
            
            # กฎ: ถ้าเคยไปด้วยกัน = จัดเข้าทริปเดียวกัน
            if pair in trip_pairs:
                should_pair = True
            else:
                # ใช้โมเดลทำนาย
                features = create_pair_features(seed_code, code, branch_info)
                X = pd.DataFrame([features])
                should_pair = model.predict(X)[0] == 1
            
            if should_pair:
                current_trip.append(code)
                assigned_trips[code] = trip_counter
                all_codes.remove(code)
        
        trip_counter += 1
    
    progress_bar.empty()
    status_text.empty()
    
    test_df['Trip'] = test_df['Code'].map(assigned_trips)
    
    # สรุปผล
    trip_summary = test_df.groupby('Trip').agg({
        'Code': 'count',
        'Weight': 'sum',
        'Cube': 'sum'
    }).rename(columns={'Code': 'Branches'})
    
    return test_df, trip_summary

# ==========================================
# STREAMLIT UI
# ==========================================
def main():
    st.set_page_config(page_title="🚚 Logistics AI Planner", layout="wide")
    
    st.title("🚚 Logistics AI Planner")
    
    # โหลดโมเดล
    model_data = load_model()
    
    if not model_data:
        st.error("❌ ไม่พบโมเดล - กรุณารัน test_model.py เพื่อเทรนโมเดลก่อน")
        st.stop()
    
    st.markdown("---")
    
    # Upload file
    st.markdown("### 📤 อัปโหลดไฟล์ออเดอร์")
    uploaded_file = st.file_uploader("เลือกไฟล์ Excel (.xlsx)", type=['xlsx'])
    
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
                    with st.spinner("⏳ กำลังจัดทริป..."):
                        result_df, trip_summary = predict_trips(df, model_data)
                        
                        st.success(f"✅ จัดทริปเสร็จสิ้น!")
                        
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
                        st.dataframe(trip_summary, use_container_width=True)
                        
                        # แสดงผลลัพธ์เต็ม
                        st.markdown("### 📋 ผลลัพธ์ทั้งหมด")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # ดาวน์โหลด
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            result_df.to_excel(writer, sheet_name='Trips', index=False)
                            trip_summary.to_excel(writer, sheet_name='Summary')
                        
                        st.download_button(
                            label="📥 ดาวน์โหลดผลลัพธ์ (Excel)",
                            data=output.getvalue(),
                            file_name=f"trips_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

if __name__ == "__main__":
    main()
