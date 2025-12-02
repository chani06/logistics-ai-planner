import streamlit as st
import pandas as pd
import pickle
import json
import os
from datetime import datetime

# ==========================================
# CONFIG
# ==========================================
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'trip_pairs.pkl')
MODEL_INFO_PATH = os.path.join(MODEL_DIR, 'model_info.json')

# ==========================================
# FUNCTIONS
# ==========================================
def normalize(val):
    """ทำให้รหัสสาขาเป็นมาตรฐาน"""
    return str(val).strip().upper().replace(" ", "").replace(".0", "")

def load_excel_sheet(file_content, sheet_name=None):
    """โหลด Excel และหา sheet ที่ต้องการ"""
    try:
        import io
        xls = pd.ExcelFile(io.BytesIO(file_content))
        
        # ถ้าระบุชื่อ sheet
        if sheet_name and sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
        else:
            # หา sheet ที่มี "punthai" หรือ sheet แรก
            target_sheet = None
            for s in xls.sheet_names:
                if 'punthai' in s.lower() or '2.' in s.lower():
                    target_sheet = s
                    break
            
            if not target_sheet:
                target_sheet = xls.sheet_names[0]
            
            df = pd.read_excel(xls, sheet_name=target_sheet)
        
        return df
    except Exception as e:
        st.error(f"❌ Error loading Excel: {e}")
        return None

def process_dataframe(df):
    """แปลงคอลัมน์เป็นรูปแบบมาตรฐาน"""
    if df is None:
        return None
    
    # Rename columns
    rename_map = {}
    for col in df.columns:
        col_upper = str(col).upper().replace(' ', '').replace('_', '')
        if 'BRANCHCODE' in col_upper or 'รหัสสาขา' in col:
            rename_map[col] = 'Code'
        elif 'BRANCH' in col_upper and 'CODE' not in col_upper:
            rename_map[col] = 'Name'
        elif col.strip() == 'Trip':
            rename_map[col] = 'Trip'
        elif 'TRIPNO' in col_upper or col.strip() == 'Trip no':
            rename_map[col] = 'Vehicle'
        elif 'WGT' in col_upper or 'น้ำหนัก' in col:
            rename_map[col] = 'Wgt'
        elif 'CUBE' in col_upper or 'คิว' in col:
            rename_map[col] = 'Cube'
    
    df = df.rename(columns=rename_map)
    
    # Normalize Code
    if 'Code' in df.columns:
        df['Code'] = df['Code'].apply(normalize)
    
    return df.reset_index(drop=True)

def learn_trip_patterns(df):
    """เรียนรู้รูปแบบการจัดทริปจากไฟล์ประวัติ"""
    if 'Trip' not in df.columns or 'Code' not in df.columns:
        return {}
    
    # แปลง Trip เป็น string
    df = df.copy()
    df['Trip'] = df['Trip'].astype(str)
    df = df[df['Trip'].notna() & (df['Trip'] != 'nan') & (df['Trip'] != '')]
    
    # สร้าง dictionary เก็บคู่ที่เคยไปด้วยกัน
    trip_pairs = {}
    
    for trip_id, group in df.groupby('Trip'):
        codes = sorted(group['Code'].unique())
        
        # บันทึกทุกคู่ในทริปนี้
        for i in range(len(codes)):
            for j in range(i+1, len(codes)):
                pair = tuple(sorted([codes[i], codes[j]]))
                trip_pairs[pair] = trip_pairs.get(pair, 0) + 1
    
    return trip_pairs

def test_accuracy(trip_pairs, test_df):
    """ทดสอบความแม่นยำของโมเดล"""
    if 'Trip' not in test_df.columns:
        return None
    
    test_df = test_df.copy()
    test_df['Trip'] = test_df['Trip'].astype(str)
    test_df = test_df[test_df['Trip'].notna() & (test_df['Trip'] != 'nan') & (test_df['Trip'] != '')]
    
    total_pairs = 0
    matched_pairs = 0
    missing_pairs = []
    
    for trip_id, group in test_df.groupby('Trip'):
        codes = sorted(group['Code'].unique())
        
        if len(codes) < 2:
            continue
        
        for i in range(len(codes)):
            for j in range(i+1, len(codes)):
                total_pairs += 1
                pair = tuple(sorted([codes[i], codes[j]]))
                
                if pair in trip_pairs:
                    matched_pairs += 1
                else:
                    missing_pairs.append((trip_id, codes[i], codes[j]))
    
    accuracy = (matched_pairs / total_pairs * 100) if total_pairs > 0 else 0
    
    return {
        'total_pairs': total_pairs,
        'matched_pairs': matched_pairs,
        'missing_pairs': missing_pairs[:20],
        'accuracy': accuracy
    }

def save_model(trip_pairs, source_files, stats):
    """บันทึกโมเดลลงไฟล์"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # บันทึกโมเดล
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(trip_pairs, f)
    
    # บันทึกข้อมูล
    model_info = {
        'created_at': datetime.now().isoformat(),
        'source_files': source_files,
        'total_pairs': len(trip_pairs),
        'stats': stats
    }
    
    with open(MODEL_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    return model_info

def load_model():
    """โหลดโมเดลจากไฟล์"""
    if not os.path.exists(MODEL_PATH):
        return None, None
    
    try:
        with open(MODEL_PATH, 'rb') as f:
            trip_pairs = pickle.load(f)
        
        with open(MODEL_INFO_PATH, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
        
        return trip_pairs, model_info
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

def predict_trips(df, trip_pairs):
    """จัดทริปตามโมเดล"""
    if 'Code' not in df.columns:
        return None
    
    # สร้าง Trip ID ใหม่
    used_codes = set()
    trips = []
    trip_id = 1
    
    df = df.copy()
    df['Code'] = df['Code'].apply(normalize)
    codes = df['Code'].unique().tolist()
    
    while codes:
        # เลือกสาขาแรก
        seed = codes.pop(0)
        current_trip = [seed]
        used_codes.add(seed)
        
        # หาสาขาที่เคยไปด้วยกัน
        for code in codes[:]:
            pair = tuple(sorted([seed, code]))
            if pair in trip_pairs:
                current_trip.append(code)
                codes.remove(code)
                used_codes.add(code)
        
        # บันทึกทริป
        for code in current_trip:
            trips.append({'Code': code, 'Trip': f"AI-{trip_id:03d}"})
        
        trip_id += 1
    
    # รวมกับข้อมูลเดิม
    trip_df = pd.DataFrame(trips)
    result = df.merge(trip_df, on='Code', how='left', suffixes=('_old', ''))
    
    return result

# ==========================================
# STREAMLIT UI
# ==========================================
def main():
    st.set_page_config(page_title="🚚 Logistics AI Planner", layout="wide")
    st.title("🚚 Logistics AI Planner - Simple Model")
    
    # แสดงสถานะโมเดล
    model_exists = os.path.exists(MODEL_PATH)
    if model_exists:
        trip_pairs, model_info = load_model()
        if model_info:
            st.success(f"✅ มีโมเดลที่เทรนไว้แล้ว (สร้างเมื่อ: {model_info['created_at'][:19]})")
            st.info(f"📊 จำนวนคู่ที่จดจำ: {model_info['total_pairs']} คู่")
    else:
        st.warning("⚠️ ยังไม่มีโมเดล - ต้องเทรนก่อนใช้งาน")
    
    st.markdown("---")
    
    # สร้าง Tabs
    tab1, tab2 = st.tabs(["🎯 ใช้งานระบบ", "🎓 เทรนโมเดล"])
    
    # ========== TAB 1: ใช้งาน ==========
    with tab1:
        st.markdown("### 📤 อัปโหลดไฟล์ออเดอร์ใหม่")
        
        test_file = st.file_uploader("เลือกไฟล์ Excel", type=['xlsx'], key='test')
        
        if st.button("🚀 จัดทริป", type="primary"):
            if not test_file:
                st.error("❌ กรุณาอัปโหลดไฟล์")
            elif not model_exists:
                st.error("❌ ยังไม่มีโมเดล - ไปที่แท็บ 'เทรนโมเดล' ก่อน")
            else:
                with st.spinner("⏳ กำลังประมวลผล..."):
                    # โหลดโมเดล
                    trip_pairs, model_info = load_model()
                    
                    # โหลดไฟล์ Test
                    test_content = test_file.read()
                    test_df = load_excel_sheet(test_content)
                    test_df = process_dataframe(test_df)
                    
                    if test_df is not None:
                        # จัดทริป
                        result = predict_trips(test_df, trip_pairs)
                        
                        if result is not None:
                            st.success("✅ จัดทริปเสร็จแล้ว!")
                            st.dataframe(result)
                            
                            # Export
                            output = io.BytesIO()
                            result.to_excel(output, index=False)
                            st.download_button(
                                "📥 ดาวน์โหลดผลลัพธ์",
                                data=output.getvalue(),
                                file_name=f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                            )
    
    # ========== TAB 2: เทรน ==========
    with tab2:
        st.markdown("### 🎓 เทรนโมเดลจากไฟล์ประวัติ")
        
        # ค้นหาไฟล์ใน DC folder
        dc_files = []
        if os.path.exists('Dc'):
            import glob
            dc_files = glob.glob('Dc/*.xlsx')
        
        if dc_files:
            st.success(f"📂 พบไฟล์ประวัติ {len(dc_files)} ไฟล์:")
            for f in dc_files:
                st.text(f"  • {os.path.basename(f)}")
        else:
            st.warning("⚠️ ไม่พบไฟล์ในโฟลเดอร์ Dc/")
        
        # ปุ่มเทรน
        col1, col2 = st.columns([1, 3])
        with col1:
            train_button = st.button("🚀 เทรนโมเดล", type="primary", use_container_width=True)
        with col2:
            if model_exists:
                if st.button("🗑️ ลบโมเดลเก่า", use_container_width=True):
                    if os.path.exists(MODEL_PATH):
                        os.remove(MODEL_PATH)
                    if os.path.exists(MODEL_INFO_PATH):
                        os.remove(MODEL_INFO_PATH)
                    st.success("✅ ลบโมเดลแล้ว")
                    st.rerun()
        
        if train_button:
            if not dc_files:
                st.error("❌ ไม่มีไฟล์ประวัติในโฟลเดอร์ Dc/")
            else:
                with st.spinner("⏳ กำลังเทรนโมเดล..."):
                    all_pairs = {}
                    source_files = []
                    total_trips = 0
                    
                    # โหลดไฟล์ทั้งหมด
                    for file_path in dc_files:
                        try:
                            with open(file_path, 'rb') as f:
                                content = f.read()
                            
                            df = load_excel_sheet(content)
                            df = process_dataframe(df)
                            
                            if df is not None and 'Trip' in df.columns:
                                pairs = learn_trip_patterns(df)
                                
                                # รวมคู่เข้าด้วยกัน
                                for pair, count in pairs.items():
                                    all_pairs[pair] = all_pairs.get(pair, 0) + count
                                
                                source_files.append(os.path.basename(file_path))
                                total_trips += df['Trip'].nunique()
                                
                                st.text(f"✅ {os.path.basename(file_path)}: {len(pairs)} คู่")
                        except Exception as e:
                            st.error(f"❌ Error: {os.path.basename(file_path)}: {e}")
                    
                    # ทดสอบความแม่นยำกับไฟล์แรก
                    st.markdown("---")
                    st.markdown("### 🎯 ทดสอบความแม่นยำ")
                    
                    with open(dc_files[0], 'rb') as f:
                        test_content = f.read()
                    test_df = load_excel_sheet(test_content)
                    test_df = process_dataframe(test_df)
                    
                    accuracy_result = test_accuracy(all_pairs, test_df)
                    
                    if accuracy_result:
                        acc = accuracy_result['accuracy']
                        
                        if acc >= 95:
                            st.success(f"✅ ความแม่นยำ: {acc:.1f}% - ยอดเยี่ยม!")
                        elif acc >= 80:
                            st.warning(f"⚠️ ความแม่นยำ: {acc:.1f}% - พอใช้")
                        else:
                            st.error(f"❌ ความแม่นยำ: {acc:.1f}% - ต่ำเกินไป")
                        
                        st.metric("คู่ที่ตรงกัน", f"{accuracy_result['matched_pairs']}/{accuracy_result['total_pairs']}")
                        
                        if accuracy_result['missing_pairs']:
                            with st.expander("🔍 ดูคู่ที่ไม่ตรงกัน (20 คู่แรก)"):
                                for trip, code1, code2 in accuracy_result['missing_pairs']:
                                    st.text(f"Trip {trip}: {code1} ↔ {code2}")
                    
                    # บันทึกโมเดล
                    stats = {
                        'total_trips': total_trips,
                        'total_files': len(source_files)
                    }
                    
                    model_info = save_model(all_pairs, source_files, stats)
                    
                    st.markdown("---")
                    st.success("✅ บันทึกโมเดลเรียบร้อย!")
                    st.json(model_info)
                    st.balloons()

if __name__ == "__main__":
    import io
    main()
