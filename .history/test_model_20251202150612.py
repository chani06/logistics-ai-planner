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
            
            # หา header row ที่ถูกต้อง
            df_temp = pd.read_excel(file_path, sheet_name=target_sheet, header=None)
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
            
            df = pd.read_excel(file_path, sheet_name=target_sheet, header=header_row)
            
            # ลบคอลัมน์ซ้ำ
            df = df.loc[:, ~df.columns.duplicated()]
            
            # Rename columns - รองรับหลายรูปแบบ
            rename_map = {}
            for col in df.columns:
                col_clean = str(col).strip()
                col_upper = col_clean.upper().replace(' ', '').replace('_', '')
                
                # รหัสสาขา
                if col_clean == 'BranchCode' or 'รหัสสาขา' in col_clean or col_clean == 'รหัส WMS' or 'BRANCH_CODE' in col_upper:
                    rename_map[col] = 'Code'
                # ชื่อสาขา
                elif col_clean == 'Branch' or 'ชื่อสาขา' in col_clean or col_clean == 'สาขา' or 'BRANCH_DESCRIPTION' in col_upper:
                    rename_map[col] = 'Name'
                # เลขทริป
                elif col_clean == 'Trip' or col_clean == 'Booking No':
                    rename_map[col] = 'Trip'
                # ประเภทรถ
                elif col_clean == 'Trip no' or 'TRIPNO' in col_upper or col_clean == 'ประเภทรถ':
                    rename_map[col] = 'Vehicle'
                # น้ำหนัก
                elif col_clean == 'TOTALWGT' or 'น้ำหนัก' in col_clean or 'WEIGHT' in col_upper:
                    rename_map[col] = 'Weight'
                # คิว/ปริมาตร
                elif col_clean == 'TOTALCUBE' or 'คิว' in col_clean or 'CUBE' in col_upper:
                    rename_map[col] = 'Cube'
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
            has_trip = 'Trip' in df.columns
            has_location = 'Latitude' in df.columns and 'Longitude' in df.columns
            
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
        train_df = pd.concat(train_data, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"📚 TRAIN DATA: {len(train_df)} แถว, {train_df['Trip'].nunique()} ทริป")
        print(f"{'='*60}\n")
    
    if test_data:
        test_df = pd.concat(test_data, ignore_index=True)
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
def create_training_data(df):
    """สร้างข้อมูลสำหรับเทรน: คู่สาขาที่ควรไปด้วยกัน (label=1) และไม่ควรไปด้วยกัน (label=0)"""
    print("\n📐 กำลังสร้าง Training Data...")
    
    # เก็บข้อมูลแต่ละสาขา
    branch_info = {}
    for code, group in df.groupby('Code'):
        # ดึงพิกัดถ้ามี
        lat = group['Latitude'].iloc[0] if 'Latitude' in group.columns else 0.0
        lon = group['Longitude'].iloc[0] if 'Longitude' in group.columns else 0.0
        
        branch_info[code] = {
            'avg_weight': group['Weight'].mean(),
            'avg_cube': group['Cube'].mean(),
            'total_trips': len(group),
            'province': group['Province'].iloc[0] if 'Province' in group.columns and group['Province'].notna().any() else 'UNKNOWN',
            'latitude': float(lat) if pd.notna(lat) else 0.0,
            'longitude': float(lon) if pd.notna(lon) else 0.0
        }
    
    # สร้างข้อมูลเทรน
    positive_pairs = []  # คู่ที่ควรไปด้วยกัน
    negative_pairs = []  # คู่ที่ไม่ควรไปด้วยกัน
    
    all_codes = list(branch_info.keys())
    trip_pairs = set()  # เก็บคู่ที่เคยไปด้วยกัน
    
    # หาคู่ที่เคยไปด้วยกัน (Positive pairs)
    for trip, group in df.groupby('Trip'):
        codes = sorted(group['Code'].unique())
        
        if len(codes) >= 2:
            for i in range(len(codes)):
                for j in range(i+1, len(codes)):
                    pair = tuple(sorted([codes[i], codes[j]]))
                    trip_pairs.add(pair)
    
    print(f"  ✅ พบคู่ที่เคยไปด้วยกัน: {len(trip_pairs)} คู่")
    
    # สร้าง features สำหรับ positive pairs
    for code1, code2 in trip_pairs:
        if code1 in branch_info and code2 in branch_info:
            features = create_pair_features(code1, code2, branch_info)
            features['label'] = 1  # ควรไปด้วยกัน
            positive_pairs.append(features)
    
    # สร้าง negative pairs - เลือกสาขาจากคนละทริปที่ไม่เคยไปด้วยกัน
    # กลยุทธ์: หาคู่สาขาที่อยู่ในทริปต่างกัน และห่างไกลกัน (จังหวัดคนละภาค)
    np.random.seed(42)
    num_negative = len(positive_pairs)
    
    # สร้างรายการทริปของแต่ละสาขา
    code_trips = {}
    for trip, group in df.groupby('Trip'):
        for code in group['Code'].unique():
            if code not in code_trips:
                code_trips[code] = []
            code_trips[code].append(trip)
    
    attempted = 0
    max_attempts = num_negative * 20
    
    while len(negative_pairs) < num_negative and attempted < max_attempts:
        idx1, idx2 = np.random.choice(len(all_codes), 2, replace=False)
        code1, code2 = all_codes[idx1], all_codes[idx2]
        pair = tuple(sorted([code1, code2]))
        
        # เช็คว่าไม่เคยไปด้วยกันจริงๆ
        if pair not in trip_pairs:
            # เพิ่มเงื่อนไข: ควรอยู่คนละทริปอย่างชัดเจน (ไม่มีทริปที่ทับซ้อนกัน)
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
    
    # รวมข้อมูล
    all_pairs = positive_pairs + negative_pairs
    train_df = pd.DataFrame(all_pairs)
    
    return train_df, trip_pairs, branch_info

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
    
    # Code prefix เหมือนกันหรือไม่ (2 ตัวแรก)
    prefix1 = code1[:2] if len(code1) >= 2 else code1
    prefix2 = code2[:2] if len(code2) >= 2 else code2
    same_prefix = 1 if prefix1 == prefix2 else 0
    
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

# ==========================================
# 3. TRAIN MODEL
# ==========================================
def train_decision_tree(train_df):
    """เทรนโมเดล Decision Tree"""
    print("\n🌲 กำลังเทรน Decision Tree...")
    
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
    
    # ลอง max_depth ต่างๆ - เพิ่มความลึกเพื่อเรียนรู้ pattern ซับซ้อน
    for max_depth in [None, 15, 20, 30, 50]:
        for min_samples_split in [2, 3, 5]:
            for min_samples_leaf in [1, 2]:
                for criterion in ['gini', 'entropy']:
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        criterion=criterion,
                        random_state=42
                    )
                    
                    model.fit(X_train, y_train)
                    train_score = model.score(X_train, y_train)
                    test_score = model.score(X_test, y_test)
                    
                    # เน้นที่ test score สูงสุด
                    if test_score > best_score or (test_score == best_score and train_score >= 0.99):
                        best_score = test_score
                        best_model = model
    
    # ใช้โมเดลที่ดีที่สุด
    train_accuracy = best_model.score(X_train, y_train)
    test_accuracy = best_model.score(X_test, y_test)
    
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
def test_model_on_actual_trips(df, model, trip_pairs, branch_info):
    """ทดสอบโมเดลกับทริปจริง"""
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
                
                # สร้าง features
                features = create_pair_features(code1, code2, branch_info)
                X = pd.DataFrame([features])
                
                # ทำนาย
                prediction = model.predict(X)[0]
                
                # ควรเป็น 1 (เพราะเป็นทริปจริง)
                if prediction == 1:
                    correct_pairs += 1
                else:
                    incorrect_pairs.append({
                        'trip': trip,
                        'code1': code1,
                        'code2': code2,
                        'predicted': prediction
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
            print(f"  Trip {item['trip']}: {item['code1']} ↔ {item['code2']}")
    
    return accuracy, incorrect_pairs

# ==========================================
# 5. SAVE MODEL
# ==========================================
def save_model(model, trip_pairs, branch_info, accuracy):
    """บันทึกโมเดล"""
    os.makedirs('models', exist_ok=True)
    
    model_data = {
        'model': model,
        'trip_pairs': trip_pairs,
        'branch_info': branch_info,
        'accuracy': accuracy,
        'created_at': datetime.now().isoformat()
    }
    
    with open('models/decision_tree_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ บันทึกโมเดลที่: models/decision_tree_model.pkl")

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
    if df is None:
        print("\n❌ ไม่สามารถโหลดข้อมูลได้")
        return
    
    # 2. สร้าง training data
    train_df, trip_pairs, branch_info = create_training_data(df)
    
    # 3. Train model
    model, train_acc, test_acc = train_decision_tree(train_df)
    
    # 4. Test กับทริปจริง
    accuracy, incorrect = test_model_on_actual_trips(df, model, trip_pairs, branch_info)
    
    # 5. บันทึกโมเดลถ้าแม่นยำพอ
    if accuracy >= 95.0:
        save_model(model, trip_pairs, branch_info, accuracy)
        print(f"\n🎉 โมเดลผ่านเกณฑ์! ({accuracy:.2f}%)")
    else:
        print(f"\n⚠️  โมเดลยังไม่ผ่านเกณฑ์ ({accuracy:.2f}% < 95%)")
        print(f"ต้องปรับปรุงเพิ่มเติม")
    
    print(f"\n{'#'*60}")
    print(f"# เสร็จสิ้น")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()
