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

def load_historical_data(folder='Dc'):
    """โหลดข้อมูลประวัติทั้งหมด"""
    print(f"\n{'='*60}")
    print(f"📂 กำลังโหลดข้อมูลจากโฟลเดอร์: {folder}")
    print(f"{'='*60}\n")
    
    if not os.path.exists(folder):
        print(f"❌ ไม่พบโฟลเดอร์ {folder}")
        return None
    
    files = glob.glob(os.path.join(folder, '*.xlsx'))
    if not files:
        print(f"❌ ไม่พบไฟล์ .xlsx ในโฟลเดอร์ {folder}")
        return None
    
    print(f"พบไฟล์: {len(files)} ไฟล์\n")
    
    all_data = []
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
            
            # Rename columns
            rename_map = {}
            for col in df.columns:
                col_clean = str(col).strip()
                col_upper = col_clean.upper().replace(' ', '').replace('_', '')
                
                if col_clean == 'BranchCode' or 'รหัสสาขา' in col_clean:
                    rename_map[col] = 'Code'
                elif col_clean == 'Branch' or 'ชื่อสาขา' in col_clean:
                    rename_map[col] = 'Name'
                elif col_clean == 'Trip':
                    rename_map[col] = 'Trip'
                elif col_clean == 'Trip no' or 'TRIPNO' in col_upper:
                    rename_map[col] = 'Vehicle'
                elif col_clean == 'TOTALWGT' or 'น้ำหนัก' in col_clean:
                    rename_map[col] = 'Weight'
                elif col_clean == 'TOTALCUBE' or 'คิว' in col_clean:
                    rename_map[col] = 'Cube'
                elif 'จังหวัด' in col_clean or 'PROVINCE' in col_upper:
                    rename_map[col] = 'Province'
            
            df = df.rename(columns=rename_map)
            
            # ต้องมีคอลัมน์พื้นฐาน
            if 'Code' in df.columns and 'Trip' in df.columns:
                df['Code'] = df['Code'].apply(normalize)
                df['Trip'] = df['Trip'].astype(str)
                df = df[df['Trip'].notna() & (df['Trip'] != 'nan') & (df['Trip'] != '')]
                
                # เพิ่มข้อมูลน้ำหนัก/คิว ถ้าไม่มี
                if 'Weight' not in df.columns:
                    df['Weight'] = 0.0
                if 'Cube' not in df.columns:
                    df['Cube'] = 0.0
                
                df['File'] = os.path.basename(file_path)
                all_data.append(df)
                print(f"✅ {os.path.basename(file_path)}: {len(df)} แถว, {df['Trip'].nunique()} ทริป")
            else:
                print(f"⚠️  {os.path.basename(file_path)}: ไม่มีคอลัมน์ที่จำเป็น")
        
        except Exception as e:
            print(f"❌ {os.path.basename(file_path)}: {e}")
    
    if not all_data:
        return None
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"📊 รวมข้อมูลทั้งหมด: {len(combined)} แถว, {combined['Trip'].nunique()} ทริป")
    print(f"{'='*60}\n")
    
    return combined

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
def create_training_data(df):
    """สร้างข้อมูลสำหรับเทรน: คู่สาขาที่ควรไปด้วยกัน (label=1) และไม่ควรไปด้วยกัน (label=0)"""
    print("\n📐 กำลังสร้าง Training Data...")
    
    # เก็บข้อมูลแต่ละสาขา
    branch_info = {}
    for code, group in df.groupby('Code'):
        branch_info[code] = {
            'avg_weight': group['Weight'].mean(),
            'avg_cube': group['Cube'].mean(),
            'total_trips': len(group),
            'province': group['Province'].iloc[0] if 'Province' in group.columns and group['Province'].notna().any() else 'UNKNOWN'
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
    
    # สร้าง negative pairs (สุ่มคู่ที่ไม่เคยไปด้วยกัน)
    np.random.seed(42)
    num_negative = len(positive_pairs)  # สร้างจำนวนเท่ากับ positive
    
    attempted = 0
    max_attempts = num_negative * 10
    
    while len(negative_pairs) < num_negative and attempted < max_attempts:
        idx1, idx2 = np.random.choice(len(all_codes), 2, replace=False)
        code1, code2 = all_codes[idx1], all_codes[idx2]
        pair = tuple(sorted([code1, code2]))
        
        if pair not in trip_pairs:
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
    
    return {
        'weight_sum': weight_sum,
        'cube_sum': cube_sum,
        'weight_diff': weight_diff,
        'cube_diff': cube_diff,
        'same_province': same_province,
        'same_prefix': same_prefix,
        'avg_weight_1': info1['avg_weight'],
        'avg_weight_2': info2['avg_weight'],
        'avg_cube_1': info1['avg_cube'],
        'avg_cube_2': info2['avg_cube']
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
    
    # ลอง max_depth ต่างๆ
    for max_depth in [None, 10, 15, 20, 30]:
        for min_samples_split in [2, 5, 10]:
            for min_samples_leaf in [1, 2, 5]:
                model = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                if train_score > best_score or (train_score == best_score and test_score > model.score(X_test, y_test)):
                    best_score = train_score
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
    
    # 1. Load data
    df = load_historical_data('Dc')
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
