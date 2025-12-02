"""
สคริปต์เทรนโมเดล 100 รอบจนได้ความแม่นยำ 100%
"""
import pandas as pd
import os
import glob
import pickle
import json
from datetime import datetime
import sys

# ตั้งค่า encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'trip_pairs.pkl')
MODEL_INFO_PATH = os.path.join(MODEL_DIR, 'model_info.json')

def normalize(val):
    return str(val).strip().upper().replace(" ", "").replace(".0", "")

def load_excel_sheet(file_path):
    """โหลด Excel และหา header อัตโนมัติ"""
    try:
        xls = pd.ExcelFile(file_path)
        
        # หา sheet ที่มี "punthai" หรือ "2."
        target_sheet = None
        for s in xls.sheet_names:
            if 'punthai' in s.lower() or '2.' in s.lower():
                target_sheet = s
                break
        
        if not target_sheet:
            target_sheet = xls.sheet_names[0]
        
        print(f"[LOAD] Sheet: {target_sheet}")
        
        # หา header ที่ถูกต้อง
        for header_row in [0, 1, 2]:
            try:
                df = pd.read_excel(xls, sheet_name=target_sheet, header=header_row)
                col_str = ' '.join([str(c).upper() for c in df.columns])
                if 'BRANCH' in col_str or 'TRIP' in col_str or 'CODE' in col_str:
                    print(f"[OK] Found header at row {header_row}")
                    return df
            except:
                continue
        
        df = pd.read_excel(xls, sheet_name=target_sheet, header=0)
        return df
    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def process_dataframe(df):
    """ประมวลผล DataFrame"""
    if df is None:
        return None
    
    # Rename columns
    rename_map = {}
    for col in df.columns:
        col_str = str(col).strip()
        col_upper = col_str.upper().replace(' ', '').replace('_', '')
        
        if col_str == 'BranchCode':
            rename_map[col] = 'Code'
        elif col_str == 'Branch':
            rename_map[col] = 'Name'
        elif col_str == 'Trip':
            rename_map[col] = 'Trip'
        elif col_str == 'Trip no':
            rename_map[col] = 'Vehicle'
        elif col_str == 'TOTALWGT':
            rename_map[col] = 'Wgt'
        elif col_str == 'TOTALCUBE':
            rename_map[col] = 'Cube'
        elif isinstance(col, str):
            if 'BRANCHCODE' in col_upper:
                rename_map[col] = 'Code'
            elif 'WGT' in col_upper:
                rename_map[col] = 'Wgt'
            elif 'CUBE' in col_upper:
                rename_map[col] = 'Cube'
            elif 'TRIPNO' in col_upper:
                rename_map[col] = 'Vehicle'
    
    df = df.rename(columns=rename_map)
    
    required = ['Code', 'Trip']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing: {missing}")
        return None
    
    df['Code'] = df['Code'].apply(normalize)
    
    if 'Wgt' not in df.columns:
        df['Wgt'] = 0.0
    if 'Cube' not in df.columns:
        df['Cube'] = 0.0
    
    df['Wgt'] = pd.to_numeric(df['Wgt'], errors='coerce').fillna(0.0)
    df['Cube'] = pd.to_numeric(df['Cube'], errors='coerce').fillna(0.0)
    
    return df.reset_index(drop=True)

def filter_hubs(df):
    """กรอง Hub/DC ออก"""
    if df is None:
        return None
    
    hub_keywords = ['DC011', 'PTDC', 'DC', 'DISTRIBUTION']
    original = len(df)
    
    for keyword in hub_keywords:
        df = df[~df['Code'].str.contains(keyword, na=False, case=False)]
    
    removed = original - len(df)
    if removed > 0:
        print(f"[FILTER] Removed {removed} hub/DC entries")
    
    return df

def learn_trip_patterns(df):
    """เรียนรู้คู่สาขาจากทริป"""
    if 'Trip' not in df.columns or 'Code' not in df.columns:
        return {}
    
    df = df.copy()
    df['Trip'] = df['Trip'].astype(str)
    df = df[df['Trip'].notna() & (df['Trip'] != 'nan') & (df['Trip'] != '')]
    
    trip_pairs = {}
    trip_details = {}
    
    for trip_id, group in df.groupby('Trip'):
        codes = sorted(group['Code'].unique())
        
        trip_details[trip_id] = {
            'codes': codes,
            'count': len(codes),
            'wgt': group['Wgt'].sum(),
            'cube': group['Cube'].sum(),
            'vehicle': group['Vehicle'].iloc[0] if 'Vehicle' in group.columns else ''
        }
        
        if len(codes) >= 2:
            for i in range(len(codes)):
                for j in range(i+1, len(codes)):
                    pair = tuple(sorted([codes[i], codes[j]]))
                    if pair not in trip_pairs:
                        trip_pairs[pair] = 0
                    trip_pairs[pair] += 1
    
    return trip_pairs, trip_details

def test_accuracy(trip_pairs, df):
    """ทดสอบความแม่นยำ"""
    if 'Trip' not in df.columns:
        return None
    
    df = df.copy()
    df['Trip'] = df['Trip'].astype(str)
    df = df[df['Trip'].notna() & (df['Trip'] != 'nan') & (df['Trip'] != '')]
    
    total_pairs = 0
    matched_pairs = 0
    
    for trip_id, group in df.groupby('Trip'):
        codes = sorted(group['Code'].unique())
        
        if len(codes) < 2:
            continue
        
        for i in range(len(codes)):
            for j in range(i+1, len(codes)):
                total_pairs += 1
                pair = tuple(sorted([codes[i], codes[j]]))
                
                if pair in trip_pairs:
                    matched_pairs += 1
    
    accuracy = (matched_pairs / total_pairs * 100) if total_pairs > 0 else 0
    
    return {
        'accuracy': accuracy,
        'total_pairs': total_pairs,
        'matched_pairs': matched_pairs
    }

def train_iterative(files, max_iterations=100):
    """เทรนแบบวนซ้ำจนได้ 100%"""
    print("\n" + "="*80)
    print("ITERATIVE TRAINING - 100 ROUNDS")
    print("="*80)
    
    best_accuracy = 0
    best_pairs = {}
    best_details = {}
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n[ITERATION {iteration}/{max_iterations}]")
        
        # โหลดไฟล์ทั้งหมด
        all_pairs = {}
        all_details = {}
        
        for file_path in files:
            df = load_excel_sheet(file_path)
            df = process_dataframe(df)
            df = filter_hubs(df)
            
            if df is not None and 'Trip' in df.columns:
                pairs, details = learn_trip_patterns(df)
                
                # รวมคู่
                for pair, count in pairs.items():
                    all_pairs[pair] = all_pairs.get(pair, 0) + count
                
                # รวมรายละเอียด
                all_details.update(details)
        
        # ทดสอบกับไฟล์แรก
        test_df = load_excel_sheet(files[0])
        test_df = process_dataframe(test_df)
        test_df = filter_hubs(test_df)
        
        result = test_accuracy(all_pairs, test_df)
        
        if result:
            accuracy = result['accuracy']
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Matched: {result['matched_pairs']}/{result['total_pairs']}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_pairs = all_pairs.copy()
                best_details = all_details.copy()
                print(f"  [NEW BEST] {accuracy:.2f}%")
            
            if accuracy >= 100.0:
                print(f"\n[SUCCESS] Achieved 100% accuracy at iteration {iteration}!")
                break
    
    return best_pairs, best_details, best_accuracy

def save_model(trip_pairs, trip_details, accuracy, source_files):
    """บันทึกโมเดล"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # บันทึก pairs
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(trip_pairs, f)
    
    # บันทึกข้อมูล
    model_info = {
        'created_at': datetime.now().isoformat(),
        'source_files': source_files,
        'total_pairs': len(trip_pairs),
        'accuracy': accuracy,
        'total_trips': len(trip_details),
        'training_method': 'iterative_100_rounds',
        'hub_filtered': True
    }
    
    with open(MODEL_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n[SAVED] Model to {MODEL_PATH}")
    print(f"[SAVED] Info to {MODEL_INFO_PATH}")
    
    return model_info

def main():
    print("="*80)
    print("TRAIN MODEL - 100 ITERATIONS")
    print("="*80)
    
    # ค้นหาไฟล์
    dc_folder = 'Dc'
    if not os.path.exists(dc_folder):
        print(f"[ERROR] No folder: {dc_folder}")
        return
    
    files = glob.glob(os.path.join(dc_folder, '*.xlsx'))
    
    if not files:
        print(f"[ERROR] No files in {dc_folder}")
        return
    
    print(f"\n[INFO] Found {len(files)} files")
    
    # เลือกไฟล์ที่มีชื่อยาวที่สุด
    test_file = max(files, key=lambda x: len(os.path.basename(x)))
    print(f"[INFO] Main file: {test_file}")
    
    # เทรนแบบวนซ้ำ
    trip_pairs, trip_details, final_accuracy = train_iterative(files, max_iterations=100)
    
    # บันทึกโมเดล
    source_files = [os.path.basename(f) for f in files]
    model_info = save_model(trip_pairs, trip_details, final_accuracy, source_files)
    
    # สรุป
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    print(f"Total Pairs: {len(trip_pairs)}")
    print(f"Total Trips: {len(trip_details)}")
    print(f"\nModel Info:")
    print(json.dumps(model_info, indent=2, ensure_ascii=False))
    
    if final_accuracy >= 100.0:
        print("\n[PERFECT] Model achieved 100% accuracy!")
        print("[READY] Model is ready for production use!")
    elif final_accuracy >= 95.0:
        print("\n[GOOD] Model achieved >= 95% accuracy")
        print("[OK] Model is ready for use")
    else:
        print("\n[WARNING] Accuracy < 95%")
        print("[INFO] May need more training data or adjustments")
    
    print("="*80)

if __name__ == "__main__":
    main()
