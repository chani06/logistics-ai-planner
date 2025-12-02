"""
สคริปต์ทดสอบโมเดล - ตรวจสอบความแม่นยำ 100%
"""
import pandas as pd
import os
import glob
from datetime import datetime

# ==========================================
# FUNCTIONS
# ==========================================
def normalize(val):
    """ทำให้รหัสสาขาเป็นมาตรฐาน"""
    return str(val).strip().upper().replace(" ", "").replace(".0", "")

def load_excel_sheet(file_path, sheet_name=None):
    """โหลด Excel และหา sheet ที่ต้องการ"""
    try:
        xls = pd.ExcelFile(file_path)
        
        # หา sheet ที่มี "punthai" หรือ "2."
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
        
        print(f"[LOAD] Sheet: {target_sheet}")
        df = pd.read_excel(xls, sheet_name=target_sheet)
        return df
    except Exception as e:
        print(f"[ERROR] Loading {file_path}: {e}")
        return None

def process_dataframe(df):
    """แปลงคอลัมน์เป็นรูปแบบมาตรฐาน"""
    if df is None:
        return None
    
    print(f"📊 Columns: {list(df.columns[:10])}")
    
    # Rename columns
    rename_map = {}
    for col in df.columns:
        col_str = str(col).strip()
        col_upper = col_str.upper().replace(' ', '').replace('_', '')
        
        # Exact match first
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
        # Partial match (check if col is string first)
        elif isinstance(col, str):
            if 'BRANCHCODE' in col_upper or 'รหัสสาขา' in col:
                rename_map[col] = 'Code'
            elif 'WGT' in col_upper or 'น้ำหนัก' in col:
                rename_map[col] = 'Wgt'
            elif 'CUBE' in col_upper or 'คิว' in col:
                rename_map[col] = 'Cube'
            elif 'TRIPNO' in col_upper:
                rename_map[col] = 'Vehicle'
    
    df = df.rename(columns=rename_map)
    
    print(f"✅ Renamed columns: {list(df.columns[:10])}")
    
    # ตรวจสอบคอลัมน์ที่จำเป็น
    required = ['Code', 'Trip']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ Missing columns: {missing}")
        return None
    
    # Normalize Code
    df['Code'] = df['Code'].apply(normalize)
    
    # เติมค่าว่างสำหรับ Wgt, Cube
    if 'Wgt' not in df.columns:
        df['Wgt'] = 0.0
    if 'Cube' not in df.columns:
        df['Cube'] = 0.0
    
    df['Wgt'] = pd.to_numeric(df['Wgt'], errors='coerce').fillna(0.0)
    df['Cube'] = pd.to_numeric(df['Cube'], errors='coerce').fillna(0.0)
    
    return df.reset_index(drop=True)

def learn_trip_patterns(df):
    """เรียนรู้รูปแบบการจัดทริปจากไฟล์ประวัติ"""
    if 'Trip' not in df.columns or 'Code' not in df.columns:
        return {}, {}
    
    # แปลง Trip เป็น string
    df = df.copy()
    df['Trip'] = df['Trip'].astype(str)
    df = df[df['Trip'].notna() & (df['Trip'] != 'nan') & (df['Trip'] != '')]
    
    print(f"🔢 จำนวนทริปทั้งหมด: {df['Trip'].nunique()}")
    print(f"🏪 จำนวนสาขาทั้งหมด: {df['Code'].nunique()}")
    
    # Dictionary เก็บคู่ที่เคยไปด้วยกัน
    trip_pairs = {}
    
    # Dictionary เก็บข้อมูลทริปจริง
    trip_details = {}
    
    for trip_id, group in df.groupby('Trip'):
        codes = sorted(group['Code'].unique())
        total_wgt = group['Wgt'].sum()
        total_cube = group['Cube'].sum()
        vehicle = group['Vehicle'].iloc[0] if 'Vehicle' in group.columns else ''
        
        # บันทึกรายละเอียดทริป
        trip_details[trip_id] = {
            'codes': codes,
            'count': len(codes),
            'wgt': total_wgt,
            'cube': total_cube,
            'vehicle': vehicle
        }
        
        # บันทึกทุกคู่ในทริปนี้
        if len(codes) >= 2:
            for i in range(len(codes)):
                for j in range(i+1, len(codes)):
                    pair = tuple(sorted([codes[i], codes[j]]))
                    if pair not in trip_pairs:
                        trip_pairs[pair] = {
                            'count': 0,
                            'trips': []
                        }
                    trip_pairs[pair]['count'] += 1
                    trip_pairs[pair]['trips'].append(trip_id)
    
    print(f"✅ เรียนรู้ได้ {len(trip_pairs)} คู่")
    
    return trip_pairs, trip_details

def reconstruct_trips(df, trip_pairs):
    """สร้างทริปใหม่จากโมเดล - พยายามจับคู่ให้ตรงเป๊ะ"""
    if 'Code' not in df.columns:
        return None
    
    df = df.copy()
    df['Code'] = df['Code'].apply(normalize)
    
    # เก็บสาขาที่ยังไม่ได้จัด
    remaining_codes = set(df['Code'].unique())
    reconstructed_trips = []
    trip_id = 1
    
    print(f"\n🔄 เริ่มต้นจัดทริป: {len(remaining_codes)} สาขา")
    
    # สร้าง adjacency list สำหรับหาเพื่อนบ้าน
    adjacency = {}
    for (code1, code2), info in trip_pairs.items():
        if code1 not in adjacency:
            adjacency[code1] = []
        if code2 not in adjacency:
            adjacency[code2] = []
        adjacency[code1].append((code2, info['count']))
        adjacency[code2].append((code1, info['count']))
    
    # เรียงตาม connectivity (สาขาที่มีเพื่อนบ้านน้อยก่อน)
    def get_connectivity(code):
        if code not in adjacency:
            return 0
        return len([c for c, count in adjacency[code] if c in remaining_codes])
    
    while remaining_codes:
        # เลือกสาขาที่มี connectivity น้อยที่สุด (ป้องกันการแยกกลุ่ม)
        seed = min(remaining_codes, key=get_connectivity)
        
        current_trip = [seed]
        remaining_codes.remove(seed)
        
        # หาเพื่อนบ้านทั้งหมดที่เคยไปด้วยกัน
        neighbors = []
        if seed in adjacency:
            for neighbor, count in adjacency[seed]:
                if neighbor in remaining_codes:
                    neighbors.append((neighbor, count))
        
        # เรียงตามจำนวนครั้งที่เคยไปด้วยกัน (มากไปน้อย)
        neighbors.sort(key=lambda x: x[1], reverse=True)
        
        # เพิ่มเพื่อนบ้านเข้าทริป
        for neighbor, count in neighbors:
            if neighbor in remaining_codes:
                # ตรวจสอบว่า neighbor นี้เคยไปกับสมาชิกในทริปปัจจุบันหรือไม่
                can_add = True
                for member in current_trip:
                    pair = tuple(sorted([member, neighbor]))
                    if pair not in trip_pairs:
                        can_add = False
                        break
                
                if can_add:
                    current_trip.append(neighbor)
                    remaining_codes.remove(neighbor)
        
        # บันทึกทริป
        for code in current_trip:
            reconstructed_trips.append({
                'Code': code,
                'ReconstructedTrip': f"AI-{trip_id:03d}",
                'TripSize': len(current_trip)
            })
        
        print(f"  Trip AI-{trip_id:03d}: {len(current_trip)} สาขา - {current_trip[:5]}{'...' if len(current_trip) > 5 else ''}")
        trip_id += 1
    
    print(f"✅ สร้างได้ {trip_id - 1} ทริป")
    
    # รวมกับข้อมูลเดิม
    recon_df = pd.DataFrame(reconstructed_trips)
    result = df.merge(recon_df, on='Code', how='left')
    
    return result

def compare_trips(df, trip_pairs):
    """เปรียบเทียบทริปที่สร้างใหม่กับทริปเดิม"""
    if 'Trip' not in df.columns or 'ReconstructedTrip' not in df.columns:
        return None
    
    df = df.copy()
    df['Trip'] = df['Trip'].astype(str)
    df = df[df['Trip'].notna() & (df['Trip'] != 'nan') & (df['Trip'] != '')]
    
    print("\n" + "="*80)
    print("📊 เปรียบเทียบทริปเดิม VS ทริปที่สร้างใหม่")
    print("="*80)
    
    # สถิติโดยรวม
    total_pairs = 0
    matched_pairs = 0
    missing_pairs = []
    extra_pairs = []
    
    # วิเคราะห์ทริปเดิม
    original_trips = {}
    for trip_id, group in df.groupby('Trip'):
        codes = sorted(group['Code'].unique())
        original_trips[trip_id] = {
            'codes': codes,
            'wgt': group['Wgt'].sum(),
            'cube': group['Cube'].sum(),
            'vehicle': group['Vehicle'].iloc[0] if 'Vehicle' in group.columns else ''
        }
    
    # วิเคราะห์ทริปที่สร้างใหม่
    reconstructed_trips = {}
    for trip_id, group in df.groupby('ReconstructedTrip'):
        codes = sorted(group['Code'].unique())
        reconstructed_trips[trip_id] = {
            'codes': codes,
            'wgt': group['Wgt'].sum(),
            'cube': group['Cube'].sum()
        }
    
    print(f"\n📈 สถิติ:")
    print(f"  • ทริปเดิม: {len(original_trips)} ทริป")
    print(f"  • ทริปใหม่: {len(reconstructed_trips)} ทริป")
    
    # ตรวจสอบความถูกต้องของคู่
    print(f"\n🔍 ตรวจสอบคู่สาขา:")
    
    for trip_id, info in original_trips.items():
        codes = info['codes']
        
        if len(codes) < 2:
            continue
        
        for i in range(len(codes)):
            for j in range(i+1, len(codes)):
                total_pairs += 1
                pair = tuple(sorted([codes[i], codes[j]]))
                
                # ตรวจสอบว่าในทริปที่สร้างใหม่ คู่นี้อยู่ด้วยกันหรือไม่
                code1_new_trip = df[df['Code'] == codes[i]]['ReconstructedTrip'].iloc[0]
                code2_new_trip = df[df['Code'] == codes[j]]['ReconstructedTrip'].iloc[0]
                
                if code1_new_trip == code2_new_trip:
                    matched_pairs += 1
                else:
                    missing_pairs.append({
                        'original_trip': trip_id,
                        'code1': codes[i],
                        'code2': codes[j],
                        'new_trip1': code1_new_trip,
                        'new_trip2': code2_new_trip
                    })
    
    # ตรวจสอบคู่ที่สร้างเกิน (ไม่มีในประวัติ)
    for trip_id, info in reconstructed_trips.items():
        codes = info['codes']
        
        if len(codes) < 2:
            continue
        
        for i in range(len(codes)):
            for j in range(i+1, len(codes)):
                pair = tuple(sorted([codes[i], codes[j]]))
                
                if pair not in trip_pairs:
                    extra_pairs.append({
                        'reconstructed_trip': trip_id,
                        'code1': codes[i],
                        'code2': codes[j]
                    })
    
    accuracy = (matched_pairs / total_pairs * 100) if total_pairs > 0 else 0
    
    print(f"  • คู่ทั้งหมด: {total_pairs}")
    print(f"  • คู่ที่ตรงกัน: {matched_pairs}")
    print(f"  • คู่ที่หายไป: {len(missing_pairs)}")
    print(f"  • คู่ที่เกิน (ไม่มีในประวัติ): {len(extra_pairs)}")
    print(f"\n{'='*80}")
    print(f"✨ ความแม่นยำ: {accuracy:.2f}%")
    print(f"{'='*80}")
    
    # แสดงรายละเอียดคู่ที่หายไป
    if missing_pairs:
        print(f"\n❌ คู่ที่หายไป (แสดง 20 คู่แรก):")
        for idx, pair in enumerate(missing_pairs[:20], 1):
            print(f"  {idx}. Trip {pair['original_trip']}: {pair['code1']} ↔ {pair['code2']}")
            print(f"      → ถูกแยกไปคนละทริป: {pair['new_trip1']} และ {pair['new_trip2']}")
    
    # แสดงรายละเอียดคู่ที่เกิน
    if extra_pairs:
        print(f"\n⚠️ คู่ที่เกิน (ไม่มีในประวัติ) - แสดง 20 คู่แรก:")
        for idx, pair in enumerate(extra_pairs[:20], 1):
            print(f"  {idx}. Trip {pair['reconstructed_trip']}: {pair['code1']} ↔ {pair['code2']}")
            print(f"      → คู่นี้ไม่เคยไปด้วยกันในประวัติ!")
    
    # เปรียบเทียบน้ำหนักและคิว
    print(f"\n⚖️ เปรียบเทียบน้ำหนักและคิว:")
    print(f"{'Trip เดิม':<15} {'Wgt เดิม':<12} {'Cube เดิม':<12} | {'Trip ใหม่':<15} {'Wgt ใหม่':<12} {'Cube ใหม่':<12}")
    print("-" * 90)
    
    # แสดง 10 ทริปแรก
    for idx, (trip_id, info) in enumerate(list(original_trips.items())[:10]):
        wgt = info['wgt']
        cube = info['cube']
        
        # หาทริปใหม่ที่มีสาขาเดียวกัน
        first_code = info['codes'][0]
        new_trip_id = df[df['Code'] == first_code]['ReconstructedTrip'].iloc[0]
        new_info = reconstructed_trips.get(new_trip_id, {})
        new_wgt = new_info.get('wgt', 0)
        new_cube = new_info.get('cube', 0)
        
        wgt_match = "✅" if abs(wgt - new_wgt) < 0.1 else "❌"
        cube_match = "✅" if abs(cube - new_cube) < 0.1 else "❌"
        
        print(f"{trip_id:<15} {wgt:<12.2f} {cube:<12.2f} | {new_trip_id:<15} {new_wgt:<12.2f} {cube_match} {new_cube:<12.2f} {wgt_match}")
    
    return {
        'accuracy': accuracy,
        'total_pairs': total_pairs,
        'matched_pairs': matched_pairs,
        'missing_pairs': missing_pairs,
        'extra_pairs': extra_pairs,
        'original_trip_count': len(original_trips),
        'reconstructed_trip_count': len(reconstructed_trips)
    }

# ==========================================
# MAIN TEST
# ==========================================
def main():
    print("="*80)
    print("TEST: Logistics AI Model")
    print("="*80)
    
    # ค้นหาไฟล์ใน DC folder
    dc_folder = 'Dc'
    if not os.path.exists(dc_folder):
        print(f"❌ ไม่พบโฟลเดอร์ {dc_folder}")
        return
    
    files = glob.glob(os.path.join(dc_folder, '*.xlsx'))
    
    if not files:
        print(f"[ERROR] No files in {dc_folder}")
        return
    
    print(f"\n[INFO] Found {len(files)} files:")
    for f in files:
        print(f"  - {os.path.basename(f)}")
    
    # เลือกไฟล์แรกสำหรับทดสอบ
    test_file = files[0]
    print(f"\n[TEST] Using file: {os.path.basename(test_file)}")
    print("="*80)
    
    # โหลดและประมวลผลไฟล์
    print("\n[STEP 1] Loading file...")
    df = load_excel_sheet(test_file)
    df = process_dataframe(df)
    
    if df is None:
        print("[ERROR] Cannot load file")
        return
    
    print(f"[OK] Loaded {len(df)} rows")
    
    # เรียนรู้รูปแบบทริป
    print("\n[STEP 2] Learning trip patterns...")
    trip_pairs, trip_details = learn_trip_patterns(df)
    
    # สร้างทริปใหม่
    print("\n[STEP 3] Reconstructing trips...")
    result = reconstruct_trips(df, trip_pairs)
    
    if result is None:
        print("[ERROR] Cannot reconstruct trips")
        return
    
    # เปรียบเทียบผลลัพธ์
    print("\n[STEP 4] Comparing results...")
    comparison = compare_trips(result, trip_pairs)
    
    # สรุปผล
    if comparison:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        if comparison['accuracy'] >= 100:
            print("[EXCELLENT] Accuracy 100% - Ready to use!")
        elif comparison['accuracy'] >= 95:
            print("[GOOD] Accuracy >= 95% - Can use")
        elif comparison['accuracy'] >= 80:
            print("[FAIR] Accuracy >= 80% - Need improvement")
        else:
            print("[BAD] Accuracy < 80% - Must fix")
        
        print(f"\nDetails:")
        print(f"  - Accuracy: {comparison['accuracy']:.2f}%")
        print(f"  - Matched pairs: {comparison['matched_pairs']}/{comparison['total_pairs']}")
        print(f"  - Original trips: {comparison['original_trip_count']}")
        print(f"  - Reconstructed trips: {comparison['reconstructed_trip_count']}")
        print(f"  - Difference: {abs(comparison['original_trip_count'] - comparison['reconstructed_trip_count'])} trips")
        
        # บันทึกผลลัพธ์
        output_file = f"test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        result.to_excel(output_file, index=False)
        print(f"\n[SAVED] Output: {output_file}")
        
        print("="*80)

if __name__ == "__main__":
    main()
