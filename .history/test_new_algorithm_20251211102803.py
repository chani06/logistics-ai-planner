"""
ทดสอบ Farthest First + Nearest Neighbor Algorithm
"""

import pandas as pd
import sys
import os
from math import radians, cos, sin, asin, sqrt

# เพิ่ม path ของโปรเจค
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def haversine_distance(lat1, lon1, lat2, lon2):
    """คำนวณระยะทางระหว่าง 2 จุด (กม.)"""
    if None in [lat1, lon1, lat2, lon2]:
        return 0
    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 6371 * c
    except:
        return 0

def load_data():
    """โหลดข้อมูล Punthai และ Master"""
    file_path = "Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx"
    master_path = "Dc/Master สถานที่ส่ง.xlsx"
    
    print(f"Loading: {file_path}")
    
    # โหลดข้อมูล
    df = pd.read_excel(file_path, header=1)
    
    # Rename columns
    col_map = {
        'สาขา': 'Code',
        'ชื่อสาขา': 'Name',
        'TOTALWGT': 'Weight',
        'TOTALCUBE': 'Cube'
    }
    df = df.rename(columns=col_map)
    
    # กรองเฉพาะแถวที่มี Code
    df = df[df['Code'].notna() & (df['Code'] != '')]
    df['Code'] = df['Code'].astype(str)
    
    # แปลง Weight/Cube เป็นตัวเลข
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce').fillna(0)
    df['Cube'] = pd.to_numeric(df['Cube'], errors='coerce').fillna(0)
    
    # รวม Weight/Cube ตาม Code
    df_grouped = df.groupby('Code').agg({
        'Name': 'first',
        'Weight': 'sum',
        'Cube': 'sum'
    }).reset_index()
    
    print(f"Branches: {len(df_grouped)}")
    print(f"Total Weight: {df_grouped['Weight'].sum():,.0f} kg")
    print(f"Total Cube: {df_grouped['Cube'].sum():,.2f}")
    
    # โหลด Master
    print(f"\nLoading Master: {master_path}")
    master_df = pd.read_excel(master_path)
    print(f"Master Data: {len(master_df)} records")
    
    return df_grouped, master_df

def analyze_trips(df, master_df):
    """วิเคราะห์ผลการจัดทริป"""
    
    # สร้าง coord lookup จาก Master
    coord_lookup = {}
    for _, row in master_df.iterrows():
        code = str(row.get('Plan Code', ''))
        if not code:
            continue
        lat = row.get('ละติจูด')
        lon = row.get('ลองติจูด')
        if pd.notna(lat) and pd.notna(lon):
            coord_lookup[code] = (float(lat), float(lon))
    
    # ดึงรายการ Trip
    trips = df.groupby('Trip').apply(lambda x: x.to_dict('records')).to_dict()
    
    LIMITS = {
        '4W': {'max_w': 2500, 'max_c': 5.0},
        'JB': {'max_w': 3500, 'max_c': 7.0},
        '6W': {'max_w': 6000, 'max_c': 20.0}
    }
    
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    # วิเคราะห์
    issues = {'over_100': [], 'far_30km': []}
    vehicle_counts = {'4W': 0, 'JB': 0, '6W': 0}
    
    for trip_num, branches in sorted(trips.items()):
        if pd.isna(trip_num):
            continue
            
        trip_weight = sum(b['Weight'] for b in branches)
        trip_cube = sum(b['Cube'] for b in branches)
        
        # เลือกรถ
        if trip_cube <= 5.0 and trip_weight <= 2500:
            vehicle = '4W'
        elif trip_cube <= 7.0 and trip_weight <= 3500:
            vehicle = 'JB'
        else:
            vehicle = '6W'
        
        vehicle_counts[vehicle] = vehicle_counts.get(vehicle, 0) + 1
        
        # เช็คเกิน 100%
        limit = LIMITS[vehicle]
        w_util = (trip_weight / limit['max_w']) * 100
        c_util = (trip_cube / limit['max_c']) * 100
        util = max(w_util, c_util)
        
        if util > 100:
            issues['over_100'].append({
                'trip': trip_num,
                'vehicle': vehicle,
                'util': util,
                'branches': len(branches)
            })
        
        # เช็คระยะห่าง
        codes = [str(b['Code']) for b in branches]
        max_dist = 0
        far_pair = None
        for i, c1 in enumerate(codes):
            for c2 in codes[i+1:]:
                if c1 in coord_lookup and c2 in coord_lookup:
                    dist = haversine_distance(
                        coord_lookup[c1][0], coord_lookup[c1][1],
                        coord_lookup[c2][0], coord_lookup[c2][1]
                    )
                    if dist > max_dist:
                        max_dist = dist
                        far_pair = (c1, c2)
        
        if max_dist > 30:
            issues['far_30km'].append({
                'trip': trip_num,
                'max_dist': max_dist,
                'branches': len(branches),
                'codes': codes[:5]  # แสดง 5 สาขาแรก
            })
    
    # แสดงผล
    total_trips = sum(vehicle_counts.values())
    print(f"\nTotal Trips: {total_trips}")
    print(f"  4W: {vehicle_counts.get('4W', 0)}")
    print(f"  JB: {vehicle_counts.get('JB', 0)}")
    print(f"  6W: {vehicle_counts.get('6W', 0)}")
    
    print(f"\nISSUES:")
    print(f"  Trips > 100%: {len(issues['over_100'])}")
    for issue in issues['over_100'][:5]:
        print(f"    Trip {issue['trip']}: {issue['util']:.1f}% ({issue['vehicle']}, {issue['branches']} branches)")
    
    print(f"\n  Trips > 30km apart: {len(issues['far_30km'])}")
    for issue in issues['far_30km'][:10]:
        print(f"    Trip {issue['trip']}: {issue['max_dist']:.1f}km ({issue['branches']} branches)")
        print(f"      Branches: {issue['codes']}")
    
    return vehicle_counts, issues

def main():
    # โหลดข้อมูล
    df, master_df = load_data()
    
    # Import predict_trips จาก app.py
    print("\n" + "="*70)
    print("RUNNING NEW ALGORITHM (Farthest First + Nearest Neighbor)")
    print("="*70)
    
    # ต้อง set Master Data ก่อน
    import app
    app.MASTER_DATA = master_df
    
    # เรียก predict_trips
    # สร้าง model_data ที่ต้องการ
    model_data = {
        'model': None,
        'trip_pairs': set(),
        'branch_info': {},
        'trip_vehicles': {},
        'branch_vehicles': {}
    }
    
    result_df, summary_df = app.predict_trips(df, model_data=model_data)
    
    print(f"\nResult columns: {result_df.columns.tolist()}")
    print(f"Result shape: {result_df.shape}")
    print(f"Trip column: {'Trip' in result_df.columns}")
    if 'Trip' in result_df.columns:
        print(f"Unique trips: {result_df['Trip'].nunique()}")
    
    # วิเคราะห์
    vehicle_counts, issues = analyze_trips(result_df, master_df)
    
    # แสดง 20 ทริปแรก
    print("\n" + "="*70)
    print("FIRST 20 TRIPS")
    print("="*70)
    
    trips = result_df.groupby('Trip').apply(lambda x: {
        'codes': x['Code'].tolist(),
        'names': x['Name'].tolist() if 'Name' in x.columns else [],
        'weight': x['Weight'].sum(),
        'cube': x['Cube'].sum()
    }).to_dict()
    
    for trip_num in sorted(trips.keys())[:20]:
        trip = trips[trip_num]
        vehicle = '4W' if trip['cube'] <= 5 else ('JB' if trip['cube'] <= 7 else '6W')
        
        # หาระยะห่างสูงสุด
        coord_lookup = {}
        for _, row in master_df.iterrows():
            code = str(row.get('Plan Code', ''))
            lat = row.get('ละติจูด')
            lon = row.get('ลองติจูด')
            if code and pd.notna(lat) and pd.notna(lon):
                coord_lookup[code] = (float(lat), float(lon))
        
        codes = trip['codes']
        max_dist = 0
        for i, c1 in enumerate(codes):
            for c2 in codes[i+1:]:
                if c1 in coord_lookup and c2 in coord_lookup:
                    dist = haversine_distance(
                        coord_lookup[c1][0], coord_lookup[c1][1],
                        coord_lookup[c2][0], coord_lookup[c2][1]
                    )
                    max_dist = max(max_dist, dist)
        
        print(f"\nTrip {trip_num} ({vehicle}): {len(codes)} branches, {trip['cube']:.2f} cube, max_dist: {max_dist:.1f}km")
        for i, (code, name) in enumerate(zip(codes, trip['names'])):
            print(f"  {i+1}. {code}: {name}")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == "__main__":
    main()
