"""
ทดสอบการจัดทริปจริง - โหลดไฟล์ Punthai และตรวจสอบผลลัพธ์
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import sys
import os

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

def load_punthai_file():
    """โหลดและเตรียมข้อมูล Punthai"""
    file_path = "Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx"
    
    print(f"\n📂 โหลด: {file_path}")
    
    # โหลดข้อมูล - skip first row (blank)
    df = pd.read_excel(file_path, header=1)
    
    # แสดงคอลัมน์
    print(f"📋 คอลัมน์: {list(df.columns)}")
    
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
    
    print(f"✅ สาขาทั้งหมด: {len(df_grouped)}")
    print(f"📊 Weight รวม: {df_grouped['Weight'].sum():,.0f} kg")
    print(f"📦 Cube รวม: {df_grouped['Cube'].sum():,.2f}")
    
    return df_grouped

def load_master_data():
    """โหลด Master Data"""
    master_path = "Dc/Master สถานที่ส่ง.xlsx"
    print(f"\n📂 โหลด Master: {master_path}")
    
    try:
        master_df = pd.read_excel(master_path)
        print(f"✅ Master Data: {len(master_df)} สาขา")
        
        # สร้าง lookup
        coord_lookup = {}
        province_lookup = {}
        subdistrict_lookup = {}
        
        for _, row in master_df.iterrows():
            code = str(row.get('Plan Code', ''))
            if not code:
                continue
                
            lat = row.get('ละติจูด') or row.get('Latitude')
            lon = row.get('ลองติจูด') or row.get('Longitude')
            prov = row.get('จังหวัด') or row.get('Province', '')
            subdist = row.get('ตำบล', '')
            district = row.get('อำเภอ', '')
            
            if pd.notna(lat) and pd.notna(lon):
                coord_lookup[code] = (float(lat), float(lon))
            if pd.notna(prov):
                province_lookup[code] = str(prov)
            if pd.notna(subdist):
                subdistrict_lookup[code] = (str(subdist), str(district) if pd.notna(district) else '')
        
        return coord_lookup, province_lookup, subdistrict_lookup
    except Exception as e:
        print(f"❌ Error: {e}")
        return {}, {}, {}

def simple_trip_assignment(df, coord_lookup, province_lookup, subdistrict_lookup):
    """จัดทริปแบบง่าย - เน้นจังหวัดเดียวกัน + ใกล้กัน"""
    print("\n" + "="*60)
    print("🚀 เริ่มจัดทริป")
    print("="*60)
    
    # กฎการจัดทริป
    LIMITS = {
        '4W': {'max_w': 2500, 'max_c': 5.0, 'max_drops': 12},
        'JB': {'max_w': 3500, 'max_c': 7.0, 'max_drops': 7},  # Punthai = 7 drops
        '6W': {'max_w': 6000, 'max_c': 20.0, 'max_drops': 999}
    }
    
    MAX_DISTANCE = 25  # km - ระยะห่างสูงสุดในทริป
    
    # จัดกลุ่มตามจังหวัด
    codes_by_province = {}
    for _, row in df.iterrows():
        code = str(row['Code'])
        prov = province_lookup.get(code, 'UNKNOWN')
        if prov not in codes_by_province:
            codes_by_province[prov] = []
        codes_by_province[prov].append(row)
    
    print(f"📍 จังหวัดทั้งหมด: {len(codes_by_province)}")
    
    trips = []
    trip_num = 1
    
    for prov, branches in codes_by_province.items():
        print(f"\n🏷️ จังหวัด: {prov} ({len(branches)} สาขา)")
        
        remaining = list(branches)
        
        while remaining:
            # เริ่มทริปใหม่
            current_trip = []
            trip_weight = 0
            trip_cube = 0
            
            # เลือก seed (สาขาแรก)
            seed = remaining.pop(0)
            current_trip.append(seed)
            trip_weight = seed['Weight']
            trip_cube = seed['Cube']
            seed_code = str(seed['Code'])
            seed_coord = coord_lookup.get(seed_code)
            
            # หาสาขาที่ใกล้กันและใส่ได้
            i = 0
            while i < len(remaining):
                branch = remaining[i]
                code = str(branch['Code'])
                coord = coord_lookup.get(code)
                
                # เช็คระยะทาง
                if seed_coord and coord:
                    dist = haversine_distance(seed_coord[0], seed_coord[1], coord[0], coord[1])
                else:
                    dist = 0
                
                # เช็คว่าใส่ได้หรือไม่
                new_weight = trip_weight + branch['Weight']
                new_cube = trip_cube + branch['Cube']
                
                # เลือกรถ
                if new_cube <= 5.0:
                    vehicle = '4W'
                elif new_cube <= 7.0:
                    vehicle = 'JB'
                else:
                    vehicle = '6W'
                
                limit = LIMITS[vehicle]
                can_fit = (new_weight <= limit['max_w'] and 
                          new_cube <= limit['max_c'] and 
                          len(current_trip) < limit['max_drops'] and
                          dist <= MAX_DISTANCE)
                
                if can_fit:
                    current_trip.append(branch)
                    trip_weight = new_weight
                    trip_cube = new_cube
                    remaining.pop(i)
                else:
                    i += 1
            
            # บันทึกทริป
            # เลือกรถ
            if trip_cube <= 5.0 and trip_weight <= 2500:
                vehicle = '4W'
            elif trip_cube <= 7.0 and trip_weight <= 3500:
                vehicle = 'JB'
            else:
                vehicle = '6W'
            
            trips.append({
                'trip_num': trip_num,
                'vehicle': vehicle,
                'branches': current_trip,
                'weight': trip_weight,
                'cube': trip_cube,
                'province': prov
            })
            trip_num += 1
    
    return trips

def analyze_trips(trips, coord_lookup):
    """วิเคราะห์ผลการจัดทริป"""
    print("\n" + "="*60)
    print("📊 ผลการจัดทริป")
    print("="*60)
    
    LIMITS = {
        '4W': {'max_w': 2500, 'max_c': 5.0},
        'JB': {'max_w': 3500, 'max_c': 7.0},
        '6W': {'max_w': 6000, 'max_c': 20.0}
    }
    
    # นับรถ
    vehicle_counts = {}
    issues = {'over': 0, 'far': 0}
    
    for trip in trips:
        v = trip['vehicle']
        vehicle_counts[v] = vehicle_counts.get(v, 0) + 1
        
        # เช็คเกิน 100%
        limit = LIMITS[v]
        w_util = (trip['weight'] / limit['max_w']) * 100
        c_util = (trip['cube'] / limit['max_c']) * 100
        if max(w_util, c_util) > 100:
            issues['over'] += 1
        
        # เช็คระยะห่าง
        codes = [str(b['Code']) for b in trip['branches']]
        max_dist = 0
        for i, c1 in enumerate(codes):
            for c2 in codes[i+1:]:
                if c1 in coord_lookup and c2 in coord_lookup:
                    dist = haversine_distance(
                        coord_lookup[c1][0], coord_lookup[c1][1],
                        coord_lookup[c2][0], coord_lookup[c2][1]
                    )
                    max_dist = max(max_dist, dist)
        if max_dist > 30:
            issues['far'] += 1
    
    print(f"\n🚛 จำนวนรถ: {sum(vehicle_counts.values())} คัน")
    for v, count in sorted(vehicle_counts.items()):
        print(f"   {v}: {count} คัน")
    
    print(f"\n⚠️ ปัญหา:")
    print(f"   - ทริปเกิน 100%: {issues['over']}")
    print(f"   - ทริปห่างเกิน 30km: {issues['far']}")
    
    # แสดง 10 ทริปแรก
    print(f"\n📋 ตัวอย่าง 10 ทริปแรก:")
    print("-" * 80)
    print(f"{'Trip':>5} {'Vehicle':>8} {'Branches':>8} {'Weight':>10} {'Cube':>8} {'Province':>15}")
    print("-" * 80)
    for trip in trips[:10]:
        print(f"{trip['trip_num']:>5} {trip['vehicle']:>8} {len(trip['branches']):>8} {trip['weight']:>10.0f} {trip['cube']:>8.2f} {trip['province'][:15]:>15}")
    
    return vehicle_counts, issues

if __name__ == "__main__":
    # โหลดข้อมูล
    df = load_punthai_file()
    coord_lookup, province_lookup, subdistrict_lookup = load_master_data()
    
    # จัดทริป
    trips = simple_trip_assignment(df, coord_lookup, province_lookup, subdistrict_lookup)
    
    # วิเคราะห์
    vehicle_counts, issues = analyze_trips(trips, coord_lookup)
    
    print("\n" + "="*60)
    print("✅ เสร็จสิ้น")
    print("="*60)
