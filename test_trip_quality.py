"""
ทดสอบคุณภาพการจัดทริป - ตรวจสอบว่าทริปถูกต้องตามกฎหรือไม่
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

# ขีดจำกัดรถ
LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5.0, 'max_drops': 12},
    'JB': {'max_w': 3500, 'max_c': 7.0, 'max_drops': 12},  # Punthai = 7 drops
    '6W': {'max_w': 6000, 'max_c': 20.0, 'max_drops': 999}
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """คำนวณระยะทางระหว่าง 2 จุด (กม.)"""
    if None in [lat1, lon1, lat2, lon2]:
        return 0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def load_and_analyze(file_path):
    """โหลดไฟล์และวิเคราะห์"""
    print(f"\n📂 กำลังโหลด: {file_path}")
    
    # โหลด Master Data
    master_path = "Dc/Master สถานที่ส่ง.xlsx"
    try:
        master_df = pd.read_excel(master_path)
        print(f"✅ โหลด Master Data: {len(master_df)} สาขา")
    except Exception as e:
        print(f"❌ ไม่สามารถโหลด Master: {e}")
        master_df = pd.DataFrame()
    
    # โหลดไฟล์ทดสอบ
    try:
        df = pd.read_excel(file_path)
        print(f"✅ โหลดไฟล์: {len(df)} แถว")
    except Exception as e:
        print(f"❌ ไม่สามารถโหลด: {e}")
        return
    
    # แสดงคอลัมน์
    print(f"📋 คอลัมน์: {list(df.columns)[:10]}...")
    
    return df, master_df

def check_trip_quality(result_df, master_df):
    """ตรวจสอบคุณภาพทริป"""
    print("\n" + "="*60)
    print("🔍 ตรวจสอบคุณภาพทริป")
    print("="*60)
    
    if 'Trip' not in result_df.columns:
        print("❌ ไม่พบคอลัมน์ Trip")
        return
    
    # สร้าง lookup สำหรับพิกัด
    coord_lookup = {}
    province_lookup = {}
    if not master_df.empty and 'Plan Code' in master_df.columns:
        for _, row in master_df.iterrows():
            code = str(row['Plan Code'])
            lat = row.get('ละติจูด') or row.get('Latitude')
            lon = row.get('ลองติจูด') or row.get('Longitude')
            prov = row.get('จังหวัด') or row.get('Province', '')
            if pd.notna(lat) and pd.notna(lon):
                coord_lookup[code] = (float(lat), float(lon))
            if pd.notna(prov):
                province_lookup[code] = str(prov)
    
    # วิเคราะห์แต่ละทริป
    issues = {
        'over_capacity': [],      # เกิน 100%
        'far_branches': [],       # สาขาห่างกันเกิน 30km
        'cross_province': [],     # ข้ามจังหวัด
        'wrong_vehicle': []       # รถไม่เหมาะสม
    }
    
    trip_stats = []
    
    for trip_num in sorted(result_df['Trip'].dropna().unique()):
        trip_data = result_df[result_df['Trip'] == trip_num]
        codes = list(trip_data['Code'].values) if 'Code' in trip_data.columns else []
        
        # คำนวณ Weight/Cube
        total_w = trip_data['Weight'].sum() if 'Weight' in trip_data.columns else 0
        total_c = trip_data['Cube'].sum() if 'Cube' in trip_data.columns else 0
        branch_count = len(trip_data)
        
        # หารถที่แนะนำ
        vehicle = trip_data['Recommended_Vehicle'].iloc[0] if 'Recommended_Vehicle' in trip_data.columns else '6W'
        
        # คำนวณ utilization
        if vehicle in LIMITS:
            w_util = (total_w / LIMITS[vehicle]['max_w']) * 100
            c_util = (total_c / LIMITS[vehicle]['max_c']) * 100
        else:
            w_util = c_util = 0
        
        max_util = max(w_util, c_util)
        
        # เช็ค 1: เกิน 100%
        if max_util > 100:
            issues['over_capacity'].append({
                'trip': trip_num,
                'vehicle': vehicle,
                'weight_util': f"{w_util:.1f}%",
                'cube_util': f"{c_util:.1f}%"
            })
        
        # เช็ค 2: ระยะห่างสาขา
        max_distance = 0
        for i, code1 in enumerate(codes):
            for code2 in codes[i+1:]:
                if str(code1) in coord_lookup and str(code2) in coord_lookup:
                    lat1, lon1 = coord_lookup[str(code1)]
                    lat2, lon2 = coord_lookup[str(code2)]
                    dist = haversine_distance(lat1, lon1, lat2, lon2)
                    max_distance = max(max_distance, dist)
        
        if max_distance > 30:
            issues['far_branches'].append({
                'trip': trip_num,
                'max_distance': f"{max_distance:.1f} km",
                'branches': branch_count
            })
        
        # เช็ค 3: ข้ามจังหวัด
        provinces = set()
        for code in codes:
            prov = province_lookup.get(str(code), '')
            if prov:
                provinces.add(prov)
        
        if len(provinces) > 1:
            issues['cross_province'].append({
                'trip': trip_num,
                'provinces': list(provinces),
                'branches': branch_count
            })
        
        # เช็ค 4: รถไม่เหมาะสม
        # 4W ควร cube ≤ 5, JB ≤ 7, 6W ควร ≥ 18
        wrong = False
        reason = ""
        if vehicle == '4W' and total_c > 5:
            wrong = True
            reason = f"4W แต่ Cube = {total_c:.1f} (เกิน 5)"
        elif vehicle == 'JB' and total_c > 7:
            wrong = True
            reason = f"JB แต่ Cube = {total_c:.1f} (เกิน 7)"
        elif vehicle == '6W' and total_c < 15:
            wrong = True
            reason = f"6W แต่ Cube = {total_c:.1f} (ต่ำ ควรใช้ JB)"
        
        if wrong:
            issues['wrong_vehicle'].append({
                'trip': trip_num,
                'vehicle': vehicle,
                'reason': reason
            })
        
        trip_stats.append({
            'trip': trip_num,
            'vehicle': vehicle,
            'branches': branch_count,
            'weight': total_w,
            'cube': total_c,
            'w_util': w_util,
            'c_util': c_util,
            'max_dist': max_distance,
            'provinces': len(provinces)
        })
    
    # แสดงผลสรุป
    print(f"\n📊 สรุป: {len(trip_stats)} ทริป")
    print("-" * 60)
    
    # นับรถแต่ละประเภท
    vehicle_counts = {}
    for stat in trip_stats:
        v = stat['vehicle']
        vehicle_counts[v] = vehicle_counts.get(v, 0) + 1
    print(f"🚛 รถ: {vehicle_counts}")
    
    # แสดง issues
    print(f"\n⚠️ ปัญหาที่พบ:")
    print(f"   - เกิน 100%: {len(issues['over_capacity'])} ทริป")
    print(f"   - สาขาห่างกันเกิน 30km: {len(issues['far_branches'])} ทริป")
    print(f"   - ข้ามจังหวัด: {len(issues['cross_province'])} ทริป")
    print(f"   - รถไม่เหมาะสม: {len(issues['wrong_vehicle'])} ทริป")
    
    # แสดงรายละเอียด issues
    if issues['over_capacity']:
        print(f"\n🔴 ทริปที่เกิน 100%:")
        for item in issues['over_capacity'][:5]:
            print(f"   Trip {item['trip']}: {item['vehicle']} - W:{item['weight_util']} C:{item['cube_util']}")
    
    if issues['far_branches']:
        print(f"\n🟠 ทริปที่สาขาห่างกันเกิน 30km:")
        for item in issues['far_branches'][:5]:
            print(f"   Trip {item['trip']}: {item['max_distance']} ({item['branches']} สาขา)")
    
    if issues['cross_province']:
        print(f"\n🟡 ทริปที่ข้ามจังหวัด:")
        for item in issues['cross_province'][:5]:
            print(f"   Trip {item['trip']}: {item['provinces']}")
    
    if issues['wrong_vehicle']:
        print(f"\n🟣 ทริปที่รถไม่เหมาะสม:")
        for item in issues['wrong_vehicle'][:5]:
            print(f"   Trip {item['trip']}: {item['reason']}")
    
    # แสดง 10 ทริปแรก
    print(f"\n📋 ตัวอย่าง 10 ทริปแรก:")
    print("-" * 80)
    print(f"{'Trip':>5} {'Vehicle':>8} {'Branches':>8} {'Weight':>8} {'Cube':>8} {'W%':>8} {'C%':>8} {'MaxDist':>10}")
    print("-" * 80)
    for stat in trip_stats[:10]:
        print(f"{stat['trip']:>5} {stat['vehicle']:>8} {stat['branches']:>8} {stat['weight']:>8.0f} {stat['cube']:>8.1f} {stat['w_util']:>7.1f}% {stat['c_util']:>7.1f}% {stat['max_dist']:>9.1f}km")
    
    return issues, trip_stats

if __name__ == "__main__":
    # ทดสอบกับไฟล์ Punthai
    file_path = "Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx"
    
    df, master_df = load_and_analyze(file_path)
    
    if df is not None:
        # ตรวจสอบว่ามี Trip column หรือไม่
        if 'Trip' not in df.columns:
            print("\n⚠️ ไฟล์นี้ยังไม่ได้จัดทริป - ต้องรันผ่าน app.py ก่อน")
            print("📋 ข้อมูลดิบ:")
            print(df.head(10))
        else:
            check_trip_quality(df, master_df)
