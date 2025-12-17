"""
🔍 ทดสอบการจัดทริป - วิเคราะห์ระยะห่างและการกระโดดข้ามทริป
"""
import pandas as pd
import math

# ไฟล์
DATA_FILE = r"Dc\แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx"
MASTER_FILE = r"Dc\Master สถานที่ส่ง.xlsx"

# ค่าคงที่รถ (ตรงกับ app.py)
LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5.0},
    'JB': {'max_w': 3500, 'max_c': 8.0},  
    '6W': {'max_w': 6000, 'max_c': 20.0}
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """คำนวณระยะทางระหว่าง 2 จุด (km)"""
    if not all([lat1, lon1, lat2, lon2]):
        return None
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

print("=" * 80)
print("🔍 วิเคราะห์การจัดทริป - ตรวจสอบการกระโดดข้ามทริป")
print("=" * 80)

# อ่านไฟล์
print("\n📂 อ่านไฟล์...")
df = pd.read_excel(DATA_FILE, sheet_name='2.Punthai', header=1)
df = df[df['BranchCode'].notna()].copy()
df = df[~df['BranchCode'].isin(['DC011', 'PTDC'])].copy()  # ตัด DC ออก
print(f"   ✅ ข้อมูล: {len(df)} สาขา")

# อ่าน Master เพื่อดึงพิกัดและจังหวัด
df_master = pd.read_excel(MASTER_FILE)
coord_map = {}
province_map = {}
district_map = {}

for _, row in df_master.iterrows():
    code = str(row.get('Plan Code', '')).strip()
    lat = row.get('ละติจูด')
    lon = row.get('ลองติจูด')
    province = row.get('จังหวัด', '')
    district = row.get('อำเภอ', '')
    
    if code and pd.notna(lat) and pd.notna(lon):
        coord_map[code] = (lat, lon)
        province_map[code] = province
        district_map[code] = district

print(f"   ✅ พิกัด: {len(coord_map)} สาขา")

# วิเคราะห์แต่ละทริป
print("\n" + "=" * 80)
print("📊 วิเคราะห์ระยะห่างในแต่ละทริป")
print("=" * 80)

swap_candidates = []  # สาขาที่ควรสลับ

for trip_no in sorted(df['Trip no'].unique()):
    if pd.isna(trip_no):
        continue
    
    trip_data = df[df['Trip no'] == trip_no].copy()
    trip_codes = list(trip_data['BranchCode'].values)
    
    if len(trip_codes) < 2:
        continue
    
    # หา centroid และสาขาที่ไกลที่สุด
    lats, lons = [], []
    for code in trip_codes:
        coord = coord_map.get(str(code).strip())
        if coord:
            lats.append(coord[0])
            lons.append(coord[1])
    
    if not lats:
        continue
    
    centroid_lat = sum(lats) / len(lats)
    centroid_lon = sum(lons) / len(lons)
    
    # หาสาขาที่ไกลจาก centroid มากที่สุด
    farthest_code = None
    farthest_dist = 0
    branch_distances = []
    
    for code in trip_codes:
        coord = coord_map.get(str(code).strip())
        if coord:
            dist = haversine_distance(centroid_lat, centroid_lon, coord[0], coord[1])
            branch_distances.append((code, dist, province_map.get(str(code).strip(), '')))
            if dist > farthest_dist:
                farthest_dist = dist
                farthest_code = code
    
    # หาทริปอื่นที่ใกล้กว่าสำหรับสาขาที่ไกล
    if farthest_dist > 50:  # ถ้าห่างจาก centroid > 50km
        far_coord = coord_map.get(str(farthest_code).strip())
        far_province = province_map.get(str(farthest_code).strip(), '')
        
        # หาทริปอื่นที่ centroid ใกล้กว่า
        for other_trip in sorted(df['Trip no'].unique()):
            if pd.isna(other_trip) or other_trip == trip_no:
                continue
            
            other_data = df[df['Trip no'] == other_trip]
            other_codes = list(other_data['BranchCode'].values)
            
            # หา centroid ของทริปอื่น
            other_lats, other_lons = [], []
            for code in other_codes:
                coord = coord_map.get(str(code).strip())
                if coord:
                    other_lats.append(coord[0])
                    other_lons.append(coord[1])
            
            if not other_lats:
                continue
            
            other_centroid_lat = sum(other_lats) / len(other_lats)
            other_centroid_lon = sum(other_lons) / len(other_lons)
            
            # ระยะห่างจากสาขาไกลไปยัง centroid ทริปอื่น
            dist_to_other = haversine_distance(far_coord[0], far_coord[1], other_centroid_lat, other_centroid_lon)
            
            if dist_to_other and dist_to_other < farthest_dist - 20:  # ใกล้กว่าอย่างน้อย 20km
                swap_candidates.append({
                    'code': farthest_code,
                    'name': trip_data[trip_data['BranchCode'] == farthest_code]['Branch'].values[0] if len(trip_data[trip_data['BranchCode'] == farthest_code]) > 0 else '',
                    'province': far_province,
                    'current_trip': trip_no,
                    'current_dist': farthest_dist,
                    'better_trip': other_trip,
                    'better_dist': dist_to_other,
                    'improvement': farthest_dist - dist_to_other
                })
                break  # หาแค่ทริปแรกที่ดีกว่า

# แสดงผล
print(f"\n🚨 พบสาขาที่ควรสลับทริป: {len(swap_candidates)} สาขา")
print("-" * 100)

if swap_candidates:
    # เรียงตาม improvement
    swap_candidates.sort(key=lambda x: x['improvement'], reverse=True)
    
    print(f"{'Code':<12} {'ชื่อสาขา':<30} {'จังหวัด':<15} {'ทริปปัจจุบัน':<12} {'ระยะ(km)':<10} {'ทริปที่ดีกว่า':<12} {'ระยะใหม่':<10} {'ดีขึ้น':<10}")
    print("-" * 130)
    
    for s in swap_candidates[:30]:  # แสดง 30 อันดับแรก
        name = str(s['name'])[:28] if s['name'] else ''
        print(f"{s['code']:<12} {name:<30} {s['province']:<15} {s['current_trip']:<12} {s['current_dist']:>8.1f}km {s['better_trip']:<12} {s['better_dist']:>8.1f}km {s['improvement']:>8.1f}km")

# วิเคราะห์ทริปที่มีสาขาหลายจังหวัด
print("\n" + "=" * 80)
print("📊 ทริปที่มีสาขาหลายจังหวัด (อาจจัดไม่ดี)")
print("=" * 80)

multi_province_trips = []

for trip_no in sorted(df['Trip no'].unique()):
    if pd.isna(trip_no):
        continue
    
    trip_data = df[df['Trip no'] == trip_no]
    trip_codes = list(trip_data['BranchCode'].values)
    
    # หาจังหวัดในทริป
    provinces = set()
    for code in trip_codes:
        prov = province_map.get(str(code).strip(), '')
        if prov:
            provinces.add(prov)
    
    if len(provinces) >= 3:  # มี 3 จังหวัดขึ้นไป
        # คำนวณระยะห่างสูงสุด
        max_dist = 0
        for i, code1 in enumerate(trip_codes):
            coord1 = coord_map.get(str(code1).strip())
            for j, code2 in enumerate(trip_codes):
                if i >= j:
                    continue
                coord2 = coord_map.get(str(code2).strip())
                if coord1 and coord2:
                    dist = haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])
                    if dist and dist > max_dist:
                        max_dist = dist
        
        multi_province_trips.append({
            'trip': trip_no,
            'branches': len(trip_codes),
            'provinces': len(provinces),
            'province_list': ', '.join(sorted(provinces)),
            'max_dist': max_dist
        })

if multi_province_trips:
    multi_province_trips.sort(key=lambda x: x['max_dist'], reverse=True)
    
    print(f"\n{'ทริป':<12} {'สาขา':>6} {'จังหวัด':>8} {'MaxDist':>10} {'จังหวัดในทริป'}")
    print("-" * 100)
    
    for t in multi_province_trips[:20]:
        print(f"{t['trip']:<12} {t['branches']:>6} {t['provinces']:>8} {t['max_dist']:>9.1f}km {t['province_list'][:60]}")

print("\n" + "=" * 80)
print("✅ วิเคราะห์เสร็จสิ้น")
print("=" * 80)
