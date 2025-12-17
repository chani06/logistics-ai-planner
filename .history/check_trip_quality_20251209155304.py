"""
🔍 ตรวจสอบคุณภาพการจัดทริป
- ระยะทางระหว่างสาขาในทริป
- ปริมาณที่ใส่ในรถ (ต้องไม่เกิน 100%)
"""
import pandas as pd
import math

# ไฟล์
DATA_FILE = r"Dc\แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx"
MASTER_FILE = r"Dc\Master สถานที่ส่ง.xlsx"

# ค่าคงที่รถ
LIMITS = {
    '4W': {'max_w': 1700, 'max_c': 8.0},
    'JB': {'max_w': 2500, 'max_c': 10.0},  
    '6W': {'max_w': 5000, 'max_c': 20.0}
}

def haversine_distance(lat1, lon1, lat2, lon2):
    """คำนวณระยะทางระหว่าง 2 จุด (km)"""
    if not all([lat1, lon1, lat2, lon2]):
        return None
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

print("=" * 80)
print("🔍 ตรวจสอบคุณภาพการจัดทริป")
print("=" * 80)

# 1. อ่านไฟล์
print("\n📂 อ่านไฟล์...")
df = pd.read_excel(DATA_FILE, sheet_name='2.Punthai', header=1)
df = df[df['BranchCode'].notna()].copy()
print(f"   ✅ ข้อมูล: {len(df)} สาขา")

# อ่าน Master เพื่อดึงพิกัด
df_master = pd.read_excel(MASTER_FILE)
coord_map = {}
for _, row in df_master.iterrows():
    code = str(row.get('Plan Code', '')).strip()
    lat = row.get('ละติจูด')
    lon = row.get('ลองติจูด')
    if code and pd.notna(lat) and pd.notna(lon):
        coord_map[code] = (lat, lon)
print(f"   ✅ พิกัด: {len(coord_map)} สาขา")

# 2. ตรวจสอบแต่ละทริป
print("\n" + "=" * 80)
print("📊 ตรวจสอบทริป")
print("=" * 80)

problems = []
trip_stats = []

for trip_no in sorted(df['Trip no'].unique()):
    if pd.isna(trip_no):
        continue
        
    trip_data = df[df['Trip no'] == trip_no].copy()
    
    # หาประเภทรถจากชื่อทริป
    trip_str = str(trip_no)
    if '6W' in trip_str:
        vehicle = '6W'
    elif '4WJ' in trip_str or 'JB' in trip_str:
        vehicle = 'JB'
    else:
        vehicle = '4W'
    
    # คำนวณ utilization
    total_cube = trip_data['TOTALCUBE'].sum()
    total_weight = trip_data['TOTALWGT'].sum()
    branch_count = len(trip_data)
    
    max_c = LIMITS[vehicle]['max_c']
    max_w = LIMITS[vehicle]['max_w']
    
    cube_util = (total_cube / max_c) * 100
    weight_util = (total_weight / max_w) * 100
    max_util = max(cube_util, weight_util)
    
    # คำนวณระยะทางระหว่างสาขา
    codes = list(trip_data['BranchCode'].values)
    distances = []
    max_distance = 0
    
    for i, code1 in enumerate(codes):
        coord1 = coord_map.get(str(code1).strip())
        for j, code2 in enumerate(codes):
            if i >= j:
                continue
            coord2 = coord_map.get(str(code2).strip())
            if coord1 and coord2:
                dist = haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])
                if dist:
                    distances.append(dist)
                    if dist > max_distance:
                        max_distance = dist
    
    avg_distance = sum(distances) / len(distances) if distances else 0
    
    # เช็คปัญหา
    status = "✅"
    issue = ""
    
    if max_util > 100:
        status = "❌"
        issue = f"เกิน {max_util:.1f}%"
        problems.append({
            'trip': trip_no,
            'issue': 'OVERLOAD',
            'detail': f"Cube: {cube_util:.1f}%, Weight: {weight_util:.1f}%"
        })
    elif max_distance > 100:
        status = "⚠️"
        issue = f"ระยะห่าง {max_distance:.0f}km"
        problems.append({
            'trip': trip_no,
            'issue': 'FAR_APART',
            'detail': f"ระยะห่างสูงสุด {max_distance:.0f}km"
        })
    
    trip_stats.append({
        'trip_no': trip_no,
        'vehicle': vehicle,
        'branches': branch_count,
        'cube': total_cube,
        'weight': total_weight,
        'cube_util': cube_util,
        'weight_util': weight_util,
        'max_util': max_util,
        'max_dist': max_distance,
        'avg_dist': avg_distance,
        'status': status,
        'issue': issue
    })

# แสดงผล
print(f"\n{'Trip':<10} {'รถ':<4} {'สาขา':>5} {'Cube':>8} {'Weight':>10} {'Cube%':>7} {'Wgt%':>7} {'Max%':>7} {'MaxDist':>8} {'Status':<6} {'Issue'}")
print("-" * 100)

for s in trip_stats:
    print(f"{s['trip_no']:<10} {s['vehicle']:<4} {s['branches']:>5} {s['cube']:>8.2f} {s['weight']:>10.2f} "
          f"{s['cube_util']:>6.1f}% {s['weight_util']:>6.1f}% {s['max_util']:>6.1f}% {s['max_dist']:>7.1f}km "
          f"{s['status']:<6} {s['issue']}")

# สรุป
print("\n" + "=" * 80)
print("📊 สรุป")
print("=" * 80)

total_trips = len(trip_stats)
overload_trips = len([s for s in trip_stats if s['max_util'] > 100])
far_trips = len([s for s in trip_stats if s['max_dist'] > 100])
good_trips = len([s for s in trip_stats if s['max_util'] <= 100 and s['max_dist'] <= 100])

print(f"   📦 ทริปทั้งหมด: {total_trips}")
print(f"   ✅ ทริปปกติ: {good_trips}")
print(f"   ❌ ทริปเกิน 100%: {overload_trips}")
print(f"   ⚠️ ทริประยะห่างมาก (>100km): {far_trips}")

# แสดงปัญหา
if problems:
    print(f"\n🚨 ปัญหาที่พบ:")
    for p in problems[:20]:
        print(f"   - {p['trip']}: {p['issue']} - {p['detail']}")

# Utilization stats
utils = [s['max_util'] for s in trip_stats]
print(f"\n📈 Utilization:")
print(f"   Min: {min(utils):.1f}%")
print(f"   Max: {max(utils):.1f}%")
print(f"   Avg: {sum(utils)/len(utils):.1f}%")

# รถที่ใช้
vehicles = {}
for s in trip_stats:
    v = s['vehicle']
    vehicles[v] = vehicles.get(v, 0) + 1

print(f"\n🚛 รถที่ใช้:")
for v, count in sorted(vehicles.items()):
    print(f"   {v}: {count} คัน")
