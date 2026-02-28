"""
Rebuild branch_clusters.json using existing distance_cache.json only.
No OSRM API calls — just reads the cache we already have.
"""
import json
import math
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DC_LAT = 14.179394
DC_LON = 100.648149

# Load existing OSRM cache
CACHE = {}
if os.path.exists('distance_cache.json'):
    with open('distance_cache.json', 'r', encoding='utf-8') as f:
        CACHE = json.load(f)
    print(f"📦 OSRM cache: {len(CACHE):,} entries")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def get_road_distance(lat1, lon1, lat2, lon2):
    k  = f"{lat1:.4f},{lon1:.4f}_{lat2:.4f},{lon2:.4f}"
    kr = f"{lat2:.4f},{lon2:.4f}_{lat1:.4f},{lon1:.4f}"
    if k in CACHE:  return CACHE[k], True
    if kr in CACHE: return CACHE[kr], True
    return round(haversine(lat1, lon1, lat2, lon2) * 1.35, 3), False

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(lat2_r)
    y = math.cos(lat1_r)*math.sin(lat2_r) - math.sin(lat1_r)*math.cos(lat2_r)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def get_direction_zone(bearing):
    directions = ['N','NE','E','SE','S','SW','W','NW']
    return directions[int((bearing + 22.5) / 45) % 8]

print("\n📥 Loading branch_data.json ...")
with open('branch_data.json', 'r', encoding='utf-8') as f:
    branch_data = json.load(f)
print(f"   {len(branch_data)} branches")

# Build branch list with distances
branches = []
no_coords = 0
for code, b in branch_data.items():
    try:
        lat = float(b.get('ละ', 0))
        lon = float(b.get('ลอง', 0))
        if not lat or not lon:
            no_coords += 1
            continue
        dist, is_road = get_road_distance(DC_LAT, DC_LON, lat, lon)
        bearing = calculate_bearing(DC_LAT, DC_LON, lat, lon)
        branches.append({
            'code': code,
            'name': b.get('สาขา', ''),
            'province': b.get('จังหวัด', ''),
            'district': b.get('อำเภอ', ''),
            'subdistrict': b.get('ตำบล', ''),
            'lat': lat, 'lon': lon,
            'distance_from_dc': round(dist, 2),
            'bearing': round(bearing, 1),
            'direction': get_direction_zone(bearing),
            'is_road_dc': is_road,
        })
    except Exception:
        no_coords += 1

print(f"   ✅ {len(branches)} with coords, {no_coords} without")

# Build clusters
from collections import defaultdict
distance_clusters = defaultdict(list)
direction_clusters = defaultdict(list)
province_clusters  = defaultdict(list)
district_clusters  = defaultdict(list)

for b in branches:
    distance_clusters[int(b['distance_from_dc'] / 50) * 50].append(b['code'])
    direction_clusters[b['direction']].append(b['code'])
    province_clusters[b['province']].append(b['code'])
    district_clusters[f"{b['province']}_{b['district']}"].append(b['code'])

print(f"   Clusters: distance={len(distance_clusters)}, direction={len(direction_clusters)},",
      f"province={len(province_clusters)}, district={len(district_clusters)}")

# Build nearby_branches (using only existing cache)
print(f"\n🔍 Computing nearby_branches (< 20 km road) ...")
nearby_branches = {}
osrm_hits = 0
fallback_hits = 0

for i, b1 in enumerate(branches):
    nearby = []
    for b2 in branches:
        if b2['code'] == b1['code']:
            continue
        # ข้ามถ้าพิกัดเดียวกัน (4dp) — ป้องกัน dist=0 bug
        if f"{b1['lat']:.4f},{b1['lon']:.4f}" == f"{b2['lat']:.4f},{b2['lon']:.4f}":
            continue
        hav = haversine(b1['lat'], b1['lon'], b2['lat'], b2['lon'])
        if hav >= 20:
            continue
        road, is_road = get_road_distance(b1['lat'], b1['lon'], b2['lat'], b2['lon'])
        if is_road:
            osrm_hits += 1
        else:
            fallback_hits += 1
        if 0 < road < 20:  # ต้อง > 0 ด้วย ป้องกัน fallback ให้ 0 เมื่อพิกัดซ้ำ
            nearby.append({'code': b2['code'], 'distance': round(road, 2), 'is_road': is_road})
    nearby.sort(key=lambda x: x['distance'])
    nearby_branches[b1['code']] = nearby[:20]

    if (i+1) % 1000 == 0 or (i+1) == len(branches):
        avg = sum(len(v) for v in nearby_branches.values()) / max(len(nearby_branches), 1)
        print(f"   ⏳ {i+1}/{len(branches)}  avg_nearby={avg:.1f}  osrm={osrm_hits:,}  fallback={fallback_hits}")

# Build branch_info
branch_info = {}
for b in branches:
    branch_info[b['code']] = {
        'lat': b['lat'], 'lon': b['lon'],
        'distance_from_dc': b['distance_from_dc'],
        'bearing': b['bearing'], 'direction': b['direction'],
        'province': b['province'], 'district': b['district'],
        'subdistrict': b['subdistrict'], 'name': b['name'],
        'district_cluster': f"{b['province']}_{b['district']}",
    }

clusters = {
    'distance':  {str(k): v for k, v in distance_clusters.items()},
    'direction': dict(direction_clusters),
    'province':  dict(province_clusters),
    'district':  dict(district_clusters),
}

cluster_data = {
    'branch_info':       branch_info,
    'nearby_branches':   nearby_branches,
    'clusters':          clusters,
    'distance_clusters': {str(k): v for k, v in distance_clusters.items()},
    'direction_clusters': dict(direction_clusters),
    'province_clusters':  dict(province_clusters),
    'district_clusters':  dict(district_clusters),
    'total_branches': len(branches),
    'dc_location': {'lat': DC_LAT, 'lon': DC_LON},
}

with open('branch_clusters.json', 'w', encoding='utf-8') as f:
    json.dump(cluster_data, f, ensure_ascii=False)

avg_nearby = sum(len(v) for v in nearby_branches.values()) / max(len(nearby_branches), 1)
road_dc_count = sum(1 for b in branches if b['is_road_dc'])
print(f"\n✅ branch_clusters.json saved")
print(f"   Branches: {len(branch_info)}")
print(f"   DC→branch OSRM hits: {road_dc_count}/{len(branches)}")
print(f"   Avg nearby per branch: {avg_nearby:.1f}")
print(f"   Nearby OSRM hits: {osrm_hits:,}  fallback: {fallback_hits}")
