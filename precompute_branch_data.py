"""
สคริปต์สำหรับ Pre-compute ข้อมูลสาขาทั้งหมด:
1. คำนวณระยะทางจาก DC วังน้อย
2. สร้าง spatial clusters
3. คำนวณระยะทางระหว่างสาขาใกล้เคียง
"""
import json
import math
import os
import requests
import sys
import time
from collections import defaultdict

# ตั้ง stdout เป็น UTF-8 เพื่อรองรับ emoji และภาษาไทยใน Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# DC วังน้อย
DC_LAT = 14.179394
DC_LON = 100.648149

# โหลด OSRM distance cache (ถ้ามี)
OSRM_CACHE = {}
if os.path.exists('distance_cache.json'):
    try:
        with open('distance_cache.json', 'r', encoding='utf-8') as f:
            OSRM_CACHE = json.load(f)
        print(f"📦 โหลด OSRM distance cache: {len(OSRM_CACHE):,} รายการ")
    except Exception as e:
        print(f"⚠️ โหลด distance_cache.json ไม่สำเร็จ: {e}")

BATCH_SIZE = 90        # จำนวน coordinates ต่อ 1 OSRM Table call (public server รองรับ ~100)
OSRM_DELAY = 0.15      # วินาที หน่วงระหว่าง call เพื่อไม่ flood public server
OSRM_TIMEOUT = 20      # timeout ต่อ request


def _osrm_table_call(coords_lonlat, retries=3):
    """
    เรียก OSRM Table API ครั้งเดียวแบบ full N×N matrix
    coords_lonlat: list ของ (lon, lat)
    คืน matrix distances[i][j] เป็น km หรือ None ถ้า fail
    """
    coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords_lonlat)
    url = f"http://router.project-osrm.org/table/v1/driving/{coord_str}?annotations=distance"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=OSRM_TIMEOUT)
            data = r.json()
            if data.get("code") == "Ok":
                raw = data["distances"]  # N×N matrix in meters
                km = [[v / 1000.0 if v else None for v in row] for row in raw]
                return km
        except KeyboardInterrupt:
            raise  # ปล่อยให้ outer handler จัดการ
        except Exception:
            if attempt < retries - 1:
                time.sleep(1 + attempt)
    return None


def build_osrm_cache_batched(branch_data):
    """
    สร้าง/เติม distance_cache.json ด้วย OSRM Table API แบบ batch:
    1) DC → สาขาทั้งหมด (batch ทีละ BATCH_SIZE-1)
    2) สาขา-สาขา ภายในจังหวัดเดียวกัน (full N×N per province) 
    """
    global OSRM_CACHE

    # รวบรวม branches ที่มีพิกัด
    branches = []
    for code, b in branch_data.items():
        try:
            lat = float(b.get('ละ', 0))
            lon = float(b.get('ลอง', 0))
            if lat and lon:
                branches.append({'code': code, 'lat': lat, 'lon': lon,
                                  'province': b.get('จังหวัด', '')})
        except Exception:
            continue

    total = len(branches)
    print(f"\n🌐 Build OSRM cache (batch) — {total} สาขา")
    new_pairs = 0

    # ——— ขั้น 1: DC → สาขาทั้งหมด ———
    dc_missing = [b for b in branches
                  if f"{DC_LAT:.4f},{DC_LON:.4f}_{b['lat']:.4f},{b['lon']:.4f}" not in OSRM_CACHE]
    print(f"  [1/2] DC → สาขา: {len(dc_missing)}/{total} ที่ยังไม่มีใน cache")
    if not dc_missing:
        print("       ✅ ครบแล้ว ข้ามไป")
    else:
        dc_coord = (DC_LON, DC_LAT)  # (lon, lat) format
        for start in range(0, len(dc_missing), BATCH_SIZE - 1):
            chunk = dc_missing[start:start + BATCH_SIZE - 1]
            coords = [dc_coord] + [(b['lon'], b['lat']) for b in chunk]
            coord_str = ";".join(f"{lon},{lat}" for lon, lat in coords)
            dst_str = ";".join(str(i) for i in range(1, len(coords)))
            url = (f"http://router.project-osrm.org/table/v1/driving/{coord_str}"
                   f"?sources=0&destinations={dst_str}&annotations=distance")
            try:
                r = requests.get(url, timeout=OSRM_TIMEOUT)
                data = r.json()
                if data.get("code") == "Ok":
                    for j, dist_m in enumerate(data["distances"][0]):
                        b = chunk[j]
                        key = f"{DC_LAT:.4f},{DC_LON:.4f}_{b['lat']:.4f},{b['lon']:.4f}"
                        if dist_m and dist_m > 0 and key not in OSRM_CACHE:
                            OSRM_CACHE[key] = round(dist_m / 1000.0, 3)
                            new_pairs += 1
            except KeyboardInterrupt:
                _save_cache()
                print(f"\n⚠️ DC→branch ถูกหยุด — บันทึก cache แล้ว (+{new_pairs} pairs)")
                raise
            except Exception:
                pass
            time.sleep(OSRM_DELAY)

            if (start // (BATCH_SIZE - 1)) % 20 == 0:
                print(f"     ⏳ DC→branch {min(start + BATCH_SIZE - 1, len(dc_missing))}/{len(dc_missing)} (+{new_pairs} ใหม่)")

    # บันทึกหลัง DC pass
    _save_cache()
    print(f"  ✅ DC→branch เสร็จ: +{new_pairs} pairs ใหม่")
    new_pairs = 0

    # ——— ขั้น 2: Branch↔Branch ภายในจังหวัด ———
    print(f"  [2/2] Branch↔Branch รายจังหวัด...")
    from collections import defaultdict as _dd
    by_prov = _dd(list)
    for b in branches:
        by_prov[b['province']].append(b)

    prov_list = sorted(by_prov.keys())
    try:
        for p_idx, prov in enumerate(prov_list):
            prov_branches = by_prov[prov]
            n = len(prov_branches)
            if n < 2:
                continue

            # ตรวจว่า province นี้มีคู่ที่ยังไม่ใน cache หรือเปล่า
            prov_missing = False
            for bi in prov_branches:
                for bj in prov_branches:
                    if bi is bj:
                        continue
                    k = f"{bi['lat']:.4f},{bi['lon']:.4f}_{bj['lat']:.4f},{bj['lon']:.4f}"
                    if k not in OSRM_CACHE:
                        prov_missing = True
                        break
                if prov_missing:
                    break
            if not prov_missing:
                continue  # ✅ ครบแล้ว ข้ามจังหวัดนี้

            # ถ้า province มี > BATCH_SIZE สาขา → split เป็น batch ย่อย
            for start in range(0, n, BATCH_SIZE):
                chunk = prov_branches[start:start + BATCH_SIZE]
                if len(chunk) < 2:
                    continue
                # ข้าม batch ถ้าทุกคู่ใน chunk มีใน cache แล้ว
                batch_missing = any(
                    f"{bi['lat']:.4f},{bi['lon']:.4f}_{bj['lat']:.4f},{bj['lon']:.4f}" not in OSRM_CACHE
                    for i, bi in enumerate(chunk) for j, bj in enumerate(chunk) if i != j
                )
                if not batch_missing:
                    continue
                coords = [(b['lon'], b['lat']) for b in chunk]
                matrix = _osrm_table_call(coords)
                if matrix is None:
                    time.sleep(2)
                    continue
                for i, bi in enumerate(chunk):
                    for j, bj in enumerate(chunk):
                        if i == j:
                            continue
                        dist_km = matrix[i][j]
                        if dist_km is None or dist_km <= 0:
                            continue
                        key = f"{bi['lat']:.4f},{bi['lon']:.4f}_{bj['lat']:.4f},{bj['lon']:.4f}"
                        if key not in OSRM_CACHE:
                            OSRM_CACHE[key] = round(dist_km, 3)
                            new_pairs += 1
                time.sleep(OSRM_DELAY)

            # บันทึกทุก 10 จังหวัด
            if (p_idx + 1) % 10 == 0:
                _save_cache()
                print(f"     ⏳ {p_idx+1}/{len(prov_list)} จังหวัด (+{new_pairs} pairs ใหม่, cache={len(OSRM_CACHE):,})")
                new_pairs = 0

    except KeyboardInterrupt:
        print(f"\n⚠️ ถูกหยุด — บันทึก cache ที่ทำไว้แล้ว ({len(OSRM_CACHE):,} entries)")
        _save_cache()
        raise

    _save_cache()
    print(f"  ✅ Branch↔Branch เสร็จ: +{new_pairs} pairs, cache รวม {len(OSRM_CACHE):,} รายการ")


def _save_cache():
    """บันทึก OSRM_CACHE ลงไฟล์"""
    try:
        with open('distance_cache.json', 'w', encoding='utf-8') as f:
            json.dump(OSRM_CACHE, f, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ บันทึก cache ไม่สำเร็จ: {e}")


def get_road_distance(lat1, lon1, lat2, lon2):
    """
    ดึงระยะทางถนนจาก OSRM cache (ที่ build ไว้ล่วงหน้าแล้ว)
    cache miss → haversine × 1.35 เป็น fallback ใน precompute
    """
    key = f"{lat1:.4f},{lon1:.4f}_{lat2:.4f},{lon2:.4f}"
    key_rev = f"{lat2:.4f},{lon2:.4f}_{lat1:.4f},{lon1:.4f}"
    if key in OSRM_CACHE:
        return OSRM_CACHE[key], True
    if key_rev in OSRM_CACHE:
        return OSRM_CACHE[key_rev], True
    # cache miss → ประมาณ haversine × 1.35
    dist = haversine(lat1, lon1, lat2, lon2) * 1.35
    return round(dist, 3), False

def haversine(lat1, lon1, lat2, lon2):
    """คำนวณระยะทาง Haversine (km)"""
    R = 6371.0  # รัศมีโลก (กม.)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """คำนวณทิศทาง (bearing) 0-360 องศา"""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    
    bearing = math.atan2(x, y)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing

def get_direction_zone(bearing):
    """แบ่งทิศทางเป็น 8 โซน"""
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = int((bearing + 22.5) / 45) % 8
    return directions[index]

def precompute_all():
    """Pre-compute ข้อมูลสาขาทั้งหมด"""
    
    print("="*60)
    print("🚀 เริ่มต้น Pre-compute ข้อมูลสาขา")
    print("="*60)
    
    # 1. โหลดข้อมูลสาขา
    print("\n📥 โหลดข้อมูลสาขา...")
    with open('branch_data.json', 'r', encoding='utf-8') as f:
        branch_data = json.load(f)
    print(f"   ✅ โหลด: {len(branch_data)} สาขา")
    
    # 2. คำนวณระยะทางจาก DC
    print("\n📏 คำนวณระยะทางจาก DC วังน้อย...")
    branches_with_distance = []
    no_coords = 0
    
    for code, branch in branch_data.items():
        try:
            lat = float(branch.get('ละ', 0))
            lon = float(branch.get('ลอง', 0))
            
            if lat == 0 or lon == 0:
                no_coords += 1
                continue
            
            # คำนวณระยะทางจาก DC (ใช้ OSRM ถ้ามี)
            distance, is_road = get_road_distance(DC_LAT, DC_LON, lat, lon)
            
            # คำนวณทิศทาง
            bearing = calculate_bearing(DC_LAT, DC_LON, lat, lon)
            direction = get_direction_zone(bearing)
            
            branches_with_distance.append({
                'code': code,
                'name': branch.get('สาขา', ''),
                'province': branch.get('จังหวัด', ''),
                'district': branch.get('อำเภอ', ''),
                'subdistrict': branch.get('ตำบล', ''),
                'lat': lat,
                'lon': lon,
                'distance_from_dc': round(distance, 2),
                'bearing': round(bearing, 1),
                'direction': direction
            })
        except Exception as e:
            no_coords += 1
            continue
    
    print(f"   ✅ คำนวณสำเร็จ: {len(branches_with_distance)} สาขา")
    print(f"   ⚠️ ไม่มีพิกัด: {no_coords} สาขา")
    
    # 3. จัดกลุ่มตามระยะทางและทิศทาง (Spatial Clusters)
    print("\n📊 สร้าง Spatial Clusters...")
    
    # กลุ่มตามระยะทาง (ทุกๆ 50 กม.)
    distance_clusters = defaultdict(list)
    for b in branches_with_distance:
        dist_group = int(b['distance_from_dc'] / 50) * 50
        distance_clusters[dist_group].append(b['code'])
    
    # กลุ่มตามทิศทาง
    direction_clusters = defaultdict(list)
    for b in branches_with_distance:
        direction_clusters[b['direction']].append(b['code'])
    
    # กลุ่มตามจังหวัด
    province_clusters = defaultdict(list)
    for b in branches_with_distance:
        province_clusters[b['province']].append(b['code'])
    
    # กลุ่มตามอำเภอ
    district_clusters = defaultdict(list)
    for b in branches_with_distance:
        key = f"{b['province']}_{b['district']}"
        district_clusters[key].append(b['code'])
    
    print(f"   ✅ Distance Clusters: {len(distance_clusters)} กลุ่ม")
    print(f"   ✅ Direction Clusters: {len(direction_clusters)} กลุ่ม")
    print(f"   ✅ Province Clusters: {len(province_clusters)} กลุ่ม")
    print(f"   ✅ District Clusters: {len(district_clusters)} กลุ่ม")
    
    # ——— Build/fill OSRM cache ก่อน ———
    build_osrm_cache_batched(branch_data)

    # 4. หาสาขาใกล้เคียง (< 15 km ตาม Haversine pre-filter → ใช้ OSRM cache)
    print("\n🔍 คำนวณสาขาใกล้เคียง (< 20 km)...")
    nearby_branches = {}
    osrm_used = 0
    penalty_used = 0
    
    for i, b1 in enumerate(branches_with_distance):
        nearby = []
        for j, b2 in enumerate(branches_with_distance):
            if i == j:
                continue
            
            # กรองเบื้องต้นด้วย Haversine ก่อน (เร็ว) เพื่อไม่เรียก OSRM คู่ที่ไกลเกิน
            hav_dist = haversine(b1['lat'], b1['lon'], b2['lat'], b2['lon'])
            if hav_dist >= 20:  # ขยาย 15→20 เผื่อถนนอ้อม
                continue
            
            # ดึงระยะทางถนนจริงจาก OSRM (cache หรือ live)
            road_dist, is_road = get_road_distance(b1['lat'], b1['lon'], b2['lat'], b2['lon'])
            if is_road:
                osrm_used += 1
            else:
                penalty_used += 1
            
            # เก็บเฉพาะที่ระยะทางถนนจริง < 20 km
            if road_dist < 20:
                nearby.append({
                    'code': b2['code'],
                    'distance': round(road_dist, 2),
                    'is_road': is_road
                })
        
        # เรียงตามระยะทาง
        nearby.sort(key=lambda x: x['distance'])
        nearby_branches[b1['code']] = nearby[:20]  # เก็บแค่ 20 สาขาที่ใกล้ที่สุด
        
        if (i + 1) % 500 == 0 or (i + 1) == len(branches_with_distance):
            print(f"   ⏳ {i+1}/{len(branches_with_distance)} สาขา (OSRM: {osrm_used:,}, est. factor: {penalty_used})")

    avg_nearby = sum(len(v) for v in nearby_branches.values()) / max(len(nearby_branches), 1)
    print(f"   ✅ คำนวณสำเร็จ: เฉลี่ย {avg_nearby:.1f} สาขาใกล้เคียง/สาขา")
    print(f"   📡 OSRM cache hit: {osrm_used:,} คู่ | haversine×1.35 est.: {penalty_used} คู่")
    
    # 5. บันทึกผลลัพธ์
    print("\n💾 บันทึกผลลัพธ์...")
    
    # สร้าง branch_info (format ที่ app.py ต้องการ)
    branch_info = {}
    for b in branches_with_distance:
        # สร้าง district_cluster key สำหรับ same_area_branches
        district_key = f"{b['province']}_{b['district']}"
        branch_info[b['code']] = {
            'lat': b['lat'],
            'lon': b['lon'],
            'distance_from_dc': b['distance_from_dc'],
            'bearing': b['bearing'],
            'direction': b['direction'],
            'province': b['province'],
            'district': b['district'],
            'subdistrict': b['subdistrict'],
            'name': b['name'],
            'district_cluster': district_key   # สำหรับ same_area_branches lookup
        }
    
    # สร้าง clusters ในรูปแบบที่ app.py ต้องการ
    # app.py ใช้ clusters.get('district', {}) สำหรับ same_area_branches
    clusters = {
        'distance': {str(k): v for k, v in distance_clusters.items()},
        'direction': {k: v for k, v in direction_clusters.items()},
        'province': {k: v for k, v in province_clusters.items()},
        'district': {k: v for k, v in district_clusters.items()}
    }
    
    # บันทึก branch_clusters.json
    cluster_data = {
        'branch_info': branch_info,                # 🆕 ข้อมูลพิกัดสาขา (app ต้องการ)
        'nearby_branches': nearby_branches,         # 🆕 ระยะทาง OSRM/haversine
        'clusters': clusters,                        # 🆕 กลุ่มตาม district/province/etc (app ต้องการ)
        # เก็บ format เดิมไว้ด้วยเพื่อ backward compatibility
        'distance_clusters': {str(k): v for k, v in distance_clusters.items()},
        'direction_clusters': {k: v for k, v in direction_clusters.items()},
        'province_clusters': {k: v for k, v in province_clusters.items()},
        'district_clusters': {k: v for k, v in district_clusters.items()},
        'total_branches': len(branches_with_distance),
        'dc_location': {'lat': DC_LAT, 'lon': DC_LON}
    }
    
    with open('branch_clusters.json', 'w', encoding='utf-8') as f:
        json.dump(cluster_data, f, ensure_ascii=False, indent=2)
    print(f"   ✅ บันทึก branch_clusters.json ({len(branch_info)} สาขา)")
    
    # สร้างสถิติ
    stats = {
        'total_branches': len(branches_with_distance),
        'total_branches_no_coords': no_coords,
        'avg_distance_from_dc': round(sum(b['distance_from_dc'] for b in branches_with_distance) / len(branches_with_distance), 2),
        'max_distance_from_dc': round(max(b['distance_from_dc'] for b in branches_with_distance), 2),
        'min_distance_from_dc': round(min(b['distance_from_dc'] for b in branches_with_distance), 2),
        'total_distance_clusters': len(distance_clusters),
        'total_direction_clusters': len(direction_clusters),
        'total_province_clusters': len(province_clusters),
        'total_district_clusters': len(district_clusters),
        'avg_nearby_branches': round(avg_nearby, 1),
        'direction_distribution': {k: len(v) for k, v in direction_clusters.items()}
    }
    
    print("\n" + "="*60)
    print("📊 สถิติข้อมูลสาขา")
    print("="*60)
    print(f"สาขาทั้งหมด: {stats['total_branches']} สาขา")
    print(f"ระยะทางเฉลี่ยจาก DC: {stats['avg_distance_from_dc']} km")
    print(f"ระยะทางไกลสุด: {stats['max_distance_from_dc']} km")
    print(f"ระยะทางใกล้สุด: {stats['min_distance_from_dc']} km")
    print(f"\nการกระจายตามทิศทาง:")
    for direction in ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']:
        count = stats['direction_distribution'].get(direction, 0)
        print(f"   {direction:3s}: {count:4d} สาขา")
    
    print("\n✅ Pre-compute เสร็จสิ้น!")
    _save_cache()
    print(f"💾 distance_cache.json: {len(OSRM_CACHE):,} รายการ")
    return stats

if __name__ == "__main__":
    try:
        stats = precompute_all()
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()
