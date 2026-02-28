"""
Pre-compute ระยะทางจริงจาก OSRM API (ไม่ใช่เส้นตรง)
เก็บเฉพาะระยะทางที่จำเป็น:
1. DC → สาขาทั้งหมด
2. สาขา → สาขาใกล้เคียง (< 15 km เท่านั้น)
"""
import json
import requests
import time
from datetime import datetime

# DC วังน้อย
DC_LAT = 14.179394
DC_LON = 100.648149

def get_osrm_distance(lat1, lon1, lat2, lon2, retry=2):
    """
    ดึงระยะทางจริงจาก OSRM API
    Returns: distance in km (0 ถ้าล้มเหลว)
    """
    # OSRM รับพิกัดแบบ lon,lat
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    
    for attempt in range(retry):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if 'routes' in data and len(data['routes']) > 0:
                    distance_m = data['routes'][0]['distance']
                    return round(distance_m / 1000, 2)  # แปลงเป็น km
        except Exception as e:
            if attempt < retry - 1:
                time.sleep(0.5)
                continue
        return 0
    return 0

def load_existing_cache():
    """โหลด cache เดิม"""
    try:
        with open('distance_cache.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def save_cache(cache_dict):
    """บันทึก cache"""
    with open('distance_cache.json', 'w', encoding='utf-8') as f:
        json.dump(cache_dict, f, ensure_ascii=False, indent=2)

def precompute_essential_distances():
    """
    Pre-compute เฉพาะระยะทางที่จำเป็น:
    1. DC → สาขาทั้งหมด
    2. สาขา → สาขาใกล้เคียง (จาก branch_clusters.json)
    """
    print("="*70)
    print("🚀 เริ่มต้น Pre-compute ระยะทางจริง (OSRM)")
    print("="*70)
    
    # โหลด cache เดิม
    print("\n📦 โหลด cache เดิม...")
    cache = load_existing_cache()
    print(f"   ✅ พบ cache เดิม: {len(cache)} รายการ")
    
    # โหลดข้อมูลสาขา
    print("\n📥 โหลดข้อมูลสาขา...")
    with open('branch_data.json', 'r', encoding='utf-8') as f:
        branch_data = json.load(f)
    
    with open('branch_clusters.json', 'r', encoding='utf-8') as f:
        clusters = json.load(f)
    
    branches = []
    for code, branch in branch_data.items():
        try:
            lat = float(branch.get('ละ', 0))
            lon = float(branch.get('ลอง', 0))
            if lat > 0 and lon > 0:
                branches.append({'code': code, 'lat': lat, 'lon': lon})
        except:
            continue
    
    print(f"   ✅ โหลด: {len(branches)} สาขา")
    
    nearby_branches = clusters.get('nearby_branches', {})
    print(f"   ✅ โหลด nearby_branches: {len(nearby_branches)} รายการ")
    
    # ===== PHASE 1: DC → สาขาทั้งหมด =====
    print("\n" + "="*70)
    print("📍 PHASE 1: คำนวณระยะทาง DC → สาขาทั้งหมด")
    print("="*70)
    
    dc_distances = {}
    new_count = 0
    cached_count = 0
    skipped_count = 0
    
    for i, branch in enumerate(branches):
        code = branch['code']
        lat, lon = branch['lat'], branch['lon']
        
        # สร้าง cache key
        cache_key = f"{DC_LAT:.4f},{DC_LON:.4f}_{lat:.4f},{lon:.4f}"
        
        if cache_key in cache:
            dc_distances[code] = cache[cache_key]
            cached_count += 1
        else:
            # คำนวณใหม่
            dist = get_osrm_distance(DC_LAT, DC_LON, lat, lon)
            if dist > 0:
                dc_distances[code] = dist
                cache[cache_key] = dist
                new_count += 1
                
                # บันทึกทุกๆ 20 รายการ (เร็วขึ้น)
                if new_count % 20 == 0:
                    save_cache(cache)
                    print(f"   💾 บันทึก... (ใหม่: {new_count}, cache: {cached_count}, ข้าม: {skipped_count})")
                
                # Rate limiting (0.3 วินาที/request = 3.3 requests/sec)
                time.sleep(0.3)
            else:
                skipped_count += 1
        
        # แสดงความคืบหน้า
        if (i + 1) % 100 == 0:
            progress = (i + 1) / len(branches) * 100
            print(f"   ⏳ {i+1}/{len(branches)} ({progress:.1f}%) | ใหม่: {new_count} | cache: {cached_count} | ข้าม: {skipped_count}")
    
    print(f"\n   ✅ PHASE 1 เสร็จสิ้น:")
    print(f"      - คำนวณใหม่: {new_count} รายการ")
    print(f"      - ใช้ cache: {cached_count} รายการ")
    print(f"      - รวม DC distances: {len(dc_distances)} สาขา")
    
    # ===== PHASE 2: สาขา → สาขาใกล้เคียง =====
    print("\n" + "="*70)
    print("🔗 PHASE 2: คำนวณระยะทางสาขา → สาขาใกล้เคียง")
    print("="*70)
    
    # นับจำนวนที่ต้องคำนวณ
    total_pairs = sum(len(neighbors) for neighbors in nearby_branches.values())
    print(f"   📊 ต้องคำนวณ: {total_pairs} คู่สาขา")
    
    branch_coords = {b['code']: (b['lat'], b['lon']) for b in branches}
    
    new_count = 0
    cached_count = 0
    skipped_count = 0
    computed = 0
    start_time = time.time()
    
    for code1, neighbors in nearby_branches.items():
        if code1 not in branch_coords:
            continue
        
        lat1, lon1 = branch_coords[code1]
        
        for neighbor_info in neighbors:
            code2 = neighbor_info['code']
            if code2 not in branch_coords:
                continue
            
            lat2, lon2 = branch_coords[code2]
            
            # สร้าง cache key (เรียงตามตัวอักษรเพื่อไม่ซ้ำ)
            if code1 < code2:
                cache_key = f"{lat1:.4f},{lon1:.4f}_{lat2:.4f},{lon2:.4f}"
            else:
                cache_key = f"{lat2:.4f},{lon2:.4f}_{lat1:.4f},{lon1:.4f}"
            
            computed += 1
            
            if cache_key in cache:
                cached_count += 1
            else:
                # คำนวณใหม่
                dist = get_osrm_distance(lat1, lon1, lat2, lon2)
                if dist > 0:
                    cache[cache_key] = dist
                    new_count += 1
                    
                    # บันทึกทุกๆ 20 รายการ
                    if new_count % 20 == 0:
                        save_cache(cache)
                        print(f"   💾 บันทึก... (ใหม่: {new_count}, cache: {cached_count}, ข้าม: {skipped_count})")
                else:
                    skipped_count += 1
                
                # Rate limiting
                time.sleep(0.3)
            
            # แสดงความคืบหน้า
            if computed % 200 == 0:
                progress = computed / total_pairs * 100
                elapsed = time.time() - start_time if 'start_time' in dir() else 0
                rate = computed / max(elapsed, 1)
                eta = (total_pairs - computed) / max(rate, 0.1) / 60
                print(f"   ⏳ {computed}/{total_pairs} ({progress:.1f}%) | ใหม่: {new_count} | cache: {cached_count} | ETA: {eta:.1f} นาที")
    
    print(f"\n   ✅ PHASE 2 เสร็จสิ้น:")
    print(f"      - คำนวณใหม่: {new_count} รายการ")
    print(f"      - ใช้ cache: {cached_count} รายการ")
    print(f"      - ข้ามไป (ล้มเหลว): {skipped_count} รายการ")
    
    # บันทึก cache ครั้งสุดท้าย
    print("\n💾 บันทึก cache ครั้งสุดท้าย...")
    save_cache(cache)
    
    # สรุปสถิติ
    print("\n" + "="*70)
    print("📊 สรุปผลลัพธ์")
    print("="*70)
    print(f"Cache ทั้งหมด: {len(cache)} รายการ")
    print(f"DC → สาขา: {len(dc_distances)} รายการ")
    print(f"สาขา ↔ สาขา: {len(cache) - len(dc_distances)} รายการ")
    
    # สร้างไฟล์สรุป DC distances
    print("\n💾 สร้างไฟล์สรุป DC distances...")
    dc_summary = {
        'updated_at': datetime.now().isoformat(),
        'total_branches': len(dc_distances),
        'distances': dc_distances
    }
    
    with open('dc_distances.json', 'w', encoding='utf-8') as f:
        json.dump(dc_summary, f, ensure_ascii=False, indent=2)
    
    print(f"   ✅ บันทึก dc_distances.json ({len(dc_distances)} สาขา)")
    
    print("\n✅ Pre-compute เสร็จสิ้นทั้งหมด!")
    print(f"⏱️  เวลาที่ใช้: ประมาณ {(new_count * 0.2 / 60):.1f} นาที")

if __name__ == "__main__":
    cache = {}  # สร้างตัวแปร global
    try:
        start_time = time.time()
        precompute_essential_distances()
        elapsed = time.time() - start_time
        print(f"\n⏱️  เวลาทั้งหมด: {elapsed/60:.1f} นาที ({elapsed:.0f} วินาที)")
    except KeyboardInterrupt:
        print("\n\n⚠️ ถูกยกเลิกโดยผู้ใช้")
        print("💾 กรุณารอสักครู่... กำลังบันทึก cache")
        if cache:  # ตรวจสอบว่ามี cache หรือไม่
            save_cache(cache)
            print("✅ บันทึก cache เรียบร้อย")
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()
