"""
ตรวจสอบสาขาจาก Google Sheets และเติมระยะทางที่ขาด
1. Sync branch_data.json จาก Sheets
2. หาสาขาใหม่และสาขาที่ขาด distance cache
3. คำนวณด้วย OSRM + อัพเดต branch_clusters.json
"""
import json
import os
import sys
import time
import math
import requests
from collections import defaultdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

DC_LAT, DC_LON = 14.179394, 100.648149
OSRM_TIMEOUT = 20
OSRM_DELAY = 0.15
BATCH_SIZE = 90

# ─── โหลดไฟล์ ─────────────────────────────────────────────────────────────
print("=" * 60)
print("🔄 ตรวจสอบสาขาจากชีตใหม่และหาข้อมูลระยะทาง")
print("=" * 60)

# 1) โหลด branch_data.json ปัจจุบัน
old_bd = {}
if os.path.exists('branch_data.json'):
    old_bd = json.load(open('branch_data.json', 'r', encoding='utf-8'))
print(f"\n📦 branch_data.json (ปัจจุบัน): {len(old_bd):,} สาขา")

# 2) โหลด distance_cache.json
dc_cache = {}
if os.path.exists('distance_cache.json'):
    dc_cache = json.load(open('distance_cache.json', 'r', encoding='utf-8'))
print(f"📦 distance_cache.json: {len(dc_cache):,} entries")

# ─── Sync จาก Google Sheets ──────────────────────────────────────────────
print("\n📡 Sync จาก Google Sheets...")
new_bd = None
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key('12DmIfECwVpsWfl8rl2r1A_LB4_5XMrmnmwlPUHKNU-o')

    # หา worksheet GID 876257177
    ws = None
    for w in sh.worksheets():
        if w.id == 876257177:
            ws = w
            break
    if ws is None:
        ws = sh.get_worksheet(0)

    print(f"   ✅ เชื่อมต่อ: {sh.title} / {ws.title}")
    data = ws.get_all_values()
    headers = data[0]
    rows = data[1:]
    print(f"   📥 ดึงมา: {len(rows)} แถว, {len(headers)} คอลัมน์")
    print(f"   คอลัมน์: {headers}")

    # หา code column
    code_col_idx = None
    for i, h in enumerate(headers):
        if h.strip() in ['Plan Code', 'Code', 'รหัสสาขา']:
            code_col_idx = i
            break
    if code_col_idx is None:
        # fallback: first column
        code_col_idx = 0
    print(f"   🔑 ใช้คอลัมน์: '{headers[code_col_idx]}' (index {code_col_idx})")

    new_bd = dict(old_bd)  # copy
    new_count = updated_count = 0
    for row in rows:
        if len(row) <= code_col_idx:
            continue
        code = str(row[code_col_idx]).strip().upper()
        if not code:
            continue
        row_dict = {headers[i]: (row[i] if i < len(row) else '') for i in range(len(headers))}
        if code in new_bd:
            if new_bd[code] != row_dict:
                new_bd[code] = row_dict
                updated_count += 1
        else:
            new_bd[code] = row_dict
            new_count += 1

    print(f"\n   🆕 สาขาใหม่: {new_count}")
    print(f"   🔄 อัพเดต:   {updated_count}")
    print(f"   📊 รวม:      {len(new_bd):,} สาขา")

    # บันทึก branch_data.json
    with open('branch_data.json', 'w', encoding='utf-8') as f:
        json.dump(new_bd, f, ensure_ascii=False, indent=2)
    print(f"   💾 บันทึก branch_data.json แล้ว")

except Exception as e:
    print(f"   ⚠️ ไม่สามารถ sync จาก Sheets: {e}")
    print("   📦 ใช้ branch_data.json เดิม")
    new_bd = dict(old_bd)

# ─── ตรวจสอบสาขาที่ขาด distance cache ────────────────────────────────────
print("\n🔍 ตรวจสอบ distance cache...")

missing_dc = []     # ขาด DC→branch
no_coords = []      # ไม่มีพิกัด

for code, b in new_bd.items():
    try:
        lat = float(b.get('\u0e25\u0e30') or b.get('lat') or b.get('_lat') or 0)
        lon = float(b.get('\u0e25\u0e2d\u0e07') or b.get('lon') or b.get('_lon') or 0)
    except:
        lat, lon = 0, 0

    if not lat or not lon:
        no_coords.append(code)
        continue

    key = f"{DC_LAT:.4f},{DC_LON:.4f}_{lat:.4f},{lon:.4f}"
    if key not in dc_cache:
        missing_dc.append({'code': code, 'lat': lat, 'lon': lon,
                           'province': b.get('\u0e08\u0e31\u0e07\u0e2b\u0e27\u0e31\u0e14', '')})

print(f"   ✅ มีพิกัด: {len(new_bd) - len(no_coords):,} สาขา")
print(f"   ⚠️  ไม่มีพิกัด: {len(no_coords)} สาขา")
print(f"   ❌ ขาด DC→branch ใน cache: {len(missing_dc)} สาขา")
if missing_dc:
    print(f"   ตัวอย่าง: {[m['code'] for m in missing_dc[:10]]}")
if no_coords:
    print(f"   ไม่มีพิกัด: {no_coords[:10]}")

# ─── คำนวณระยะทาง DC→branch ที่ขาด ──────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

new_pairs = 0
if missing_dc:
    print(f"\n🌐 คำนวณ DC→branch {len(missing_dc)} สาขา ด้วย OSRM...")
    dc_coord = (DC_LON, DC_LAT)
    for start in range(0, len(missing_dc), BATCH_SIZE - 1):
        chunk = missing_dc[start:start + BATCH_SIZE - 1]
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
                    if dist_m and dist_m > 0:
                        dc_cache[key] = round(dist_m / 1000.0, 3)
                    else:
                        # fallback haversine x1.35
                        dc_cache[key] = round(haversine(DC_LAT, DC_LON, b['lat'], b['lon']) * 1.35, 3)
                    new_pairs += 1
                print(f"   ✅ batch {start//( BATCH_SIZE-1)+1}: +{len(chunk)} สาขา")
            else:
                # fallback
                for b in chunk:
                    key = f"{DC_LAT:.4f},{DC_LON:.4f}_{b['lat']:.4f},{b['lon']:.4f}"
                    dc_cache[key] = round(haversine(DC_LAT, DC_LON, b['lat'], b['lon']) * 1.35, 3)
                    new_pairs += 1
        except Exception as e:
            print(f"   ⚠️ OSRM error: {e} — ใช้ haversine fallback")
            for b in chunk:
                key = f"{DC_LAT:.4f},{DC_LON:.4f}_{b['lat']:.4f},{b['lon']:.4f}"
                dc_cache[key] = round(haversine(DC_LAT, DC_LON, b['lat'], b['lon']) * 1.35, 3)
                new_pairs += 1
        time.sleep(OSRM_DELAY)

    with open('distance_cache.json', 'w', encoding='utf-8') as f:
        json.dump(dc_cache, f, ensure_ascii=False)
    print(f"   💾 อัพเดต distance_cache.json: +{new_pairs} pairs → รวม {len(dc_cache):,}")
else:
    print("   ✅ DC→branch ครบทุกสาขาแล้ว!")

# ─── ตรวจสอบ branch↔branch ภายในจังหวัดที่ขาด ─────────────────────────
print("\n🔍 ตรวจสอบ branch↔branch รายจังหวัด...")

# จัดกลุ่มสาขาตามจังหวัด
by_prov = defaultdict(list)
for code, b in new_bd.items():
    try:
        lat = float(b.get('\u0e25\u0e30') or b.get('lat') or b.get('_lat') or 0)
        lon = float(b.get('\u0e25\u0e2d\u0e07') or b.get('lon') or b.get('_lon') or 0)
    except:
        lat, lon = 0, 0
    if lat and lon:
        prov = b.get('\u0e08\u0e31\u0e07\u0e2b\u0e27\u0e31\u0e14', '')
        by_prov[prov].append({'code': code, 'lat': lat, 'lon': lon})

missing_provs = []
for prov, blist in by_prov.items():
    if len(blist) < 2:
        continue
    has_missing = any(
        f"{bi['lat']:.4f},{bi['lon']:.4f}_{bj['lat']:.4f},{bj['lon']:.4f}" not in dc_cache
        for bi in blist for bj in blist if bi is not bj
    )
    if has_missing:
        missing_provs.append(prov)

print(f"   จังหวัดที่ขาดข้อมูล branch↔branch: {len(missing_provs)}/{len(by_prov)}")
if missing_provs:
    print(f"   จังหวัด: {missing_provs[:20]}")

# ─── สรุปผล ─────────────────────────────────────────────────────────────--
print("\n" + "=" * 60)
print("📊 สรุปผล")
print("=" * 60)
print(f"  สาขาทั้งหมด (ชีต):   {len(new_bd):,}")
print(f"  สาขาเพิ่มใหม่:        {len(new_bd) - len(old_bd)}")
print(f"  ไม่มีพิกัด:           {len(no_coords)}")
print(f"  DC→branch pairs ใหม่: {new_pairs}")
print(f"  Distance cache รวม:   {len(dc_cache):,}")
print(f"  จังหวัดขาด bb-cache:  {len(missing_provs)}")

if missing_provs:
    print("\n⚠️  ยังขาด branch↔branch ของบางจังหวัด")
    print("    รัน: python precompute_branch_data.py  เพื่อเติม")
else:
    print("\n✅ distance cache ครบทุกสาขา!")

if len(missing_provs) > 0:
    ans = input("\nต้องการรัน precompute_branch_data.py เพื่อเติม branch↔branch ตอนนี้เลยไหม? (y/n): ").strip().lower()
    if ans == 'y':
        import subprocess
        subprocess.run([sys.executable, 'precompute_branch_data.py'])
