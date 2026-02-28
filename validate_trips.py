"""
validate_trips.py
=================
ตรวจสอบความถูกต้องของการจัดกลุ่มสาขาตามเงื่อนไขที่ตั้งใน app.py

เงื่อนไขที่ตรวจ:
  1. Zone consistency   - ทุกสาขาในทริปต้องอยู่ zone/highway เดียวกัน
  2. NO_CROSS_ZONE_PAIRS- ห้ามรวมจังหวัดที่กำหนดไว้ใน NO_CROSS_ZONE_PAIRS
  3. Geographic spread  - ห้ามสาขาในทริปห่างจาก centroid > 80 km
  4. Max drops         - 4W≤12, JB≤12 (Punthai 4W≤5, JB≤7)  [กรณีไม่มีน้ำหนักจริง snapshot ~1 drop/branch]
  5. Highway corridor  - สาขาในทริปต้องใช้ highway เดียวกัน (set intersection ≥ 1)
  6. Bearing coherence - ทิศทางสาขาจาก DC ไม่ควรต่างกันเกิน 135° ในทริปเดียว

ผลลัพธ์:
  validate_report.html  - รายงาน + แผนที่แสดง violations
"""

import json, math, sys, os, importlib.util
from collections import defaultdict

try:
    import folium
    from folium import plugins
except ImportError:
    print("pip install folium"); sys.exit(1)

# ──────────────────────────────────────────────────────────────
# 1.  Import LOGISTICS_ZONES, NO_CROSS_ZONE_PAIRS, LIMITS
#     โดยตรงจาก app.py  (ไม่ run side-effects)
# ──────────────────────────────────────────────────────────────
def _safe_import_from_app():
    """Import เฉพาะ constants จาก app.py แบบ safe (ไม่ run streamlit)."""
    import ast, re

    with open("app.py", "r", encoding="utf-8") as f:
        src = f.read()

    # eval namespace ที่ปลอดภัย
    ns = {}
    # ดึง LOGISTICS_ZONES, NO_CROSS_ZONE_PAIRS, LIMITS, PUNTHAI_LIMITS, REGION_ORDER
    blocks = {
        "LOGISTICS_ZONES": None,
        "NO_CROSS_ZONE_PAIRS": None,
        "LIMITS": None,
        "PUNTHAI_LIMITS": None,
        "DC_WANG_NOI_LAT": None,
        "DC_WANG_NOI_LON": None,
    }

    for name in blocks:
        # หา assignment: NAME = ...
        m = re.search(rf"^({re.escape(name)}\s*=\s*)", src, re.MULTILINE)
        if not m:
            continue
        start = m.start()
        chunk = src[start: start + 50000]  # ตัดพอ
        try:
            tree = ast.parse(chunk)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for t in node.targets:
                        if isinstance(t, ast.Name) and t.id == name:
                            val = ast.literal_eval(node.value)
                            blocks[name] = val
                            break
                if blocks[name] is not None:
                    break
        except Exception:
            pass

    return blocks


print("📥 Loading constants from app.py …")
_consts = _safe_import_from_app()

LOGISTICS_ZONES   = _consts.get("LOGISTICS_ZONES") or {}
NO_CROSS_ZONE_PAIRS = set()
for pair in (_consts.get("NO_CROSS_ZONE_PAIRS") or []):
    if len(pair) == 2:
        NO_CROSS_ZONE_PAIRS.add(tuple(pair))
        NO_CROSS_ZONE_PAIRS.add((pair[1], pair[0]))

LIMITS         = _consts.get("LIMITS") or {'4W':{'max_w':2500,'max_c':5,'max_drops':12},'JB':{'max_w':3500,'max_c':7,'max_drops':12},'6W':{'max_w':6000,'max_c':20,'max_drops':999}}
PUNTHAI_LIMITS = _consts.get("PUNTHAI_LIMITS") or LIMITS
DC_LAT = _consts.get("DC_WANG_NOI_LAT") or 14.1794
DC_LON = _consts.get("DC_WANG_NOI_LON") or 100.6481

print(f"   Zones: {len(LOGISTICS_ZONES)}")
print(f"   NO_CROSS pairs: {len(NO_CROSS_ZONE_PAIRS)//2}")
print(f"   DC: {DC_LAT}, {DC_LON}")

# ──────────────────────────────────────────────────────────────
# 2.  Build zone lookup (province/district/subdistrict → zone key)
# ──────────────────────────────────────────────────────────────
def _build_zone_lookup():
    """Create fast lookup: (province, district) → zone_name"""
    prov_map  = {}  # province → [zone_names]
    dist_map  = {}  # (province, district) → [zone_names]
    subdist_map = {}  # (province, district, subdistrict) → zone_name

    for zname, zdef in LOGISTICS_ZONES.items():
        provinces = zdef.get("provinces", [])
        districts = zdef.get("districts", [])
        subdistricts = zdef.get("subdistricts", [])

        if subdistricts:
            for p in provinces:
                for d in districts or ['']:
                    for s in subdistricts:
                        subdist_map[(p, d, s)] = zname
        elif districts:
            for p in provinces:
                for d in districts:
                    dist_map.setdefault((p, d), []).append(zname)
        else:
            for p in provinces:
                prov_map.setdefault(p, []).append(zname)

    return prov_map, dist_map, subdist_map

_prov_map, _dist_map, _subdist_map = _build_zone_lookup()

def get_zone(province, district="", subdistrict=""):
    province   = str(province  or "").strip()
    district   = str(district  or "").strip()
    subdistrict= str(subdistrict or "").strip()
    key3 = (province, district, subdistrict)
    if key3 in _subdist_map:
        return _subdist_map[key3]
    key2 = (province, district)
    if key2 in _dist_map:
        return _dist_map[key2][0]
    if province in _prov_map:
        return _prov_map[province][0]
    return "UNKNOWN"

def get_zone_hw(zone_name):
    return LOGISTICS_ZONES.get(zone_name, {}).get("highway", "UNK")

def get_zone_priority(zone_name):
    return LOGISTICS_ZONES.get(zone_name, {}).get("priority", 99)

# ──────────────────────────────────────────────────────────────
# 3.  Haversine + bearing + Road distance (from cache)
# ──────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def bearing(lat1, lon1, lat2, lon2):
    dL = math.radians(lon2 - lon1)
    x  = math.sin(dL) * math.cos(math.radians(lat2))
    y  = math.cos(math.radians(lat1))*math.sin(math.radians(lat2)) - \
         math.sin(math.radians(lat1))*math.cos(math.radians(lat2))*math.cos(dL)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def bearing_diff(b1, b2):
    d = abs(b1 - b2) % 360
    return d if d <= 180 else 360 - d

# ──────────────────────────────────────────────────────────────
# 3b. Load distance_cache.json  (road distances in km)
# ──────────────────────────────────────────────────────────────
print("\n📂 Loading distance_cache.json …")
_ROAD_CACHE: dict = {}
try:
    with open("distance_cache.json", "r", encoding="utf-8") as _f:
        _ROAD_CACHE = json.load(_f)
    print(f"   {len(_ROAD_CACHE):,} road-distance entries loaded")
except FileNotFoundError:
    print("   ⚠️  distance_cache.json not found — using haversine fallback")

_road_hit = _road_miss = 0

def _fmt(v: float) -> str:
    """Format lat/lon to 4 decimal places (matches cache key format)"""
    return f"{v:.4f}"

def road_dist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return road distance (km) from cache; fall back to haversine if missing."""
    global _road_hit, _road_miss
    key  = f"{_fmt(lat1)},{_fmt(lon1)}_{_fmt(lat2)},{_fmt(lon2)}"
    rkey = f"{_fmt(lat2)},{_fmt(lon2)}_{_fmt(lat1)},{_fmt(lon1)}"
    val  = _ROAD_CACHE.get(key) or _ROAD_CACHE.get(rkey)
    if val is not None:
        _road_hit += 1
        return float(val)
    _road_miss += 1
    return haversine(lat1, lon1, lat2, lon2)  # fallback

def road_dist_b(b1: dict, b2: dict) -> float:
    """road_dist between two branch dicts"""
    return road_dist(b1["lat"], b1["lon"], b2["lat"], b2["lon"])

def road_dist_dc(b: dict) -> float:
    """road_dist from DC to branch dict"""
    return road_dist(DC_LAT, DC_LON, b["lat"], b["lon"])

# ──────────────────────────────────────────────────────────────
# 4.  Load branch cache
# ──────────────────────────────────────────────────────────────
print("\n📂 Loading branch_clusters.json …")
with open("branch_clusters.json", "r", encoding="utf-8") as f:
    bc = json.load(f)

branch_info = bc.get("branch_info", {})
print(f"   {len(branch_info):,} branches")

# Filter arg
filter_prov = sys.argv[1] if len(sys.argv) > 1 else None

branches = []
for code, info in branch_info.items():
    lat = float(info.get("lat") or 0)
    lon = float(info.get("lon") or 0)
    if lat == 0 or lon == 0:
        continue
    prov = str(info.get("province") or "").strip()
    dist = str(info.get("district") or "").strip()
    subd = str(info.get("subdistrict") or "").strip()
    if filter_prov and filter_prov not in prov:
        continue
    zone = get_zone(prov, dist, subd)
    hw   = get_zone_hw(zone)
    brg  = bearing(DC_LAT, DC_LON, lat, lon)
    dist_dc = road_dist(DC_LAT, DC_LON, lat, lon)   # road distance (km)
    branches.append({
        "code": code,
        "name": info.get("name", code),
        "lat": lat, "lon": lon,
        "province": prov,
        "district": dist,
        "subdistrict": subd,
        "zone": zone,
        "hw": hw,
        "bearing": brg,
        "dist_dc": dist_dc,
    })

print(f"   Valid with coords: {len(branches):,}")

# ──────────────────────────────────────────────────────────────
# 5.  Simulate trips (same greedy NN as map_all_branches.py)
#     แต่ตอนนี้ group ตาม zone highway corridor ก่อน
# ──────────────────────────────────────────────────────────────
print("\n🔄 Simulating trips …")

# group by (zone, province) เพื่อให้ใกล้เคียง app.py
zone_groups = defaultdict(list)
for b in branches:
    zone_groups[b["zone"]].append(b)

all_trips = []  # each trip = list of branch dicts (+ meta)

for zone_name, zone_branches in sorted(zone_groups.items(), key=lambda x: get_zone_priority(x[0])):
    zone_hw = get_zone_hw(zone_name)
    # sort farthest first (LIFO)
    zone_branches_s = sorted(zone_branches, key=lambda b: -b["dist_dc"])

    used = set()
    for seed in zone_branches_s:
        if seed["code"] in used:
            continue

        trip = [seed]
        used.add(seed["code"])
        cur_lat, cur_lon = seed["lat"], seed["lon"]
        candidates = [b for b in zone_branches_s if b["code"] not in used]

        while candidates:
            # nearest (road km) within 120km and bearing within 90° of seed
            best = None
            best_d = float("inf")
            for b in candidates:
                d = road_dist(cur_lat, cur_lon, b["lat"], b["lon"])
                bd = bearing_diff(seed["bearing"], b["bearing"])
                if d < best_d and d <= 120 and bd <= 90:
                    best_d = d
                    best = b
            if best is None:
                break
            trip.append(best)
            used.add(best["code"])
            cur_lat, cur_lon = best["lat"], best["lon"]
            candidates = [b for b in zone_branches_s if b["code"] not in used]
            if len(trip) >= 25:
                break

        all_trips.append({
            "zone": zone_name,
            "hw": zone_hw,
            "branches": trip,
        })

    # leftovers
    for b in zone_branches_s:
        if b["code"] not in used:
            all_trips.append({"zone": zone_name, "hw": zone_hw, "branches": [b]})

print(f"   Trips formed: {len(all_trips):,}")

# ──────────────────────────────────────────────────────────────
# 6.  VALIDATION  — ตรวจเงื่อนไขทุกข้อ
# ──────────────────────────────────────────────────────────────
print("\n🔍 Validating trips …")

VIOLATIONS = []  # list of dicts

def hw_overlap(hw1, hw2):
    """True ถ้า highway set มี overlap"""
    s1 = set(str(hw1).split("/"))
    s2 = set(str(hw2).split("/"))
    return bool(s1 & s2)

for tidx, trip in enumerate(all_trips):
    blist = trip["branches"]
    if not blist:
        continue

    codes     = [b["code"] for b in blist]
    zones     = [b["zone"] for b in blist]
    provs     = list({b["province"] for b in blist})
    hws       = [b["hw"] for b in blist]
    bearings_ = [b["bearing"] for b in blist]
    drops     = len(blist)

    trip_zone = trip["zone"]
    trip_hw   = trip["hw"]

    # -- C1: Zone consistency (ทุกสาขาต้องอยู่ zone เดียวกัน)
    if len(set(zones)) > 1:
        VIOLATIONS.append({
            "trip_id": tidx,
            "type": "ZONE_MIX",
            "severity": "HIGH",
            "msg": f"รวมหลาย zone: {set(zones)}",
            "codes": codes,
            "blist": blist,
        })

    # -- C2: NO_CROSS_ZONE_PAIRS (ห้ามรวมจังหวัดคู่นี้)
    for i in range(len(provs)):
        for j in range(i + 1, len(provs)):
            pair = (provs[i], provs[j])
            if pair in NO_CROSS_ZONE_PAIRS:
                VIOLATIONS.append({
                    "trip_id": tidx,
                    "type": "NO_CROSS",
                    "severity": "HIGH",
                    "msg": f"ห้ามรวม {provs[i]} + {provs[j]}",
                    "codes": codes,
                    "blist": blist,
                })

    # -- C3: Highway coherence (highway ต้องมี overlap)
    unique_hws = list({b["hw"] for b in blist})
    if len(unique_hws) > 1:
        # ตรวจ pairwise
        incoherent = []
        for i in range(len(unique_hws)):
            for j in range(i+1, len(unique_hws)):
                if not hw_overlap(unique_hws[i], unique_hws[j]):
                    incoherent.append((unique_hws[i], unique_hws[j]))
        if incoherent:
            VIOLATIONS.append({
                "trip_id": tidx,
                "type": "HW_MISMATCH",
                "severity": "MED",
                "msg": f"Highway ไม่ตรงกัน: {incoherent}",
                "codes": codes,
                "blist": blist,
            })

    # -- C4: Geographic spread — ใช้ road distance สูงสุดระหว่างคู่สาขาในทริป
    if len(blist) >= 2:
        max_pair_road = 0.0
        worst_pair = (None, None)
        for _i in range(len(blist)):
            for _j in range(_i+1, len(blist)):
                _d = road_dist_b(blist[_i], blist[_j])
                if _d > max_pair_road:
                    max_pair_road = _d
                    worst_pair = (blist[_i], blist[_j])
        if max_pair_road > 120:   # road km threshold (ห่างกันเกิน 120km ถนนจริง)
            _wp = f"{worst_pair[0]['code']}↔{worst_pair[1]['code']}" if worst_pair[0] else ""
            VIOLATIONS.append({
                "trip_id": tidx,
                "type": "GEO_SPREAD",
                "severity": "MED",
                "msg": f"[{trip_zone}] สาขาห่างกัน (road) เกิน: {max_pair_road:.0f} km (สูงสุด 120km) — {_wp} — จังหวัด: {sorted(set(b['province'] for b in blist))}",
                "codes": codes,
                "blist": blist,
            })

    # -- C5: Bearing coherence (ทิศทางจาก DC ไม่ควรต่างกัน > 155°)
    # ยกเว้น: สาขาใกล้ DC < 60km ถนน (ปริมณฑล/ใกล้) ทิศได้ทุกทาง
    far_branches = [b for b in blist if b["dist_dc"] > 60]
    if len(far_branches) >= 2:
        far_bearings = [b["bearing"] for b in far_branches]
        max_brg_diff = max(
            bearing_diff(far_bearings[i], far_bearings[j])
            for i in range(len(far_bearings))
            for j in range(i+1, len(far_bearings))
        )
        if max_brg_diff > 155:
            VIOLATIONS.append({
                "trip_id": tidx,
                "type": "BEARING_JUMP",
                "severity": "HIGH",
                "msg": f"[{trip_zone}] สาขาอยู่คนละทิศจาก DC: max bearing diff = {max_brg_diff:.0f}° (เฉพาะ >40km) — จังหวัด: {sorted(set(b['province'] for b in far_branches))}",
                "codes": [b["code"] for b in far_branches],
                "blist": far_branches,
            })

    # -- C6: Drops limit (simulate: 1 drop/branch, ไม่มีข้อมูลรถจริง)
    # 4W/JB ≤ 12 drops, 6W = ไม่จำกัด (999)
    # proxy: ถ้าทุกสาขาอยู่ห่าง DC ≤ 100km (ถนนจริง) = ปริมณฑล/ใกล้ → อาจใช้ 4W/JB
    is_short_haul = all(b["dist_dc"] <= 100 for b in blist)
    if drops > 12 and is_short_haul:
        VIOLATIONS.append({
            "trip_id": tidx,
            "type": "OVER_DROPS",
            "severity": "LOW",
            "msg": f"สาขาในทริปเกิน 12 drops ({drops} branches) ระยะใกล้ — ถ้าใช้ 4W/JB ต้อง split",
            "codes": codes,
            "blist": blist,
        })

# -- C7: UNKNOWN zone branches
unknown_branches = [b for b in branches if b["zone"] == "UNKNOWN"]
if unknown_branches:
    from collections import Counter as _C
    unk_provs = _C(b["province"] for b in unknown_branches).most_common(5)
    prov_str = ", ".join(f"{p}({c})" for p,c in unk_provs)
    VIOLATIONS.append({
        "trip_id": -1,
        "type": "UNKNOWN_ZONE",
        "severity": "LOW",
        "msg": f"ไม่พบ zone ใน LOGISTICS_ZONES: {len(unknown_branches)} สาขา ({prov_str}) — หมายเหตุ: app.py มี .strip() แล้ว ตรวจสอบ LOGISTICS_ZONES ว่าครอบจังหวัดเหล่านี้",
        "codes": [b["code"] for b in unknown_branches],
        "blist": unknown_branches,
    })

# -- C8: Zone with too many provinces (zone too wide)
for zname, zbrs in zone_groups.items():
    if zname == "UNKNOWN":
        continue
    zprovs = {b["province"] for b in zbrs}
    if len(zprovs) >= 5:  # ≥5 จังหวัดใน zone เดียว = กว้างเกินไป
        VIOLATIONS.append({
            "trip_id": -1,
            "type": "ZONE_TOO_WIDE",
            "severity": "MED",
            "msg": f"Zone {zname} ครอบ {len(zprovs)} จังหวัด: {sorted(zprovs)}",
            "codes": [b["code"] for b in zbrs[:6]],
            "blist": zbrs[:6],
        })

# Summary
severity_count = defaultdict(int)
type_count = defaultdict(int)
for v in VIOLATIONS:
    severity_count[v["severity"]] += 1
    type_count[v["type"]] += 1

print(f"\n{'='*55}")
print(f"  VIOLATIONS FOUND: {len(VIOLATIONS)}")
print(f"{'='*55}")
for sev in ["HIGH", "MED", "LOW"]:
    print(f"  {sev}: {severity_count.get(sev, 0)}")
print(f"{'─'*55}")
for tp, cnt in sorted(type_count.items(), key=lambda x: -x[1]):
    print(f"  {tp:<20}: {cnt}")
print(f"{'='*55}")

# ──────────────────────────────────────────────────────────────
# 7.  Build Validation Report HTML (Map + Table)
# ──────────────────────────────────────────────────────────────
print("\n🗺️  Building validation map …")

SEV_COLOR = {"HIGH": "#d62728", "MED": "#ff7f0e", "LOW": "#bcbd22"}
TYPE_ICON = {
    "ZONE_MIX":    "⚠️ Zone Mix",
    "NO_CROSS":    "🚫 No Cross",
    "HW_MISMATCH": "🛣️ Highway",
    "GEO_SPREAD":  "📍 Spread",
    "BEARING_JUMP":"🧭 Bearing",
    "OVER_DROPS":  "📦 Drops",
    "UNKNOWN_ZONE":"❓ Unknown",
}

center_lat = sum(b["lat"] for b in branches) / max(1, len(branches))
center_lon = sum(b["lon"] for b in branches) / max(1, len(branches))

m = folium.Map(location=[center_lat, center_lon], zoom_start=6,
               tiles="CartoDB Positron", prefer_canvas=True)
plugins.Fullscreen(position="topleft", title="เต็มจอ", force_separate_button=True).add_to(m)

# DC marker
folium.Marker(
    [DC_LAT, DC_LON],
    popup="<b>🏭 DC Wang Noi</b>",
    tooltip="DC Wang Noi",
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(m)

# ─ Layer: All branches (background gray dots)
fg_all = folium.FeatureGroup(name="✅ ทุกสาขา (background)", show=True)
for b in branches:
    folium.CircleMarker(
        [b["lat"], b["lon"]], radius=3,
        color="#aaa", fill=True, fill_color="#aaa", fill_opacity=0.5,
        tooltip=f"{b['code']} {b['province']} {b['district']} | zone:{b['zone']}",
    ).add_to(fg_all)
fg_all.add_to(m)

# ─ Layer: Violations
added_coords = set()
fg_viol = {"HIGH": folium.FeatureGroup(name="🔴 HIGH violations", show=True),
           "MED":  folium.FeatureGroup(name="🟠 MED violations",  show=True),
           "LOW":  folium.FeatureGroup(name="🟡 LOW violations",  show=True)}

for v in VIOLATIONS:
    sev   = v["severity"]
    color = SEV_COLOR[sev]
    vtype = TYPE_ICON.get(v["type"], v["type"])
    fg    = fg_viol[sev]

    blist = v.get("blist", [])
    if not blist:
        continue

    # draw route lines for the offending trip
    if len(blist) >= 2:
        route_pts = [[DC_LAT, DC_LON]] + [[b["lat"], b["lon"]] for b in blist] + [[DC_LAT, DC_LON]]
        folium.PolyLine(
            route_pts, weight=3, color=color, opacity=0.8, dash_array="8 4",
            tooltip=f"{vtype}: {v['msg']}"
        ).add_to(fg)

    for b in blist:
        key = (round(b["lat"], 4), round(b["lon"], 4))
        if key in added_coords:
            continue
        added_coords.add(key)

        popup_html = (
            f"<b style='color:{color}'>{vtype}</b><br>"
            f"<b>{b['code']}</b> {b['name']}<br>"
            f"{b['district']} {b['province']}<br>"
            f"zone: {b['zone']} | hw: {b['hw']}<br>"
            f"<i>{v['msg']}</i>"
        )
        folium.CircleMarker(
            [b["lat"], b["lon"]], radius=7,
            color=color, fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{b['code']} [{v['type']}]",
        ).add_to(fg)

for fg in fg_viol.values():
    fg.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

# Road cache hit rate
print(f"\n📊 Road cache: {_road_hit:,} hits / {_road_miss:,} misses ({100*_road_hit/max(1,_road_hit+_road_miss):.1f}% coverage)")

# ─ Stats panel
stats_rows = ""
for v in VIOLATIONS[:200]:  # แสดงสูงสุด 200 rows
    sev   = v["severity"]
    color = SEV_COLOR[sev]
    vtype = TYPE_ICON.get(v["type"], v["type"])
    codes_str = ", ".join(v["codes"][:6]) + ("…" if len(v["codes"]) > 6 else "")
    stats_rows += (
        f'<tr style="border-bottom:1px solid #eee">'
        f'<td style="color:{color};font-weight:bold">{sev}</td>'
        f'<td>{vtype}</td>'
        f'<td style="max-width:180px;word-wrap:break-word;font-size:10px">{v["msg"]}</td>'
        f'<td style="font-size:10px;color:#555">{codes_str}</td>'
        f'</tr>'
    )

summary_rows = "".join(
    f'<tr><td>{tp}</td><td align="right">{cnt}</td></tr>'
    for tp, cnt in sorted(type_count.items(), key=lambda x: -x[1])
)

panel = f"""
<div id="val-panel" style="position:fixed;top:10px;right:10px;z-index:9999;
     background:rgba(255,255,255,0.96);padding:12px 14px;border-radius:8px;
     font-family:Arial;font-size:12px;box-shadow:0 2px 12px rgba(0,0,0,0.35);
     max-height:92vh;overflow-y:auto;max-width:420px;">
  <b style="font-size:14px">🔍 Validation Report</b>
  {"<br><i>(filter: "+filter_prov+")</i>" if filter_prov else ""}
  <br><span style="color:#888">สาขา: {len(branches):,} | ทริป: {len(all_trips):,} | road cache: {_road_hit:,} hits</span>
  <hr style="margin:6px 0">
  <b>Violations: {len(VIOLATIONS)}</b>
  &emsp;🔴 HIGH:{severity_count.get('HIGH',0)}
  &emsp;🟠 MED:{severity_count.get('MED',0)}
  &emsp;🟡 LOW:{severity_count.get('LOW',0)}
  <hr style="margin:6px 0">
  <table style="font-size:11px;border-collapse:collapse;width:100%">
    <tr style="background:#f0f0f0"><th>ประเภท</th><th align="right">จำนวน</th></tr>
    {summary_rows}
  </table>
  <hr style="margin:6px 0">
  <details open>
    <summary style="cursor:pointer;font-weight:bold">รายการ violations</summary>
    <table style="font-size:10px;border-collapse:collapse;width:100%;margin-top:4px">
      <tr style="background:#f5f5f5">
        <th>Sev</th><th>ประเภท</th><th>รายละเอียด</th><th>สาขา</th>
      </tr>
      {stats_rows}
    </table>
  </details>
  <hr style="margin:6px 0">
  <small style="color:#666">คลิกสาขาสีแดง/ส้ม เพื่อดูรายละเอียด</small>
</div>
"""
m.get_root().html.add_child(folium.Element(panel))

out_file = "validate_report.html"
m.save(out_file)
print(f"\n✅ Saved: {out_file}")
print(f"   HIGH violations : {severity_count.get('HIGH', 0)}")
print(f"   MED  violations : {severity_count.get('MED',  0)}")
print(f"   LOW  violations : {severity_count.get('LOW',  0)}")
print("\nOpen validate_report.html in browser to see detailed map.")

# ──────────────────────────────────────────────────────────────
# 8.  Plain-text summary for quick review
# ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DETAIL: HIGH violations")
print("="*60)
for v in VIOLATIONS:
    if v["severity"] != "HIGH":
        continue
    codes_str = ", ".join(v["codes"][:8]) + ("…" if len(v["codes"]) > 8 else "")
    print(f"  [{v['type']}] Trip#{v['trip_id']} | {v['msg']}")
    print(f"    codes: {codes_str}")

print("\n" + "="*60)
print("DETAIL: MED violations")
print("="*60)
for v in VIOLATIONS:
    if v["severity"] != "MED":
        continue
    codes_str = ", ".join(v["codes"][:8]) + ("…" if len(v["codes"]) > 8 else "")
    print(f"  [{v['type']}] Trip#{v['trip_id']} | {v['msg']}")
    print(f"    codes: {codes_str}")
