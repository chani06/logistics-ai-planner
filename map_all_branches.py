"""
map_all_branches.py - Diagnostic: Plot all cached branches + optimal group by travel distance

Usage:
    python map_all_branches.py [province]

Output:
    map_branches.html - Interactive Folium map
"""
import json, math, sys
from collections import defaultdict

try:
    import folium
    from folium import plugins
except ImportError:
    print("pip install folium"); sys.exit(1)

# ──────────────────────────────────────────
# Constants
# ──────────────────────────────────────────
DC_LAT, DC_LON = 14.1459, 100.6873

# Highway corridors: (corridor_name, highway_tags, max_combined_km_between_branches)
HIGHWAY_CORRIDORS = {
    "HW1_เหนือ":       {"tags": {"1"}, "color": "#1f77b4"},
    "HW2_อีสานเหนือ": {"tags": {"2"}, "color": "#ff7f0e"},
    "HW3_ตะวันออก":   {"tags": {"3", "331"}, "color": "#2ca02c"},
    "HW304_ชลฉะ":     {"tags": {"304", "331"}, "color": "#9467bd"},
    "HW4_ใต้":        {"tags": {"4", "35", "401", "402"}, "color": "#8c564b"},
    "HW11_แพร่น่าน":  {"tags": {"11", "101"}, "color": "#e377c2"},
    "HW12_พิษณุโลก":  {"tags": {"12"}, "color": "#7f7f7f"},
    "HW21_เพชรบูรณ์": {"tags": {"21"}, "color": "#bcbd22"},
    "HW24_อีสานใต้":  {"tags": {"24"}, "color": "#17becf"},
    "BKK_ปริมณฑล":    {"tags": {"BKK", "กทม", "CBD", "9"}, "color": "#aec7e8"},
}

# Province → Highway (simplified)
PROVINCE_HW = {
    "พะเยา":"1", "เชียงราย":"1", "ลำพูน":"1", "ลำปาง":"1",
    "แพร่":"11", "น่าน":"101", "อุตรดิตถ์":"11", "สุโขทัย":"11",
    "พิษณุโลก":"12", "พิจิตร":"1", "กำแพงเพชร":"1", "นครสวรรค์":"1",
    "เพชรบูรณ์":"21", "ชัยภูมิ":"2", "นครราชสีมา":"2",
    "ขอนแก่น":"2", "อุดรธานี":"2", "หนองคาย":"2", "เลย":"2",
    "บึงกาฬ":"2", "สกลนคร":"2", "นครพนม":"2",
    "หนองบัวลำภู":"2",
    "อุบลราชธานี":"24", "ศรีสะเกษ":"24", "สุรินทร์":"24", "บุรีรัมย์":"24",
    "ยโสธร":"24", "อำนาจเจริญ":"24", "มุกดาหาร":"24",
    "ร้อยเอ็ด":"2", "มหาสารคาม":"2", "กาฬสินธุ์":"2",
    "ชลบุรี":"3", "ระยอง":"3", "จันทบุรี":"3", "ตราด":"3",
    "ฉะเชิงเทรา":"304", "ปราจีนบุรี":"304", "สระแก้ว":"33",
    "เพชรบุรี":"4", "ประจวบคีรีขันธ์":"4", "ชุมพร":"4",
    "สุราษฎร์ธานี":"4", "นครศรีธรรมราช":"4",
    "สงขลา":"4", "ปัตตานี":"4", "ยะลา":"4", "นราธิวาส":"4",
    "ระนอง":"4", "กระบี่":"4", "พังงา":"4", "ภูเก็ต":"4",
    "ตรัง":"4", "สตูล":"4", "พัทลุง":"4",
    "กรุงเทพมหานคร":"BKK", "นนทบุรี":"BKK", "ปทุมธานี":"BKK",
    "สมุทรปราการ":"BKK", "สมุทรสาคร":"BKK", "นครปฐม":"BKK",
    "สมุทรสงคราม":"BKK","พระนครศรีอยุธยา":"BKK", "สระบุรี":"BKK",
    "อ่างทอง":"BKK", "สิงห์บุรี":"BKK", "ชัยนาท":"BKK", "ลพบุรี":"BKK",
    "นครนายก":"BKK", "ราชบุรี":"4", "กาญจนบุรี":"4", "สุพรรณบุรี":"BKK",
    "อุทัยธานี":"BKK", "ตาก":"1",
}

# Override สมุทรปราการ บางบ่อ/บางเสาธง → Hwy 3 (ทิศทางเดียวกับชลบุรี)
DISTRICT_HW_OVERRIDE = {
    ("สมุทรปราการ","บางบ่อ"):     "3",
    ("สมุทรปราการ","บางเสาธง"):   "3",
    # ชลบุรีในแผ่นดิน → Hwy 304/331
    ("ชลบุรี","พนัสนิคม"):        "304",
    ("ชลบุรี","บ่อทอง"):          "304",
    ("ชลบุรี","หนองใหญ่"):        "304",
    ("ชลบุรี","เกาะจันทร์"):      "304",
}


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def get_hw(province, district=""):
    key = (province, district)
    if key in DISTRICT_HW_OVERRIDE:
        return DISTRICT_HW_OVERRIDE[key]
    return PROVINCE_HW.get(province, "UNK")


def nearest_neighbor_sort(branches, dc_lat=DC_LAT, dc_lon=DC_LON):
    """Sort branches by nearest-neighbor from DC to minimize total distance."""
    if not branches:
        return []
    remaining = list(branches)
    path = []
    cur_lat, cur_lon = dc_lat, dc_lon
    while remaining:
        nearest = min(remaining, key=lambda b: haversine(cur_lat, cur_lon, b["lat"], b["lon"]))
        path.append(nearest)
        cur_lat, cur_lon = nearest["lat"], nearest["lon"]
        remaining.remove(nearest)
    return path


def greedy_trips(branches, max_weight=7000.0, max_cube=25.0, dc_lat=DC_LAT, dc_lon=DC_LON):
    """Group branches into trips with nearest-neighbor & weight/cube constraints."""
    remaining = sorted(branches, key=lambda b: haversine(dc_lat, dc_lon, b["lat"], b["lon"]), reverse=True)
    trips = []
    used = set()

    while remaining:
        # Pick farthest unused as trip seed
        seed = next((b for b in remaining if b["code"] not in used), None)
        if not seed:
            break

        trip = [seed]
        used.add(seed["code"])
        trip_w = seed.get("weight", 0)
        trip_c = seed.get("cube", 0)
        cur_lat, cur_lon = seed["lat"], seed["lon"]

        # Greedy NN fill
        candidates = [b for b in remaining if b["code"] not in used]
        while candidates:
            best = None
            best_dist = float("inf")
            for b in candidates:
                d = haversine(cur_lat, cur_lon, b["lat"], b["lon"])
                if d < best_dist:
                    best_dist = d
                    best = b
            if best is None or best_dist > 100:   # stop if nearest is >100km away
                break
            if trip_w + best.get("weight", 100) > max_weight:
                break
            if trip_c + best.get("cube", 1) > max_cube:
                break
            trip.append(best)
            used.add(best["code"])
            trip_w += best.get("weight", 0)
            trip_c += best.get("cube", 0)
            cur_lat, cur_lon = best["lat"], best["lon"]
            candidates = [b for b in remaining if b["code"] not in used]

        trips.append(trip)

    # Re-order remaining into single trips
    leftover = [b for b in remaining if b["code"] not in used]
    for b in leftover:
        trips.append([b])

    return trips


def total_route_distance(trips, dc_lat=DC_LAT, dc_lon=DC_LON):
    total = 0
    for trip in trips:
        prev_lat, prev_lon = dc_lat, dc_lon
        for b in trip:
            total += haversine(prev_lat, prev_lon, b["lat"], b["lon"])
            prev_lat, prev_lon = b["lat"], b["lon"]
        total += haversine(prev_lat, prev_lon, dc_lat, dc_lon)
    return total


# ──────────────────────────────────────────
# Load branch cache
# ──────────────────────────────────────────
print("📂 Loading branch cache...")
with open("branch_clusters.json", "r", encoding="utf-8") as f:
    bc = json.load(f)

branch_info = bc.get("branch_info", {})
print(f"   {len(branch_info):,} branches loaded")

# Filter argument
filter_prov = sys.argv[1] if len(sys.argv) > 1 else None

# Build branch list
branches_by_hw = defaultdict(list)
all_branches = []

for code, info in branch_info.items():
    lat = float(info.get("lat", 0) or 0)
    lon = float(info.get("lon", 0) or 0)
    prov = info.get("province", "")
    dist = info.get("district", "")
    name = info.get("name", code)

    if lat == 0 or lon == 0:
        continue
    if filter_prov and filter_prov not in prov:
        continue

    hw = get_hw(prov, dist)
    b = {
        "code": code, "name": name,
        "lat": lat, "lon": lon,
        "province": prov, "district": dist,
        "subdistrict": info.get("subdistrict", ""),
        "hw": hw,
        "dist_from_dc": haversine(DC_LAT, DC_LON, lat, lon),
        "weight": 500.0,  # placeholder (no actual weight in cache)
        "cube": 1.0,
    }
    all_branches.append(b)
    branches_by_hw[hw].append(b)

print(f"   Valid with coords: {len(all_branches):,}")
print(f"   Highway corridors: {len(branches_by_hw)}")
for hw, bl in sorted(branches_by_hw.items(), key=lambda x: -len(x[1])):
    print(f"      {hw:20s}: {len(bl):4d} branches")

# ──────────────────────────────────────────
# Run grouping by highway corridor + nearest-neighbor
# ──────────────────────────────────────────
print("\n🔄 Grouping branches into trips per highway corridor...")

# Color palette (50 distinct colors per corridor)
PALETTE = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
    "#a65628","#f781bf","#1b9e77","#d95f02","#7570b3",
    "#e7298a","#66a61e","#e6ab02","#a6761d","#666666",
    "#1f78b4","#33a02c","#fb9a99","#fdbf6f","#cab2d6",
    "#b15928","#8dd3c7","#ffffb3","#bebada","#fb8072",
    "#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9",
]

HW_BASE_COLOR = {
    "1": "#1f77b4", "11": "#e377c2", "101": "#de8cca",
    "12": "#7f7f7f", "2": "#ff7f0e", "3": "#2ca02c",
    "304": "#9467bd", "331": "#9467bd", "33": "#8c564b",
    "4": "#8c564b", "35": "#c49c94", "401": "#c5b0d5",
    "21": "#bcbd22", "24": "#17becf",
    "BKK": "#aec7e8", "UNK": "#999",
}

all_trips_by_hw = {}
trip_summary = []

for hw, hw_branches in branches_by_hw.items():
    # Sort by distance (farthest first within corridor)
    hw_branches_sorted = sorted(hw_branches, key=lambda b: -b["dist_from_dc"])

    # Split into sub-groups by province proximity (branches >150km apart in same corridor split)
    # Also split: ชลบุรีชายฝั่ง vs ชลบุรีเหนือ/ฉะเชิงเทรา
    prov_groups = defaultdict(list)
    for b in hw_branches_sorted:
        key = f"{b['province']}_{b['district'][:2]}"
        prov_groups[key].append(b)

    # Greedy nearest-neighbor trips within corridor
    # Use cube limit = float('inf') because we don't have real weights in cache
    hw_trips = []
    remaining = list(hw_branches_sorted)
    used_codes = set()

    while remaining:
        seed = next((b for b in remaining if b["code"] not in used_codes), None)
        if not seed:
            break

        trip = [seed]
        used_codes.add(seed["code"])
        cur_lat, cur_lon = seed["lat"], seed["lon"]

        candidates = [b for b in remaining if b["code"] not in used_codes]
        while candidates:
            best = min(candidates, key=lambda b: haversine(cur_lat, cur_lon, b["lat"], b["lon"]))
            dist_to_best = haversine(cur_lat, cur_lon, best["lat"], best["lon"])

            # Stop if nearest is >80km away (different sub-zone)
            if dist_to_best > 80:
                break

            trip.append(best)
            used_codes.add(best["code"])
            cur_lat, cur_lon = best["lat"], best["lon"]
            candidates = [b for b in remaining if b["code"] not in used_codes]

            # Max 25 branches per trip
            if len(trip) >= 25:
                break

        hw_trips.append(trip)

    # Handle leftovers
    for b in remaining:
        if b["code"] not in used_codes:
            hw_trips.append([b])

    all_trips_by_hw[hw] = hw_trips

    # Stats
    total_dist = total_route_distance(hw_trips)
    trip_summary.append({
        "highway": hw,
        "branches": len(hw_branches),
        "trips": len(hw_trips),
        "total_km": total_dist,
        "avg_km": total_dist / max(1, len(hw_trips)),
    })

# Sort summary by branches desc
trip_summary.sort(key=lambda x: -x["branches"])

print("\n📊 Grouping results:")
print(f"{'Highway':<22} {'Branches':>8} {'Trips':>6} {'Total km':>10} {'Avg km/trip':>12}")
print("-" * 62)
grand_branches = grand_trips = grand_km = 0
for s in trip_summary:
    print(f"{s['highway']:<22} {s['branches']:>8,} {s['trips']:>6,} {s['total_km']:>10,.0f} {s['avg_km']:>12,.0f}")
    grand_branches += s["branches"]
    grand_trips    += s["trips"]
    grand_km       += s["total_km"]
print("-" * 62)
print(f"{'TOTAL':<22} {grand_branches:>8,} {grand_trips:>6,} {grand_km:>10,.0f} {grand_km/max(1,grand_trips):>12,.0f}")

# ──────────────────────────────────────────
# Build Folium map
# ──────────────────────────────────────────
print("\n🗺️  Building map...")

center_lat = sum(b["lat"] for b in all_branches) / max(1, len(all_branches))
center_lon = sum(b["lon"] for b in all_branches) / max(1, len(all_branches))

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

# Feature groups per highway
layer_groups = {}
trip_global_idx = 0

for hw, trips in sorted(all_trips_by_hw.items(), key=lambda x: -len(x[1])):
    hw_color = HW_BASE_COLOR.get(hw, "#888")
    fg = folium.FeatureGroup(name=f"Hwy {hw} ({sum(len(t) for t in trips)} branches, {len(trips)} trips)", show=True)
    layer_groups[hw] = fg

    for trip_local_idx, trip in enumerate(trips):
        trip_color = PALETTE[trip_global_idx % len(PALETTE)]
        trip_global_idx += 1

        if not trip:
            continue

        # Draw route line: DC → branch1 → branch2 → ... → DC
        route_latlons = [[DC_LAT, DC_LON]] + [[b["lat"], b["lon"]] for b in trip] + [[DC_LAT, DC_LON]]
        route_dist = sum(
            haversine(route_latlons[i][0], route_latlons[i][1],
                      route_latlons[i+1][0], route_latlons[i+1][1])
            for i in range(len(route_latlons)-1)
        )

        folium.PolyLine(
            locations=route_latlons,
            weight=2,
            color=trip_color,
            opacity=0.5,
            tooltip=f"Hwy {hw} Trip {trip_local_idx+1} ({len(trip)} branches, {route_dist:.0f}km)",
        ).add_to(fg)

        # Branch markers
        for seq, b in enumerate(trip):
            label = f'<div style="background:{trip_color};color:#fff;border-radius:50%;' \
                    f'width:20px;height:20px;text-align:center;line-height:20px;font-size:9px;' \
                    f'font-weight:bold;border:1px solid #333;">{seq+1}</div>'

            popup_html = (
                f"<b>{b['code']}</b><br>"
                f"{b['name']}<br>"
                f"<i>{b['district']} {b['province']}</i><br>"
                f"Hwy {hw} | Trip {trip_local_idx+1} seq {seq+1}<br>"
                f"ห่าง DC: {b['dist_from_dc']:.1f} km"
            )
            folium.Marker(
                location=[b["lat"], b["lon"]],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{b['code']} {b['province']} {b['district']}",
                icon=folium.DivIcon(html=label, icon_size=(20, 20), icon_anchor=(10, 10)),
            ).add_to(fg)

    fg.add_to(m)

# Layer control
folium.LayerControl(collapsed=True).add_to(m)

# ──────────────────────────────────────────
# Stats panel (HTML overlay)
# ──────────────────────────────────────────
stats_html = f"""
<div style="position:fixed;top:10px;right:10px;z-index:9999;background:rgba(255,255,255,0.95);
            padding:12px 16px;border-radius:8px;font-family:Arial;font-size:12px;
            box-shadow:0 2px 10px rgba(0,0,0,0.3);max-height:90vh;overflow-y:auto;max-width:320px;">
  <b style="font-size:14px;">📊 Branch Map Summary</b>
  {"<br><i>(filter: " + filter_prov + ")</i>" if filter_prov else ""}
  <hr style="margin:6px 0">
  <b>Total branches:</b> {grand_branches:,}<br>
  <b>Grouped into trips:</b> {grand_trips:,}<br>
  <b>Total travel km:</b> {grand_km:,.0f} km<br>
  <hr style="margin:6px 0">
  <table style="width:100%;font-size:11px;border-collapse:collapse">
    <tr style="background:#f0f0f0">
      <th align="left">Highway</th>
      <th align="right">Branches</th>
      <th align="right">Trips</th>
      <th align="right">Avg km</th>
    </tr>
"""
for s in trip_summary:
    stats_html += (
        f'<tr><td>{s["highway"]}</td>'
        f'<td align="right">{s["branches"]}</td>'
        f'<td align="right">{s["trips"]}</td>'
        f'<td align="right">{s["avg_km"]:.0f}</td></tr>'
    )
stats_html += """
  </table>
  <hr style="margin:6px 0">
  <small style="color:#666">คลิก branch เพื่อดูรายละเอียด<br>
  เส้นสี = trip เดียวกัน<br>
  Layer control (ขวาบน) แสดง/ซ่อน highway corridor</small>
</div>
"""
m.get_root().html.add_child(folium.Element(stats_html))

# Save
out_file = "map_branches.html"
m.save(out_file)
print(f"\n✅ Saved: {out_file}")
print(f"   Branches plotted: {grand_branches:,}")
print(f"   Trips formed:     {grand_trips:,}")
print(f"   Total route km:   {grand_km:,.0f}")
print("\nOpen map_branches.html in browser to see results.")
