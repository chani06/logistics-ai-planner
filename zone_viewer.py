"""
zone_viewer.py — แผนที่โซนจัดส่งสาขา (Standalone · Deploy-ready)
รันบนเครื่อง : streamlit run zone_viewer.py --server.port 8889
Deploy       : streamlit.io/cloud  →  point to this file
"""
import json, math, io, os
from collections import defaultdict

import pandas as pd
import streamlit as st

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import folium
    from streamlit_folium import folium_static
    FOLIUM_OK = True
except ImportError:
    FOLIUM_OK = False

try:
    import xlsxwriter  # noqa
    XLSX_OK = True
except ImportError:
    XLSX_OK = False

# ─── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="โซนจัดส่งสาขา",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  section[data-testid="stSidebar"] { min-width:260px; max-width:290px; }
  .mbox { text-align:center; padding:8px 4px; border-radius:8px;
          background:#1e2937; color:#fff; margin:2px; }
  .mbox .v { font-size:1.5rem; font-weight:700; }
  .mbox .l { font-size:.72rem; opacity:.75; }
  .badge { display:inline-flex; align-items:center; gap:5px;
           padding:3px 9px; border-radius:20px; font-size:12px;
           font-weight:600; margin:2px 2px; white-space:nowrap; }
  .dot   { width:9px; height:9px; border-radius:50%; flex-shrink:0; }
</style>
""", unsafe_allow_html=True)

# ─── data path ────────────────────────────────────────────────────────────────
_DIR        = os.path.dirname(os.path.abspath(__file__))
BRANCH_JSON = os.path.join(_DIR, "branch_data.json")
_GEO_CACHE  = os.path.join(_DIR, "_thai_amphoe.geojson")  # downloaded once

# ════════════════════════════════════════════════════════════════════════════════
# PROVINCE → ZONE LOOKUP
# ════════════════════════════════════════════════════════════════════════════════
PROVINCE_ZONE: dict = {
    "กรุงเทพมหานคร": "__BKK__", "กรุงเทพฯ": "__BKK__",
    "กทม": "__BKK__", "กทม.": "__BKK__",
    # ปริมณฑล
    "นนทบุรี":"ปริมณฑล_นนทบุรี","ปทุมธานี":"ปริมณฑล_ปทุมธานี",
    "สมุทรปราการ":"ปริมณฑล_สมุทรปราการ","นครปฐม":"ปริมณฑล_นครปฐม",
    "สมุทรสาคร":"ปริมณฑล_สมุทรสาคร","สมุทรสงคราม":"ปริมณฑล_สมุทรสงคราม",
    "พระนครศรีอยุธยา":"ปริมณฑล_อยุธยา","สระบุรี":"ปริมณฑล_สระบุรี",
    "อ่างทอง":"ปริมณฑล_อ่างทอง","สิงห์บุรี":"ปริมณฑล_สิงห์บุรี",
    "ชัยนาท":"ปริมณฑล_ชัยนาท","ลพบุรี":"ปริมณฑล_ลพบุรี",
    # ภาคเหนือ
    "นครสวรรค์":"เหนือ_นครสวรรค์","อุทัยธานี":"เหนือ_อุทัยธานี",
    "กำแพงเพชร":"เหนือ_กำแพงเพชร","ตาก":"เหนือ_ตาก",
    "สุโขทัย":"เหนือ_สุโขทัย","พิษณุโลก":"เหนือ_พิษณุโลก",
    "พิจิตร":"เหนือ_พิจิตร","เพชรบูรณ์":"เหนือ_เพชรบูรณ์",
    "อุตรดิตถ์":"เหนือ_อุตรดิตถ์","แพร่":"เหนือ_แพร่",
    "น่าน":"เหนือ_น่าน","พะเยา":"เหนือ_พะเยา",
    "เชียงราย":"เหนือ_เชียงราย","เชียงใหม่":"เหนือ_เชียงใหม่",
    "ลำพูน":"เหนือ_ลำพูน","ลำปาง":"เหนือ_ลำปาง",
    "แม่ฮ่องสอน":"เหนือ_แม่ฮ่องสอน",
    # ภาคอีสาน
    "หนองบัวลำภู":"อีสาน_หนองบัวลำภู","อุดรธานี":"อีสาน_อุดรธานี",
    "หนองคาย":"อีสาน_หนองคาย","บึงกาฬ":"อีสาน_บึงกาฬ",
    "เลย":"อีสาน_เลย","สกลนคร":"อีสาน_สกลนคร",
    "นครพนม":"อีสาน_นครพนม","มุกดาหาร":"อีสาน_มุกดาหาร",
    "ชัยภูมิ":"อีสาน_ชัยภูมิ","ขอนแก่น":"อีสาน_ขอนแก่น",
    "กาฬสินธุ์":"อีสาน_กาฬสินธุ์","มหาสารคาม":"อีสาน_มหาสารคาม",
    "ร้อยเอ็ด":"อีสาน_ร้อยเอ็ด","นครราชสีมา":"อีสาน_นครราชสีมา",
    "บุรีรัมย์":"อีสาน_บุรีรัมย์","สุรินทร์":"อีสาน_สุรินทร์",
    "ศรีสะเกษ":"อีสาน_ศรีสะเกษ","อุบลราชธานี":"อีสาน_อุบลราชธานี",
    "ยโสธร":"อีสาน_ยโสธร","อำนาจเจริญ":"อีสาน_อำนาจเจริญ",
    # ภาคตะวันออก
    "ฉะเชิงเทรา":"ตะวันออก_ฉะเชิงเทรา","นครนายก":"ตะวันออก_นครนายก",
    "ปราจีนบุรี":"ตะวันออก_ปราจีนบุรี","สระแก้ว":"ตะวันออก_สระแก้ว",
    "ชลบุรี":"ตะวันออก_ชลบุรี","ระยอง":"ตะวันออก_ระยอง",
    "จันทบุรี":"ตะวันออก_จันทบุรี","ตราด":"ตะวันออก_ตราด",
    # ภาคตะวันตก
    "กาญจนบุรี":"ตะวันตก_กาญจนบุรี","ราชบุรี":"ตะวันตก_ราชบุรี",
    "สุพรรณบุรี":"ตะวันตก_สุพรรณบุรี","เพชรบุรี":"ตะวันตก_เพชรบุรี",
    "ประจวบคีรีขันธ์":"ตะวันตก_ประจวบคีรีขันธ์",
    # ภาคใต้
    "ชุมพร":"ใต้_ชุมพร","ระนอง":"ใต้_ระนอง",
    "สุราษฎร์ธานี":"ใต้_สุราษฎร์ธานี","นครศรีธรรมราช":"ใต้_นครศรีธรรมราช",
    "พังงา":"ใต้_พังงา","กระบี่":"ใต้_กระบี่","ภูเก็ต":"ใต้_ภูเก็ต",
    "ตรัง":"ใต้_ตรัง","พัทลุง":"ใต้_พัทลุง","สตูล":"ใต้_สตูล",
    "สงขลา":"ใต้_สงขลา","ปัตตานี":"ใต้_ปัตตานี",
    "ยะลา":"ใต้_ยะลา","นราธิวาส":"ใต้_นราธิวาส",
}

ZONE_REGION: dict = {
    "ปริมณฑล":  "🌆 ปริมณฑล",
    "เหนือ":    "🏔️ เหนือ",
    "อีสาน":    "🌾 อีสาน",
    "ตะวันออก": "🌊 ตะวันออก",
    "ตะวันตก":  "🌳 ตะวันตก",
    "ใต้":      "🏝️ ใต้",
}

# ════════════════════════════════════════════════════════════════════════════════
# BKK — classified by district (เขต) from อำเภอ field
# ════════════════════════════════════════════════════════════════════════════════

# Monochromatic palettes — each REGION gets shades of its own hue family
# so zones in different regions are always easily distinguished by color tone.
_PAL: dict = {
    "🌆 ปริมณฑล":  [  # Blue-gray / slate
        "#37474F","#455A64","#546E7A","#607D8B","#263238",
        "#2C3E48","#324753","#38505E","#3F5A69","#456474",
        "#3B5560","#425D68","#4A6570","#526E78","#5A7780",
    ],
    "🏔️ เหนือ":    [  # Blues
        "#0D47A1","#1565C0","#1976D2","#283593","#303F9F",
        "#003B7A","#0E4F9E","#1258B0","#1862C0","#1E70CC",
        "#0277BD","#01579B","#0061A8","#006BB8","#0075C8",
    ],
    "🌾 อีสาน":    [  # Greens
        "#1B5E20","#2E7D32","#388E3C","#1A3A1F","#1E502C",
        "#245828","#2B6230","#326C38","#387640","#3E8048",
        "#33691E","#3B7223","#427B28","#4A852D","#528F32",
    ],
    "🌊 ตะวันออก": [  # Teals
        "#004D40","#00695C","#00796B","#006064","#00838F",
        "#004248","#004C54","#005760","#00616C","#006B78",
        "#0097A7","#0087A0","#007A94","#006D88","#00607C",
    ],
    "🌳 ตะวันตก":  [  # Purples
        "#4A148C","#6A1B9A","#7B1FA2","#311B92","#4527A0",
        "#380579","#420A85","#4C1091","#56169D","#601CA9",
        "#512DA8","#5B36B0","#6540B8","#6F4AC0","#7954C8",
    ],
    "🏝️ ใต้":      [  # Reds / deep oranges
        "#BF360C","#D84315","#E64A19","#B71C1C","#C62828",
        "#8B1500","#9C2005","#AE2C0A","#C0380F","#D24414",
        "#6D2B00","#7D3300","#8E3B00","#9F4400","#B04C00",
    ],
}

# BKK districts get shades of pink/magenta (distinct from all province regions)
_BKK_PAL = [
    "#880E4F","#AD1457","#C2185B","#D81B60","#6A0032",
    "#790040","#8A0A4E","#9A1458","#AB1E62","#BC286C",
    "#6D004A","#7D0052","#8D0A5A","#9D1462","#AE1E6A",
]

def _lum(h): h=h.lstrip("#"); r,g,b=int(h[:2],16),int(h[2:4],16),int(h[4:],16); return 0.299*r+0.587*g+0.114*b
def _fg(h): return "#FFFFFF" if _lum(h)<140 else "#1A1A1A"

def _norm_dist(s: str) -> str:
    """Strip เขต/อำเภอ prefix so 'เขตพระนคร' and 'พระนคร' both match."""
    s = str(s).strip()
    for pre in ("เขต", "อำเภอ"):
        if s.startswith(pre):
            return s[len(pre):].strip()
    return s

def _norm_prov(s: str) -> str:
    """Strip จังหวัด prefix from GADM NL_NAME_1 (e.g. 'จังหวัดเชียงใหม่' → 'เชียงใหม่')."""
    s = str(s).strip()
    if s.startswith("จังหวัด"):
        return s[6:].strip()
    return s

@st.cache_data(show_spinner="🌏 โหลดขอบเขตอำเภอ…", ttl=86400*7)
def _thai_geo() -> dict | None:
    """Return Thailand district GeoJSON (local cache first, then download)."""
    if os.path.exists(_GEO_CACHE):
        try:
            with open(_GEO_CACHE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    import requests
    for url in [
        "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_THA_2.json",  # GADM level 2 = อำเภอ
    ]:
        try:
            r = requests.get(url, timeout=30)
            if r.ok:
                d = r.json()
                with open(_GEO_CACHE, "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False)
                return d
        except Exception:
            continue
    return None

def _convex_hull(latlons):
    """Return ordered (lat, lon) list forming convex hull. None if < 3 unique pts."""
    pts = list({(round(la,6),round(lo,6)) for la,lo in latlons})
    if len(pts) < 3:
        return None
    try:
        import numpy as np
        from scipy.spatial import ConvexHull
        arr = np.array(pts)
        hull = ConvexHull(arr)
        return [pts[i] for i in hull.vertices]
    except Exception:
        try:  # fallback: angular sort from centroid
            import numpy as np
            arr = np.array(pts)
            cx, cy = arr[:,0].mean(), arr[:,1].mean()
            order = sorted(range(len(pts)), key=lambda i: __import__('math').atan2(pts[i][0]-cx, pts[i][1]-cy))
            return [pts[i] for i in order]
        except Exception:
            return None

# ════════════════════════════════════════════════════════════════════════════════
# LOAD & CLASSIFY
# ════════════════════════════════════════════════════════════════════════════════
def _region_zone_groups(df: pd.DataFrame):
    """Returns {region: set(zone_keys)} excluding BKK and unclassified."""
    rg: dict = defaultdict(set)
    for _,row in df.iterrows():
        z=row["zone"]
        if not z.startswith("BKK_") and not z.startswith("ไม่ระบุ"):
            rg[row["region"]].add(z)
    return rg.items()


def _prov_key(zone: str) -> str:
    """Strip district suffix → province-level key, e.g. เหนือ_เชียงใหม่_เมือง → เหนือ_เชียงใหม่"""
    parts = zone.split("_")
    return "_".join(parts[:2]) if len(parts) >= 3 else zone


@st.cache_data(show_spinner="📂 โหลดข้อมูล…")
def load_and_classify() -> pd.DataFrame:
    if not os.path.exists(BRANCH_JSON):
        return pd.DataFrame()
    with open(BRANCH_JSON, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for code, info in raw.items():
        prov = str(info.get("จังหวัด","") or "").strip()
        for alias,full in [("กรุงเทพฯ","กรุงเทพมหานคร"),("กทม","กรุงเทพมหานคร"),("กทม.","กรุงเทพมหานคร")]:
            if prov==alias: prov=full; break
        try: lat=float(info.get("ละ",0) or 0)
        except: lat=0.
        try: lon=float(info.get("ลอง",0) or 0)
        except: lon=0.

        dist = str(info.get("อำเภอ","") or "").strip()
        rz = PROVINCE_ZONE.get(prov)
        if rz == "__BKK__":
            zone   = f"BKK_{dist}" if dist else "BKK_ไม่ระบุ"
            region = "🏙️ กรุงเทพฯ"
            color  = ""
            label  = f"กทม. เขต{dist}" if dist else "กทม. ไม่ระบุเขต"
            prov_z = "BKK"
        elif rz:
            prefix     = rz.split("_")[0]
            prov_short = rz.split("_",1)[1] if "_" in rz else rz
            region     = ZONE_REGION.get(prefix, prefix)
            prov_z     = rz  # province-level key used for color grouping
            if dist:
                zone  = f"{prefix}_{prov_short}_{dist}"
                label = f"{prov_short} · {dist}"
            else:
                zone  = rz
                label = prov_short
            color = ""
        else:
            zone   = f"ไม่ระบุ_{prov}" if prov else "ไม่ระบุ"
            region = "❓ ไม่ระบุ"
            label  = prov or "ไม่ระบุ"
            color  = "#9E9E9E"
            prov_z = zone

        rows.append(dict(
            code=str(info.get("Plan Code",code)).strip().upper(),
            name=str(info.get("สาขา","")),
            province=prov,
            district=dist,
            lat=lat, lon=lon,
            truck=str(info.get("MaxTruckType","") or ""),
            zone=zone, region=region, label=label, color=color,
            prov_z=prov_z,
        ))

    df = pd.DataFrame(rows)

    # Assign UNIQUE color per district zone for provinces
    for reg, zset in _region_zone_groups(df):
        pal = _PAL.get(reg, ["#37474F","#E65100","#1B5E20","#0D47A1","#880E4F",
                              "#006064","#F57F17","#4A148C","#BF360C","#33691E"]*6)
        for i, z in enumerate(sorted(zset)):  # each district-zone = distinct color
            df.loc[df["zone"]==z, "color"] = pal[i % len(pal)]

    # Assign shades-of-pink per BKK เขต
    bkk_zones_list = sorted(df[df["zone"].str.startswith("BKK_")]["zone"].unique())
    for i, z in enumerate(bkk_zones_list):
        df.loc[df["zone"]==z, "color"] = _BKK_PAL[i % len(_BKK_PAL)]

    # ── Auto-export zone assignments for main app ──────────────────────────────
    _save_branch_zones(df)

    return df


# ════════════════════════════════════════════════════════════════════════════════
# EXPORT — branch_zones.json  (ใช้โดย app.py เพื่อ zone isolation)
# Format: { "BRANCH_CODE": "zone_string" }
# zone_string เช่น "BKK_ลาดพร้าว", "เหนือ_เชียงใหม่_เมือง", "อีสาน_ขอนแก่น_เมือง"
# ════════════════════════════════════════════════════════════════════════════════
_ZONE_EXPORT = os.path.join(_DIR, "branch_zones.json")

def _save_branch_zones(df: pd.DataFrame) -> int:
    """Save {code: zone} mapping to branch_zones.json for main app to consume."""
    try:
        mapping = {
            str(row["code"]).upper(): row["zone"]
            for _, row in df.iterrows()
            if row["code"] and row["zone"]
        }
        with open(_ZONE_EXPORT, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, separators=(",", ":"))
        return len(mapping)
    except Exception:
        return 0def build_map(df: pd.DataFrame):
    m = folium.Map(location=[13.0,101.5], zoom_start=6,
                   tiles="CartoDB positron", control_scale=True)

    # ── Real district GeoJSON boundary layer ────────────────────────────────────
    geo = _thai_geo()
    use_geo = False
    if geo and geo.get("features"):
        # ── Point-in-polygon: map each GADM feature → zone color ──────────────
        # Avoids all Thai name-matching/encoding issues by using branch lat/lon
        try:
            from shapely.geometry import shape, Point
            from shapely.strtree import STRtree

            feats      = geo["features"]
            shp_list   = []
            gid2_list  = []
            for feat in feats:
                try:
                    g = shape(feat["geometry"])
                    if not g.is_valid:
                        g = g.buffer(0)
                    shp_list.append(g)
                    gid2_list.append(feat["properties"].get("GID_2",""))
                except Exception:
                    shp_list.append(None)
                    gid2_list.append("")

            # Build STRtree on valid shapes only
            valid_idx  = [i for i,s in enumerate(shp_list) if s is not None]
            valid_shps = [shp_list[i] for i in valid_idx]
            tree       = STRtree(valid_shps)

            gid2_color: dict = {}
            gid2_label: dict = {}
            for _, row in df.iterrows():
                lat = float(row["lat"] or 0)
                lon = float(row["lon"] or 0)
                c   = str(row.get("color","")).strip()
                if not lat or not lon or not c:
                    continue
                pt = Point(lon, lat)   # GeoJSON uses (lon, lat)
                try:                   # shapely 2.x — 'intersects' finds polygon that contains the point
                    hits = tree.query(pt, predicate="intersects")
                except TypeError:      # shapely 1.x fallback
                    hits = [valid_idx.index(i) for i,s in enumerate(shp_list)
                            if s is not None and s.contains(pt)]
                for hit in hits:
                    gid2 = gid2_list[valid_idx[hit]]
                    if gid2 and gid2 not in gid2_color:
                        gid2_color[gid2] = c
                        gid2_label[gid2] = str(row["label"])

            def _geo_style(feature):
                gid2 = feature.get("properties",{}).get("GID_2","")
                c = gid2_color.get(gid2)
                if c:
                    return {"color": c, "weight": 2.0, "fillColor": c,
                            "fillOpacity": 0.22, "opacity": 0.85}
                return {"color": "#78909C", "weight": 0.5, "fillColor": "#90A4AE",
                        "fillOpacity": 0.04, "opacity": 0.25}

            geo_fg = folium.FeatureGroup(name="📐 ขอบเขตอำเภอ/เขต", show=True)
            folium.GeoJson(geo, style_function=_geo_style, smooth_factor=1.5).add_to(geo_fg)
            geo_fg.add_to(m)
            use_geo = True
        except Exception:
            use_geo = False

    # ── Branch point markers ────────────────────────────────────────────────────
    zone_rows: dict = defaultdict(list)
    for _,row in df.iterrows():
        if row["lat"] and row["lon"] and row["lat"]!=0:
            zone_rows[row["zone"]].append(row)

    for zone, rows in sorted(zone_rows.items()):
        color = rows[0]["color"] or "#9E9E9E"
        label = rows[0]["label"]
        fg = folium.FeatureGroup(
            name=f'<span style="color:{color};font-size:15px">●</span> {label} ({len(rows)})',
            show=True,
        )

        lats, lons = [], []
        for r in rows:
            lats.append(r["lat"]); lons.append(r["lon"])
            tc = _fg(color)
            tip = (f"<b style='color:{color}'>{r['code']}</b><br>{r['name']}<br>"
                   f"<span style='background:{color};color:{tc};padding:1px 6px;"
                   f"border-radius:3px;font-size:11px'>{label}</span>")
            folium.CircleMarker(
                location=[r["lat"],r["lon"]],
                radius=6, color=color, weight=2.5,
                fill=True, fill_color=color, fill_opacity=0.85,
                tooltip=folium.Tooltip(tip, sticky=False),
            ).add_to(fg)

        # Convex-hull fallback only when no real GeoJSON loaded
        if not use_geo and len(rows) >= 3:
            hull = _convex_hull(list(zip(lats, lons)))
            if hull:
                folium.Polygon(
                    locations=hull, color=color, weight=2,
                    fill=True, fill_color=color, fill_opacity=0.10,
                    tooltip=folium.Tooltip(f"{label}  ({len(rows)} สาขา)", sticky=False),
                ).add_to(fg)

        fg.add_to(m)

    folium.LayerControl(collapsed=True).add_to(m)
    return m


# ════════════════════════════════════════════════════════════════════════════════
# EXCEL
# ════════════════════════════════════════════════════════════════════════════════
def build_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        wb = writer.book
        hdr = wb.add_format({"bold":True,"bg_color":"#263238","font_color":"#FFFFFF",
                              "border":1,"align":"center","font_name":"TH Sarabun New"})
        clookup = dict(zip(df["zone"],df["color"]))

        def rfmt(z):
            c = clookup.get(z,"#ECEFF1")
            return wb.add_format({"bg_color":c,"font_color":_fg(c),
                                   "border":1,"font_name":"TH Sarabun New"})

        def write_sheet(sname, data, zone_col=None):
            data.to_excel(writer, sheet_name=sname, index=False)
            ws = writer.sheets[sname]
            for ci,cn in enumerate(data.columns):
                ws.write(0,ci,cn,hdr); ws.set_column(ci,ci,max(18,len(str(cn))+4))
            for ri,tup in enumerate(data.itertuples(index=False),1):
                z = getattr(tup, str(zone_col).replace(" ","_"), None) if zone_col else None
                fmt = rfmt(z)
                for ci,v in enumerate(tup): ws.write(ri,ci,v,fmt)

        # Sheet 1 — sorted by region → zone → branch name
        s1 = df[["code","name","province","district","zone","region","label","lat","lon","truck"]].copy()
        s1.columns=["รหัส","ชื่อสาขา","จังหวัด","อำเภอ","โซน","ภาค","ชื่อโซน","ละ","ลอง","รถ"]
        s1 = s1.sort_values(["ภาค","โซน","ชื่อสาขา"], ignore_index=True)
        write_sheet("สาขาทั้งหมด",s1,"โซน")

        # Sheet 2 summary
        s2 = (df.groupby(["zone","region","label"])["code"].count()
               .reset_index().rename(columns={"code":"จำนวน"})
               .sort_values(["region","zone"]))
        s2.columns=["โซน","ภาค","ชื่อโซน","จำนวน"]
        s2.to_excel(writer, sheet_name="สรุปโซน", index=False)
        ws2 = writer.sheets["สรุปโซน"]
        for ci,cn in enumerate(s2.columns): ws2.write(0,ci,cn,hdr); ws2.set_column(ci,ci,22)
        for ri,tup in enumerate(s2.itertuples(index=False),1):
            fmt=rfmt(tup[0])
            for ci,v in enumerate(tup): ws2.write(ri,ci,v,fmt)

        # BKK — sorted by district (เขต)
        bkk=df[df["region"]=="🏙️ กรุงเทพฯ"][["code","name","district","zone","label"]].copy()
        bkk = bkk.sort_values(["district","name"], ignore_index=True)
        bkk.columns=["รหัส","ชื่อสาขา","เขต","Sub-Zone","ชื่อ Sub-Zone"]
        if not bkk.empty: write_sheet("กรุงเทพ_SubZone",bkk,"Sub-Zone")

        # Per region — sorted by zone → branch name
        for reg in sorted(df["region"].unique()):
            if reg in ["🏙️ กรุงเทพฯ","❓ ไม่ระบุ"]: continue
            rd=df[df["region"]==reg][["code","name","province","district","zone","label","lat","lon"]].copy()
            rd = rd.sort_values(["zone","name"], ignore_index=True)
            rd.columns=["รหัส","ชื่อสาขา","จังหวัด","อำเภอ","โซน","ชื่อโซน","ละ","ลอง"]
            write_sheet(reg.split(" ",1)[-1][:28],rd,"โซน")

    buf.seek(0); return buf.getvalue()


# ════════════════════════════════════════════════════════════════════════════════
# COLOR BADGE HELPER
# ════════════════════════════════════════════════════════════════════════════════
def badge(color: str, text: str, count: int = 0) -> str:
    fg = _fg(color)
    cnt = f'<span style="font-size:10px;opacity:.8"> · {count}</span>' if count else ""
    return (f'<span class="badge" style="background:{color};color:{fg};'
            f'border:1px solid {fg}33">'
            f'<span class="dot" style="background:{fg}66"></span>{text}{cnt}</span>')


# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════
df = load_and_classify()
if df.empty:
    st.error(f"❌ ไม่พบ branch_data.json ที่: {BRANCH_JSON}")
    st.stop()

# ─── sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗺️ โซนจัดส่งสาขา")
    st.divider()

    all_regs = ["ทั้งหมด"] + sorted(df["region"].unique().tolist())
    sel_reg  = st.selectbox("ภาค", all_regs)
    if sel_reg != "ทั้งหมด":
        provs = ["ทั้งหมด"] + sorted(df[df["region"]==sel_reg]["province"].dropna().unique().tolist())
    else:
        provs = ["ทั้งหมด"]
    sel_prov = st.selectbox("จังหวัด", provs)
    if sel_prov != "ทั้งหมด":
        dists = ["ทั้งหมด"] + sorted(
            d for d in df[df["province"]==sel_prov]["district"].dropna().unique() if d
        )
    else:
        dists = ["ทั้งหมด"]
    sel_dist = st.selectbox("อำเภอ", dists)
    hide_unk  = st.toggle("ซ่อนไม่ระบุโซน", value=True)

    st.divider()
    st.markdown("**📥 ดาวน์โหลด**")
    csv_all = df[["code","name","province","zone","region","label","lat","lon","truck"]
               ].rename(columns={"code":"รหัส","name":"ชื่อสาขา","province":"จังหวัด",
                                  "zone":"โซน","region":"ภาค","label":"ชื่อโซน",
                                  "lat":"ละ","lon":"ลอง","truck":"รถ"}
               ).to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ CSV ทั้งหมด", csv_all, "branch_zones.csv", "text/csv",
                       use_container_width=True)
    if XLSX_OK:
        if st.button("🔧 สร้าง Excel", use_container_width=True):
            with st.spinner("กำลังสร้าง Excel…"):
                xb = build_excel(df)
            st.download_button("⬇️ Excel (ระบายสีโซน)", xb, "branch_zones.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

    st.divider()
    st.markdown("**🔗 ส่งโซนให้แอพหลัก**")
    _saved_n = len(json.load(open(_ZONE_EXPORT, encoding="utf-8"))) if os.path.exists(_ZONE_EXPORT) else 0
    if _saved_n:
        st.success(f"✅ branch_zones.json: {_saved_n:,} สาขา")
    if st.button("🔄 Refresh zones → main app", use_container_width=True):
        n = _save_branch_zones(df)
        st.success(f"✅ บันทึก {n:,} สาขา → branch_zones.json")
        st.cache_data.clear()

    # Legend in sidebar — show province level
    st.divider()
    st.markdown("**🎨 สีโซน**")
    bkk_df_leg = df[df["region"]=="🏙️ กรุงเทพฯ"]
    bkk_zone_leg = sorted(bkk_df_leg["zone"].unique())
    st.markdown(f"**🏙️ กรุงเทพฯ** ({len(bkk_df_leg):,} สาขา · {len(bkk_zone_leg)} เขต)")
    for zk in bkk_zone_leg[:8]:
        mask = bkk_df_leg["zone"]==zk
        if not mask.any(): continue
        cnt = int(mask.sum())
        c = bkk_df_leg.loc[mask,"color"].iloc[0] or "#9E9E9E"
        st.markdown(badge(c, zk.replace("BKK_",""), cnt), unsafe_allow_html=True)
    if len(bkk_zone_leg) > 8:
        st.caption(f"…+{len(bkk_zone_leg)-8} เขต (ดูใน tab โซนทั้งหมด)")
    for reg, zset in sorted(_region_zone_groups(df)):
        if not zset: continue
        st.markdown(f"**{reg}**")
        # Group by province-level key, show province badge
        prov_color: dict = {}
        prov_count: dict = defaultdict(int)
        for _, r in df[df["region"]==reg].iterrows():
            pk = _prov_key(r["zone"])
            prov_color[pk] = r["color"]
            prov_short = pk.split("_",1)[-1] if "_" in pk else pk
            prov_count[prov_short] += 1
        shown_provs = sorted(prov_color.keys())[:6]
        for pk in shown_provs:
            c = prov_color.get(pk, "#9E9E9E")
            prov_short = pk.split("_",1)[-1] if "_" in pk else pk
            st.markdown(badge(c, prov_short, prov_count[prov_short]), unsafe_allow_html=True)
        total_provs = len(set(_prov_key(z) for z in zset))
        if total_provs > 6:
            st.caption(f"…+{total_provs-6} จังหวัด (ดูใน tab)")


# ─── apply filters ────────────────────────────────────────────────────────────
view = df.copy()
if sel_reg  != "ทั้งหมด": view = view[view["region"]==sel_reg]
if sel_prov != "ทั้งหมด": view = view[view["province"]==sel_prov]
if sel_dist != "ทั้งหมด": view = view[view["district"]==sel_dist]
if hide_unk:               view = view[view["region"]!="❓ ไม่ระบุ"]

# ─── top metrics ──────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
c1.markdown(f'<div class="mbox"><div class="v">{len(view):,}</div><div class="l">📍 สาขา</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="mbox"><div class="v">{view["zone"].nunique()}</div><div class="l">🗂️ โซน</div></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="mbox"><div class="v">{view[view["region"]=="🏙️ กรุงเทพฯ"]["zone"].nunique()}</div><div class="l">🏙️ BKK sub</div></div>', unsafe_allow_html=True)
c4.markdown(f'<div class="mbox"><div class="v">{int((view["region"]=="❓ ไม่ระบุ").sum())}</div><div class="l">❓ ไม่ระบุ</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── tabs ─────────────────────────────────────────────────────────────────────
tab_map, tab_zones, tab_tbl = st.tabs(["🗺️ แผนที่","🎨 โซนทั้งหมด","📋 ตาราง"])

# ── Map ────────────────────────────────────────────────────────────────────────
with tab_map:
    if not FOLIUM_OK:
        st.warning("⚠️ ต้องติดตั้ง folium และ streamlit-folium")
    else:
        map_src = view if sel_reg!="ทั้งหมด" else df
        if hide_unk: map_src = map_src[map_src["region"]!="❓ ไม่ระบุ"]
        with st.spinner("🔄 กำลังสร้างแผนที่…"):
            fmap = build_map(map_src)
        folium_static(fmap, width=1150, height=680)

# ── All zones ──────────────────────────────────────────────────────────────────
with tab_zones:
    bkk_tab_df = df[df["region"]=="🏙️ กรุงเทพฯ"]
    bkk_tab_zones = sorted(bkk_tab_df["zone"].unique())
    st.markdown(f"### 🏙️ กรุงเทพฯ — {len(bkk_tab_zones)} เขต  ({len(bkk_tab_df):,} สาขา)")
    cols = st.columns(5)
    for i, zk in enumerate(bkk_tab_zones):
        mask = bkk_tab_df["zone"]==zk
        if not mask.any(): continue
        cnt = int(mask.sum())
        c = bkk_tab_df.loc[mask,"color"].iloc[0] or "#9E9E9E"
        fg = _fg(c)
        dist_name = zk.replace("BKK_","")
        cols[i%5].markdown(
            f'<div style="background:{c};color:{fg};padding:8px 10px;'
            f'border-radius:8px;margin:3px 0;font-size:12px">'
            f'<b>เขต{dist_name}</b><br>'
            f'<span style="font-size:1.1rem;font-weight:700">{cnt}</span>'
            f'<span style="font-size:10px"> สาขา</span></div>',
            unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🗂️ โซนจังหวัด (ระดับอำเภอ)")

    for reg, zset in sorted(_region_zone_groups(df)):
        # Group districts by province
        prov_districts: dict = defaultdict(list)  # prov_key → [zone,...]
        prov_color_map: dict = {}
        for z in zset:
            pk = _prov_key(z)
            prov_districts[pk].append(z)
            # get color from df
            mask = df["zone"]==z
            if mask.any():
                prov_color_map[pk] = df.loc[mask, "color"].iloc[0]

        n_prov = len(prov_districts)
        n_dist = len(zset)
        total_reg = int((df["region"]==reg).sum())
        with st.expander(f"{reg}  ·  {total_reg:,} สาขา  ·  {n_prov} จังหวัด  ·  {n_dist} อำเภอ"):
            for pk in sorted(prov_districts.keys()):
                prov_name = pk.split("_",1)[-1] if "_" in pk else pk
                base_c = prov_color_map.get(pk, "#9E9E9E")
                fg_base = _fg(base_c)
                prov_cnt = int(df[df["prov_z"]==pk]["code"].count())
                st.markdown(
                    f'<div style="background:{base_c};color:{fg_base};'
                    f'padding:5px 10px;border-radius:6px;margin:6px 0 2px;'
                    f'font-size:13px;font-weight:700">'
                    f'📍 {prov_name}  <span style="font-weight:400;font-size:11px">({prov_cnt} สาขา)</span></div>',
                    unsafe_allow_html=True)
                dcols = st.columns(4)
                for j,z in enumerate(sorted(prov_districts[pk])):
                    mask = df["zone"]==z
                    if not mask.any(): continue
                    cnt = int(mask.sum())
                    lbl = df.loc[mask, "label"].iloc[0]
                    dist_name = lbl.split(" · ")[-1] if " · " in lbl else lbl
                    dist_c = df.loc[mask, "color"].iloc[0] or base_c
                    dcols[j%4].markdown(
                        f'<div style="background:{dist_c}28;color:inherit;'
                        f'border-left:4px solid {dist_c};padding:5px 8px;'
                        f'border-radius:4px;margin:2px 0;font-size:12px">'
                        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{dist_c};margin-right:4px"></span>'
                        f'{dist_name}<br><span style="font-size:11px;opacity:.7">{cnt} สาขา</span></div>',
                        unsafe_allow_html=True)

# ── Table ──────────────────────────────────────────────────────────────────────
with tab_tbl:
    tbl = view[["code","name","province","district","zone","region","label","lat","lon","truck"]].copy()
    tbl.columns=["รหัส","ชื่อสาขา","จังหวัด","อำเภอ","โซน","ภาค","ชื่อโซน","ละ","ลอง","รถ"]
    st.caption(f"แสดง {len(tbl):,} สาขา")
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=500)
    st.download_button(
        "⬇️ CSV (ตามตัวกรอง)", tbl.to_csv(index=False).encode("utf-8-sig"),
        "filtered_zones.csv","text/csv",
    )
