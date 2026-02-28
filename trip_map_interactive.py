"""
trip_map_interactive.py
=======================
สร้าง interactive HTML map สำหรับจัดการ trip สาขา
- Leaflet.js (ไม่โหลด API ซ้ำ, tile cached โดย browser)
- คลิกสาขา → ย้าย trip ผ่าน popup
- คำนวณ Weight/Cube real-time
- เลือกประเภทรถต่อ trip
- เตือนถ้าเกิน limit แต่ยืนยันได้
- Export Excel (SheetJS)
ทุกอย่างทำใน JS ไม่ต้อง roundtrip Python
"""

import json
import pathlib
import pandas as pd
import io

# ─────────────────────────────────────────────────────
# Load Leaflet assets from local static/ dir (avoids Streamlit iframe CDN block)
# Falls back to jsDelivr CDN if local files not found.
_STATIC = pathlib.Path(__file__).parent / 'static'

def _read_local(fname: str) -> str:
    p = _STATIC / fname
    return p.read_text(encoding='utf-8') if p.exists() else ''

_LEAFLET_CSS_CONTENT = _read_local('leaflet.css')
_LEAFLET_JS_CONTENT  = _read_local('leaflet.js')

# ─────────────────────────────────────────────────────
VEHICLE_LIMITS = {
    "4W": {"max_w": 2500, "max_c": 5.0,  "max_drops": 12},
    "JB": {"max_w": 3500, "max_c": 7.0,  "max_drops": 12},
    "6W": {"max_w": 6000, "max_c": 20.0, "max_drops": 999},
}
PUNTHAI_VEHICLE_LIMITS = {
    "4W": {"max_w": 2500, "max_c": 5.0,  "max_drops": 5},
    "JB": {"max_w": 3500, "max_c": 7.0,  "max_drops": 7},
    "6W": {"max_w": 6000, "max_c": 20.0, "max_drops": 999},
}

TRIP_COLORS = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628",
    "#f781bf","#1b9e77","#d95f02","#7570b3","#e7298a","#66a61e",
    "#e6ab02","#a6761d","#333333","#1f78b4","#33a02c","#fb9a99",
    "#fdbf6f","#cab2d6","#b15928","#8dd3c7","#bebada","#fb8072",
    "#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd",
    "#ccebc5","#ffed6f","#8dd3c7","#ffffb3","#bebada","#fb8072",
    "#80b1d3","#fdb462","#b3de69","#fccde5","#d9d9d9","#bc80bd",
]


def _safe(v, default=""):
    if v is None:
        return default
    if isinstance(v, float) and pd.isna(v):
        return default
    try:
        if pd.isna(v):   # catches pd.NA, pd.NaT
            return default
    except (TypeError, ValueError):
        pass
    s = str(v)
    if s in ("None", "nan", "NaN", "NaT", "<NA>", "NA"):
        return default
    return v


def build_interactive_map_html(
    result_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    limits: dict | None = None,
    punthai_limits: dict | None = None,
    punthai_buffer: float = 1.0,
    maxmart_buffer: float = 1.1,
    trip_no_map: dict | None = None,
    dc_lat: float = 14.1459,
    dc_lon: float = 100.6873,
) -> str:

    lim   = limits         or VEHICLE_LIMITS
    plim  = punthai_limits or PUNTHAI_VEHICLE_LIMITS
    tno_map = trip_no_map or {}

    # Build inline or CDN tags for Leaflet
    if _LEAFLET_CSS_CONTENT:
        _leaflet_css_tag = f'<style>\n{_LEAFLET_CSS_CONTENT}\n</style>'
    else:
        _leaflet_css_tag = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" crossorigin="">'
    if _LEAFLET_JS_CONTENT:
        _leaflet_js_tag = f'<script>\n{_LEAFLET_JS_CONTENT}\n</script>'
    else:
        _leaflet_js_tag = '<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>'

    # ──────────────────────────────────────────────────
    # 1.  Prepare branch rows (only assigned + has coords)
    # ──────────────────────────────────────────────────
    lat_col = "_lat" if "_lat" in result_df.columns else "Latitude"
    lon_col = "_lon" if "_lon" in result_df.columns else "Longitude"

    df = result_df.copy()
    df = df[df["Trip"] > 0].copy()
    if lat_col in df.columns and lon_col in df.columns:
        df = df[(df[lat_col].fillna(0) > 0) & (df[lon_col].fillna(0) > 0)].copy()
    else:
        return "<p style='color:red'>ไม่พบคอลัมน์พิกัด (_lat/_lon) ในผลลัพธ์</p>"

    def _col(c, default=""):
        return df[c].fillna(default).tolist() if c in df.columns else [default] * len(df)

    branches_json = []
    for _, row in df.iterrows():
        bu = str(_safe(row.get("BU", ""), "")).strip().upper()
        is_pt = bu in ("211", "PUNTHAI")
        truck_raw = str(_safe(row.get("Truck", "6W"), "6W")).strip()
        vtype = truck_raw.split()[0] if truck_raw else "6W"
        if vtype == "4WJ":
            vtype = "JB"
        mv = str(_safe(row.get("MaxVehicle", "6W"), "6W")).strip()

        branches_json.append({
            "code":     str(_safe(row.get("Code", ""))),
            "name":     str(_safe(row.get("Name", ""))),
            "province": str(_safe(row.get("Province", ""))),
            "district": str(_safe(row.get("District", row.get("_district", "")))),
            "lat":      float(row[lat_col]),
            "lon":      float(row[lon_col]),
            "trip":     int(_safe(row.get("Trip", 0), 0)),
            "weight":   float(_safe(row.get("Weight", 0), 0)),
            "cube":     float(_safe(row.get("Cube", 0), 0)),
            "vtype":    vtype,
            "maxVeh":   mv,
            "isPunthai": is_pt,
            "bu":        bu,
            "vc":       str(_safe(row.get("VehicleCheck", ""), "")),
        })

    # ──────────────────────────────────────────────────
    # 2.  Trip info from summary
    # ──────────────────────────────────────────────────
    trips_json = {}
    if summary_df is not None and not summary_df.empty:
        for _, row in summary_df.iterrows():
            t = int(_safe(row.get("Trip", 0), 0))
            if t == 0:
                continue
            truck_raw = str(_safe(row.get("Truck", "6W"), "6W")).strip()
            vt = truck_raw.split()[0] if truck_raw else "6W"
            if vt == "4WJ":
                vt = "JB"
            trips_json[str(t)] = {
                "trip":     t,
                "truck":    vt,
                "tripNo":   tno_map.get(t, f"T{t:03d}"),
                "branches": int(_safe(row.get("Branches", 0), 0)),
            }

    # Fill any missing trips
    for b in branches_json:
        tid = str(b["trip"])
        if tid not in trips_json:
            trips_json[tid] = {
                "trip": b["trip"], "truck": b["vtype"],
                "tripNo": tno_map.get(b["trip"], f"T{b['trip']:03d}"),
                "branches": 0,
            }

    lim_json  = json.dumps(lim,  ensure_ascii=False)
    plim_json = json.dumps(plim, ensure_ascii=False)

    branches_js  = json.dumps(branches_json,          ensure_ascii=False)
    trips_js     = json.dumps(trips_json,              ensure_ascii=False)
    colors_js    = json.dumps(TRIP_COLORS,             ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<title>Interactive Trip Map</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
{_leaflet_css_tag}
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  html,body{{width:100%;height:780px;margin:0;padding:0;overflow:hidden;font-family:'Segoe UI',sans-serif;font-size:13px;background:#f4f6f9}}
  #app{{display:flex;width:100%;height:780px;position:relative}}
  #sidebar{{width:340px;min-width:270px;max-width:400px;height:780px;background:#fff;
            border-right:1px solid #ddd;display:flex;flex-direction:column;
            resize:horizontal;overflow:hidden}}
  #sidebar-header{{padding:10px 12px;background:#1976d2;color:#fff;font-weight:700;font-size:14px;
                   display:flex;align-items:center;justify-content:space-between;flex-shrink:0}}
  #toolbar{{display:flex;gap:6px;padding:8px 10px;background:#f8f9fa;border-bottom:1px solid #e0e0e0;flex-shrink:0;flex-wrap:wrap}}
  #trip-list{{flex:1;overflow-y:auto;padding:6px}}
  .trip-card{{background:#fff;border:1px solid #e0e0e0;border-radius:8px;margin-bottom:6px;
              overflow:hidden;transition:box-shadow .15s}}
  .trip-card:hover{{box-shadow:0 2px 8px rgba(0,0,0,.12)}}
  .trip-header{{padding:8px 10px;cursor:pointer;display:flex;align-items:center;gap:8px;
                 user-select:none}}
  .trip-dot{{width:14px;height:14px;border-radius:50%;flex-shrink:0}}
  .trip-title{{font-weight:600;flex:1}}
  .trip-stats{{font-size:11px;color:#555;line-height:1.4}}
  .stat-bar{{height:6px;border-radius:3px;background:#e0e0e0;margin-top:2px;overflow:hidden}}
  .stat-bar-fill{{height:100%;border-radius:3px;transition:width .3s}}
  .trip-body{{padding:0 10px 8px;display:none}}
  .trip-body.open{{display:block}}
  .veh-sel{{padding:4px 6px;border:1px solid #ccc;border-radius:4px;font-size:12px;
             background:#fff;width:100%;margin-top:6px;cursor:pointer}}
  .branch-mini{{font-size:11px;color:#444;padding:2px 0;border-bottom:1px solid #f0f0f0;
                display:flex;justify-content:space-between}}
  .over-limit{{background:#fff3e0;border-color:#ff9800}}
  .over-error{{background:#ffebee;border-color:#f44336}}
  #map{{flex:1;height:780px;min-height:0;min-width:0}}
  #info-panel{{position:absolute;bottom:0;left:340px;right:0;background:rgba(255,255,255,.95);
               border-top:2px solid #1976d2;padding:10px 14px;z-index:1000;
               display:none;max-height:220px;overflow-y:auto}}
  .info-title{{font-weight:700;font-size:14px;color:#1976d2}}
  .move-row{{display:flex;align-items:center;gap:8px;margin-top:8px;flex-wrap:wrap}}
  .move-sel{{flex:1;min-width:150px;padding:5px 8px;border:1px solid #ccc;
              border-radius:4px;font-size:12px}}
  .btn-sm-custom{{padding:5px 12px;border:none;border-radius:4px;cursor:pointer;
                  font-size:12px;font-weight:600}}
  .btn-primary2{{background:#1976d2;color:#fff}}
  .btn-primary2:hover{{background:#1565c0}}
  .btn-danger2{{background:#d32f2f;color:#fff}}
  .btn-danger2:hover{{background:#b71c1c}}
  .btn-success2{{background:#388e3c;color:#fff}}
  .btn-success2:hover{{background:#2e7d32}}
  .btn-warn{{background:#f57c00;color:#fff}}
  .btn-warn:hover{{background:#e65100}}
  .badge-veh{{display:inline-block;padding:2px 7px;border-radius:10px;font-size:10px;font-weight:700}}
  .badge-4W{{background:#e3f2fd;color:#1976d2}}
  .badge-JB{{background:#f3e5f5;color:#7b1fa2}}
  .badge-6W{{background:#e8f5e9;color:#388e3c}}
  #toast{{position:fixed;top:10px;left:50%;transform:translateX(-50%);
          background:#323232;color:#fff;padding:8px 20px;border-radius:20px;
          font-size:12px;z-index:9999;display:none;pointer-events:none}}
  .warn-badge{{background:#ff9800;color:#fff;border-radius:10px;padding:1px 6px;font-size:10px}}
  .err-badge{{background:#f44336;color:#fff;border-radius:10px;padding:1px 6px;font-size:10px}}
  ::-webkit-scrollbar{{width:5px}}
  ::-webkit-scrollbar-thumb{{background:#ccc;border-radius:3px}}
</style>
</head>
<body>
<div id="app">
 <div id="sidebar">
  <div id="sidebar-header">
    🚚 จัดการทริป
    <div style="display:flex;gap:6px">
      <span id="total-badge" style="background:rgba(255,255,255,.25);padding:2px 8px;
            border-radius:10px;font-size:11px;font-weight:400"></span>
    </div>
  </div>
  <div id="toolbar">
    <button class="btn-sm-custom btn-success2" onclick="exportExcel()">📥 Export Excel</button>
    <button class="btn-sm-custom" style="background:#546e7a;color:#fff" onclick="toggleAllTrips()">📋 ขยาย/ย่อ</button>
    <button class="btn-sm-custom" style="background:#7b1fa2;color:#fff" onclick="fitAll()">🔍 Fit All</button>
    <button class="btn-sm-custom" id="route-btn" style="background:#0288d1;color:#fff" onclick="toggleRoutes()">🛣️ เส้นทาง</button>
    <input id="search-inp" placeholder="🔍 ค้นหาสาขา..." style="flex:1;min-width:80px;
           padding:4px 8px;border:1px solid #ccc;border-radius:4px;font-size:12px">
  </div>
  <div id="trip-list"></div>
 </div>
 <div id="map"></div>
 <div id="info-panel">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
   <div>
    <div class="info-title" id="info-code">-</div>
    <div id="info-detail" style="font-size:12px;color:#555;margin-top:2px"></div>
   </div>
   <button onclick="closeInfo()" style="background:none;border:none;font-size:18px;cursor:pointer;color:#888">✕</button>
  </div>
  <div class="move-row">
    <span style="font-size:12px;font-weight:600">ย้ายไป Trip:</span>
    <select class="move-sel" id="move-sel"></select>
    <button class="btn-sm-custom btn-primary2" onclick="confirmMove()">✅ ย้าย</button>
    <button class="btn-sm-custom btn-danger2"  onclick="closeInfo()">ยกเลิก</button>
  </div>
  <div id="move-warn" style="font-size:11px;color:#f57c00;margin-top:4px"></div>
 </div>
</div>

<div id="toast"></div>

<script>
// JS error display — shows any error inside the map iframe
window.onerror = function(msg, src, line, col, err) {{
  document.getElementById('map').innerHTML =
    '<pre style="color:red;background:#fff;padding:12px;font-size:12px">JS Error: ' +
    msg + '\nLine: ' + line + '\n' + (err ? err.stack : '') + '</pre>';
  return true;
}};
</script>
{_leaflet_js_tag}
<script>
// xlsx โหลดเฉพาะตอน export (lazy)
let _xlsxReady=false;
function _loadXlsx(cb){{
  if(_xlsxReady){{cb();return;}}
  const s=document.createElement('script');
  s.src='https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js';
  s.onload=()=>{{_xlsxReady=true;cb();}};
  s.onerror=()=>alert('โหลด xlsx ไม่ได้ ตรวจสอบเน็ต');
  document.head.appendChild(s);
}}
</script>
<script>
// ═══════════════════════════════════════════════════
// DATA (embedded from Python)
// ═══════════════════════════════════════════════════
const RAW_BRANCHES = {branches_js};
const RAW_TRIPS    = {trips_js};
const COLORS       = {colors_js};
const LIMITS       = {lim_json};
const PLIMITS      = {plim_json};
const PUNTHAI_BUF  = {punthai_buffer};
const MAXMART_BUF  = {maxmart_buffer};
const DC           = [{dc_lat}, {dc_lon}];

// ═══════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════
let branches  = JSON.parse(JSON.stringify(RAW_BRANCHES));  // deep copy
let trips     = JSON.parse(JSON.stringify(RAW_TRIPS));
let markers   = {{}};   // code -> leaflet marker
let selectedCode = null;
let _allOpen = false;

// ── perf cache ──────────────────────────────────
let _cachedTripKeys = null;
let _summaryCache   = {{}};
function _invalidateCache(){{ _cachedTripKeys=null; _summaryCache={{}}; }}

// ═══════════════════════════════════════════════════
// MAP INIT
// ═══════════════════════════════════════════════════
const _renderer = L.canvas({{padding:0.5}});
const map = L.map('map', {{zoomControl:true,preferCanvas:true,renderer:_renderer}})
  .setView(DC, 6);

L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom:18, attribution:'© OpenStreetMap',
  crossOrigin:true
}}).addTo(map);

// DC marker
L.marker(DC, {{icon: L.divIcon({{
  html:'<div style="background:#1976d2;color:#fff;border-radius:50%;width:28px;height:28px;' +
       'display:flex;align-items:center;justify-content:center;font-size:13px;' +
       'box-shadow:0 2px 6px rgba(0,0,0,.4)">🏭</div>',
  iconSize:[28,28], iconAnchor:[14,14], className:''
}})}})).addTo(map).bindPopup('<b>DC Wang Noi</b>');

// ═══════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════
function tripColor(tripId) {{
  const keys = sortedTripKeys();
  const idx  = keys.indexOf(String(tripId));
  return COLORS[idx % COLORS.length] || '#888';
}}

function sortedTripKeys() {{
  if(!_cachedTripKeys) _cachedTripKeys=Object.keys(trips).sort((a,b)=>parseInt(a)-parseInt(b));
  return _cachedTripKeys;
}}

function getLimit(vtype, isPunthai) {{
  const L = isPunthai ? PLIMITS : LIMITS;
  return L[vtype] || L['6W'];
}}

function tripSummary(tripId) {{
  const tid = String(tripId);
  if(_summaryCache[tid]) return _summaryCache[tid];
  const brs = branches.filter(b => String(b.trip) === tid);
  const w = brs.reduce((s,b) => s+b.weight, 0);
  const c = brs.reduce((s,b) => s+b.cube,   0);
  const drops = brs.length;
  const isPt  = brs.length > 0 && brs.every(b => b.isPunthai);
  const vtype = trips[tid] ? trips[tid].truck : '6W';
  const buf   = isPt ? PUNTHAI_BUF : MAXMART_BUF;
  const lim   = getLimit(vtype, isPt);
  const maxW  = lim.max_w * buf;
  const maxC  = lim.max_c * buf;
  const maxD  = lim.max_drops;
  const res = {{w, c, drops, maxW, maxC, maxD, isPt, vtype,
           wPct: maxW>0?w/maxW*100:0, cPct: maxC>0?c/maxC*100:0,
           overW: w>maxW, overC: c>maxC, overD: drops>maxD}};
  _summaryCache[tid] = res;
  return res;
}}

function vehBadge(v) {{
  return `<span class="badge-veh badge-${{v}}">${{v}}</span>`;
}}

// ═══════════════════════════════════════════════════
// RENDER SIDEBAR
// ═══════════════════════════════════════════════════
function renderSidebar() {{
  const list = document.getElementById('trip-list');
  const tkeys = sortedTripKeys();
  let html = '';
  let totalBranches = branches.length;
  let warnTrips=0, errTrips=0;

  for (const tid of tkeys) {{
    const t    = trips[tid];
    const brs  = branches.filter(b => String(b.trip) === tid);
    const s    = tripSummary(tid);
    const col  = tripColor(parseInt(tid));
    const over = s.overW || s.overC || s.overD;
    const cardClass = over ? (s.overW&&s.wPct>110||s.overC&&s.cPct>110 ? 'over-error':'over-limit') : '';
    if(s.overW||s.overC||s.overD) {{
      if(s.wPct>110||s.cPct>110) errTrips++; else warnTrips++;
    }}

    const vtyp = t.truck || '6W';
    const vOpts = ['4W','JB','6W'].map(v =>
      `<option value="${{v}}" ${{v===vtyp?'selected':''}}>${{v}}</option>`
    ).join('');

    const brHtml = brs.slice(0,8).map(b =>
      `<div class="branch-mini"><span>${{b.code}} ${{b.name.slice(0,14)}}</span>
       <span style="color:#888">${{b.weight.toFixed(1)}}kg/${{b.cube.toFixed(2)}}㎥</span></div>`
    ).join('') + (brs.length>8 ? `<div style="font-size:10px;color:#999;padding-top:2px">...+${{brs.length-8}} สาขา</div>` : '');

    html += `
    <div class="trip-card ${{cardClass}}" id="tc-${{tid}}">
      <div class="trip-header" onclick="toggleTrip('${{tid}}')">
        <div class="trip-dot" style="background:${{col}}"></div>
        <div>
          <div class="trip-title">Trip ${{tid}} &nbsp;${{vehBadge(vtyp)}}
            ${{s.overW||s.overC||s.overD ? '<span class="err-badge">⚠</span>':''}}
          </div>
          <div class="trip-stats">
            ${{brs.length}} สาขา · W:${{s.w.toFixed(0)}}/${{s.maxW.toFixed(0)}}kg · C:${{s.c.toFixed(2)}}/${{s.maxC.toFixed(2)}}㎥
          </div>
          <div class="stat-bar"><div class="stat-bar-fill"
            style="width:${{Math.min(s.wPct,100).toFixed(1)}}%;
                   background:${{s.wPct>100?'#f44336':s.wPct>85?'#ff9800':'#4caf50'}}"></div></div>
          <div class="stat-bar"><div class="stat-bar-fill"
            style="width:${{Math.min(s.cPct,100).toFixed(1)}}%;
                   background:${{s.cPct>100?'#f44336':s.cPct>85?'#ff9800':'#2196f3'}}"></div></div>
        </div>
      </div>
      <div class="trip-body" id="tb-${{tid}}">
        <select class="veh-sel" onchange="changeVehicle('${{tid}}',this.value)">
          <option value="" disabled>-- เลือกประเภทรถ --</option>
          ${{vOpts}}
        </select>
        <div style="margin-top:6px">${{brHtml}}</div>
        <button class="btn-sm-custom btn-primary2" style="margin-top:6px;width:100%"
          onclick="focusTrip(${{parseInt(tid)}})">🗺️ Zoom Trip</button>
      </div>
    </div>`;
  }}
  list.innerHTML = html;

  const badge = document.getElementById('total-badge');
  badge.textContent = `${{totalBranches}} สาขา · ${{tkeys.length}} ทริป`
    + (errTrips ? ` · ❌${{errTrips}}` : '') + (warnTrips ? ` · ⚠${{warnTrips}}` : '');
}}

function toggleTrip(tid) {{
  const body = document.getElementById('tb-'+tid);
  if(!body) return;
  body.classList.toggle('open');
}}

function toggleAllTrips() {{
  _allOpen = !_allOpen;
  sortedTripKeys().forEach(tid=>{{
    const body = document.getElementById('tb-'+tid);
    if(body) body.classList.toggle('open', _allOpen);
  }});
}}

// ═══════════════════════════════════════════════════
// MARKERS  (circleMarker = canvas, ไม่สร้าง DOM element)
// ═══════════════════════════════════════════════════
let _showRoutes = false;

function markerStyle(b, selected=false) {{
  const col  = tripColor(b.trip);
  const s    = tripSummary(String(b.trip));
  const over = s.overW||s.overC||s.overD;
  return {{
    radius:      selected ? 11 : 7,
    fillColor:   col,
    color:       selected ? '#fff' : over ? '#ff5722' : '#333',
    weight:      selected ? 3 : over ? 2 : 1,
    opacity:     1,
    fillOpacity: selected ? 1.0 : 0.82,
    renderer:    _renderer,
  }};
}}

function renderMarkers() {{
  Object.values(markers).forEach(m => map.removeLayer(m));
  markers = {{}};
  _clearRoutes();
  if(_showRoutes) _drawRoutes();

  for(const b of branches) {{
    const mk = L.circleMarker([b.lat, b.lon], markerStyle(b));
    mk.bindTooltip(`T${{b.trip}} · ${{b.code||''}} ${{(b.name||'').slice(0,14)}}`, {{direction:'top',offset:[0,-4]}});
    mk.on('click', () => showInfo(b.code));
    mk.addTo(map);
    markers[b.code] = mk;
  }}
}}

function _clearRoutes() {{
  if(window._routeLines) window._routeLines.forEach(l=>map.removeLayer(l));
  window._routeLines = [];
}}
function _drawRoutes() {{
  window._routeLines = [];
  for(const tid of sortedTripKeys()) {{
    const brs = branches.filter(b=>String(b.trip)===tid);
    if(brs.length < 2) continue;
    const col = tripColor(parseInt(tid));
    const line = L.polyline([DC,...brs.map(b=>[b.lat,b.lon]),DC],
      {{color:col,weight:2,opacity:0.5,dashArray:'5,4',renderer:_renderer}});
    line.addTo(map);
    window._routeLines.push(line);
  }}
}}

function refreshMarker(code) {{
  const b = branches.find(x=>x.code===code);
  if(!b || !markers[code]) return;
  markers[code].setStyle(markerStyle(b, code===selectedCode));
}}

function refreshTripMarkers(tripId) {{
  branches.filter(b=>String(b.trip)===String(tripId)).forEach(b=>refreshMarker(b.code));
}}

// ═══════════════════════════════════════════════════
// INFO PANEL
// ═══════════════════════════════════════════════════
function showInfo(code) {{
  selectedCode = code;
  const b   = branches.find(x=>x.code===code);
  if(!b) return;

  document.getElementById('info-code').textContent = `${{b.code}} — ${{b.name}}`;
  document.getElementById('info-detail').textContent =
    `${{b.district}} ${{b.province}} · Trip ${{b.trip}} · W:${{b.weight.toFixed(1)}}kg C:${{b.cube.toFixed(2)}}㎥ · MaxVeh:${{b.maxVeh}}`;

  // Build trip selector
  const sel = document.getElementById('move-sel');
  sel.innerHTML = '';
  for(const tid of sortedTripKeys()) {{
    const t  = trips[tid];
    const s  = tripSummary(tid);
    const opt = document.createElement('option');
    opt.value = tid;
    const warn = (s.overW||s.overC||s.overD) ? ' ⚠' : '';
    opt.textContent = `Trip ${{tid}} (${{t.truck||'6W'}}) ${{s.drops}}สาขา W${{s.w.toFixed(0)}}/${{s.maxW.toFixed(0)}}${{warn}}`;
    if(String(b.trip)===tid) opt.selected=true;
    sel.appendChild(opt);
  }};
  // Add new trip option
  const newOpt = document.createElement('option');
  const nextTripId = Math.max(0,...sortedTripKeys().map(Number))+1;
  newOpt.value = 'NEW';
  newOpt.textContent = `+ สร้าง Trip ใหม่ (Trip ${{nextTripId}})`;
  sel.appendChild(newOpt);

  document.getElementById('move-warn').textContent = '';
  document.getElementById('info-panel').style.display='block';
  refreshMarker(code);
  sel.onchange = () => previewMove();
  previewMove();
  map.panTo([b.lat, b.lon]);
}}

function previewMove() {{
  const b   = branches.find(x=>x.code===selectedCode);
  if(!b) return;
  const sel = document.getElementById('move-sel');
  const tid = sel.value;
  if(tid==='NEW' || tid===String(b.trip)) {{
    document.getElementById('move-warn').textContent='';
    return;
  }}
  // Check if target trip would overflow after adding this branch
  const s  = tripSummary(tid);
  const nw = s.w + b.weight;
  const nc = s.c + b.cube;
  const nd = s.drops + 1;
  const msgs = [];
  if(nw > s.maxW) msgs.push(`⚠️ น้ำหนักเกิน: ${{nw.toFixed(0)}}/${{s.maxW.toFixed(0)}}kg`);
  if(nc > s.maxC) msgs.push(`⚠️ คิวเกิน: ${{nc.toFixed(2)}}/${{s.maxC.toFixed(2)}}㎥`);
  if(nd > s.maxD) msgs.push(`⚠️ Drops เกิน: ${{nd}}/${{s.maxD}}`);
  document.getElementById('move-warn').textContent = msgs.join('  ');
}}

function confirmMove() {{
  const b   = branches.find(x=>x.code===selectedCode);
  if(!b) return;
  const sel = document.getElementById('move-sel');
  let targetTid = sel.value;

  if(targetTid==='NEW') {{
    const nextId = Math.max(0,...sortedTripKeys().map(Number))+1;
    targetTid = String(nextId);
    trips[targetTid] = {{trip:nextId, truck:b.vtype, tripNo:`T${{nextId.toString().padStart(3,'0')}}`, branches:0}};
  }}

  const warnMsg = document.getElementById('move-warn').textContent;
  if(warnMsg && !window._overrideConfirmed) {{
    if(!confirm(`${{warnMsg}}\\n\\nต้องการย้ายต่อไปหรือไม่?`)) return;
    window._overrideConfirmed = true;
  }}
  window._overrideConfirmed = false;

  const oldTrip = String(b.trip);
  b.trip = parseInt(targetTid);
  _invalidateCache();

  // Refresh markers
  refreshTripMarkers(oldTrip);
  refreshTripMarkers(targetTid);
  refreshMarker(b.code);

  renderSidebar();
  closeInfo();
  showToast(`✅ ย้าย ${{b.code}} → Trip ${{targetTid}} แล้ว`);
}}

function closeInfo() {{
  document.getElementById('info-panel').style.display='none';
  if(selectedCode) {{ refreshMarker(selectedCode); selectedCode=null; }}
}}

// ═══════════════════════════════════════════════════
// VEHICLE CHANGE
// ═══════════════════════════════════════════════════
function changeVehicle(tid, vtype) {{
  if(trips[tid]) trips[tid].truck = vtype;
  branches.filter(b=>String(b.trip)===tid).forEach(b=>b.vtype=vtype);
  _invalidateCache();
  renderSidebar();
  refreshTripMarkers(tid);
  showToast(`🚚 Trip ${{tid}} → ${{vtype}}`);
}}

// ═══════════════════════════════════════════════════
// FOCUS / FIT
// ═══════════════════════════════════════════════════
function focusTrip(tripId) {{
  const brs = branches.filter(b=>b.trip===tripId);
  if(!brs.length) return;
  const bounds = L.latLngBounds(brs.map(b=>[b.lat,b.lon]));
  map.fitBounds(bounds, {{padding:[30,30]}});
}}

function fitAll() {{
  if(!branches.length) return;
  const bounds = L.latLngBounds(branches.map(b=>[b.lat,b.lon]));
  map.fitBounds(bounds, {{padding:[20,20]}});
}}

function toggleRoutes() {{
  _showRoutes = !_showRoutes;
  const btn = document.getElementById('route-btn');
  if(_showRoutes) {{
    btn.style.background='#01579b';
    _drawRoutes();
  }} else {{
    btn.style.background='#0288d1';
    _clearRoutes();
  }}
}}

// ═══════════════════════════════════════════════════
// SEARCH
// ═══════════════════════════════════════════════════
document.getElementById('search-inp').addEventListener('input', function() {{
  const q = this.value.trim().toLowerCase();
  if(!q) {{ renderMarkers(); return; }}
  const found = branches.find(b =>
    (b.code||'').toLowerCase().includes(q) || (b.name||'').toLowerCase().includes(q));
  if(found) {{
    map.setView([found.lat, found.lon], 14);
    showInfo(found.code);
  }}
}});

// ═══════════════════════════════════════════════════
// TOAST
// ═══════════════════════════════════════════════════
let _toastTimer;
function showToast(msg) {{
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.style.display='block';
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(()=>t.style.display='none', 2500);
}}

// ═══════════════════════════════════════════════════
// EXPORT EXCEL
// ═══════════════════════════════════════════════════
function exportExcel() {{
  _loadXlsx(()=>_doExport());
}}
function _doExport() {{
  const rows = [['Trip','Trip No','ประเภทรถ','รหัสสาขา','ชื่อสาขา','จังหวัด','อำเภอ','น้ำหนัก(kg)','คิว(m³)','MaxVehicle','BU','หมายเหตุ']];

  // Sort by trip then code
  const sorted = [...branches].sort((a,b) => a.trip-b.trip || a.code.localeCompare(b.code));
  for(const b of sorted) {{
    const tid = String(b.trip);
    const t   = trips[tid] || {{}};
    const s   = tripSummary(tid);
    const warn = (s.overW||s.overC||s.overD) ?
      [(s.overW?`W:${{s.w.toFixed(0)}}/${{s.maxW.toFixed(0)}}`:''),
       (s.overC?`C:${{s.c.toFixed(2)}}/${{s.maxC.toFixed(2)}}`:''),
       (s.overD?`D:${{s.drops}}/${{s.maxD}}`:'')].filter(x=>x).join(',') : '';
    rows.push([b.trip, t.tripNo||`T${{b.trip}}`, t.truck||'6W',
               b.code, b.name, b.province, b.district,
               b.weight, b.cube, b.maxVeh, b.bu, warn]);
  }}

  // Summary sheet
  const sumRows = [['Trip','Trip No','ประเภทรถ','สาขา','น้ำหนักรวม','คิวรวม','Max W','Max C','%W','%C','สถานะ']];
  for(const tid of sortedTripKeys()) {{
    const t = trips[tid]||{{}};
    const s = tripSummary(tid);
    const status = (s.overW||s.overC||s.overD)?'⚠️ เกิน':'✅ ปกติ';
    sumRows.push([parseInt(tid), t.tripNo||`T${{tid}}`, t.truck||'6W',
                  s.drops, +s.w.toFixed(2), +s.c.toFixed(3),
                  +s.maxW.toFixed(2), +s.maxC.toFixed(3),
                  +(s.wPct.toFixed(1)), +(s.cPct.toFixed(1)), status]);
  }}

  const wb = XLSX.utils.book_new();
  const ws1 = XLSX.utils.aoa_to_sheet(rows);
  const ws2 = XLSX.utils.aoa_to_sheet(sumRows);
  XLSX.utils.book_append_sheet(wb, ws1, 'Trip Detail');
  XLSX.utils.book_append_sheet(wb, ws2, 'Trip Summary');
  XLSX.writeFile(wb, 'trips_interactive.xlsx');
  showToast('📥 บันทึก trips_interactive.xlsx แล้ว');
}}

// ═══════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════
renderMarkers();
renderSidebar();
fitAll();

// 🔧 Streamlit srcdoc iframe: ใช้ fixed px height ตาม components.v1.html(height=780)
function _fixMapSize() {{
  var mapEl = document.getElementById('map');
  mapEl.style.height = '780px';
  map.invalidateSize({{reset:true, animate:false}});
  fitAll();
}}

document.addEventListener('DOMContentLoaded', function() {{
  setTimeout(_fixMapSize, 50);
}});
setTimeout(_fixMapSize, 100);
setTimeout(_fixMapSize, 400);
setTimeout(_fixMapSize, 900);
setTimeout(_fixMapSize, 1800);
window.addEventListener('resize', _fixMapSize);
</script>
</body>
</html>"""
    return html
