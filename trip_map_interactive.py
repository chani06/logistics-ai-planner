"""
trip_map_interactive.py — Modern Interactive Trip Map v4
White/Green theme, OSRM routes, trip confirmation, vehicle edit, Excel export with styles.
"""
import json
import pathlib
import pandas as pd

_STATIC = pathlib.Path(__file__).parent / "static"

try:
    _LEAFLET_CSS_CONTENT = (_STATIC / "leaflet.css").read_text(encoding="utf-8")
except Exception:
    _LEAFLET_CSS_CONTENT = ""

try:
    _LEAFLET_JS_CONTENT = (_STATIC / "leaflet.js").read_text(encoding="utf-8")
except Exception:
    _LEAFLET_JS_CONTENT = ""

VEHICLE_LIMITS = {
    "4W":  {"max_w": 2500, "max_c": 5.0,  "max_drops": 12},
    "JB":  {"max_w": 3500, "max_c": 7.0,  "max_drops": 12},
    "6W":  {"max_w": 6000, "max_c": 20.0, "max_drops": 999},
}

PUNTHAI_VEHICLE_LIMITS = {
    "4W":  {"max_w": 2500, "max_c": 5.0,  "max_drops": 5},
    "JB":  {"max_w": 3500, "max_c": 7.0,  "max_drops": 7},
    "6W":  {"max_w": 6000, "max_c": 20.0, "max_drops": 999},
}

TRIP_COLORS = [
    "#ef4444","#f97316","#eab308","#22c55e","#14b8a6","#3b82f6","#8b5cf6","#ec4899",
    "#dc2626","#ea580c","#ca8a04","#16a34a","#0d9488","#2563eb","#7c3aed","#db2777",
    "#b91c1c","#c2410c","#a16207","#15803d","#0f766e","#1d4ed8","#6d28d9","#be185d",
    "#991b1b","#9a3412","#854d0e","#166534","#115e59","#1e40af","#5b21b6","#9d174d",
    "#7f1d1d","#7c2d12","#713f12","#14532d","#134e4a","#1e3a8a","#4c1d95","#831843",
    "#f43f5e","#06b6d4",
]


def _safe(v, default=""):
    if v is None:
        return default
    try:
        import math
        if isinstance(v, float) and math.isnan(v):
            return default
    except Exception:
        pass
    return v


def build_interactive_map_html(
    result_df,
    summary_df,
    limits=None,
    punthai_limits=None,
    punthai_buffer=1.0,
    maxmart_buffer=1.1,
    trip_no_map=None,
    dc_lat=14.1459,
    dc_lon=100.6873,
    route_cache=None,
):
    lim     = limits         or VEHICLE_LIMITS
    plim    = punthai_limits or PUNTHAI_VEHICLE_LIMITS
    tno_map = trip_no_map    or {}

    if _LEAFLET_CSS_CONTENT:
        _leaflet_css_tag = "<style>" + _LEAFLET_CSS_CONTENT + "</style>"
    else:
        _leaflet_css_tag = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css" crossorigin="">'
    if _LEAFLET_JS_CONTENT:
        _leaflet_js_tag = "<script>" + _LEAFLET_JS_CONTENT + "</script>"
    else:
        _leaflet_js_tag = '<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>'

    lat_col = "_lat" if "_lat" in result_df.columns else "Latitude"
    lon_col = "_lon" if "_lon" in result_df.columns else "Longitude"

    df = result_df.copy()
    df = df[df["Trip"] > 0].copy()
    if lat_col in df.columns and lon_col in df.columns:
        df = df[(df[lat_col].fillna(0) > 0) & (df[lon_col].fillna(0) > 0)].copy()
    else:
        return "<p style='color:red'>ไม่พบคอลัมน์พิกัด (_lat/_lon) ในผลลัพธ์</p>"

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
            "code":        str(_safe(row.get("Code", ""))),
            "name":        str(_safe(row.get("Name", ""))),
            "province":    str(_safe(row.get("Province", ""))),
            "district":    str(_safe(row.get("District", row.get("_district", "")))),
            "subdistrict": str(_safe(row.get("Subdistrict", row.get("_subdistrict", "")), "")),
            "route":       str(_safe(row.get("Reference", row.get("Route", "")), "")),
            "origQty":     int(float(_safe(row.get("OriginalQty", row.get("Original QTY", 0)) or 0, 0))),
            "lat":         float(row[lat_col]),
            "lon":         float(row[lon_col]),
            "trip":        int(_safe(row.get("Trip", 0), 0)),
            "weight":      float(_safe(row.get("Weight", 0), 0)),
            "cube":        float(_safe(row.get("Cube", 0), 0)),
            "vtype":       vtype,
            "maxVeh":      mv,
            "isPunthai":   is_pt,
            "bu":          bu,
            "vc":          str(_safe(row.get("VehicleCheck", ""), "")),
        })

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
    for b in branches_json:
        tid = str(b["trip"])
        if tid not in trips_json:
            trips_json[tid] = {
                "trip":     b["trip"],
                "truck":    b["vtype"],
                "tripNo":   tno_map.get(b["trip"], f"T{b['trip']:03d}"),
                "branches": 0,
            }

    def _clean_str(s):
        """Remove/replace surrogate characters that would break UTF-8 encoding."""
        if not isinstance(s, str):
            return s
        try:
            s.encode('utf-8')
            return s
        except UnicodeEncodeError:
            return s.encode('utf-8', errors='replace').decode('utf-8')

    def _clean_obj(o):
        if isinstance(o, dict):
            return {k: _clean_obj(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_clean_obj(i) for i in o]
        if isinstance(o, str):
            return _clean_str(o)
        return o

    lim_json     = json.dumps(_clean_obj(lim),           ensure_ascii=False)
    plim_json    = json.dumps(_clean_obj(plim),          ensure_ascii=False)
    branches_js  = json.dumps(_clean_obj(branches_json), ensure_ascii=False)
    trips_js     = json.dumps(_clean_obj(trips_json),    ensure_ascii=False)
    colors_js    = json.dumps(TRIP_COLORS,               ensure_ascii=False)

    # Build pre-computed route lookup indexed by trip ID
    tid_routes = {}
    if route_cache:
        from collections import defaultdict as _dd
        trip_br_map = _dd(list)
        for b in branches_json:
            trip_br_map[str(b['trip'])].append(b)
        for tid, tbrs in trip_br_map.items():
            wps = [[dc_lat, dc_lon]] + [[b['lat'], b['lon']] for b in tbrs] + [[dc_lat, dc_lon]]
            ckey = '|'.join([f'{lat:.4f},{lon:.4f}' for lat, lon in wps])
            if ckey in route_cache:
                rv = route_cache[ckey]
                coords_f = rv.get('coords', [])
                if len(coords_f) > 500:
                    step = max(1, len(coords_f) // 500)
                    coords_f = coords_f[::step]
                tid_routes[tid] = {'coords': coords_f, 'distance_m': rv.get('distance', 0) * 1000}
    tid_routes_js = json.dumps(tid_routes, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<title>Trip Map</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
{_leaflet_css_tag}
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html,body{{width:100%;height:830px;overflow:hidden;font-family:'Inter','Segoe UI',system-ui,sans-serif;font-size:13px;background:#f0fdf4;color:#0f172a}}
#app{{display:flex;width:100%;height:830px;position:relative;overflow:hidden;background:#f0fdf4}}
#sidebar{{width:370px;min-width:260px;max-width:480px;height:830px;display:flex;flex-direction:column;background:#fff;border-right:1px solid #d1fae5;flex-shrink:0;position:relative;z-index:10;box-shadow:4px 0 24px rgba(5,150,105,.07)}}
#map-wrap{{flex:1;height:830px;position:relative;overflow:hidden;background:#e8f5e9}}
#map{{width:100%;height:100%}}
/* ── Sidebar header ── */
#sb-header{{background:linear-gradient(135deg,#064e3b 0%,#059669 60%,#10b981 100%);padding:14px 16px 12px;flex-shrink:0;border-bottom:1px solid rgba(255,255,255,.15);position:relative;overflow:hidden}}
#sb-header::after{{content:'';position:absolute;top:-18px;right:-18px;width:90px;height:90px;background:rgba(255,255,255,.06);border-radius:50%;pointer-events:none}}
#sb-header h1{{font-size:15px;font-weight:800;letter-spacing:.4px;color:#fff;display:flex;align-items:center;gap:8px;text-shadow:0 1px 4px rgba(0,0,0,.2)}}
#sb-chips{{margin-top:9px;display:flex;gap:6px;flex-wrap:wrap}}
.chip{{padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;background:rgba(255,255,255,.2);border:1px solid rgba(255,255,255,.3);color:#fff;letter-spacing:.2px}}
.chip-warn{{background:rgba(251,191,36,.35);border-color:rgba(251,191,36,.6);color:#fef3c7}}
.chip-err{{background:rgba(239,68,68,.35);border-color:rgba(239,68,68,.6);color:#fee2e2}}
/* ── Toolbar ── */
#toolbar{{padding:10px 12px 6px;background:#fff;flex-shrink:0;border-bottom:1px solid #f0fdf4}}
#search-inp{{width:100%;padding:8px 12px 8px 36px;background:#f0fdf4;border:1.5px solid #bbf7d0;border-radius:10px;color:#0f172a;font-size:12px;outline:none;transition:all .2s;font-family:inherit;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='14' height='14' fill='%2334d399' viewBox='0 0 16 16'%3E%3Cpath d='M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001q.044.06.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1 1 0 0 0-.115-.099zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:10px center}}
#search-inp::placeholder{{color:#94a3b8}}
#search-inp:focus{{border-color:#10b981;background:#fff;box-shadow:0 0 0 3px rgba(16,185,129,.12)}}
/* Trip filter bar */
#filter-bar{{display:flex;align-items:center;gap:6px;padding:5px 12px 7px;background:#fff;flex-shrink:0}}
#trip-filter{{flex:1;padding:6px 10px;background:#f0fdf4;border:1.5px solid #bbf7d0;border-radius:10px;color:#0f172a;font-size:12px;outline:none;cursor:pointer;font-family:inherit}}
#trip-filter:focus{{border-color:#10b981;box-shadow:0 0 0 3px rgba(16,185,129,.12)}}
#filter-lbl{{font-size:11px;color:#6b7280;font-weight:600;white-space:nowrap}}
/* Tools */
#tools{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;padding:6px 10px 4px;background:#fff;flex-shrink:0}}
#tools2{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;padding:4px 10px 8px;background:#fff;border-bottom:2px solid #f0fdf4;flex-shrink:0}}
.tbtn{{padding:8px 6px;border:none;border-radius:10px;cursor:pointer;font-size:11px;font-weight:700;transition:all .18s;white-space:nowrap;display:flex;align-items:center;justify-content:center;gap:4px;line-height:1.2;letter-spacing:.1px;font-family:inherit}}
.tbtn:hover{{transform:translateY(-2px);box-shadow:0 4px 14px rgba(0,0,0,.15)}}
.tbtn:active{{transform:translateY(0);filter:brightness(.93)}}
.tbtn-green{{background:linear-gradient(135deg,#059669,#10b981);color:#fff;box-shadow:0 2px 8px rgba(16,185,129,.3)}}
.tbtn-slate{{background:#f1f5f9;color:#475569;border:1.5px solid #e2e8f0}}
.tbtn-slate:hover{{background:#e2e8f0;border-color:#cbd5e1}}
.tbtn-teal{{background:#d1fae5;color:#065f46;border:1.5px solid #6ee7b7;font-weight:700}}
.tbtn-teal:hover{{background:#a7f3d0;border-color:#34d399}}
.tbtn-blue{{background:linear-gradient(135deg,#2563eb,#3b82f6);color:#fff;box-shadow:0 2px 8px rgba(59,130,246,.3)}}
.tbtn-blue.active{{background:linear-gradient(135deg,#1d4ed8,#2563eb);box-shadow:0 0 0 3px rgba(59,130,246,.35),0 2px 8px rgba(59,130,246,.3)}}
.tbtn-amber{{background:#fef9c3;color:#92400e;border:1.5px solid #fde68a;font-weight:700}}
.tbtn-amber.active{{background:linear-gradient(135deg,#d97706,#f59e0b);color:#fff;border-color:#d97706;box-shadow:0 0 0 3px rgba(251,191,36,.4)}}
#mode-banner{{display:none;padding:6px 14px;background:linear-gradient(135deg,#fef3c7,#fde68a);border-bottom:1px solid #f59e0b;font-size:11px;color:#78350f;flex-shrink:0;text-align:center;font-weight:700;letter-spacing:.3px;animation:pulse-banner 2.5s ease-in-out infinite}}
#mode-banner.visible{{display:block}}
@keyframes pulse-banner{{0%,100%{{opacity:1}}50%{{opacity:.65}}}}
/* ── Trip list ── */
#trip-list{{flex:1;overflow-y:auto;padding:8px 8px 20px;background:#f8fafc}}
#trip-list::-webkit-scrollbar{{width:5px}}
#trip-list::-webkit-scrollbar-thumb{{background:#bbf7d0;border-radius:3px}}
#trip-list::-webkit-scrollbar-track{{background:transparent}}
/* ── Trip card ── */
.tc{{background:#fff;border:1.5px solid #e2e8f0;border-left:4px solid #10b981;border-radius:12px;margin-bottom:6px;overflow:hidden;transition:all .2s;cursor:default}}
.tc:hover{{box-shadow:0 4px 20px rgba(5,150,105,.12);border-color:#bbf7d0;transform:translateY(-1px)}}
.tc.tc-active{{box-shadow:0 0 0 2.5px #10b981,0 4px 16px rgba(16,185,129,.2)!important;border-color:#10b981!important}}
.tc.tc-confirmed{{border-left-color:#15803d!important;background:#f0fdf4}}
.tc.tc-warn{{border-left-color:#f59e0b!important}}
.tc.tc-err{{border-left-color:#ef4444!important}}
.tc-head{{padding:10px 11px 5px;cursor:pointer;display:flex;align-items:flex-start;gap:8px;user-select:none}}
.tc-dot{{width:11px;height:11px;border-radius:50%;flex-shrink:0;margin-top:4px;box-shadow:0 1px 4px rgba(0,0,0,.2)}}
.tc-main{{flex:1;min-width:0}}
.tc-title{{font-weight:700;font-size:13px;color:#0f172a;display:flex;align-items:center;gap:5px;flex-wrap:wrap}}
.tc-sub{{font-size:11px;color:#6b7280;margin-top:3px}}
.tc-right{{display:flex;flex-direction:column;align-items:flex-end;gap:3px;flex-shrink:0}}
.tc-dist{{font-size:11px;font-weight:600;color:#059669}}
.tc-dist.dist-err{{color:#dc2626}}
.tc-chev{{font-size:9px;color:#9ca3af;transition:transform .2s;margin-top:1px}}
.tc-chev.open{{transform:rotate(90deg)}}
/* ── Progress bars ── */
.bars{{padding:3px 10px 6px 30px}}
.bar-row{{display:flex;align-items:center;gap:6px;margin-bottom:3px}}
.bar-lbl{{font-size:12px;color:#9ca3af;width:18px;flex-shrink:0;text-align:center}}
.bar-track{{flex:1;height:5px;border-radius:3px;background:#f0fdf4;overflow:hidden;border:1px solid #d1fae5}}
.bar-fill{{height:100%;border-radius:3px;transition:width .4s cubic-bezier(.4,0,.2,1)}}
.bar-val{{font-size:10px;color:#6b7280;white-space:nowrap;min-width:30px;text-align:right;font-weight:600}}
/* ── Expanded body ── */
.tc-body{{padding:0 10px 10px 30px;display:none}}
.tc-body.open{{display:block}}
.veh-sel{{width:100%;padding:7px 24px 7px 10px;margin-top:5px;background:#f0fdf4;color:#0f172a;border:1.5px solid #bbf7d0;border-radius:9px;font-size:12px;cursor:pointer;outline:none;appearance:none;font-family:inherit;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%2310b981'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 8px center;transition:border .15s}}
.veh-sel:focus{{border-color:#10b981;box-shadow:0 0 0 3px rgba(16,185,129,.12)}}
.veh-sel:disabled{{opacity:.5;cursor:default}}
.veh-sel option{{background:#fff;color:#0f172a}}
.br-list{{margin-top:8px}}
.br-row{{font-size:11px;color:#475569;padding:4px 0;border-bottom:1px solid #f0fdf4;display:flex;justify-content:space-between;gap:4px;align-items:center}}
.br-row:last-child{{border-bottom:none}}
.br-dist{{font-size:10px;color:#10b981;font-weight:600;white-space:nowrap;flex-shrink:0}}
.tc-btns{{display:flex;gap:5px;margin-top:9px}}
.tc-btn{{flex:1;padding:6px 6px;border:none;border-radius:9px;cursor:pointer;font-size:11px;font-weight:700;transition:all .18s;font-family:inherit}}
.tc-btn:hover{{transform:translateY(-1px);box-shadow:0 3px 10px rgba(0,0,0,.15)}}
.btn-zoom{{background:#f0fdf4;color:#065f46;border:1.5px solid #6ee7b7}}
.btn-zoom:hover{{background:#d1fae5;border-color:#34d399}}
.btn-confirm{{background:linear-gradient(135deg,#059669,#10b981);color:#fff;border:none}}
.btn-confirm.locked{{background:#f0fdf4;color:#6b7280;border:1.5px solid #d1d5db;cursor:default}}
.btn-confirm.locked:hover{{transform:none;box-shadow:none}}
/* ── Badges ── */
.badge{{display:inline-flex;align-items:center;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;line-height:1.5}}
.b4W{{background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe}}
.bJB{{background:#f5f3ff;color:#6d28d9;border:1px solid #ddd6fe}}
.b6W{{background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0}}
.bwarn{{background:#fffbeb;color:#b45309;border:1px solid #fde68a}}
.berr{{background:#fef2f2;color:#b91c1c;border:1px solid #fecaca}}
.block{{background:#f0fdf4;color:#15803d;border:1px solid #86efac}}
/* ── Info panel ── */
#info-panel{{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:7000;background:#fff;backdrop-filter:blur(20px);border:1.5px solid #bbf7d0;border-radius:18px;color:#0f172a;padding:20px 24px 20px;display:none;opacity:0;transition:opacity .22s ease;width:620px;max-width:96vw;box-shadow:0 20px 70px rgba(5,150,105,.18),0 4px 20px rgba(0,0,0,.1)}}
#info-panel.visible{{opacity:1}}
.ip-head{{display:flex;align-items:flex-start;justify-content:space-between;gap:8px}}
.ip-code{{font-weight:800;font-size:15px;color:#059669}}
.ip-detail{{font-size:11px;color:#6b7280;margin-top:3px;line-height:1.6}}
.ip-close{{background:none;border:1.5px solid #e2e8f0;color:#9ca3af;font-size:16px;cursor:pointer;padding:0;width:28px;height:28px;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:all .15s}}
.ip-close:hover{{background:#f0fdf4;border-color:#10b981;color:#059669}}
.ip-move{{display:flex;flex-direction:column;gap:8px;margin-top:14px}}
.ip-sel{{flex:1;padding:8px 12px;background:#f0fdf4;color:#0f172a;border:1.5px solid #bbf7d0;border-radius:10px;font-size:12px;outline:none;cursor:pointer;height:auto;font-family:inherit;transition:border .15s}}
.ip-sel:focus{{border-color:#10b981;box-shadow:0 0 0 3px rgba(16,185,129,.12)}}
.ip-sel option{{background:#fff;color:#0f172a;padding:4px 8px}}
.ip-btn{{padding:8px 16px;border:none;border-radius:10px;cursor:pointer;font-size:12px;font-weight:700;transition:all .18s;font-family:inherit}}
.ip-btn:hover{{transform:translateY(-1px);box-shadow:0 3px 10px rgba(0,0,0,.15)}}
.ip-move-btn{{background:linear-gradient(135deg,#059669,#10b981);color:#fff}}
.ip-cancel-btn{{background:#f1f5f9;color:#6b7280;border:1.5px solid #e2e8f0}}
.ip-cancel-btn:hover{{background:#e2e8f0}}
.ip-warn{{font-size:11px;color:#b45309;margin-top:5px;line-height:1.5;background:#fffbeb;border-radius:8px;padding:7px 10px;display:none;border:1px solid #fde68a}}
.ip-warn:not(:empty){{display:block}}
.ip-actions{{display:flex;gap:8px;margin-top:4px}}
/* ── Selection panel ── */
#sel-panel{{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);z-index:5000;background:#fff;backdrop-filter:blur(24px);padding:0;display:none;border:1.5px solid #bbf7d0;border-radius:22px;box-shadow:0 24px 80px rgba(5,150,105,.22),0 4px 20px rgba(0,0,0,.08);min-width:560px;max-width:96vw;width:720px;max-height:90vh;overflow-y:auto;animation:selIn .2s cubic-bezier(.34,1.56,.64,1)}}
@keyframes selIn{{from{{opacity:0;transform:translate(-50%,-50%) scale(.88)}}to{{opacity:1;transform:translate(-50%,-50%) scale(1)}}}}
#sel-panel-bg{{position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.35);z-index:4999;display:none;backdrop-filter:blur(3px)}}
#sel-title-bar{{background:linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%);border-bottom:1px solid #bbf7d0;padding:16px 20px 14px;border-radius:20px 20px 0 0;display:flex;align-items:center;gap:10px}}
#sel-count{{font-weight:800;color:#065f46;font-size:15px}}
#sel-summary{{font-size:12px;color:#047857;background:#d1fae5;padding:3px 12px;border-radius:20px;border:1px solid #6ee7b7;font-weight:600}}
#sel-trips{{font-size:11px;color:#9ca3af;margin-left:2px}}
#sel-body{{padding:18px 20px 20px}}
/* Distance grid in sel panel */
#sel-dist-grid{{font-size:11px;background:#f0fdf4;border-radius:10px;padding:8px 12px;margin-bottom:12px;border:1px solid #d1fae5;line-height:1.7;max-height:90px;overflow-y:auto;display:none}}
#sel-dist-grid.show{{display:block}}
.sel-trip-row{{display:flex;align-items:center;gap:10px;margin-bottom:12px}}
.sel-trip-label{{font-size:12px;color:#374151;width:90px;flex-shrink:0;font-weight:600}}
.sel-veh-grp{{display:flex;gap:5px}}
.veh-chip{{padding:6px 14px;border-radius:20px;font-size:12px;font-weight:700;cursor:pointer;border:1.5px solid #e2e8f0;background:#f9fafb;color:#6b7280;transition:all .18s}}
.veh-chip:hover{{background:#f0fdf4;border-color:#6ee7b7;color:#059669}}
.veh-chip.vc-active-4W{{background:#eff6ff;color:#1d4ed8;border-color:#3b82f6;box-shadow:0 0 0 2px rgba(59,130,246,.2)}}
.veh-chip.vc-active-6W{{background:#f0fdf4;color:#15803d;border-color:#10b981;box-shadow:0 0 0 2px rgba(16,185,129,.2)}}
.veh-chip.vc-active-JB{{background:#f5f3ff;color:#6d28d9;border-color:#8b5cf6;box-shadow:0 0 0 2px rgba(139,92,246,.2)}}
.sel-target-wrap{{display:flex;align-items:center;gap:10px;margin-bottom:14px}}
.sel-trip-label2{{font-size:12px;color:#374151;width:90px;flex-shrink:0;font-weight:600}}
#sel-target{{flex:1;padding:9px 14px;background:#f0fdf4;color:#0f172a;border:1.5px solid #bbf7d0;border-radius:11px;font-size:13px;outline:none;cursor:pointer;font-family:inherit;transition:border .15s}}
#sel-target:focus{{border-color:#10b981;box-shadow:0 0 0 3px rgba(16,185,129,.12)}}
#sel-target option{{background:#fff;color:#0f172a}}
.sel-actions{{display:flex;gap:8px;margin-top:4px}}
.sel-btn{{flex:1;padding:11px 0;border:none;border-radius:12px;cursor:pointer;font-size:13px;font-weight:700;transition:all .18s;letter-spacing:.2px;font-family:inherit}}
.sel-btn:hover{{transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,0,0,.18)}}
.sel-btn-move{{background:linear-gradient(135deg,#059669,#10b981);color:#fff}}
.sel-btn-swap{{background:linear-gradient(135deg,#7c3aed,#8b5cf6);color:#fff}}
.sel-btn-cancel{{background:#f9fafb;color:#6b7280;border:1.5px solid #e2e8f0}}
.sel-btn-cancel:hover{{background:#f1f5f9;box-shadow:none}}
#sel-warn{{font-size:11px;color:#b45309;margin-top:10px;min-height:14px;line-height:1.6;background:#fffbeb;border-radius:9px;padding:7px 12px;display:none;border:1px solid #fde68a}}
#sel-warn.show{{display:block}}
/* ── Quick trip-buttons grid ── */
.tqb-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(90px,1fr));gap:6px;margin:8px 0 4px;max-height:220px;overflow-y:auto;min-height:44px}}
.tqb{{padding:8px 4px;border-radius:9px;font-size:12px;font-weight:700;cursor:pointer;border:2.5px solid transparent;transition:all .15s;color:#fff;text-align:center;line-height:1.3;width:100%;box-sizing:border-box}}
.tqb:hover{{transform:scale(1.06);box-shadow:0 2px 10px rgba(0,0,0,.28);filter:brightness(1.12)}}
.tqb:active{{transform:scale(.96)}}
.tqb-locked{{opacity:.45;cursor:not-allowed;filter:grayscale(.4)}}
.tqb-over{{border-color:#ef4444!important;border-width:2.5px!important}}
.tqb-current{{opacity:.38;cursor:default;border-style:dashed;border-color:rgba(255,255,255,.55)!important}}
.tqb-new{{background:#6b7280!important}}
.tqb-sub{{font-size:10px;opacity:.88;display:block;margin-top:2px;font-weight:500}}
.sel-section-lbl{{font-size:10px;font-weight:700;color:#374151;margin-bottom:3px;margin-top:4px}}
/* ── Sidebar swap button styles ── */
.tc-btn.btn-swap{{background:linear-gradient(135deg,#7c3aed,#8b5cf6);color:#fff}}
.tc-btn.btn-swap-ok{{background:linear-gradient(135deg,#d97706,#f59e0b);color:#fff;animation:pulse-amber .75s infinite alternate}}
.tc-btn.btn-swap-target{{background:linear-gradient(135deg,#dc2626,#ef4444);color:#fff}}
@keyframes pulse-amber{{from{{box-shadow:none}}to{{box-shadow:0 0 8px rgba(251,191,36,.65)}}}}
/* ── Floating map controls ── */
#map-controls{{position:absolute;top:10px;right:10px;z-index:910;display:flex;flex-direction:column;gap:6px}}
.map-ctrl-btn{{width:38px;height:38px;border-radius:12px;background:rgba(255,255,255,.96);border:1.5px solid #d1fae5;cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center;transition:all .18s;box-shadow:0 2px 12px rgba(5,150,105,.12);color:#059669;backdrop-filter:blur(8px)}}
.map-ctrl-btn:hover{{background:#fff;border-color:#10b981;box-shadow:0 4px 16px rgba(5,150,105,.22);transform:scale(1.08)}}
.map-ctrl-btn:active{{transform:scale(.95)}}
#route-ind{{position:absolute;top:10px;left:10px;z-index:900;background:rgba(255,255,255,.95);color:#059669;padding:7px 14px;border-radius:10px;font-size:11px;font-weight:700;border:1.5px solid #bbf7d0;display:none;pointer-events:none;box-shadow:0 2px 8px rgba(5,150,105,.12);backdrop-filter:blur(6px)}}
/* ── Settings panel ── */
#settings-panel{{position:absolute;top:58px;right:10px;z-index:950;background:rgba(255,255,255,.98);backdrop-filter:blur(20px);border:1.5px solid #bbf7d0;border-radius:16px;padding:16px 18px 18px;width:260px;display:none;box-shadow:0 12px 40px rgba(5,150,105,.2);animation:fadeInDown .2s ease}}
@keyframes fadeInDown{{from{{opacity:0;transform:translateY(-8px)}}to{{opacity:1;transform:translateY(0)}}}}
#settings-panel h3{{font-size:13px;font-weight:800;color:#065f46;margin-bottom:14px;display:flex;align-items:center;gap:7px}}
.setting-row{{margin-bottom:12px}}
.setting-lbl{{font-size:12px;font-weight:600;color:#374151;margin-bottom:5px;display:block}}
.setting-row-inline{{display:flex;align-items:center;gap:6px}}
.setting-inp{{width:80px;padding:7px 10px;background:#f0fdf4;border:1.5px solid #bbf7d0;border-radius:9px;color:#0f172a;font-size:13px;font-weight:700;outline:none;text-align:right;font-family:inherit;transition:border .15s}}
.setting-inp:focus{{border-color:#10b981;box-shadow:0 0 0 3px rgba(16,185,129,.12)}}
.setting-unit{{font-size:12px;color:#6b7280;font-weight:500}}
.setting-apply{{width:100%;padding:9px;background:linear-gradient(135deg,#059669,#10b981);color:#fff;border:none;border-radius:10px;font-size:12px;font-weight:700;cursor:pointer;margin-top:6px;transition:all .18s;font-family:inherit}}
.setting-apply:hover{{box-shadow:0 4px 14px rgba(16,185,129,.4);transform:translateY(-1px)}}
/* ── Legend panel ── */
#legend-panel{{position:absolute;bottom:10px;right:10px;z-index:900;background:rgba(255,255,255,.97);backdrop-filter:blur(16px);border:1.5px solid #d1fae5;border-radius:14px;padding:12px 14px;min-width:160px;max-width:220px;display:none;box-shadow:0 8px 30px rgba(5,150,105,.14);animation:fadeInUp .2s ease}}
@keyframes fadeInUp{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
#legend-panel h4{{font-size:12px;font-weight:800;color:#065f46;margin-bottom:8px;display:flex;align-items:center;gap:6px}}
.legend-item{{display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:11px;color:#374141}}
.legend-dot{{width:14px;height:14px;border-radius:50%;flex-shrink:0;border:1.5px solid rgba(0,0,0,.1);box-shadow:0 1px 3px rgba(0,0,0,.15);display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:700;color:#fff}}
.legend-sym{{width:24px;height:24px;border-radius:6px;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:13px}}
#legend-trips{{max-height:120px;overflow-y:auto;padding-right:2px}}
#legend-trips::-webkit-scrollbar{{width:3px}}
#legend-trips::-webkit-scrollbar-thumb{{background:#d1fae5;border-radius:2px}}
/* ── Confirm modal ── */
#confirm-overlay{{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.5);z-index:20000;align-items:center;justify-content:center;backdrop-filter:blur(4px)}}
#confirm-overlay.show{{display:flex}}
#confirm-box{{background:#fff;border:1.5px solid #bbf7d0;border-radius:16px;padding:24px 26px;max-width:360px;width:90%;box-shadow:0 16px 50px rgba(5,150,105,.2)}}
#confirm-msg{{color:#0f172a;font-size:13px;line-height:1.7;margin-bottom:18px;white-space:pre-wrap}}
.confirm-btns{{display:flex;gap:8px;justify-content:flex-end}}
/* ── Toast ── */
#toast{{position:fixed;top:16px;left:50%;transform:translateX(-50%);background:#065f46;color:#fff;border:1.5px solid #059669;padding:9px 22px;border-radius:24px;font-size:12px;font-weight:600;z-index:19999;display:none;pointer-events:none;box-shadow:0 6px 24px rgba(5,150,105,.35)}}
/* ── Leaflet overrides ── */
.leaflet-control-zoom{{border:1.5px solid #bbf7d0!important;border-radius:10px!important;overflow:hidden;box-shadow:0 2px 12px rgba(5,150,105,.12)!important}}
.leaflet-control-zoom a{{background:#fff!important;color:#059669!important;border-bottom:1px solid #d1fae5!important;font-size:16px!important;line-height:26px!important}}
.leaflet-control-zoom a:hover{{background:#f0fdf4!important}}
.leaflet-tooltip{{background:rgba(255,255,255,.97);border:1.5px solid #d1fae5;color:#0f172a;font-size:12px;border-radius:9px;box-shadow:0 4px 16px rgba(5,150,105,.14);padding:5px 11px;font-weight:500}}
.leaflet-tooltip::before{{display:none}}
.leaflet-popup-content-wrapper{{background:#fff;border:1.5px solid #d1fae5;color:#0f172a;border-radius:14px;box-shadow:0 8px 30px rgba(5,150,105,.15)}}
.leaflet-popup-tip{{background:#fff}}
.leaflet-popup-close-button{{color:#9ca3af!important}}
</style>
</head>
<body>
<div id="app">
  <div id="sidebar">
    <div id="sb-header">
      <h1>&#128666; Trip Planner</h1>
      <div id="sb-chips">
        <span class="chip" id="chip-br">-</span>
        <span class="chip" id="chip-tr">-</span>
        <span class="chip chip-warn" id="chip-w" style="display:none"></span>
        <span class="chip chip-err"  id="chip-e" style="display:none"></span>
      </div>
    </div>
    <div id="toolbar">
      <input id="search-inp" type="search" placeholder="&#128269; ค้นหาสาขา / รหัส..." autocomplete="off">
    </div>
    <div id="filter-bar">
      <span id="filter-lbl">&#127966; แสดง:</span>
      <select id="trip-filter" onchange="applyTripFilter()">
        <option value="ALL">ทั้งหมด</option>
      </select>
    </div>
    <div id="tools">
      <button class="tbtn tbtn-blue"  id="route-btn" onclick="toggleRoutes()" title="แสดง/ซ่อนเส้นทาง">&#128739; เส้นทาง</button>
      <button class="tbtn tbtn-slate" onclick="fitAll()" title="ย่อแผนที่ให้พอดี">&#9711; Fit</button>
      <button class="tbtn tbtn-slate" onclick="toggleAllTrips()" title="ขยาย/ย่อทุกทริป">&#8862; ขยาย/ย่อ</button>
    </div>
    <div id="tools2">
      <button class="tbtn tbtn-amber" id="select-btn" onclick="toggleSelectMode()" title="เลือกสาขาหลายๆ สาขา">&#9633; เลือกสาขา</button>
      <button class="tbtn tbtn-teal"  onclick="_cleanupAndRefresh()" title="ลบทริปว่าง เรียงหมายเลขใหม่">&#10024; ล้างว่าง</button>
      <button class="tbtn tbtn-green" onclick="exportExcel()" title="ส่งออก Excel">&#128229; Export</button>
    </div>
    <div id="mode-banner">&#9670; โหมดเลือกสาขา &nbsp;&mdash;&nbsp; คลิกสาขา หรือลากกรอบบนแผนที่</div>
    <div id="trip-list"></div>
  </div>
  <div id="map-wrap">
    <div id="map"></div>
    <div id="sel-panel-bg" onclick="clearSelection()"></div>
    <div id="sel-panel">
      <div id="sel-title-bar">
        <span>&#128203;</span>
        <span id="sel-count"></span>
        <span id="sel-summary"></span>
        <span id="sel-trips"></span>
        <button onclick="clearSelection()" class="ip-btn ip-cancel-btn" style="margin-left:auto;padding:4px 12px;border-radius:8px;font-size:12px">&#10005;</button>
      </div>
      <div id="sel-body">
        <div id="sel-dist-grid"></div>
        <div class="sel-trip-row">
          <span class="sel-trip-label">&#128666; ประเภทรถ</span>
          <div class="sel-veh-grp" id="sel-veh-grp">
            <span class="veh-chip" id="vchip-4W" onclick="setSelVeh('4W')">4W</span>
            <span class="veh-chip" id="vchip-JB" onclick="setSelVeh('JB')">JB</span>
            <span class="veh-chip vc-active-6W" id="vchip-6W" onclick="setSelVeh('6W')">6W</span>
          </div>
          <span id="sel-veh-hint" style="font-size:11px;color:#6b7280;margin-left:4px"></span>
        </div>
        <div class="sel-target-wrap">
          <div class="sel-section-lbl" style="color:#065f46">&#8599; ย้ายไป Trip (คลิกเลย)</div>
          <div id="sel-move-btns" class="tqb-grid"></div>
          <div id="sel-swap-row" style="display:none;margin-top:6px">
            <div class="sel-section-lbl" style="color:#7c3aed">&#8644; สลับกับ Trip</div>
            <div id="sel-swap-btns" class="tqb-grid"></div>
          </div>
          <select id="sel-target" style="display:none" onchange="previewSelTarget()"></select>
        </div>
        <div class="sel-actions">
          <button class="sel-btn sel-btn-cancel" onclick="clearSelection()">&#10005; ยกเลิก</button>
        </div>
        <div id="sel-warn"></div>
      </div>
    </div>
    <!-- Floating map controls -->
    <div id="map-controls">
      <button class="map-ctrl-btn" id="fs-btn"  onclick="toggleFullscreen()" title="เต็มจอ / ย่อจอ">&#x26F6;</button>
      <button class="map-ctrl-btn" id="cfg-btn" onclick="toggleSettings()" title="ตั้งค่าการจัดทริป">&#9881;&#65039;</button>
      <button class="map-ctrl-btn" id="leg-btn" onclick="toggleLegend()" title="สัญลักษณ์">&#127370;</button>
    </div>
    <div id="route-ind">&#9203; <span id="route-prog">กำลังโหลด...</span></div>
    <!-- Settings panel -->
    <div id="settings-panel">
      <h3>&#9881;&#65039; ตั้งค่าการจัดทริป</h3>
      <div class="setting-row">
        <label class="setting-lbl">&#127153; Punthai Buffer</label>
        <div class="setting-row-inline">
          <input class="setting-inp" type="number" id="cfg-pt-buf" min="50" max="200" step="5" value="100">
          <span class="setting-unit">%</span>
        </div>
      </div>
      <div class="setting-row">
        <label class="setting-lbl">&#127164; Maxmart/ผสม Buffer</label>
        <div class="setting-row-inline">
          <input class="setting-inp" type="number" id="cfg-mm-buf" min="50" max="200" step="5" value="110">
          <span class="setting-unit">%</span>
        </div>
      </div>
      <button class="setting-apply" onclick="applySettings()">&#9989; ใช้การตั้งค่า</button>
    </div>
    <!-- Legend panel -->
    <div id="legend-panel">
      <h4>&#128204; สัญลักษณ์แผนที่</h4>
      <div class="legend-item"><div class="legend-sym">&#127981;</div><span>DC ศูนย์กระจายสินค้า</span></div>
      <div class="legend-item"><div class="legend-sym" style="background:#f0fdf4;border-radius:50%;border:1.5px solid #10b981;font-size:9px;font-weight:700;color:#059669;width:18px;height:18px">1</div><span>สาขา (ตัวเลข = Trip)</span></div>
      <div class="legend-item"><div class="legend-dot" style="background:#fcd34d;width:18px;height:18px;border:2.5px solid #fcd34d"></div><span>สาขาที่เลือก</span></div>
      <div class="legend-item"><div class="legend-dot" style="background:#10b981;width:18px;height:18px"></div><span>ทริปยืนยันแล้ว</span></div>
      <div class="legend-item"><div class="legend-dot" style="background:#ef4444;width:18px;height:18px"></div><span>ทริปเกินขีดจำกัด</span></div>
      <div style="font-size:11px;font-weight:700;color:#065f46;margin:8px 0 5px">ทริปทั้งหมด</div>
      <div id="legend-trips"></div>
    </div>
    <!-- Info panel -->
    <div id="info-panel">
      <div class="ip-head">
        <div>
          <div class="ip-code" id="ip-code">-</div>
          <div class="ip-detail" id="ip-detail"></div>
        </div>
        <button class="ip-close" onclick="closeInfo()">&#10005;</button>
      </div>
      <div class="ip-move">
        <span style="font-size:12px;font-weight:700;color:#374151">&#8594; ย้ายไป Trip (คลิกเลย):</span>
        <div id="ip-trip-btns" class="tqb-grid"></div>
        <select class="ip-sel" id="move-sel" style="display:none"></select>
        <div class="ip-actions">
          <button class="ip-btn ip-cancel-btn" onclick="closeInfo()">ยกเลิก</button>
        </div>
      </div>
      <div class="ip-warn" id="move-warn"></div>
    </div>
  </div>
</div>
<div id="confirm-overlay">
  <div id="confirm-box">
    <div id="confirm-msg"></div>
    <div class="confirm-btns">
      <button class="tbtn tbtn-slate" onclick="_confirmResolve(false)">ยกเลิก</button>
      <button class="tbtn tbtn-green" onclick="_confirmResolve(true)">ตกลง</button>
    </div>
  </div>
</div>
<div id="toast"></div>
<script>
window.onerror = function(msg, src, line, col, err) {{
  // Suppress Leaflet null-ref when map container detaches during Streamlit re-render
  if (msg && (msg.indexOf('offsetWidth') !== -1 || msg.indexOf('offsetHeight') !== -1)) return true;
  const el = document.getElementById('map');
  if (el) el.innerHTML =
    '<pre style="color:#f87171;background:#0f172a;padding:16px;font-size:12px;margin:0">' +
    'JS Error: ' + msg + '\\nLine: ' + line + '\\n' + (err ? err.stack : '') + '</pre>';
  return true;
}};
</script>
{_leaflet_js_tag}
<script>
let _xlsxReady = false;
function _loadXlsx(cb) {{
  if (_xlsxReady) {{ cb(); return; }}
  const s = document.createElement('script');
  s.src = 'https://cdn.jsdelivr.net/npm/exceljs@4.4.0/dist/exceljs.min.js';
  s.onload = () => {{ _xlsxReady = true; cb(); }};
  s.onerror = () => alert('โหลด Excel library ล้มเหลว ตรวจสอบการเชื่อมต่อเน็ต');
  document.head.appendChild(s);
}}
</script>
<script>
// ── DATA ──────────────────────────────────────────────────────────────────
const RAW_BRANCHES = {branches_js};
const RAW_TRIPS    = {trips_js};
const COLORS       = {colors_js};
const LIMITS       = {lim_json};
const PLIMITS      = {plim_json};
const PUNTHAI_BUF  = {punthai_buffer};
const MAXMART_BUF  = {maxmart_buffer};
const DC           = [{dc_lat}, {dc_lon}];
const TID_ROUTES   = {tid_routes_js};

// ── STATE ─────────────────────────────────────────────────────────────────
let branches     = JSON.parse(JSON.stringify(RAW_BRANCHES));
let trips        = JSON.parse(JSON.stringify(RAW_TRIPS));
let markers      = {{}};
let selectedCode = null;
let _allOpen     = false;
let _showRoutes  = false;
let _mapInited   = false;
let _expandedTrips = new Set();
let _cachedTripKeys = null;
let _summaryCache   = {{}};
let _filterTrip     = 'ALL';   // 'ALL' or trip id string
let _settingsOpen   = false;
let _legendOpen     = false;
window._osrmCache        = {{}};
window._routeLines       = [];
window._tripDistances    = {{}};
window._tripRouteStatus  = {{}};
window._confirmedTrips   = new Set();
let _selectMode       = false;
let _selectedBranches = new Set();
let _selVeh           = null;
let _swapSrcTid       = null;   // for sidebar swap-mode
let _boxStart  = null;
let _boxRect   = null;
// Live buffers (updated by settings panel)
let _ptBuf = PUNTHAI_BUF;
let _mmBuf = MAXMART_BUF;

function _invalidateCache() {{ _cachedTripKeys = null; _summaryCache = {{}}; }}

// ── MAP ───────────────────────────────────────────────────────────────────
let map = null, _renderer = null;
function _initMap() {{
  const el = document.getElementById('map');
  if (!el || map) return;
  if (!el.offsetWidth || !el.offsetHeight) return;  // not visible yet — defer
  _renderer = L.canvas({{ padding: 0.5 }});
  try {{
  map = L.map('map', {{ center: DC, zoom: 6, zoomControl: true, preferCanvas: true, renderer: _renderer }});
  L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
    maxZoom: 19, subdomains: 'abcd',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
  }}).addTo(map);
  L.marker(DC, {{ icon: L.divIcon({{
    html: '<div style="background:#1d4ed8;color:#fff;border-radius:50%;width:32px;height:32px;display:flex;align-items:center;justify-content:center;font-size:15px;box-shadow:0 2px 10px rgba(0,0,0,.5);border:2px solid #93c5fd">&#127981;</div>',
    iconSize: [32,32], iconAnchor: [16,16], className: '',
  }}) }}).addTo(map).bindPopup('<b style="color:#60a5fa">DC Wang Noi</b>');
  // ── BOX SELECT ──────────────────────────────────────────────────────────
  const _mc = map.getContainer();
  _mc.addEventListener('mousedown', (e) => {{
    if (!_selectMode || e.button !== 0) return;
    if (e.target.closest('.leaflet-marker-icon')) return;
    const r = _mc.getBoundingClientRect();
    _boxStart = map.containerPointToLatLng(L.point(e.clientX - r.left, e.clientY - r.top));
    if (map && map.dragging) map.dragging.disable();
    const onMove = (ev) => {{
      if (!_boxStart) return;
      const r2 = _mc.getBoundingClientRect();
      const cur = map.containerPointToLatLng(L.point(ev.clientX - r2.left, ev.clientY - r2.top));
      if (_boxRect) map.removeLayer(_boxRect);
      _boxRect = L.rectangle([_boxStart, cur], {{ weight:2, color:'#fcd34d', fillColor:'#fcd34d', fillOpacity:0.08, dashArray:'5,4' }}).addTo(map);
    }};
    const onUp = (ev) => {{
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      if (!_boxStart) return;
      const r2 = _mc.getBoundingClientRect();
      const cur = map.containerPointToLatLng(L.point(ev.clientX - r2.left, ev.clientY - r2.top));
      const bnds = L.latLngBounds([_boxStart, cur]);
      if (bnds.getNorthEast().distanceTo(bnds.getSouthWest()) > 50) {{
        branches.filter(b => bnds.contains([b.lat, b.lon])).forEach(b => _selectedBranches.add(b.code));
        renderMarkers(); updateSelPanel();
      }}
      if (_boxRect) {{ map.removeLayer(_boxRect); _boxRect = null; }}
      _boxStart = null;
      if (map && map.dragging && map.getContainer()) map.dragging.enable();
    }};
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }});
  }} catch(e) {{ console.warn('Map init error:', e); map = null; _mapInited = false; }}
}}

// ── HELPERS ───────────────────────────────────────────────────────────────
function sortedTripKeys() {{
  if (!_cachedTripKeys) _cachedTripKeys = Object.keys(trips).sort((a,b) => parseInt(a)-parseInt(b));
  return _cachedTripKeys;
}}
function tripColor(tripId) {{
  const idx = sortedTripKeys().indexOf(String(tripId));
  return COLORS[idx % COLORS.length] || '#888';
}}
function getLimit(vtype, isPt) {{
  const L = isPt ? PLIMITS : LIMITS; return L[vtype] || L['6W'];
}}
function tripSummary(tid) {{
  const key = String(tid);
  if (_summaryCache[key]) return _summaryCache[key];
  const brs = branches.filter(b => String(b.trip) === key);
  const w = brs.reduce((s,b) => s+b.weight, 0);
  const c = brs.reduce((s,b) => s+b.cube,   0);
  const drops = brs.length;
  const isPt  = brs.length > 0 && brs.every(b => b.isPunthai);
  const vt    = (trips[key] || {{}}).truck || '6W';
  const buf   = isPt ? _ptBuf : _mmBuf;
  const lim   = getLimit(vt, isPt);
  const maxW  = lim.max_w * buf, maxC = lim.max_c * buf, maxD = lim.max_drops;
  const r = {{
    w, c, drops, maxW, maxC, maxD, isPt, vtype: vt,
    wPct: maxW > 0 ? w/maxW*100 : 0,
    cPct: maxC > 0 ? c/maxC*100 : 0,
    dPct: maxD < 999 ? drops/maxD*100 : 0,
    overW: w > maxW, overC: c > maxC, overD: drops > maxD,
  }};
  _summaryCache[key] = r; return r;
}}
function _pct(val) {{ return Math.min(val, 100).toFixed(1); }}
function _barColor(p) {{ return p > 100 ? '#ef4444' : p > 85 ? '#f59e0b' : '#10b981'; }}
function _vehBadge(v) {{ return '<span class="badge b' + v + '">' + v + '</span>'; }}
function _distLabel(tid) {{
  const d = window._tripDistances[String(tid)];
  return d != null ? d.toFixed(1) + ' km' : '';
}}

// ── SIDEBAR ───────────────────────────────────────────────────────────────
function _haversineKm(lat1,lon1,lat2,lon2) {{
  const R=6371, r=Math.PI/180;
  const dLat=(lat2-lat1)*r, dLon=(lon2-lon1)*r;
  const a=Math.sin(dLat/2)**2+Math.cos(lat1*r)*Math.cos(lat2*r)*Math.sin(dLon/2)**2;
  return R*2*Math.atan2(Math.sqrt(a),Math.sqrt(1-a));
}}
function _buildTripFilterOptions() {{
  const sel = document.getElementById('trip-filter');
  if (!sel) return;
  const prev = sel.value;
  sel.innerHTML = '<option value="ALL">ทั้งหมด (' + sortedTripKeys().length + ' ทริป)</option>';
  for (const tid of sortedTripKeys()) {{
    const t = trips[tid]||{{}}, s = tripSummary(tid);
    const opt = document.createElement('option');
    opt.value = tid;
    opt.textContent = 'Trip ' + tid + ' · ' + (t.truck||'6W') + ' · ' + s.drops + 'จุด';
    sel.appendChild(opt);
  }}
  sel.value = (prev && (prev==='ALL' || trips[prev])) ? prev : 'ALL';
}}
function applyTripFilter() {{
  const sel = document.getElementById('trip-filter');
  _filterTrip = sel ? sel.value : 'ALL';
  renderMarkers();
  renderSidebar(true);
}}
function renderSidebar(skipFilter) {{
  const list  = document.getElementById('trip-list');
  const tkeys = sortedTripKeys();
  if (!skipFilter) _buildTripFilterOptions();
  let warnCnt = 0, errCnt = 0, html = '';
  const filteredKeys = _filterTrip === 'ALL' ? tkeys : tkeys.filter(t => t === _filterTrip);
  for (const tid of filteredKeys) {{
    const t    = trips[tid] || {{}};
    const brs  = branches.filter(b => String(b.trip) === tid);
    const s    = tripSummary(tid);
    const col  = tripColor(parseInt(tid));
    const conf = window._confirmedTrips.has(tid);
    const over = s.overW || s.overC || s.overD;
    const sev  = over && (s.wPct > 110 || s.cPct > 110) ? 'err' : over ? 'warn' : '';
    if (sev === 'err') errCnt++; else if (sev === 'warn') warnCnt++;
    const cardCls = conf ? 'tc-confirmed' : sev === 'err' ? 'tc-err' : sev === 'warn' ? 'tc-warn' : '';
    let badge2 = '';
    if (conf)             badge2 = '<span class="badge block">&#128274; ยืนยัน</span>';
    else if (sev==='err') badge2 = '<span class="badge berr">&#10060; เกินขีด</span>';
    else if (sev==='warn')badge2 = '<span class="badge bwarn">&#9888; เกิน</span>';
    const rs   = window._tripRouteStatus[tid];
    const dist = _distLabel(tid);
    let distHtml = '';
    if      (rs==='loading') distHtml = '<span class="tc-dist" style="color:#9ca3af">&#9203;</span>';
    else if (rs==='error')   distHtml = '<span class="tc-dist dist-err">&#10006; ไม่มีเส้นทาง</span>';
    else if (dist)           distHtml = '<span class="tc-dist">&#128205; ' + dist + '</span>';
    const vt    = t.truck || '6W';
    const vOpts = ['4W','JB','6W'].map(v =>
      '<option value="' + v + '"' + (v===vt?' selected':'') + '>' + v + '</option>'
    ).join('');
    // Branch rows with distance from DC
    const dcLat = DC[0], dcLon = DC[1];
    const brRows = brs.slice(0,8).map(b => {{
      const dkm = b.lat && b.lon ? _haversineKm(dcLat,dcLon,b.lat,b.lon).toFixed(0) + 'km' : '';
      return '<div class="br-row"><span style="font-weight:600;color:#374151">' + b.code + '</span>' +
        '<span style="color:#6b7280;flex:1;margin:0 4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">' + (b.name||'').slice(0,14) + '</span>' +
        '<span class="br-dist">&#128205;' + dkm + '</span>' +
        '<span style="font-size:10px;color:#9ca3af;margin-left:4px">' + b.weight.toFixed(0) + 'kg</span></div>';
    }}).join('') + (brs.length > 8 ? '<div style="font-size:10px;color:#9ca3af;padding-top:4px;text-align:center">+' + (brs.length-8) + ' สาขาอื่น</div>' : '');
    const dRow = s.maxD < 999
      ? '<div class="bar-row"><span class="bar-lbl" title="จุดส่ง">&#128205;</span><div class="bar-track"><div class="bar-fill" style="width:' + _pct(s.dPct) + '%;background:' + _barColor(s.dPct) + '"></div></div><span class="bar-val">' + s.drops + '/' + s.maxD + '</span></div>'
      : '<div style="font-size:11px;color:#6b7280;margin:2px 0 1px;font-weight:600">' + brs.length + ' จุดส่ง</div>';
    const isOpen = _expandedTrips.has(tid);
    const tidN = parseInt(tid);
    const btnConf = conf
      ? '<button class="tc-btn btn-confirm locked" onclick="toggleConfirm(' + tidN + ')">&#128274; ยืนยันแล้ว</button>'
      : '<button class="tc-btn btn-confirm" onclick="toggleConfirm(' + tidN + ')">&#9989; ยืนยันทริป</button>';
    const swapBtnHtml = conf ? '' : (
      _swapSrcTid === tid
        ? '<button class="tc-btn btn-swap-target" onclick="swapModeStart(' + tidN + ')">&#10005; ยกเลิก</button>'
        : _swapSrcTid
          ? '<button class="tc-btn btn-swap-ok" onclick="swapModeStart(' + tidN + ')">&#8644; สลับกับ ' + _swapSrcTid + '</button>'
          : '<button class="tc-btn btn-swap" onclick="swapModeStart(' + tidN + ')">&#8644; สลับ</button>'
    );
    html +=
      '<div class="tc ' + cardCls + '" id="tc-' + tid + '" style="border-left-color:' + col + '">' +
        '<div class="tc-head" onclick="toggleTrip(' + tidN + ')">' +
          '<div class="tc-dot" style="background:' + col + '"></div>' +
          '<div class="tc-main">' +
            '<div class="tc-title" style="color:#0f172a">Trip ' + tid + ' ' + _vehBadge(vt) + ' ' + badge2 + '</div>' +
            '<div class="tc-sub">' + brs.length + ' จุด &nbsp;·&nbsp; ' + s.w.toFixed(0) + '/' + s.maxW.toFixed(0) + 'kg &nbsp;·&nbsp; ' + s.c.toFixed(2) + '/' + s.maxC.toFixed(2) + 'm³</div>' +
          '</div>' +
          '<div class="tc-right">' + distHtml + '<span class="tc-chev ' + (isOpen?'open':'') + '" id="chev-' + tid + '">&#9658;</span></div>' +
        '</div>' +
        '<div class="bars">' +
          '<div class="bar-row"><span class="bar-lbl" title="น้ำหนัก">&#9878;</span><div class="bar-track"><div class="bar-fill" style="width:' + _pct(s.wPct) + '%;background:' + _barColor(s.wPct) + '"></div></div><span class="bar-val">' + s.wPct.toFixed(0) + '%</span></div>' +
          '<div class="bar-row"><span class="bar-lbl" title="คิว">&#128230;</span><div class="bar-track"><div class="bar-fill" style="width:' + _pct(s.cPct) + '%;background:' + _barColor(s.cPct) + '"></div></div><span class="bar-val">' + s.cPct.toFixed(0) + '%</span></div>' +
          dRow +
        '</div>' +
        '<div class="tc-body ' + (isOpen?'open':'') + '" id="tb-' + tid + '">' +
          '<select class="veh-sel"' + (conf?' disabled':'') + ' onchange="changeVehicle(' + tidN + ',this.value)">' + vOpts + '</select>' +
          '<div class="br-list">' + brRows + '</div>' +
          '<div class="tc-btns">' +
            '<button class="tc-btn btn-zoom" onclick="focusTrip(' + parseInt(tid) + ')">&#128506; Zoom</button>' +
            swapBtnHtml +
            btnConf +
          '</div>' +
        '</div>' +
      '</div>';
  }}
  list.innerHTML = html;
  document.getElementById('chip-br').textContent = branches.length + ' สาขา';
  document.getElementById('chip-tr').textContent = tkeys.length + ' ทริป';
  const cw = document.getElementById('chip-w');
  const ce = document.getElementById('chip-e');
  cw.style.display = warnCnt ? '' : 'none'; cw.textContent = '⚠ ' + warnCnt;
  ce.style.display = errCnt  ? '' : 'none'; ce.textContent = '❌ ' + errCnt;
  _updateLegend();
}}

function _updateLegend() {{
  const el = document.getElementById('legend-trips');
  if (!el) return;
  const tkeys = sortedTripKeys();
  el.innerHTML = tkeys.slice(0,20).map(tid => {{
    const col = tripColor(parseInt(tid));
    const t = trips[tid]||{{}};
    const s = tripSummary(tid);
    return '<div class="legend-item"><div class="legend-dot" style="background:' + col + '">' + tid + '</div>' +
      '<span>Trip ' + tid + ' · ' + (t.truck||'6W') + ' · ' + s.drops + 'จุด</span></div>';
  }}).join('') + (tkeys.length > 20 ? '<div style="font-size:10px;color:#9ca3af;margin-top:4px">+' + (tkeys.length-20) + ' ทริปอื่น</div>' : '');
}}

function scrollToTrip(tid) {{
  const el = document.getElementById('tc-' + tid);
  if (el) {{
    el.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
    el.classList.add('tc-active');
    setTimeout(() => el.classList.remove('tc-active'), 1800);
  }}
}}

function toggleTrip(tid) {{
  tid = String(tid);
  const body = document.getElementById('tb-'+tid);
  const chev = document.getElementById('chev-'+tid);
  if (!body) return;
  const open = body.classList.toggle('open');
  if (open) _expandedTrips.add(tid); else _expandedTrips.delete(tid);
  if (chev) chev.classList.toggle('open', open);
}}
function toggleAllTrips() {{
  _allOpen = !_allOpen;
  sortedTripKeys().forEach(tid => {{
    const body = document.getElementById('tb-'+tid);
    const chev = document.getElementById('chev-'+tid);
    if (body) body.classList.toggle('open', _allOpen);
    if (chev) chev.classList.toggle('open', _allOpen);
    if (_allOpen) _expandedTrips.add(tid); else _expandedTrips.delete(tid);
  }});
}}

// ── SETTINGS ──────────────────────────────────────────────────────────────
function toggleSettings() {{
  _settingsOpen = !_settingsOpen;
  document.getElementById('settings-panel').style.display = _settingsOpen ? 'block' : 'none';
  if (_settingsOpen) {{
    document.getElementById('cfg-pt-buf').value = Math.round(_ptBuf * 100);
    document.getElementById('cfg-mm-buf').value = Math.round(_mmBuf * 100);
    if (_legendOpen) {{ _legendOpen = false; document.getElementById('legend-panel').style.display = 'none'; }}
  }}
}}
function applySettings() {{
  const ptVal = parseFloat(document.getElementById('cfg-pt-buf').value) / 100;
  const mmVal = parseFloat(document.getElementById('cfg-mm-buf').value) / 100;
  if (isNaN(ptVal) || ptVal < 0.5 || isNaN(mmVal) || mmVal < 0.5) {{
    showToast('⚠ กรุณาใส่ค่า buffer ที่ถูกต้อง (≥50%)'); return;
  }}
  _ptBuf = ptVal; _mmBuf = mmVal;
  _invalidateCache(); renderSidebar(); renderMarkers();
  toggleSettings();
  showToast('✅ Buffer: Punthai ' + Math.round(_ptBuf*100) + '% · Maxmart ' + Math.round(_mmBuf*100) + '%');
}}

// ── LEGEND ────────────────────────────────────────────────────────────────
function toggleLegend() {{
  _legendOpen = !_legendOpen;
  document.getElementById('legend-panel').style.display = _legendOpen ? 'block' : 'none';
  if (_legendOpen) {{
    _updateLegend();
    if (_settingsOpen) {{ _settingsOpen = false; document.getElementById('settings-panel').style.display = 'none'; }}
  }}
}}

// ── FULLSCREEN ────────────────────────────────────────────────────────────
function toggleFullscreen() {{
  const btn = document.getElementById('fs-btn');
  if (!document.fullscreenElement && !document.webkitFullscreenElement) {{
    const el = document.getElementById('app');
    const fn = el.requestFullscreen || el.webkitRequestFullscreen || el.mozRequestFullScreen || el.msRequestFullscreen;
    if (fn) fn.call(el);
    btn.innerHTML = '&#x26F7;';
    btn.title = 'ย่อจอ';
  }} else {{
    const fn = document.exitFullscreen || document.webkitExitFullscreen || document.mozCancelFullScreen || document.msExitFullscreen;
    if (fn) fn.call(document);
    btn.innerHTML = '&#x26F6;';
    btn.title = 'เต็มจอ';
  }}
  if (map) setTimeout(() => map.invalidateSize(), 150);
}}
document.addEventListener('fullscreenchange', () => {{
  const btn = document.getElementById('fs-btn');
  if (btn) btn.innerHTML = document.fullscreenElement ? '&#x26F7;' : '&#x26F6;';
  if (map) setTimeout(() => map.invalidateSize(), 150);
}});
// Make onclick-callable explicitly global
window.toggleFullscreen = toggleFullscreen;
window.toggleSettings   = toggleSettings;
window.toggleLegend     = toggleLegend;

// ── MARKERS ───────────────────────────────────────────────────────────────
function _mkIcon(b) {{
  const col  = tripColor(b.trip);
  const sel  = _selectedBranches.has(b.code);
  const conf = window._confirmedTrips.has(String(b.trip));
  const over = tripSummary(String(b.trip)).overW || tripSummary(String(b.trip)).overC;
  const bord = sel ? '#fbbf24' : conf ? '#15803d' : over ? '#ef4444' : '#fff';
  const sz   = sel ? 32 : 28;
  const ring = sel ? 'box-shadow:0 0 0 3px #fbbf24,0 2px 8px rgba(0,0,0,.3);' : '0 3px 8px rgba(0,0,0,.25);';
  return L.divIcon({{
    html: '<div style="background:' + col + ';border:2.5px solid ' + bord +
          ';border-radius:50%;width:' + sz + 'px;height:' + sz + 'px;display:flex;' +
          'align-items:center;justify-content:center;font-size:9px;font-weight:800;' +
          'color:#fff;box-shadow:' + ring + 'transition:transform .15s">' +
          b.trip + '</div>',
    iconSize: [sz, sz], iconAnchor: [sz/2, sz/2], className: '',
  }});
}}
function renderMarkers() {{
  if (!map) return;
  Object.values(markers).forEach(m => map.removeLayer(m));
  markers = {{}};
  for (const b of branches) {{
    // Hide markers not in current filter
    if (_filterTrip !== 'ALL' && String(b.trip) !== _filterTrip) continue;
    const mk = L.marker([b.lat, b.lon], {{ icon: _mkIcon(b), zIndexOffset: _selectedBranches.has(b.code) ? 1000 : 0 }});
    mk.bindTooltip('T' + b.trip + ' · ' + b.code + ' ' + (b.name||'').slice(0,15) + ' · ' + b.weight.toFixed(0) + 'kg', {{ direction:'top', offset:[0,-15] }});
    mk.on('click', (e) => {{
      if (_selectMode) {{ L.DomEvent.stop(e); toggleSelectBranch(b.code); }}
      else {{
        showInfo(b.code);
        scrollToTrip(String(b.trip));
      }}
    }});
    mk.addTo(map);
    markers[b.code] = mk;
  }}
}}
function refreshMarker(code) {{
  const b = branches.find(x => x.code === code);
  if (!b || !markers[code]) return;
  markers[code].setIcon(_mkIcon(b));
  markers[code].setZIndexOffset(_selectedBranches.has(code) ? 1000 : 0);
}}
function refreshTripMarkers(tid) {{
  branches.filter(b => String(b.trip) === String(tid)).forEach(b => refreshMarker(b.code));
}}

// ── ROUTES ────────────────────────────────────────────────────────────────
function _clearRoutes() {{
  if (!map) return;
  window._routeLines.forEach(l => map.removeLayer(l));
  window._routeLines = [];
}}
async function _fetchOsrmRoute(waypoints) {{
  let wps = waypoints;
  if (wps.length > 20) {{
    const step = Math.max(1, Math.floor((wps.length-2) / 18));
    const s2 = [wps[0]];
    for (let i = 1; i < wps.length-1; i += step) s2.push(wps[i]);
    s2.push(wps[wps.length-1]);
    wps = s2;
  }}
  const cacheKey = JSON.stringify(wps);
  if (window._osrmCache[cacheKey]) return window._osrmCache[cacheKey];
  const coords = wps.map(([lat,lon]) => lon + ',' + lat).join(';');
  const url = 'https://router.project-osrm.org/route/v1/driving/' + coords + '?overview=full&geometries=geojson';
  try {{
    const ctrl = new AbortController();
    const timeout = setTimeout(() => ctrl.abort(), 12000);
    const resp = await fetch(url, {{ signal: ctrl.signal }});
    clearTimeout(timeout);
    const data = await resp.json();
    if (data.routes && data.routes[0]) {{
      const r = data.routes[0];
      const result = {{ coords: r.geometry.coordinates.map(([lon,lat]) => [lat, lon]), distance_m: r.distance }};
      window._osrmCache[cacheKey] = result; return result;
    }}
  }} catch(e) {{}}
  return null;
}}
async function _drawRoutes() {{
  if (!map) return;
  _clearRoutes();
  const tkeys = sortedTripKeys();
  const ind = document.getElementById('route-ind');
  const prog = document.getElementById('route-prog');
  if (ind) ind.style.display = 'block';
  let done = 0, failed = 0;
  const promises = tkeys.map(async (tid) => {{
    const brs = branches.filter(b => String(b.trip) === tid);
    if (!brs.length) return;
    window._tripRouteStatus[tid] = 'loading';
    const col = tripColor(parseInt(tid));
    // ── ลองใช้ pre-computed route ก่อน (ไม่ต้องเรียก OSRM) ────────────
    let result = null;
    if (TID_ROUTES && TID_ROUTES[tid]) {{
      result = TID_ROUTES[tid];
    }} else {{
      const wps = [DC].concat(brs.map(b => [b.lat, b.lon])).concat([DC]);
      result = await _fetchOsrmRoute(wps);
    }}
    if (!_showRoutes) return;
    if (result && result.coords && result.coords.length > 1) {{
      const line = L.polyline(result.coords, {{ color: col, weight: 4, opacity: 0.82, renderer: _renderer, lineJoin: 'round', lineCap: 'round' }});
      line.bindTooltip('Trip ' + tid + ' · ' + (result.distance_m/1000).toFixed(1) + ' km', {{ sticky: true }});
      line.addTo(map);
      window._routeLines.push(line);
      window._tripDistances[tid]   = result.distance_m / 1000;
      window._tripRouteStatus[tid] = 'ok';
      done++;
    }} else {{
      // Straight-line fallback when OSRM unavailable
      const wps2 = [[DC[0],DC[1]]].concat(brs.map(b => [b.lat,b.lon])).concat([[DC[0],DC[1]]]);
      const line2 = L.polyline(wps2, {{ color:col, weight:2.5, opacity:0.55, dashArray:'7,5' }});
      line2.bindTooltip('Trip ' + tid + ' (เส้นตรง · OSRM unavailable)', {{ sticky:true }});
      line2.addTo(map);
      window._routeLines.push(line2);
      window._tripRouteStatus[tid] = 'fallback';
      failed++;
    }}
    _patchDistChip(tid);
    if (prog) prog.textContent = 'โหลด ' + (done+failed) + '/' + tkeys.length + ' ทริป' + (failed ? ' (⚠ ' + failed + ' ล้มเหลว)' : '');
  }});
  await Promise.all(promises);
  if (ind) ind.style.display = 'none';
  if (_showRoutes) {{
    const btn = document.getElementById('route-btn');
    if (btn) {{ btn.textContent = '&#128739; ซ่อนเส้นทาง' + (failed ? ' ⚠' : ''); btn.classList.add('active'); }}
  }}
  renderSidebar();
}}
function _patchDistChip(tid) {{
  const card = document.getElementById('tc-'+tid);
  if (!card) return;
  const right = card.querySelector('.tc-right');
  if (!right) return;
  const rs = window._tripRouteStatus[tid], dist = _distLabel(tid);
  let el = right.querySelector('.tc-dist');
  if (!el) {{
    el = document.createElement('span'); el.className = 'tc-dist';
    const chev = right.querySelector('.tc-chev');
    right.insertBefore(el, chev);
  }}
  if (rs === 'error') {{ el.textContent = '✕ ไม่มีเส้นทาง'; el.style.color = '#f87171'; }}
  else if (rs === 'fallback') {{ el.textContent = '~ เส้นตรง'; el.style.color = '#d97706'; }}
  else if (dist) {{ el.textContent = '📍 ' + dist; el.style.color = '#60a5fa'; }}
}}
function toggleRoutes() {{
  if (!map) return;
  _showRoutes = !_showRoutes;
  const btn = document.getElementById('route-btn');
  if (_showRoutes) {{
    if (btn) {{ btn.textContent = '⏳ กำลังโหลด...'; btn.classList.add('active'); }}
    _drawRoutes();
  }} else {{
    if (btn) {{ btn.textContent = '&#128739; เส้นทาง'; btn.classList.remove('active'); }}
    _clearRoutes();
    sortedTripKeys().forEach(tid => {{ delete window._tripRouteStatus[tid]; }});
    renderSidebar();
  }}
}}

// ── SELECTION MODE ────────────────────────────────────────────────────────
function setSelVeh(v) {{
  _selVeh = v;
  ['4W','JB','6W'].forEach(x => {{
    const el = document.getElementById('vchip-'+x);
    if (el) el.className = 'veh-chip' + (x===v ? ' vc-active-'+x : '');
  }});
  previewSelTarget();
}}
function toggleSelectMode() {{
  _selectMode = !_selectMode;
  const btn = document.getElementById('select-btn');
  const banner = document.getElementById('mode-banner');
  if (btn) {{
    btn.classList.toggle('active', _selectMode);
    btn.textContent = _selectMode ? '✓ เลือกสาขา' : '◻ เลือกสาขา';
  }}
  if (banner) banner.classList.toggle('visible', _selectMode);
  if (_selectMode) {{
    closeInfo();
    if (map) map.getContainer().style.cursor = 'crosshair';
  }} else {{
    if (_boxRect) {{ map.removeLayer(_boxRect); _boxRect = null; }}
    _boxStart = null;
    if (map && map.dragging && map.getContainer()) {{ map.dragging.enable(); map.getContainer().style.cursor = ''; }}
    clearSelection();
  }}
}}
function _cleanupAndRefresh() {{
  const n = _cleanupEmptyTrips();
  renderMarkers();
  renderSidebar();
  showToast(n ? '\u2705 ล้างทริปว่าง + เรียงใหม่' : 'ℹ️ ไม่มีทริปว่าง');
}}
function toggleSelectBranch(code) {{
  if (_selectedBranches.has(code)) _selectedBranches.delete(code);
  else _selectedBranches.add(code);
  refreshMarker(code);
  updateSelPanel();
}}
function clearSelection() {{
  const prev = new Set(_selectedBranches);
  _selectedBranches.clear();
  _selVeh = null;
  const bg = document.getElementById('sel-panel-bg');
  const pn = document.getElementById('sel-panel');
  if (bg) bg.style.display = 'none';
  if (pn) pn.style.display = 'none';
  prev.forEach(code => refreshMarker(code));
}}
function updateSelPanel() {{
  const panel = document.getElementById('sel-panel');
  const bg    = document.getElementById('sel-panel-bg');
  if (!panel) return;
  const n = _selectedBranches.size;
  if (!n) {{
    panel.style.display = 'none';
    if (bg) bg.style.display = 'none';
    return;
  }}
  panel.style.display = 'block';
  if (bg) bg.style.display = 'block';
  const selBrs  = branches.filter(b => _selectedBranches.has(b.code));
  const totalW  = selBrs.reduce((a,b) => a+b.weight, 0);
  const totalC  = selBrs.reduce((a,b) => a+b.cube, 0);
  document.getElementById('sel-count').textContent = n + ' สาขา';
  document.getElementById('sel-summary').textContent = 'W:'+totalW.toFixed(0)+'kg  C:'+totalC.toFixed(2)+'m³';
  const tripIds = [...new Set(selBrs.map(b => b.trip))].sort((a,b)=>a-b);
  document.getElementById('sel-trips').textContent = '(จาก Trip: ' + tripIds.join(', ') + ')';
  // Distance grid between selected branches
  const distGrid = document.getElementById('sel-dist-grid');
  if (distGrid) {{
    if (selBrs.length >= 2) {{
      let dHtml = '<b style="color:#065f46">ระยะห่าง:</b> ';
      const rows = [];
      for (let i = 0; i < Math.min(selBrs.length - 1, 4); i++) {{
        const d = _haversineKm(selBrs[i].lat, selBrs[i].lon, selBrs[i+1].lat, selBrs[i+1].lon).toFixed(1);
        rows.push('<span style="color:#374151;font-weight:600">' + selBrs[i].code + '</span>↔<span style="color:#374151;font-weight:600">' + selBrs[i+1].code + '</span>: <span style="color:#059669;font-weight:700">' + d + ' km</span>');
      }}
      if (selBrs.length >= 2) {{
        const dDC = selBrs.map(b => '<span style="color:#374151;font-weight:600">' + b.code + '</span>←DC:<span style="color:#059669;font-weight:700">' + _haversineKm(DC[0],DC[1],b.lat,b.lon).toFixed(0) + 'km</span>').slice(0,4).join('  ');
        dHtml += rows.join('  ') + '<br>' + dDC;
      }}
      distGrid.innerHTML = dHtml;
      distGrid.classList.add('show');
    }} else {{
      distGrid.classList.remove('show');
    }}
  }}
  // Set vehicle chip to match source trip by default
  if (!_selVeh) {{
    const srcVeh = selBrs[0]?.vtype || '6W';
    _selVeh = srcVeh;
    ['4W','JB','6W'].forEach(x => {{
      const el = document.getElementById('vchip-'+x);
      if (el) el.className = 'veh-chip' + (x===srcVeh ? ' vc-active-'+x : '');
    }});
    _selVeh = null; // don't override yet, just highlight default
  }}
  const selEl = document.getElementById('sel-target');
  if (!selEl) return;
  const prev  = selEl.value;
  selEl.innerHTML = '';
  for (const tid of sortedTripKeys()) {{
    const s = tripSummary(tid), t = trips[tid]||{{}};
    const locked = window._confirmedTrips.has(tid);
    const willOver = (s.w + totalW > s.maxW || s.c + totalC > s.maxC);
    const opt = document.createElement('option');
    opt.value = tid;
    opt.textContent = 'Trip ' + tid + ' (' + (t.truck||'6W') + ')  '
      + s.drops + 'จุด  '
      + s.w.toFixed(0) + '/' + s.maxW.toFixed(0) + 'kg'
      + (willOver ? '  ⚠' : ' ✓') + (locked ? '  🔒' : '');
    if (locked) opt.disabled = true;
    selEl.appendChild(opt);
  }}
  const nextId = Math.max.apply(null, sortedTripKeys().map(Number)) + 1;
  const newOpt = document.createElement('option');
  newOpt.value = 'NEW'; newOpt.textContent = '+ สร้าง Trip ใหม่ (Trip ' + nextId + ')';
  selEl.appendChild(newOpt);
  if (prev) selEl.value = prev;
  previewSelTarget();
  // ── Build move trip-buttons grid ─────────────────────────────────────
  const moveBtns = document.getElementById('sel-move-btns');
  if (moveBtns) {{
    moveBtns.innerHTML = '';
    const selBrs2 = branches.filter(b => _selectedBranches.has(b.code));
    const addW2 = selBrs2.reduce((a,b)=>a+b.weight,0);
    const addC2 = selBrs2.reduce((a,b)=>a+b.cube,0);
    for (const tid of sortedTripKeys()) {{
      const t = trips[tid]||{{}}, sm = tripSummary(tid);
      const col = tripColor(parseInt(tid));
      const locked = window._confirmedTrips.has(tid);
      const willOver = (sm.w + addW2 > sm.maxW || sm.c + addC2 > sm.maxC);
      const btn = document.createElement('button');
      btn.className = 'tqb' + (locked ? ' tqb-locked' : '') + (willOver ? ' tqb-over' : '');
      btn.style.background = col;
      btn.innerHTML = 'Trip ' + tid + '<span class="tqb-sub">' + (t.truck||'6W') + ' ' + sm.wPct.toFixed(0) + '%W</span>';
      if (!locked) btn.onclick = (function(t){{ return function(){{
        const sel2 = document.getElementById('sel-target'); if (sel2) sel2.value = t;
        moveSelected();
      }}; }})(tid);
      moveBtns.appendChild(btn);
    }}
    const newMBtn = document.createElement('button');
    newMBtn.className = 'tqb tqb-new';
    newMBtn.innerHTML = '+ ใหม่<span class="tqb-sub">Trip ' + nextId + '</span>';
    newMBtn.onclick = function(){{ const sel2=document.getElementById('sel-target'); if(sel2) sel2.value='NEW'; moveSelected(); }};
    moveBtns.appendChild(newMBtn);
  }}
  // ── Build swap trip-buttons grid (only if all from 1 trip) ───────────
  const selBrsSwap = branches.filter(b => _selectedBranches.has(b.code));
  const srcTripsSwap = [...new Set(selBrsSwap.map(b => String(b.trip)))];
  const swapRow = document.getElementById('sel-swap-row');
  const swapBtns = document.getElementById('sel-swap-btns');
  if (swapRow && swapBtns) {{
    if (srcTripsSwap.length === 1) {{
      const srcTid = srcTripsSwap[0];
      swapBtns.innerHTML = '';
      for (const tid of sortedTripKeys()) {{
        if (tid === srcTid) continue;
        const t = trips[tid]||{{}}, sm = tripSummary(tid);
        const col = tripColor(parseInt(tid));
        const locked = window._confirmedTrips.has(tid);
        const btn = document.createElement('button');
        btn.className = 'tqb' + (locked ? ' tqb-locked' : '');
        btn.style.background = col;
        btn.innerHTML = 'Trip ' + tid + '<span class="tqb-sub">' + (t.truck||'6W') + ' ' + sm.drops + 'จุด</span>';
        if (!locked) btn.onclick = (function(t){{ return function(){{
          const sel2 = document.getElementById('sel-target'); if (sel2) sel2.value = t;
          swapSelected();
        }}; }})(tid);
        swapBtns.appendChild(btn);
      }}
      swapRow.style.display = '';
    }} else {{
      swapRow.style.display = 'none';
    }}
  }}
}}
function previewSelTarget() {{
  const warnEl = document.getElementById('sel-warn');
  if (!warnEl) return;
  const selEl = document.getElementById('sel-target');
  if (!selEl) return;
  const tid = selEl.value;
  // Update vehicle hint
  const hint = document.getElementById('sel-veh-hint');
  if (hint && tid && tid !== 'NEW' && _selVeh) {{
    const t = trips[tid]||{{}};
    hint.textContent = _selVeh !== (t.truck||'6W') ? '\u26a0 จะเปลี่ยนรถของ Trip '+tid+' เป็น '+_selVeh : '';
  }} else if (hint) hint.textContent = '';
  if (!tid || tid === 'NEW') {{ warnEl.textContent = ''; warnEl.classList.remove('show'); return; }}
  const s = tripSummary(tid);
  const selBrs = branches.filter(b => _selectedBranches.has(b.code));
  if (!selBrs.length) {{ warnEl.textContent = ''; warnEl.classList.remove('show'); return; }}
  const addW = selBrs.reduce((a,b)=>a+b.weight,0);
  const addC = selBrs.reduce((a,b)=>a+b.cube,0);
  const newW = s.w+addW, newC = s.c+addC, newD = s.drops+selBrs.length;
  const msgs = [];
  if (newW > s.maxW) msgs.push('⚠ น้ำหนัก: ' + newW.toFixed(0) + '/' + s.maxW.toFixed(0) + ' kg (เกิน ' + (newW-s.maxW).toFixed(0) + 'kg)');
  if (newC > s.maxC) msgs.push('⚠ คิว: ' + newC.toFixed(2) + '/' + s.maxC.toFixed(2) + ' m³');
  if (newD > s.maxD) msgs.push('⚠ จุดส่ง: ' + newD + '/' + s.maxD);
  if (msgs.length) {{ warnEl.textContent = msgs.join('   '); warnEl.classList.add('show'); }}
  else {{ warnEl.textContent = ''; warnEl.classList.remove('show'); }}
}}
async function moveSelected() {{
  let targetTid = document.getElementById('sel-target').value;
  if (!targetTid || !_selectedBranches.size) return;
  if (targetTid === 'NEW') {{
    const nextId = Math.max.apply(null, sortedTripKeys().map(Number)) + 1;
    targetTid = String(nextId);
    const veh = _selVeh || (branches.find(b => _selectedBranches.has(b.code))||{{}}).vtype || '6W';
    trips[targetTid] = {{ trip: parseInt(targetTid), truck: veh,
                         tripNo: 'T' + String(targetTid).padStart(3,'0'), branches: 0 }};
    _invalidateCache();
  }} else if (_selVeh) {{
    // Apply vehicle change to target trip if user selected one
    changeVehicle(targetTid, _selVeh);
  }}
  if (window._confirmedTrips.has(targetTid)) {{ showToast('🔒 ทริปปลายทางถูกยืนยันแล้ว'); return; }}
  const chkBrs = branches.filter(b => _selectedBranches.has(b.code));
  const s0 = tripSummary(targetTid);
  const addW0 = chkBrs.reduce((a,b)=>a+b.weight,0), addC0 = chkBrs.reduce((a,b)=>a+b.cube,0);
  const newW0 = s0.w+addW0, newC0 = s0.c+addC0, newD0 = s0.drops+chkBrs.length;
  const warns0 = [];
  if (newW0 > s0.maxW) warns0.push('น้ำหนัก ' + newW0.toFixed(0) + '/' + s0.maxW.toFixed(0) + ' kg');
  if (newC0 > s0.maxC) warns0.push('คิว ' + newC0.toFixed(2) + '/' + s0.maxC.toFixed(2) + ' m³');
  if (newD0 > s0.maxD) warns0.push('จุดส่ง ' + newD0 + '/' + s0.maxD);
  if (warns0.length) {{
    const ok0 = await _askConfirm('⚠ ทริป ' + targetTid + ' จะเกินขีดจำกัด: ' + warns0.join(' + ') + '  \u2014  ต้องการย้ายต่อไปหรือไม่?');
    if (!ok0) return;
  }}
  const srcTrips = new Set(chkBrs.map(b => String(b.trip)));
  const n = _selectedBranches.size;
  chkBrs.forEach(b => {{ b.trip = parseInt(targetTid); }});
  _selectedBranches.clear(); _invalidateCache();
  srcTrips.forEach(tid => refreshTripMarkers(tid));
  refreshTripMarkers(targetTid);
  const renumbered = _cleanupEmptyTrips();
  if (renumbered) renderMarkers();
  renderSidebar(); updateSelPanel();
  showToast('✅ ย้าย ' + n + ' สาขา → Trip ' + targetTid);
}}
async function swapSelected() {{
  const targetTid = document.getElementById('sel-target').value;
  if (!targetTid || targetTid === 'NEW' || !_selectedBranches.size) {{
    showToast('⚠ กรุณาเลือกทริปปลายทาง (ไม่รองรับสร้างใหม่)');
    return;
  }}
  if (window._confirmedTrips.has(targetTid)) {{ showToast('🔒 ทริปปลายทางถูกยืนยันแล้ว'); return; }}
  const selArr  = [..._selectedBranches];
  const srcBrs  = branches.filter(b => selArr.includes(b.code));
  const srcTrips = [...new Set(srcBrs.map(b => String(b.trip)))];
  if (srcTrips.length !== 1) {{
    document.getElementById('sel-warn').textContent = '⚠ สลับได้เฉพาะสาขาจาก 1 ทริปเท่านั้น';
    return;
  }}
  const srcTid = srcTrips[0];
  if (srcTid === String(targetTid)) {{ showToast('⚠ ทริปต้นทางและปลายทางเหมือนกัน'); return; }}
  if (window._confirmedTrips.has(srcTid)) {{ showToast('🔒 ทริปต้นทางถูกยืนยันแล้ว'); return; }}
  const tgtBrs = branches.filter(b => String(b.trip) === String(targetTid));
  const n1 = selArr.length, n2 = tgtBrs.length;
  const sSrc = tripSummary(srcTid), sTgt = tripSummary(targetTid);
  const srcW = srcBrs.reduce((a,b)=>a+b.weight,0), srcC = srcBrs.reduce((a,b)=>a+b.cube,0);
  const tgtW = tgtBrs.reduce((a,b)=>a+b.weight,0), tgtC = tgtBrs.reduce((a,b)=>a+b.cube,0);
  const swW = [];
  if (srcW > sTgt.maxW) swW.push('Trip '+targetTid+' น้ำหนัก '+srcW.toFixed(0)+'/'+sTgt.maxW.toFixed(0)+'kg');
  if (srcC > sTgt.maxC) swW.push('Trip '+targetTid+' คิว '+srcC.toFixed(2)+'/'+sTgt.maxC.toFixed(2)+'m³');
  if (tgtW > sSrc.maxW) swW.push('Trip '+srcTid+' น้ำหนัก '+tgtW.toFixed(0)+'/'+sSrc.maxW.toFixed(0)+'kg');
  if (tgtC > sSrc.maxC) swW.push('Trip '+srcTid+' คิว '+tgtC.toFixed(2)+'/'+sSrc.maxC.toFixed(2)+'m³');
  if (swW.length) {{
    const okSw = await _askConfirm('⚠ เกินขีดจำกัดหลังสลับ: ' + swW.join(' + ') + '  \u2014  ต้องการสลับต่อไปหรือไม่?');
    if (!okSw) return;
  }}
  srcBrs.forEach(b => b.trip = parseInt(targetTid));
  tgtBrs.forEach(b => b.trip = parseInt(srcTid));
  _selectedBranches.clear(); _invalidateCache();
  refreshTripMarkers(srcTid); refreshTripMarkers(targetTid);
  if (_cleanupEmptyTrips()) renderMarkers();
  renderSidebar(); updateSelPanel();
  showToast('⇄ สลับ ' + n1 + ' สาขา (Trip ' + srcTid + ') ↔ ' + n2 + ' สาขา (Trip ' + targetTid + ')');
}}

// ── QUICK SWAP TWO WHOLE TRIPS (no prior selection needed) ────────────────
async function quickSwapTrips(tidA, tidB) {{
  tidA = String(tidA); tidB = String(tidB);
  if (tidA === tidB) {{ showToast('⚠ ทริปเหมือนกัน'); return; }}
  if (window._confirmedTrips.has(tidA)) {{ showToast('🔒 Trip ' + tidA + ' ถูกยืนยันแล้ว'); return; }}
  if (window._confirmedTrips.has(tidB)) {{ showToast('🔒 Trip ' + tidB + ' ถูกยืนยันแล้ว'); return; }}
  const aBrs = branches.filter(b => String(b.trip) === tidA);
  const bBrs = branches.filter(b => String(b.trip) === tidB);
  const sA = tripSummary(tidA), sB = tripSummary(tidB);
  const aW = aBrs.reduce((x,b)=>x+b.weight,0), aC = aBrs.reduce((x,b)=>x+b.cube,0);
  const bW = bBrs.reduce((x,b)=>x+b.weight,0), bC = bBrs.reduce((x,b)=>x+b.cube,0);
  const warns = [];
  if (bW > sA.maxW) warns.push('Trip '+tidA+' น้ำหนัก '+bW.toFixed(0)+'/'+sA.maxW.toFixed(0)+'kg');
  if (bC > sA.maxC) warns.push('Trip '+tidA+' คิว '+bC.toFixed(2)+'/'+sA.maxC.toFixed(2)+'m³');
  if (aW > sB.maxW) warns.push('Trip '+tidB+' น้ำหนัก '+aW.toFixed(0)+'/'+sB.maxW.toFixed(0)+'kg');
  if (aC > sB.maxC) warns.push('Trip '+tidB+' คิว '+aC.toFixed(2)+'/'+sB.maxC.toFixed(2)+'m³');
  if (warns.length) {{
    const ok = await _askConfirm('⚠ เกินขีดจำกัดหลังสลับ: ' + warns.join(' · ') + ' — ต้องการสลับต่อไปหรือไม่?');
    if (!ok) return;
  }}
  aBrs.forEach(b => b.trip = parseInt(tidB));
  bBrs.forEach(b => b.trip = parseInt(tidA));
  _invalidateCache();
  refreshTripMarkers(tidA); refreshTripMarkers(tidB);
  if (_cleanupEmptyTrips()) renderMarkers();
  renderSidebar();
  showToast('⇄ สลับ Trip ' + tidA + ' (' + aBrs.length + 'จุด) ↔ Trip ' + tidB + ' (' + bBrs.length + 'จุด)');
}}

// ── SWAP MODE (sidebar two-click swap) ────────────────────────────────────
function swapModeStart(tid) {{
  tid = String(tid);
  if (_swapSrcTid) {{
    if (_swapSrcTid === tid) {{
      // Cancel swap mode
      _swapSrcTid = null;
      renderSidebar();
      showToast('ยกเลิกโหมดสลับ');
    }} else {{
      const src = _swapSrcTid;
      _swapSrcTid = null;
      quickSwapTrips(src, tid);
    }}
  }} else {{
    _swapSrcTid = tid;
    renderSidebar();
    showToast('✓ Trip ' + tid + ' — คลิก ⇄ สลับ บน Trip ที่ต้องการสลับด้วย (คลิกซ้ำเพื่อยกเลิก)');
  }}
}}

// ── CLEANUP & RENUMBER TRIPS ──────────────────────────────────────────────
function _cleanupEmptyTrips() {{
  const emptyTids = Object.keys(trips).filter(tid => !branches.some(b => String(b.trip) === tid));
  if (!emptyTids.length) return false;
  emptyTids.forEach(tid => {{ delete trips[tid]; window._confirmedTrips.delete(tid); }});
  const remaining = Object.keys(trips).map(Number).sort((a,b)=>a-b);
  const hasGaps = remaining.some((num,i) => num !== i+1);
  if (hasGaps) {{
    const remap = {{}};
    remaining.forEach((oldNum,i) => {{ remap[String(oldNum)] = String(i+1); }});
    const newTrips = {{}};
    remaining.forEach(oldNum => {{
      const o = String(oldNum), nt = remap[o];
      newTrips[nt] = {{ ...trips[o], trip: parseInt(nt), tripNo: 'T'+nt.padStart(3,'0') }};
    }});
    trips = newTrips;
    branches.forEach(b => {{ const nt = remap[String(b.trip)]; if (nt) b.trip = parseInt(nt); }});
    window._confirmedTrips = new Set([...window._confirmedTrips].map(t => remap[t]).filter(Boolean));
    _expandedTrips = new Set([..._expandedTrips].map(t => remap[t]).filter(Boolean));
    if (_showRoutes) {{
      _showRoutes = false; _clearRoutes();
      const rbtn = document.getElementById('route-btn');
      if (rbtn) {{ rbtn.textContent = '&#128739; เส้นทาง'; rbtn.classList.remove('active'); }}
    }}
    window._osrmCache = {{}}; window._tripDistances = {{}}; window._tripRouteStatus = {{}};
    showToast('🔢 เรียงเลขทริปใหม่ ' + remaining.length + ' ทริป (ลบทริปว่าง ' + emptyTids.length + ' ทริป)');
  }}
  _invalidateCache();
  return true;
}}

// ── INFO PANEL ────────────────────────────────────────────────────────────
function showInfo(code) {{
  selectedCode = code;
  const b = branches.find(x => x.code === code);
  if (!b) return;
  document.getElementById('ip-code').textContent   = b.code + ' — ' + b.name;
  document.getElementById('ip-detail').textContent =
    b.district + ' ' + b.province + ' · Trip ' + b.trip +
    ' · W:' + b.weight.toFixed(1) + 'kg C:' + b.cube.toFixed(2) + 'm³ · ' + b.maxVeh;
  const srcConf = window._confirmedTrips.has(String(b.trip));
  const sel = document.getElementById('move-sel');
  sel.disabled = srcConf;
  sel.innerHTML = '';
  for (const tid of sortedTripKeys()) {{
    const t = trips[tid] || {{}}, s = tripSummary(tid);
    const locked = window._confirmedTrips.has(tid);
    const opt = document.createElement('option');
    opt.value = tid;
    opt.textContent = 'Trip ' + tid + ' (' + (t.truck||'6W') + ') ' +
      s.drops + 'จุด W' + s.w.toFixed(0) + '/' + s.maxW.toFixed(0) +
      ((s.overW||s.overC||s.overD) ? ' ⚠' : '') + (locked ? ' 🔒' : '');
    if (locked && String(b.trip) !== tid) opt.disabled = true;
    if (String(b.trip) === tid) opt.selected = true;
    sel.appendChild(opt);
  }}
  const nextId = Math.max.apply(null, sortedTripKeys().map(Number)) + 1;
  const newOpt = document.createElement('option');
  newOpt.value = 'NEW';
  newOpt.textContent = '+ สร้าง Trip ใหม่ (Trip ' + nextId + ')';
  sel.appendChild(newOpt);
  // ── Build trip quick-buttons grid ──────────────────────────────────────
  const btnsEl = document.getElementById('ip-trip-btns');
  if (btnsEl) {{
    btnsEl.innerHTML = '';
    for (const tid of sortedTripKeys()) {{
      const t = trips[tid]||{{}}, sm = tripSummary(tid);
      const col = tripColor(parseInt(tid));
      const isCurrent = String(b.trip) === tid;
      const locked = window._confirmedTrips.has(tid);
      const willOver = !isCurrent && (sm.w + b.weight > sm.maxW || sm.c + b.cube > sm.maxC);
      const btn = document.createElement('button');
      btn.className = 'tqb' + (isCurrent ? ' tqb-current' : '') + (locked && !isCurrent ? ' tqb-locked' : '') + (willOver ? ' tqb-over' : '');
      btn.style.background = col;
      btn.innerHTML = 'Trip ' + tid + '<span class="tqb-sub">' + (t.truck||'6W') + ' ' + sm.wPct.toFixed(0) + '%W</span>';
      if (!isCurrent && !locked && !srcConf) {{
        btn.onclick = (function(t){{ return function(){{
          document.getElementById('move-sel').value = t;
          previewMove();
          confirmMove();
        }}; }})(tid);
      }}
      btnsEl.appendChild(btn);
    }}
    const newBtn = document.createElement('button');
    newBtn.className = 'tqb tqb-new';
    newBtn.innerHTML = '+ ใหม่<span class="tqb-sub">Trip ' + nextId + '</span>';
    if (!srcConf) newBtn.onclick = function(){{ document.getElementById('move-sel').value = 'NEW'; confirmMove(); }};
    btnsEl.appendChild(newBtn);
  }}
  const wEl = document.getElementById('move-warn');
  if (srcConf) {{ wEl.textContent = '🔒 ทริปนี้ถูกยืนยันแล้ว ไม่สามารถย้ายสาขาออกได้'; }}
  else {{ wEl.textContent = ''; sel.onchange = () => previewMove(); previewMove(); }}
  const panel = document.getElementById('info-panel');
  panel.style.display = 'block';
  requestAnimationFrame(() => panel.classList.add('visible'));
  refreshMarker(code);
  if (map) map.panTo([b.lat, b.lon]);
}}
function previewMove() {{
  const b = branches.find(x => x.code === selectedCode);
  if (!b) return;
  const sel = document.getElementById('move-sel');
  const tid = sel.value;
  const wEl = document.getElementById('move-warn');
  if (tid === 'NEW' || tid === String(b.trip)) {{ wEl.textContent = ''; return; }}
  if (window._confirmedTrips.has(tid)) {{ wEl.textContent = '🔒 ทริปปลายทางถูกยืนยันแล้ว'; return; }}
  const s  = tripSummary(tid);
  const nw = s.w + b.weight, nc = s.c + b.cube, nd = s.drops + 1;
  const msgs = [];
  if (nw > s.maxW) msgs.push('⚠ น้ำหนัก: ' + nw.toFixed(0) + '/' + s.maxW.toFixed(0) + 'kg');
  if (nc > s.maxC) msgs.push('⚠ คิว: ' + nc.toFixed(2) + '/' + s.maxC.toFixed(2) + 'm³');
  if (nd > s.maxD) msgs.push('⚠ จุดส่ง: ' + nd + '/' + s.maxD);
  wEl.textContent = msgs.join('  ');
}}
async function confirmMove() {{
  const b = branches.find(x => x.code === selectedCode);
  if (!b) return;
  if (window._confirmedTrips.has(String(b.trip))) {{ showToast('🔒 ทริปต้นทางถูกยืนยันแล้ว'); return; }}
  const sel = document.getElementById('move-sel');
  let targetTid = sel.value;
  if (window._confirmedTrips.has(targetTid)) {{ showToast('🔒 ทริปปลายทางถูกยืนยันแล้ว'); return; }}
  if (targetTid === 'NEW') {{
    const nextId = Math.max.apply(null, sortedTripKeys().map(Number)) + 1;
    targetTid = String(nextId);
    trips[targetTid] = {{ trip: nextId, truck: b.vtype, tripNo: 'T' + String(nextId).padStart(3,'0'), branches: 0 }};
  }}
  const warnMsg = document.getElementById('move-warn').textContent;
  if (warnMsg) {{
    const ok = await _askConfirm(warnMsg + '\\n\\nต้องการย้ายต่อไปหรือไม่?');
    if (!ok) return;
  }}
  const oldTrip = String(b.trip);
  b.trip = parseInt(targetTid);
  _invalidateCache();
  refreshTripMarkers(oldTrip); refreshTripMarkers(targetTid); refreshMarker(b.code);
  renderSidebar();
  if (_showRoutes) {{
    delete window._tripDistances[oldTrip]; delete window._tripDistances[targetTid];
    delete window._tripRouteStatus[oldTrip]; delete window._tripRouteStatus[targetTid];
    _showRoutes = false;
    const btn = document.getElementById('route-btn');
    if (btn) {{ btn.textContent = '&#128739; เส้นทาง'; btn.classList.remove('active'); }}
    _clearRoutes();
  }}
  closeInfo();
  showToast('✅ ย้าย ' + b.code + ' → Trip ' + targetTid);
}}
function closeInfo() {{
  const panel = document.getElementById('info-panel');
  panel.classList.remove('visible');
  setTimeout(() => {{ panel.style.display = 'none'; }}, 230);
  if (selectedCode) {{ refreshMarker(selectedCode); selectedCode = null; }}
}}

// ── VEHICLE CHANGE ────────────────────────────────────────────────────────
function changeVehicle(tid, vtype) {{
  tid = String(tid);
  if (trips[tid]) trips[tid].truck = vtype;
  branches.filter(b => String(b.trip) === tid).forEach(b => b.vtype = vtype);
  _invalidateCache(); renderSidebar(); refreshTripMarkers(tid);
  showToast('🚚 Trip ' + tid + ' → ' + vtype);
}}

// ── TRIP CONFIRMATION ─────────────────────────────────────────────────────
async function toggleConfirm(tid) {{
  tid = String(tid);
  if (window._confirmedTrips.has(tid)) {{
    const ok = await _askConfirm('ยืนยันการยกเลิกการล็อก Trip ' + tid + '?');
    if (!ok) return;
    window._confirmedTrips.delete(tid);
    showToast('🔓 ยกเลิกยืนยัน Trip ' + tid);
  }} else {{
    window._confirmedTrips.add(tid);
    showToast('✅ ยืนยัน Trip ' + tid + ' เรียบร้อย');
  }}
  renderSidebar(); refreshTripMarkers(tid);
}}

// ── FOCUS / FIT ───────────────────────────────────────────────────────────
function focusTrip(tripId) {{
  if (!map) return;
  const brs = branches.filter(b => b.trip === tripId);
  if (!brs.length) return;
  map.fitBounds(L.latLngBounds(brs.map(b => [b.lat, b.lon])), {{ padding: [30,30] }});
}}
function fitAll() {{
  if (!map || !branches.length) return;
  map.fitBounds(L.latLngBounds(branches.map(b => [b.lat, b.lon])), {{ padding: [20,20] }});
}}

// ── SEARCH ────────────────────────────────────────────────────────────────
(function() {{
  const _si = document.getElementById('search-inp');
  if (_si) _si.addEventListener('input', function() {{
    if (!map) return;
    const q = this.value.trim().toLowerCase();
    if (!q) return;
    const found = branches.find(b => (b.code+'').toLowerCase().includes(q) || (b.name+'').toLowerCase().includes(q));
    if (found) {{ map.setView([found.lat, found.lon], 14); showInfo(found.code); }}
  }});
}})();

// ── CUSTOM CONFIRM DIALOG ─────────────────────────────────────────────────
let _confirmResolve = null;
function _askConfirm(msg) {{
  return new Promise(resolve => {{
    document.getElementById('confirm-msg').textContent = msg;
    document.getElementById('confirm-overlay').classList.add('show');
    _confirmResolve = function(v) {{
      document.getElementById('confirm-overlay').classList.remove('show');
      resolve(v);
    }};
  }});
}}

// ── TOAST ─────────────────────────────────────────────────────────────────
let _toastTimer;
function showToast(msg) {{
  const t = document.getElementById('toast');
  t.textContent = msg; t.style.display = 'block';
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.style.display = 'none', 2800);
}}

// ── EXPORT ────────────────────────────────────────────────────────────────
function exportExcel() {{ _loadXlsx(() => _doExport()); }}
async function _doExport() {{
  // Styles — matching Python xlsxwriter format exactly
  const THIN = {{style:'thin'}};
  const BRD  = {{top:THIN,left:THIN,bottom:THIN,right:THIN}};
  const HDR_FILL = {{type:'pattern',pattern:'solid',fgColor:{{argb:'FFD9D9D9'}}}};
  const YFILL    = {{type:'pattern',pattern:'solid',fgColor:{{argb:'FFFFE699'}}}};
  const WFILL    = {{type:'pattern',pattern:'solid',fgColor:{{argb:'FFFFFFFF'}}}};
  const RED_FONT = {{color:{{argb:'FFFF0000'}},bold:true}};
  const NRM_FONT = {{color:{{argb:'FF000000'}},bold:false}};
  const ORA_FONT = {{color:{{argb:'FFCD5C11'}},bold:true}};
  const NUM_FMT  = '#,##0.00';

  // Per-trip alternating fill + under-load flag
  let _yel = true;
  const _tc = {{}};
  for (const tid of sortedTripKeys()) {{
    const s = tripSummary(tid);
    _tc[tid] = {{fill: _yel ? YFILL : WFILL, failed: s.wPct<90 && s.cPct<90}};
    _yel = !_yel;
  }}

  const wb = new ExcelJS.Workbook();
  wb.creator = 'TripMap';

  // ══ Sheet 1: 2.Punthai — คอลัมน์เหมือนกัน Python xlsxwriter ════════════
  const ws1 = wb.addWorksheet('2.Punthai');
  ws1.columns = [
    {{header:'Sep.',        key:'sep',   width:6}},
    {{header:'BU',          key:'bu',    width:6}},
    {{header:'รหัสสาขา',   key:'code',  width:12}},
    {{header:'รหัส WMS',   key:'wms',   width:12}},
    {{header:'สาขา',       key:'name',  width:30}},
    {{header:'ตำบล',       key:'sub',   width:14}},
    {{header:'อำเภอ',      key:'dist',  width:14}},
    {{header:'จังหวัด',    key:'prov',  width:16}},
    {{header:'Route',       key:'route', width:12}},
    {{header:'Total Cube',  key:'cube',  width:11}},
    {{header:'Total Wgt',   key:'wgt',   width:11}},
    {{header:'Original QTY',key:'qty',   width:12}},
    {{header:'Trip',        key:'trip',  width:6}},
    {{header:'Trip no',     key:'tripno',width:10}},
  ];
  const h1 = ws1.getRow(1);
  h1.height = 18;
  h1.eachCell(c => {{
    c.fill=HDR_FILL; c.font={{bold:true}}; c.border=BRD;
    c.alignment={{horizontal:'center',wrapText:true}};
  }});

  // Sort branches in trip order
  const _tOrder = {{}};
  sortedTripKeys().forEach((tid,i) => {{ _tOrder[tid]=i; }});
  const _brSorted = branches.slice().sort((a,b) => (_tOrder[String(a.trip)]||0)-(_tOrder[String(b.trip)]||0));

  let sepNum = 1;
  for (const b of _brSorted) {{
    const tid  = String(b.trip);
    const t    = trips[tid]||{{}};
    const tc   = _tc[tid]||{{fill:WFILL,failed:false}};
    const font = tc.failed ? RED_FONT : NRM_FONT;
    const r = ws1.addRow({{
      sep:    sepNum++,
      bu:     b.bu||'211',
      code:   b.code,
      wms:    b.code,
      name:   b.name,
      sub:    b.subdistrict||'',
      dist:   b.district||'',
      prov:   b.province||'',
      route:  b.route||'',
      cube:   +b.cube.toFixed(2),
      wgt:    +b.weight.toFixed(2),
      qty:    b.origQty||0,
      trip:   b.trip,
      tripno: t.tripNo||('T'+tid.padStart(3,'0')),
    }});
    r.eachCell({{includeEmpty:true}}, (c,col) => {{
      c.fill=tc.fill; c.border=BRD; c.font=font;
      if (col===10||col===11) c.numFmt=NUM_FMT;  // Total Cube, Total Wgt
    }});
  }}
  ws1.views=[{{state:'frozen',ySplit:1}}];

  // ══ Sheet 2: สรุปทริป ════════════════════════════════════════════════════
  const ws2 = wb.addWorksheet('สรุปทริป');
  ws2.columns = [
    {{key:'a',width:6}},  {{key:'b',width:11}}, {{key:'c',width:9}},
    {{key:'d',width:9}},  {{key:'e',width:17}}, {{key:'f',width:13}},
    {{key:'g',width:11}}, {{key:'h',width:11}}, {{key:'i',width:8}},
    {{key:'j',width:8}},  {{key:'k',width:13}}, {{key:'l',width:11}}, {{key:'m',width:12}},
  ];
  const h2 = ws2.addRow(['Trip','ทริป No','ประเภทรถ','จำนวนสาขา','น้ำหนักรวม(kg)','คิวรวม(m³)','Max W(kg)','Max C(m³)','%W','%C','ระยะทาง(km)','เส้นทาง','สถานะ']);
  h2.height = 22;
  h2.eachCell(c => {{ c.fill=HDR_FILL; c.font={{bold:true}}; c.border=BRD; c.alignment={{horizontal:'center',wrapText:true}}; }});
  for (const tid of sortedTripKeys()) {{
    const t=trips[tid]||{{}}, s=tripSummary(tid);
    const conf=window._confirmedTrips.has(tid);
    const dist=window._tripDistances[tid];
    const rs=window._tripRouteStatus[tid];
    const tc=_tc[tid]||{{fill:WFILL,failed:false}};
    const r2 = ws2.addRow([
      parseInt(tid), t.tripNo||('T'+tid.padStart(3,'0')), t.truck||'6W',
      s.drops,
      +s.w.toFixed(2), +s.c.toFixed(3), +s.maxW.toFixed(2), +s.maxC.toFixed(3),
      +s.wPct.toFixed(1), +s.cPct.toFixed(1),
      dist?+dist.toFixed(1):'',
      rs==='ok'?'เส้นจริง':rs==='fallback'?'เส้นตรง':'-',
      conf?'✅ ยืนยัน':(s.overW||s.overC||s.overD?'⚠ เกิน':'ปกติ'),
    ]);
    r2.eachCell({{includeEmpty:true}}, (c,col) => {{
      c.fill=tc.fill; c.border=BRD;
      if(tc.failed) c.font=RED_FONT;
      if(col>=5&&col<=8) c.numFmt='#,##0.00';
      if(col===9||col===10) {{ c.numFmt='#,##0.0'; if(col===9&&s.overW) c.font=ORA_FONT; if(col===10&&s.overC) c.font=ORA_FONT; }}
    }});
  }}
  ws2.views=[{{state:'frozen',ySplit:1}}];

  // ─ Download
  const _n=new Date();
  const _ts=_n.getFullYear()+String(_n.getMonth()+1).padStart(2,'0')+String(_n.getDate()).padStart(2,'0')+'_'+String(_n.getHours()).padStart(2,'0')+String(_n.getMinutes()).padStart(2,'0')+String(_n.getSeconds()).padStart(2,'0');
  const _fname='ผลจัดทริป_'+_ts+'.xlsx';
  const buf=await wb.xlsx.writeBuffer();
  const blob=new Blob([buf],{{type:'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}});
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a'); a.href=url; a.download=_fname; a.click();
  setTimeout(()=>URL.revokeObjectURL(url),5000);
  showToast('📥 บันทึก '+_fname+' แล้ว');
}}

// ── INIT ──────────────────────────────────────────────────────────────────
renderSidebar();
function _doMapInit() {{
  if (_mapInited) return; _mapInited = true;
  _initMap(); renderMarkers();
  [100, 300, 600, 1000, 1500, 2500].forEach(ms => setTimeout(() => {{ if (map) map.invalidateSize({{ animate: false }}); }}, ms));
  [1000, 1800, 3000].forEach(ms => setTimeout(() => {{ if (map) fitAll(); }}, ms));
}}
if (document.readyState === 'loading') {{
  document.addEventListener('DOMContentLoaded', () => setTimeout(_doMapInit, 80));
}} else {{
  setTimeout(_doMapInit, 80);
}}
setTimeout(_doMapInit, 250); setTimeout(_doMapInit, 700); setTimeout(_doMapInit, 1500); setTimeout(_doMapInit, 3000);
if (typeof ResizeObserver !== 'undefined') {{
  const ro = new ResizeObserver(entries => {{
    for (const e of entries) {{
      if (e.contentRect.width > 0 && e.contentRect.height > 0) {{
        if (!_mapInited) _doMapInit(); else if (map) map.invalidateSize({{ animate: false }});
      }}
    }}
  }});
  ['map','app','map-wrap'].forEach(id => {{ const el=document.getElementById(id); if(el) ro.observe(el); }});
}}
window.addEventListener('resize', () => {{ if (map) map.invalidateSize({{ animate: false }}); }});
window.addEventListener('beforeunload', () => {{ if (map) {{ try {{ map.remove(); }} catch(e){{}} map = null; }} }});
</script>
</body>
</html>"""
    return html
