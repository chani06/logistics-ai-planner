"""
🚚 Sequential Zone-first Trip Planner  v4
Algorithm:
  1. เรียง zone → district → subdistrict ตามระยะจาก DC
  2. ภายในตำบล เรียง branch ด้วย nearest-neighbor
  3. Sequential fill: ยัดทีละ branch จนเต็ม แล้วเปิดทริปใหม่
     → ถ้าตำบลหนึ่งเกิน capacity ให้แบ่งข้ามทริป (ส่วนที่เหลือไปรวมตำบลถัดไป)
  4. Consolidate: รวมทริปเล็กภายในอำเภอ
  5. Cross-merge: จังหวัดน้อยรวมกับจังหวัดใกล้เคียง
"""

import pandas as pd
import numpy as np
import math
from typing import List, Dict, Optional, Tuple

# ── Vehicle limits ────────────────────────────────────────────────────────
LIMITS = {
    '4W': {'max_w': 2500,  'max_c': 5.0},
    'JB': {'max_w': 3500,  'max_c': 7.0},
    '6W': {'max_w': 6000,  'max_c': 20.0},
}
_RANK = {'4W': 0, 'JB': 1, '6W': 2}


def _hav(lat1, lon1, lat2, lon2) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2-lat1)/2)**2 +
         math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2)
    return 2*R*math.asin(math.sqrt(max(0.0, min(1.0, a))))


def _s(v) -> str:
    x = str(v).strip()
    return '' if x.lower() in ('nan','none','') else x


def _best_truck(w: float, c: float, max_veh: str = '6W') -> str:
    """รถเล็กที่สุดที่จุได้ ภายใต้ max_veh"""
    cap = _RANK.get(max_veh if max_veh in LIMITS else '6W', 2)
    for t in ('4W', 'JB', '6W'):
        if _RANK[t] > cap:
            continue
        if w <= LIMITS[t]['max_w'] and c <= LIMITS[t]['max_c']:
            return t
    return max_veh if max_veh in LIMITS else '6W'


def _get_mv(df: pd.DataFrame, ri: int) -> str:
    if '_max_vehicle' not in df.columns:
        return '6W'
    mv = _s(df.at[ri, '_max_vehicle']) or '6W'
    return mv if mv in LIMITS else '6W'


def _nn_order(rows: List[int], df: pd.DataFrame,
              start_lat: float, start_lon: float) -> List[int]:
    """เรียง rows ด้วย nearest-neighbor จาก start point"""
    if len(rows) <= 1:
        return list(rows)
    rem = set(rows)
    ordered = []
    cur = (start_lat, start_lon)
    while rem:
        nxt = min(rem, key=lambda i: _hav(cur[0], cur[1],
                                           float(df.at[i, '_lat']),
                                           float(df.at[i, '_lon'])))
        ordered.append(nxt)
        rem.remove(nxt)
        cur = (float(df.at[nxt, '_lat']), float(df.at[nxt, '_lon']))
    return ordered


# ── Sequential bin-fill ───────────────────────────────────────────────────
class _Trip:
    __slots__ = ('id', 'rows', 'w', 'c', 'mv_strict')

    def __init__(self, tid: int):
        self.id       = tid
        self.rows:    List[int] = []
        self.w        = 0.0
        self.c        = 0.0
        self.mv_strict = '6W'   # เข้มงวดที่สุดในทริปนี้

    @property
    def truck(self) -> str:
        return _best_truck(self.w, self.c, self.mv_strict)

    def _update_strict(self, mv: str):
        if _RANK.get(mv, 2) < _RANK.get(self.mv_strict, 2):
            self.mv_strict = mv

    def can_add(self, w: float, c: float, mv: str) -> bool:
        # ต้องเช็คกับรถที่เข้มงวดที่สุดที่จะเป็นหลังเพิ่ม branch นี้
        new_strict = mv if _RANK.get(mv,2) < _RANK.get(self.mv_strict,2) else self.mv_strict
        lim = LIMITS[new_strict]
        return (self.w + w <= lim['max_w'] and
                self.c + c <= lim['max_c'])

    def add(self, ri: int, w: float, c: float, mv: str):
        self.rows.append(ri)
        self.w += w
        self.c += c
        self._update_strict(mv)


def _loc_groups(ordered_rows: List[int], df: pd.DataFrame) -> List[List[int]]:
    """
    จัดกลุ่ม branches ที่มีพิกัดเดียวกัน (lat/lon ปัดเป็น 5 ทศนิยม)
    ให้อยู่กลุ่มเดียวกัน ลำดับภายในกลุ่มตาม ordered_rows
    """
    groups: List[List[int]] = []
    seen: Dict[tuple, int] = {}   # (lat_r, lon_r) → index ใน groups
    for ri in ordered_rows:
        key = (round(float(df.at[ri,'_lat']), 5),
               round(float(df.at[ri,'_lon']), 5))
        if key in seen:
            groups[seen[key]].append(ri)
        else:
            seen[key] = len(groups)
            groups.append([ri])
    return groups


def _sequential_fill(
    ordered_rows: List[int],
    df: pd.DataFrame,
    trip_start: int,
) -> Tuple[Dict[int,int], Dict[int,str], int]:
    """
    Sequential fill แบบ location-group:
    - จัดกลุ่ม branches ที่พิกัดเดียวกันให้อยู่กลุ่มเดียว
    - ยัดทีละ location-group ตามลำดับ
    - ถ้า group ทั้งกลุ่มใส่ไม่ได้ → ปิดทริปปัจจุบัน เปิดทริปใหม่
    → ไม่มีการแยกสาขาที่อยู่จุดเดียวกันออกจากกัน
    คืน (trip_map, truck_map, next_trip_id)
    """
    tm:  Dict[int,int] = {}
    trm: Dict[int,str] = {}
    tid = trip_start
    cur = _Trip(tid); tid += 1

    for grp in _loc_groups(ordered_rows, df):
        # คำนวณ weight/cube/mv รวมของทั้งกลุ่ม
        gw = sum(float(df.at[ri,'Weight']) for ri in grp)
        gc = sum(float(df.at[ri,'Cube'])   for ri in grp)
        gm = min((_get_mv(df, ri) for ri in grp),
                 key=lambda x: _RANK.get(x, 2))   # mv เข้มงวดสุดในกลุ่ม

        if cur.can_add(gw, gc, gm):
            for ri in grp:
                cur.add(ri, float(df.at[ri,'Weight']),
                            float(df.at[ri,'Cube']), _get_mv(df, ri))
        else:
            # ปิดทริปปัจจุบัน แล้วเปิดใหม่พร้อมกลุ่มนี้
            if cur.rows:
                for r in cur.rows:
                    tm[r]  = cur.id
                    trm[r] = cur.truck
            cur = _Trip(tid); tid += 1
            for ri in grp:
                cur.add(ri, float(df.at[ri,'Weight']),
                            float(df.at[ri,'Cube']), _get_mv(df, ri))

    if cur.rows:
        for r in cur.rows:
            tm[r]  = cur.id
            trm[r] = cur.truck

    return tm, trm, tid


# ── helpers สร้าง info dict ──────────────────────────────────────────────────
def _most_common(series) -> str:
    from collections import Counter
    vals = [_s(v) for v in series if _s(v)]
    return Counter(vals).most_common(1)[0][0] if vals else ''


def _build_trip_info(df: pd.DataFrame) -> Dict[int, dict]:
    info = {}
    has_trip = df['Trip'] > 0
    for tid in sorted(df.loc[has_trip, 'Trip'].unique()):
        mask  = df['Trip'] == tid
        t_df  = df[mask]
        truck = _s(t_df['Truck'].iloc[0]) if 'Truck' in t_df.columns else '6W'
        if truck not in LIMITS: truck = '6W'
        lim   = LIMITS[truck]
        tw    = float(t_df['Weight'].sum())
        tc    = float(t_df['Cube'].sum())
        gz = _most_common(t_df['_gz']) if '_gz' in t_df.columns else ''
        gp = _most_common(t_df['_gp']) if '_gp' in t_df.columns else ''
        gd = _most_common(t_df['_gd']) if '_gd' in t_df.columns else ''
        gs = _most_common(t_df['_gs']) if '_gs' in t_df.columns else ''
        clat  = float(t_df['_lat'].mean()) if '_lat' in t_df.columns else 0.0
        clon  = float(t_df['_lon'].mean()) if '_lon' in t_df.columns else 0.0
        fill  = max(tw / lim['max_w'], tc / lim['max_c'])
        info[tid] = {
            'truck': truck, 'w': tw, 'c': tc, 'lim': lim,
            'scope_z': gz,
            'scope_p': f'{gz}||{gp}',
            'scope_d': f'{gz}||{gp}||{gd}',
            'scope_s': f'{gz}||{gp}||{gd}||{gs}',
            'fill': fill, 'lat': clat, 'lon': clon,
            'rows': t_df.index.tolist(),
            'strict': truck,
        }
    return info


def _do_merge(info: dict, sid: int, tid2: int, df: pd.DataFrame) -> bool:
    s  = info[sid]
    t2 = info[tid2]
    cw = s['w'] + t2['w']
    cc = s['c'] + t2['c']
    new_strict = min(s['strict'], t2['strict'], key=lambda x: _RANK.get(x, 2))
    lim = LIMITS[new_strict]
    if cw <= lim['max_w'] and cc <= lim['max_c']:
        new_tk = _best_truck(cw, cc, new_strict)
        all_rows = s['rows'] + t2['rows']
        for ri in all_rows:
            df.at[ri, 'Trip']  = sid
            df.at[ri, 'Truck'] = new_tk
        s['w'] = cw; s['c'] = cc; s['truck'] = new_tk
        s['strict'] = new_strict
        s['fill'] = max(cw/lim['max_w'], cc/lim['max_c'])
        s['rows'] = all_rows
        return True
    return False


# ── Tiered Consolidation: ตำบล → อำเภอ → จังหวัด → zone ─────────────────
# ระยะ centroid สูงสุดแต่ละระดับ
_DIST_SUBD = 20_000   # ตำบล: ≤ 20 km
_DIST_DIST = 40_000   # อำเภอ: ≤ 40 km
_DIST_PROV = 60_000   # จังหวัด: ≤ 60 km
_DIST_ZONE = 80_000   # zone: ≤ 80 km


_DIST_CROSS_PROV = 20_000   # cross-province merge สูงสุด 20km


def _consolidate_one_level(
    df: pd.DataFrame,
    scope_key: str,          # 'scope_s' | 'scope_d' | 'scope_p' | 'scope_z'
    fill_ratio: float,
    max_dist: float,
    prefer_key: Optional[str] = None,   # prefer same value of this key first
) -> pd.DataFrame:
    """
    รวมทริปที่ fill < fill_ratio ภายใน scope_key เดียวกัน
    sort candidates: prefer_key เดียวกันก่อน → ใกล้ก่อน
    ถ้า scope_key = scope_z (level 4) และ province ต่างกัน → ใช้ _DIST_CROSS_PROV
    วนซ้ำจนไม่มีการ merge
    """
    if 'Trip' not in df.columns:
        return df
    is_zone_level = (scope_key == 'scope_z')
    changed = True
    while changed:
        changed = False
        info = _build_trip_info(df)
        small = sorted([t for t in info.values() if t['fill'] < fill_ratio],
                       key=lambda t: t['fill'])
        used: set = set()
        for s in small:
            sid = next((k for k, v in info.items() if v is s), None)
            if sid is None or sid in used:
                continue
            s_scope = s.get(scope_key, '')
            s_pref  = s.get(prefer_key, '') if prefer_key else None
            s_prov  = s.get('scope_p', '')

            def _dist_ok(v):
                d = _hav(s['lat'], s['lon'], v['lat'], v['lon'])
                if is_zone_level and v.get('scope_p','') != s_prov:
                    return d <= _DIST_CROSS_PROV   # cross-province: 20km max
                return d <= max_dist

            candidates = sorted(
                [(k, v) for k, v in info.items()
                 if k != sid and k not in used
                 and v.get(scope_key, '') == s_scope
                 and _dist_ok(v)],
                key=lambda x: (
                    0 if (prefer_key and x[1].get(prefer_key,'') == s_pref) else 1,
                    _hav(s['lat'], s['lon'], x[1]['lat'], x[1]['lon'])
                ))

            for tid2, _ in candidates:
                if tid2 in used:
                    continue
                if _do_merge(info, sid, tid2, df):
                    used.add(tid2)
                    changed = True
                    if info[sid]['fill'] >= fill_ratio:
                        break
    return df


def _consolidate_tiered(df: pd.DataFrame) -> pd.DataFrame:
    """
    4 ระดับ: ตำบล → อำเภอ → จังหวัด → zone
    แต่ละระดับรวมทริปใกล้ที่สุดก่อน prefer ระดับย่อยกว่าก่อนเสมอ
    """
    # เพิ่ม scope_p (province) ใน df info ผ่านการสร้างใน _build_trip_info
    # Level 1: ตำบล (fill < 0.95, ≤ 20km) — aggressive
    df = _consolidate_one_level(df, 'scope_s', 0.95, _DIST_SUBD)
    # Level 2: อำเภอ (fill < 0.80, ≤ 40km) prefer ตำบลเดียวกันก่อน
    df = _consolidate_one_level(df, 'scope_d', 0.80, _DIST_DIST, prefer_key='scope_s')
    # Level 3: จังหวัด (fill < 0.65, ≤ 60km) prefer อำเภอเดียวกันก่อน
    df = _consolidate_one_level(df, 'scope_p', 0.65, _DIST_PROV, prefer_key='scope_d')
    # Level 4: zone (fill < 0.50, ≤ 80km) prefer จังหวัดเดียวกันก่อน
    df = _consolidate_one_level(df, 'scope_z', 0.50, _DIST_ZONE, prefer_key='scope_p')
    return df


# compat aliases
def _consolidate_subd(df, fill_ratio=0.95):
    return _consolidate_one_level(df, 'scope_s', fill_ratio, _DIST_SUBD)

def _consolidate(df, fill_ratio=0.50):
    return _consolidate_one_level(df, 'scope_d', fill_ratio, _DIST_DIST, prefer_key='scope_s')


# ── Cross-merge: จังหวัดน้อยรวมกับจังหวัดใกล้เคียง ──────────────────────
def _cross_merge(df: pd.DataFrame, fill_ratio: float = 0.40,
                 max_dist_m: float = 15_000) -> pd.DataFrame:
    """
    ทริปที่ fill < fill_ratio + centroid ≤ max_dist_m
    จาก province ต่างกัน → merge ถ้าไม่เกิน capacity
    """
    has_trip = df['Trip'] > 0
    if not has_trip.any(): return df
    prov_col = '_province' if '_province' in df.columns else None
    if not prov_col: return df

    info = {}
    for tid in sorted(df.loc[has_trip, 'Trip'].unique()):
        mask  = df['Trip'] == tid
        t_df  = df[mask]
        truck = _s(t_df['Truck'].iloc[0]) if 'Truck' in t_df.columns else '6W'
        if truck not in LIMITS: truck = '6W'
        lim   = LIMITS[truck]
        tw    = float(t_df['Weight'].sum())
        tc    = float(t_df['Cube'].sum())
        prov  = _s(t_df[prov_col].iloc[0])
        gz    = _s(t_df['_gz'].iloc[0]) if '_gz' in t_df.columns else ''
        clat  = float(t_df['_lat'].mean())
        clon  = float(t_df['_lon'].mean())
        fill  = max(tw/lim['max_w'], tc/lim['max_c'])
        info[tid] = {'truck': truck, 'w': tw, 'c': tc, 'lim': lim,
                     'prov': prov, 'gz': gz, 'fill': fill,
                     'lat': clat, 'lon': clon,
                     'rows': t_df.index.tolist(), 'strict': truck}

    small = sorted([t for t in info.values() if t['fill'] < fill_ratio],
                   key=lambda t: t['fill'])
    used = set()

    for s in small:
        sid = next((k for k,v in info.items() if v is s), None)
        if sid is None or sid in used: continue

        candidates = sorted(
            [t for tid2,t in info.items()
             if tid2 != sid and tid2 not in used
             and t['gz'] == s['gz']           # zone เดียวกันเท่านั้น
             and t['prov'] != s['prov']        # ต่าง province
             and _hav(s['lat'],s['lon'],t['lat'],t['lon']) <= max_dist_m],
            key=lambda t: _hav(s['lat'],s['lon'],t['lat'],t['lon']))

        for t2 in candidates:
            tid2 = next((k for k,v in info.items() if v is t2), None)
            if tid2 is None: continue
            cw = s['w']+t2['w']; cc = s['c']+t2['c']
            new_strict = min(s['strict'],t2['strict'],key=lambda x:_RANK.get(x,2))
            lim = LIMITS[new_strict]
            if cw <= lim['max_w'] and cc <= lim['max_c']:
                new_tk = _best_truck(cw, cc, new_strict)
                all_rows = s['rows'] + t2['rows']
                for ri in all_rows:
                    df.at[ri,'Trip']  = sid
                    df.at[ri,'Truck'] = new_tk
                s['w']=cw; s['c']=cc; s['truck']=new_tk
                s['fill']=max(cw/lim['max_w'],cc/lim['max_c'])
                s['rows']=all_rows; s['strict']=new_strict
                used.add(tid2)
                break

    return df


# ── Split trips ที่เกิน capacity ──────────────────────────────────────────
def _split_over(df: pd.DataFrame, next_tid: int,
                dc_lat: float, dc_lon: float) -> Tuple[pd.DataFrame, int]:
    """
    ตรวจทุกทริปว่า weight หรือ cube เกิน limit จริงของ truck ที่ assign ไหม
    ถ้าเกิน → re-pack ด้วย sequential fill (location-group aware)
    วนซ้ำจนไม่มีทริปเกิน
    """
    max_pass = 5
    for _ in range(max_pass):
        found = False
        for tid in sorted(df.loc[df['Trip']>0, 'Trip'].unique()):
            mask  = df['Trip'] == tid
            t_df  = df[mask]
            # หา limit จาก max_vehicle ที่เข้มงวดที่สุดในทริป
            rows  = t_df.index.tolist()
            strict = _mv_strict(rows, df)   # ฟังก์ชันใหม่
            lim   = LIMITS[strict]
            tw    = float(t_df['Weight'].sum())
            tc    = float(t_df['Cube'].sum())
            if tw <= lim['max_w'] and tc <= lim['max_c']:
                continue
            # เกิน → reset + re-pack
            df.loc[rows, 'Trip']  = 0
            df.loc[rows, 'Truck'] = ''
            ordered = _nn_order(rows, df, dc_lat, dc_lon)
            tm, trm, next_tid = _sequential_fill(ordered, df, next_tid)
            for ri, t in tm.items():
                df.at[ri,'Trip']  = t
                df.at[ri,'Truck'] = trm[ri]
            found = True
        if not found:
            break
    return df, next_tid


def _mv_strict(rows: List[int], df: pd.DataFrame) -> str:
    """หา max_vehicle เข้มงวดสุดในกลุ่ม rows"""
    if '_max_vehicle' not in df.columns:
        return '6W'
    best = '6W'
    for r in rows:
        mv = _s(df.at[r,'_max_vehicle']) or '6W'
        if mv not in LIMITS: mv = '6W'
        if _RANK[mv] < _RANK[best]: best = mv
    return best


# ── ตรวจตำบลกระจายเกิน 2 ทริป → consolidate ──────────────────────────────
def _fix_subd_spread(df: pd.DataFrame, next_tid: int,
                     dc_lat: float, dc_lon: float) -> Tuple[pd.DataFrame, int]:
    """
    ถ้าตำบลเดียวกันกระจายอยู่ > 2 ทริป → รวม trips ที่เล็กที่สุดเข้าหากัน
    จนเหลือ ≤ 2 ทริปต่อตำบล (หรือจนเต็ม)
    """
    if '_gs' not in df.columns:
        return df, next_tid

    for subd in df.loc[df['Trip']>0, '_gs'].unique():
        s_mask  = df['_gs'] == subd
        s_trips = df.loc[s_mask & (df['Trip']>0), 'Trip'].unique().tolist()
        if len(s_trips) <= 2:
            continue

        # เรียง trips ตาม fill (น้อย→มาก) แล้ว merge ที่เล็กเข้าที่ใหญ่กว่า
        trip_data = []
        for tid in s_trips:
            t_mask = (df['Trip']==tid)
            tw = float(df.loc[t_mask,'Weight'].sum())
            tc = float(df.loc[t_mask,'Cube'].sum())
            strict = _mv_strict(df.loc[t_mask].index.tolist(), df)
            lim    = LIMITS[strict]
            fill   = max(tw/lim['max_w'], tc/lim['max_c'])
            trip_data.append({'id':tid,'w':tw,'c':tc,'strict':strict,'fill':fill,
                               'rows':df.loc[t_mask].index.tolist()})
        trip_data.sort(key=lambda x: x['fill'])   # เล็กก่อน

        # merge trips เล็กๆ เข้า trip ใหญ่ที่สุด (ตัวสุดท้าย)
        base = trip_data[-1]
        for small in trip_data[:-1]:
            cw = base['w'] + small['w']
            cc = base['c'] + small['c']
            new_strict = min(base['strict'], small['strict'],
                             key=lambda x: _RANK.get(x,2))
            lim = LIMITS[new_strict]
            if cw <= lim['max_w'] and cc <= lim['max_c']:
                new_tk = _best_truck(cw, cc, new_strict)
                all_rows = base['rows'] + small['rows']
                for ri in all_rows:
                    df.at[ri,'Trip']  = base['id']
                    df.at[ri,'Truck'] = new_tk
                base['w']=cw; base['c']=cc
                base['strict']=new_strict; base['rows']=all_rows

    return df, next_tid


# ── Public API ──────────────────────────────────────────────────────────────
def solve_vrp_by_province(
    branches_df: pd.DataFrame,
    dc_lat: float, dc_lon: float,
    buffer: float = 1.10,
    time_limit_sec: int = 15,
    max_vehicles_per_type: Optional[Dict[str,int]] = None,
) -> pd.DataFrame:
    """
    Sequential zone-first packing v4
    1. เรียง zone→district→subdistrict ตามระยะ DC
    2. ภายในตำบล: nearest-neighbor ordering
    3. Sequential fill: ยัดจนเต็มแล้วเปิดทริปใหม่ (ตำบลแบ่งข้ามทริปได้)
    4. Consolidate ทริปเล็กในอำเภอเดียวกัน
    5. Cross-merge จังหวัดน้อยกับจังหวัดใกล้เคียง
    6. Split ทริปที่เกิน
    """
    df = branches_df.copy().reset_index(drop=True)
    df['Trip']  = 0
    df['Truck'] = ''

    for col, dv in [('_lat',0),('_lon',0),('Weight',0),('Cube',0)]:
        df[col] = pd.to_numeric(df.get(col, dv), errors='coerce').fillna(dv)

    has_coord    = (df['_lat'] > 0) & (df['_lon'] > 0)
    no_coord_idx = df[~has_coord].index.tolist()

    def _gcol(name):
        if name in df.columns:
            return df[name].apply(_s)
        return pd.Series([''] * len(df), index=df.index)

    zone_s = _gcol('_logistics_zone')
    prov_s = _gcol('_province')
    dist_s = _gcol('_district')
    subd_s = _gcol('_subdistrict')
    zone_s = zone_s.where(zone_s != '', prov_s).where(lambda x: x != '', 'Z_UNK')

    df['_gz'] = zone_s
    df['_gp'] = prov_s.where(prov_s != '', '__NP__')
    df['_gd'] = dist_s.where(dist_s != '', '__ND__')
    df['_gs'] = subd_s.where(subd_s != '', '__NS__')

    vdf = df[has_coord]

    def _ctr(sub):
        return _hav(dc_lat, dc_lon,
                    float(sub['_lat'].mean()), float(sub['_lon'].mean()))

    # เรียง zone: province ก่อน (ตามระยะ centroid ของ province จาก DC)
    # ภายใน province เรียง zone ตามระยะ — ทำให้ทริปของแต่ละจังหวัดอยู่ติดกัน
    def _prov_of_zone(z):
        from collections import Counter
        zdf = vdf[vdf['_gz'] == z]
        pv = Counter(zdf['_gp'].tolist()).most_common(1)
        return pv[0][0] if pv else ''

    prov_dist: Dict[str, float] = {}
    for pv in vdf['_gp'].unique():
        pv_df = vdf[vdf['_gp'] == pv]
        prov_dist[pv] = _hav(dc_lat, dc_lon,
                              float(pv_df['_lat'].mean()),
                              float(pv_df['_lon'].mean()))

    zones = sorted(vdf['_gz'].unique(),
                   key=lambda z: (prov_dist.get(_prov_of_zone(z), 0),
                                  _ctr(vdf[vdf['_gz'] == z])))

    # ── Sequential fill แยกตาม zone (ป้องกันทริปข้ามโซน) ────────────────────
    trip_counter = 1
    prev_lat, prev_lon = dc_lat, dc_lon

    for zone in zones:
        z_mask = has_coord & (df['_gz'] == zone)
        z_df   = df[z_mask]
        dists  = sorted(z_df['_gd'].unique(),
                        key=lambda d: _ctr(z_df[z_df['_gd']==d]))
        zone_ordered: List[int] = []

        for dist in dists:
            d_mask = z_mask & (df['_gd'] == dist)
            d_df   = df[d_mask]
            if d_df.empty: continue
            subds = sorted(d_df['_gs'].unique(),
                           key=lambda s: _ctr(d_df[d_df['_gs']==s]))
            for subd in subds:
                s_mask = d_mask & (df['_gs'] == subd)
                s_rows = df[s_mask].index.tolist()
                if not s_rows: continue
                ordered = _nn_order(s_rows, df, prev_lat, prev_lon)
                zone_ordered.extend(ordered)
                last = ordered[-1]
                prev_lat = float(df.at[last, '_lat'])
                prev_lon = float(df.at[last, '_lon'])

        if zone_ordered:
            tm, trm, trip_counter = _sequential_fill(zone_ordered, df, trip_counter)
            for ri, t in tm.items():
                df.at[ri, 'Trip']  = t
                df.at[ri, 'Truck'] = trm[ri]

    # ── Fix: ตำบลกระจายเกิน 2 ทริป → consolidate ──────────────────────────
    df, trip_counter = _fix_subd_spread(df, trip_counter, dc_lat, dc_lon)

    # ── Tiered consolidation: ตำบล → อำเภอ → จังหวัด → zone ───────────────
    df = _consolidate_tiered(df)

    # ── Split ทริปเกิน (รัน หลัง merge ทุกขั้น) ─────────────────────────────
    df, trip_counter = _split_over(df, trip_counter, dc_lat, dc_lon)

    # ── สาขาไม่มีพิกัด ─────────────────────────────────────────────────────
    for ri in no_coord_idx:
        df.at[ri, 'Trip']  = trip_counter
        df.at[ri, 'Truck'] = '6W'
        trip_counter += 1

    df.drop(columns=['_gz','_gp','_gd','_gs'], inplace=True, errors='ignore')
    return df


def solve_vrp(branches_df, dc_lat, dc_lon, buffer=1.10,
              time_limit_sec=15, max_vehicles_per_type=None):
    return solve_vrp_by_province(branches_df, dc_lat, dc_lon, buffer,
                                  time_limit_sec, max_vehicles_per_type)
