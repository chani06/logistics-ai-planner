"""
ทดสอบ backend ทุกขั้นตอน — ตรวจสอบผลปรับแก้
Usage: python _test_run.py
"""
import sys, os, traceback
import pandas as pd
import importlib.util

# ──────────────────────────────────────────────
# 0. ค้นหาไฟล์ test
# ──────────────────────────────────────────────
TEST_FILES = [
    r"Dc/test.xlsx",
    r"ตัวอย่างไฟล์_Upload_20251224.xlsx",
]
test_file = None
for f in TEST_FILES:
    if os.path.exists(f):
        test_file = f
        break

if not test_file:
    print("❌ ไม่พบไฟล์ทดสอบ กรุณาวางไฟล์ .xlsx ไว้ใน Dc/test.xlsx")
    sys.exit(1)

print("=" * 70)
print(f"📁 ไฟล์ทดสอบ: {test_file}")
print("=" * 70)

# ──────────────────────────────────────────────
# 1. โหลดแอพ
# ──────────────────────────────────────────────
print("\n⚙️  กำลัง import app.py ...")
spec = importlib.util.spec_from_file_location("app", "app.py")
app = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(app)
    print("✅ import สำเร็จ")
except Exception as e:
    print(f"❌ import ล้มเหลว: {e}")
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────
# 2. อ่านไฟล์ + process
# ──────────────────────────────────────────────
print(f"\n📄 อ่าน Excel ...")
try:
    xl = pd.ExcelFile(test_file)
    print(f"   Sheets: {xl.sheet_names}")
    # เลือก sheet 2.Punthai หรือ sheet แรก
    sheet = "2.Punthai" if "2.Punthai" in xl.sheet_names else xl.sheet_names[0]
    raw_df = pd.read_excel(test_file, sheet_name=sheet, header=1)
    print(f"   {sheet} → {len(raw_df)} แถว, {len(raw_df.columns)} คอลัมน์")
except Exception as e:
    print(f"❌ อ่านไฟล์ล้มเหลว: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n🔄 process_dataframe ...")
try:
    df = app.process_dataframe(raw_df)
    if df is None or df.empty:
        print("❌ process_dataframe คืนข้อมูลว่าง")
        sys.exit(1)
    # กรองเฉพาะแถวที่มี Code และน้ำหนัก > 0
    before = len(df)
    df = df[df['Code'].notna() & (df['Code'].str.strip() != '') & (df['Weight'] > 0)].copy()
    print(f"✅ {before} แถว → {len(df)} แถว (กรอง Code/Weight ว่าง)")
    print(f"   BU unique: {df['BU'].unique().tolist()[:10]}")
    print(f"   Weight รวม: {df['Weight'].sum():.1f} kg  |  Cube รวม: {df['Cube'].sum():.2f} m³")
except Exception as e:
    print(f"❌ process_dataframe error: {e}")
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────
# 3. predict_trips
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("🚛 predict_trips ...")
print("=" * 70)

# ── Fleet ที่มี — ตั้งเพื่อทดสอบ consolidation ──
FLEET_LIMITS = {
    'JB': 24,
    '6W': 13,
    '4W': 13,
}
print(f"   🎯 Fleet target: 4W={FLEET_LIMITS['4W']}, JB={FLEET_LIMITS['JB']}, 6W={FLEET_LIMITS['6W']} (รวม {sum(FLEET_LIMITS.values())} คัน)")

try:
    result_df, summary_df, fleet_used = app.predict_trips(
        test_df=df,
        model_data=app.MASTER_DATA,
        punthai_buffer=1.0,
        maxmart_buffer=1.10,
        fleet_limits=FLEET_LIMITS,
    )
    if result_df is None or result_df.empty:
        print("❌ predict_trips คืนข้อมูลว่าง")
        sys.exit(1)
    print(f"✅ จัดทริปสำเร็จ: {result_df['Trip'].max()} ทริป")
except Exception as e:
    print(f"❌ predict_trips error: {e}")
    traceback.print_exc()
    sys.exit(1)

# ──────────────────────────────────────────────
# 4. ตรวจสอบ
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("🔍 ตรวจสอบทุกเงื่อนไข")
print("=" * 70)

LIMITS = {
    '4W':  {'max_w': 2500, 'max_c': 5.0,  'max_drops': 12},
    'JB':  {'max_w': 3500, 'max_c': 7.0,  'max_drops': 12},
    '6W':  {'max_w': 6000, 'max_c': 20.0, 'max_drops': 999},
}
PUNTHAI_LIMITS = {
    '4W':  {'max_w': 2500, 'max_c': 5.0,  'max_drops': 5},
    'JB':  {'max_w': 3500, 'max_c': 7.0,  'max_drops': 7},
    '6W':  {'max_w': 6000, 'max_c': 20.0, 'max_drops': 999},
}
VEHICLE_RANK = {'4W': 1, 'JB': 2, '6W': 3}

errors = []
warnings = []

# ── 4.1 ทุกสาขาถูกจัด? ──
unassigned = result_df[result_df['Trip'] == 0]
if len(unassigned):
    warnings.append(f"⚠️  สาขาไม่ถูกจัด {len(unassigned)} สาขา: {unassigned['Code'].tolist()[:10]}")
else:
    print("✅ ทุกสาขาถูกจัดทริปครบ")

# ── 4.2 เรียงทริปตามระยะทาง (Trip 1 = ไกลสุด) ──
if '_distance_from_dc' in result_df.columns:
    trip_max_dist = result_df[result_df['Trip'] > 0].groupby('Trip')['_distance_from_dc'].max()
    trips_sorted = trip_max_dist.sort_index()
    # ตรวจว่า trip ก่อนหน้า >= trip ถัดไป (descending)
    trip_list = trip_max_dist.sort_index().items()
    prev_trip, prev_dist = None, None
    out_of_order = []
    for t, d in sorted(trip_max_dist.items()):
        if prev_dist is not None and d > prev_dist * 1.05:  # tolerence 5%
            out_of_order.append((prev_trip, round(prev_dist,1), t, round(d,1)))
        prev_trip, prev_dist = t, d
    if out_of_order:
        warnings.append(f"⚠️  ลำดับทริปไม่ถูกต้อง (Trip N ใกล้กว่า N+1): {out_of_order[:5]}")
    else:
        print(f"✅ ลำดับทริปถูกต้อง (ไกลก่อน)\n   Trip 1 = {trip_max_dist.get(1,0):.1f}km  |  Trip {int(trip_max_dist.index.max())} = {trip_max_dist.max():.1f}km  |  Min = {trip_max_dist.min():.1f}km")
else:
    warnings.append("⚠️  ไม่มีคอลัมน์ _distance_from_dc — ไม่สามารถตรวจลำดับทริปได้")

# ── 4.3 เรียงแถวภายในทริปตามระยะทาง (ไกลก่อน) ──
if '_distance_from_dc' in result_df.columns:
    row_order_errors = []
    for t in sorted(result_df[result_df['Trip'] > 0]['Trip'].unique()):
        tdata = result_df[result_df['Trip'] == t].reset_index(drop=True)
        for i in range(len(tdata) - 1):
            d_now = tdata.loc[i, '_distance_from_dc']
            d_next = tdata.loc[i+1, '_distance_from_dc']
            if d_next > d_now * 1.10:  # tolerance 10%
                row_order_errors.append(
                    f"Trip {t} แถว {i}({tdata.loc[i,'Code']},{d_now:.0f}km) < แถว {i+1}({tdata.loc[i+1,'Code']},{d_next:.0f}km)"
                )
    if row_order_errors:
        warnings.append(f"⚠️  เรียงแถวในทริปผิดลำดับ {len(row_order_errors)} จุด:\n     " + "\n     ".join(row_order_errors[:5]))
    else:
        print("✅ เรียงแถวภายในทริปถูกต้อง (ไกลก่อน)")

# ── 4.4 Vehicle constraint ──
if 'Truck' in result_df.columns and '_max_vehicle' in result_df.columns:
    veh_violations = []
    for _, row in result_df.iterrows():
        if row['Trip'] == 0:
            continue
        truck_str = str(row.get('Truck', '')).split()[0]
        if truck_str == '4WJ':
            truck_str = 'JB'
        max_v = str(row.get('_max_vehicle', '6W'))
        if truck_str in VEHICLE_RANK and max_v in VEHICLE_RANK:
            if VEHICLE_RANK[truck_str] > VEHICLE_RANK[max_v]:
                veh_violations.append(
                    f"Trip {int(row['Trip'])} สาขา {row['Code']}: ใช้ {truck_str} แต่ max={max_v}"
                )
    if veh_violations:
        errors.append(f"❌ Vehicle constraint ละเมิด {len(veh_violations)} สาขา:\n     " + "\n     ".join(veh_violations[:5]))
    else:
        print("✅ Vehicle constraint ถูกต้องทุกสาขา")
else:
    warnings.append("⚠️  ไม่มีคอลัมน์ Truck/_max_vehicle")

# ── 4.5 weight/cube ไม่เกิน buffer ──
over_limit = []
impossible_cases = []  # สาขาเดียวเกิน capacity ของรถที่สาขานั้นรับได้ = impossible
for t in sorted(result_df[result_df['Trip'] > 0]['Trip'].unique()):
    tdata = result_df[result_df['Trip'] == t]
    truck_raw = str(tdata.iloc[0].get('Truck', '6W')).split()[0]
    truck = 'JB' if truck_raw == '4WJ' else truck_raw
    if truck not in LIMITS:
        truck = '6W'
    is_pt = all(str(r).upper() in ('211','PUNTHAI') for r in tdata.get('BU', pd.Series()))
    lim = (PUNTHAI_LIMITS if is_pt else LIMITS)[truck]
    buf = 1.0 if is_pt else 1.10
    tw = tdata['Weight'].sum()
    tc = tdata['Cube'].sum()
    dr = tdata['Code'].nunique()  # นับ unique codes (จุดส่ง) ไม่ใช่ rows
    is_over = (tw > lim['max_w'] * buf) or (tc > lim['max_c'] * buf) or (dr > lim['max_drops'])
    if is_over:
        # single-branch ที่ตัวเองเกิน capacity ของรถที่รับได้ = impossible case (ไม่ใช่ bug)
        if len(tdata) == 1:
            br = tdata.iloc[0]
            impossible_cases.append(
                f"Trip {t} ({truck}) สาขา {br['Code']}: cube={br.get('Cube',0):.2f}m³ > JB limit={lim['max_c']:.1f}m³ [impossible: singleBranch+maxV={br.get('_max_vehicle','?')}]"
            )
        else:
            if tw > lim['max_w'] * buf:
                over_limit.append(f"Trip {t} ({truck}): น้ำหนัก {tw:.0f} > {lim['max_w']*buf:.0f} kg")
            if tc > lim['max_c'] * buf:
                over_limit.append(f"Trip {t} ({truck}): คิว {tc:.2f} > {lim['max_c']*buf:.2f} m³")
            if dr > lim['max_drops']:
                over_limit.append(f"Trip {t} ({truck}): drops {dr} > {lim['max_drops']}")

if impossible_cases:
    for ic in impossible_cases:
        warnings.append(f"⚠️ Impossible single-branch overflow (ไม่ใช่ bug): {ic}")
if over_limit:
    errors.append(f"❌ เกิน buffer {len(over_limit)} ทริป:\n     " + "\n     ".join(over_limit[:8]))
    # แสดงรายละเอียดสาขาในทริปที่เกิน
    for t in sorted(result_df[result_df['Trip'] > 0]['Trip'].unique()):
        tdata = result_df[result_df['Trip'] == t]
        truck_raw = str(tdata.iloc[0].get('Truck', '6W')).split()[0]
        truck = 'JB' if truck_raw == '4WJ' else truck_raw
        if truck not in LIMITS: truck = '6W'
        is_pt = all(str(r).upper() in ('211','PUNTHAI') for r in tdata.get('BU', pd.Series()))
        lim = (PUNTHAI_LIMITS if is_pt else LIMITS)[truck]
        buf = 1.0 if is_pt else 1.10
        tw = tdata['Weight'].sum()
        tc = tdata['Cube'].sum()
        if len(tdata) > 1 and (tw > lim['max_w']*buf or tc > lim['max_c']*buf):
            print(f"\n   --- Trip {t} ({truck}) รายละเอียด ---")
            print(f"   Weight: {tw:.1f}/{lim['max_w']*buf:.1f}  Cube: {tc:.2f}/{lim['max_c']*buf:.2f}")
            for _, r in tdata.iterrows():
                print(f"   {r['Code']:15s}  w={r.get('Weight', 0):.1f}  c={r.get('Cube', 0):.2f}  maxV={r.get('_max_vehicle','?')}  dist={r.get('_distance_from_dc',0):.1f}km  truck={str(r.get('Truck','?')).split()[0]}")
else:
    if not impossible_cases:
        print("✅ ทุกทริปไม่เกิน buffer")
    else:
        print(f"✅ ทุกทริปไม่เกิน buffer (มี {len(impossible_cases)} impossible single-branch case)")

# ── 4.6 Region mixing ──
if '_province' in result_df.columns:
    get_region_name = app.get_region_name
    region_mix = []
    for t in sorted(result_df[result_df['Trip'] > 0]['Trip'].unique()):
        tdata = result_df[result_df['Trip'] == t]
        regions = set()
        for p in tdata['_province'].dropna().unique():
            r = get_region_name(str(p))
            if r and r not in ('', 'ไม่ระบุ'):
                regions.add(r)
        if len(regions) > 1:
            region_mix.append(f"Trip {t}: ปนภาค {regions}")
    if region_mix:
        errors.append(f"❌ ปนภาค {len(region_mix)} ทริป:\n     " + "\n     ".join(region_mix[:5]))
    else:
        print("✅ ไม่มีทริปที่ปนภาค")

# ── 4.7 Cross-province within same trip (ข้ามจังหวัด) ──
if '_province' in result_df.columns:
    cross_prov = []
    bkk_violations = []
    BKK_PROV = 'กรุงเทพมหานคร'
    for t in sorted(result_df[result_df['Trip'] > 0]['Trip'].unique()):
        tdata = result_df[result_df['Trip'] == t]
        provs = [p for p in tdata['_province'].dropna().unique() if str(p).strip()]
        if len(provs) > 1:
            cross_prov.append((t, provs, len(tdata)))
            # เช็ค BKK isolation (ERROR ถ้ากรุงเทพปนกับจังหวัดอื่น)
            has_bkk = BKK_PROV in provs
            has_non_bkk = any(p != BKK_PROV for p in provs)
            if has_bkk and has_non_bkk:
                truck = str(tdata.iloc[0].get('Truck', '?')).split()[0] if 'Truck' in tdata.columns else '?'
                dist_max = tdata['_distance_from_dc'].max() if '_distance_from_dc' in tdata.columns else 0
                bkk_violations.append((t, truck, len(tdata), round(dist_max,1), provs))
    print(f"\n📊 ทริปที่มีหลายจังหวัด: {len(cross_prov)}/{result_df['Trip'].max()} ทริป")
    for t, provs, n in cross_prov[:20]:
        tdata = result_df[result_df['Trip'] == t]
        truck = str(tdata.iloc[0].get('Truck', '?')).split()[0] if 'Truck' in tdata.columns else '?'
        dist_max = tdata['_distance_from_dc'].max() if '_distance_from_dc' in tdata.columns else 0
        flag = ' ❌ BKK+จังหวัดอื่น!' if BKK_PROV in provs and any(p != BKK_PROV for p in provs) else ''
        print(f"  Trip {t:3d} ({truck}) {n:3d}สาขา {dist_max:6.1f}km: {' + '.join(provs[:5])}{flag}")
    if bkk_violations:
        print(f"\n❌ BKK ISOLATION - กรุงเทพปนกับจังหวัดอื่น: {len(bkk_violations)} ทริป")
        for t, truck, n, dist, provs in bkk_violations:
            print(f"  Trip {t:3d} ({truck}) {n}สาขา {dist}km → {' + '.join(str(p) for p in provs)}")
        errors.append(f"BKK ISOLATION: กรุงเทพปนกับจังหวัดอื่นใน {len(bkk_violations)} ทริป")
    else:
        print("  ✅ กรุงเทพไม่ปนกับจังหวัดอื่น")

    # เช็ค ZONE_NEARBY cross-province (ERROR: ZONE_NEARBY_xxx ต้องไม่ปนกัน)
    nearby_violations = []
    if '_logistics_zone' in result_df.columns:
        for t, provs, n in cross_prov:
            tdata = result_df[result_df['Trip'] == t]
            nearby_zones = [z for z in tdata['_logistics_zone'].dropna().unique()
                            if str(z).startswith('ZONE_NEARBY_')]
            if len(set(nearby_zones)) > 1:
                truck = str(tdata.iloc[0].get('Truck','?')).split()[0] if 'Truck' in tdata.columns else '?'
                dist_max = tdata['_distance_from_dc'].max() if '_distance_from_dc' in tdata.columns else 0
                nearby_violations.append((t, truck, n, round(dist_max,1), provs, list(set(nearby_zones))))
    if nearby_violations:
        print(f"\n❌ ZONE_NEARBY ข้ามจังหวัด: {len(nearby_violations)} ทริป")
        for t, truck, n, dist, provs, zones in nearby_violations:
            print(f"  Trip {t:3d} ({truck}) {n}สาขา {dist}km  จังหวัด: {' + '.join(str(p) for p in provs)}")
            print(f"     zones: {zones}")
        errors.append(f"ZONE_NEARBY cross-province: {len(nearby_violations)} ทริป (ต้องแยกจังหวัด)")
    else:
        print("  ✅ ZONE_NEARBY ไม่มีข้ามจังหวัด")

# ── 4.8 Utilization per trip ──
print(f"\n📊 Utilization ทุกทริป:")
PUNTHAI_LIMITS_LOC = {'4W':{'max_w':2500,'max_c':5.0},'JB':{'max_w':3500,'max_c':7.0},'6W':{'max_w':6000,'max_c':20.0}}
LIMITS_LOC =         {'4W':{'max_w':2500,'max_c':5.0},'JB':{'max_w':3500,'max_c':7.0},'6W':{'max_w':6000,'max_c':20.0}}
low_util = []
for t in sorted(result_df[result_df['Trip'] > 0]['Trip'].unique()):
    tdata = result_df[result_df['Trip'] == t]
    truck_raw = str(tdata.iloc[0].get('Truck','6W')).split()[0]
    truck = 'JB' if truck_raw == '4WJ' else truck_raw
    if truck not in LIMITS_LOC: truck = '6W'
    is_pt = all(str(r).upper() in ('211','PUNTHAI') for r in tdata.get('BU', pd.Series()))
    lim = PUNTHAI_LIMITS_LOC[truck] if is_pt else LIMITS_LOC[truck]
    tw = tdata['Weight'].sum()
    tc = tdata['Cube'].sum()
    wu = tw / lim['max_w'] * 100
    cu = tc / lim['max_c'] * 100
    util = max(wu, cu)
    dist_max = tdata['_distance_from_dc'].max() if '_distance_from_dc' in tdata.columns else 0
    provs = list(tdata['_province'].dropna().unique()) if '_province' in tdata.columns else []
    flag = ''
    if util < 60: flag = ' ⬇LOW'
    if util < 30: flag = ' ⬇⬇VERY_LOW'
    print(f"  T{t:3d} {truck:<4} {len(tdata):3d}สาขา {dist_max:6.0f}km  W{wu:5.1f}% C{cu:5.1f}%{flag}  [{', '.join(provs[:3])}]")
    if util < 60 and len(tdata) > 1:
        low_util.append((t, truck, round(util,1), len(tdata), round(dist_max,1), provs))
print(f"\n  ✍️ ทริป util<60% (>1สาขา): {len(low_util)} ทริป")

# ──────────────────────────────────────────────
# 5. สรุปทริป (แสดง 10 ทริปแรก)
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("📋 สรุป 10 ทริปแรก (ลำดับตามระยะทาง)")
print("=" * 70)
print(f"{'Trip':>5}  {'Truck':<8}  {'สาขา':>5}  {'Weight':>8}  {'Cube':>7}  {'MaxDist':>8}  {'Region'}")
print("-" * 70)
for t in sorted(result_df[result_df['Trip'] > 0]['Trip'].unique())[:10]:
    tdata = result_df[result_df['Trip'] == t]
    truck = str(tdata.iloc[0].get('Truck', '?')).split()[0] if 'Truck' in tdata.columns else '?'
    n = len(tdata)
    tw = tdata['Weight'].sum()
    tc = tdata['Cube'].sum()
    md = tdata['_distance_from_dc'].max() if '_distance_from_dc' in tdata.columns else 0
    reg = ''
    if '_province' in tdata.columns:
        provs = tdata['_province'].dropna().unique()
        regs = set()
        for p in provs:
            r = app.get_region_name(str(p))
            if r and r not in ('', 'ไม่ระบุ'):
                regs.add(r)
        reg = '/'.join(sorted(regs))
    print(f"{t:>5}  {truck:<8}  {n:>5}  {tw:>8.0f}  {tc:>7.2f}  {md:>8.1f}  {reg}")

# ── สรุปรถแต่ละประเภท ──
if 'Truck' in result_df.columns:
    print("\n📊 รถแต่ละประเภท:")
    trucks = result_df[result_df['Trip'] > 0].drop_duplicates('Trip')['Truck'].str.split().str[0].value_counts()
    for v, c in trucks.items():
        print(f"   {v}: {c} ทริป")

# ──────────────────────────────────────────────
# 6. ผลสรุปสุดท้าย
# ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("🏁 ผลสรุปการทดสอบ")
print("=" * 70)

if warnings:
    print("\n⚠️  คำเตือน:")
    for w in warnings:
        print(f"   {w}")

if errors:
    print("\n❌ พบปัญหา:")
    for e in errors:
        print(f"   {e}")
    print("\n❌ ทดสอบ: ยังมีปัญหาที่ต้องแก้ไข")
    sys.exit(1)
else:
    if not warnings:
        print("\n✅✅ ผ่านทุกเงื่อนไข — พร้อมใช้งาน!")
    else:
        print("\n✅ ผ่านเงื่อนไขหลัก (มีคำเตือนบางรายการ)")
