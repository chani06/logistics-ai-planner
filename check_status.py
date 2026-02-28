import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DC_LAT, DC_LON = 14.1794, 100.6481

with open('distance_cache.json','r',encoding='utf-8') as f:
    dc = json.load(f)
print(f"=== distance_cache.json: {len(dc):,} entries ===")
dc_branch = [k for k in dc if k.startswith(f"{DC_LAT},{DC_LON}_")]
print(f"DC->branch forward keys: {len(dc_branch)}")

with open('branch_clusters.json','r',encoding='utf-8') as f:
    bc = json.load(f)
bi = bc['branch_info']
nb = bc['nearby_branches']
print(f"\n=== branch_clusters.json ===")
print(f"branch_info: {len(bi)} branches")

zero=0; road=0; fallback=0; total_nearby=0
for code, items in nb.items():
    for item in items:
        total_nearby += 1
        if item['distance'] == 0: zero += 1
        if item.get('is_road'): road += 1
        else: fallback += 1
print(f"nearby total pairs: {total_nearby:,}")
print(f"  OSRM road: {road:,} ({road*100//max(total_nearby,1)}%)")
print(f"  fallback (hav x1.35): {fallback:,}")
print(f"  dist==0 (bug entries): {zero:,}")

# DC coverage
missing = []
for code, b in bi.items():
    k  = f"{DC_LAT:.4f},{DC_LON:.4f}_{b['lat']:.4f},{b['lon']:.4f}"
    kr = f"{b['lat']:.4f},{b['lon']:.4f}_{DC_LAT:.4f},{DC_LON:.4f}"
    if k not in dc and kr not in dc:
        missing.append(code)
print(f"\nDC->branch OSRM coverage: {len(bi)-len(missing)}/{len(bi)} ({(len(bi)-len(missing))*100//len(bi)}%)")
print(f"Missing DC->branch: {len(missing)}")
if missing[:5]:
    print(f"  Examples: {missing[:5]}")
