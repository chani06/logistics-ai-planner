import pandas as pd, json, sys, io, os
from collections import defaultdict, Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

df = pd.read_excel('Truck Fulfillment(1).xlsx', sheet_name='Master', header=0)
df.columns = [str(c).strip().replace('\n','') for c in df.columns]
df['Date'] = pd.to_datetime(df['Date'])
df_clean = df[df['BU'] != 'PROJECT'].copy()
df_clean['code'] = df_clean['รหัส WMS'].astype(str).str.strip().str.upper()

TRUCK_MAX_DROPS = {'4W': 12, 'JB': 12, '6W': 999}

trip_pairs = {}
skipped = 0
for (date, trip_no), grp in df_clean.groupby(['Date', 'Trip no']):
    truck = grp['Truck Type'].iloc[0]
    max_d = TRUCK_MAX_DROPS.get(truck, 12)
    codes = sorted(grp['code'].unique().tolist())
    if len(codes) > max_d:
        skipped += 1
        continue
    for i in range(len(codes)):
        for j in range(i+1, len(codes)):
            a, b = codes[i], codes[j]
            k = a + '|' + b
            trip_pairs[k] = trip_pairs.get(k, 0) + 1

print(f'Valid trips seeded: {len(trip_pairs)} pairs | Skipped large trips: {skipped}')

# re-seed trip_history.json
sessions = [{
    'date': '2026-02-25T00:00:00',
    'source': 'branch_history_fulfillment',
    'trips': int(df_clean.groupby(['Date', 'Trip no']).ngroups),
    'pair_count': len(trip_pairs),
    'note': 'filtered: trips <= max_drops only',
}]
tdata_new = {
    'pair_freq': trip_pairs,
    'sessions': sessions,
    'total_sessions': len(sessions),
}
with open('trip_history.json', 'w', encoding='utf-8') as f:
    json.dump(tdata_new, f, ensure_ascii=False, indent=2)
print('Saved trip_history.json:', len(tdata_new['pair_freq']), 'pairs')
