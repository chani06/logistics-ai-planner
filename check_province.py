import pickle

model = pickle.load(open('models/decision_tree_model.pkl', 'rb'))
bi = model['branch_info']
trip_pairs = model['trip_pairs']

print('=== ตรวจสอบจังหวัดของสาขาตัวอย่าง ===')
test_codes = ['11004255', 'NX74', 'NK11', 'ZC005', '11004388', 'W2011', 'W2215']

for c in test_codes:
    if c in bi:
        prov = bi[c].get('province', 'N/A')
        print(f'{c}: {prov}')
    else:
        print(f'{c}: ไม่พบใน model')

# หา trip_pairs ที่มีสาขากรุงเทพคู่กับสุพรรณบุรี
print('\n=== trip_pairs ที่มีสาขากรุงเทพคู่กับสุพรรณบุรี ===')
bkk_codes = set()
suph_codes = set()

for code, info in bi.items():
    prov = info.get('province', '')
    if 'กรุงเทพ' in str(prov):
        bkk_codes.add(code)
    if 'สุพรรณ' in str(prov):
        suph_codes.add(code)

count = 0
for pair in trip_pairs:
    c1, c2 = pair
    if (c1 in bkk_codes and c2 in suph_codes) or (c2 in bkk_codes and c1 in suph_codes):
        p1 = bi.get(c1, {}).get('province', 'N/A')
        p2 = bi.get(c2, {}).get('province', 'N/A')
        print(f'{c1} ({p1}) <-> {c2} ({p2})')
        count += 1

print(f'\nพบทั้งหมด {count} คู่')

# ตรวจสอบว่ามี pair ที่มี code ขึ้นต้น 110 กับ สุพรรณบุรี
print('\n=== trip_pairs ที่มี code ขึ้นต้น 110 กับสุพรรณบุรี ===')
for pair in trip_pairs:
    c1, c2 = pair
    if c1.startswith('110') or c2.startswith('110'):
        if c1 in suph_codes or c2 in suph_codes:
            p1 = bi.get(c1, {}).get('province', 'N/A')
            p2 = bi.get(c2, {}).get('province', 'N/A')
            print(f'{c1} ({p1}) <-> {c2} ({p2})')
