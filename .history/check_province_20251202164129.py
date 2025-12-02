import pickle

model = pickle.load(open('models/decision_tree_model.pkl', 'rb'))
bi = model['branch_info']

print('=== ตรวจสอบจังหวัดของสาขาตัวอย่าง ===')
test_codes = ['11004255', 'NX74', 'NK11', 'ZC005', '11004388', 'W2011', 'W2215']

for c in test_codes:
    if c in bi:
        prov = bi[c].get('province', 'N/A')
        print(f'{c}: {prov}')
    else:
        print(f'{c}: ไม่พบใน model')

# ดูตัวอย่างสาขาที่มีจังหวัดกรุงเทพ
print('\n=== สาขาที่มีจังหวัดกรุงเทพ ===')
for code, info in bi.items():
    prov = info.get('province', '')
    if 'กรุงเทพ' in str(prov):
        print(f'{code}: {prov}')

# ดูตัวอย่างสาขาที่มีจังหวัดสุพรรณบุรี
print('\n=== สาขาที่มีจังหวัดสุพรรณบุรี ===')
for code, info in bi.items():
    prov = info.get('province', '')
    if 'สุพรรณ' in str(prov):
        print(f'{code}: {prov}')
