import pandas as pd
import pickle

# โหลด model เพื่อดูประวัติ
model = pickle.load(open('models/decision_tree_model.pkl', 'rb'))
branch_info = model['branch_info']

codes = ['N802', 'N820', 'NX01', 'N113']
print('ข้อมูลสาขา:')
for code in codes:
    if code in branch_info:
        info = branch_info[code]
        prov = info.get('province', 'N/A')
        print(f'{code}: จังหวัด={prov}')
    else:
        print(f'{code}: ไม่พบใน model')

print('\nชื่อสาขา:')
names = {
    'N802': 'PUN-ตลาดเทศบาลสกลนคร',
    'N820': 'PUN-ถนนเหล่านาดี ขอนแก่น',
    'NX01': 'PUN-ตลาดร้อยช่างมาร์เก็ต',
    'N113': 'PUN-บางแวก(พุทธมณฑลสาย3)'
}
for code, name in names.items():
    print(f'{code}: {name}')
