import pandas as pd

# --- 1. อ่านข้อจำกัดรถจาก Autoplan.xlsx ชีต Info ---
# ตัวอย่างคอลัมน์: Branch_Code, MaxTruckType
# (ปรับชื่อคอลัมน์ตามไฟล์จริง)

def load_vehicle_restrictions(filepath='Autoplan.xlsx', sheet='Info'):
    df = pd.read_excel(filepath, sheet_name=sheet)
    # สมมติคอลัมน์ชื่อ 'Branch_Code' และ 'MaxTruckType'
    # MaxTruckType: 4W, JB, 6W (หรืออื่นๆ)
    restrictions = {}
    for _, row in df.iterrows():
        code = str(row['Branch_Code']).strip()
        max_truck = str(row['MaxTruckType']).strip().upper()
        # กำหนด allowed vehicles ตาม max_truck
        if max_truck == '4W':
            allowed = ['4W']
        elif max_truck == 'JB':
            allowed = ['4W', 'JB']
        elif max_truck == '6W':
            allowed = ['4W', 'JB', '6W']
        else:
            allowed = ['4W', 'JB', '6W']  # default
        restrictions[code] = allowed
    return restrictions

# --- 2. ฟังก์ชันเลือกขนาดรถโดยเช็คข้อจำกัดจากไฟล์ ---
def get_allowed_vehicle_for_branch(branch_code, zone, restrictions):
    allowed = restrictions.get(str(branch_code).strip(), ['4W', 'JB', '6W'])
    # ข้อจำกัดโซน (เช่น CENTRAL ห้าม 6W)
    if zone == 'CENTRAL' and '6W' in allowed:
        allowed = [v for v in allowed if v != '6W']
    # เลือกขนาดใหญ่สุดที่อนุญาต
    for v in ['6W', 'JB', '4W']:
        if v in allowed:
            return v
    return allowed[0]

# --- 3. ตัวอย่างการใช้งาน ---
if __name__ == '__main__':
    restrictions = load_vehicle_restrictions('Autoplan.xlsx', 'Info')
    # ตัวอย่าง branch/zone
    print(get_allowed_vehicle_for_branch('1001', 'CENTRAL', restrictions))
    print(get_allowed_vehicle_for_branch('1002', 'NE', restrictions))
