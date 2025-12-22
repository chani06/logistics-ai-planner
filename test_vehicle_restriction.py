import pandas as pd
from app import load_vehicle_restrictions, get_allowed_vehicle_for_branch

def main():
    # โหลดข้อจำกัดรถจาก Auto planning (1).xlsx
    restrictions = load_vehicle_restrictions('Dc/Auto planning (1).xlsx', 'Info')
    # โหลดตัวอย่าง branch_code/zone จาก test.xlsx (ถ้ามี)
    try:
        df = pd.read_excel('Dc/test.xlsx')
        # สมมติว่ามีคอลัมน์ 'branch_code' และ 'zone' ใน test.xlsx
        code_col = [c for c in df.columns if 'branch' in c.lower() or 'location' in c.lower()][0]
        zone_col = [c for c in df.columns if 'zone' in c.lower()][0]
        for _, row in df.iterrows():
            branch_code = str(row[code_col])
            zone = str(row[zone_col])
            vehicle = get_allowed_vehicle_for_branch(branch_code, zone, restrictions)
            print(f"Branch: {branch_code} | Zone: {zone} | Allowed Vehicle: {vehicle}")
    except Exception as e:
        print("Error reading test.xlsx or missing columns:", e)
        # ทดสอบแบบ manual ถ้าไม่มี test.xlsx
        for branch_code, zone in [('1001', 'CENTRAL'), ('1002', 'NE'), ('1003', 'NORTH')]:
            vehicle = get_allowed_vehicle_for_branch(branch_code, zone, restrictions)
            print(f"Branch: {branch_code} | Zone: {zone} | Allowed Vehicle: {vehicle}")

if __name__ == '__main__':
    main()
