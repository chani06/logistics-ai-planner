"""
สคริปต์สำหรับ sync ข้อมูลจาก Google Sheets ลง JSON
"""
import json
import os
import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def sync_branch_data_from_sheets():
    """
    ดึงข้อมูลจาก Google Sheets และ sync กับ JSON file
    """
    json_file = 'branch_data.json'
    
    # โหลดข้อมูลเก่าจาก JSON
    existing_data = {}
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            print(f"📦 โหลดข้อมูลเก่า: {len(existing_data)} สาขา")
        except Exception as e:
            print(f"⚠️ ไม่สามารถอ่าน JSON: {e}")
    
    # เชื่อมต่อ Google Sheets
    try:
        print("🔄 เชื่อมต่อ Google Sheets...")
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json', scope)
        client = gspread.authorize(creds)
        
        # เปิด Google Sheets
        spreadsheet_id = '12DmIfECwVpsWfl8rl2r1A_LB4_5XMrmnmwlPUHKNU-o'
        sh = client.open_by_key(spreadsheet_id)
        worksheet = sh.get_worksheet_by_id(876257177)
        
        print(f"✅ เชื่อมต่อสำเร็จ: {sh.title}")
        
        # ดึงข้อมูลทั้งหมด
        data = worksheet.get_all_records()
        df_new = pd.DataFrame(data)
        
        print(f"📥 ดึงข้อมูลจาก Sheets: {len(df_new)} แถว")
        
        # หาคอลัมน์รหัสสาขา
        code_col = None
        for col in df_new.columns:
            if 'Code' in col or 'code' in col or 'รหัส' in col:
                code_col = col
                break
        
        if not code_col:
            print("❌ ไม่พบคอลัมน์รหัสสาขา")
            print(f"คอลัมน์ที่มี: {df_new.columns.tolist()}")
            return None
        
        print(f"📋 ใช้คอลัมน์: {code_col}")
        
        # นับข้อมูลใหม่
        new_count = 0
        updated_count = 0
        unchanged_count = 0
        
        # อัปเดตข้อมูล
        for idx, row in df_new.iterrows():
            code = str(row[code_col]).strip().upper()
            if not code or code == '':
                continue
            
            # แปลง row เป็น dict
            row_dict = row.to_dict()
            
            if code in existing_data:
                # ข้อมูลเก่า - เช็คว่ามีการเปลี่ยนแปลงจริงหรือไม่
                if existing_data[code] != row_dict:
                    existing_data[code] = row_dict
                    updated_count += 1
                else:
                    unchanged_count += 1
            else:
                # ข้อมูลใหม่
                existing_data[code] = row_dict
                new_count += 1
        
        # บันทึกกลับเป็น JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Sync เสร็จสิ้น:")
        print(f"   📊 รวมทั้งหมด: {len(existing_data)} สาขา")
        print(f"   🆕 สาขาใหม่: {new_count}")
        print(f"   🔄 อัปเดต: {updated_count}")
        print(f"   ✔️ ไม่เปลี่ยนแปลง: {unchanged_count}")
        
        # ตรวจสอบ DC วังน้อย
        if '8NVDC011' in existing_data:
            dc = existing_data['8NVDC011']
            print(f"\n🏢 DC วังน้อย (8NVDC011): {dc.get('สาขา', 'N/A')}")
        
        return len(existing_data)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("🔄 เริ่มต้น Sync จาก Google Sheets")
    print("=" * 60)
    result = sync_branch_data_from_sheets()
    if result:
        print(f"\n✅ สำเร็จ! ข้อมูลล่าสุด: {result} สาขา")
    else:
        print("\n❌ Sync ล้มเหลว")
