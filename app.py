# Logistics Planner
# Version: 2025-12-26-v3.4

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import glob
from datetime import datetime, time as datetime_time, timedelta
import io
from math import radians, sin, cos, sqrt, atan2
import json
import time as time_module
import sys

# ฟังก์ชัน safe print สำหรับ Windows console
def safe_print(*args, **kwargs):
    """Print with fallback encoding for Windows console"""
    text = ' '.join(str(arg) for arg in args)
    try:
        # ใช้ buffer แทน stdout โดยตรงเพื่อหลีกเลี่ยง Streamlit wrapper
        output = sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else sys.stdout
        if hasattr(output, 'write'):
            if isinstance(output, (type(sys.stdout.buffer), type(sys.stderr.buffer))):
                output.write((text + '\n').encode('utf-8', errors='replace'))
            else:
                output.write(text + '\n')
            output.flush() if hasattr(output, 'flush') else None
    except Exception:
        try:
            # Fallback: ใช้ ASCII
            sys.__stdout__.write(text.encode('ascii', 'replace').decode('ascii') + '\n')
            sys.__stdout__.flush()
        except Exception:
            pass

# Map visualization
try:
    import folium
    from folium import plugins
    from streamlit_folium import folium_static  # ใช้ folium_static แทน st_folium เพื่อไม่ให้โหลดซ้ำ
    import requests
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

# Google Sheets Integration
try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    
    # ตรวจสอบว่ามีไฟล์ credentials.json หรือ Streamlit secrets
    credentials_file = 'credentials.json'
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    
    creds = None
    
    # 1️⃣ ลองใช้ Streamlit Secrets ก่อน (สำหรับ Streamlit Cloud)
    try:
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            creds = ServiceAccountCredentials.from_json_keyfile_dict(
                dict(st.secrets['gcp_service_account']), 
                scope
            )
            safe_print("✅ ใช้ credentials จาก Streamlit Secrets")
    except Exception as e:
        safe_print(f"⚠️ Streamlit Secrets ไม่พร้อมใช้งาน: {e}")
    
    # 2️⃣ ถ้าไม่มี secrets ให้ใช้ไฟล์ local
    if creds is None:
        if os.path.exists(credentials_file):
            try:
                creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
                safe_print(f"✅ ใช้ credentials จาก {credentials_file}")
            except Exception as e:
                safe_print(f"⚠️ ไม่สามารถอ่านไฟล์ {credentials_file}: {e}")
        else:
            safe_print(f"⚠️ ไม่พบ {credentials_file} และไม่มี Streamlit Secrets")
            safe_print(f"💡 ดูวิธีตั้งค่าได้ที่: CREDENTIALS_SETUP.md")
    
    # เชื่อมต่อ Google Sheets
    if creds:
        try:
            gc = gspread.authorize(creds)
            SPREADSHEET_ID = '12DmIfECwVpsWfl8rl2r1A_LB4_5XMrmnmwlPUHKNU-o'
            sh = gc.open_by_key(SPREADSHEET_ID)
            SHEETS_AVAILABLE = True
            safe_print("✅ เชื่อมต่อ Google Sheets สำเร็จ")
        except Exception as e:
            safe_print(f"⚠️ Google Sheets Error: {e}")
            safe_print(f"💡 ตรวจสอบ credentials หรือดูคู่มือที่ CREDENTIALS_SETUP.md")
            SHEETS_AVAILABLE = False
            gc = None
            sh = None
    else:
        SHEETS_AVAILABLE = False
        gc = None
        sh = None
        
except ImportError:
    safe_print("⚠️ ไม่พบ gspread library - ติดตั้งด้วย: pip install gspread oauth2client")
    SHEETS_AVAILABLE = False
    gc = None
    sh = None

# Auto-refresh component
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False
    # แสดง warning เฉพาะใน local dev (ไม่แสดงใน deployment)
    if os.environ.get('ENVIRONMENT') != 'production':
        pass  # ไม่แสดง warning - ใช้ manual refresh แทน

# ==========================================
# CACHE SYSTEM - ป้องกันการโหลดซ้ำและเพิ่มความเร็ว
# ==========================================
USE_CACHE = True  # เปิดใช้งาน cache system
DISTANCE_CACHE_FILE = 'distance_cache.json'
ROUTE_CACHE_FILE = 'route_cache.json'

# โหลด cache จากไฟล์
@st.cache_data(show_spinner=False)
def load_distance_cache():
    """โหลด distance cache จากไฟล์"""
    if os.path.exists(DISTANCE_CACHE_FILE):
        try:
            with open(DISTANCE_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_distance_cache(cache_dict):
    """บันทึก distance cache ลงไฟล์"""
    try:
        with open(DISTANCE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        safe_print(f"⚠️ ไม่สามารถบันทึก distance cache: {e}")

@st.cache_data(show_spinner=False)
def load_route_cache():
    """โหลด route cache จากไฟล์"""
    if os.path.exists(ROUTE_CACHE_FILE):
        try:
            with open(ROUTE_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_route_cache(cache_dict):
    """บันทึก route cache ลงไฟล์ (compact JSON เพื่อความเร็ว)"""
    try:
        with open(ROUTE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_dict, f, ensure_ascii=False, separators=(',', ':'))
    except Exception as e:
        safe_print(f"⚠️ ไม่สามารถบันทึก route cache: {e}")

# โหลด cache ตอนเริ่มต้น
if USE_CACHE:
    DISTANCE_CACHE = load_distance_cache()
    ROUTE_CACHE_DATA = load_route_cache()
    
    # แยกประเภท cache
    dc_distances = sum(1 for k in DISTANCE_CACHE.keys() if k.startswith('14.1') or k.startswith('14.2'))
    branch_distances = len(DISTANCE_CACHE) - dc_distances
    
    safe_print(f"✅ โหลด distance_cache.json: {len(DISTANCE_CACHE):,} รายการ")
    if dc_distances > 0 or branch_distances > 0:
        safe_print(f"   - DC→สาขา: ~{dc_distances:,} รายการ")
        safe_print(f"   - สาขา↔สาขา: ~{branch_distances:,} รายการ")
    safe_print(f"✅ โหลด route_cache.json: {len(ROUTE_CACHE_DATA):,} เส้นทาง")
else:
    DISTANCE_CACHE = {}
    ROUTE_CACHE_DATA = {}

# ==========================================
# GOOGLE SHEETS SYNC FUNCTION
# ==========================================
def sync_branch_data_from_sheets():
    """
    ดึงข้อมูลจาก Google Sheets และ sync กับ JSON file
    ใช้รหัสสาขา (Code/Plan Code) เป็น key หลัก
    
    Returns:
        DataFrame หรือ None ถ้าล้มเหลว
    """
    global SHEETS_AVAILABLE, sh
    
    json_file = 'branch_data.json'
    
    # โหลดข้อมูลเก่าจาก JSON
    existing_data = {}
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except Exception as e:
            safe_print(f"⚠️ ไม่สามารถอ่าน JSON: {e}")
    
    # ถ้าไม่มี Google Sheets ให้ใช้ข้อมูลเก่า
    if not SHEETS_AVAILABLE or sh is None:
        if existing_data:
            safe_print(f"⚠️ Google Sheets ไม่พร้อม - ใช้ข้อมูลจาก JSON ({len(existing_data)} สาขา)")
            df = pd.DataFrame.from_dict(existing_data, orient='index')
            # ตรวจสอบว่ามีคอลัมน์ Plan Code หรือไม่
            if 'Plan Code' not in df.columns:
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Plan Code'}, inplace=True)
            else:
                df.reset_index(drop=True, inplace=True)
            return df
        else:
            safe_print("❌ ไม่พบข้อมูล: ไม่มี Google Sheets และไม่มี JSON cache")
            return pd.DataFrame()  # Return empty DataFrame แทน None
    
    try:
        # ดึงข้อมูลจาก Sheets (GID: 876257177)
        worksheet = None
        for ws in sh.worksheets():
            if ws.id == 876257177:
                worksheet = ws
                break
        
        if worksheet is None:
            worksheet = sh.get_worksheet(0)
        
        # ดึงข้อมูลทั้งหมด
        data = worksheet.get_all_values()
        if not data or len(data) < 2:
            return None
        
        # สร้าง DataFrame
        headers = data[0]
        df_new = pd.DataFrame(data[1:], columns=headers)
        
        # หา column รหัสสาขา
        code_col = None
        for col in ['Code', 'Plan Code', 'รหัสสาขา', 'สาขา']:
            if col in df_new.columns:
                code_col = col
                break
        
        if not code_col:
            safe_print("❌ ไม่พบคอลัมน์รหัสสาขา")
            return None
        
        # นับข้อมูลใหม่
        new_count = 0
        updated_count = 0
        
        # อัปเดตข้อมูลจาก Google Sheets (รวม DC วังน้อยที่มีอยู่ใน Sheets)
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
                # ถ้าข้อมูลเหมือนเดิม ไม่นับเป็น update
            else:
                # ข้อมูลใหม่ - เพิ่ม
                existing_data[code] = row_dict
                new_count += 1
        
        # บันทึกกลับเป็น JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
        safe_print(f"✅ Sync เสร็จสิ้น: {new_count} สาขาใหม่, {updated_count} สาขาอัปเดต, รวม {len(existing_data)} สาขา")
        
        # แปลงกลับเป็น DataFrame (ใช้ index เป็น Plan Code)
        df = pd.DataFrame.from_dict(existing_data, orient='index')
        
        # ตรวจสอบว่ามีคอลัมน์ Plan Code หรือไม่ ถ้าไม่มีให้สร้างจาก index
        if 'Plan Code' not in df.columns and code_col in df.columns:
            df['Plan Code'] = df[code_col]
        elif 'Plan Code' not in df.columns:
            # ใช้ index เป็น Plan Code
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Plan Code'}, inplace=True)
        else:
            df.reset_index(drop=True, inplace=True)
        
        return df
        
    except Exception as e:
        safe_print(f"❌ Error: {e}")
        # ถ้าเกิด error ให้ใช้ข้อมูลเก่า
        if existing_data:
            safe_print(f"📦 ใช้ข้อมูลเก่าจาก JSON")
            df = pd.DataFrame.from_dict(existing_data, orient='index')
            # ตรวจสอบว่ามีคอลัมน์ Plan Code หรือไม่
            if 'Plan Code' not in df.columns:
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Plan Code'}, inplace=True)
            else:
                df.reset_index(drop=True, inplace=True)
            return df
        return None

# ==========================================
# CONFIG
# ==========================================
MODEL_PATH = 'models/decision_tree_model.pkl'

# ขีดจำกัดรถแต่ละประเภท (มาตรฐาน)
LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5.0, 'max_drops': 12},   # ไม่เกิน 12 จุด, Cube ≤ 5
    'JB': {'max_w': 3500, 'max_c': 7.0, 'max_drops': 12},   # ไม่เกิน 12 จุด, Cube ≤ 7
    '6W': {'max_w': 6000, 'max_c': 20.0, 'max_drops': 999}  # ไม่จำกัดจุด, Cube ต้องเต็ม, Weight ≤ 6000
}

# 🔒 ขีดจำกัดสำหรับ Punthai ล้วน (ห้ามเกิน 100%)
PUNTHAI_LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5.0, 'max_drops': 5},   # Punthai ล้วน 4W: สูงสุด 5 สาขา
    'JB': {'max_w': 3500, 'max_c': 7.0, 'max_drops': 7},  # Punthai ล้วน JB: สูงสุด 7 สาขา
    '6W': {'max_w': 6000, 'max_c': 20.0, 'max_drops': 999}
}

#  Geographic Clustering Config
MAX_DISTRICT_DISTANCE_KM = 30  # คนละจังหวัด: ห่างกันเกิน 30km ไม่ควรรวมทริป (จังหวัดเดียวกันสามารถ 80km)

# Utilization Config (ใช้ buffer จากหน้าเว็บเท่านั้น ไม่ fix ตายตัว)
MIN_VEHICLE_UTILIZATION = 0.80  # เป้าหมาย: รถต้องใช้อย่างน้อย 80% (แสดง warning ถ้าต่ำกว่า)

# ==========================================
# REGION ORDER CONFIG (Far-to-Near Sorting)
# ==========================================
# ลำดับการจัด: เหนือ → อีสาน → ใต้ → ตะวันออก → กลาง
REGION_ORDER = {
    'เหนือ': 1, 'NORTH': 1,
    'อีสาน': 2, 'NE': 2,
    'ใต้': 3, 'SOUTH': 3,
    'ตะวันออก': 4, 'EAST': 4,
    'ตะวันตก': 5, 'WEST': 5,
    'กลาง': 6, 'CENTRAL': 6,
    'ไม่ระบุ': 99
}

# รายการสาขาที่ไม่ต้องการจัดส่ง (ตัดออก)
EXCLUDE_BRANCHES = ['DC011', 'PTDC', 'PTG DISTRIBUTION CENTER']

# รายชื่อที่ต้องตัดออก (ใช้ตรวจสอบชื่อ)
EXCLUDE_NAMES = ['Distribution Center', 'PTG Distribution', 'บ.พีทีจี เอ็นเนอยี']

# พิกัด DC วังน้อย (จุดกลาง)
DC_WANG_NOI_LAT = 14.179394
DC_WANG_NOI_LON = 100.648149

# ==========================================
# 🚛 HIGHWAY-BASED LOGISTICS ROUTES & ZONES
# ==========================================
# หลักการ: "อย่าลากเส้นตรง ให้ลากตามถนน"
# ยึดเลขทางหลวงแผ่นดินเป็นเกณฑ์หลักในการจัดกลุ่ม ไม่ใช่เขตจังหวัด

HIGHWAY_ROUTES = {
    # สาย 1 (พหลโยธิน): ภาคเหนือตอนบน
    'ROUTE_1_พหลโยธิน': {
        'highway': '1',
        'description': 'กทม → สระบุรี → นครสวรรค์ → ตาก → ลำปาง → เชียงใหม่ → เชียงราย',
        'provinces': ['สระบุรี', 'ลพบุรี', 'นครสวรรค์', 'กำแพงเพชร', 'ตาก', 'ลำปาง', 'ลำพูน', 'เชียงใหม่', 'เชียงราย'],
        'branches': ['พหลโยธิน', 'เอเชีย'],
    },
    # สาย 11 (เอเชียสายเก่า): พิษณุโลก-แพร่-น่าน
    'ROUTE_11_เอเชียสายเก่า': {
        'highway': '11',
        'description': 'นครสวรรค์ → พิจิตร → พิษณุโลก → อุตรดิตถ์ → แพร่',
        'provinces': ['นครสวรรค์', 'พิจิตร', 'พิษณุโลก', 'อุตรดิตถ์', 'แพร่'],
    },
    # สาย 101: แพร่-น่าน
    'ROUTE_101_แพร่น่าน': {
        'highway': '101',
        'description': 'แพร่ → น่าน (หุบเขา)',
        'provinces': ['แพร่', 'น่าน'],
    },
    # สาย 32 (สายเอเชีย): ภาคเหนือตอนล่าง
    'ROUTE_32_สายเอเชีย': {
        'highway': '32',
        'description': 'กทม → อยุธยา → อ่างทอง → สิงห์บุรี → ชัยนาท → นครสวรรค์',
        'provinces': ['พระนครศรีอยุธยา', 'อ่างทอง', 'สิงห์บุรี', 'ชัยนาท', 'นครสวรรค์'],
    },
    # สาย 2 (มิตรภาพ): อีสานเหนือ
    'ROUTE_2_มิตรภาพ': {
        'highway': '2',
        'description': 'สระบุรี → นครราชสีมา → ขอนแก่น → อุดรธานี → หนองคาย',
        'provinces': ['นครราชสีมา', 'ขอนแก่น', 'อุดรธานี', 'หนองคาย', 'เลย', 'หนองบัวลำภู', 'สกลนคร', 'นครพนม', 'มุกดาหาร', 'กาฬสินธุ์', 'มหาสารคาม', 'ร้อยเอ็ด'],
    },
    # สาย 24 (เดชอุดม): อีสานใต้
    'ROUTE_24_เดชอุดม': {
        'highway': '24',
        'description': 'นครราชสีมา → บุรีรัมย์ → สุรินทร์ → อุบลราชธานี',
        'provinces': ['บุรีรัมย์', 'สุรินทร์', 'ศรีสะเกษ', 'อุบลราชธานี', 'ยโสธร', 'อำนาจเจริญ'],
    },
    # สาย 304: ปราจีนบุรี-โคราช
    'ROUTE_304_ปราจีนโคราช': {
        'highway': '304',
        'description': 'ชลบุรี → ปราจีนบุรี → นครราชสีมา',
        'provinces': ['ปราจีนบุรี', 'นครราชสีมา'],
    },
    # สาย 4 (เพชรเกษม): ภาคใต้
    'ROUTE_4_เพชรเกษม': {
        'highway': '4',
        'description': 'กทม → เพชรบุรี → ประจวบฯ → ชุมพร → สุราษฎร์ → นครศรีฯ → สงขลา',
        'provinces': ['เพชรบุรี', 'ประจวบคีรีขันธ์', 'ชุมพร', 'ระนอง', 'สุราษฎร์ธานี', 'นครศรีธรรมราช', 'พัทลุง', 'สงขลา', 'ปัตตานี', 'ยะลา', 'นราธิวาส'],
    },
    # สาย 401/402: อันดามัน
    'ROUTE_401_อันดามัน': {
        'highway': '401/402',
        'description': 'สุราษฎร์ → กระบี่ → ภูเก็ต',
        'provinces': ['กระบี่', 'พังงา', 'ภูเก็ต', 'ตรัง', 'สตูล'],
    },
    # สาย 3 (สุขุมวิท): ภาคตะวันออก
    'ROUTE_3_สุขุมวิท': {
        'highway': '3',
        'description': 'กทม → ชลบุรี → ระยอง → จันทบุรี → ตราด',
        'provinces': ['ชลบุรี', 'ระยอง', 'จันทบุรี', 'ตราด'],
    },
    # สาย 331/344: EEC
    'ROUTE_331_EEC': {
        'highway': '331/344',
        'description': 'ชลบุรี-ระยอง (เขตเศรษฐกิจพิเศษ)',
        'provinces': ['ชลบุรี', 'ระยอง'],
    },
    # สาย 9 (กาญจนาภิเษก): ปริมณฑลด้านเหนือ
    'ROUTE_9_กาญจนาภิเษก': {
        'highway': '9',
        'description': 'รอบนอกกรุงเทพ นนทบุรี-ปทุมธานี-สมุทรปราการ',
        'provinces': ['กรุงเทพมหานคร', 'นนทบุรี', 'ปทุมธานี', 'สมุทรปราการ'],
    },
    # สาย 35 (บรมราชชนนี): ปริมณฑลด้านตะวันตก
    'ROUTE_35_บรมราชชนนี': {
        'highway': '35',
        'description': 'กทม → นนทบุรี → นครปฐม → สมุทรสาคร',
        'provinces': ['กรุงเทพมหานคร', 'นนทบุรี', 'นครปฐม', 'สมุทรสาคร'],
    },
    # สาย 305: สระบุรี-ฉะเชิงเทรา
    'ROUTE_305_สระบุรีฉะเชิงเทรา': {
        'highway': '305',
        'description': 'สระบุรี → นครนายก → ฉะเชิงเทรา',
        'provinces': ['สระบุรี', 'นครนายก', 'ฉะเชิงเทรา', 'ปราจีนบุรี'],
    },
    # สาย 340: สุพรรณบุรี-ชัยนาท
    'ROUTE_340_สุพรรณ': {
        'highway': '340',
        'description': 'สุพรรณบุรี → ชัยนาท → อุทัยธานี',
        'provinces': ['สุพรรณบุรี', 'ชัยนาท', 'อุทัยธานี'],
    },
    # สาย 321: ราชบุรี-กาญจนบุรี
    'ROUTE_321_ราชบุรีกาญจน': {
        'highway': '321',
        'description': 'ราชบุรี → กาญจนบุรี → สังขละบุรี',
        'provinces': ['ราชบุรี', 'กาญจนบุรี'],
    },
}

# ===== NO CROSS-ZONE RULES (ห้ามข้ามโซน) =====
# หลักการ: จังหวัดที่ไปคนละทาง/ต้องข้ามภูเขา
NO_CROSS_ZONE_PAIRS = [
    ('เพชรบูรณ์', 'ชัยภูมิ'),
    ('เพชรบูรณ์', 'เลย'),
    ('ตาก', 'สุโขทัย'),
    ('ใต้ฝั่งอันดามัน', 'ใต้ฝั่งอ่าวไทย'),
    ('กระบี่', 'สุราษฎร์ธานี'),
    # ฝั่งตะวันตกของสมุทรปราการ (เมือง/พระประแดง) ≠ ตะวันออก (ชลบุรี/ฉะเชิงเทรา)
    # บางบ่อ/บางเสาธง แยกออกไปอยู่ ZONE_EAST_สมุทรปราการ แล้ว จึงไม่ต้องปิดกั้นทั้งจังหวัด
]
# ✅ น่าน-แพร่-พะเยา รวมกันได้ (เส้นทาง: DC → แพร่ (สาย 11) → น่าน (สาย 101) หรือ → พะเยา (สาย 1))
# ✅ แพร่-อุตรดิตถ์ รวมได้ (สาย 11 เดียวกัน)

# 🎯 ROUTE GROUPS: กลุ่มจังหวัดที่ควรรวมกัน (ไปทางเดียวกัน)
ROUTE_GROUPS = {
    'ROUTE_สาย1_พะเยา': {
        'provinces': ['พะเยา', 'เชียงราย', 'ลำปาง'],
        'description': 'สาย 1 พหลโยธิน ไปเชียงราย-พะเยา',
        'next_routes': [],  # พะเยาไม่ควรรวมกับจังหวัดอื่นนอกสาย 1
    },
    'ROUTE_สาย11_แพร่น่าน': {
        'provinces': ['แพร่', 'น่าน'],
        'description': 'สาย 11/101 แพร่-น่าน (ต้องผ่านแพร่)',
        'next_routes': ['ROUTE_สาย11_อุตรดิตถ์'],  # รวมกับอุตรดิตถ์ได้
    },
    'ROUTE_สาย11_อุตรดิตถ์': {
        'provinces': ['อุตรดิตถ์', 'สุโขทัย'],
        'description': 'สาย 11 อุตรดิตถ์-สุโขทัย',
        'next_routes': ['ROUTE_สาย11_แพร่น่าน', 'ROUTE_สาย11_พิษณุโลก'],
    },
    'ROUTE_สาย11_พิษณุโลก': {
        'provinces': ['พิษณุโลก', 'พิจิตร'],
        'description': 'สาย 11 พิษณุโลก-พิจิตร',
        'next_routes': ['ROUTE_สาย11_อุตรดิตถ์', 'ROUTE_สาย32_นครสวรรค์'],
    },
}

LOGISTICS_ZONES = {
    # ============ ภาคเหนือ - สาย 1/11/101 ============
    'ZONE_A_พะเยา': {
        'provinces': ['พะเยา'],
        'districts': ['เมืองพะเยา', 'แม่ใจ', 'เชียงคำ', 'เชียงม่วน', 'ดอกคำใต้', 'ปง', 'จุน', 'ภูซาง', 'ภูกามยาว'],
        'highway': '1',
        'priority': 1,
        'distance_from_dc_km': 680,
        'description': 'โซนเหนือสุด สาย 1 ปลายทาง'
    },
    'ZONE_B_น่าน': {
        'provinces': ['น่าน'],
        'districts': ['เมืองน่าน', 'ท่าวังผา', 'ปัว', 'เชียงกลาง', 'ทุ่งช้าง', 'บ่อเกลือ', 'เวียงสา', 'นาน้อย', 'นาหมื่น', 'แม่จริม', 'บ้านหลวง', 'สันติสุข', 'ภูเพียง', 'สองแคว', 'เฉลิมพระเกียรติ'],
        'highway': '101',
        'priority': 2,
        'distance_from_dc_km': 620,
        'description': 'หุบเขา ต้องผ่านแพร่ (สาย 101)'
    },
    'ZONE_C_แพร่': {
        'provinces': ['แพร่'],
        'districts': ['เมืองแพร่', 'สูงเม่น', 'เด่นชัย', 'ร้องกวาง', 'สอง', 'ลอง', 'วังชิ้น', 'หนองม่วงไข่'],
        'highway': '11',
        'priority': 3,
        'distance_from_dc_km': 540,
        'description': 'Gateway เหนือตอนบน สาย 11 → น่าน/พะเยา'
    },
    'ZONE_D_อุตรดิตถ์': {
        'provinces': ['อุตรดิตถ์', 'สุโขทัย'],
        'districts': ['เมืองอุตรดิตถ์', 'ตรอน', 'ท่าปลา', 'น้ำปาด', 'ฟากท่า', 'บ้านโคก', 'พิชัย', 'ลับแล', 'ทองแสนขัน',
                      'เมืองสุโขทัย', 'กงไกรลาศ', 'คีรีมาศ', 'ศรีสำโรง', 'สวรรคโลก', 'ศรีนคร', 'บ้านด่านลานหอย', 'ทุ่งเสลี่ยม', 'ศรีสัชนาลัย'],
        'highway': '11',
        'priority': 4,
        'distance_from_dc_km': 450,
        'description': 'หน้าด่านก่อนเข้าแพร่ สาย 11'
    },
    'ZONE_E1_พิษณุโลก_ในเมือง': {
        'provinces': ['พิษณุโลก'],
        'districts': ['เมืองพิษณุโลก'],
        'subdistricts': ['วัดจันทร์', 'ในเมือง', 'บ้านคลอง', 'หัวรอ', 'บึงพระ', 'ท่าทอง', 'บ้านกร่าง'],
        'highway': '11',
        'priority': 5,
        'distance_from_dc_km': 380,
        'description': 'Hub ใหญ่ โซนในเมือง+ตลาด'
    },
    'ZONE_E2_พิษณุโลก_มหาวิทยาลัย': {
        'provinces': ['พิษณุโลก'],
        'districts': ['เมืองพิษณุโลก'],
        'subdistricts': ['ท่าโพธิ์', 'อรัญญิก', 'แม่กา', 'สมอแข', 'บ้านป่า'],
        'highway': '11',
        'priority': 6,
        'distance_from_dc_km': 385,
        'description': 'โซน ม.นเรศวร'
    },
    'ZONE_E3_พิษณุโลก_ตะวันออก': {
        'provinces': ['พิษณุโลก', 'เพชรบูรณ์'],
        'districts': ['วังทอง', 'พรหมพิราม', 'เนินมะปราง', 'บางระกำ', 'ชาติตระการ', 'นครไทย',
                      'หล่มสัก', 'หล่มเก่า', 'เขาค้อ'],
        'highway': '12',
        'priority': 7,
        'distance_from_dc_km': 400,
        'description': 'โซนตะวันออก สาย 12 ไปเขาค้อ'
    },
    'ZONE_F1_พิจิตร_สายหลัก': {
        'provinces': ['พิจิตร'],
        'districts': ['เมืองพิจิตร', 'สากเหล็ก', 'สามง่าม', 'วังทรายพูน'],
        'highway': '11',
        'priority': 8,
        'distance_from_dc_km': 330,
        'description': 'พิจิตรสายหลัก สาย 11'
    },
    'ZONE_F2_พิจิตร_ตะวันออก': {
        'provinces': ['พิจิตร'],
        'districts': ['ตะพานหิน', 'ทับคล้อ', 'ดงเจริญ', 'บางมูลนาก'],
        'highway': '113',
        'priority': 9,
        'distance_from_dc_km': 340,
        'description': 'พิจิตรตะวันออก สาย 113'
    },
    'ZONE_F3_พิจิตร_สาย117': {
        'provinces': ['พิจิตร'],
        'districts': ['โพธิ์ประทับช้าง', 'บึงนาราง', 'วชิรบารมี', 'โพทะเล'],
        'highway': '117',
        'priority': 10,
        'distance_from_dc_km': 320,
        'description': 'พิจิตรสาย 117'
    },
    'ZONE_G_นครสวรรค์': {
        'provinces': ['นครสวรรค์'],
        'districts': ['เมืองนครสวรรค์', 'หนองบัว', 'ท่าตะโก', 'ไพศาลี', 'ตาคลี', 'บรรพตพิสัย', 'ชุมตาบง', 'ลาดยาว', 'ตากฟ้า', 'พยุหะคีรี', 'โกรกพระ', 'เก้าเลี้ยว', 'ชุมแสง', 'แม่วงก์', 'แม่เปิน'],
        'highway': '1/32',
        'priority': 11,
        'distance_from_dc_km': 240,
        'description': 'ประตูเหนือ สาย 1/32'
    },
    # ============ ภาคอีสาน - สาย 2/24 (เพิ่มโซนย่อย) ============
    'ZONE_H1_โคราช_เมือง': {
        'provinces': ['นครราชสีมา'],
        'districts': ['เมืองนครราชสีมา', 'ปักธงชัย'],
        'highway': '2',
        'priority': 12,
        'distance_from_dc_km': 260,
        'description': 'โคราชในเมือง ประตูอีสาน'
    },
    'ZONE_H2_โคราช_ตะวันออก': {
        'provinces': ['นครราชสีมา'],
        'districts': ['บัวใหญ่', 'ครบุรี', 'สีคิ้ว', 'สูงเนิน', 'โนนสูง', 'โนนแดง', 'ด่านขุนทด'],
        'highway': '2/304',
        'priority': 12,
        'distance_from_dc_km': 280,
        'description': 'โคราชตะวันออก-เขาใหญ่'
    },
    'ZONE_H3_โคราช_เหนือ': {
        'provinces': ['นครราชสีมา'],
        'districts': ['พิมาย', 'ห้วยแถลง', 'บ้านเหลื่อม', 'โชคชัย', 'แก้งสนามนาง', 'เทพารักษ์'],
        'highway': '2',
        'priority': 12,
        'distance_from_dc_km': 270,
        'description': 'โคราชเหนือ-เส้นมิตรภาพ'
    },
    'ZONE_H4_โคราช_ใต้': {
        'provinces': ['นครราชสีมา'],
        'districts': ['ปากช่อง', 'วังน้ำเขียว', 'เฉลิมพระเกียรติ', 'คง', 'ชุมพวง'],
        'highway': '2/304',
        'priority': 12,
        'distance_from_dc_km': 290,
        'description': 'โคราชใต้-เส้น304'
    },
    'ZONE_I1_ขอนแก่น_เมือง': {
        'provinces': ['ขอนแก่น'],
        'districts': ['เมืองขอนแก่น', 'น้ำพอง', 'อุบลรัตน์', 'บ้านไผ่'],
        'highway': '2',
        'priority': 13,
        'distance_from_dc_km': 450,
        'description': 'ขอนแก่นในเมือง Hub อีสานกลาง'
    },
    'ZONE_I2_ขอนแก่น_ใต้': {
        'provinces': ['ขอนแก่น'],
        'districts': ['บ้านฝาง', 'ชนบท', 'พล', 'แวงใหญ่', 'แวงน้อย', 'มัญจาคีรี'],
        'highway': '2',
        'priority': 13,
        'distance_from_dc_km': 470,
        'description': 'ขอนแก่นใต้'
    },
    'ZONE_I3_ขอนแก่น_เหนือ': {
        'provinces': ['ขอนแก่น'],
        'districts': ['กระนวน', 'ซำสูง', 'เปือยน้อย', 'พระยืน', 'ภูผาม่าน', 'หนองสองห้อง', 'หนองเรือ'],
        'highway': '2',
        'priority': 13,
        'distance_from_dc_km': 460,
        'description': 'ขอนแก่นเหนือ'
    },
    'ZONE_I4_มหาสารคาม': {
        'provinces': ['มหาสารคาม'],
        'districts': ['เมืองมหาสารคาม', 'กันทรวิชัย', 'แกดำ', 'โกสุมพิสัย', 'ชื่นชม', 'นาเชือก', 'นาดูน', 'บรบือ', 'พยัคฆภูมิพิสัย', 'วาปีปทุม', 'ยางสีสุราช'],
        'highway': '2',
        'priority': 13,
        'distance_from_dc_km': 480,
        'description': 'มหาสารคาม'
    },
    'ZONE_I5_ร้อยเอ็ด': {
        'provinces': ['ร้อยเอ็ด'],
        'districts': ['เมืองร้อยเอ็ด', 'เกษตรวิสัย', 'ปทุมรัตต์', 'ธวัชบุรี', 'พนมไพร', 'โพนทอง', 'เมืองสรวง', 'เสลภูมิ', 'สุวรรณภูมิ', 'อาจสามารถ'],
        'highway': '2/214',
        'priority': 13,
        'distance_from_dc_km': 510,
        'description': 'ร้อยเอ็ด'
    },
    'ZONE_I6_กาฬสินธุ์': {
        'provinces': ['กาฬสินธุ์'],
        'districts': ['เมืองกาฬสินธุ์', 'กมลาไสย', 'กุฉินารายณ์', 'เขาวง', 'คำม่วง', 'ดอนจาน', 'ท่าคันโท', 'นาคู', 'นามน', 'ยางตลาด', 'ฆ้องชัย', 'ร่องคำ', 'สหัสขันธ์', 'สมเด็จ', 'สามชัย', 'หนองกุงศรี', 'ห้วยเม็ก', 'ห้วยผึ้ง'],
        'highway': '213',
        'priority': 13,
        'distance_from_dc_km': 520,
        'description': 'กาฬสินธุ์'
    },
    'ZONE_J_อุดร': {
        'provinces': ['อุดรธานี', 'หนองคาย', 'หนองบัวลำภู', 'เลย', 'สกลนคร', 'นครพนม', 'บึงกาฬ'],
        'highway': '2',
        'priority': 14,
        'distance_from_dc_km': 560,
        'description': 'อีสานเหนือ สาย 2 ปลายทาง'
    },
    'ZONE_K_อีสานใต้': {
        'provinces': ['บุรีรัมย์', 'สุรินทร์', 'ศรีสะเกษ', 'อุบลราชธานี', 'ยโสธร', 'อำนาจเจริญ', 'มุกดาหาร'],
        'highway': '24',
        'priority': 15,
        'distance_from_dc_km': 500,
        'description': 'อีสานใต้ สาย 24'
    },
    # ============ ภาคตะวันออก - สาย 3 ============
    'ZONE_L_ชลบุรีระยอง': {
        'provinces': ['ชลบุรี', 'ระยอง'],
        # เฉพาะชลบุรีชายฝั่ง (เมือง/ศรีราชา/บ้านบึง/พัทยา/สัตหีบ) + ระยอง ทางสาย 3 (Bang Na-Trat)
        'highway': '3',
        'priority': 16,
        'distance_from_dc_km': 120,
        'description': 'ชลบุรีชายฝั่ง+ระยอง สาย 3 (Bang Na→Trat) EEC'
    },
    # ชลบุรีเหนือ/ในแผ่นดิน: พนัสนิคม/บ่อทอง/หนองใหญ่ เข้าถึงผ่านสาย 304→31 จากฉะเชิงเทรา
    'ZONE_L1_ชลบุรีเหนือ': {
        'provinces': ['ชลบุรี'],
        'districts': ['พนัสนิคม', 'บ่อทอง', 'หนองใหญ่', 'เกาะจันทร์'],
        'highway': '304/331',
        'priority': 16.5,
        'distance_from_dc_km': 90,
        'description': 'ชลบุรีเหนือในแผ่นดิน สาย 304→331 ผ่านฉะเชิงเทรา'
    },
    # สมุทรปราการ ฝั่งตะวันออก: บางบ่อ-บางเสาธง อยู่บนสาย 3 เส้นทางเดียวกับชลบุรี
    'ZONE_EAST_สมุทรปราการ': {
        'provinces': ['สมุทรปราการ'],
        'districts': ['บางบ่อ', 'บางเสาธง'],
        'highway': '3/331',
        'priority': 28,
        'distance_from_dc_km': 55,
        'description': 'สมุทรปราการตะวันออก บางบ่อ-บางเสาธง สาย 3 ต่อชลบุรี'
    },
    'ZONE_M_จันทบุรีตราด': {
        'provinces': ['จันทบุรี', 'ตราด'],
        'highway': '3',
        'priority': 17,
        'distance_from_dc_km': 300,
        'description': 'ตะวันออกไกล สาย 3 ปลายทาง'
    },
    # ============ ภาคใต้ - สาย 4 ============
    'ZONE_N_ใต้ตอนบน': {
        'provinces': ['เพชรบุรี', 'ประจวบคีรีขันธ์', 'ระนอง'],
        'highway': '4',
        'priority': 18,
        'distance_from_dc_km': 400,
        'description': 'ใต้ตอนบน สาย 4 (ไม่รวมชุมพร)'
    },
    # ชุมพร - แยกโซนย่อย
    'ZONE_N1_ชุมพรเหนือ': {
        'provinces': ['ชุมพร'],
        'districts': ['ปะทิว', 'สวี', 'ละแม', 'เมืองชุมพร'],
        'highway': '4',
        'priority': 18.1,
        'distance_from_dc_km': 420,
        'description': 'ชุมพรเหนือ (ปะทิว-สวี-เมือง)'
    },
    'ZONE_N2_ชุมพรใต้': {
        'provinces': ['ชุมพร'],
        'districts': ['ทุ่งตะโก', 'พะโต๊ะ', 'หลังสวน', 'ท่าแซะ'],
        'highway': '4',
        'priority': 18.2,
        'distance_from_dc_km': 450,
        'description': 'ชุมพรใต้ (ทุ่งตะโก-หลังสวน)'
    },
    'ZONE_N3_ชุมพรกลาง': {
        'provinces': ['ชุมพร'],
        'districts': ['บางสะพานน้อย', 'ทับสะแก', 'บางสะพาน'],
        'highway': '4',
        'priority': 18.3,
        'distance_from_dc_km': 440,
        'description': 'ชุมพรกลาง (บางสะพานน้อย-ทับสะแก-บางสะพาน)'
    },
    'ZONE_O_ใต้อ่าวไทย': {
        'provinces': ['สุราษฎร์ธานี', 'นครศรีธรรมราช', 'พัทลุง', 'สงขลา', 'ปัตตานี', 'ยะลา', 'นราธิวาส'],
        'highway': '4',
        'priority': 19,
        'distance_from_dc_km': 700,
        'description': 'ใต้ฝั่งอ่าวไทย สาย 4'
    },
    'ZONE_P_ใต้อันดามัน': {
        'provinces': ['กระบี่', 'พังงา', 'ภูเก็ต', 'ตรัง', 'สตูล'],
        'highway': '401/402',
        'priority': 20,
        'distance_from_dc_km': 850,
        'description': 'ใต้ฝั่งอันดามัน สาย 401/402'
    },
    # ============ ปริมณฑล (แบ่งโซนละเอียด) ============
    'ZONE_BKK_เหนือ': {
        'provinces': ['กรุงเทพมหานคร'],
        'districts': ['หลักสี่', 'ดอนเมือง', 'สายไหม', 'จตุจักร', 'พญาไท', 'ดินแดง', 'ห้วยขวาง'],
        'highway': '1/9',
        'priority': 95,
        'distance_from_dc_km': 50,
        'description': 'กทม.เหนือ ใกล้ทางด่วน-สนามบินดอนเมือง'
    },
    'ZONE_BKK_กลาง': {
        'provinces': ['กรุงเทพมหานคร'],
        'districts': ['ปทุมวัน', 'วัฒนา', 'คลองเตย', 'ราชเทวี', 'บางรัก', 'สาทร', 'ยานนาวา'],
        'highway': 'CBD',
        'priority': 96,
        'distance_from_dc_km': 55,
        'description': 'กทม.กลาง CBD-สีลม-สุขุมวิท'
    },
    'ZONE_BKK_ใต้': {
        'provinces': ['กรุงเทพมหานคร'],
        'districts': ['บางแค', 'ราษฎร์บูรณะ', 'ทุ่งครุ', 'จอมทอง', 'บางบอน', 'บางขุนเทียน', 'ประเวศ', 'หนองจอก'],
        'highway': '35',
        'priority': 97,
        'distance_from_dc_km': 60,
        'description': 'กทม.ใต้ พระราม 2-บางขุนเทียน'
    },
    'ZONE_BKK_ตะวันออก': {
        'provinces': ['กรุงเทพมหานคร'],
        'districts': ['บางกะปิ', 'บึงกุ่ม', 'สะพานสูง', 'ลาดกระบัง', 'มีนบุรี', 'คลองสามวา', 'หนองจอก'],
        'highway': '3/9',
        'priority': 98,
        'distance_from_dc_km': 45,
        'description': 'กทม.ตะวันออก รามอินทรา-ลาดกระบัง'
    },
    'ZONE_NEARBY_นนทบุรี': {
        'provinces': ['นนทบุรี'],
        'districts': ['เมืองนนทบุรี', 'บางกรวย', 'บางใหญ่', 'บางบัวทอง', 'ไทรน้อย', 'ปากเกร็ด'],
        'highway': '9/35',
        'priority': 99,
        'distance_from_dc_km': 35,
        'description': 'นนทบุรี ใกล้กทม'
    },
    'ZONE_NEARBY_ปทุมธานี': {
        'provinces': ['ปทุมธานี'],
        'districts': ['เมืองปทุมธานี', 'คลองหลวง', 'ธัญบุรี', 'หนองเสือ', 'ลาดหลุมแก้ว', 'ลำลูกกา', 'สามโคก'],
        'highway': '1/9/305',
        'priority': 99,
        'distance_from_dc_km': 25,
        'description': 'ปทุมธานี ใกล้ DC วังน้อย'
    },
    'ZONE_NEARBY_สมุทรปราการ': {
        'provinces': ['สมุทรปราการ'],
        'districts': ['เมืองสมุทรปราการ', 'บางพลี', 'พระประแดง', 'พระสมุทรเจดีย์'],
        'highway': '3/9/34',
        'priority': 99,
        'distance_from_dc_km': 40,
        'description': 'สมุทรปราการตะวันตก เมือง-พระประแดง-บางพลี (ฝั่งกทม)'
    },
    'ZONE_NEARBY_สมุทรสาคร': {
        'provinces': ['สมุทรสาคร'],
        'districts': ['เมืองสมุทรสาคร', 'กระทุ่มแบน', 'บ้านแพ้ว'],
        'highway': '35',
        'priority': 99,
        'distance_from_dc_km': 50,
        'description': 'สมุทรสาคร มหาชัย'
    },
    'ZONE_NEARBY_นครปฐม': {
        'provinces': ['นครปฐม'],
        'districts': ['เมืองนครปฐม', 'กำแพงแสน', 'นครชัยศรี', 'ดอนตูม', 'บางเลน', 'สามพราน', 'พุทธมณฑล'],
        'highway': '35/4',
        'priority': 99,
        'distance_from_dc_km': 55,
        'description': 'นครปฐม ม.เกษตร-สามพราน'
    },
    'ZONE_NEARBY_อยุธยา': {
        'provinces': ['พระนครศรีอยุธยา'],
        'districts': ['เมืองพระนครศรีอยุธยา', 'ท่าเรือ', 'นครหลวง', 'บางไทร', 'บางปะหัน', 'บางซ้าย', 'ผักไห่', 'ภาชี', 'ลาดบัวหลวง', 'วังน้อย', 'เสนา', 'บางปะอิน', 'อุทัย'],
        'highway': '1/32',
        'priority': 99,
        'distance_from_dc_km': 20,
        'description': 'อยุธยา-วังน้อย (DC อยู่ที่นี่!)'
    },
    # ============ โซนเพิ่มเติม - จังหวัดใกล้เคียง DC ============
    'ZONE_F4_กำแพงเพชร': {
        'provinces': ['กำแพงเพชร'],
        'highway': '1',
        'priority': 10.5,
        'distance_from_dc_km': 340,
        'description': 'กำแพงเพชร สาย 1 ภาคเหนือตอนล่าง'
    },
    'ZONE_F4_สุโขทัย': {
        'provinces': ['สุโขทัย'],
        'highway': '1',
        'priority': 11,
        'distance_from_dc_km': 400,
        'description': 'สุโขทัย สาย 1 ภาคเหนือตอนล่าง'
    },
    'ZONE_NEARBY_สิงห์บุรี': {
        'provinces': ['สิงห์บุรี'],
        'highway': '1/32',
        'priority': 99,
        'distance_from_dc_km': 100,
        'description': 'สิงห์บุรี สาย 1/32 ภาคกลางตอนบน'
    },
    'ZONE_NEARBY_อ่างทอง': {
        'provinces': ['อ่างทอง'],
        'highway': '1/32',
        'priority': 99,
        'distance_from_dc_km': 80,
        'description': 'อ่างทอง สาย 1/32 ภาคกลางตอนบน'
    },
    'ZONE_NEARBY_ชัยนาท': {
        'provinces': ['ชัยนาท'],
        'highway': '1/32',
        'priority': 99,
        'distance_from_dc_km': 150,
        'description': 'ชัยนาท สาย 1/32 ภาคกลางตอนบน'
    },
    'ZONE_NEARBY_ลพบุรี': {
        'provinces': ['ลพบุรี'],
        'highway': '1/21',
        'priority': 99,
        'distance_from_dc_km': 140,
        'description': 'ลพบุรี สาย 1/21 ภาคกลางตอนบน'
    },
    'ZONE_NEARBY_สระบุรี': {
        'provinces': ['สระบุรี'],
        'highway': '1/2',
        'priority': 99,
        'distance_from_dc_km': 80,
        'description': 'สระบุรี สาย 1/2 ประตูอีสาน-เหนือ'
    },
    # ============ โซนเพิ่มเติม - กรุงเทพฯ แยกละเอียด ============
    'ZONE_BKK_เหนือ': {
        'provinces': ['กรุงเทพมหานคร'],
        'districts': ['จตุจักร', 'หลักสี่', 'ดอนเมือง', 'สายไหม', 'บางเขน', 'ลาดพร้าว', 'บึงกุ่ม', 'บางกะปิ', 'วังทองหลาง', 'คันนายาว'],
        'highway': 'กทม-เหนือ',
        'priority': 99,
        'distance_from_dc_km': 30,
        'description': 'กรุงเทพเหนือ (ใกล้ DC วังน้อย)'
    },
    'ZONE_BKK_ตะวันออก': {
        'provinces': ['กรุงเทพมหานคร'],
        'districts': ['มีนบุรี', 'คลองสามวา', 'หนองจอก', 'ลาดกระบัง', 'สะพานสูง', 'ประเวศ', 'สวนหลวง', 'พระโขนง', 'บางนา', 'คลองเตย', 'วัฒนา'],
        'highway': 'กทม-ตะวันออก',
        'priority': 99,
        'distance_from_dc_km': 55,
        'description': 'กรุงเทพตะวันออก'
    },
    'ZONE_BKK_ใต้': {
        'provinces': ['กรุงเทพมหานคร'],
        'districts': ['บางขุนเทียน', 'บางบอน', 'จอมทอง', 'ราษฎร์บูรณะ', 'ทุ่งครุ', 'บางคอแหลม', 'ยานนาวา', 'สาทร', 'บางรัก', 'ปทุมวัน'],
        'highway': 'กทม-ใต้',
        'priority': 99,
        'distance_from_dc_km': 70,
        'description': 'กรุงเทพใต้'
    },
    'ZONE_BKK_ตะวันตก': {
        'provinces': ['กรุงเทพมหานคร'],
        'districts': ['บางพลัด', 'ตลิ่งชัน', 'ทวีวัฒนา', 'หนองแขม', 'บางแค', 'ภาษีเจริญ', 'บางกอกใหญ่', 'บางกอกน้อย', 'ธนบุรี', 'คลองสาน', 'ราชเทวี', 'พญาไท', 'ดินแดง', 'ห้วยขวาง'],
        'highway': 'กทม-ตะวันตก',
        'priority': 99,
        'distance_from_dc_km': 60,
        'description': 'กรุงเทพตะวันตก/ธนบุรี'
    },
    'ZONE_BKK_กลาง': {
        'provinces': ['กรุงเทพมหานคร'],
        'districts': ['พระนคร', 'ป้อมปราบศัตรูพ่าย', 'สัมพันธวงศ์', 'ดุสิต', 'บางซื่อ'],
        'highway': 'กทม-กลาง',
        'priority': 99,
        'distance_from_dc_km': 50,
        'description': 'กรุงเทพกลาง/เกาะรัตนโกสินทร์'
    },
    # Fallback สำหรับกรุงเทพเมื่อไม่ระบุเขต
    'ZONE_BKK_ทั่วไป': {
        'provinces': ['กรุงเทพมหานคร'],
        'highway': 'กทม',
        'priority': 99,
        'distance_from_dc_km': 50,
        'description': 'กรุงเทพทั่วไป (fallback)'
    },
    # ============ โซนปริมณฑล ============
    'ZONE_CENTRAL_นนทบุรี': {
        'provinces': ['นนทบุรี'],
        'highway': 'กทม',
        'priority': 99,
        'distance_from_dc_km': 40,
        'description': 'นนทบุรี ปริมณฑล'
    },
    'ZONE_CENTRAL_ปทุมธานี': {
        'provinces': ['ปทุมธานี'],
        'highway': 'กทม',
        'priority': 99,
        'distance_from_dc_km': 30,
        'description': 'ปทุมธานี ใกล้ DC'
    },
    'ZONE_CENTRAL_สมุทรปราการ': {
        'provinces': ['สมุทรปราการ'],
        'highway': 'กทม',
        'priority': 99,
        'distance_from_dc_km': 60,
        'description': 'สมุทรปราการ ปริมณฑล'
    },
    'ZONE_CENTRAL_นครปฐม': {
        'provinces': ['นครปฐม'],
        'highway': '35/4',
        'priority': 99,
        'distance_from_dc_km': 55,
        'description': 'นครปฐม สาย 35/4'
    },
    'ZONE_CENTRAL_สมุทรสาคร': {
        'provinces': ['สมุทรสาคร'],
        'highway': '35',
        'priority': 99,
        'distance_from_dc_km': 70,
        'description': 'สมุทรสาคร สาย 35'
    },
    'ZONE_CENTRAL_สมุทรสงคราม': {
        'provinces': ['สมุทรสงคราม'],
        'highway': '35',
        'priority': 99,
        'distance_from_dc_km': 90,
        'description': 'สมุทรสงคราม สาย 35'
    },
    # ============ โซนเพิ่มเติม - ภาคกลางตอนบน ============
    'ZONE_F4_นครสวรรค์': {
        'provinces': ['นครสวรรค์'],
        'highway': '1',
        'priority': 10,
        'distance_from_dc_km': 240,
        'description': 'นครสวรรค์ สาย 1 ภาคกลางตอนบน'
    },
    'ZONE_CENTRAL_อยุธยา': {
        'provinces': ['พระนครศรีอยุธยา'],
        'highway': '1/32',
        'priority': 99,
        'distance_from_dc_km': 25,
        'description': 'อยุธยา ใกล้ DC'
    },
    # ============ โซนเพิ่มเติม - ภาคตะวันออก ============
    'ZONE_EAST_นครนายก': {
        'provinces': ['นครนายก'],
        'highway': '305',
        'priority': 90,
        'distance_from_dc_km': 100,
        'description': 'นครนายก สาย 305'
    },
    'ZONE_EAST_ฉะเชิงเทรา': {
        'provinces': ['ฉะเชิงเทรา'],
        'highway': '304/331',  # เส้นทางเดียวกับชลบุรีเหนือ
        'priority': 17,  # อยู่ถัดชลบุรีเหนือ (สามารถรวมทริปได้ผ่าน highway 304/331)
        'distance_from_dc_km': 80,
        'description': 'ฉะเชิงเทรา สาย 304/331 (รวมทริปกับชลบุรีเหนือได้)'
    },
    'ZONE_EAST_ปราจีนบุรี': {
        'provinces': ['ปราจีนบุรี'],
        'highway': '304',
        'priority': 85,
        'distance_from_dc_km': 130,
        'description': 'ปราจีนบุรี สาย 304'
    },
    'ZONE_EAST_สระแก้ว': {
        'provinces': ['สระแก้ว'],
        'highway': '33',
        'priority': 80,
        'distance_from_dc_km': 220,
        'description': 'สระแก้ว สาย 33'
    },
    # ============ โซนเพิ่มเติม - ภาคตะวันตก ============
    'ZONE_WEST_ราชบุรี': {
        'provinces': ['ราชบุรี'],
        'highway': '4',
        'priority': 85,
        'distance_from_dc_km': 100,
        'description': 'ราชบุรี สาย 4'
    },
    'ZONE_WEST_กาญจนบุรี': {
        'provinces': ['กาญจนบุรี'],
        'highway': '323',
        'priority': 80,
        'distance_from_dc_km': 150,
        'description': 'กาญจนบุรี สาย 323'
    },
    'ZONE_WEST_สุพรรณบุรี': {
        'provinces': ['สุพรรณบุรี'],
        'highway': '340',
        'priority': 85,
        'distance_from_dc_km': 110,
        'description': 'สุพรรณบุรี สาย 340'
    },
    # ============ โซนเพิ่มเติม - ภาคเหนือ ============
    'ZONE_NORTH_พะเยา': {
        'provinces': ['พะเยา'],
        'highway': '1',
        'priority': 1,
        'distance_from_dc_km': 680,
        'description': 'พะเยา สาย 1'
    },
    'ZONE_NORTH_น่าน': {
        'provinces': ['น่าน'],
        'highway': '101',
        'priority': 2,
        'distance_from_dc_km': 620,
        'description': 'น่าน สาย 101'
    },
    'ZONE_NORTH_แพร่': {
        'provinces': ['แพร่'],
        'highway': '11',
        'priority': 3,
        'distance_from_dc_km': 540,
        'description': 'แพร่ สาย 11'
    },
    'ZONE_NORTH_อุตรดิตถ์': {
        'provinces': ['อุตรดิตถ์'],
        'highway': '11',
        'priority': 4,
        'distance_from_dc_km': 450,
        'description': 'อุตรดิตถ์ สาย 11'
    },
    'ZONE_F4_พิษณุโลก': {
        'provinces': ['พิษณุโลก'],
        'highway': '12',
        'priority': 12,
        'distance_from_dc_km': 380,
        'description': 'พิษณุโลก สาย 12'
    },
    'ZONE_F4_พิจิตร': {
        'provinces': ['พิจิตร'],
        'highway': '1',
        'priority': 11,
        'distance_from_dc_km': 330,
        'description': 'พิจิตร สาย 1'
    },
    'ZONE_F4_เพชรบูรณ์': {
        'provinces': ['เพชรบูรณ์'],
        'highway': '21',
        'priority': 15,
        'distance_from_dc_km': 350,
        'description': 'เพชรบูรณ์ สาย 21'
    },
    'ZONE_F4_ตาก': {
        'provinces': ['ตาก'],
        'highway': '1',
        'priority': 13,
        'distance_from_dc_km': 430,
        'description': 'ตาก สาย 1'
    },
    'ZONE_F4_อุทัยธานี': {
        'provinces': ['อุทัยธานี'],
        'highway': '333',
        'priority': 14,
        'distance_from_dc_km': 230,
        'description': 'อุทัยธานี สาย 333'
    },
    'ZONE_NORTH_เชียงใหม่': {
        'provinces': ['เชียงใหม่'],
        'highway': '11',
        'priority': 5,
        'distance_from_dc_km': 700,
        'description': 'เชียงใหม่ สาย 11'
    },
    'ZONE_NORTH_เชียงราย': {
        'provinces': ['เชียงราย'],
        'highway': '1',
        'priority': 3,
        'distance_from_dc_km': 780,
        'description': 'เชียงราย สาย 1'
    },
    'ZONE_NORTH_ลำพูน': {
        'provinces': ['ลำพูน'],
        'highway': '11',
        'priority': 6,
        'distance_from_dc_km': 680,
        'description': 'ลำพูน สาย 11'
    },
    'ZONE_NORTH_ลำปาง': {
        'provinces': ['ลำปาง'],
        'highway': '11',
        'priority': 7,
        'distance_from_dc_km': 600,
        'description': 'ลำปาง สาย 11'
    },
    'ZONE_NORTH_แม่ฮ่องสอน': {
        'provinces': ['แม่ฮ่องสอน'],
        'highway': '108',
        'priority': 2,
        'distance_from_dc_km': 850,
        'description': 'แม่ฮ่องสอน สาย 108 (ไกลสุด)'
    },
    # ============ โซนเพิ่มเติม - ภาคอีสาน ============
    'ZONE_ISAN_นครราชสีมา': {
        'provinces': ['นครราชสีมา'],
        'highway': '2',
        'priority': 50,
        'distance_from_dc_km': 260,
        'description': 'นครราชสีมา สาย 2 (มิตรภาพ)'
    },
    'ZONE_ISAN_ขอนแก่น': {
        'provinces': ['ขอนแก่น'],
        'highway': '2',
        'priority': 45,
        'distance_from_dc_km': 450,
        'description': 'ขอนแก่น สาย 2 (มิตรภาพ)'
    },
    'ZONE_ISAN_ชัยภูมิ': {
        'provinces': ['ชัยภูมิ'],
        'highway': '201',
        'priority': 48,
        'distance_from_dc_km': 340,
        'description': 'ชัยภูมิ สาย 201'
    },
    'ZONE_ISAN_กาฬสินธุ์': {
        'provinces': ['กาฬสินธุ์'],
        'highway': '12',
        'priority': 40,
        'distance_from_dc_km': 500,
        'description': 'กาฬสินธุ์ สาย 12'
    },
    'ZONE_ISAN_มหาสารคาม': {
        'provinces': ['มหาสารคาม'],
        'highway': '2',
        'priority': 42,
        'distance_from_dc_km': 470,
        'description': 'มหาสารคาม สาย 2'
    },
    'ZONE_ISAN_ร้อยเอ็ด': {
        'provinces': ['ร้อยเอ็ด'],
        'highway': '23',
        'priority': 38,
        'distance_from_dc_km': 520,
        'description': 'ร้อยเอ็ด สาย 23'
    },
    # ============ โซนเพิ่มเติม - ภาคใต้ตอนบน ============
    'ZONE_SOUTH_ชุมพร': {
        'provinces': ['ชุมพร'],
        'highway': '4',
        'priority': 60,
        'distance_from_dc_km': 470,
        'description': 'ชุมพร สาย 4 (ประตูใต้)'
    },
}

# ==========================================
# ZONE/REGION CONFIG - รหัสภาคและจังหวัด
# ==========================================
# รหัสภาค: 1=กลาง, 2=ตะวันออก, 3=ตะวันตก, 4=เหนือ, 5=อีสาน, 6=ใต้
REGION_CODE = {
    # ภาคกลาง (รหัส 1)
    'กรุงเทพมหานคร': '10', 'กรุงเทพฯ': '10',
    'นนทบุรี': '11',
    'ปทุมธานี': '12',
    'พระนครศรีอยุธยา': '13', 'อยุธยา': '13',
    'สระบุรี': '14',
    'ลพบุรี': '15',
    'สิงห์บุรี': '16',
    'อ่างทอง': '17',
    'ชัยนาท': '18',
    'นครปฐม': '19',
    'สมุทรปราการ': '1A',
    'สมุทรสาคร': '1B',
    'สมุทรสงคราม': '1C',
    
    # ภาคตะวันออก (รหัส 2)
    'ชลบุรี': '20',
    'ระยอง': '21',
    'จันทบุรี': '22',
    'ตราด': '23',
    'ฉะเชิงเทรา': '24',
    'ปราจีนบุรี': '25',
    'สระแก้ว': '26',
    'นครนายก': '27',
    
    # ภาคตะวันตก (รหัส 3)
    'ราชบุรี': '30',
    'กาญจนบุรี': '31',
    'สุพรรณบุรี': '32',
    'เพชรบุรี': '33',
    'ประจวบคีรีขันธ์': '34',
    
    # ภาคเหนือ (รหัส 4)
    'นครสวรรค์': '40',
    'อุทัยธานี': '41',
    'กำแพงเพชร': '42',
    'ตาก': '43',
    'สุโขทัย': '44',
    'พิษณุโลก': '45',
    'พิจิตร': '46',
    'เพชรบูรณ์': '47',
    'อุตรดิตถ์': '48',
    'แพร่': '49',
    'น่าน': '4A',
    'พะเยา': '4B',
    'เชียงราย': '4C',
    'เชียงใหม่': '4D',
    'แม่ฮ่องสอน': '4E',
    'ลำพูน': '4F',
    'ลำปาง': '4G',
    
    # ภาคตะวันออกเฉียงเหนือ/อีสาน (รหัส 5)
    'นครราชสีมา': '50', 'โคราช': '50',
    'บุรีรัมย์': '51',
    'สุรินทร์': '52',
    'ศรีสะเกษ': '53',
    'อุบลราชธานี': '54',
    'ยโสธร': '55',
    'ชัยภูมิ': '56',
    'อำนาจเจริญ': '57',
    'หนองบัวลำภู': '58',
    'ขอนแก่น': '59',
    'อุดรธานี': '5A',
    'เลย': '5B',
    'หนองคาย': '5C',
    'มหาสารคาม': '5D',
    'ร้อยเอ็ด': '5E',
    'กาฬสินธุ์': '5F',
    'สกลนคร': '5G',
    'นครพนม': '5H',
    'มุกดาหาร': '5I',
    'บึงกาฬ': '5J',
    
    # ภาคใต้ (รหัส 6)
    'ชุมพร': '60',
    'ระนอง': '61',
    'สุราษฎร์ธานี': '62',
    'พังงา': '63',
    'กระบี่': '64',
    'ภูเก็ต': '65',
    'นครศรีธรรมราช': '66',
    'ตรัง': '67',
    'พัทลุง': '68',
    'สงขลา': '69',
    'สตูล': '6A',
    'ปัตตานี': '6B',
    'ยะลา': '6C',
    'นราธิวาส': '6D',
}

# ชื่อภาค
REGION_NAMES = {
    '1': 'กลาง',
    '2': 'ตะวันออก',
    '3': 'ตะวันตก',
    '4': 'เหนือ',
    '5': 'อีสาน',
    '6': 'ใต้',
    '9': 'ไม่ระบุ'
}

# ==========================================
# HELPER: ZONE/REGION FUNCTIONS
# ==========================================
def get_region_code(province):
    """ดึงรหัสภาค/โซนจากจังหวัด"""
    if not province or str(province).strip() == '' or str(province) == 'nan':
        return '99'  # ไม่ระบุ
    province = clean_name(str(province).strip())  # ลบ จ./อ./ต. prefix
    # normalize aliases (พระนครศรีอยุธยา → อยุธยา ฯลฯ)
    _alias = {
        'พระนครศรีอยุธยา': 'อยุธยา',
        'กรุงเทพฯ': 'กรุงเทพมหานคร',
        'กทม': 'กรุงเทพมหานคร',
        'กทม.': 'กรุงเทพมหานคร',
        'โคราช': 'นครราชสีมา',
    }
    province = _alias.get(province, province)
    return REGION_CODE.get(province, '99')

def get_region_name(province):
    """ดึงชื่อภาคจากจังหวัด"""
    code = get_region_code(province)
    if code == '99':
        return 'ไม่ระบุ'
    region_prefix = code[0]
    return REGION_NAMES.get(region_prefix, 'ไม่ระบุ')

# ==========================================
# LOGISTICS ZONE FUNCTIONS
# ==========================================
def get_logistics_zone(province, district='', subdistrict=''):
    """
    หาโซนโลจิสติกส์จาก จังหวัด/อำเภอ/ตำบล
    
    หลักการ: ใช้โซนหลัก (ระดับจังหวัด) ก่อน แล้วค่อยใช้โซนย่อย (ระดับอำเภอ/ตำบล)
    
    Returns:
        zone_name (str): เช่น 'ZONE_A_พะเยา', 'ZONE_NEARBY_กทม', None ถ้าไม่พบ
    """
    if not province or str(province).strip() == '':
        return None
    
    province = str(province).strip()
    district = str(district).strip() if district else ''
    subdistrict = str(subdistrict).strip() if subdistrict else ''
    
    # 🎯 หลักการ: ใช้โซนหลักก่อน (ไม่มี districts/subdistricts กำหนด)
    # แล้วค่อยไล่ลงไปโซนย่อย (มี districts/subdistricts กำหนด)
    
    main_zones = []  # โซนหลัก (ระดับจังหวัด)
    sub_zones = []   # โซนย่อย (ระดับอำเภอ/ตำบล)
    
    # แยกโซนเป็นหลัก/ย่อย
    for zone_name, zone_info in LOGISTICS_ZONES.items():
        if province in zone_info['provinces']:
            # ถ้าไม่มี districts กำหนด = โซนหลัก
            if 'districts' not in zone_info or not zone_info['districts']:
                main_zones.append((zone_name, zone_info))
            else:
                # มี districts กำหนด = โซนย่อย
                sub_zones.append((zone_name, zone_info))
    
    # 1️⃣ ลองหาโซนย่อยก่อน (ถ้ามีอำเภอระบุมา)
    if district:
        for zone_name, zone_info in sub_zones:
            if district in zone_info['districts']:
                # ถ้ามี subdistricts กำหนดด้วย → เช็คให้แม่นยำยิ่งขึ้น
                if 'subdistricts' in zone_info and zone_info['subdistricts']:
                    if subdistrict and subdistrict in zone_info['subdistricts']:
                        return zone_name
                else:
                    # ไม่มี subdistricts กำหนด → return โซนย่อยนี้
                    return zone_name
    
    # 2️⃣ ถ้าไม่เจอโซนย่อย → ใช้โซนหลัก
    if main_zones:
        # เลือกโซนแรกที่เจอ (เรียงตาม priority)
        main_zones_sorted = sorted(main_zones, key=lambda x: x[1].get('priority', 999))
        return main_zones_sorted[0][0]
    
    return None

def get_zone_priority(zone_name):
    """
    ดึงค่า Priority ของโซน (สำหรับ LIFO: ไกลส่งก่อน ใกล้ส่งทีหลัง)
    
    Returns:
        int: 1-99 (1 = ไกลสุด, 99 = ใกล้สุด)
    """
    if not zone_name or zone_name not in LOGISTICS_ZONES:
        return 999  # ไม่รู้จักโซน → ให้ส่งทีหลังสุด
    
    return LOGISTICS_ZONES[zone_name]['priority']

def get_zone_highway(zone_name):
    """
    ดึงทางหลวงหลักของโซน
    
    Returns:
        str: เช่น 'สาย 1 (พหลโยธิน)', 'สาย 2 (มิตรภาพ)'
    """
    if not zone_name or zone_name not in LOGISTICS_ZONES:
        return ''
    
    return LOGISTICS_ZONES[zone_name].get('highway', '')

def can_combine_zones_by_highway(zone1, zone2):
    """
    เช็คว่า 2 โซนอยู่บนทางหลวงเดียวกันหรือไม่
    (ถ้าใช่ → สามารถรวมทริปได้)
    
    Returns:
        bool: True ถ้าอยู่ทางเดียวกัน
    """
    if not zone1 or not zone2:
        return False
    
    highway1 = get_zone_highway(zone1)
    highway2 = get_zone_highway(zone2)
    
    if not highway1 or not highway2:
        return False
    
    # เช็คว่าทางหลวงมีส่วนร่วมกัน (set intersection) รองรับ '304' == '304/331'
    return bool(set(highway1.split('/')) & set(highway2.split('/')))

def is_cross_zone_violation(province1, province2):
    """
    เช็คว่าจังหวัดทั้ง 2 อยู่ใน NO_CROSS_ZONE_PAIRS หรือไม่
    (พยายามหลีกเลี่ยง - Soft Rule)
    
    Returns:
        bool: True ถ้าควรหลีกเลี่ยงการรวมโซน
    """
    if not province1 or not province2:
        return False
    
    prov1 = str(province1).strip()
    prov2 = str(province2).strip()
    
    # เช็คทั้ง 2 ทาง
    return (prov1, prov2) in NO_CROSS_ZONE_PAIRS or (prov2, prov1) in NO_CROSS_ZONE_PAIRS

def are_provinces_on_same_route(province1, province2):
    """
    เช็คว่าจังหวัดทั้ง 2 อยู่ใน ROUTE เดียวกันหรือไม่
    
    Returns:
        bool: True ถ้าอยู่ route เดียวกัน (ควรรวมกัน)
    """
    if not province1 or not province2:
        return False
    
    prov1 = str(province1).strip()
    prov2 = str(province2).strip()
    
    # หา route ของแต่ละจังหวัด
    route1 = None
    route2 = None
    
    for route_name, route_info in ROUTE_GROUPS.items():
        if prov1 in route_info['provinces']:
            route1 = route_name
        if prov2 in route_info['provinces']:
            route2 = route_name
    
    # ถ้าอยู่ route เดียวกัน
    if route1 and route2:
        if route1 == route2:
            return True
        # เช็ค next_routes
        if route2 in ROUTE_GROUPS.get(route1, {}).get('next_routes', []):
            return True
        if route1 in ROUTE_GROUPS.get(route2, {}).get('next_routes', []):
            return True
    
    return False

def calculate_district_centroid(district_df):
    """คำนวณจุดกลางของอำเภอจากพิกัดสาขา"""
    valid_coords = district_df[district_df['_lat'] > 0]
    if valid_coords.empty:
        return None, None
    return valid_coords['_lat'].mean(), valid_coords['_lon'].mean()

def check_geographic_proximity(district1_df, district2_df, max_distance_km=MAX_DISTRICT_DISTANCE_KM):
    """ตรวจสอบว่า 2 อำเภอใกล้กันพอที่จะอยู่ทริปเดียวกันได้หรือไม่"""
    # ตรวจสอบจังหวัด
    prov1 = district1_df['_province'].iloc[0] if not district1_df.empty else ''
    prov2 = district2_df['_province'].iloc[0] if not district2_df.empty else ''
    
    # 🚨 ถ้าคนละจังหวัด → เช็ค Logistics Zone ก่อน
    if prov1 and prov2 and prov1 != prov2:
        # แช็คว่าอยู่ Logistics Zone เดียวกันหรือไม่
        zone1 = get_logistics_zone(prov1, '', '')
        zone2 = get_logistics_zone(prov2, '', '')
        
        if zone1 and zone2:
            if zone1 == zone2:
                # ✅ Zone เดียวกัน → ไม่จำกัดระยะทาง (เลือกใกล้ที่สุดในโซน)
                return True
            else:
                # คนละ Zone → เช็คว่าอยู่บนทางหลวงเดียวกันหรือไม่
                if not can_combine_zones_by_highway(zone1, zone2):
                    return False  # คนละทางหลวง → ห้ามรวม
    
    # คำนวณระยะห่างระหว่าง centroids
    lat1, lon1 = calculate_district_centroid(district1_df)
    lat2, lon2 = calculate_district_centroid(district2_df)
    
    if lat1 is None or lat2 is None:
        return True  # ไม่มีพิกัด ให้ผ่าน
    
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    
    if prov1 and prov2 and prov1 == prov2:
        # ✅ จังหวัดเดียวกัน → ใช้ threshold กว้างกว่า (60km)
        return distance <= (max_distance_km * 2.0)  # 30km * 2.0 = 60km
    else:
        # ⚠️ คนละจังหวัด + คนละ Zone → ใช้ threshold เข้มงวด (30km)
        return distance <= max_distance_km

def sort_branches_by_region_route(branches_df, master_data=None):
    """
    จัดเรียงสาขาตามภาค → จังหวัด → อำเภอ → ตำบล → Route
    เพื่อให้ทริปเรียงติดกันไม่กระโดด
    """
    if branches_df.empty:
        return branches_df
    
    df = branches_df.copy()
    
    # หาชื่อคอลัมน์จังหวัด (รองรับทั้ง Province และ จังหวัด)
    province_col = 'Province' if 'Province' in df.columns else 'จังหวัด' if 'จังหวัด' in df.columns else None
    
    # เพิ่มคอลัมน์สำหรับ sort
    df['_region_code'] = df[province_col].apply(get_region_code) if province_col else '99'
    df['_province'] = df[province_col].fillna('') if province_col else ''
    df['_district'] = df['District'].fillna('') if 'District' in df.columns else ''
    df['_subdistrict'] = df['Subdistrict'].fillna('') if 'Subdistrict' in df.columns else ''
    
    # แยก Route number
    if 'Route' in df.columns:
        df['_route_num'] = df['Route'].apply(lambda x: int(str(x).replace('CD', '')) if pd.notna(x) and str(x).startswith('CD') else 99999)
    else:
        df['_route_num'] = 99999
    
    # Sort
    df = df.sort_values(by=['_region_code', '_province', '_district', '_subdistrict', '_route_num'])
    
    # ลบคอลัมน์ชั่วคราว
    df = df.drop(columns=['_region_code', '_province', '_district', '_subdistrict', '_route_num'])
    
    return df.reset_index(drop=True)

def check_trip_route_spread(trip_df):
    """
    ตรวจสอบว่าทริปมี Route กระจายมากไหม
    คืนค่า: (route_range, is_spread, provinces)
    """
    if trip_df.empty or 'Route' not in trip_df.columns:
        return 0, False, []
    
    routes = trip_df['Route'].dropna().unique()
    route_nums = []
    for r in routes:
        if pd.notna(r) and str(r).startswith('CD'):
            try:
                route_nums.append(int(str(r).replace('CD', '')))
            except:
                pass
    
    if len(route_nums) < 2:
        return 0, False, trip_df['Province'].dropna().unique().tolist() if 'Province' in trip_df.columns else []
    
    route_range = max(route_nums) - min(route_nums)
    is_spread = route_range > 4000  # ถ้ามากกว่า 4000 ถือว่ากระจาย
    
    provinces = trip_df['Province'].dropna().unique().tolist() if 'Province' in trip_df.columns else []
    
    return route_range, is_spread, provinces

# ==========================================
# LOAD MASTER DATA
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)  # Cache 1 ชม. (เพิ่มความเร็ว)
def load_master_data():
    """โหลด Master Data จาก Google Sheets หรือ JSON (auto-sync)"""
    try:
        # ใช้ข้อมูลจาก Google Sheets ที่ sync มาแล้ว
        df_from_sheets = sync_branch_data_from_sheets()
        
        if df_from_sheets is None or df_from_sheets.empty:
            safe_print("⚠️ ไม่สามารถโหลดข้อมูล - ตรวจสอบ Google Sheets หรือ branch_data.json")
            return pd.DataFrame()
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_cols = ['Plan Code']
        missing = [c for c in required_cols if c not in df_from_sheets.columns]
        if missing:
            safe_print(f"⚠️ ขาดคอลัมน์: {missing}")
        
        # แปลงชื่อคอลัมน์ที่อาจต่างกัน
        col_mapping = {
            'ละ': 'ละติจูด',
            'ลอง': 'ลองติจูด'
        }
        df_from_sheets = df_from_sheets.rename(columns=col_mapping)
        
        # ทำความสะอาด Plan Code
        if 'Plan Code' in df_from_sheets.columns:
            df_from_sheets['Plan Code'] = df_from_sheets['Plan Code'].astype(str).str.strip().str.upper()
            df_from_sheets = df_from_sheets[df_from_sheets['Plan Code'] != '']
        
        safe_print(f"✅ โหลด MASTER_DATA: {len(df_from_sheets)} สาขา")
        
        # 🔍 Debug: แสดงคอลัมน์ทั้งหมดที่มี
        safe_print(f"📋 คอลัมน์ทั้งหมด ({len(df_from_sheets.columns)}): {', '.join(df_from_sheets.columns.tolist())}")
        
        # 🔍 Debug: แสดงคอลัมน์ที่เกี่ยวข้องกับรถ (ค้นหาแบบยืดหยุ่น)
        vehicle_cols = [
            'MaxTruckType', 'Max Truck Type', 'MaxVehicle', 'Max Vehicle', 
            'รถสูงสุด', 'Max_Truck_Type', 'max_truck', 'MaxTruck',
            'ข้อจำกัดรถ', 'Truck', 'truck_type', 'TruckType',
            'ประเภทรถ', 'Vehicle', 'vehicle_type', 'VehicleType'
        ]
        found_vehicle_cols = [col for col in vehicle_cols if col in df_from_sheets.columns]
        if found_vehicle_cols:
            safe_print(f"✅ พบคอลัมน์ข้อจำกัดรถ: {', '.join(found_vehicle_cols)}")
            # แสดงสถิติข้อจำกัดรถ
            for col in found_vehicle_cols:
                vehicle_counts = df_from_sheets[col].value_counts(dropna=False)
                safe_print(f"   - {col}: {dict(vehicle_counts)}")
        else:
            safe_print(f"⚠️ ไม่พบคอลัมน์ข้อจำกัดรถ!")
            # ค้นหาคอลัมน์ที่อาจเกี่ยวข้อง
            for col in df_from_sheets.columns:
                if 'truck' in col.lower() or 'vehicle' in col.lower() or 'รถ' in col:
                    safe_print(f"   💡 คอลัมน์ที่อาจเกี่ยวข้อง: '{col}'")
        
        return df_from_sheets
        
    except Exception as e:
        safe_print(f"❌ Error loading MASTER_DATA: {e}")
        return pd.DataFrame()

# โหลด Master Data จาก Google Sheets
MASTER_DATA = load_master_data()

# ==========================================
# 🔄 BRANCH GROUPING (จุดส่งเดียวกัน ≤200 เมตร)
# โหลดจาก branch_groups.json (สร้างโดย test_groups.py ด้วย haversine 200m)
# ==========================================
@st.cache_data(show_spinner=False)
def load_branch_groups():
    """
    โหลด branch_groups.json ที่สร้างด้วย haversine 200m
    Return: (groups_dict, branch_to_group_dict)
    """
    try:
        with open('branch_groups.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        groups = data.get('groups', {})  # {group_id: [codes]}
        branch_to_group = data.get('branch_to_group', {})  # {code: group_id}
        
        safe_print(f"✅ โหลด branch_groups.json: {len(groups)} กลุ่ม, {len(branch_to_group)} สาขา")
        return groups, branch_to_group
    except FileNotFoundError:
        safe_print("⚠️ ไม่พบ branch_groups.json - รัน python test_groups.py ก่อน")
        return {}, {}
    except Exception as e:
        safe_print(f"⚠️ โหลด branch_groups.json ไม่สำเร็จ: {e}")
        return {}, {}

# โหลด branch groups
BRANCH_GROUPS, BRANCH_TO_GROUP = load_branch_groups()

def get_group_branches(code: str) -> list:
    """
    ดึงสาขาทั้งหมดในกลุ่มเดียวกัน (จุดส่งเดียวกัน ≤200 เมตร)
    ถ้าไม่มีกลุ่ม return [code] (สาขาเดียว)
    """
    code_upper = str(code).strip().upper()
    group_id = BRANCH_TO_GROUP.get(code_upper)
    if group_id:
        return BRANCH_GROUPS.get(group_id, [code_upper])
    return [code_upper]

def is_same_group(code1: str, code2: str) -> bool:
    """เช็คว่า 2 สาขาอยู่กลุ่มเดียวกันหรือไม่"""
    c1 = str(code1).strip().upper()
    c2 = str(code2).strip().upper()
    g1 = BRANCH_TO_GROUP.get(c1)
    g2 = BRANCH_TO_GROUP.get(c2)
    return g1 and g2 and g1 == g2

# ==========================================
# 🚀 BRANCH CLUSTERS & SPATIAL DATA (Pre-computed)
# โหลดจาก branch_clusters.json (สร้างโดย precompute_branch_data.py)
# ==========================================
@st.cache_data(show_spinner=False)
def load_branch_clusters():
    """
    โหลด branch_clusters.json ที่มี:
    - branch_info: พิกัด, ระยะห่างจาก DC, ทิศทาง, cluster
    - nearby_branches: สาขาใกล้เคียง (< 15km)
    - clusters: กลุ่มตามระยะทาง, ทิศทาง, จังหวัด, อำเภอ
    """
    try:
        with open('branch_clusters.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        branch_info = {str(k).strip().upper(): v for k, v in data.get('branch_info', {}).items()}
        nearby_branches = {str(k).strip().upper(): v for k, v in data.get('nearby_branches', {}).items()}
        clusters = data.get('clusters', {})
        
        safe_print(f"✅ โหลด branch_clusters.json:")
        safe_print(f"   - {len(branch_info)} สาขามีข้อมูล spatial")
        safe_print(f"   - {len(nearby_branches)} สาขามี nearby branches")
        if clusters:
            safe_print(f"   - Distance clusters: {len(clusters.get('distance', {}))}")
            safe_print(f"   - Direction clusters: {len(clusters.get('direction', {}))}")
            safe_print(f"   - Province clusters: {len(clusters.get('province', {}))}")
            safe_print(f"   - District clusters: {len(clusters.get('district', {}))}")
        
        return branch_info, nearby_branches, clusters
    except FileNotFoundError:
        safe_print("⚠️ ไม่พบ branch_clusters.json - รัน python precompute_branch_data.py ก่อน")
        return {}, {}, {}
    except Exception as e:
        safe_print(f"⚠️ โหลด branch_clusters.json ไม่สำเร็จ: {e}")
        return {}, {}, {}

# โหลด branch clusters
BRANCH_INFO, NEARBY_BRANCHES, BRANCH_CLUSTERS = load_branch_clusters()

# ==========================================
# 🚀 PRE-COMPUTE: Distance Matrix & Nearby Branches
# ใช้ข้อมูลจาก branch_clusters.json แทนการคำนวณใหม่
# ==========================================
@st.cache_data(ttl=3600)  # Cache 1 ชั่วโมง
def precompute_branch_distances(master_df):
    """
    โหลดข้อมูล pre-computed จาก branch_clusters.json
    หรือคำนวณใหม่ถ้าไม่มีไฟล์
    """
    # ถ้ามีข้อมูล pre-computed ให้ใช้เลย
    if BRANCH_INFO and NEARBY_BRANCHES:
        safe_print("✅ ใช้ข้อมูล pre-computed จาก branch_clusters.json")
        
        # แปลง branch_info เป็น branch_coords (ใช้ uppercase key ทั้งหมด)
        branch_coords = {}
        for code, info in BRANCH_INFO.items():
            if 'lat' in info and 'lon' in info:
                branch_coords[str(code).strip().upper()] = (info['lat'], info['lon'])
        
        # แปลง nearby_branches เป็นรูปแบบที่ต้องการ
        nearby_dict = {}
        for code, nearby_list in NEARBY_BRANCHES.items():
            code_upper = str(code).strip().upper()
            # nearby_list มีได้สองรูปแบบ:
            #   - [code1, code2, ...]  (รูปแบบเก่า)
            #   - [{"code": code1, "distance": d1}, ...]  (รูปแบบใหม่จาก precompute_branch_data.py)
            nearby_with_dist = []
            if code_upper in branch_coords:
                lat1, lon1 = branch_coords[code_upper]
                for item in nearby_list:
                    # รองรับทั้งสองรูปแบบ
                    if isinstance(item, dict):
                        nearby_code = str(item.get('code', '')).strip().upper()
                        pre_dist = item.get('distance', None)
                    else:
                        nearby_code = str(item).strip().upper()
                        pre_dist = None

                    if not nearby_code or nearby_code not in branch_coords:
                        continue

                    lat2, lon2 = branch_coords[nearby_code]
                    if pre_dist is not None and pre_dist > 0:
                        # ใช้ค่า pre-computed จาก branch_clusters.json (OSRM road dist)
                        dist = pre_dist
                    else:
                        # Inline: ตรวจ DISTANCE_CACHE ก่อน → fallback haversine × 1.35
                        _ck  = f"{lat1:.4f},{lon1:.4f}_{lat2:.4f},{lon2:.4f}"
                        _ckr = f"{lat2:.4f},{lon2:.4f}_{lat1:.4f},{lon1:.4f}"
                        if USE_CACHE and _ck in DISTANCE_CACHE:
                            dist = DISTANCE_CACHE[_ck]
                        elif USE_CACHE and _ckr in DISTANCE_CACHE:
                            dist = DISTANCE_CACHE[_ckr]
                        else:
                            from math import radians, sin, cos, sqrt, atan2
                            _phi1, _phi2 = radians(lat1), radians(lat2)
                            _a = sin((radians(lat2-lat1))/2)**2 + cos(_phi1)*cos(_phi2)*sin((radians(lon2-lon1))/2)**2
                            dist = round(6371.0 * 2 * atan2(sqrt(_a), sqrt(1-_a)) * 1.35, 2)
                    nearby_with_dist.append((nearby_code, round(dist, 2)))
            nearby_dict[code_upper] = sorted(nearby_with_dist, key=lambda x: x[1])
        
        # สร้าง same_area_branches จาก clusters (ใช้ uppercase key)
        same_area_branches = {}
        if BRANCH_CLUSTERS and 'district' in BRANCH_CLUSTERS:
            district_clusters = BRANCH_CLUSTERS['district']
            for code, info in BRANCH_INFO.items():
                code_up = str(code).strip().upper()
                district_id = info.get('district_cluster')
                if district_id and district_id in district_clusters:
                    same_area_branches[code_up] = [str(c).strip().upper() for c in district_clusters[district_id] if str(c).strip().upper() != code_up]
                else:
                    same_area_branches[code_up] = []
        
        safe_print(f"   ✅ {len(branch_coords)} สาขามีพิกัด, {len(nearby_dict)} สาขามี nearby")
        return branch_coords, nearby_dict, same_area_branches
    
    # ถ้าไม่มี pre-computed ให้คำนวณใหม่
    safe_print("⚠️ ไม่มี pre-computed data - คำนวณใหม่...")
    
    if master_df.empty:
        return {}, {}, {}
    
    # ดึงพิกัดสาขาทั้งหมด
    branch_coords = {}
    for _, row in master_df.iterrows():
        code = str(row.get('Plan Code', '')).strip().upper()
        lat = row.get('ละติจูด') or row.get('Latitude') or row.get('ละ', 0)
        lon = row.get('ลองติจูด') or row.get('Longitude') or row.get('ลอง', 0)
        if code and lat and lon:
            try:
                lat_float = float(lat)
                lon_float = float(lon)
                if lat_float > 0 and lon_float > 0:
                    branch_coords[code] = (lat_float, lon_float)
            except (ValueError, TypeError):
                pass
    
    safe_print(f"   📍 พบ {len(branch_coords)} สาขาที่มีพิกัด")
    
    # สร้าง nearby_branches (คำนวณแบบเร็ว)
    nearby_branches = {}
    same_area_branches = {}
    
    codes = list(branch_coords.keys())
    for code in codes:
        nearby_branches[code] = []
        same_area_branches[code] = []
    
    safe_print(f"   ✅ เตรียมข้อมูลเบื้องต้นเสร็จสิ้น")
    
    return branch_coords, nearby_branches, same_area_branches

# Pre-compute distances
BRANCH_COORDS, NEARBY_BRANCHES, SAME_AREA_BRANCHES = precompute_branch_distances(MASTER_DATA)

# ==========================================
# CLEAN NAME FUNCTION (สำหรับทำ Join_Key)
# ==========================================
def clean_name(text):
    """
    ทำความสะอาดชื่อ: ลบ prefix จ./อ./ต. และ trim whitespace
    ใช้สำหรับสร้าง Join_Key เพื่อเทียบกับ Master Data
    """
    if pd.isna(text) or text is None:
        return ''
    text = str(text)
    # ลบ prefix ภาษาไทย
    text = text.replace('จ. ', '').replace('จ.', '')
    text = text.replace('อ. ', '').replace('อ.', '')
    text = text.replace('ต. ', '').replace('ต.', '')
    # ลบ prefix ภาษาอังกฤษ (ถ้ามี)
    text = text.replace('Tambon ', '').replace('Amphoe ', '').replace('Changwat ', '')
    return text.strip()

def normalize_province_name(province):
    """
    แปลงชื่อจังหวัดให้เป็นมาตรฐาน (แก้ปัญหาชื่อเพี้ยน)
    """
    if pd.isna(province) or province is None:
        return ''
    province = clean_name(province)
    # Mapping ชื่อที่พบบ่อย
    province_mapping = {
        'พระนครศรีอยุธยา': 'อยุธยา',
        'กรุงเทพฯ': 'กรุงเทพมหานคร',
        'กทม': 'กรุงเทพมหานคร',
        'กทม.': 'กรุงเทพมหานคร',
        'โคราช': 'นครราชสีมา',
    }
    return province_mapping.get(province, province)

def normalize(val):
    """ทำให้รหัสสาขาเป็นมาตรฐาน"""
    return str(val).strip().upper().replace(" ", "").replace(".0", "")

def load_master_dist_data():
    """
    โหลดไฟล์ Master Dist.xlsx สำหรับ:
    1. ระยะทางระดับตำบล
    2. Sum_Code (Sort_Code) สำหรับเรียงลำดับตามภูมิศาสตร์
    
    หลักการ: ใช้ Join_Key (จังหวัด_อำเภอ_ตำบล) เป็นตัวเชื่อม
    เพื่อดึง Sum_Code มาใช้ในการ Sort
    """
    try:
        file_path = 'Dc/Master Dist.xlsx'
        df = pd.read_excel(file_path)
        
        # สร้าง lookup dict - สอง key: Sum_Code และ Join_Key (จังหวัด_อำเภอ_ตำบล)
        dist_lookup = {}   # key = Sum_Code
        name_lookup = {}   # key = Join_Key (จังหวัด_อำเภอ_ตำบล)
        
        for _, row in df.iterrows():
            sum_code = str(row.get('Sum_Code', '')).strip()
            
            # ข้อมูลสำคัญ: เพิ่ม sum_code (Sort_Code) เข้าไปด้วย!
            data = {
                'sum_code': sum_code,  # 🔑 กุญแจสำคัญสำหรับ Sort!
                'region': row.get('Region', ''),
                'region_code': row.get('Region_Code', ''),
                'province': row.get('Province', ''),
                'prov_code': row.get('Prov_Code', ''),
                'district': row.get('District', ''),
                'dist_code': row.get('Dist_Code', ''),
                'subdistrict': row.get('Subdistrict', ''),
                'subdist_code': row.get('Subdist_Code', ''),
                'dist_from_dc_km': float(row.get('Dist_from_DC_km', 9999)) if pd.notna(row.get('Dist_from_DC_km')) else 9999,
                'prov_dist_km': float(row.get('Prov_Dist_km', 0)) if pd.notna(row.get('Prov_Dist_km')) else 0,
                'dist_subdist_km': float(row.get('Dist_Subdist_km', 0)) if pd.notna(row.get('Dist_Subdist_km')) else 0,
            }
            
            # Key 1: Sum_Code (สำหรับ lookup โดยตรง)
            if sum_code:
                dist_lookup[sum_code] = data
            
            # Key 2: Join_Key (จังหวัด_อำเภอ_ตำบล) - หัวใจของ Lookup!
            prov_raw = str(row.get('Province', ''))
            dist_raw = str(row.get('District', ''))
            subdist_raw = str(row.get('Subdistrict', ''))
            
            # Clean name สำหรับ Join
            prov_clean = clean_name(prov_raw)
            dist_clean = clean_name(dist_raw)
            subdist_clean = clean_name(subdist_raw)
            
            # Join_Key แบบ clean (มาตรฐาน)
            join_key = f"{prov_clean}_{dist_clean}_{subdist_clean}"
            if join_key and join_key != '__':
                name_lookup[join_key] = data
            
            # Join_Key แบบ normalized province (เผื่อชื่อเพี้ยน)
            prov_normalized = normalize_province_name(prov_raw)
            if prov_normalized != prov_clean:
                alt_key = f"{prov_normalized}_{dist_clean}_{subdist_clean}"
                if alt_key and alt_key != '__':
                    name_lookup[alt_key] = data
            
            # Join_Key แบบมี prefix (เผื่อข้อมูลมี prefix)
            raw_key = f"{prov_raw.strip()}_{dist_raw.strip()}_{subdist_raw.strip()}"
            if raw_key and raw_key != '__' and raw_key not in name_lookup:
                name_lookup[raw_key] = data
        
        return {'by_code': dist_lookup, 'by_name': name_lookup}
    except Exception as e:
        return {'by_code': {}, 'by_name': {}}

# โหลด Master Dist Data
MASTER_DIST_DATA = load_master_dist_data()

# ==========================================
# PUNTHAI/MAXMART BUFFER FUNCTIONS (REMOVED - ใช้โลจิกใหม่แล้ว)
# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_max_vehicle_for_branch(branch_code, test_df=None, debug=False):
    """ดึงรถใหญ่สุดที่สาขานี้รองรับ - อ่านจาก MASTER_DATA (Google Sheets) เท่านั้น"""
    branch_code_str = str(branch_code).strip().upper()
    
    # 🎯 อ่านจาก MASTER_DATA (Google Sheets) เท่านั้น - ไม่มี fallback
    if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
        # ลอง match หลายแบบ
        master_codes = MASTER_DATA['Plan Code'].str.strip().str.upper()
        
        # 1. Match แบบ exact
        branch_row = MASTER_DATA[master_codes == branch_code_str]
        
        # 2. ถ้าไม่พบ ลอง match แบบ partial (Code อาจมี prefix/suffix ต่างกัน)
        if branch_row.empty:
            # ลองตัด prefix ที่พบบ่อย
            prefixes = ['PUN-', 'MAX-', 'MM-', 'PT-', 'N', 'S', 'E', 'W', 'C', 'PUN', 'MAX', 'MM', 'PT']
            
            # กรณี branch_code มี prefix
            code_clean = branch_code_str
            for p in prefixes:
                if code_clean.startswith(p):
                    code_clean = code_clean[len(p):]
                    break
            
            # ลอง match กับ master codes ที่ตัด prefix เหมือนกัน
            for idx, mc in enumerate(master_codes):
                mc_clean = mc
                for p in prefixes:
                    if mc_clean.startswith(p):
                        mc_clean = mc_clean[len(p):]
                        break
                
                if code_clean == mc_clean:
                    branch_row = MASTER_DATA.iloc[[idx]]
                    break
                    
        # 3. (ยกเลิกการ match แบบ loose partial 'in' เพราะทำให้ได้รถผิดประเภท)
        
        if not branch_row.empty:
            # ลองหาคอลัมน์ข้อจำกัดรถหลายชื่อ (เพิ่มแบบยืดหยุ่นมากขึ้น)
            possible_cols = [
                'MaxTruckType', 'Max Truck Type', 'MaxVehicle', 'Max Vehicle', 
                'รถสูงสุด', 'Max_Truck_Type', 'max_truck', 'MaxTruck',
                'ข้อจำกัดรถ', 'Truck', 'truck_type', 'TruckType',
                'ประเภทรถ', 'Vehicle', 'vehicle_type', 'VehicleType'
            ]
            for col in possible_cols:
                if col in branch_row.columns and pd.notna(branch_row.iloc[0][col]):
                    max_truck = str(branch_row.iloc[0][col]).strip().upper()
                    # แปลงชื่อรถหลายแบบ
                    if max_truck in ['4W', '4 W', '4-W']:
                        return '4W'
                    elif max_truck in ['JB', 'J B', 'J-B', '4WJ', '4WJ ']:
                        return 'JB'
                    elif max_truck in ['6W', '6 W', '6-W']:
                        return '6W'
                    elif max_truck and max_truck not in ['', 'NAN', 'NONE', '-']:
                        if debug:
                            safe_print(f"⚠️ Branch {branch_code_str}: ค่าข้อจำกัดรถไม่รู้จัก '{max_truck}' จากคอลัมน์ '{col}'")
    
    # Default: ไม่มีข้อจำกัด = ใช้รถใหญ่ได้
    return '6W'

def get_max_vehicle_for_trip(trip_codes):
    """
    หารถใหญ่สุดที่ทริปนี้ใช้ได้ (เช็คข้อจำกัดของทุกสาขาในทริป)
    
    Args:
        trip_codes: set ของ branch codes ในทริป
    
    Returns:
        str: '4W', 'JB', หรือ '6W'
    """
    vehicle_priority = {'4W': 1, 'JB': 2, '6W': 3}
    max_allowed = '6W'  # เริ่มจากใหญ่สุด แล้วจำกัดตามข้อจำกัดสาขา
    min_priority = 3  # ค่าใหญ่สุดคือไม่มีข้อจำกัด
    
    for code in trip_codes:
        branch_max = get_max_vehicle_for_branch(code)
        priority = vehicle_priority.get(branch_max, 3)
        
        # 🔒 เลือกรถที่เล็กที่สุด (ข้อจำกัดมากที่สุด) จากทุกสาขาในทริป
        if priority < min_priority:
            min_priority = priority
            max_allowed = branch_max
    
    return max_allowed

def get_route_osrm(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, max_retries=1):
    """
    ขอเส้นทางจริงจาก OSRM API (วิ่งตามถนน)
    ตรวจ ROUTE_CACHE_DATA ก่อนเสมอ — เรียก API เฉพาะตอนที่ยังไม่เคย cache ไว้
    """
    if not FOLIUM_AVAILABLE:
        return [[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]]

    # ตรวจ cache ก่อน
    cache_key = f"{pickup_lat:.4f},{pickup_lon:.4f}|{dropoff_lat:.4f},{dropoff_lon:.4f}"
    if USE_CACHE and cache_key in ROUTE_CACHE_DATA:
        cached = ROUTE_CACHE_DATA[cache_key]
        if isinstance(cached, dict):
            return cached.get('coords', [[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]])
        return cached  # backward compat (list)

    # OSRM Public Server (lon, lat format!)
    loc = f"{pickup_lon},{pickup_lat};{dropoff_lon},{dropoff_lat}"
    url = f"http://router.project-osrm.org/route/v1/driving/{loc}?overview=full&geometries=geojson"

    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=4)
            res = r.json()

            if "routes" in res and len(res["routes"]) > 0:
                coords = res["routes"][0]["geometry"]["coordinates"]
                route_coords = [[lat, lon] for lon, lat in coords]
                # บันทึก cache ทันที
                if USE_CACHE:
                    ROUTE_CACHE_DATA[cache_key] = {'coords': route_coords, 'distance': 0}
                    save_route_cache(ROUTE_CACHE_DATA)
                return route_coords
            else:
                return [[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]]
        except Exception:
            return [[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]]

    return [[pickup_lat, pickup_lon], [dropoff_lat, dropoff_lon]]


def get_multi_point_route_osrm(waypoints, max_retries=2):
    """
    ขอเส้นทางจริงจาก OSRM API สำหรับหลายจุด พร้อม cache
    
    Args:
        waypoints: list ของ [lat, lon] เช่น [[14.1, 100.6], [14.2, 100.7], ...]
        max_retries: จำนวนครั้งที่ลองใหม่
    
    Returns:
        tuple: (route_coords, distance_km) - พิกัดเส้นทาง และระยะทางรวม
    """
    if not FOLIUM_AVAILABLE or len(waypoints) < 2:
        return waypoints, 0
    
    # สร้าง cache key
    cache_key = "|".join([f"{lat:.4f},{lon:.4f}" for lat, lon in waypoints])
    
    # ตรวจสอบ cache ก่อน
    if USE_CACHE and cache_key in ROUTE_CACHE_DATA:
        cached = ROUTE_CACHE_DATA[cache_key]
        return cached['coords'], cached['distance']
    
    # OSRM รับพิกัดแบบ lon,lat (ไม่ใช่ lat,lon!)
    coords_str = ";".join([f"{lon},{lat}" for lat, lon in waypoints])
    url = f"http://router.project-osrm.org/route/v1/driving/{coords_str}?overview=full&geometries=geojson"
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=10)
            res = r.json()
            
            if "routes" in res and len(res["routes"]) > 0:
                route = res["routes"][0]
                # แปลง GeoJSON coordinates (lon, lat) เป็น (lat, lon)
                coords = route["geometry"]["coordinates"]
                route_coords = [[lat, lon] for lon, lat in coords]
                # ระยะทางจาก OSRM เป็นเมตร
                distance_km = route.get("distance", 0) / 1000
                
                # บันทึก cache ทันทีทุกครั้งที่ได้ข้อมูลใหม่
                if USE_CACHE:
                    ROUTE_CACHE_DATA[cache_key] = {
                        'coords': route_coords,
                        'distance': distance_km
                    }
                    save_route_cache(ROUTE_CACHE_DATA)
                
                return route_coords, distance_km
            else:
                return waypoints, 0
        except Exception as e:
            if attempt < max_retries - 1:
                time_module.sleep(0.3)
                continue
            return waypoints, 0
    
    return waypoints, 0

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    คำนวณทิศทาง (bearing) จากจุด 1 ไปจุด 2 เป็นองศา (0-360)
    0 = เหนือ, 90 = ตะวันออก, 180 = ใต้, 270 = ตะวันตก
    """
    from math import radians, sin, cos, atan2, degrees
    
    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)
    dlon = radians(lon2 - lon1)
    
    x = sin(dlon) * cos(lat2_rad)
    y = cos(lat1_rad) * sin(lat2_rad) - sin(lat1_rad) * cos(lat2_rad) * cos(dlon)
    
    bearing = atan2(x, y)
    bearing = degrees(bearing)
    bearing = (bearing + 360) % 360  # Normalize to 0-360
    
    return bearing

def get_bearing_zone(bearing):
    """
    แบ่งทิศทางเป็น 8 โซน (ทุก 45 องศา)
    0-1 = N, 2-3 = NE, 4-5 = E, 6-7 = SE, 8-9 = S, 10-11 = SW, 12-13 = W, 14-15 = NW
    """
    # แบ่งเป็น 16 โซน (ทุก 22.5 องศา) เพื่อจัดกลุ่มสาขาที่อยู่ทิศเดียวกัน
    zone = int((bearing + 11.25) / 22.5) % 16
    return zone

def get_osrm_distance_live(lat1, lon1, lat2, lon2):
    """
    เรียก OSRM Table API เพื่อดึงระยะทางถนนจริงระหว่างสองจุด (km)
    คืนค่า float km ถ้าสำเร็จ หรือ None ถ้าล้มเหลว
    """
    try:
        url = (
            f"http://router.project-osrm.org/table/v1/driving/"
            f"{lon1},{lat1};{lon2},{lat2}?annotations=distance"
        )
        r = requests.get(url, timeout=6)
        data = r.json()
        if data.get("code") == "Ok":
            dist_m = data["distances"][0][1]  # จากจุด 0 → จุด 1
            if dist_m and dist_m > 0:
                return dist_m / 1000.0
    except Exception:
        pass
    return None


def haversine_distance(lat1, lon1, lat2, lon2, use_osrm_cache=True):
    """
    คืนค่าระยะทางถนนจริง (km) แบบ Cache-Only (ไม่เรียก live API ระหว่าง runtime)
    ลำดับ:
      1. OSRM distance cache → คืนค่าระยะทางถนนจริง (แม่นที่สุด)
      2. Cache miss → haversine × ROAD_FACTOR (ประมาณระยะถนน ~1.35×เส้นตรง)
    ใช้ precompute_branch_data.py เพื่อ build cache ล่วงหน้า
    """
    # 1. ตรวจ DISTANCE_CACHE (OSRM road distance) ก่อนเสมอ
    cache_key = f"{lat1:.4f},{lon1:.4f}_{lat2:.4f},{lon2:.4f}"
    cache_key_reverse = f"{lat2:.4f},{lon2:.4f}_{lat1:.4f},{lon1:.4f}"

    if USE_CACHE:
        if cache_key in DISTANCE_CACHE:
            return DISTANCE_CACHE[cache_key]
        if cache_key_reverse in DISTANCE_CACHE:
            return DISTANCE_CACHE[cache_key_reverse]

    # 2. ไม่เจอใน cache → ประมาณระยะถนนด้วย haversine × road factor
    # (ไม่เรียก OSRM live API ระหว่าง runtime เพื่อความเร็ว)
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    straight_km = R * 2 * atan2(sqrt(a), sqrt(1 - a))

    # Road factor 1.35× คือค่าเฉลี่ยของถนนไทยที่อ้อม ~35% จากเส้นตรง
    ROAD_FACTOR = 1.35
    return round(straight_km * ROAD_FACTOR, 2)

def load_model():
    """โหลดโมเดลที่เทรนไว้"""
    if not os.path.exists(MODEL_PATH):
        return None
    
    try:
        import warnings as _warnings
        with _warnings.catch_warnings(record=True) as _caught:
            _warnings.simplefilter("always")
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
        # แสดง version mismatch warning ใน UI (ไม่แสดงแค่ใน terminal)
        for w in _caught:
            if 'InconsistentVersionWarning' in str(w.category.__name__) or 'version' in str(w.message).lower():
                import sklearn as _sk
                st.warning(f"⚠️ **Model Version Warning:** โมเดลถูก train ด้วย sklearn เวอร์ชันต่างกัน — ผลลัพธ์อาจแตกต่างเล็กน้อย (ติดตั้ง sklearn ใหม่แล้ว retrain model เพื่อความแม่นยำสูงสุด)")
                break
        return model_data
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def load_excel(file_content, sheet_name=None):
    """โหลด Excel"""
    try:
        xls = pd.ExcelFile(io.BytesIO(file_content))
        
        target_sheet = None
        if sheet_name and sheet_name in xls.sheet_names:
            target_sheet = sheet_name
        else:
            for s in xls.sheet_names:
                if 'punthai' in s.lower() or '2.' in s.lower():
                    target_sheet = s
                    break
        
        if not target_sheet:
            target_sheet = xls.sheet_names[0]
        
        # หา header row
        df_temp = pd.read_excel(xls, sheet_name=target_sheet, header=None)
        header_row = 0
        
        for i in range(min(10, len(df_temp))):
            row_values = df_temp.iloc[i].astype(str).str.upper()
            match_count = sum([
                'BRANCH' in ' '.join(row_values),
                'TRIP' in ' '.join(row_values),
                'รหัสสาขา' in ' '.join(df_temp.iloc[i].astype(str))
            ])
            if match_count >= 2:
                header_row = i
                break
        
        df = pd.read_excel(xls, sheet_name=target_sheet, header=header_row)
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def process_dataframe(df):
    """แปลงคอลัมน์เป็นรูปแบบมาตรฐาน"""
    if df is None:
        return None
    
    rename_map = {}
    
    # ใช้ลำดับตำแหน่งคอลัมน์ตามไฟล์ test.xlsx sheet 2.Punthai
    # 0:Sep, 1:BU, 2:BranchCode, 3:รหัสWMS, 4:Branch, 5:TOTALCUBE, 6:TOTALWGT, 7:OriginalQTY, ...
    col_list = list(df.columns)
    
    # ลำดับ 1 = BU
    if len(col_list) > 1:
        rename_map[col_list[1]] = 'BU'
    # ลำดับ 2 = รหัสสาขา (BranchCode)
    if len(col_list) > 2:
        rename_map[col_list[2]] = 'Code'
    # ลำดับ 4 = สาขา/ชื่อ (Branch)
    if len(col_list) > 4:
        rename_map[col_list[4]] = 'Name'
    # ลำดับ 5 = TOTALCUBE
    if len(col_list) > 5:
        rename_map[col_list[5]] = 'Cube'
    # ลำดับ 6 = TOTALWGT
    if len(col_list) > 6:
        rename_map[col_list[6]] = 'Weight'
    # ลำดับ 15 = latitude
    if len(col_list) > 15:
        rename_map[col_list[15]] = 'Latitude'
    # ลำดับ 16 = longitude
    if len(col_list) > 16:
        rename_map[col_list[16]] = 'Longitude'
    
    # ตรวจสอบเพิ่มเติมจากชื่อคอลัมน์ (สำรองถ้าไฟล์มีคอลัมน์น้อยหรือโครงสร้างต่าง)
    for col in df.columns:
        if col in rename_map.values():  # ถ้า map แล้วข้าม
            continue
        if col in rename_map:  # ถ้าเป็น key ใน map แล้วข้าม
            continue
        col_clean = str(col).strip()
        col_upper = col_clean.upper().replace(' ', '').replace('_', '')
        
        # BU
        if col_upper == 'BU' or col_clean == 'BU':
            rename_map[col] = 'BU'
        # Code
        elif col_clean == 'BranchCode' or 'รหัสสาขา' in col_clean or col_clean ==  'BRANCH_CODE' in col_upper or 'CODE' in col_upper:
            if 'Weight' not in col_upper and 'Cube' not in col_upper:  # ป้องกันไม่ให้จับ WeightCode
                rename_map[col] = 'Code'
        # Name
        elif col_clean == 'Branch' or 'ชื่อสาขา' in col_clean or col_clean == 'สาขา' or ('BRANCH' in col_upper and 'CODE' not in col_upper):
            rename_map[col] = 'Name'
        # ตำบล
        elif 'ตำบล' in col_clean or 'SUBDISTRICT' in col_upper or 'TAMBON' in col_upper:
            rename_map[col] = 'Subdistrict'
        # อำเภอ
        elif 'อำเภอ' in col_clean or ('DISTRICT' in col_upper and 'SUB' not in col_upper) or 'AMPHOE' in col_upper or 'AMPHUR' in col_upper:
            rename_map[col] = 'District'
        # จังหวัด
        elif 'จังหวัด' in col_clean or 'PROVINCE' in col_upper or 'CHANGWAT' in col_upper:
            rename_map[col] = 'Province'
        # Weight - ตรวจสอบหลายรูปแบบ
        elif ('น้ำหนัก' in col_clean or 
              'WEIGHT' in col_upper or 
              'WGT' in col_upper or 
              'TOTALWGT' in col_upper or
              'น้ําหนัก' in col_clean or  # รองรับ ำ ที่พิมพ์ผิด
              col_upper in ['WEIGHT', 'WGT', 'TOTALWEIGHT']):
            rename_map[col] = 'Weight'
        # Cube - ตรวจสอบหลายรูปแบบ
        elif ('คิว' in col_clean or 
              'CUBE' in col_upper or 
              'TOTALCUBE' in col_upper or
              'CBM' in col_upper or
              col_upper in ['CUBE', 'CBM', 'TOTALCUBE']):
            rename_map[col] = 'Cube'
        # Latitude
        elif 'latitude' in col_clean.lower() or col_clean == 'ละติจูด' or 'LAT' == col_upper or col_upper == 'LATITUDE':
            rename_map[col] = 'Latitude'
        # Longitude
        elif 'longitude' in col_clean.lower() or col_clean == 'ลองติจูด' or col_upper in ['LONG', 'LNG', 'LON', 'LONGITUDE']:
            rename_map[col] = 'Longitude'
        # Trip
        elif col_upper in ['TRIPNO', 'TRIP_NO', 'TRIPNUMBER'] or col_clean == 'Trip no':
            rename_map[col] = 'TripNo'
        elif col_upper == 'TRIP' or 'ทริป' in col_clean or 'เที่ยว' in col_clean:
            rename_map[col] = 'Trip'
        # Booking
        elif 'BOOKING' in col_upper:
            rename_map[col] = 'Booking'
    
    df = df.rename(columns=rename_map)
    
    # ลบคอลัมน์ซ้ำ
    df = df.loc[:, ~df.columns.duplicated()]
    
    if 'Code' in df.columns:
        df['Code'] = df['Code'].apply(normalize)
        
        # ตัดสาขาที่ไม่ต้องการออก (รหัส)
        df = df[~df['Code'].isin(EXCLUDE_BRANCHES)]
        
        # ตัดสาขาที่ชื่อมี keyword ที่ไม่ต้องการ
        if 'Name' in df.columns:
            exclude_pattern = '|'.join(EXCLUDE_NAMES)
            df = df[~df['Name'].str.contains(exclude_pattern, case=False, na=False)]
    
    for col in ['Weight', 'Cube']:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # เพิ่มจังหวัดจาก Master ถ้ายังไม่มี (รองรับทั้ง Province และ จังหวัด)
    province_col = 'Province' if 'Province' in df.columns else 'จังหวัด' if 'จังหวัด' in df.columns else None
    
    if not province_col or df[province_col].isna().all():
        if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns and 'Code' in df.columns:
            # สร้าง mapping จาก Master
            province_map = {}
            for _, row in MASTER_DATA.iterrows():
                code = row.get('Plan Code', '')
                province = row.get('จังหวัด', '')
                if code and province:
                    province_map[code] = province
            
            # ฟังก์ชันค้นหาจังหวัดจากชื่อสาขา
            def find_province_by_name(code, name):
                # ลองหาจาก code ก่อน
                if code in province_map:
                    return province_map[code]
                
                # ถ้าไม่เจอ ลองค้นหาจากชื่อสาขา
                if not name or pd.isna(name):
                    return ''
                
                # แยกคำสำคัญจากชื่อ (เอาคำแรกที่ไม่ใช่ prefix)
                keywords = str(name).replace('MAX MART-', '').replace('PUNTHAI-', '').replace('LUBE', '').strip()
                if not keywords:
                    return ''
                
                # ค้นหาในชื่อสาขาของ Master (ใช้ vectorized แทน iterrows)
                name_lookup = {}
                if not MASTER_DATA.empty and 'สาขา' in MASTER_DATA.columns and 'จังหวัด' in MASTER_DATA.columns:
                    for rec in MASTER_DATA[['สาขา', 'จังหวัด']].dropna().to_dict('records'):
                        name_lookup[str(rec['สาขา'])[:10]] = str(rec.get('จังหวัด', ''))
                for prefix, prov in name_lookup.items():
                    if keywords[:10] == prefix or prefix in keywords:
                        return prov if prov else ''
                return ''
            
            # ใส่จังหวัดให้แต่ละสาขา (สร้างคอลัมน์ Province ถ้ายังไม่มี)
            target_col = 'Province' if 'Province' in df.columns else 'จังหวัด'
            if 'Name' in df.columns:
                df[target_col] = df.apply(lambda row: find_province_by_name(row['Code'], row.get('Name', '')), axis=1).fillna('')
            else:
                df[target_col] = df['Code'].map(province_map).fillna('')
            
            # สร้าง Province ถ้ายังไม่มี (เพื่อ backward compatibility)
            if 'Province' not in df.columns and 'จังหวัด' in df.columns:
                df['Province'] = df['จังหวัด']
    
    return df.reset_index(drop=True)

def predict_trips(test_df, model_data, punthai_buffer=1.0, maxmart_buffer=1.10):
    """
    จัดทริปแบบใหม่ - เรียบง่ายและมีประสิทธิภาพ
    
    หลักการ:
    1. เรียงตาม: ภาค → จังหวัด → อำเภอ → ตำบล → Route (ใช้ระยะทางจาก Master Dist.xlsx ไม่ใช่ตัวอักษร)
    2. จับกลุ่ม Route เดียวกัน รวมน้ำหนักไว้ด้วยกัน
    3. เรียงจากไกลมาใกล้ (จาก DC)
    4. ตัดเป็นทริปตามน้ำหนัก/คิวของรถแต่ละประเภท
    5. ใช้ BUFFER ตาม BU (ตรวจจากชื่อสาขา)
    
    Args:
        test_df: DataFrame ข้อมูลสาขาที่จะจัดทริป
        model_data: ข้อมูลโมเดล (branch_vehicles, etc.)
        punthai_buffer: Buffer สำหรับ Punthai (เช่น 1.0 = 100%)
        maxmart_buffer: Buffer สำหรับ Maxmart/ผสม (เช่น 1.10 = 110%)
    """
    branch_vehicles = model_data.get('branch_vehicles', {})
    
    # ==========================================
    # Step 1: เตรียม Master Dist Lookup (Join_Key → Sort_Code)
    # หลักการ: ใช้ Join_Key (จังหวัด_อำเภอ_ตำบล) เป็นตัวเชื่อม
    # เพื่อดึง Sum_Code (Sort_Code) มาใช้ในการเรียงลำดับ
    # ==========================================
    subdistrict_dist_lookup = {}  # {Join_Key: {sum_code, dist_from_dc, ...}}
    if MASTER_DIST_DATA and 'by_name' in MASTER_DIST_DATA:
        subdistrict_dist_lookup = MASTER_DIST_DATA['by_name']
    
    # สร้าง location_map จากข้อมูล test_df (จาก Google Sheets) เป็นหลัก
    location_map = {}  # {code: {province, district, subdistrict, route, sum_code, ...}}
    
    # 🎯 ใช้ข้อมูลจาก Google Sheets (MASTER_DATA) เป็นหลัก - ไม่ใช่จาก Excel upload
    for _, row in test_df.iterrows():
        code = str(row.get('Code', '')).strip().upper()
        if not code:
            continue
        
        # 🌟 ใช้ข้อมูลจาก MASTER_DATA (Google Sheets) เป็นหลัก
        province = ''
        district = ''
        subdistrict = ''
        route = ''
        
        # ลองหาจาก MASTER_DATA ก่อน (ข้อมูลล่าสุดจาก Sheets)
        if isinstance(model_data, pd.DataFrame) and not model_data.empty and 'Plan Code' in model_data.columns:
            master_row = model_data[model_data['Plan Code'] == code]
            if not master_row.empty:
                province = str(master_row.iloc[0].get('จังหวัด', '')).strip() if pd.notna(master_row.iloc[0].get('จังหวัด')) else ''
                district = str(master_row.iloc[0].get('อำเภอ', '')).strip() if pd.notna(master_row.iloc[0].get('อำเภอ')) else ''
                subdistrict = str(master_row.iloc[0].get('ตำบล', '')).strip() if pd.notna(master_row.iloc[0].get('ตำบล')) else ''
                route = str(master_row.iloc[0].get('Route', '')).strip() if pd.notna(master_row.iloc[0].get('Route')) else ''
        
        # 🔄 Fallback: ถ้าไม่มีใน MASTER_DATA → ใช้จาก Excel upload (ข้อมูลเก่า)
        if not province:
            province = str(row.get('Province', '')).strip() if pd.notna(row.get('Province')) else ''
        if not district:
            district = str(row.get('District', '')).strip() if pd.notna(row.get('District')) else ''
        if not subdistrict:
            subdistrict = str(row.get('Subdistrict', '')).strip() if pd.notna(row.get('Subdistrict')) else ''
        if not route:
            route = str(row.get('Route', '')).strip() if pd.notna(row.get('Route')) else ''
        
        # 🔧 normalize province: ลบ จ./อ./ต. prefix และ alias
        province   = clean_name(province)
        _prov_alias = {'พระนครศรีอยุธยา':'อยุธยา','กรุงเทพฯ':'กรุงเทพมหานคร',
                       'กทม':'กรุงเทพมหานคร','กทม.':'กรุงเทพมหานคร','โคราช':'นครราชสีมา'}
        province   = _prov_alias.get(province, province)
        district   = clean_name(district)
        subdistrict = clean_name(subdistrict)
        
        # 🌍 ใช้พิกัดจาก Sheets เป็นหลัก
        lat = 0
        lon = 0
        
        # ลองหาคอลัมน์พิกัดหลายแบบ
        lat_cols = ['Latitude', 'latitude', 'ละติจูด', 'lat','ละ']
        lon_cols = ['Longitude', 'longitude', 'ลองจิจูด', 'ลองติจูด', 'lon', 'long','ลอง']
        
        for lat_col in lat_cols:
            if lat_col in row and pd.notna(row[lat_col]):
                try:
                    lat = float(row[lat_col])
                    break
                except:
                    pass
        
        for lon_col in lon_cols:
            if lon_col in row and pd.notna(row[lon_col]):
                try:
                    lon = float(row[lon_col])
                    break
                except:
                    pass
        
        # 🔄 ถ้า Sheets ไม่มีพิกัด → Fallback ไปหาใน MASTER_DATA
        if (lat == 0 or lon == 0) and not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
            master_row = MASTER_DATA[MASTER_DATA['Plan Code'] == code]
            if not master_row.empty:
                lat = float(master_row.iloc[0].get('ละติจูด', 0)) if pd.notna(master_row.iloc[0].get('ละติจูด')) else 0
                lon = float(master_row.iloc[0].get('ลองติจูด', 0)) if pd.notna(master_row.iloc[0].get('ลองติจูด')) else 0
        
        # 🔑 สร้าง Join_Key เพื่อเทียบกับ Master Dist (VLOOKUP)
        prov_clean = clean_name(province)
        dist_clean = clean_name(district)
        subdist_clean = clean_name(subdistrict)
        join_key = f"{prov_clean}_{dist_clean}_{subdist_clean}"
        
        # ลองหลาย key เผื่อชื่อไม่ตรง
        dist_data = subdistrict_dist_lookup.get(join_key, {})
        if not dist_data:
            # ลอง normalize ชื่อจังหวัด
            prov_normalized = normalize_province_name(province)
            alt_key = f"{prov_normalized}_{dist_clean}_{subdist_clean}"
            dist_data = subdistrict_dist_lookup.get(alt_key, {})
        
        # ดึงข้อมูลจาก Master Dist (ถ้ามี)
        if dist_data:
            sum_code = dist_data.get('sum_code', '')  # 🎯 Sort_Code หลัก!
            dist_from_dc = dist_data.get('dist_from_dc_km', 9999)
            region_code = dist_data.get('region_code', '')
            prov_code = dist_data.get('prov_code', '')
            dist_code_val = dist_data.get('dist_code', '')
            subdist_code = dist_data.get('subdist_code', '')
        else:
            # Fallback: สร้าง sort_code จาก region code และคำนวณระยะทางจาก lat/lon
            region_code = get_region_code(province)
            sum_code = f"R99P999D9999S99999"  # Default สำหรับไม่พบ
            dist_from_dc = 9999
            if lat and lon:
                dist_from_dc = haversine_distance(DC_WANG_NOI_LAT, DC_WANG_NOI_LON, lat, lon)
            prov_code = 'P999'
            dist_code_val = 'D9999'
            subdist_code = 'S99999'
        
        region_name = get_region_name(province)
        
        location_map[code] = {
            'province': province,
            'district': district,
            'subdistrict': subdistrict,
            'route': route,
            'lat': lat,
            'lon': lon,
            'join_key': join_key,  # 🔑 Join_Key ที่ใช้ lookup
            'sum_code': sum_code,  # 🎯 Sort_Code หลัก (จาก Master Dist)
            'distance_from_dc': dist_from_dc,
            'region_code': region_code,
            'prov_code': prov_code,
            'dist_code': dist_code_val,
            'subdist_code': subdist_code,
            'region_name': region_name
        }
    
    # ==========================================
    # Step 2: เพิ่มข้อมูลพื้นที่ให้แต่ละสาขา (pd.merge แบบ manual)
    # ==========================================
    df = test_df.copy()
    
    def get_location_info(code):
        code_upper = str(code).strip().upper()
        return location_map.get(code_upper, {
            'province': '', 'district': '', 'subdistrict': '', 'route': '',
            'lat': 0, 'lon': 0, 'join_key': '', 
            'sum_code': 'R99P999D9999S99999',  # Default sort_code
            'distance_from_dc': 9999,
            'region_code': 'R99', 'prov_code': 'P999', 'dist_code': 'D9999', 'subdist_code': 'S99999',
            'region_name': 'ไม่ระบุ'
        })
    
    # เพิ่มคอลัมน์ข้อมูลพื้นที่ (รวม sum_code สำหรับ sort)
    df['_sum_code'] = df['Code'].apply(lambda c: get_location_info(c)['sum_code'])  # 🎯 Sort_Code!
    df['_join_key'] = df['Code'].apply(lambda c: get_location_info(c)['join_key'])
    df['_region_code'] = df['Code'].apply(lambda c: get_location_info(c)['region_code'])
    df['_region_name'] = df['Code'].apply(lambda c: get_location_info(c)['region_name'])
    df['_prov_code'] = df['Code'].apply(lambda c: get_location_info(c)['prov_code'])
    df['_dist_code'] = df['Code'].apply(lambda c: get_location_info(c)['dist_code'])
    df['_subdist_code'] = df['Code'].apply(lambda c: get_location_info(c)['subdist_code'])
    df['_province'] = df['Code'].apply(lambda c: get_location_info(c)['province'])
    df['_district'] = df['Code'].apply(lambda c: get_location_info(c)['district'])
    df['_subdistrict'] = df['Code'].apply(lambda c: get_location_info(c)['subdistrict'])
    df['_route'] = df['Code'].apply(lambda c: get_location_info(c)['route'])
    df['_distance_from_dc'] = df['Code'].apply(lambda c: get_location_info(c)['distance_from_dc'])
    df['_lat'] = df['Code'].apply(lambda c: get_location_info(c)['lat'])
    df['_lon'] = df['Code'].apply(lambda c: get_location_info(c)['lon'])
    
    # 🎯 คำนวณ Bearing (ทิศทาง) จาก DC เพื่อจัดกลุ่มสาขาที่อยู่ทิศเดียวกัน
    DC_LAT = 14.117451
    DC_LON = 100.633408
    
    def calc_bearing(row):
        if row['_lat'] > 0 and row['_lon'] > 0:
            return calculate_bearing(DC_LAT, DC_LON, row['_lat'], row['_lon'])
        return 0
    
    df['_bearing_from_dc'] = df.apply(calc_bearing, axis=1)
    df['_bearing_zone'] = df['_bearing_from_dc'].apply(get_bearing_zone)
    
    # 🚨 เพิ่ม Logistics Zone สำหรับ routing ตามทางหลวง
    df['_logistics_zone'] = df.apply(
        lambda row: get_logistics_zone(row['_province'], row['_district'], row['_subdistrict']),
        axis=1
    )
    df['_zone_priority'] = df['_logistics_zone'].apply(get_zone_priority)
    df['_zone_highway'] = df['_logistics_zone'].apply(get_zone_highway)
    
    # ==========================================
    # Step 3: เรียงลำดับแบบ Hierarchical (Zone Priority > Region > Province Max Dist > District Max Dist > Distance)
    # 🎯 หัวใจสำคัญ: เรียงตาม Region Order ก่อน (ไกลมาใกล้)
    # ==========================================
    
    # เพิ่ม Region Order สำหรับ sorting
    df['_region_order'] = df['_region_name'].map(REGION_ORDER).fillna(99)
    
    # คำนวณ Province Max Distance (จังหวัดไหนมีจุดไกลสุดมาก่อน)
    prov_max_dist = df.groupby('_province')['_distance_from_dc'].max().reset_index()
    prov_max_dist.columns = ['_province', '_prov_max_dist']
    df = df.merge(prov_max_dist, on='_province', how='left')
    df['_prov_max_dist'] = df['_prov_max_dist'].fillna(9999)
    
    # คำนวณ District Max Distance (อำเภอไหนมีจุดไกลสุดมาก่อน)
    dist_max_dist = df.groupby(['_province', '_district'])['_distance_from_dc'].max().reset_index()
    dist_max_dist.columns = ['_province', '_district', '_dist_max_dist']
    df = df.merge(dist_max_dist, on=['_province', '_district'], how='left')
    df['_dist_max_dist'] = df['_dist_max_dist'].fillna(9999)
    
    # 🎯 Sort: LIFO (Last In First Out) - ไกลส่งก่อน, ใกล้ส่งทีหลัง
    # Zone Priority (Asc - 1=ไกลสุด, 99=ใกล้สุด) → Region Order → Province Max Dist → District Max Dist → 
    # Sum_Code → Route → Distance (Desc - ไกลก่อนในโซนเดียวกัน)
    df = df.sort_values(
        ['_zone_priority', '_region_order', '_prov_max_dist', '_dist_max_dist', '_sum_code', '_route', '_distance_from_dc'],
        ascending=[True, True, False, False, True, True, False]  # Zone Priority + ไกลมาใกล้
    ).reset_index(drop=True)
    
    # ==========================================
    # Step 4: จับกลุ่ม Route เดียวกัน รวมน้ำหนัก
    # ==========================================
    # สร้าง grouping key จาก route (ถ้ามี) หรือ ตำบล+อำเภอ+จังหวัด
    def get_group_key(row):
        route = row['_route']
        if route and route.strip():
            return f"R_{route}"
        # ถ้าไม่มี route ใช้ รหัสตำบล (เรียงตามระยะทาง)
        return f"L_{row['_subdist_code']}_{row['_dist_code']}_{row['_prov_code']}"
    
    df['_group_key'] = df.apply(get_group_key, axis=1)
    
    # ==========================================
    # Step 5: หารถที่เหมาะสมจากข้อจำกัดสาขา + Central Region Rule
    # ==========================================
    def get_max_vehicle_for_code(code):
        """หารถที่ใหญ่ที่สุดที่สาขาสามารถใช้ได้ - อ่านจาก Sheets"""
        max_vehicle = get_max_vehicle_for_branch(code, test_df=test_df)
        return max_vehicle
    
    def get_allowed_vehicles_for_region(region_name):
        """หารถที่ใช้ได้ (อิงตาม Master data เท่านั้น)"""
        return ['4W', 'JB', '6W']  # All vehicles - restrictions from Master data only
    
    df['_max_vehicle'] = df['Code'].apply(get_max_vehicle_for_code)
    df['_region_allowed_vehicles'] = df['_region_name'].apply(get_allowed_vehicles_for_region)
    
    # 🎯 สร้าง Vehicle Priority: สาขา 4W = 1 (จัดก่อน), JB = 2, 6W = 3 (จัดทีหลัง)
    vehicle_priority_map = {'4W': 1, 'JB': 2, '6W': 3}
    df['_vehicle_priority'] = df['_max_vehicle'].map(vehicle_priority_map).fillna(3)
    
    # 🎯 Sort: ใช้ BEARING ZONE เป็นหลัก เพื่อจัดกลุ่มสาขาทิศเดียวกันอยู่ติดกัน
    # หลักการ: สาขาที่อยู่ทิศเดียวกันจาก DC ต้องอยู่ในทริปเดียวกัน ไม่กระจาย
    # 1. Bearing Zone (ทิศทางจาก DC) - สาขาทิศเดียวกันอยู่ติดกัน
    # 2. ระยะทางจาก DC (ไกลมาใกล้) - LIFO
    # 3. จังหวัด/อำเภอ/ตำบล - จัดกลุ่มในพื้นที่เดียวกัน
    df = df.sort_values(
        [
            '_bearing_zone',        # 1. 🧭 ทิศทางจาก DC - สาขาทิศเดียวกันอยู่ติดกัน!
            '_distance_from_dc',    # 2. ระยะทางไกลก่อน (LIFO)
            '_province',            # 3. จังหวัดเดียวกัน
            '_district',            # 4. อำเภอเดียวกัน
            '_subdistrict',         # 5. ตำบลเดียวกัน
            '_vehicle_priority'     # 6. ข้อจำกัดรถ (secondary)
        ],
        ascending=[
            True,   # ทิศทาง 0-15 (เหนือ→ตะวันออก→ใต้→ตะวันตก)
            False,  # ไกลมาใกล้ (LIFO)
            True,   # จังหวัดเรียง A-Z
            True,   # อำเภอเรียง A-Z
            True,   # ตำบลเรียง A-Z
            True    # ข้อจำกัดมากก่อน
        ]
    ).reset_index(drop=True)
    
    safe_print(f"📊 DEBUG: Bearing zones = {df['_bearing_zone'].unique().tolist()}")
    
    # ==========================================
    # Step 6: DISTRICT CLUSTERING ALLOCATION (OPTIMIZED)
    # จัดทริปตาม District Buckets พร้อม Split เมื่อเกิน
    # ==========================================
    trip_counter = 1
    df['Trip'] = 0
    
    vehicle_priority = {'4W': 1, 'JB': 2, '6W': 3}
    
    # 🚀 CACHE: Pre-compute branch constraints และ BU type จาก Excel upload
    branch_max_vehicle_cache = {}
    branch_bu_cache = {}
    for _, row in df.iterrows():
        code = row['Code']
        branch_max_vehicle_cache[code] = row['_max_vehicle']
        
        # 📊 ดึง BU จากไฟล์ Excel ที่ upload มา (คอลัมน์ BU)
        bu = row.get('BU', None)  # อ่านค่าจริงจาก Excel
        
        # เช็คว่าเป็น Punthai หรือไม่
        is_punthai = False
        if bu is not None:
            bu_str = str(bu).strip().upper()
            # เช็ค BU = 211 หรือ 'PUNTHAI'
            is_punthai = bu_str in ['211', 'PUNTHAI']
        else:
            # ถ้าไม่มีคอลัมน์ BU → ลองเช็คจากชื่อสาขา (fallback)
            name = str(row.get('Name', '')).upper()
            is_punthai = 'PUNTHAI' in name or 'PUN-' in name
        
        branch_bu_cache[code] = is_punthai
    
    # 🚀 Pre-compute limits with buffer
    def get_max_limits(allowed_vehicles, is_punthai):
        """หา capacity สูงสุดที่ใช้ได้"""
        buffer_mult = punthai_buffer if is_punthai else maxmart_buffer
        max_vehicle = '6W' if '6W' in allowed_vehicles else ('JB' if 'JB' in allowed_vehicles else '4W')
        limits_to_use = PUNTHAI_LIMITS if is_punthai else LIMITS
        lim = limits_to_use.get(max_vehicle, LIMITS['6W'])
        return {
            'max_w': lim.get('max_w', 6000) * buffer_mult,
            'max_c': lim.get('max_c', 20.0) * buffer_mult,
            'max_d': lim.get('max_drops', 12)
        }
    
    # Helper function: เลือกรถที่เหมาะสม (STRICT - บังคับข้อจำกัด)
    def select_vehicle_for_load(weight, cube, drops, is_punthai, allowed_vehicles, strict_constraint=True):
        """
        เลือกรถที่เหมาะสมตามโหลดและข้อจำกัด
        
        Logic: ใช้ buffer จากหน้าเว็บ (punthai_buffer, maxmart_buffer)
        - Punthai: buffer = 100% (ห้ามเกิน)
        - Maxmart: buffer = 110% (เกินได้ 10%)
        - strict_constraint=True: ห้ามใช้รถที่ใหญ่กว่า allowed_vehicles (บังคับข้อจำกัด)
        """
        buffer_mult = punthai_buffer if is_punthai else maxmart_buffer
        limits_to_use = PUNTHAI_LIMITS if is_punthai else LIMITS
        
        # 🚨 STRICT MODE: ใช้เฉพาะรถที่อนุญาตเท่านั้น ห้ามเกิน
        vehicle_rank = {'4W': 1, 'JB': 2, '6W': 3}
        max_allowed_rank = max([vehicle_rank[v] for v in allowed_vehicles if v in vehicle_rank], default=3)
        
        # เรียงจากเล็กไปใหญ่ แต่ห้ามเกิน max_allowed
        vehicle_order = ['4W', 'JB', '6W']
        
        for v in vehicle_order:
            # ข้ามรถที่ใหญ่กว่าที่อนุญาต
            if strict_constraint and vehicle_rank.get(v, 3) > max_allowed_rank:
                continue
            
            if v not in allowed_vehicles:
                continue
                
            lim = limits_to_use[v]
            
            # เช็คตาม buffer ที่ตั้งหน้าเว็บเท่านั้น
            if (weight <= lim['max_w'] * buffer_mult and 
                cube <= lim['max_c'] * buffer_mult and 
                drops <= lim.get('max_drops', 12)):
                return v
        
        return None
    
    # Helper function: เช็ค Geographic Spread ภายในทริป
    def check_intra_trip_spread(trip_codes_list):
        """ตรวจสอบว่าสาขาในทริปไม่กระจายทางภูมิศาสตร์เกินไป (ห้ามคนละทิศ)"""
        if len(trip_codes_list) < 2:
            return True  # 1 สาขา = OK
        
        trip_df = df[df['Code'].isin(trip_codes_list)]
        if trip_df.empty:
            return True
        
        # คำนวณ centroid ของทริป
        trip_lat_mean = trip_df['_lat'].mean()
        trip_lon_mean = trip_df['_lon'].mean()
        
        # เช็คว่าทุกสาขาห่างจาก centroid ไม่เกิน 80km
        max_dist_from_center = 0
        for _, row in trip_df.iterrows():
            if row['_lat'] > 0 and row['_lon'] > 0:
                dist = haversine_distance(trip_lat_mean, trip_lon_mean, row['_lat'], row['_lon'])
                max_dist_from_center = max(max_dist_from_center, dist)
        
        # ถ้า spread เกิน 80km ถือว่ากระจายเกินไป (คนละทิศ)
        return max_dist_from_center <= 80
    
    # Helper function: เช็คว่าเป็น Punthai ล้วนหรือไม่ (Optimized - ใช้ cache)
    def is_all_punthai_codes(codes):
        if not codes:
            return False
        return all(branch_bu_cache.get(c, False) for c in codes)
    
    # Helper function: หา allowed vehicles จาก codes (Optimized)
    def get_allowed_from_codes(codes, base_allowed):
        """หา allowed vehicles โดยรวม branch constraints"""
        result = set(base_allowed)
        for code in codes:
            branch_max = branch_max_vehicle_cache.get(code, '6W')
            if branch_max == 'JB':
                result.discard('6W')
            elif branch_max == '4W':
                result.discard('6W')
                result.discard('JB')
        return list(result)
    
    # Step 6.4: 🎯 ZONE-STRICT GREEDY - จัดทริปแบบแยกโซน + ห้ามข้ามโซน
    # หลักการ: ใช้ LOGISTICS_ZONES + NO_CROSS_ZONE_PAIRS
    # ==========================================
    safe_print("🎯 กำลังจัดทริปใหม่แบบ Zone-Strict (LOGISTICS_ZONES + NO_CROSS_ZONE_PAIRS)...")

    # _logistics_zone, _zone_priority, _zone_highway คำนวณไว้แล้วตั้งแต่ Step 2

    # 2️⃣ เรียงลำดับตาม zone priority (priority ต่ำ = ไกล DC = จัดก่อน)
    df = df.sort_values(['_zone_priority', '_distance_from_dc'], ascending=[True, False]).reset_index(drop=True)
    
    # 🚗 คำนวณ MaxVehicle ของแต่ละสาขา (4W=1, JB=2, 6W=3)
    vehicle_rank = {'4W': 1, 'JB': 2, '6W': 3}
    df['_max_vehicle'] = df['Code'].apply(lambda c: get_max_vehicle_for_branch(c))
    df['_vehicle_rank'] = df['_max_vehicle'].map(vehicle_rank).fillna(3).astype(int)
    
    # รีเซ็ตทริปทั้งหมด
    df['Trip'] = 0
    trip_counter = 1
    
    # สร้าง set ของสาขาที่ยังไม่ได้จัด
    unassigned = set(df['Code'].tolist())
    
    # 3️⃣ จัดทริปแยกตามโซน + ประเภทรถ (4W ก่อน → JB → 6W)
    zones_processed = set()
    
    while unassigned:
        # หาสาขาที่ยังไม่ได้จัด
        unassigned_df = df[df['Code'].isin(unassigned)]
        if unassigned_df.empty:
            break
        
        # 🚗 เรียงตาม: 1) zone priority (ไกลก่อน) 2) vehicle_rank (4W ก่อน) 3) ระยะทาง (ไกลก่อน)
        unassigned_df = unassigned_df.sort_values(
            ['_zone_priority', '_vehicle_rank', '_distance_from_dc'], 
            ascending=[True, True, False]
        )
        
        # เลือกสาขาแรก (ไกลสุด + ข้อจำกัดมากสุด)
        farthest_row = unassigned_df.iloc[0]
        start_code = farthest_row['Code']
        start_lat = farthest_row['_lat']
        start_lon = farthest_row['_lon']
        start_max_vehicle = farthest_row['_max_vehicle']  # 🚗 รถที่ใหญ่ที่สุดที่ใช้ได้
        
        # 🎯 กำหนดโซนของทริปจากสาขาแรก
        trip_province = farthest_row.get('_province', '')
        trip_district = farthest_row.get('_district', '')  # เพิ่มอำเภอ
        trip_subdistrict = farthest_row.get('_subdistrict', '')  # เพิ่มตำบล
        trip_bearing_zone = farthest_row.get('_bearing_zone', 0)
        trip_region = farthest_row.get('_region_name', '')
        trip_logistics_zone = farthest_row.get('_logistics_zone', '')  # 🎯 LOGISTICS_ZONE
        trip_max_vehicle = start_max_vehicle  # 🚗 รถสูงสุดของทริป (จากสาขาแรก)

        # 🔒 บันทึก province + highway + region ต้นทาง → ใช้ตรวจ zone-compat ตลอด (ป้องกัน chain-hop)
        trip_original_province = trip_province
        # คำนวณ region จาก province → fallback หลายชั้น
        _derived_region = get_region_name(str(trip_original_province)) if trip_original_province else ''
        if not _derived_region or _derived_region == 'ไม่ระบุ':
            _derived_region = farthest_row.get('_region_name', '')
        if not _derived_region or _derived_region == 'ไม่ระบุ':
            # last resort: ดึงจาก _region_code column (พร้อมไว้แล้วใน Step 2)
            _rc = str(farthest_row.get('_region_code', '99'))
            _derived_region = REGION_NAMES.get(_rc[0] if len(_rc) > 0 else '9', 'ไม่ระบุ')
        trip_original_region = _derived_region
        _orig_hw_str = get_zone_highway(trip_logistics_zone)
        trip_original_hws: set = set(_orig_hw_str.split('/')) if _orig_hw_str else set()
        
        # 🎯 เก็บ set ของตำบล/อำเภอที่อยู่ในทริป (ใช้หาสาขาตำบลเดียวกัน)
        trip_subdistricts = {trip_subdistrict} if trip_subdistrict else set()
        trip_districts = {trip_district} if trip_district else set()
        
        # เริ่มทริปใหม่ด้วยสาขาไกลสุด
        # 🎯 ดึงสาขาทั้งกลุ่ม (จุดส่งเดียวกัน ≤200m) ของสาขาแรก
        start_group_codes = get_group_branches(start_code)
        start_group_unassigned = [c for c in start_group_codes if c in unassigned or c.upper() in [str(x).upper() for x in unassigned]]
        if not start_group_unassigned:
            start_group_unassigned = [start_code]
        
        # คำนวณ weight/cube รวมทั้งกลุ่ม
        trip_codes = []
        trip_weight = 0
        trip_cube = 0
        for gc in start_group_unassigned:
            gc_row = df[df['Code'].apply(lambda x: str(x).upper() == str(gc).upper())]
            if not gc_row.empty:
                actual_code = gc_row.iloc[0]['Code']
                # 🔒 FINAL REGION GUARD: กรองสมาชิกกลุ่ม start ที่ต่างภาค
                if trip_original_region and trip_original_region not in ('', 'ไม่ระบุ'):
                    _sg_prov = str(gc_row.iloc[0].get('_province', '') or '')
                    _sg_region = get_region_name(_sg_prov) if _sg_prov else ''
                    if _sg_region and _sg_region not in ('', 'ไม่ระบุ') and _sg_region != trip_original_region:
                        safe_print(f"      🛑 START-GROUP GUARD: ตัด {actual_code} ภาค {_sg_region} ≠ {trip_original_region}")
                        continue  # ไม่เพิ่ม — สาขานี้จะเป็นทริปตัวเองทีหลัง
                    # ⚡ ถ้าไม่รู้ province/region → เช็คระยะจาก farthest_row (สาขาเริ่มต้น)
                    if (not _sg_region or _sg_region == 'ไม่ระบุ') and gc.upper() != str(start_code).upper():
                        _sg_lat = float(gc_row.iloc[0].get('_lat', 0) or 0)
                        _sg_lon = float(gc_row.iloc[0].get('_lon', 0) or 0)
                        if _sg_lat > 0 and _sg_lon > 0 and start_lat > 0 and start_lon > 0:
                            _sg_dist = haversine_distance(_sg_lat, _sg_lon, start_lat, start_lon)
                            if _sg_dist > 10.0:
                                safe_print(f"      🛑 START-GROUP DIST GUARD: ตัด {actual_code} ห่าง {_sg_dist:.1f}km (ไม่รู้จังหวัด)")
                                continue
                trip_codes.append(actual_code)
                trip_weight += gc_row.iloc[0]['Weight']
                trip_cube += gc_row.iloc[0]['Cube']
                # ลบออกจาก unassigned
                if actual_code in unassigned:
                    unassigned.remove(actual_code)
                else:
                    for u in list(unassigned):
                        if str(u).upper() == str(actual_code).upper():
                            unassigned.remove(u)
                            break
        
        if len(trip_codes) > 1:
            safe_print(f"  🌏 ทริปใหม่ #{trip_counter} เริ่มที่ {start_code} | จังหวัด='{trip_original_province}' | ภาค='{trip_original_region}' | zone='{trip_logistics_zone}'")
            safe_print(f"      🔗 สาขาในกลุ่ม: {trip_codes}")
        else:
            safe_print(f"   🚀 Trip {trip_counter}: {start_code} ({trip_province}) - {trip_logistics_zone} - {trip_max_vehicle} - {farthest_row['_distance_from_dc']:.0f}km")
        
        # หา allowed vehicles จาก constraints (จำกัดตาม trip_max_vehicle)
        trip_allowed = get_allowed_from_codes(trip_codes, ['4W', 'JB', '6W'])
        trip_is_punthai = all(branch_bu_cache.get(c, False) for c in trip_codes)
        
        # 2️⃣ Greedy: หาสาขาใกล้สุดมาเติมจนเต็ม buffer
        while unassigned:
            remaining_df = df[df['Code'].isin(unassigned)].copy()
            if remaining_df.empty:
                break
            
            # ✅ รีเซ็ต same_zone_df ทุก iteration ป้องกัน stale value จาก iteration ก่อน
            same_zone_df = None
            filter_level = ""

            # 🎯 0️⃣ เช็คสาขาใกล้มากๆ ก่อน (< 6km) — [FIX2] เช็คทุกสาขาในทริป ไม่ใช่แค่สาขาสุดท้าย
            very_close_df = None
            unassigned_upper_set_vc = {str(c).strip().upper() for c in unassigned}
            very_close_codes = set()
            ultra_close_codes = set()  # < 3km — [FIX3] bypass highway filter (แต่ยังตรวจ region)
            for _tc in trip_codes:
                _tc_upper = str(_tc).strip().upper()
                if _tc_upper in NEARBY_BRANCHES:
                    for nearby_code, dist in NEARBY_BRANCHES[_tc_upper]:
                        if dist < 6.0 and nearby_code in unassigned_upper_set_vc:
                            very_close_codes.add(nearby_code)
                        if dist < 3.0 and nearby_code in unassigned_upper_set_vc:
                            ultra_close_codes.add(nearby_code)

            if very_close_codes:
                very_close_df = remaining_df[remaining_df['Code'].apply(lambda x: str(x).strip().upper() in very_close_codes)].copy()

            # ถ้ามีสาขาใกล้มาก → กรอง highway แต่ bypass ถ้า < 3km จากสาขาใดๆ ในทริป
            # [FIX3] ultra_close (< 3km) ข้าม highway filter ได้ — region filter ทำงานต่อตามปกติ
            if very_close_df is not None and not very_close_df.empty:
                trip_hw = get_zone_highway(trip_logistics_zone) if trip_logistics_zone else ''
                # กรองเฉพาะสาขาที่: < 3km (ultra-close bypass) หรือ จังหวัดเดียวกัน หรือ highway เดียวกัน
                def _hw_compatible(row_hw):
                    if not trip_hw:
                        return True  # ทริปไม่มี zone → อนุญาตแก่ neighbor (region filter จะจัดการต่อ)
                    if not row_hw:
                        return False  # candidate ไม่มี zone/highway → ปฏิเสธ
                    # เช็ค primary highway (ตัวแรกก่อน '/')
                    trip_primaries = set(trip_hw.split('/'))
                    row_primaries = set(row_hw.split('/'))
                    return bool(trip_primaries & row_primaries)
                compatible_df = very_close_df[
                    (very_close_df['Code'].apply(lambda c: str(c).strip().upper() in ultra_close_codes)) |  # < 3km bypass
                    (very_close_df['_province'] == trip_province) |
                    very_close_df['_zone_highway'].apply(_hw_compatible)
                ].copy()
                if not compatible_df.empty:
                    same_zone_df = compatible_df
                    filter_level = "ใกล้มาก(<6km)"
                # ถ้าไม่มีสาขา compatible → ปล่อยต่อ same_zone_df = None (ใช้ zone filter ปกติ)
            else:
                # 🎯 กรองลำดับ: ตำบล → อำเภอ → จังหวัด → โซน (ห้ามข้ามจนกว่าจะหมด!)
                same_zone_df = None
                filter_level = ""
                
                # 🏙️ กรณีพิเศษ BKK: ใช้ radius แทน zone (เพราะสาขาติดกัน)
                is_bkk_trip = trip_logistics_zone and 'BKK' in trip_logistics_zone
                if is_bkk_trip and trip_codes:
                    # หาสาขาทั้งหมดที่อยู่ใน radius 15km จากสาขาใดๆ ในทริป
                    bkk_nearby = set()
                    unassigned_upper_set = {str(c).strip().upper() for c in unassigned}
                    for tc in trip_codes:
                        tc_upper = str(tc).strip().upper()
                        if tc_upper in NEARBY_BRANCHES:
                            for nearby_code, dist in NEARBY_BRANCHES[tc_upper]:
                                if dist <= 15.0 and nearby_code in unassigned_upper_set:
                                    bkk_nearby.add(nearby_code)
                    
                    if bkk_nearby:
                        bkk_nearby_df = remaining_df[remaining_df['Code'].apply(lambda x: str(x).strip().upper() in bkk_nearby)].copy()
                        if not bkk_nearby_df.empty:
                            same_zone_df = bkk_nearby_df
                            filter_level = "BKK radius 15km"
                
                # 1️⃣ ตำบลเดียวกันก่อน (priority สูงสุด)
                if same_zone_df is None and trip_subdistricts and trip_districts:
                    subdistrict_df = remaining_df[
                        (remaining_df['_subdistrict'].isin(trip_subdistricts)) & 
                        (remaining_df['_district'].isin(trip_districts))
                    ].copy()
                    if not subdistrict_df.empty:
                        same_zone_df = subdistrict_df
                        filter_level = "ตำบล"
                
                # 2️⃣ อำเภอเดียวกัน (ถ้าหมดตำบลแล้ว)
                if same_zone_df is None and trip_districts:
                    district_df = remaining_df[remaining_df['_district'].isin(trip_districts)].copy()
                    if not district_df.empty:
                        same_zone_df = district_df
                        filter_level = "อำเภอ"
            
            # 3️⃣ จังหวัดเดียวกัน (ถ้าหมดอำเภอแล้ว)
            # ⚠️ ต้องตรวจ trip_province ไม่ว่าง — มิฉะนั้น '' == '' คืน ALL สาขาไม่รู้จังหวัด (ต่างภาคปนกัน)
            if same_zone_df is None and trip_province:
                province_df = remaining_df[remaining_df['_province'] == trip_province].copy()
                if not province_df.empty:
                    same_zone_df = province_df
                    filter_level = "จังหวัด"
            
            # 4️⃣ LOGISTICS_ZONE เดียวกัน (ถ้าหมดจังหวัดแล้ว)
            if same_zone_df is None and trip_logistics_zone:
                zone_df = remaining_df[remaining_df['_logistics_zone'] == trip_logistics_zone].copy()
                if not zone_df.empty:
                    same_zone_df = zone_df
                    filter_level = "โซน"
                    # รีเซ็ตตำบล/อำเภอ เพราะเปลี่ยนจังหวัดแล้ว
                    trip_subdistricts = set()
                    trip_districts = set()
            
            # 5️⃣ Highway เดียวกัน (ถ้าหมดโซนแล้ว แต่ยังมีโซนอื่นใน highway เดียวกัน)
            if same_zone_df is None:
                trip_highway = get_zone_highway(trip_logistics_zone)
                if trip_highway:
                    # ใช้ set intersection แทน exact match เพื่อรองรับ '304' กับ '304/331' เป็นต้น
                    trip_hw_set = set(trip_highway.split('/'))
                    def _hw_match(row_hw):
                        if not row_hw:
                            return False
                        return bool(trip_hw_set & set(str(row_hw).split('/')))
                    highway_df = remaining_df[remaining_df['_zone_highway'].apply(_hw_match)].copy()
                    if not highway_df.empty:
                        # เลือกโซนที่ใกล้สาขาล่าสุดที่สุด
                        last_code = trip_codes[-1]
                        last_row = df[df['Code'] == last_code].iloc[0]
                        last_lat, last_lon = last_row['_lat'], last_row['_lon']
                        
                        if last_lat > 0 and last_lon > 0:
                            highway_df['_dist_to_last'] = highway_df.apply(
                                lambda r: haversine_distance(r['_lat'], r['_lon'], last_lat, last_lon) 
                                if r['_lat'] > 0 and r['_lon'] > 0 else 999, axis=1
                            )
                            # 🔒 กรองเฉพาะสาขาที่ไม่ไกลเกิน 120km จากสาขาล่าสุด (ป้องกันกระโดดไกล)
                            highway_df = highway_df[highway_df['_dist_to_last'] <= 120].copy()
                            if highway_df.empty:
                                same_zone_df = None
                            else:
                                nearest_row = highway_df.loc[highway_df['_dist_to_last'].idxmin()]
                                nearest_zone = nearest_row['_logistics_zone']
                                nearest_prov = nearest_row['_province']

                                # อัพเดตโซน (BKK trips ไม่อัพเดตจังหวัดออกจาก BKK)
                                _orig_zone_name = str(farthest_row.get('_logistics_zone', ''))
                                _is_bkk_start = (trip_original_province == 'กรุงเทพมหานคร' or
                                                 'BKK' in _orig_zone_name)
                                _nearest_is_bkk = 'BKK' in str(nearest_zone)
                                trip_logistics_zone = nearest_zone
                                if _is_bkk_start and not _nearest_is_bkk:
                                    safe_print(f"      🏙️ BKK trip: ไม่อัพเดต trip_province ({trip_province} → {nearest_prov} ข้าม)")
                                else:
                                    trip_province = nearest_prov
                                trip_subdistricts = set()
                                trip_districts = set()

                                # 🔒 จำกัด same_zone_df เฉพาะ nearest_zone (ป้องกันกระโดดข้ามโซนใน highway)
                                same_zone_df = highway_df[highway_df['_logistics_zone'] == nearest_zone].copy()
                                if same_zone_df.empty:
                                    # fallback: รัศมี 30km จาก nearest branch
                                    same_zone_df = highway_df[highway_df['_dist_to_last'] <= 30].copy()
                                filter_level = f"Highway→Zone {nearest_zone}"
                                safe_print(f"      🛣️ ขยายไปโซน {nearest_zone} ใน Highway {trip_highway} ({len(same_zone_df)} สาขา)")
            
            # 6️⃣ หมด Highway แล้ว → ปิดทริป! (ห้ามข้ามไป Highway อื่น)
            # ตามหลัก "หมดเส้นทางจริงๆถึงไปเส้นอื่น"
            if same_zone_df is None:
                safe_print(f"      🛑 หมดสาขาใน Highway {get_zone_highway(trip_logistics_zone)} แล้ว → ปิดทริป {trip_counter}")
                break  # ปิดทริปเลย ไม่ขยายไป Highway อื่น

            # 🔒 กรองภาค: ห้ามเพิ่มสาขาจากต่างภาค (ใช้ทุกเส้นทาง)
            if not trip_original_region or trip_original_region in ('ไม่ระบุ', ''):
                # trip_original_region ไม่รู้ → ดึงจาก candidates แล้วล็อค
                _cand_region_counts: dict = {}
                for _, _cr in same_zone_df.iterrows():
                    _cp = _cr.get('_province', '')
                    _crn = get_region_name(str(_cp)) if _cp else ''
                    if _crn and _crn != 'ไม่ระบุ':
                        _cand_region_counts[_crn] = _cand_region_counts.get(_crn, 0) + 1
                if _cand_region_counts:
                    # ล็อคภาคจาก candidate ที่มีมากที่สุด
                    trip_original_region = max(_cand_region_counts, key=_cand_region_counts.get)
                    safe_print(f"      🔒 ล็อคภาคอัตโนมัติ #{trip_counter}: '{trip_original_region}' (จาก candidates {_cand_region_counts})")
                    _region_filtered = same_zone_df[
                        same_zone_df['_province'].apply(lambda p: get_region_name(str(p)) if p else '') == trip_original_region
                    ]
                    if _region_filtered.empty:
                        safe_print(f"      🛑 ไม่มีสาขาในภาค {trip_original_region} → ปิดทริป {trip_counter}")
                        break
                    same_zone_df = _region_filtered.copy()
                else:
                    # ไม่รู้ภาคเลย → ปิดทริปปัจจุบัน (คงเหลือสาขาไม่รู้จังหวัด → จะเป็นทริปตัวเองทีหลัง)
                    safe_print(f"      🛑 ไม่รู้ภาคของ candidates ใน #{trip_counter} → ปิดทริป (สาขาไม่รู้จังหวัดจะเป็นทริปตัวเอง)")
                    break
            else:
                _region_filtered = same_zone_df[
                    same_zone_df['_province'].apply(lambda p: get_region_name(str(p)) if p else '') == trip_original_region
                ]
                _dropped = len(same_zone_df) - len(_region_filtered)
                if _dropped > 0:
                    _dropped_provs = same_zone_df[~same_zone_df.index.isin(_region_filtered.index)]['_province'].unique().tolist()
                    safe_print(f"      🔒 กรองภาค #{trip_counter}: ตัด {_dropped} สาขา (ต่างภาค: {_dropped_provs})")
                if _region_filtered.empty:
                    safe_print(f"      🛑 ไม่มีสาขาในภาค {trip_original_region} (หลังกรองภาค) → ปิดทริป {trip_counter}")
                    break
                same_zone_df = _region_filtered.copy()

            # 🎯 Priority: ตำบลเดียวกัน > อำเภอเดียวกัน > จังหวัด > โซน
            same_zone_df['_priority'] = 4  # default = โซนเดียวกัน
            
            # ตำบลเดียวกัน → priority 1
            if trip_subdistricts and trip_districts:
                mask_subdistrict = same_zone_df['_subdistrict'].isin(trip_subdistricts) & same_zone_df['_district'].isin(trip_districts)
                same_zone_df.loc[mask_subdistrict, '_priority'] = 1
            
            # อำเภอเดียวกัน → priority 2
            if trip_districts:
                mask_district = same_zone_df['_district'].isin(trip_districts) & (same_zone_df['_priority'] > 2)
                same_zone_df.loc[mask_district, '_priority'] = 2
            
            # จังหวัดเดียวกัน → priority 3
            mask_province = (same_zone_df['_province'] == trip_province) & (same_zone_df['_priority'] > 3)
            same_zone_df.loc[mask_province, '_priority'] = 3
            
            # 🎯 คำนวณระยะทาง - ใช้ pre-computed ถ้ามี
            unassigned_upper = {str(c).strip().upper() for c in unassigned}
            
            candidate_distances = {}
            for tc in trip_codes:
                tc_upper = str(tc).strip().upper()
                if tc_upper in NEARBY_BRANCHES:
                    for nearby_code, dist in NEARBY_BRANCHES[tc_upper]:
                        if nearby_code in unassigned_upper:
                            if nearby_code not in candidate_distances or dist < candidate_distances[nearby_code]:
                                candidate_distances[nearby_code] = dist
            
            # คำนวณระยะทาง — ใช้ NEARBY_BRANCHES cache ถ้ามี, fallback haversine จาก branch ล่าสุด
            _last_code_d = trip_codes[-1]
            _last_row_d = df[df['Code'] == _last_code_d].iloc[0]
            _last_lat_d, _last_lon_d = _last_row_d['_lat'], _last_row_d['_lon']

            def _dist_for_row(row):
                cu = str(row['Code']).strip().upper()
                if cu in candidate_distances:
                    return candidate_distances[cu]
                # fallback: haversine จาก branch ล่าสุดในทริป
                if _last_lat_d > 0 and _last_lon_d > 0 and row['_lat'] > 0 and row['_lon'] > 0:
                    return haversine_distance(row['_lat'], row['_lon'], _last_lat_d, _last_lon_d)
                return 999

            same_zone_df['_dist_to_trip'] = same_zone_df.apply(_dist_for_row, axis=1)
            
            # เรียงตาม priority + distance
            same_zone_df = same_zone_df.sort_values(['_priority', '_dist_to_trip'])
            
            found_candidate = False
            
            for _, candidate_row in same_zone_df.iterrows():
                candidate_code = candidate_row['Code']
                candidate_dist = candidate_row['_dist_to_trip']
                
                # 🚫 ถ้าไกลเกิน 80km → ข้ามไปสาขาถัดไป (อย่า break — priority อื่นอาจใกล้กว่า)
                if candidate_dist > 80:
                    continue  # ลองสาขาถัดไป (priority ต่ำกว่า / ใกล้กว่า)
                
                # 🚫 Zone/province/region axis check (Step 6 greedy) — ห้ามรวมคนละทิศ/highway/ภาค
                _c_prov   = candidate_row.get('_province', '')
                _c_zone   = candidate_row.get('_logistics_zone', '')
                _c_hw     = candidate_row.get('_zone_highway', '')
                _c_hws    = set(str(_c_hw).split('/')) if _c_hw else set()
                # ✅ ตรวจภาคก่อน — คำนวณจาก province โดยตรง (ไม่ใช้ _region_name ที่อาจว่าง)
                _c_region_calc = get_region_name(str(_c_prov)) if _c_prov else ''
                _orig_region_calc = trip_original_region  # ล็อคไว้แล้วตอนเริ่มทริปหรือ chokepoint
                _region_compat = (
                    _c_region_calc in ('', 'ไม่ระบุ') or   # candidate ไม่รู้จักภาค → อนุญาต
                    _orig_region_calc in ('', 'ไม่ระบุ') or  # ทริปยังไม่รู้จักภาค → อนุญาต (จะถูกล็อคที่ chokepoint ต่อไป)
                    _c_region_calc == _orig_region_calc   # ภาคเดียวกัน
                )
                if not _region_compat:
                    safe_print(f"      🚫 step6 skip {candidate_code} ภาคต่างกัน ({_c_region_calc}/{_c_prov} ≠ {_orig_region_calc}/{trip_original_province})")
                    continue
                # ✅ ตรวจ province/zone — ห้ามข้ามโซน/จังหวัดโดยไม่มีเหตุผล
                # [STRICT] ตัด highway-wide bypass ออก — กระโดดข้ามโซนใน highway เดียวกันได้
                _zone_compat = (
                    not _c_prov or not trip_original_province or   # ไม่มีข้อมูล → อนุญาต
                    _c_prov == trip_original_province or           # จังหวัดเดียวกับต้นทาง
                    _c_prov == trip_province or                    # จังหวัดเดียวกับปัจจุบัน
                    _c_zone == trip_logistics_zone                 # โซนเดียวกับปัจจุบัน
                )
                if not _zone_compat:
                    safe_print(f"      🚫 step6 skip {candidate_code} ({_c_prov}/{_c_zone}) ≠ trip ({trip_original_province}/{trip_province}/{trip_logistics_zone})")
                    continue   # ลองสาขาถัดไปใน same_zone_df

                # 🎯 ดึงสาขาทั้งกลุ่ม (จุดส่งเดียวกัน ≤200m)
                group_codes = get_group_branches(candidate_code)
                # กรองเฉพาะสาขาที่ยังไม่ได้จัดและมีใน df
                group_codes_unassigned = [c for c in group_codes if c in unassigned or c.upper() in [str(x).upper() for x in unassigned]]
                if not group_codes_unassigned:
                    group_codes_unassigned = [candidate_code]
                
                # คำนวณ weight/cube รวมทั้งกลุ่ม
                group_weight = 0
                group_cube = 0
                group_codes_valid = []
                for gc in group_codes_unassigned:
                    gc_row = df[df['Code'].apply(lambda x: str(x).upper() == str(gc).upper())]
                    if not gc_row.empty:
                        # 🔒 FINAL REGION GUARD: ตรวจภาคของสมาชิกกลุ่ม candidate ก่อนเพิ่ม
                        if trip_original_region and trip_original_region not in ('', 'ไม่ระบุ'):
                            _cg_prov = str(gc_row.iloc[0].get('_province', '') or '')
                            _cg_region = get_region_name(_cg_prov) if _cg_prov else ''
                            if _cg_region and _cg_region not in ('', 'ไม่ระบุ') and _cg_region != trip_original_region:
                                safe_print(f"      🛑 CAND-GROUP GUARD: ตัด {gc} ภาค {_cg_region} ≠ {trip_original_region}")
                                continue  # ไม่เพิ่มเข้ากลุ่ม — น้ำหนัก/คิวก็ไม่นับ
                            # ⚡ ถ้าไม่รู้ province/region → เช็คระยะจาก candidate
                            if (not _cg_region or _cg_region == 'ไม่ระบุ') and str(gc).upper() != str(candidate_code).upper():
                                _cg_lat = float(gc_row.iloc[0].get('_lat', 0) or 0)
                                _cg_lon = float(gc_row.iloc[0].get('_lon', 0) or 0)
                                _cd_lat = float(candidate_row.get('_lat', 0) or 0)
                                _cd_lon = float(candidate_row.get('_lon', 0) or 0)
                                if _cg_lat > 0 and _cg_lon > 0 and _cd_lat > 0 and _cd_lon > 0:
                                    _cg_dist = haversine_distance(_cg_lat, _cg_lon, _cd_lat, _cd_lon)
                                    if _cg_dist > 5.0:
                                        safe_print(f"      🛑 CAND-GROUP DIST GUARD: ตัด {gc} ห่าง {_cg_dist:.1f}km (ไม่รู้จังหวัด)")
                                        continue
                        group_weight += gc_row.iloc[0]['Weight']
                        group_cube += gc_row.iloc[0]['Cube']
                        group_codes_valid.append(gc_row.iloc[0]['Code'])
                
                if not group_codes_valid:
                    continue
                
                # 🚫 เช็ค vehicle constraint ของสาขาทั้งกลุ่ม
                vehicle_rank = {'4W': 1, 'JB': 2, '6W': 3}
                group_min_max_rank = 3
                for gc in group_codes_valid:
                    gc_max_vehicle = get_max_vehicle_for_branch(gc)
                    gc_max_rank = vehicle_rank.get(gc_max_vehicle, 3)
                    group_min_max_rank = min(group_min_max_rank, gc_max_rank)
                
                # เช็ค allowed vehicles (รวมสาขาทั้งกลุ่ม)
                test_codes = trip_codes + group_codes_valid
                test_allowed = get_allowed_from_codes(test_codes, ['4W', 'JB', '6W'])
                if not test_allowed:
                    # ข้อจำกัดรถไม่เข้ากัน → ปิดทริป (ไม่ข้ามไปสาขาอื่น เพื่อไม่ให้มั่วสาขา)
                    if len(group_codes_valid) > 1:
                        safe_print(f"      🛑 กลุ่ม {group_codes_valid} (ใกล้สุด) ข้อจำกัดรถไม่เข้ากัน → ปิดทริป {trip_counter}")
                    else:
                        safe_print(f"      🛑 สาขา {candidate_code} (ใกล้สุด) ข้อจำกัดรถไม่เข้ากัน → ปิดทริป {trip_counter}")
                    break
                
                # เช็คน้ำหนัก/ปริมาตร/drops รวมทั้งกลุ่ม
                test_weight = trip_weight + group_weight
                test_cube = trip_cube + group_cube
                test_drops = len(test_codes)

                # หา buffer ที่ใช้
                test_is_punthai = all(branch_bu_cache.get(c, False) for c in test_codes)
                buffer = punthai_buffer if test_is_punthai else maxmart_buffer
                limits = PUNTHAI_LIMITS if test_is_punthai else LIMITS

                # 🎯 ลอจิกมนุษย์: เลือกรถเล็กสุดที่รับโหลดได้ ไม่เกินข้อจำกัดสาขา
                # ลำดับ: 4W → JB → 6W  เล็กก่อน ถ้าเกิน → ขึ้นไปรถใหญ่ขึ้น
                # ถ้าข้อจำกัดสาขาห้าม upgrade (group_min_max_rank=1) หรือโหลดเกินทุกรถ → ปิดทริป
                selected_vehicle = None
                for veh in ['4W', 'JB', '6W']:
                    veh_rank = vehicle_rank.get(veh, 3)
                    if veh_rank > group_min_max_rank:
                        break  # เกินข้อจำกัดสาขา — ไม่ลอง upgrade ต่อ
                    if veh not in test_allowed:
                        continue
                    lim = limits[veh]
                    if (test_weight <= lim['max_w'] * buffer and
                            test_cube <= lim['max_c'] * buffer and
                            test_drops <= lim['max_drops']):
                        selected_vehicle = veh
                        break  # เล็กสุดที่รับโหลดได้

                if not selected_vehicle:
                    # ไม่มีรถที่รับโหลดได้ในข้อจำกัดสาขา → ปิดทริป
                    if len(group_codes_valid) > 1:
                        safe_print(f"      🛑 กลุ่ม {len(group_codes_valid)} สาขา (ใกล้สุด) โหลดเกินทุกรถที่อนุญาต → ปิดทริป {trip_counter}")
                    else:
                        safe_print(f"      🛑 สาขา {candidate_code} (ใกล้สุด) โหลดเกินทุกรถที่อนุญาต → ปิดทริป {trip_counter}")
                    break
                
                # ✅ เพิ่มสาขาทั้งกลุ่มเข้าทริป (จุดส่งเดียวกัน ≤200m)
                # กำหนด max_w/max_c จาก selected_vehicle (ใช้เช็ค utilization ด้านล่าง)
                max_w = limits[selected_vehicle]['max_w'] * buffer
                max_c = limits[selected_vehicle]['max_c'] * buffer
                for gc in group_codes_valid:
                    if gc not in trip_codes:
                        trip_codes.append(gc)
                    # ลบออกจาก unassigned
                    if gc in unassigned:
                        unassigned.remove(gc)
                    else:
                        # กรณี code เป็น uppercase/lowercase ต่างกัน
                        for u in list(unassigned):
                            if str(u).upper() == str(gc).upper():
                                unassigned.remove(u)
                                break
                
                trip_weight = test_weight
                trip_cube = test_cube
                trip_allowed = test_allowed
                trip_is_punthai = test_is_punthai
                found_candidate = True
                
                if len(group_codes_valid) > 1:
                    safe_print(f"      🔗 เพิ่มกลุ่ม {len(group_codes_valid)} สาขา (จุดส่งเดียวกัน): {group_codes_valid}")
                
                # 🎯 อัพเดตตำบล/อำเภอของทริป (เพิ่มสาขาใหม่)
                cand_subdistrict = candidate_row.get('_subdistrict', '')
                cand_district = candidate_row.get('_district', '')
                if cand_subdistrict:
                    trip_subdistricts.add(cand_subdistrict)
                if cand_district:
                    trip_districts.add(cand_district)
                
                # 🔒 ล็อคภาค: ถ้า trip_original_region ยังไม่รู้ → ล็อคจาก candidate แรกที่รู้จักภาค
                if (not trip_original_region or trip_original_region == 'ไม่ระบุ'):
                    _cand_prov = candidate_row.get('_province', '')
                    if _cand_prov:
                        _new_region = get_region_name(str(_cand_prov))
                        if _new_region and _new_region != 'ไม่ระบุ':
                            trip_original_region = _new_region
                            safe_print(f"      🔒 ล็อคภาค #{trip_counter}: '{trip_original_region}' (จาก {_cand_prov})")
                
                # เช็คว่าเต็มหรือยัง (>= 90%)
                w_util = trip_weight / max_w
                c_util = trip_cube / max_c
                if max(w_util, c_util) >= 0.90:
                    safe_print(f"      ✅ Trip {trip_counter} เต็ม {max(w_util, c_util)*100:.1f}% ({len(trip_codes)} สาขา)")
                    break  # เต็มแล้ว
                
                break  # หาสาขาเพิ่มได้กลุ่ม/สาขา → วนลูปใหม่หา centroid ใหม่
            
            if not found_candidate:
                # ไม่มีสาขาที่เข้ากันได้ในระยะ 100km → ปิดทริป
                break
        
        # 3️⃣ Assign ทริป
        for code in trip_codes:
            df.loc[df['Code'] == code, 'Trip'] = trip_counter
        
        safe_print(f"   📦 Trip {trip_counter}: {len(trip_codes)} สาขา, {trip_weight:.0f} kg")
        trip_counter += 1
    
    safe_print(f"🎯 จัดทริปเสร็จ: {trip_counter - 1} ทริป")

    # ==========================================
    # Step 6.5: เรียงลำดับทริปใหม่ตามระยะทาง (ไกล → ใกล้)
    # ==========================================
    # คำนวณระยะทางเฉลี่ยของแต่ละทริป
    trip_avg_distances = {}
    for trip_num in df[df['Trip'] > 0]['Trip'].unique():
        trip_data = df[df['Trip'] == trip_num]
        avg_dist = trip_data['_distance_from_dc'].mean()
        trip_avg_distances[trip_num] = avg_dist
    
    # เรียงทริปตามระยะทาง (ไกล → ใกล้)
    sorted_trips = sorted(trip_avg_distances.items(), key=lambda x: x[1], reverse=True)
    
    # สร้าง mapping เลขทริปใหม่
    trip_mapping = {old_num: new_num for new_num, (old_num, _) in enumerate(sorted_trips, start=1)}
    
    # อัพเดตเลขทริปใหม่
    df['Trip'] = df['Trip'].map(lambda x: trip_mapping.get(x, x) if x > 0 else x)

    # ==========================================
    # Step 6.6: 🔄 BRANCH-LEVEL MERGE - ดึงสาขาจากทริปถัดไปมาเติมทริปปัจจุบัน
    # หลักการ: เริ่มจากทริปไกลสุด ถ้ายังไม่เต็ม ดึงสาขาที่ใกล้จากทริปถัดไปมาทีละสาขา
    # ==========================================
    safe_print("🔄 กำลังเติมทริปที่ไม่เต็ม buffer ด้วยสาขาใกล้เคียง...")
    
    def get_trip_capacity(trip_num):
        """คำนวณความจุที่เหลือของทริป"""
        trip_data = df[df['Trip'] == trip_num]
        if len(trip_data) == 0:
            return None
        
        total_w = trip_data['Weight'].sum()
        total_c = trip_data['Cube'].sum()
        codes = trip_data['Code'].tolist()
        
        # เช็ค BU
        is_punthai = all(branch_bu_cache.get(c, False) for c in codes)
        buffer = punthai_buffer if is_punthai else maxmart_buffer
        
        # หารถที่รับ constraint ได้
        max_vehicles = [branch_max_vehicle_cache.get(c, '6W') for c in codes]
        min_priority = min(vehicle_priority.get(v, 3) for v in max_vehicles)
        allowed_vehicle = {1: '4W', 2: 'JB', 3: '6W'}.get(min_priority, '6W')
        
        limits = PUNTHAI_LIMITS if is_punthai else LIMITS
        max_w = limits[allowed_vehicle]['max_w'] * buffer
        max_c = limits[allowed_vehicle]['max_c'] * buffer
        max_drops = limits[allowed_vehicle]['max_drops']
        
        # รวบรวม zone/highway เพื่อตรวจ compatibility ใน merge loop
        _provinces = set(trip_data['_province'].dropna().unique()) if '_province' in trip_data.columns else set()
        _zones     = set(trip_data['_logistics_zone'].dropna().unique()) if '_logistics_zone' in trip_data.columns else set()
        _hws: set  = set()
        if '_zone_highway' in trip_data.columns:
            for _hw in trip_data['_zone_highway'].dropna().unique():
                _hws.update(str(_hw).split('/'))
        # 🔒 รวบรวม regions ของทริป (ใช้กรองไม่ให้ข้ามภาค)
        _regions: set = set()
        for _p in _provinces:
            _r = get_region_name(str(_p)) if _p else 'ไม่ระบุ'
            if _r and _r != 'ไม่ระบุ':
                _regions.add(_r)
        # fallback: ดึงจาก _region_name column
        if not _regions and '_region_name' in trip_data.columns:
            for _rn in trip_data['_region_name'].dropna().unique():
                if _rn and _rn != 'ไม่ระบุ':
                    _regions.add(_rn)
        return {
            'weight': total_w,
            'cube': total_c,
            'codes': codes,
            'drops': len(codes),
            'max_w': max_w,
            'max_c': max_c,
            'max_drops': max_drops,
            'is_punthai': is_punthai,
            'allowed_vehicle': allowed_vehicle,
            'min_priority': min_priority,
            'centroid_lat': trip_data['_lat'].mean(),
            'centroid_lon': trip_data['_lon'].mean(),
            'provinces': _provinces,
            'logistics_zones': _zones,
            'highways': _hws,
            'regions': _regions,
        }
    
    def can_add_branch_to_trip(branch_row, trip_capacity):
        """เช็คว่าสามารถเพิ่มสาขานี้เข้าทริปได้หรือไม่"""
        branch_code = branch_row['Code']
        branch_w = branch_row['Weight']
        branch_c = branch_row['Cube']
        branch_vehicle = branch_max_vehicle_cache.get(branch_code, '6W')
        branch_priority = vehicle_priority.get(branch_vehicle, 3)

        # 🚫 เช็ค vehicle constraint: effective vehicle = min(trip, branch)
        # ถ้าเพิ่มสาขานี้แล้วต้อง downgrade รถ → ตรวจว่าโหลดรวมยังพอดีรถเล็กลงไหม
        effective_priority = min(trip_capacity['min_priority'], branch_priority)
        effective_vehicle = {1: '4W', 2: 'JB', 3: '6W'}.get(effective_priority, '6W')
        is_punthai = trip_capacity.get('is_punthai', False)
        eff_limits = (PUNTHAI_LIMITS if is_punthai else LIMITS)[effective_vehicle]
        eff_buffer = punthai_buffer if is_punthai else maxmart_buffer

        # เช็คน้ำหนัก/ปริมาตร/drops กับรถที่ effective จริงๆ
        new_w = trip_capacity['weight'] + branch_w
        new_c = trip_capacity['cube'] + branch_c
        new_drops = trip_capacity['drops'] + 1

        if new_w > eff_limits['max_w'] * eff_buffer:
            return False, f"น้ำหนักเกิน ({effective_vehicle})"
        if new_c > eff_limits['max_c'] * eff_buffer:
            return False, f"ปริมาตรเกิน ({effective_vehicle})"
        if new_drops > eff_limits['max_drops']:
            return False, f"Drop เกิน ({effective_vehicle})"

        return True, "OK"
    
    def get_nearby_branches(branch_row, all_branches_df, max_dist_km=6.0):
        """หาสาขาที่อยู่ใกล้กัน (ตำบลเดียวกัน หรือ ห่างกัน < 6 km)"""
        branch_lat = branch_row['_lat']
        branch_lon = branch_row['_lon']
        branch_subdistrict = branch_row.get('_subdistrict', '')
        branch_code = branch_row['Code']
        
        nearby_codes = []
        
        for _, other_row in all_branches_df.iterrows():
            other_code = other_row['Code']
            if other_code == branch_code:
                continue
            
            # 1. ตำบลเดียวกัน → ต้องมาด้วยกัน
            if other_row.get('_subdistrict', '') == branch_subdistrict and branch_subdistrict:
                nearby_codes.append(other_code)
                continue
            
            # 2. ห่างกัน < 6 km → ต้องมาด้วยกัน
            other_lat = other_row['_lat']
            other_lon = other_row['_lon']
            if other_lat > 0 and other_lon > 0 and branch_lat > 0 and branch_lon > 0:
                dist = haversine_distance(branch_lat, branch_lon, other_lat, other_lon)
                if dist <= max_dist_km:
                    nearby_codes.append(other_code)
        
        return nearby_codes
    
    # วนลูปทริปจากไกลสุด (1) ไปใกล้สุด
    all_trips = sorted(df[df['Trip'] > 0]['Trip'].unique())
    moved_branches = 0
    
    for i, current_trip in enumerate(all_trips[:-1]):  # ไม่รวมทริปสุดท้าย
        trip_cap = get_trip_capacity(current_trip)
        if not trip_cap:
            continue
        
        # เช็คว่าทริปนี้ยังมีที่เหลือไหม
        w_util = trip_cap['weight'] / trip_cap['max_w']
        c_util = trip_cap['cube'] / trip_cap['max_c']
        
        if max(w_util, c_util) >= 0.95:  # ถ้าเต็มแล้วไม่ต้องเติม
            continue
        
        # หาสาขาจากทริปถัดไปที่ใกล้กับทริปนี้
        for next_trip in all_trips[i+1:]:
            next_trip_data = df[df['Trip'] == next_trip].copy()
            if len(next_trip_data) == 0:
                continue
            
            # คำนวณระยะห่างของแต่ละสาขาใน next_trip จาก centroid ของ current_trip
            next_trip_data['_dist_to_current'] = next_trip_data.apply(
                lambda row: haversine_distance(
                    trip_cap['centroid_lat'], trip_cap['centroid_lon'],
                    row['_lat'], row['_lon']
                ) if row['_lat'] > 0 and row['_lon'] > 0 else 999,
                axis=1
            )
            
            # เรียงตามระยะใกล้สุดก่อน
            next_trip_data = next_trip_data.sort_values('_dist_to_current')
            
            # เก็บสาขาที่ย้ายแล้วเพื่อไม่ให้ซ้ำ
            already_moved = set()
            
            # ดึงสาขาที่ใกล้และเข้ากันได้
            for _, branch_row in next_trip_data.iterrows():
                branch_code = branch_row['Code']
                
                # ข้ามถ้าย้ายไปแล้ว
                if branch_code in already_moved:
                    continue
                
                dist_to_trip = branch_row['_dist_to_current']
                
                # ไกลเกิน 80 km ไม่ดึง
                if dist_to_trip > 80:
                    continue
                
                # อัพเดต trip_cap เพราะอาจมีการเพิ่มสาขาแล้ว
                trip_cap = get_trip_capacity(current_trip)
                if not trip_cap:
                    break
                
                # เช็คว่าเต็มหรือยัง
                w_util = trip_cap['weight'] / trip_cap['max_w']
                c_util = trip_cap['cube'] / trip_cap['max_c']
                if max(w_util, c_util) >= 0.95:
                    break  # เต็มแล้ว หยุด
                
                # 🚫 Zone + Region compatibility: ห้ามรวมสาขาคนละทิศ/highway/ภาค
                _b_prov = branch_row.get('_province', '')
                _b_zone = branch_row.get('_logistics_zone', '')
                _b_hw   = branch_row.get('_zone_highway', '')
                _b_hws  = set(str(_b_hw).split('/')) if _b_hw else set()
                # 🔒 ตรวจภาคก่อน — คำนวณจาก province โดยตรง
                _b_region = get_region_name(str(_b_prov)) if _b_prov else ''
                _trip_regions = trip_cap.get('regions', set())
                _region_ok = (
                    not _b_region or _b_region == 'ไม่ระบุ' or   # candidate ไม่ทราบภาค
                    (not _trip_regions and (not _b_region or _b_region == 'ไม่ระบุ')) or  # ทั้งคู่ไม่ทราบ → zone guard
                    (len(_trip_regions) == 1 and _b_region in _trip_regions) or  # ทริปมีภาคเดียว ตรงกัน
                    (len(_trip_regions) > 1 and _b_region in _trip_regions)   # ทริปมีหลายภาค (ผิดปกติ — zone guard ดูแล)
                )
                if not _region_ok:
                    safe_print(f"      🚫 merge skip {branch_code} ภาคต่างกัน ({_b_region}/{_b_prov} ≠ {_trip_regions})")
                    continue
                # 🛑 Distance fallback: สาขาไม่รู้จังหวัด + ทริปมีภาคที่รู้ → เช็คระยะ (>15km block)
                if (not _b_region or _b_region == 'ไม่ระบุ') and _trip_regions:
                    _bx_lat = float(branch_row.get('_lat', 0) or 0)
                    _bx_lon = float(branch_row.get('_lon', 0) or 0)
                    _tx_lat = float(trip_cap.get('centroid_lat', 0) or 0)
                    _tx_lon = float(trip_cap.get('centroid_lon', 0) or 0)
                    if _bx_lat and _tx_lat:
                        _dp6 = radians(_bx_lat - _tx_lat); _dl6 = radians(_bx_lon - _tx_lon)
                        _a6 = sin(_dp6/2)**2 + cos(radians(_tx_lat))*cos(radians(_bx_lat))*sin(_dl6/2)**2
                        _dist_mg = 2*6371*atan2(sqrt(_a6), sqrt(1-_a6))
                        if _dist_mg > 15.0:
                            safe_print(f"      🛑 MERGE DIST GUARD: ตัด {branch_code} ห่าง {_dist_mg:.1f}km (ไม่รู้จังหวัด ภาคทริป={_trip_regions})")
                            continue
                _zone_ok = (
                    _b_prov in trip_cap.get('provinces', set()) or
                    _b_zone in trip_cap.get('logistics_zones', set()) or
                    bool(trip_cap.get('highways', set()) & _b_hws)
                )
                if not _zone_ok:
                    safe_print(f"      🚫 merge skip {branch_code} ({_b_prov}/{_b_zone}) ≠ trip zone {trip_cap.get('provinces')}")
                    continue

                # เช็คว่าเพิ่มสาขานี้ได้ไหม
                can_add, reason = can_add_branch_to_trip(branch_row, trip_cap)
                
                if can_add:
                    # ✅ ย้ายสาขานี้มาทริปปัจจุบัน
                    df.loc[df['Code'] == branch_code, 'Trip'] = current_trip
                    already_moved.add(branch_code)
                    moved_branches += 1
                    safe_print(f"   ✅ ย้าย {branch_code} จาก Trip {next_trip} → Trip {current_trip} (ห่าง {dist_to_trip:.1f} km)")
                    
                    # 🔗 หาสาขาใกล้เคียง (ตำบลเดียวกัน หรือ ห่าง < 6 km) แล้วย้ายมาด้วย
                    nearby_codes = get_nearby_branches(branch_row, next_trip_data[~next_trip_data['Code'].isin(already_moved)])
                    
                    for nearby_code in nearby_codes:
                        if nearby_code in already_moved:
                            continue
                        
                        # อัพเดต trip_cap อีกครั้ง
                        trip_cap = get_trip_capacity(current_trip)
                        if not trip_cap:
                            break
                        
                        # เช็คว่าเต็มหรือยัง
                        w_util = trip_cap['weight'] / trip_cap['max_w']
                        c_util = trip_cap['cube'] / trip_cap['max_c']
                        if max(w_util, c_util) >= 0.95:
                            break
                        
                        nearby_row = next_trip_data[next_trip_data['Code'] == nearby_code]
                        if len(nearby_row) == 0:
                            continue
                        nearby_row = nearby_row.iloc[0]
                        
                        # zone + region check สำหรับ nearby สาขาด้วย
                        _nb_prov = nearby_row.get('_province', '')
                        _nb_zone = nearby_row.get('_logistics_zone', '')
                        _nb_hw   = nearby_row.get('_zone_highway', '')
                        _nb_hws  = set(str(_nb_hw).split('/')) if _nb_hw else set()
                        # 🔒 region check ก่อน
                        _nb_region = get_region_name(str(_nb_prov)) if _nb_prov else ''
                        _trip_regions_nb = trip_cap.get('regions', set())
                        _nb_region_ok = (
                            not _nb_region or _nb_region == 'ไม่ระบุ' or   # candidate ไม่ทราบภาค
                            (not _trip_regions_nb and (not _nb_region or _nb_region == 'ไม่ระบุ')) or  # ทั้งคู่ไม่ทราบ
                            (len(_trip_regions_nb) == 1 and _nb_region in _trip_regions_nb) or  # ทริปมีภาคเดียว
                            (len(_trip_regions_nb) > 1 and _nb_region in _trip_regions_nb)  # ทริปมีหลายภาค
                        )
                        if not _nb_region_ok:
                            continue
                        # 🛑 Nearby distance fallback: สาขาไม่รู้จังหวัด + ทริปมีภาค → >15km block
                        if (not _nb_region or _nb_region == 'ไม่ระบุ') and _trip_regions_nb:
                            _nbx_lat = float(nearby_row.get('_lat', 0) or 0)
                            _nbx_lon = float(nearby_row.get('_lon', 0) or 0)
                            _txn_lat = float(trip_cap.get('centroid_lat', 0) or 0)
                            _txn_lon = float(trip_cap.get('centroid_lon', 0) or 0)
                            if _nbx_lat and _txn_lat:
                                _dp7 = radians(_nbx_lat - _txn_lat); _dl7 = radians(_nbx_lon - _txn_lon)
                                _a7 = sin(_dp7/2)**2 + cos(radians(_txn_lat))*cos(radians(_nbx_lat))*sin(_dl7/2)**2
                                _dist_mg7 = 2*6371*atan2(sqrt(_a7), sqrt(1-_a7))
                                if _dist_mg7 > 15.0:
                                    continue
                        _nb_zone_ok = (
                            _nb_prov in trip_cap.get('provinces', set()) or
                            _nb_zone in trip_cap.get('logistics_zones', set()) or
                            bool(trip_cap.get('highways', set()) & _nb_hws)
                        )
                        if not _nb_zone_ok:
                            continue
                        can_add_nearby, _ = can_add_branch_to_trip(nearby_row, trip_cap)
                        if can_add_nearby:
                            df.loc[df['Code'] == nearby_code, 'Trip'] = current_trip
                            already_moved.add(nearby_code)
                            moved_branches += 1
                            safe_print(f"   🔗 ย้ายด้วย {nearby_code} (ใกล้กัน/ตำบลเดียวกัน)")
        
        # หลังจากเติมเสร็จ เช็คอีกครั้ง
        trip_cap = get_trip_capacity(current_trip)
        if trip_cap:
            w_util = trip_cap['weight'] / trip_cap['max_w']
            c_util = trip_cap['cube'] / trip_cap['max_c']
            safe_print(f"   📊 Trip {current_trip}: {max(w_util, c_util)*100:.1f}% ({len(trip_cap['codes'])} สาขา)")
    
    if moved_branches > 0:
        safe_print(f"🔄 ย้ายสาขาเสร็จ: ย้าย {moved_branches} สาขา")
        
        # ลบทริปที่ว่างเปล่า
        empty_trips = [t for t in df['Trip'].unique() if t > 0 and len(df[df['Trip'] == t]) == 0]
        
        # Renumber ทริปใหม่หลังย้าย
        remaining_trips = sorted(df[df['Trip'] > 0]['Trip'].unique())
        trip_renumber = {old: new for new, old in enumerate(remaining_trips, start=1)}
        df['Trip'] = df['Trip'].map(lambda x: trip_renumber.get(x, x) if x > 0 else x)

    # ==========================================
    # Step 6.7: 🔍 REGION AUDIT — ตรวจและแยกทริปที่มีการปนภาค
    # ==========================================
    safe_print("🔍 ตรวจสอบการปนภาคใน trips...")
    _audit_fixed = 0
    _max_trip_now = df[df['Trip'] > 0]['Trip'].max() if len(df[df['Trip'] > 0]) > 0 else 0
    for _aud_trip in sorted(df[df['Trip'] > 0]['Trip'].unique()):
        _aud_data = df[df['Trip'] == _aud_trip]
        _aud_regions = {}
        for _, _aud_row in _aud_data.iterrows():
            _ap = str(_aud_row.get('_province', '') or '')
            _ar = get_region_name(_ap) if _ap else ''
            # fallback: ใช้ _region_name column ถ้า _province ว่าง
            if (not _ar or _ar == 'ไม่ระบุ'):
                _ar = str(_aud_row.get('_region_name', '') or '')
            if _ar and _ar != 'ไม่ระบุ':
                _aud_regions[_ar] = _aud_regions.get(_ar, 0) + 1
        if len(_aud_regions) <= 1:
            continue  # ไม่มีการปนภาค
        # พบการปนภาค → แยกสาขา minority ออกเป็นทริปใหม่
        # ใช้ region ที่มีสาขามากที่สุดเป็น dominant (ถ้าเท่ากัน ใช้ตามตำแหน่งใน sort)
        _dominant = max(_aud_regions, key=lambda k: (_aud_regions[k], ['เหนือ','อีสาน','ตะวันออก','กลาง','ตะวันตก','ใต้'].index(k) if k in ['เหนือ','อีสาน','ตะวันออก','กลาง','ตะวันตก','ใต้'] else 99))
        _minority_codes = []
        for _, _aud_row in _aud_data.iterrows():
            _ap2 = str(_aud_row.get('_province', '') or '')
            _ar2 = get_region_name(_ap2) if _ap2 else ''
            if (not _ar2 or _ar2 == 'ไม่ระบุ'):
                _ar2 = str(_aud_row.get('_region_name', '') or '')
            if _ar2 and _ar2 != 'ไม่ระบุ' and _ar2 != _dominant:
                _minority_codes.append(_aud_row['Code'])
        if _minority_codes:
            _max_trip_now += 1
            df.loc[df['Code'].isin(_minority_codes), 'Trip'] = _max_trip_now
            safe_print(f"   ⚠️ AUDIT: Trip {_aud_trip} ปนภาค {_aud_regions} → แยก {_minority_codes} → Trip ใหม่ {_max_trip_now}")
            _audit_fixed += 1
    if _audit_fixed > 0:
        safe_print(f"   🔧 AUDIT: แก้ไขการปนภาค {_audit_fixed} ทริป")
        # Renumber หลัง audit
        _aud_remaining = sorted(df[df['Trip'] > 0]['Trip'].unique())
        _aud_remap = {old: new for new, old in enumerate(_aud_remaining, start=1)}
        df['Trip'] = df['Trip'].map(lambda x: _aud_remap.get(x, x) if x > 0 else x)
    else:
        safe_print("   ✅ ไม่พบการปนภาค")

    # ==========================================
    # Step 7: สร้าง Summary + Central Rule + Punthai Drop Limits
    # ==========================================
    summary_data = []
    
    for trip_num in sorted(df['Trip'].unique()):
        if trip_num == 0:
            continue
        
        trip_data = df[df['Trip'] == trip_num]
        total_w = trip_data['Weight'].sum()
        total_c = trip_data['Cube'].sum()
        trip_codes = trip_data['Code'].unique()
        trip_drops = len(trip_codes)
        
        # หาภาคของทริป (ใช้ภาคแรก)
        trip_region = trip_data['_region_name'].iloc[0] if '_region_name' in trip_data.columns else 'ไม่ระบุ'
        
        # หารถที่เหมาะสม (รวม Central Rule)
        max_vehicles = [get_max_vehicle_for_branch(c) for c in trip_codes]
        min_max_size = min(vehicle_priority.get(v, 3) for v in max_vehicles)
        max_allowed_vehicle = {1: '4W', 2: 'JB', 3: '6W'}.get(min_max_size, '6W')
        
        # ตรวจ BU ของทริป
        is_punthai_only_trip = True
        for _, r in trip_data.iterrows():
            bu = str(r.get('BU', '')).upper()
            if bu not in ['211', 'PUNTHAI']:
                is_punthai_only_trip = False
                break
        
        buffer = punthai_buffer if is_punthai_only_trip else maxmart_buffer
        buffer_pct = int(buffer * 100)
        buffer_label = f"🅿️ {buffer_pct}%" if is_punthai_only_trip else f"🅼 {buffer_pct}%"
        trip_type = 'punthai' if is_punthai_only_trip else 'maxmart'
        
        # 🎯 เลือกรถตามภาค + ข้อจำกัดสาขา
        # เหนือ/ใต้ → ใช้รถใหญ่สุดที่อนุญาตเสมอ (ไม่ downgrade — เส้นทางไกล)
        # ภาคอื่น  → เล็กสุดที่รับโหลดได้ (ประหยัดรถ)
        limits_to_check = PUNTHAI_LIMITS if is_punthai_only_trip else LIMITS
        is_long_haul = str(trip_region) in ('เหนือ', 'ใต้')
        suggested = max_allowed_vehicle  # fallback = รถใหญ่สุดที่อนุญาต
        source = "📋 จำกัดสาขา" if min_max_size < 3 else "🤖 อัตโนมัติ"
        if is_long_haul:
            # เหนือ/ใต้: ใช้ max_allowed_vehicle ตรงๆ ไม่ลอง downgrade
            if min_max_size >= 3:
                source = "🚛 ไกล (เหนือ/ใต้)"
            else:
                source = "📋 จำกัดสาขา (เหนือ/ใต้)"
        else:
            # ภาคอื่น: ลอง 4W → JB → 6W เลือกเล็กสุดที่รับโหลดได้
            for _veh in ['4W', 'JB', '6W']:
                _vr = vehicle_priority.get(_veh, 3)
                if _vr > min_max_size:
                    break  # ห้ามเกินข้อจำกัดสาขา
                _lim = limits_to_check[_veh]
                if (total_w <= _lim['max_w'] * buffer and
                        total_c <= _lim['max_c'] * buffer and
                        trip_drops <= _lim['max_drops']):
                    suggested = _veh
                    if _vr < min_max_size:
                        source = "🔽 Downgrade (ขนาดพอดี)"
                    break  # เล็กสุดที่รับโหลดได้
        
        # 🔒 Punthai Drop Limit Check
        if is_punthai_only_trip:
            punthai_drop_limit = PUNTHAI_LIMITS.get(suggested, {}).get('max_drops', 999)
            if trip_drops > punthai_drop_limit:
                # ต้องเพิ่มขนาดรถเพื่อรองรับ drops - แต่ห้ามเกินข้อจำกัดสาขา!
                if suggested == '4W' and trip_drops <= PUNTHAI_LIMITS['JB']['max_drops']:
                    # เช็คว่าสาขาอนุญาต JB ไหม
                    if min_max_size >= 2:  # JB หรือ 6W
                        suggested = 'JB'
                        source += " → JB (Drop Limit)"
                    else:
                        # สาขาจำกัดแค่ 4W - ไม่สามารถ upgrade ได้!
                        source += " ⚠️ Drop เกิน (แต่สาขาจำกัด 4W)"
                elif suggested == 'JB' or trip_drops > PUNTHAI_LIMITS['JB']['max_drops']:
                    # เช็คว่าสาขาอนุญาต 6W ไหม
                    if min_max_size >= 3:  # 6W
                        suggested = '6W'
                        source += " → 6W (Drop Limit)"
                    else:
                        # 🚫 สาขาจำกัดไม่เกิน JB - ห้ามใช้ 6W!
                        suggested = max_allowed_vehicle  # ใช้รถตามข้อจำกัดสาขา (JB หรือ 4W)
                        source += f" ⚠️ Drop เกิน (แต่สาขาจำกัด {max_allowed_vehicle})"
        
        # คำนวณ utilization - ใช้ limits ตาม BU type
        max_util_threshold = buffer * 100  # 100% หรือ 110% ตาม BU
        limits_for_util = PUNTHAI_LIMITS if is_punthai_only_trip else LIMITS
        if suggested in limits_for_util:
            w_util = (total_w / limits_for_util[suggested]['max_w']) * 100
            c_util = (total_c / limits_for_util[suggested]['max_c']) * 100
            max_util = max(w_util, c_util)
            
            # ถ้าเกิน threshold ตาม BU ต้องเพิ่มขนาดรถ
            if max_util > max_util_threshold:
                # 🚫 ห้าม upgrade เกินข้อจำกัดสาขา!
                if suggested == '4W' and min_max_size >= 2:
                    jb_util = max((total_w / limits_for_util['JB']['max_w']), (total_c / limits_for_util['JB']['max_c'])) * 100
                    if jb_util <= max_util_threshold:
                        suggested = 'JB'
                        source += " → JB"
                        w_util = (total_w / limits_for_util['JB']['max_w']) * 100
                        c_util = (total_c / limits_for_util['JB']['max_c']) * 100
                    elif min_max_size >= 3:  # สาขาอนุญาต 6W
                        suggested = '6W'
                        source += " → 6W"
                        w_util = (total_w / limits_for_util['6W']['max_w']) * 100
                        c_util = (total_c / limits_for_util['6W']['max_c']) * 100
                    else:
                        # 🚫 ไม่สามารถ upgrade ได้ (สาขาจำกัด JB) → ยังคงใช้ JB (จะเกิน buffer)
                        suggested = 'JB'
                        source += " ⚠️ เกินแต่สาขาจำกัด"
                        w_util = (total_w / limits_for_util['JB']['max_w']) * 100
                        c_util = (total_c / limits_for_util['JB']['max_c']) * 100
                elif suggested == 'JB' and min_max_size >= 3:  # สาขาอนุญาต 6W
                    suggested = '6W'
                    source += " → 6W"
                    w_util = (total_w / limits_for_util['6W']['max_w']) * 100
                    c_util = (total_c / limits_for_util['6W']['max_c']) * 100
                elif suggested == 'JB' and min_max_size < 3:
                    # 🚫 ไม่สามารถ upgrade เป็น 6W ได้ (สาขาจำกัด JB)
                    source += " ⚠️ เกินแต่สาขาจำกัด"
                elif suggested == '4W' and min_max_size < 2:
                    # 🚫 ไม่สามารถ upgrade ได้ (สาขาจำกัด 4W)
                    source += " ⚠️ เกินแต่สาขาจำกัด"
        else:
            w_util = c_util = 0
        
        # คำนวณระยะทางรวม - ใช้พิกัดจาก DataFrame โดยตรง
        total_distance = 0
        branch_coords = []
        for code in trip_codes:
            # ดึงพิกัดจาก df (มีคอลัมน์ _lat, _lon)
            branch_data = df[df['Code'] == code]
            if not branch_data.empty:
                lat = branch_data.iloc[0].get('_lat', 0)
                lon = branch_data.iloc[0].get('_lon', 0)
                if lat > 0 and lon > 0:
                    branch_coords.append((lat, lon))
        
        if branch_coords:
            # DC → สาขาแรก
            total_distance += haversine_distance(DC_WANG_NOI_LAT, DC_WANG_NOI_LON, branch_coords[0][0], branch_coords[0][1])
            # สาขา → สาขา
            for i in range(len(branch_coords) - 1):
                total_distance += haversine_distance(branch_coords[i][0], branch_coords[i][1], branch_coords[i+1][0], branch_coords[i+1][1])
            # สาขาสุดท้าย → DC
            total_distance += haversine_distance(branch_coords[-1][0], branch_coords[-1][1], DC_WANG_NOI_LAT, DC_WANG_NOI_LON)
        
        summary_data.append({
            'Trip': trip_num,
            'Branches': len(trip_codes),
            'Weight': total_w,
            'Cube': total_c,
            'Truck': f"{suggested} {source}",
            'BU_Type': trip_type,
            'Buffer': buffer_label,
            'Weight_Use%': w_util,
            'Cube_Use%': c_util,
            'Total_Distance': round(total_distance, 1)
        })
    
    # ==========================================
    # 🚨 Step 7.5: ตัดสาขาออกถ้าเกิน buffer หรือรถผิดประเภท (Strict Enforcement)
    # ==========================================
    safe_print("\n📋 Step 7.5: ตรวจสอบและตัดสาขาที่เกิน Buffer + ข้อจำกัดรถ...")
    overflow_branches = []
    
    for i, trip_summary in enumerate(summary_data):
        trip_num = trip_summary['Trip']
        buffer_pct = float(trip_summary['Buffer'].replace('🅿️ ', '').replace('🅼 ', '').replace('%', ''))
        
        # ดึงข้อมูลทริป
        trip_data = df[df['Trip'] == trip_num].copy()
        if trip_data.empty:
            continue
            
        trip_codes = trip_data['Code'].tolist()
        
        # 🚗 หารถที่ถูกต้องตามข้อจำกัดสาขา (รถเล็กสุดที่รับโหลดได้)
        max_vehicles = [get_max_vehicle_for_branch(c) for c in trip_codes]
        vehicle_priority_map = {'4W': 1, 'JB': 2, '6W': 3}
        min_max_size = min(vehicle_priority_map.get(v, 3) for v in max_vehicles)
        max_allowed_v = {1: '4W', 2: 'JB', 3: '6W'}.get(min_max_size, '6W')
        
        # ดึง limits ตาม BU
        bu_type = trip_summary['BU_Type']
        is_punthai = (bu_type == 'punthai')
        buffer = punthai_buffer if is_punthai else maxmart_buffer
        limits = PUNTHAI_LIMITS if is_punthai else LIMITS
        
        # คำนวณน้ำหนัก/คิวก่อน (ใช้เลือกรถ)
        total_w = trip_data['Weight'].sum()
        total_c = trip_data['Cube'].sum()
        
        # 🎯 เลือกรถตามภาค + ข้อจำกัดสาขา (เหนือ/ใต้ → รถใหญ่สุดที่อนุญาต)
        trip_region_75 = trip_data['_region_name'].iloc[0] if '_region_name' in trip_data.columns else 'ไม่ระบุ'
        is_long_haul_75 = str(trip_region_75) in ('เหนือ', 'ใต้')
        correct_vehicle = max_allowed_v  # fallback = รถใหญ่สุดที่อนุญาต
        if not is_long_haul_75:
            # ภาคอื่น: เล็กสุดที่รับโหลดได้
            for _veh in ['4W', 'JB', '6W']:
                _vr = vehicle_priority_map.get(_veh, 3)
                if _vr > min_max_size:
                    break  # ห้ามเกินข้อจำกัดสาขา
                _lim = limits[_veh]
                if (total_w <= _lim['max_w'] * buffer and
                        total_c <= _lim['max_c'] * buffer and
                        len(trip_codes) <= _lim['max_drops']):
                    correct_vehicle = _veh
                    break  # เล็กสุดที่รับโหลดได้
        # เหนือ/ใต้: correct_vehicle = max_allowed_v แล้ว (ไม่ downgrade)
        
        max_w = limits[correct_vehicle]['max_w'] * buffer
        max_c = limits[correct_vehicle]['max_c'] * buffer
        max_drops = limits[correct_vehicle]['max_drops']
        
        w_util = (total_w / limits[correct_vehicle]['max_w']) * 100
        c_util = (total_c / limits[correct_vehicle]['max_c']) * 100
        max_util = max(w_util, c_util)
        
        # อัพเดต summary ด้วยรถที่ถูกต้อง
        if is_long_haul_75:
            truck_source = "🚛 ไกล (เหนือ/ใต้)" if min_max_size >= 3 else "📋 จำกัดสาขา (เหนือ/ใต้)"
        elif min_max_size < 3:
            truck_source = "📋 จำกัดสาขา"
        elif vehicle_priority_map.get(correct_vehicle, 3) < min_max_size:
            truck_source = "🔽 Downgrade (ขนาดพอดี)"
        else:
            truck_source = "🤖 อัตโนมัติ"
        summary_data[i]['Truck'] = f"{correct_vehicle} {truck_source}"
        summary_data[i]['Weight_Use%'] = w_util
        summary_data[i]['Cube_Use%'] = c_util
        
        # ตรวจสอบว่าเกิน buffer หรือ drops หรือไม่
        is_over_buffer = total_w > max_w or total_c > max_c
        is_over_drops = len(trip_codes) > max_drops
        
        if is_over_buffer or is_over_drops:
            reason = "เกิน buffer" if is_over_buffer else f"เกิน drops ({len(trip_codes)}>{max_drops})"
            safe_print(f"   ⚠️ Trip {trip_num} {reason}: {max_util:.1f}% (รถ {correct_vehicle})")
            
            # 🚨 ถ้ามีแค่ 1 สาขา แต่เกิน buffer → ตัดสาขานั้นไป overflow ทั้งหมด
            if len(trip_data) <= 1:
                code = trip_data.iloc[0]['Code'] if len(trip_data) == 1 else None
                if code:
                    df.loc[df['Code'] == code, 'Trip'] = 0
                    overflow_branches.append(code)
                    safe_print(f"      🔪 ตัด {code} ออก (1 สาขาแต่เกิน buffer → overflow)")
                    # ลบ summary ของทริปนี้
                    summary_data[i]['Branches'] = 0
                    summary_data[i]['Weight'] = 0
                    summary_data[i]['Cube'] = 0
                    summary_data[i]['Weight_Use%'] = 0
                    summary_data[i]['Cube_Use%'] = 0
                continue
            
            # เรียงตามระยะทางใกล้สุดก่อน (ตัดสาขาไกลออก)
            trip_data = trip_data.sort_values('_distance_from_dc', ascending=False)
            
            # ใช้รถที่ถูกต้องตามข้อจำกัดสาขา (correct_vehicle ที่คำนวณด้านบน)
            truck_str = correct_vehicle
            
            if truck_str not in limits:
                continue
            
            max_w = limits[truck_str]['max_w'] * buffer
            max_c = limits[truck_str]['max_c'] * buffer
            
            # คำนวณน้ำหนัก/คิวปัจจุบัน
            current_w = trip_data['Weight'].sum()
            current_c = trip_data['Cube'].sum()
            current_drops = len(trip_data)
            
            # ตัดสาขาออกจนกว่าจะไม่เกิน (weight, cube, และ drops)
            codes_to_remove = []
            for _, row in trip_data.iterrows():
                # เช็คทั้ง buffer และ drops
                if current_w <= max_w and current_c <= max_c and current_drops <= max_drops:
                    break  # พอดีแล้ว
                
                code = row['Code']
                codes_to_remove.append(code)
                current_w -= row['Weight']
                current_c -= row['Cube']
                current_drops -= 1
                overflow_branches.append(code)
                safe_print(f"      🔪 ตัด {code} ออก (ไกลสุด {row['_distance_from_dc']:.1f} km)")
            
            # ลบสาขาออกจากทริป (Trip = 0)
            for code in codes_to_remove:
                df.loc[df['Code'] == code, 'Trip'] = 0
            
            # อัพเดต summary
            if codes_to_remove:
                new_trip_data = df[df['Trip'] == trip_num]
                new_w = new_trip_data['Weight'].sum()
                new_c = new_trip_data['Cube'].sum()
                new_w_util = (new_w / (limits[truck_str]['max_w'])) * 100
                new_c_util = (new_c / (limits[truck_str]['max_c'])) * 100
                
                summary_data[i]['Branches'] = len(new_trip_data)
                summary_data[i]['Weight'] = new_w
                summary_data[i]['Cube'] = new_c
                summary_data[i]['Weight_Use%'] = new_w_util
                summary_data[i]['Cube_Use%'] = new_c_util
    
    # จัดทริปใหม่สำหรับ overflow branches
    if overflow_branches:
        safe_print(f"\n   📦 สาขาที่ถูกตัด: {len(overflow_branches)} สาขา → จัดทริปใหม่...")
        max_trip = df['Trip'].max()
        
        # 🎯 แยกตามข้อจำกัดรถ เพื่อไม่ให้ JB/4W ไปรวมกับ 6W
        overflow_by_max_vehicle = {}
        for code in overflow_branches:
            max_veh = get_max_vehicle_for_branch(code)
            if max_veh not in overflow_by_max_vehicle:
                overflow_by_max_vehicle[max_veh] = []
            overflow_by_max_vehicle[max_veh].append(code)
        
        # จัดทริปแยกตามข้อจำกัด + แบ่งตาม buffer
        for max_veh in ['4W', 'JB', '6W']:
            if max_veh not in overflow_by_max_vehicle:
                continue
            
            codes_for_veh = overflow_by_max_vehicle[max_veh]
            if not codes_for_veh:
                continue
            
            # 🎯 แบ่งสาขา overflow เป็นทริปย่อยตาม buffer limit
            remaining_codes = list(codes_for_veh)
            
            while remaining_codes:
                new_trip = max_trip + 1
                max_trip = new_trip
                
                # คำนวณ limits - ต้องเช็ค BU ของสาขาก่อน!
                # ตรวจสอบว่าสาขาที่เหลือเป็น Punthai ล้วนหรือไม่
                first_code = remaining_codes[0]
                first_row = df[df['Code'] == first_code]
                first_bu = str(first_row['BU'].values[0] if len(first_row) > 0 else '').upper()
                is_punthai_overflow = first_bu in ['211', 'PUNTHAI']
                
                overflow_buffer = punthai_buffer if is_punthai_overflow else maxmart_buffer
                overflow_limits = PUNTHAI_LIMITS if is_punthai_overflow else LIMITS
                max_w = overflow_limits[max_veh]['max_w'] * overflow_buffer
                max_c = overflow_limits[max_veh]['max_c'] * overflow_buffer
                max_drops = overflow_limits[max_veh]['max_drops']
                
                # เพิ่มสาขาจนกว่าจะเต็ม buffer
                trip_codes = []
                trip_weight = 0
                trip_cube = 0
                
                for code in list(remaining_codes):
                    code_row = df[df['Code'] == code]
                    if code_row.empty:
                        remaining_codes.remove(code)
                        continue
                    
                    code_w = code_row.iloc[0]['Weight']
                    code_c = code_row.iloc[0]['Cube']
                    
                    # เช็คว่าเพิ่มได้หรือไม่
                    if (trip_weight + code_w <= max_w and 
                        trip_cube + code_c <= max_c and 
                        len(trip_codes) < max_drops):
                        trip_codes.append(code)
                        trip_weight += code_w
                        trip_cube += code_c
                        remaining_codes.remove(code)
                    elif len(trip_codes) == 0:
                        # ถ้าสาขาเดียวเกิน buffer ก็ต้องเพิ่มอยู่ดี
                        trip_codes.append(code)
                        trip_weight += code_w
                        trip_cube += code_c
                        remaining_codes.remove(code)
                        break
                    else:
                        # เต็มแล้ว ปิดทริปนี้
                        break
                
                # Assign trip
                for code in trip_codes:
                    df.loc[df['Code'] == code, 'Trip'] = new_trip
                
                # เพิ่ม summary
                if trip_codes:
                    is_overflow_punthai = all(
                        str(df[df['Code'] == c]['BU'].values[0] if len(df[df['Code'] == c]) > 0 else '').upper() in ['211', 'PUNTHAI'] 
                        for c in trip_codes
                    )
                    overflow_limits_final = PUNTHAI_LIMITS if is_overflow_punthai else LIMITS
                    overflow_buffer_final = punthai_buffer if is_overflow_punthai else maxmart_buffer
                    buffer_label = f"🅿️ {int(overflow_buffer_final*100)}%" if is_overflow_punthai else f"🅼 {int(overflow_buffer_final*100)}%"
                    
                    summary_data.append({
                        'Trip': new_trip,
                        'Branches': len(trip_codes),
                        'Weight': trip_weight,
                        'Cube': trip_cube,
                        'Truck': f'{max_veh} 🔪 ตัดออก',
                        'BU_Type': 'punthai' if is_overflow_punthai else 'mixed',
                        'Buffer': buffer_label,
                        'Weight_Use%': (trip_weight / overflow_limits_final[max_veh]['max_w']) * 100,
                        'Cube_Use%': (trip_cube / overflow_limits_final[max_veh]['max_c']) * 100,
                        'Total_Distance': 0
                    })
                    safe_print(f"   ✅ สร้าง Trip {new_trip} ใหม่สำหรับสาขา {max_veh} ({len(trip_codes)} สาขา, {trip_weight:.0f}kg)")
    
    summary_df = pd.DataFrame(summary_data)
    
    # ==========================================
    # Step 8: เพิ่มคอลัมน์เสริม
    # ==========================================
    # เพิ่มคอลัมน์รถ
    trip_truck_map = {}
    for _, row in summary_df.iterrows():
        trip_truck_map[row['Trip']] = row['Truck']
    df['Truck'] = df['Trip'].map(trip_truck_map)
    
    # เพิ่มคอลัมน์ Region
    df['Region'] = df['_region_name']
    
    # เพิ่มคอลัมน์ Province/District/Subdistrict (ถ้ายังไม่มี)
    if 'Province' not in df.columns:
        df['Province'] = df['_province']
    if 'District' not in df.columns:
        df['District'] = df['_district']
    if 'Subdistrict' not in df.columns:
        df['Subdistrict'] = df['_subdistrict']
    
    # เพิ่มคอลัมน์ระยะทางจาก DC
    df['Distance_from_DC'] = df['_distance_from_dc'].round(1)
    
    # เพิ่มคอลัมน์ MaxVehicle constraint
    df['MaxVehicle'] = df['_max_vehicle']
    
    # 🚨 เพิ่มคอลัมน์เช็ครถ - ตรวจสอบว่ารถที่จัดตรงกับข้อจำกัดหรือไม่
    def check_vehicle_compliance(row):
        """ตรวจสอบว่ารถที่จัดไปตรงกับข้อจำกัดหรือไม่"""
        if row['Trip'] == 0:
            return '⚠️ ไม่ได้จัด'
        
        max_allowed = row['_max_vehicle']
        truck_assigned = str(row.get('Truck', '')).split()[0] if pd.notna(row.get('Truck')) else ''
        
        # แปลง JB เป็น 4WJ ถ้าจำเป็น
        if truck_assigned == '4WJ':
            truck_assigned = 'JB'
        
        # Vehicle hierarchy: 4W < JB < 6W
        vehicle_rank = {'4W': 1, 'JB': 2, '6W': 3}
        
        if max_allowed not in vehicle_rank or truck_assigned not in vehicle_rank:
            return '✅ ใช้ได้'
        
        # ตรวจสอบว่ารถที่จัดเล็กกว่าหรือเท่ากับรถที่อนุญาต
        if vehicle_rank[truck_assigned] <= vehicle_rank[max_allowed]:
            return '✅ ใช้ได้'
        else:
            return f'❌ เกินข้อจำกัด (Max: {max_allowed}, ใช้: {truck_assigned})'
    
    df['VehicleCheck'] = df.apply(check_vehicle_compliance, axis=1)
    
    # ==========================================
    # 🚨 Step 8.5: บังคับแก้ไขสาขาที่เกินข้อจำกัดรถ (Enforce Vehicle Constraints)
    # ==========================================
    safe_print("\n📋 Step 8.5: บังคับข้อจำกัดรถ...")
    vehicle_violations = df[df['VehicleCheck'].str.contains('❌', na=False)]
    
    if len(vehicle_violations) > 0:
        safe_print(f"   ⚠️ พบ {len(vehicle_violations)} สาขาที่ใช้รถเกินข้อจำกัด")
        
        # แยกสาขาที่เกินข้อจำกัดออกมาจัดทริปใหม่
        for _, viol_row in vehicle_violations.iterrows():
            viol_code = viol_row['Code']
            viol_trip = viol_row['Trip']
            max_allowed = viol_row['_max_vehicle']
            
            # หาสาขาอื่นในทริปเดียวกันที่มีข้อจำกัดเดียวกันหรือน้อยกว่า
            same_trip = df[df['Trip'] == viol_trip]
            
            # ตรวจสอบว่าสาขาอื่นในทริปมีข้อจำกัดอย่างไร
            vehicle_rank = {'4W': 1, 'JB': 2, '6W': 3}
            max_allowed_rank = vehicle_rank.get(max_allowed, 3)
            
            # หาสาขาที่ทำให้ต้องใช้รถใหญ่ (น้ำหนัก/คิวมาก หรือ max vehicle ใหญ่กว่า)
            other_branches = same_trip[same_trip['Code'] != viol_code]
            
            if len(other_branches) > 0:
                # ตรวจสอบว่าสาขาอื่นมีข้อจำกัดใหญ่กว่าหรือไม่
                other_max_vehicles = other_branches['_max_vehicle'].apply(lambda x: vehicle_rank.get(x, 3))
                min_other_rank = other_max_vehicles.min()
                
                if min_other_rank > max_allowed_rank:
                    # สาขาอื่นมีข้อจำกัดใหญ่กว่า → ย้ายสาขานี้ออก
                    df.loc[df['Code'] == viol_code, 'Trip'] = 0  # ย้ายออกไปจัดใหม่
                    safe_print(f"      🔄 ย้าย {viol_code} ออกจาก Trip {viol_trip} (Max: {max_allowed})")
    
    # จัดทริปใหม่สำหรับสาขาที่ถูกย้ายออก
    unassigned_violations = df[df['Trip'] == 0]
    if len(unassigned_violations) > 0:
        safe_print(f"   📦 จัดทริปใหม่สำหรับ {len(unassigned_violations)} สาขา...")
        max_trip = df[df['Trip'] > 0]['Trip'].max() if len(df[df['Trip'] > 0]) > 0 else 0
        
        # จัดกลุ่มตาม max_vehicle
        for max_veh in ['4W', 'JB', '6W']:
            veh_branches = unassigned_violations[unassigned_violations['_max_vehicle'] == max_veh]
            if len(veh_branches) == 0:
                continue
            
            # สร้างทริปใหม่สำหรับสาขาที่มี max_vehicle เดียวกัน
            new_trip = max_trip + 1
            
            # เช็คว่าเป็น Punthai หรือไม่
            is_punthai = False
            if 'BU' in veh_branches.columns and len(veh_branches) > 0:
                bu_val = str(veh_branches['BU'].iloc[0]).upper()
                is_punthai = bu_val in ['211', 'PUNTHAI']
            limits = PUNTHAI_LIMITS if is_punthai else LIMITS
            
            current_w = 0
            current_c = 0
            current_drops = 0
            max_w = limits[max_veh]['max_w']
            max_c = limits[max_veh]['max_c']
            max_d = limits[max_veh]['max_drops']
            
            for _, br in veh_branches.iterrows():
                br_w = br['Weight']
                br_c = br['Cube']
                
                if current_w + br_w > max_w or current_c + br_c > max_c or current_drops >= max_d:
                    # ปิดทริปปัจจุบัน เริ่มทริปใหม่
                    new_trip += 1
                    current_w = 0
                    current_c = 0
                    current_drops = 0
                
                df.loc[df['Code'] == br['Code'], 'Trip'] = new_trip
                current_w += br_w
                current_c += br_c
                current_drops += 1
            
            max_trip = new_trip
            safe_print(f"      ✅ จัด {len(veh_branches)} สาขา {max_veh} เสร็จ")
        
        # อัพเดต Truck และ VehicleCheck หลังจัดใหม่
        for trip_num in df[df['Trip'] > 0]['Trip'].unique():
            trip_codes = df[df['Trip'] == trip_num]['Code'].tolist()
            max_vehicles = [get_max_vehicle_for_branch(c) for c in trip_codes]
            vehicle_priority = {'4W': 1, 'JB': 2, '6W': 3}
            min_rank = min(vehicle_priority.get(v, 3) for v in max_vehicles)
            suggested = {1: '4W', 2: 'JB', 3: '6W'}.get(min_rank, '6W')
            df.loc[df['Trip'] == trip_num, 'Truck'] = f"{suggested} 📋 จัดใหม่"
        
        df['VehicleCheck'] = df.apply(check_vehicle_compliance, axis=1)
    
    # ==========================================
    # Step 9: เรียงทริปใหม่ตามภาค → จังหวัด → ระยะทาง
    # ==========================================
    safe_print("\n📋 Step 9: เรียงทริปใหม่ตามภาค → จังหวัด → ระยะทาง...")
    
    # หาระยะทางไกลสุดและ dominant province/region ของแต่ละทริป
    trip_max_distances = {}
    trip_sort9_keys = {}
    for trip_num in df[df['Trip'] > 0]['Trip'].unique():
        trip_data = df[df['Trip'] == trip_num]
        max_dist = trip_data['_distance_from_dc'].max() if '_distance_from_dc' in trip_data.columns else 0
        trip_max_distances[trip_num] = max_dist if pd.notna(max_dist) else 0
        # dominant province (most frequent in trip)
        _prov_col9 = '_province' if '_province' in trip_data.columns else ('Province' if 'Province' in trip_data.columns else None)
        _dom_prov9 = ''
        _rorder9 = 99
        if _prov_col9:
            _vc9 = trip_data[_prov_col9].value_counts()
            if len(_vc9):
                _dom_prov9 = _vc9.index[0]
                _rorder9 = REGION_ORDER.get(get_region_name(str(_dom_prov9)), 99)
        _dist_col9 = '_district' if '_district' in trip_data.columns else ('District' if 'District' in trip_data.columns else None)
        _dom_dist9 = ''
        if _dist_col9:
            _vcd9 = trip_data[_dist_col9].value_counts()
            if len(_vcd9):
                _dom_dist9 = _vcd9.index[0]
        trip_sort9_keys[trip_num] = (_rorder9, _dom_prov9, _dom_dist9, -(trip_max_distances[trip_num]))
    
    # เรียงทริปตาม ภาค → จังหวัด → อำเภอ → ระยะทาง(ไกลก่อน)
    sorted_trips = sorted(trip_max_distances.keys(), key=lambda x: trip_sort9_keys.get(x, (99, '๿', '๿', 0)))
    
    # สร้าง mapping ใหม่
    trip_renumber = {old_trip: new_trip for new_trip, old_trip in enumerate(sorted_trips, 1)}
    df['Trip'] = df['Trip'].map(lambda x: trip_renumber.get(x, 0) if x > 0 else 0)
    
    # อัพเดต summary_df ใหม่ทั้งหมดหลัง renumber (ให้ข้อมูลตรงกับ df)
    summary_data_new = []
    for trip_num in sorted(df[df['Trip'] > 0]['Trip'].unique()):
        trip_data = df[df['Trip'] == trip_num]
        total_w = trip_data['Weight'].sum()
        total_c = trip_data['Cube'].sum()
        trip_codes_list = trip_data['Code'].tolist()
        max_dist = trip_data['_distance_from_dc'].max() if '_distance_from_dc' in trip_data.columns else 0
        
        # หารถจากคอลัมน์ Truck ใน df
        truck = trip_data['Truck'].iloc[0] if 'Truck' in trip_data.columns and len(trip_data) > 0 else '6W'
        truck_str = str(truck).split()[0] if pd.notna(truck) else '6W'
        
        # หา BU type
        is_punthai = all(str(r.get('BU', '')).upper() in ['211', 'PUNTHAI'] for _, r in trip_data.iterrows())
        limits = PUNTHAI_LIMITS if is_punthai else LIMITS
        
        max_w = limits.get(truck_str, limits['6W'])['max_w']
        max_c = limits.get(truck_str, limits['6W'])['max_c']
        
        summary_data_new.append({
            'Trip': trip_num,
            'Branches': len(trip_codes_list),
            'Weight': total_w,
            'Cube': total_c,
            'Truck': truck,
            'BU_Type': 'punthai' if is_punthai else 'maxmart',
            'Buffer': f"🅿️ {int(punthai_buffer*100)}%" if is_punthai else f"🅼 {int(maxmart_buffer*100)}%",
            'Weight_Use%': (total_w / max_w) * 100,
            'Cube_Use%': (total_c / max_c) * 100,
            'Total_Distance': max_dist if pd.notna(max_dist) else 0
        })
    
    summary_df = pd.DataFrame(summary_data_new)
    summary_df = summary_df.sort_values('Trip').reset_index(drop=True)
    
    safe_print(f"   ✅ เรียงใหม่: {len(sorted_trips)} ทริป (Trip 1 = ไกลสุด {trip_max_distances[sorted_trips[0]]:.0f} km)")
    
    df = df.sort_values(['Trip', '_distance_from_dc'], ascending=[True, False]).reset_index(drop=True)
    
    # ลบคอลัมน์ชั่วคราว (เก็บ _province, _district, _subdistrict, _max_vehicle, _lat, _lon, _distance_from_dc ไว้สำหรับแผนที่)
    cols_to_drop = ['_region_code', '_region_name', '_prov_code', '_dist_code', '_subdist_code', '_route', '_group_key', '_region_order', '_prov_max_dist', '_dist_max_dist', '_region_allowed_vehicles', '_vehicle_priority']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
    
    return df, summary_df
def main():
    st.set_page_config(
        page_title="ระบบจัดเที่ยว",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # ป้องกันการโหลดโซนซ้ำ - ใช้ session state
    if 'zones_loaded' not in st.session_state:
        st.session_state.zones_loaded = False
    if 'cache_stats_shown' not in st.session_state:
        st.session_state.cache_stats_shown = False
    
    # แสดงสถานะ cache (ครั้งแรกเท่านั้น)
    if USE_CACHE and not st.session_state.cache_stats_shown:
        st.session_state.cache_stats_shown = True
        if len(DISTANCE_CACHE) > 0 or len(ROUTE_CACHE_DATA) > 0:
            st.success(f"✅ ใช้ข้อมูลแคช: {len(DISTANCE_CACHE)} ระยะทาง, {len(ROUTE_CACHE_DATA)} เส้นทาง")
    
    # 🔄 Auto-refresh (Optional - ไม่กระทบการใช้งานหลักถ้าไม่มี)
    # ใช้สำหรับ refresh cache ทุกเที่ยงคืน (เฉพาะ local dev)
    if AUTOREFRESH_AVAILABLE:
        try:
            now = datetime.now()
            # คำนวณเวลาถึงเที่ยงคืน (00:00:00)
            midnight = datetime.combine(now.date(), datetime_time(0, 0, 0))
            
            # ถ้ายังไม่ถึงเที่ยงคืน เอาเที่ยงคืนวันถัดไป
            if now < midnight:
                next_midnight = midnight
            else:
                next_midnight = midnight + timedelta(days=1)
            
            # คำนวณเวลาที่เหลือ (วินาที)
            seconds_until_midnight = int((next_midnight - now).total_seconds())
            
            # Refresh ทุกเที่ยงคืน (เฉพาะถ้ามี autorefresh)
            if seconds_until_midnight > 0:
                # เช็คในช่วง 5 นาทีก่อนเที่ยงคืน (หลัง 23:55)
                if seconds_until_midnight <= 300:  # 5 minutes
                    st.info(f"🔄 ระบบจะ Refresh อัตโนมัติใน {seconds_until_midnight // 60} นาที")
                    st_autorefresh(interval=seconds_until_midnight * 1000, key="midnight_refresh")
                else:
                    # ตรวจสอบทุก 1 ชั่วโมง
                    st_autorefresh(interval=3600000, limit=24, key="hourly_check")
        except Exception as e:
            # ถ้า autorefresh มีปัญหา → ไม่แสดง error (ฟีเจอร์เสริมเท่านั้น)
            pass
    
    # Header
    st.title("🚚 ระบบจัดเที่ยว - Route Optimizer")
    
    # 📊 Status Bar - แสดงสถานะระบบแบบกระชับ
    status_cols = st.columns([3, 1])
    with status_cols[0]:
        if SHEETS_AVAILABLE:
            st.success("📊 **Google Sheets:** เชื่อมต่อสำเร็จ | Auto-sync ทุก 5 นาที")
        else:
            st.warning("📊 **Data Source:** branch_data.json (local cache)")
    with status_cols[1]:
        st.metric("📍 Master Data", f"{len(MASTER_DATA):,} สาขา")
    
    st.markdown("---")
    
    # โหลดโมเดล
    model_data = load_model()
    
    if not model_data:
        st.error("❌ ไม่พบข้อมูลโมเดล กรุณาเทรนโมเดลก่อนใช้งาน")
        st.info("💡 รันคำสั่ง: `python test_model.py`")
        st.stop()
    
    # อัปโหลดไฟล์ครั้งเดียว
    st.markdown("### 📂 อัปโหลดไฟล์รายการออเดอร์")
    uploaded_file = st.file_uploader(
        "เลือกไฟล์ Excel (.xlsx)", 
        type=['xlsx'],
        help="อัปโหลดไฟล์ Excel ที่มีรายการสาขาและออเดอร์"
    )
    
    if uploaded_file:
        # เก็บไฟล์ต้นฉบับไว้ใน session_state เพื่อใช้ตอน export
        uploaded_file_content = uploaded_file.read()
        st.session_state['original_file_content'] = uploaded_file_content
        
        with st.spinner("⏳ กำลังอ่านข้อมูล..."):
            df = load_excel(uploaded_file_content)
            df = process_dataframe(df)
            
            if df is not None and 'Code' in df.columns:
                total_rows = len(df)
                unique_codes = df['Code'].nunique()
                duplicate_count = total_rows - unique_codes
                
                st.success(f"✅ อ่านข้อมูลสำเร็จ: **{total_rows:,}** รายการ")
                
                # ⚠️ แจ้งเตือนถ้ามี duplicate
                if duplicate_count > 0:
                    st.warning(f"⚠️ พบ **{duplicate_count}** รายการซ้ำ (Code ซ้ำกัน) - จะรวมยอดให้อัตโนมัติ")
                    with st.expander("🔍 ดู Code ที่ซ้ำ"):
                        dup_codes = df[df.duplicated(subset=['Code'], keep=False)].groupby('Code').size().reset_index(name='จำนวนซ้ำ')
                        st.dataframe(dup_codes[dup_codes['จำนวนซ้ำ'] > 1], width="stretch")
                    
                    # รวมยอด duplicate codes
                    agg_cols = {'Weight': 'sum', 'Cube': 'sum'}
                    # เก็บ column อื่นๆ ไว้ (ใช้ค่าแรก)
                    for col in df.columns:
                        if col not in ['Code', 'Weight', 'Cube']:
                            agg_cols[col] = 'first'
                    df = df.groupby('Code', as_index=False).agg(agg_cols)
                    st.info(f"📊 หลังรวมยอดซ้ำ: **{len(df):,}** สาขา")
                
                # แสดงข้อมูลพื้นฐาน
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📍 จำนวนสาขา", f"{len(df):,}")
                with col2:
                    st.metric("⚖️ น้ำหนักรวม", f"{df['Weight'].sum():,.0f} kg")
                with col3:
                    st.metric("📦 คิวรวม", f"{df['Cube'].sum():.1f} m³")
                with col4:
                    provinces = df['Province'].nunique() if 'Province' in df.columns else 0
                    st.metric("🗺️ จังหวัด", f"{provinces}")
                
                # แสดงตัวอย่างข้อมูล
                with st.expander("🔍 ดูข้อมูลตัวอย่าง"):
                    st.dataframe(df.head(10), width="stretch")
                
                # ==========================================
                # เติมข้อมูลพื้นที่จาก Master (vectorized - เร็วกว่า iterrows)
                # ==========================================
                if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
                    _m = MASTER_DATA[['Plan Code', 'จังหวัด', 'อำเภอ', 'ตำบล']].copy()
                    _m['_code'] = _m['Plan Code'].astype(str).str.strip().str.upper()
                    # ใช้ชื่อ column พิเศษเพื่อหลีกเลี่ยง collision กับ column ที่มีอยู่ใน df
                    _m = _m.rename(columns={'จังหวัด': '_m_prov', 'อำเภอ': '_m_dist', 'ตำบล': '_m_subdist'})
                    df['_code'] = df['Code'].astype(str).str.strip().str.upper()
                    df = df.merge(_m[['_code', '_m_prov', '_m_dist', '_m_subdist']].drop_duplicates('_code'),
                                  on='_code', how='left')
                    
                    # เติม Province ถ้าว่าง
                    need_prov = df['Province'].isna() | (df['Province'] == '') | (df['Province'] == 'UNKNOWN') if 'Province' in df.columns else pd.Series([True]*len(df))
                    filled_count = int(need_prov.sum())
                    if 'Province' not in df.columns:
                        df['Province'] = df['_m_prov']
                    else:
                        df.loc[need_prov, 'Province'] = df.loc[need_prov, '_m_prov']
                    
                    # เติม District/Subdistrict ถ้าว่าง
                    for col_upload, col_master in [('District', '_m_dist'), ('Subdistrict', '_m_subdist')]:
                        if col_upload not in df.columns:
                            df[col_upload] = df[col_master].fillna('')
                        else:
                            need = df[col_upload].isna() | (df[col_upload] == '')
                            df.loc[need, col_upload] = df.loc[need, col_master]
                    
                    df = df.drop(columns=['_code', '_m_prov', '_m_dist', '_m_subdist'], errors='ignore')
                    
                    if filled_count > 0:
                        st.info(f"📍 เติมข้อมูลพื้นที่จาก Master แล้ว {filled_count} รายการ")
                
                # ตรวจสอบว่ายังมีข้อมูลที่ขาดหรือไม่ (แสดงรายละเอียด)
                if 'Province' in df.columns:
                    missing_df = df[(df['Province'].isna()) | (df['Province'] == '') | (df['Province'] == 'UNKNOWN')]
                    if len(missing_df) > 0:
                        st.warning(f"⚠️ ยังมี {len(missing_df)} สาขาที่ไม่พบข้อมูลพื้นที่ใน Master")
                        with st.expander("📋 ดูรายละเอียดสาขาที่ขาดข้อมูล"):
                            _show_cols = [c for c in ['Code', 'Name', 'Province', 'District'] if c in missing_df.columns]
                            st.dataframe(missing_df[_show_cols].reset_index(drop=True), hide_index=True)
                
                st.markdown("---")
                
                # แท็บหลัก
                tab1, tab2 = st.tabs([
                    "📦 จัดเที่ยว (ตามน้ำหนัก)", 
                    "🗺️ จัดกลุ่มตามภาค"
                ])
                    
                # ==========================================
                # แท็บ 1: จัดเที่ยว (ตามน้ำหนัก)
                # ==========================================
                with tab1:
                    # เพิ่ม Region ถ้ายังไม่มี
                    if 'Region' not in df.columns and 'Province' in df.columns:
                        df['Region'] = df['Province'].apply(get_region_name)
                    
                    # ==========================================
                    # ตัวเลือกการตั้งค่า
                    # ==========================================
                    st.markdown("#### ⚙️ ตั้งค่าการจัดทริป")
                    
                    # กรอก Buffer แยกตามประเภท
                    col_buf1, col_buf2 = st.columns(2)
                    
                    with col_buf1:
                        punthai_buffer = st.number_input(
                            "🅿️ Punthai Buffer %",
                            min_value=80,
                            max_value=120,
                            value=100,
                            step=5
                        )
                    
                    with col_buf2:
                        maxmart_buffer = st.number_input(
                            "🅼 Maxmart/ผสม Buffer %",
                            min_value=80,
                            max_value=150,
                            value=110,
                            step=5
                        )
                    
                    # แปลงเป็น buffer value
                    punthai_buffer_value = punthai_buffer / 100.0
                    maxmart_buffer_value = maxmart_buffer / 100.0
                    
                    st.markdown("---")
                    st.markdown("### 📋 สรุปข้อจำกัดรถจาก Master Data")
                    
                    # บึงแคช: vehicle_restrictions (เร็ว - ไม่ใช้ iterrows)
                    vehicle_restrictions = {code: get_max_vehicle_for_branch(code) for code in df['Code']}
                    unmatched_codes = []
                    
                    if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
                        master_codes_set = set(MASTER_DATA['Plan Code'].str.strip().str.upper())
                        for code in df['Code']:
                            code_clean = str(code).strip().upper()
                            if code_clean not in master_codes_set:
                                found = any(code_clean in mc or mc in code_clean for mc in master_codes_set)
                                if not found:
                                    unmatched_codes.append(code_clean)
                    
                    restriction_counts = pd.Series(vehicle_restrictions).value_counts()
                    total_branches = len(df)
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📍 สาขาทั้งหมด", f"{total_branches}")
                    with col2:
                        four_w_count = restriction_counts.get('4W', 0)
                        st.metric("🚗 จำกัด 4W", f"{four_w_count}", 
                                 delta=f"{(four_w_count/total_branches*100):.1f}%" if total_branches > 0 else "0%")
                    with col3:
                        jb_count = restriction_counts.get('JB', 0)
                        st.metric("🚚 จำกัด JB", f"{jb_count}",
                                 delta=f"{(jb_count/total_branches*100):.1f}%" if total_branches > 0 else "0%")
                    with col4:
                        six_w_count = restriction_counts.get('6W', 0)
                        st.metric("🚛 ใช้ 6W ได้", f"{six_w_count}",
                                 delta=f"{(six_w_count/total_branches*100):.1f}%" if total_branches > 0 else "0%")
                    
                    # ⚠️ แสดงสาขาที่ไม่พบใน Master Data
                    if unmatched_codes:
                        st.warning(f"⚠️ มี {len(unmatched_codes)} สาขาที่ไม่พบใน Master Data (ใช้ 6W เป็น default)")
                        with st.expander(f"🔍 ดูรายละเอียดสาขาที่ไม่พบ ({len(unmatched_codes)} สาขา)"):
                            # แสดง 10 ตัวอย่างแรก
                            sample_codes = unmatched_codes[:20]
                            unmatched_df = df[df['Code'].isin(sample_codes)][['Code', 'Name']].copy()
                            unmatched_df.columns = ['รหัสสาขา (ไฟล์ Upload)', 'ชื่อสาขา']
                            st.dataframe(unmatched_df, width="stretch")
                            
                            if len(unmatched_codes) > 20:
                                st.caption(f"... และอีก {len(unmatched_codes) - 20} สาขา")
                    
                    # แสดงรายละเอียดสาขาที่มีข้อจำกัด
                    if four_w_count > 0 or jb_count > 0:
                        with st.expander(f"🔍 ดูรายละเอียดสาขาที่มีข้อจำกัด ({four_w_count + jb_count} สาขา)"):
                            restricted_branches = df[df['Code'].isin([k for k, v in vehicle_restrictions.items() if v in ['4W', 'JB']])].copy()
                            restricted_branches['MaxVehicle'] = restricted_branches['Code'].map(vehicle_restrictions)
                            display_restricted = restricted_branches[['Code', 'Name', 'MaxVehicle']].copy()
                            display_restricted.columns = ['รหัสสาขา', 'ชื่อสาขา', 'รถสูงสุด']
                            st.dataframe(display_restricted.sort_values('รถสูงสุด'), width="stretch", height=300)
                    
                    st.markdown("---")
                    
                    # ปุ่มจัดทริป
                    if st.button("🚀 เริ่มจัดเที่ยว", type="primary", width="stretch"):
                        # สร้าง status container แบบ popup
                        with st.status("🚀 กำลังประมวลผล...", expanded=True) as status:
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.write("⏳ กำลังเตรียมข้อมูล...")
                            progress_bar.progress(10)
                            
                            # จัดเรียงตามภาค/จังหวัด/อำเภอ/ตำบล/Route (ในฟังก์ชัน predict_trips)
                            df_to_process = df.copy()
                            
                            progress_bar.progress(20)
                            
                            import time as time_module
                            start_time = time_module.time()
                            
                            # ส่ง buffer แยกตาม BU
                            result_df, summary = predict_trips(
                                df_to_process, 
                                model_data, 
                                punthai_buffer=punthai_buffer_value,
                                maxmart_buffer=maxmart_buffer_value
                            )
                            
                            elapsed_time = time_module.time() - start_time
                            progress_bar.progress(90)
                            
                            # 💾 เก็บผลลัพธ์ใน session_state เพื่อใช้ตอน export
                            st.session_state['trip_result'] = result_df
                            st.session_state['trip_summary'] = summary
                            st.session_state['trip_buffers'] = {
                                'punthai': punthai_buffer_value,
                                'maxmart': maxmart_buffer_value
                            }
                            st.session_state['_trip_result_fresh'] = True  # แสดง balloons ครั้งแรกเท่านั้น
                            st.session_state['_trip_elapsed'] = elapsed_time  # เก็บเวลาจัดทริป
                            
                            progress_bar.progress(100)
                            status_text.write(f"✅ จัดทริปเสร็จสิ้น! (ใช้เวลา {elapsed_time:.1f} วินาที)")
                            status.update(label=f"✅ ประมวลผลเสร็จสมบูรณ์! ({elapsed_time:.1f}s)", state="complete", expanded=False)
                    
                    # 📊 แสดงผลลัพธ์ถ้ามีข้อมูลใน session_state
                    if 'trip_result' in st.session_state and 'trip_summary' in st.session_state:
                        result_df = st.session_state['trip_result']
                        summary = st.session_state['trip_summary']
                        
                        # ตรวจสอบสาขาที่ไม่ได้จัดทริป (Trip = 0)
                        unassigned_count = len(result_df[result_df['Trip'] == 0])
                        if unassigned_count > 0:
                            st.warning(f"⚠️ มี {unassigned_count} สาขาที่ไม่ได้จัดทริป (Trip = 0)")
                        
                        # กรองเฉพาะสาขาที่จัดทริปแล้ว สำหรับการแสดงผล
                        assigned_df = result_df[result_df['Trip'] > 0].copy()
                        
                        # แสดง balloons เฉพาะครั้งแรกที่ผลลัพธ์ใหม่ (ไม่เป็นทุก rerender)
                        if st.session_state.get('_trip_result_fresh', False):
                            st.balloons()
                            st.session_state['_trip_result_fresh'] = False
                        st.success(f"✅ **จัดทริปเสร็จสมบูรณ์!** รวม **{len(summary)}** ทริป ({len(assigned_df)} สาขา)")
                        
                        st.markdown("---")
                        
                        # สถิติโดยรวม
                        st.markdown("### 📊 สรุปผลการจัดทริป")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("🚚 จำนวนทริป", len(summary))
                        with col2:
                            st.metric("📍 จำนวนสาขา", len(assigned_df))
                        with col3:
                            avg_branches = len(assigned_df) / max(1, assigned_df['Trip'].nunique())
                            st.metric("📊 เฉลี่ยสาขา/ทริป", f"{avg_branches:.1f}")
                        with col4:
                            avg_util = summary['Cube_Use%'].mean() if len(summary) > 0 else 0
                            st.metric("📈 การใช้รถเฉลี่ย", f"{avg_util:.0f}%")
                        
                        # ⏱️ แสดง timing dashboard
                        _trip_elapsed = st.session_state.get('_trip_elapsed', 0)
                        _map_elapsed  = st.session_state.get('_imap_build_time', None)
                        if _trip_elapsed or _map_elapsed is not None:
                            with st.expander("⏱️ เวลาประมวลผล", expanded=False):
                                _tc1, _tc2, _tc3 = st.columns(3)
                                if _trip_elapsed:
                                    _tc1.metric("🔄 จัดทริป", f"{_trip_elapsed:.1f}s")
                                if _map_elapsed is not None:
                                    _tc2.metric("🗺️ สร้างแผนที่", f"{_map_elapsed:.1f}s")
                                _tc3.metric("💾 cache", f"{len(st.session_state.get('_imap_key',''))*0:.0f}+{len(summary)} trips")

                        st.markdown("### 🚛 รายละเอียดแต่ละทริป")
                        
                        # ตรวจสอบว่า summary มีคอลัมน์ที่ต้องการหรือไม่
                        format_dict = {}
                        gradient_cols = []
                        
                        if 'Weight' in summary.columns:
                            format_dict['Weight'] = '{:.2f}'
                        if 'Cube' in summary.columns:
                            format_dict['Cube'] = '{:.2f}'
                        if 'Weight_Use%' in summary.columns:
                            format_dict['Weight_Use%'] = '{:.1f}%'
                            gradient_cols.append('Weight_Use%')
                        if 'Cube_Use%' in summary.columns:
                            format_dict['Cube_Use%'] = '{:.1f}%'
                            gradient_cols.append('Cube_Use%')
                        if 'Total_Distance' in summary.columns:
                            format_dict['Total_Distance'] = '{:.1f} km'
                        
                        # สร้าง styled dataframe
                        if format_dict:
                            styled_df = summary.style.format(format_dict)
                            if gradient_cols:
                                styled_df = styled_df.background_gradient(
                                    subset=gradient_cols,
                                    cmap='RdYlGn',
                                    vmin=0,
                                    vmax=100
                                )
                            st.dataframe(styled_df, width="stretch", height=400)
                        else:
                            st.dataframe(summary, width="stretch", height=400)

                        with st.expander("📋 ดูรายละเอียดรายสาขา (เรียงตามน้ำหนัก)"):
                            # จัดเรียงคอลัมน์ที่สำคัญ
                            display_cols = ['Trip', 'Code', 'Name']
                            if 'Province' in result_df.columns:
                                display_cols.append('Province')
                            if 'Region' in result_df.columns:
                                display_cols.append('Region')
                            display_cols.extend(['Max_Distance_in_Trip', 'Weight', 'Cube', 'Truck', 'VehicleCheck'])
                            
                            # กรองคอลัมน์ที่มีอยู่จริง
                            display_cols = [col for col in display_cols if col in result_df.columns]
                            display_df = result_df[display_cols].copy()
                            
                            # ตั้งชื่อคอลัมน์ภาษาไทย
                            col_names = {'Trip': 'ทริป', 'Code': 'รหัส', 'Name': 'ชื่อสาขา', 'Province': 'จังหวัด', 
                                       'Region': 'ภาค', 'Max_Distance_in_Trip': 'ระยะทาง Max(km)', 
                                       'Weight': 'น้ำหนัก(kg)', 'Cube': 'คิว(m³)', 'Truck': 'รถ', 'VehicleCheck': 'ตรวจสอบรถ'}
                            display_df.columns = [col_names.get(c, c) for c in display_cols]
                            
                            # จัดรูปแบบคอลัมน์ระยะทาง
                            st.dataframe(
                                display_df.style.format({
                                    'ระยะทาง Max(km)': '{:.1f}',
                                    'น้ำหนัก(kg)': '{:.2f}',
                                    'คิว(m³)': '{:.2f}'
                                }),
                                width="stretch", 
                                height=400
                            )
                        
                        # แสดงสาขาที่มีคำเตือน - รวมทั้ง ⚠️ และ ❌
                        warning_branches = result_df[result_df['VehicleCheck'].str.contains('⚠️|❌', na=False, regex=True)]
                        if len(warning_branches) > 0:
                            # นับจำนวนแต่ละประเภท
                            error_count = len(result_df[result_df['VehicleCheck'].str.contains('❌', na=False)])
                            warning_count = len(result_df[result_df['VehicleCheck'].str.contains('⚠️', na=False)])
                            
                            with st.expander(f"🚨 สาขาที่มีปัญหา ({len(warning_branches)} สาขา: ❌ {error_count} ข้อจำกัด, ⚠️ {warning_count} อื่นๆ)", expanded=(error_count > 0)):
                                if error_count > 0:
                                    st.error(f"❌ มี {error_count} สาขาที่ใช้รถเกินข้อจำกัดจาก Master Data!")
                                if warning_count > 0:
                                    st.warning(f"⚠️ มี {warning_count} สาขาที่มีคำเตือนอื่นๆ")
                                
                                display_cols_warn = ['Trip', 'Code', 'Name', 'MaxVehicle', 'Truck', 'VehicleCheck']
                                display_warn_df = warning_branches[display_cols_warn].copy()
                                display_warn_df.columns = ['ทริป', 'รหัส', 'ชื่อสาขา', 'รถ Max', 'รถที่จัด', 'สถานะ']
                                st.dataframe(display_warn_df, width="stretch")
                        
                        # ── 📥 Excel build (cached) — สร้างครั้งเดียว ไม่ rebuild ถ้า result ไม่เปลี่ยน ──
                        import hashlib as _hl_xl
                        _xl_sig = f"v6|{len(result_df)}|{int(result_df['Trip'].max())}|{sorted(result_df['Trip'].unique().tolist())}"
                        _xl_key = _hl_xl.md5(_xl_sig.encode()).hexdigest()[:12]

                        if st.session_state.get('_excel_key') != _xl_key:
                            with st.spinner("📊 กำลังสร้างไฟล์ Excel..."):
                                import xlsxwriter as _xlw

                                # ── 1. location_map → vectorized dict map ──
                                _loc_sp = {}; _loc_sd = {}; _loc_sv = {}; _loc_rt = {}
                                if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
                                    _lm_cols = [c for c in ['Plan Code','ตำบล','อำเภอ','จังหวัด','Reference'] if c in MASTER_DATA.columns]
                                    _lm = MASTER_DATA[_lm_cols].copy()
                                    _lm['_k'] = _lm['Plan Code'].astype(str).str.strip().str.upper()
                                    for _r in _lm.to_dict('records'):
                                        _k = _r['_k']
                                        if _k:
                                            _loc_sp[_k] = str(_r.get('ตำบล', '') or '')
                                            _loc_sd[_k] = str(_r.get('อำเภอ', '') or '')
                                            _loc_sv[_k] = str(_r.get('จังหวัด', '') or '')
                                            _loc_rt[_k] = str(_r.get('Reference', '') or '')

                                # ── 2. Pre-join ทั้งหมดแบบ vectorized ──
                                _rd = result_df[result_df['Trip'] != 0].copy()
                                _rd['_key_u'] = _rd['Code'].astype(str).str.strip().str.upper()
                                _rd['_sp'] = _rd['_key_u'].map(_loc_sp).fillna('')
                                _rd['_sd'] = _rd['_key_u'].map(_loc_sd).fillna('')
                                _rd['_sv'] = _rd['_key_u'].map(_loc_sv).fillna('')
                                _rd['_rt'] = _rd['_key_u'].map(_loc_rt).fillna('')

                                # fallback → ใช้ planning columns ถ้า MASTER_DATA ว่าง
                                _prov_col  = '_province'    if '_province'    in _rd.columns else ('Province'    if 'Province'    in _rd.columns else None)
                                _dist_col2 = '_district'    if '_district'    in _rd.columns else ('District'    if 'District'    in _rd.columns else None)
                                _subd_col2 = '_subdistrict' if '_subdistrict' in _rd.columns else ('Subdistrict' if 'Subdistrict' in _rd.columns else None)
                                _empty = ''
                                _rd['_sv_eff'] = _rd['_sv'].where(_rd['_sv'].str.strip() != '', _rd[_prov_col]  if _prov_col  else _empty)
                                _rd['_sd_eff'] = _rd['_sd'].where(_rd['_sd'].str.strip() != '', _rd[_dist_col2] if _dist_col2 else _empty)
                                _rd['_sp_eff'] = _rd['_sp'].where(_rd['_sp'].str.strip() != '', _rd[_subd_col2] if _subd_col2 else _empty)

                                # ── 3. pre-build province→rorder dict ครั้งเดียว ──
                                _prov_rorder: dict = {}
                                for _p in _rd['_sv_eff'].unique():
                                    _prov_rorder[_p] = REGION_ORDER.get(get_region_name(str(_p)), 99)
                                _rd['_rorder'] = _rd['_sv_eff'].map(_prov_rorder).fillna(99).astype(int)

                                # ── 4. distance from result_df directly (ไม่ต้อง MASTER_DATA loop) ──
                                _dist_src_col = '_distance_from_dc' if '_distance_from_dc' in _rd.columns else None

                                # ── 5. sort keys ──
                                trip_no_map = {}
                                vehicle_counts = {'4W': 0, '4WJ': 0, '6W': 0}
                                trip_sort_keys = {}
                                for _tnum, _tg in _rd.groupby('Trip', sort=False):
                                    if _tnum == 0: continue
                                    _rorder = int(_tg['_rorder'].mode().iloc[0]) if len(_tg) else 99
                                    if _dist_src_col:
                                        _pmx = float(_tg[_dist_src_col].max() or 0)
                                    else:
                                        _pmx = 0.0
                                    _pvc = _tg['_sv_eff'].value_counts()
                                    _dvc = _tg['_sd_eff'].value_counts()
                                    trip_sort_keys[_tnum] = (
                                        _rorder,
                                        _pvc.index[0] if len(_pvc) else '',
                                        _dvc.index[0] if len(_dvc) else '',
                                        -_pmx
                                    )

                                sorted_trips = sorted(
                                    [t for t in result_df['Trip'].unique() if t != 0],
                                    key=lambda t: trip_sort_keys.get(t, (99, '๿', '๿', 0))
                                )

                                for _tnum in sorted_trips:
                                    _ts = summary[summary['Trip'] == _tnum]
                                    if len(_ts) > 0:
                                        _vi = _ts.iloc[0]['Truck']
                                        _vt = _vi.split()[0] if _vi else '6W'
                                        if _vt == 'JB': _vt = '4WJ'
                                        vehicle_counts[_vt] = vehicle_counts.get(_vt, 0) + 1
                                        trip_no_map[_tnum] = f"{_vt}{vehicle_counts[_vt]:03d}"

                                # ── 6. sort rows ──
                                _trip_order_map = {t: i for i, t in enumerate(sorted_trips)}
                                _rd['_trip_order'] = _rd['Trip'].map(_trip_order_map)
                                _rd = _rd.sort_values(['_trip_order', '_sv_eff', '_sd_eff', '_sp_eff', 'Code'])

                                # ── 7. pre-group rows ──
                                _trip_rows: dict = {}
                                for _rec in _rd.to_dict('records'):
                                    _trip_rows.setdefault(int(_rec['Trip']), []).append(_rec)

                                # ── 8. failed_trips ──
                                failed_trips = set()
                                for _t in sorted_trips:
                                    _rows_t = _trip_rows.get(_t, [])
                                    if not _rows_t: continue
                                    _is_pt = all(str(_r.get('BU', '')).upper() in ('211', 'PUNTHAI') for _r in _rows_t)
                                    _tno = trip_no_map.get(_t, '6W001')
                                    _vt2 = 'JB' if _tno.startswith('4WJ') else ('4W' if _tno.startswith('4W') else '6W')
                                    _lim = (PUNTHAI_LIMITS if _is_pt else LIMITS).get(_vt2, LIMITS['6W'])
                                    _tw  = sum(float(_r.get('Weight', 0) or 0) for _r in _rows_t)
                                    _tc  = sum(float(_r.get('Cube',   0) or 0) for _r in _rows_t)
                                    if (_tw / _lim['max_w'] * 100) < 90 and (_tc / _lim['max_c'] * 100) < 90:
                                        failed_trips.add(_t)

                                # ── 9. xlsxwriter write ──
                                _output = io.BytesIO()
                                try:
                                    _wb_xl = _xlw.Workbook(_output, {'in_memory': True, 'constant_memory': True})
                                    _ws_xl = _wb_xl.add_worksheet('2.Punthai')

                                    _hdr_fmt = _wb_xl.add_format({'bold':True,'border':1,'bg_color':'#D9D9D9','align':'center'})
                                    _yfmt    = _wb_xl.add_format({'bg_color':'#FFE699','border':1})
                                    _wfmt    = _wb_xl.add_format({'bg_color':'#FFFFFF','border':1})
                                    _yfmt_r  = _wb_xl.add_format({'bg_color':'#FFE699','border':1,'font_color':'#FF0000','bold':True})
                                    _wfmt_r  = _wb_xl.add_format({'bg_color':'#FFFFFF','border':1,'font_color':'#FF0000','bold':True})
                                    _ynfmt   = _wb_xl.add_format({'bg_color':'#FFE699','border':1,'num_format':'#,##0.00'})
                                    _wnfmt   = _wb_xl.add_format({'bg_color':'#FFFFFF','border':1,'num_format':'#,##0.00'})
                                    _ynfmt_r = _wb_xl.add_format({'bg_color':'#FFE699','border':1,'num_format':'#,##0.00','font_color':'#FF0000','bold':True})
                                    _wnfmt_r = _wb_xl.add_format({'bg_color':'#FFFFFF','border':1,'num_format':'#,##0.00','font_color':'#FF0000','bold':True})

                                    _hdrs = ['Sep.','BU','รหัสสาขา','รหัส WMS','สาขา','ตำบล','อำเภอ','จังหวัด','Route',
                                             'Total Cube','Total Wgt','Original QTY','Trip','Trip no']
                                    _ws_xl.write_row(0, 0, _hdrs, _hdr_fmt)
                                    _ws_xl.set_row(0, 18)
                                    for _ci_w, _cw in enumerate([6,6,12,12,30,14,14,16,12,11,11,12,6,10]):
                                        _ws_xl.set_column(_ci_w, _ci_w, _cw)

                                    use_yellow = True
                                    _row_xl = 1
                                    sep_num = 1

                                    for _tnum in sorted_trips:
                                        _rows = _trip_rows.get(_tnum, [])
                                        _tno  = trip_no_map.get(_tnum, '')
                                        _is_f = _tnum in failed_trips
                                        _tf = (_yfmt_r if _is_f else _yfmt) if use_yellow else (_wfmt_r if _is_f else _wfmt)
                                        _nf = (_ynfmt_r if _is_f else _ynfmt) if use_yellow else (_wnfmt_r if _is_f else _wnfmt)
                                        use_yellow = not use_yellow
                                        _tnum_int = int(_tnum)
                                        _tno_str  = str(_tno)
                                        for _rec in _rows:
                                            _bc = str(_rec.get('Code', ''))
                                            _ws_xl.write_row(_row_xl, 0, [
                                                sep_num,
                                                _rec.get('BU', 211),
                                                _bc, _bc,
                                                str(_rec.get('Name', '')),
                                                str(_rec.get('_sp_eff', '') or _rec.get('_sp', '')),
                                                str(_rec.get('_sd_eff', '') or _rec.get('_sd', '')),
                                                str(_rec.get('_sv_eff', '') or _rec.get('_sv', '')),
                                                str(_rec.get('_rt', '')),
                                            ], _tf)
                                            _ws_xl.write(_row_xl,  9, round(float(_rec.get('Cube',        0) or 0), 2), _nf)
                                            _ws_xl.write(_row_xl, 10, round(float(_rec.get('Weight',      0) or 0), 2), _nf)
                                            _ws_xl.write(_row_xl, 11, int(  float(_rec.get('OriginalQty', 0) or 0)), _tf)
                                            _ws_xl.write(_row_xl, 12, _tnum_int, _tf)
                                            _ws_xl.write(_row_xl, 13, _tno_str,  _tf)
                                            _row_xl += 1
                                            sep_num  += 1

                                    _wb_xl.close()
                                    _output.seek(0)

                                except Exception as _xe:
                                    st.warning(f"⚠️ xlsxwriter error: {_xe} — fallback to basic")
                                    _output = io.BytesIO()
                                    with pd.ExcelWriter(_output, engine='xlsxwriter') as _writer:
                                        _exp = _rd.drop(columns=[c for c in ['_key_u','_sp','_sd','_sv','_rt','_trip_order','_sp_eff','_sd_eff','_sv_eff','_rorder'] if c in _rd.columns], errors='ignore').copy()
                                        _exp['Trip_No'] = _exp['Trip'].map(lambda x: trip_no_map.get(x, ''))
                                        _exp.to_excel(_writer, sheet_name='รายละเอียดทริป', index=False)
                                        summary.to_excel(_writer, sheet_name='สรุปทริป', index=False)

                                st.session_state['_excel_bytes']  = _output.getvalue()
                                st.session_state['_excel_key']    = _xl_key
                                st.session_state['_trip_no_map']  = trip_no_map

                        # trip_no_map ต้องพร้อมสำหรับแผนที่ด้านล่าง
                        trip_no_map = st.session_state.get('_trip_no_map', {})

                        st.download_button(
                            label="📥 ดาวน์โหลดผลลัพธ์ (Excel)",
                            data=st.session_state.get('_excel_bytes', b''),
                            file_name=f"ผลจัดทริป_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            type="primary",
                            width="stretch"
                        )
                        
                        st.markdown("---")
                        
                        # 🗺️ แผนที่เส้นทาง (Interactive - Leaflet.js)
                        with st.expander("🗺️ แผนที่เส้นทาง (Interactive)", expanded=True):
                            try:
                                from trip_map_interactive import build_interactive_map_html as _build_imap
                                import streamlit.components.v1 as _cmp2
                                import hashlib as _hl

                                _imap_sig = f"v10|{len(assigned_df)}|{int(assigned_df['Trip'].max())}|{sorted(assigned_df['Trip'].unique().tolist())}"
                                _imap_key = _hl.md5(_imap_sig.encode()).hexdigest()[:12]

                                if st.session_state.get('_imap_key') != _imap_key:
                                    with st.spinner("🗺️ กำลังสร้างแผนที่..."):
                                        _t_map = time_module.time()
                                        st.session_state['_imap_html'] = _build_imap(
                                            result_df=assigned_df,
                                            summary_df=summary,
                                            limits=LIMITS,
                                            punthai_limits=PUNTHAI_LIMITS,
                                            trip_no_map=trip_no_map,
                                            dc_lat=14.1459, dc_lon=100.6873,
                                        )
                                        st.session_state['_imap_key'] = _imap_key
                                        st.session_state['_imap_build_time'] = time_module.time() - _t_map

                                _cmp2.html(st.session_state['_imap_html'], height=780, scrolling=False)
                            except Exception as _e:
                                import traceback as _tb
                                st.error(f"❌ Interactive map error: {_e}")
                                st.code(_tb.format_exc(), language='text')
                                st.info(f"📋 assigned_df columns: {list(assigned_df.columns)}\n\nrows: {len(assigned_df)}, trips: {sorted(assigned_df['Trip'].unique().tolist())}")
                            else:
                                _FOLIUM_FALLBACK_ = False

                        # ── FOLIUM FALLBACK (ใช้เมื่อ interactive map error) ──
                        if 'FOLIUM_AVAILABLE' in dir() and FOLIUM_AVAILABLE and locals().get('_FOLIUM_FALLBACK_', False):
                            with st.expander("🗺️ แผนที่เส้นทางแต่ละทริป (Fallback)", expanded=True):
                                # ตัวกรอง
                                col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 1])
                                
                                with col_filter1:
                                    # กรองตามเลขทริป - เรียงจากไกลมาใกล้
                                    trip_distances = {}
                                    for t in assigned_df['Trip'].unique():
                                        if t > 0 and '_distance_from_dc' in assigned_df.columns:
                                            max_dist = assigned_df[assigned_df['Trip'] == t]['_distance_from_dc'].max()
                                            trip_distances[t] = max_dist if pd.notna(max_dist) else 0
                                    sorted_trips = sorted(trip_distances.keys(), key=lambda x: trip_distances.get(x, 0), reverse=True)
                                    trip_options = ['ทั้งหมด'] + [f"Trip {t} ({trip_distances.get(t, 0):.0f}km)" for t in sorted_trips]
                                    selected_trip = st.selectbox("🚚 เลือกทริป (ไกล→ใกล้)", trip_options, key="map_trip_filter")
                                
                                with col_filter2:
                                    # กรองตามประเภทรถ
                                    truck_types = ['ทั้งหมด']
                                    if 'Truck' in assigned_df.columns:
                                        unique_trucks = assigned_df['Truck'].dropna().unique()
                                        truck_types.extend(sorted(set([t.split()[0] for t in unique_trucks if t])))
                                    selected_truck = st.selectbox("🚛 ประเภทรถ", truck_types, key="map_truck_filter")
                                
                                with col_filter3:
                                    # เลือกแสดงเส้นทาง
                                    show_route = st.checkbox("🛣️ แสดงเส้นทาง", value=True, key="map_show_route")
                                
                                # กรองข้อมูล
                                map_df = assigned_df.copy()
                                if selected_trip != 'ทั้งหมด':
                                    trip_num = int(selected_trip.split()[1])
                                    map_df = map_df[map_df['Trip'] == trip_num]
                                if selected_truck != 'ทั้งหมด':
                                    map_df = map_df[map_df['Truck'].str.startswith(selected_truck, na=False)]
                                
                                if len(map_df) == 0:
                                    st.warning("⚠️ ไม่มีข้อมูลตามเงื่อนไขที่เลือก")
                                else:
                                    # ตรวจสอบว่ามีพิกัด
                                    if '_lat' in map_df.columns and '_lon' in map_df.columns:
                                        valid_coords = map_df[(map_df['_lat'] > 0) & (map_df['_lon'] > 0)]
                                        
                                        if len(valid_coords) == 0:
                                            st.warning("⚠️ ไม่มีข้อมูลพิกัดสำหรับแสดงแผนที่")
                                        else:
                                            # สร้างแผนที่พร้อม progress
                                            # Map cache - ตรวจสอบว่ามีแผนที่ cached หรือไม่
                                            map_cache_key = f'map|{selected_trip}|{selected_truck}|{show_route}|{len(valid_coords)}'
                                            _map_is_cached = (st.session_state.get('_map_cache_key') == map_cache_key and '_map_html' in st.session_state)
                                            if not _map_is_cached:
                                                with st.spinner("🗺️ กำลังสร้างแผนที่..."):
                                                    # DC Wang Noi coordinates
                                                    DC_LAT, DC_LON = 14.1459, 100.6873
                                                
                                                    # หาจุดกึ่งกลาง
                                                    center_lat = valid_coords['_lat'].mean()
                                                    center_lon = valid_coords['_lon'].mean()
                                                
                                                    # สร้างแผนที่
                                                    m = folium.Map(
                                                        location=[center_lat, center_lon],
                                                        zoom_start=8,
                                                        tiles='OpenStreetMap',
                                                        prefer_canvas=True  # เร็วขึ้น
                                                    )
                                                
                                                    # เพิ่มปุ่ม Fullscreen
                                                    plugins.Fullscreen(
                                                        position='topleft',
                                                        title='เต็มจอ',
                                                        title_cancel='ออกจากโหมดเต็มจอ',
                                                        force_separate_button=True
                                                    ).add_to(m)
                                                
                                                    # เพิ่ม DC Marker
                                                    folium.Marker(
                                                        location=[DC_LAT, DC_LON],
                                                        popup="<b>🏭 DC Wang Noi</b>",
                                                        tooltip="DC Wang Noi",
                                                        icon=folium.Icon(color='black', icon='home', prefix='fa')
                                                    ).add_to(m)
                                                
                                                    # สี palette สำหรับแต่ละทริป - 50 สีไม่ซ้ำกัน
                                                    colors = [
                                                        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',  # 1-5
                                                        '#a65628', '#f781bf', '#1b9e77', '#d95f02', '#7570b3',  # 6-10
                                                        '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666',  # 11-15
                                                        '#1f78b4', '#33a02c', '#fb9a99', '#fdbf6f', '#cab2d6',  # 16-20
                                                        '#b15928', '#8dd3c7', '#ffffb3', '#bebada', '#fb8072',  # 21-25
                                                        '#80b1d3', '#fdb462', '#b3de69', '#fccde5', '#d9d9d9',  # 26-30
                                                        '#bc80bd', '#ccebc5', '#ffed6f', '#e31a1c', '#1b7837',  # 31-35
                                                        '#762a83', '#e66101', '#5e3c99', '#d53e4f', '#3288bd',  # 36-40
                                                        '#f46d43', '#fdae61', '#fee08b', '#66c2a5', '#3d9970',  # 41-45
                                                        '#001f3f', '#39cccc', '#85144b', '#ff4136', '#2ecc40'   # 46-50
                                                    ]
                                                
                                                    # เรียงทริปตามเลข (Trip 1, 2, 3...) เพราะ renumber แล้ว
                                                    trip_max_dist = {}
                                                    for trip_id in valid_coords['Trip'].unique():
                                                        if '_distance_from_dc' in valid_coords.columns:
                                                            max_d = valid_coords[valid_coords['Trip'] == trip_id]['_distance_from_dc'].max()
                                                            trip_max_dist[trip_id] = max_d if pd.notna(max_d) else 0
                                                        else:
                                                            trip_max_dist[trip_id] = 0
                                                    sorted_trip_ids = sorted(trip_max_dist.keys(), key=lambda x: trip_max_dist[x], reverse=True)
                                                
                                                    # ฟังก์ชันเรียงสาขาแบบ Nearest Neighbor (ป้องกันกระโดด)
                                                    def optimize_branch_order(trip_df, dc_lat, dc_lon):
                                                        """เรียงสาขาให้ต่อเนื่องกัน: DC → สาขาไกลสุด → nearest → nearest → ... → DC"""
                                                        if len(trip_df) <= 1:
                                                            return trip_df
                                                    
                                                        df = trip_df.copy()
                                                        # เริ่มจากสาขาไกลสุด
                                                        ordered = []
                                                        remaining = df.to_dict('records')
                                                    
                                                        # หาสาขาไกลสุดเป็นจุดเริ่มต้น
                                                        farthest_idx = max(range(len(remaining)), key=lambda i: remaining[i]['_distance_from_dc'])
                                                        current = remaining.pop(farthest_idx)
                                                        ordered.append(current)
                                                    
                                                        # Nearest neighbor: หาสาขาใกล้สุดกับสาขาปัจจุบัน
                                                        while remaining:
                                                            current_lat, current_lon = current['_lat'], current['_lon']
                                                            nearest_idx = 0
                                                            nearest_dist = float('inf')
                                                        
                                                            for i, branch in enumerate(remaining):
                                                                # ใช้ cache distance
                                                                dist_key = f"{current_lat:.4f},{current_lon:.4f}_{branch['_lat']:.4f},{branch['_lon']:.4f}"
                                                                dist_key_rev = f"{branch['_lat']:.4f},{branch['_lon']:.4f}_{current_lat:.4f},{current_lon:.4f}"
                                                            
                                                                if dist_key in DISTANCE_CACHE:
                                                                    dist = DISTANCE_CACHE[dist_key]
                                                                elif dist_key_rev in DISTANCE_CACHE:
                                                                    dist = DISTANCE_CACHE[dist_key_rev]
                                                                else:
                                                                    # คำนวณ haversine
                                                                    dlat = radians(branch['_lat'] - current_lat)
                                                                    dlon = radians(branch['_lon'] - current_lon)
                                                                    a = sin(dlat/2)**2 + cos(radians(current_lat)) * cos(radians(branch['_lat'])) * sin(dlon/2)**2
                                                                    c = 2 * atan2(sqrt(a), sqrt(1-a))
                                                                    dist = 6371 * c
                                                            
                                                                if dist < nearest_dist:
                                                                    nearest_dist = dist
                                                                    nearest_idx = i
                                                        
                                                            current = remaining.pop(nearest_idx)
                                                            ordered.append(current)
                                                    
                                                        return pd.DataFrame(ordered)
                                                
                                                    # สร้าง Feature Groups สำหรับ Layer Control
                                                    trip_groups = {}
                                                
                                                    # วนลูปแต่ละทริป
                                                    for idx, trip_id in enumerate(sorted_trip_ids):
                                                        trip_data = valid_coords[valid_coords['Trip'] == trip_id].copy()
                                                        # เรียงสาขาแบบ Nearest Neighbor (ป้องกันกระโดด)
                                                        trip_data = optimize_branch_order(trip_data, DC_LAT, DC_LON)
                                                    
                                                        trip_color = colors[idx % len(colors)]
                                                        max_dist = trip_max_dist.get(trip_id, 0)
                                                    
                                                        # ดึงชื่อรถจาก summary
                                                        truck_info = summary[summary['Trip'] == trip_id]['Truck'].iloc[0] if trip_id in summary['Trip'].values else 'N/A'
                                                    
                                                        # สร้าง Feature Group สำหรับทริปนี้
                                                        fg = folium.FeatureGroup(name=f"Trip {trip_id} ({max_dist:.0f}km) - {truck_info}")
                                                        trip_groups[trip_id] = fg
                                                    
                                                        # เก็บพิกัดสาขา
                                                        points = []
                                                        point_names = []
                                                        point_distances = []
                                                    
                                                        for _, row in trip_data.iterrows():
                                                            if row['_lat'] > 0 and row['_lon'] > 0:
                                                                points.append([row['_lat'], row['_lon']])
                                                                point_names.append(f"{row.get('Name', row.get('Code', 'Unknown'))}")
                                                                point_distances.append(row.get('_distance_from_dc', 0))
                                                    
                                                        if len(points) == 0:
                                                            continue
                                                    
                                                        # 🛣️ ดึงเส้นทางจริงจาก OSRM (DC → สาขา1 → สาขา2 → ... → DC)
                                                        waypoints = [[DC_LAT, DC_LON]] + points + [[DC_LAT, DC_LON]]
                                                    
                                                        # ลองดึงเส้นทางจริง - ใช้ cache
                                                        cache_key = f"route_{trip_id}_{len(points)}"
                                                        if 'route_cache' not in st.session_state:
                                                            st.session_state['route_cache'] = {}
                                                    
                                                        if cache_key in st.session_state['route_cache']:
                                                            real_route_coords, total_trip_distance = st.session_state['route_cache'][cache_key]
                                                        else:
                                                            real_route_coords, total_trip_distance = get_multi_point_route_osrm(waypoints)
                                                            st.session_state['route_cache'][cache_key] = (real_route_coords, total_trip_distance)
                                                    
                                                        # ถ้า OSRM ไม่ได้ระยะทาง → ลองดึงระยะทางจาก DISTANCE_CACHE แทน
                                                        if total_trip_distance == 0:
                                                            for i in range(len(waypoints) - 1):
                                                                lat1, lon1 = waypoints[i]
                                                                lat2, lon2 = waypoints[i + 1]
                                                                seg_dist = haversine_distance(lat1, lon1, lat2, lon2)
                                                                if seg_dist < 9999:
                                                                    total_trip_distance += seg_dist
                                                    
                                                        # ปักหมุดแต่ละจุด
                                                        for i, (point, name, dist) in enumerate(zip(points, point_names, point_distances)):
                                                            # 🎯 แสดง T{trip}({ลำดับ}) บนหมุด เช่น T1(1), T1(2)
                                                            trip_label = f'<div style="background-color:{trip_color};color:#fff;border-radius:12px;min-width:50px;height:24px;text-align:center;line-height:24px;font-weight:bold;font-size:10px;border:2px solid #000;box-shadow:2px 2px 6px rgba(0,0,0,0.5);padding:0 4px;">T{trip_id}({i+1})</div>'
                                                        
                                                            popup_html = f"""
                                                            <div style="font-family:Arial;min-width:200px;">
                                                                <h4 style="margin:0;color:{trip_color};">🚚 Trip {trip_id}</h4>
                                                                <hr style="margin:5px 0;">
                                                                <b>ลำดับ:</b> {i+1}/{len(points)}<br>
                                                                <b>สาขา:</b> {name}<br>
                                                                <b>ห่างจาก DC:</b> {dist:.1f} km<br>
                                                                <b>รถ:</b> {truck_info}<br>
                                                                <hr style="margin:5px 0;">
                                                                <b>📏 ระยะทางรวมทริป:</b> {total_trip_distance:.1f} km<br>
                                                                <b>📍 จำนวนจุด:</b> {len(points)} สาขา
                                                            </div>
                                                            """
                                                        
                                                            folium.Marker(
                                                                location=point,
                                                                popup=folium.Popup(popup_html, max_width=300),
                                                                tooltip=f"Trip {trip_id} - {i+1}. {name} ({dist:.1f}km)",
                                                                icon=folium.DivIcon(html=trip_label)
                                                            ).add_to(fg)
                                                    
                                                        # วาดเส้นทางจริง DC → สาขา → DC (ถ้าเปิด)
                                                        if show_route and len(points) >= 1:
                                                            # ใช้เส้นทางจริงจาก OSRM
                                                            folium.PolyLine(
                                                                locations=real_route_coords,
                                                                weight=4,
                                                                color=trip_color,
                                                                opacity=0.8,
                                                                popup=f"Trip {trip_id}: {total_trip_distance:.1f} km (เส้นทางจริง)",
                                                                tooltip=f"🛣️ Trip {trip_id} - ระยะทาง {total_trip_distance:.1f} km"
                                                            ).add_to(fg)
                                                    
                                                        fg.add_to(m)
                                                
                                                    # เพิ่ม Layer Control สำหรับเปิด/ปิดแต่ละทริป
                                                    folium.LayerControl(collapsed=False).add_to(m)
                                            
                                                # แสดงแผนที่
                                                folium_static(m, width=1200, height=700)

                                                # บันทึก Map Cache
                                                st.session_state['_map_cache_key'] = map_cache_key
                                                st.session_state['_map_html'] = m._repr_html_()

                                            # แสดงแผนที่จาก cache เมื่อ cached
                                            if _map_is_cached and '_map_html' in st.session_state:
                                                import streamlit.components.v1 as _cmp
                                                _cmp.html(st.session_state['_map_html'], height=720, scrolling=False)
                                            
                                            # สรุประยะทางแต่ละทริป - แสดงข้อมูลละเอียด
                                            st.markdown("#### 📏 ระยะทางแต่ละทริป (เรียงจากไกล→ใกล้)")
                                            
                                            # สร้าง DataFrame สำหรับแสดงตาราง
                                            trip_details = []
                                            for trip_id in sorted_trip_ids:
                                                trip_data = valid_coords[valid_coords['Trip'] == trip_id].copy()
                                                if len(trip_data) == 0:
                                                    continue
                                                
                                                # เรียงตามระยะทางจาก DC (ไกล → ใกล้)
                                                trip_data = trip_data.sort_values('_distance_from_dc', ascending=False).reset_index(drop=True)
                                                
                                                # ดึงข้อมูลรถจาก summary
                                                truck_info = summary[summary['Trip'] == trip_id]['Truck'].iloc[0] if trip_id in summary['Trip'].values else 'N/A'
                                                truck_type = truck_info.split()[0] if truck_info else 'N/A'
                                                
                                                # คำนวณน้ำหนักและคิวรวม
                                                total_weight = trip_data['Weight'].sum()
                                                total_cube = trip_data['Cube'].sum()
                                                
                                                # สาขาไกลสุดจาก DC
                                                max_dist_from_dc = trip_data['_distance_from_dc'].max()
                                                
                                                # คำนวณระยะทางรวม (DC → สาขา1 → สาขา2 → ... → สาขาสุดท้าย) - ไม่รวมกลับ DC
                                                points = []
                                                for _, row in trip_data.iterrows():
                                                    if row['_lat'] > 0 and row['_lon'] > 0:
                                                        points.append([row['_lat'], row['_lon']])
                                                
                                                route_distance = 0  # DC → สาขา1 → ... → สาขาสุดท้าย
                                                inter_branch_distance = 0  # สาขา → สาขา (ไม่รวม DC)
                                                
                                                if len(points) > 0:
                                                    # DC → สาขาแรก (ไกลสุด)
                                                    route_distance += haversine_distance(DC_LAT, DC_LON, points[0][0], points[0][1])
                                                    
                                                    # สาขา → สาขา
                                                    for j in range(len(points) - 1):
                                                        seg_dist = haversine_distance(points[j][0], points[j][1], points[j+1][0], points[j+1][1])
                                                        route_distance += seg_dist
                                                        inter_branch_distance += seg_dist
                                                
                                                trip_details.append({
                                                    'ทริป': trip_id,
                                                    'รถ': truck_type,
                                                    'สาขา': len(trip_data),
                                                    'น้ำหนัก (kg)': f"{total_weight:,.0f}",
                                                    'คิว (m³)': f"{total_cube:.1f}",
                                                    'ไกลสุดจาก DC': f"{max_dist_from_dc:.1f} km",
                                                    'ระยะทางรวม': f"{route_distance:.1f} km",
                                                    'ระหว่างสาขา': f"{inter_branch_distance:.1f} km"
                                                })
                                            
                                            # แสดงตาราง (หลังวนครบทุก trip แล้ว)
                                            if trip_details:
                                                trip_df = pd.DataFrame(trip_details)
                                                st.dataframe(
                                                    trip_df,
                                                    width="stretch",
                                                    hide_index=True,
                                                    column_config={
                                                        'ทริป': st.column_config.NumberColumn('🚚 ทริป', width='small'),
                                                        'รถ': st.column_config.TextColumn('🚛 รถ', width='small'),
                                                        'สาขา': st.column_config.NumberColumn('📍 สาขา', width='small'),
                                                        'น้ำหนัก (kg)': st.column_config.TextColumn('⚖️ น้ำหนัก', width='small'),
                                                        'คิว (m³)': st.column_config.TextColumn('📦 คิว', width='small'),
                                                        'ไกลสุดจาก DC': st.column_config.TextColumn('🎯 ไกลสุด', width='small'),
                                                        'ระยะทางรวม': st.column_config.TextColumn('📏 รวม (DC→สาขา)', width='medium'),
                                                        'ระหว่างสาขา': st.column_config.TextColumn('↔️ ระหว่างสาขา', width='small')
                                                    }
                                                )
                                                
                                                # สรุปรวม
                                                total_route = sum(float(d['ระยะทางรวม'].replace(' km', '').replace(',', '')) for d in trip_details)
                                                total_inter = sum(float(d['ระหว่างสาขา'].replace(' km', '').replace(',', '')) for d in trip_details)
                                                st.caption(f"📊 **รวมทั้งหมด:** {len(trip_details)} ทริป | ระยะทางรวม: {total_route:,.1f} km | ระหว่างสาขา: {total_inter:,.1f} km")
                                            
                                            st.caption(f"📍 แสดง {len(valid_coords)} สาขาใน {len(sorted_trip_ids)} ทริป | 💡 คลิกปุ่มมุมซ้ายบนเพื่อเต็มจอ | ใช้ Layer Control ด้านขวาเพื่อเปิด/ปิดทริป")
                                    else:
                                        st.warning("⚠️ ไม่มีข้อมูลพิกัดในผลลัพธ์ (ต้องมีคอลัมน์ _lat และ _lon)")
                        
                # ==========================================
                # แท็บ 2: จัดกลุ่มสาขาตามภาค (ไม่สนน้ำหนัก)
                # ==========================================
                with tab2:
                    df_region = df.copy()
                    
                    # จัดกลุ่มตามภาค
                    branch_info = model_data.get('branch_info', {})
                    trip_pairs = model_data.get('trip_pairs', set())
                    
                    # สร้างข้อมูลภาคสำหรับแต่ละสาขา (จากไฟล์ประวัติ)
                    region_groups = {
                        'ภาคกลาง-กรุงเทพชั้นใน': ['กรุงเทพมหานคร'],
                        'ภาคกลาง-กรุงเทพชั้นกลาง': ['กรุงเทพมหานคร'],
                        'ภาคกลาง-กรุงเทพชั้นนอก': ['กรุงเทพมหานคร'],
                        'ภาคกลาง-ปริมณฑล': ['นครปฐม', 'นนทบุรี', 'ปทุมธานี', 'สมุทรปราการ', 'สมุทรสาคร'],
                        'ภาคกลาง-กลางตอนบน': ['ชัยนาท', 'พระนครศรีอยุธยา', 'ลพบุรี', 'สระบุรี', 'สิงห์บุรี', 'อ่างทอง', 'อยุธยา'],
                        'ภาคกลาง-กลางตอนล่าง': ['สมุทรสงคราม', 'สุพรรณบุรี'],
                        'ภาคตะวันตก': ['กาญจนบุรี', 'ประจวบคีรีขันธ์', 'ราชบุรี', 'เพชรบุรี'],
                        'ภาคตะวันออก': ['จันทบุรี', 'ชลบุรี', 'ตราด', 'นครนายก', 'ปราจีนบุรี', 'ระยอง', 'สระแก้ว', 'ฉะเชิงเทรา'],
                        'ภาคอีสาน-อีสานเหนือ': ['นครพนม', 'บึงกาฬ', 'มุกดาหาร', 'สกลนคร', 'หนองคาย', 'หนองบัวลำภู', 'อุดรธานี', 'เลย'],
                        'ภาคอีสาน-อีสานกลาง': ['กาฬสินธุ์', 'ขอนแก่น', 'ชัยภูมิ', 'มหาสารคาม', 'ร้อยเอ็ด'],
                        'ภาคอีสาน-อีสานใต้': ['นครราชสีมา', 'โคราช', 'บุรีรัมย์', 'ยโสธร', 'ศรีสะเกษ', 'สุรินทร์', 'อำนาจเจริญ', 'อุบลราชธานี'],
                        'ภาคเหนือ-เหนือตอนบน': ['น่าน', 'พะเยา', 'ลำปาง', 'ลำพูน', 'เชียงราย', 'เชียงใหม่', 'แพร่', 'แม่ฮ่องสอน'],
                        'ภาคเหนือ-เหนือตอนล่าง': ['กำแพงเพชร', 'ตาก', 'นครสวรรค์', 'พิจิตร', 'พิษณุโลก', 'สุโขทัย', 'อุตรดิตถ์', 'อุทัยธานี', 'เพชรบูรณ์'],
                        'ภาคใต้-ใต้ฝั่งอันดามัน': ['กระบี่', 'ตรัง', 'พังงา', 'ภูเก็ต', 'ระนอง', 'สตูล'],
                        'ภาคใต้-ใต้ฝั่งอ่าวไทย': ['ชุมพร', 'นครศรีธรรมราช', 'พัทลุง', 'ยะลา', 'สงขลา', 'สุราษฎร์ธานี', 'ปัตตานี', 'นราธิวาส']
                    }
                    
                    def get_region(province):
                        if pd.isna(province) or not province or str(province).strip() in ['', 'nan', 'UNKNOWN']:
                            return 'ไม่ระบุ'
                        
                        # 🚨 Override: ฉะเชิงเทรา → ภาคตะวันออก (ไม่ใช่ปริมณฑล)
                        if 'ฉะเชิงเทรา' in str(province):
                            return 'ภาคตะวันออก'
                        
                        for region, provinces in region_groups.items():
                            if any(p in str(province) for p in provinces):
                                return region
                        return 'อื่นๆ'
                    
                    # เพิ่มคอลัมน์ภาค - ดึงจังหวัดจาก Master ถ้าไม่มี
                    # รองรับทั้งชื่อคอลัมน์ภาษาอังกฤษ (Province) และไทย (จังหวัด)
                    province_col = None
                    if 'Province' in df_region.columns:
                        province_col = 'Province'
                    elif 'จังหวัด' in df_region.columns:
                        province_col = 'จังหวัด'
                    
                    # ถ้าไม่มีคอลัมน์จังหวัดเลย หรือมีแต่เป็น NaN ทั้งหมด → ดึงจาก MASTER_DATA
                    # Vectorized: สร้าง province_map ครั้งเดียวแทน iterrows
                    _prov_map = {}
                    if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns and 'จังหวัด' in MASTER_DATA.columns:
                        _pm = MASTER_DATA[['Plan Code', 'จังหวัด']].dropna(subset=['Plan Code'])
                        _prov_map = dict(zip(_pm['Plan Code'].astype(str).str.strip(), _pm['จังหวัด'].fillna('')))
                    
                    if not province_col or df_region[province_col].isna().all():
                        df_region['จังหวัด'] = df_region['Code'].astype(str).map(_prov_map).fillna('UNKNOWN')
                        province_col = 'จังหวัด'
                    elif province_col and df_region[province_col].isna().any():
                        # เติมเฉพาะที่เป็น NaN
                        _missing = df_region[province_col].isna()
                        df_region.loc[_missing, province_col] = df_region.loc[_missing, 'Code'].astype(str).map(_prov_map).fillna('UNKNOWN')
                    
                    # ตรวจสอบอีกครั้งว่ามีคอลัมน์จังหวัดแล้ว
                    if not province_col or province_col not in df_region.columns:
                        st.error("❌ ไม่พบข้อมูลจังหวัด กรุณาตรวจสอบไฟล์ข้อมูล")
                        return
                    
                    df_region['Region'] = df_region[province_col].apply(get_region)
                    
                    # หากลุ่มสาขา (ใช้ Booking No. เป็นหลัก)
                    def find_paired_branches(code, code_province, df_data):
                        paired = set()
                        
                        # หา Booking No. ของสาขานี้
                        code_rows = df_data[df_data['Code'] == code]
                        if len(code_rows) == 0:
                            return paired
                        
                        # เช็คว่ามีคอลัมน์ Booking หรือไม่
                        if 'Booking' not in df_data.columns and 'Trip' not in df_data.columns:
                            return paired
                        
                        booking_col = 'Booking' if 'Booking' in df_data.columns else 'Trip'
                        code_bookings = set(code_rows[booking_col].dropna().astype(str))
                        
                        if not code_bookings:
                            return paired
                        
                        # หาสาขาอื่นที่อยู่ Booking เดียวกัน (ไม่สนจังหวัด)
                        for booking in code_bookings:
                            if booking == 'nan' or not booking.strip():
                                continue
                            
                            same_booking = df_data[df_data[booking_col].astype(str) == booking]
                            for _, other_row in same_booking.iterrows():
                                other_code = other_row['Code']
                                
                                # เงื่อนไข: Booking เดียวกัน = รวมกลุ่ม (ไม่สนจังหวัด)
                                if other_code != code:
                                    paired.add(other_code)
                        
                        return paired
                    
                    all_codes_set = set(df_region['Code'].unique())
                    
                    # สร้างกลุ่มสาขาแบบ Union-Find (ตามลำดับ: ตำบล → อำเภอ → จังหวัด)
                    # Step 1: เริ่มจากแต่ละสาขาเป็นกลุ่มๆ พร้อมข้อมูล Master
                    initial_groups = {}
                    for code in all_codes_set:
                        # ดึงข้อมูลจาก Master
                        location = {}
                        if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
                            master_row = MASTER_DATA[MASTER_DATA['Plan Code'] == code]
                            if len(master_row) > 0:
                                master_row = master_row.iloc[0]
                                location = {
                                    'subdistrict': master_row.get('ตำบล', ''),
                                    'district': master_row.get('อำเภอ', ''),
                                    'province': master_row.get('จังหวัด', 'UNKNOWN'),
                                    'lat': master_row.get('ละติจูด', 0),
                                    'lon': master_row.get('ลองติจูด', 0)
                                }
                        
                        # ถ้าไม่มีใน Master ลองดึงจากไฟล์อัปโหลด
                        if not location or location.get('province', 'UNKNOWN') == 'UNKNOWN':
                            c_row = df_region[df_region['Code'] == code].iloc[0] if len(df_region[df_region['Code'] == code]) > 0 else None
                            if c_row is not None:
                                location = {
                                    'subdistrict': '',
                                    'district': '',
                                    'province': c_row.get('Province', 'UNKNOWN'),
                                    'lat': 0,
                                    'lon': 0
                                }
                        
                        if location:
                            initial_groups[(code,)] = {code: location}
                    
                    # ใช้ initial_groups แทน booking_groups
                    booking_groups = initial_groups
                    
                    # Step 2: รวมกลุ่มตามลำดับ ตำบล → อำเภอ → จังหวัด
                    def groups_can_merge(locs1, locs2):
                        """เช็คว่า 2 กลุ่มควรรวมกันไหม (ตามลำดับความละเอียด)"""
                        # 1. เช็คตำบลเดียวกัน (ต้องมีข้อมูลตำบล)
                        subdistricts1 = set(loc.get('subdistrict', '') for loc in locs1.values() if loc.get('subdistrict', ''))
                        subdistricts2 = set(loc.get('subdistrict', '') for loc in locs2.values() if loc.get('subdistrict', ''))
                        if subdistricts1 and subdistricts2 and (subdistricts1 & subdistricts2):
                            return True, 'ตำบล'
                        
                        # 2. เช็คอำเภอเดียวกัน (ต้องมีข้อมูลอำเภอและจังหวัดเดียวกัน)
                        districts1 = {(loc.get('district', ''), loc.get('province', '')) for loc in locs1.values() if loc.get('district', '')}
                        districts2 = {(loc.get('district', ''), loc.get('province', '')) for loc in locs2.values() if loc.get('district', '')}
                        if districts1 and districts2:
                            # เช็คว่ามีอำเภอและจังหวัดตรงกัน
                            for d1, p1 in districts1:
                                for d2, p2 in districts2:
                                    if d1 == d2 and p1 == p2 and p1:
                                        return True, 'อำเภอ'
                        
                        # 3. เช็คจังหวัดเดียวกัน
                        provinces1 = set(loc.get('province', '') for loc in locs1.values() if loc.get('province', ''))
                        provinces2 = set(loc.get('province', '') for loc in locs2.values() if loc.get('province', ''))
                        if provinces1 & provinces2:
                            return True, 'จังหวัด'
                        
                        return False, None
                    
                    merged_groups = []
                    used_groups = set()
                    
                    for group1, locs1 in booking_groups.items():
                        if group1 in used_groups:
                            continue
                        
                        merged_codes = set(group1)
                        merged_locs = locs1.copy()
                        used_groups.add(group1)
                        
                        # หากลุ่มอื่นที่ใกล้เคียง
                        changed = True
                        while changed:
                            changed = False
                            for group2, locs2 in booking_groups.items():
                                if group2 in used_groups:
                                    continue
                                can_merge, level = groups_can_merge(merged_locs, locs2)
                                if can_merge:
                                    merged_codes |= set(group2)
                                    merged_locs.update(locs2)
                                    used_groups.add(group2)
                                    changed = True
                        
                        merged_groups.append({
                            'codes': merged_codes,
                            'locations': merged_locs
                        })
                    
                    # Step 3: แปลงเป็น groups format
                    groups = []
                    for mg in merged_groups:
                        rep_code = list(mg['codes'])[0]
                        rep_row = df_region[df_region['Code'] == rep_code].iloc[0]
                        # กรองเฉพาะจังหวัดที่ไม่ใช่ UNKNOWN และไม่เป็น NaN
                        provinces = set(
                            str(loc.get('province', '')).strip() 
                            for loc in mg['locations'].values() 
                            if loc.get('province') and str(loc.get('province', '')).strip() not in ['UNKNOWN', 'nan', '']
                        )
                        
                        # ถ้าไม่มีจังหวัดเลย ใส่ "ไม่ระบุ"
                        province_str = ', '.join(sorted(provinces)) if provinces else 'ไม่ระบุ'
                        
                        groups.append({
                            'codes': mg['codes'],
                            'region': str(rep_row.get('Region') or 'ไม่ระบุ'),
                            'province': province_str
                        })
                    
                    # แสดงสถิติ
                    st.markdown("---")
                    st.markdown("### 📊 สรุปการจัดกลุ่ม")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📍 จำนวนสาขา", df_region['Code'].nunique())
                    with col2:
                        st.metric("🗂️ จำนวนกลุ่ม", len(groups))
                    with col3:
                        regions_count = df_region['Region'].nunique()
                        st.metric("🗺️ จำนวนภาค", regions_count)
                    
                    # แสดงตามภาค
                    st.markdown("---")
                    st.markdown("### 🗺️ สาขาแยกตามภาค")
                    
                    region_summary = df_region.groupby('Region').agg({
                        'Code': 'nunique',
                        'Weight': 'sum',
                        'Cube': 'sum'
                    }).reset_index()
                    region_summary.columns = ['ภาค', 'จำนวนสาขา', 'น้ำหนักรวม', 'คิวรวม']
                    st.dataframe(region_summary, width="stretch")
                    
                    # แสดงรายละเอียดแต่ละภาค
                    for region in sorted(df_region['Region'].unique()):
                        region_data = df_region[df_region['Region'] == region]
                        with st.expander(f"📍 {region} ({region_data['Code'].nunique()} สาขา)"):
                            display_cols = ['Code', 'Name', 'Province', 'Weight', 'Cube']
                            display_cols = [c for c in display_cols if c in region_data.columns]
                            
                            region_display = region_data[display_cols].drop_duplicates('Code')
                            col_names = {'Code': 'รหัส', 'Name': 'ชื่อสาขา', 'Province': 'จังหวัด', 'Weight': 'น้ำหนัก', 'Cube': 'คิว'}
                            region_display.columns = [col_names.get(c, c) for c in display_cols]
                            st.dataframe(region_display, width="stretch")
                    
                    # แสดงกลุ่มสาขาที่เคยไปด้วยกัน
                    st.markdown("---")
                    st.markdown("### 🔗 กลุ่มสาขาที่เคยไปด้วยกัน (จากประวัติ)")
                    
                    paired_groups = [g for g in groups if len(g['codes']) > 1]
                    if paired_groups:
                        for i, group in enumerate(paired_groups, 1):
                            codes_list = list(group['codes'])
                            names = []
                            for c in codes_list:
                                name_row = df_region[df_region['Code'] == c]
                                if len(name_row) > 0 and 'Name' in name_row.columns:
                                    _nm = name_row['Name'].iloc[0]
                                    _nm_str = str(_nm) if (_nm is not None and not (isinstance(_nm, float) and pd.isna(_nm))) else ''
                                    names.append(f"{c}" + (f" ({_nm_str})" if _nm_str else ''))
                                else:
                                    names.append(str(c))
                            
                            region_label = group['region'] or 'ไม่ระบุ'
                            st.write(f"**กลุ่ม {i}** — {region_label}: {', '.join(names)}")
                    else:
                        st.info("ไม่พบกลุ่มสาขาที่เคยไปด้วยกันในรายการนี้")
                    
                    # ดาวน์โหลด
                    st.markdown("---")
                    output_region = io.BytesIO()
                    with pd.ExcelWriter(output_region, engine='xlsxwriter') as writer:
                        df_region.to_excel(writer, sheet_name='สาขาทั้งหมด', index=False)
                        region_summary.to_excel(writer, sheet_name='สรุปตามภาค', index=False)
                    
                    st.download_button(
                        label="📥 ดาวน์โหลดข้อมูลจัดกลุ่ม (Excel)",
                        data=output_region.getvalue(),
                        file_name=f"จัดกลุ่มสาขา_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        width="stretch"
                    )

if __name__ == "__main__":
    try:
        main()
    finally:
        # บันทึก cache ก่อนปิดโปรแกรม
        if USE_CACHE:
            save_distance_cache(DISTANCE_CACHE)
            save_route_cache(ROUTE_CACHE_DATA)
            safe_print(f"💾 บันทึก cache: {len(DISTANCE_CACHE)} ระยะทาง, {len(ROUTE_CACHE_DATA)} เส้นทาง")

