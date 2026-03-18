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

# ── AI TRIP LEARNING SYSTEM ──────────────────────────────────────────────────
# บันทึก/โหลดประวัติการจัดทริป เพื่อเรียนรู้ว่าสาขาไหนมักอยู่ทริปเดียวกัน

TRIP_HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'trip_history.json')
_TRIP_HISTORY_CACHE = None   # in-process cache (cleared each save)

def load_trip_history() -> dict:
    """โหลด pair_freq dict: {"CODE_A|CODE_B": count}  (sorted keys)"""
    global _TRIP_HISTORY_CACHE
    if _TRIP_HISTORY_CACHE is not None:
        return _TRIP_HISTORY_CACHE
    if os.path.exists(TRIP_HISTORY_FILE):
        try:
            with open(TRIP_HISTORY_FILE, encoding='utf-8') as f:
                data = json.load(f)
            _TRIP_HISTORY_CACHE = data.get('pair_freq', {})
            return _TRIP_HISTORY_CACHE
        except Exception:
            pass
    return {}

def save_trip_history(assigned_df) -> int:
    """
    เรียกหลัง export — บันทึกว่าสาขาไหนอยู่ทริปเดียวกัน
    คืนค่าจำนวน pairs ที่บันทึกเพิ่มใน session นี้
    """
    global _TRIP_HISTORY_CACHE
    # โหลดข้อมูลเดิม
    if os.path.exists(TRIP_HISTORY_FILE):
        try:
            with open(TRIP_HISTORY_FILE, encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    pair_freq   = data.get('pair_freq', {})
    sessions    = data.get('sessions', [])
    total_saved = 0

    # สร้าง group: trip_id → list of codes
    trip_groups = {}
    for _, row in assigned_df.iterrows():
        tid  = str(int(row.get('Trip', 0)))
        code = str(row.get('Code', '')).strip().upper()
        if code and tid != '0':
            trip_groups.setdefault(tid, []).append(code)

    session_pairs = []
    import itertools
    for tid, codes in trip_groups.items():
        codes = sorted(set(codes))
        for a, b in itertools.combinations(codes, 2):
            key = f"{a}|{b}"
            pair_freq[key] = pair_freq.get(key, 0) + 1
            session_pairs.append(key)
            total_saved += 1

    sessions.append({
        'date':       datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        'trips':      len(trip_groups),
        'pair_count': total_saved,
    })
    # เก็บเฉพาะ 500 sessions ล่าสุด
    if len(sessions) > 500:
        sessions = sessions[-500:]

    data = {'pair_freq': pair_freq, 'sessions': sessions,
            'total_sessions': len(sessions)}

    with open(TRIP_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    _TRIP_HISTORY_CACHE = pair_freq   # refresh in-process cache
    return total_saved

def get_trip_learning_stats() -> dict:
    """สรุปสถิติการเรียนรู้เพื่อแสดงใน UI"""
    if not os.path.exists(TRIP_HISTORY_FILE):
        return {'sessions': 0, 'unique_pairs': 0, 'top_pairs': []}
    try:
        with open(TRIP_HISTORY_FILE, encoding='utf-8') as f:
            data = json.load(f)
        pf  = data.get('pair_freq', {})
        top = sorted(pf.items(), key=lambda x: -x[1])[:10]
        return {
            'sessions':     len(data.get('sessions', [])),
            'unique_pairs': len(pf),
            'top_pairs':    top,
        }
    except Exception:
        return {'sessions': 0, 'unique_pairs': 0, 'top_pairs': []}

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
    
    # เชื่อมต่อ Google Sheets (ใส่ socket timeout ป้องกัน hang ตอน startup)
    if creds:
        import socket as _socket
        _prev_timeout = _socket.getdefaulttimeout()
        _socket.setdefaulttimeout(10)  # 10 วินาที max สำหรับ network calls
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
        finally:
            _socket.setdefaulttimeout(_prev_timeout)  # restore
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
            import re as _re_j
            with open(json_file, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            # normalize inner dict keys (column names อาจมี space/newline)
            for k, v in raw.items():
                if isinstance(v, dict):
                    existing_data[k] = {_re_j.sub(r'[\s\n\r\t]+', '', str(ck)): cv for ck, cv in v.items()}
                else:
                    existing_data[k] = v
        except Exception as e:
            safe_print(f"⚠️ ไม่สามารถอ่าน JSON: {e}")
    
    # ถ้าไม่มี Google Sheets ให้ใช้ข้อมูลเก่า
    if not SHEETS_AVAILABLE or sh is None:
        if existing_data:
            safe_print(f"⚠️ Google Sheets ไม่พร้อม - ใช้ข้อมูลจาก JSON ({len(existing_data)} สาขา)")
            df = pd.DataFrame.from_dict(existing_data, orient='index')
            # ตรวจสอบว่ามีคอลัมน์ Plan Code หรือไม่
            # (JSON keys ถูก normalize แล้ว → 'Plan Code' กลายเป็น 'PlanCode')
            if 'Plan Code' in df.columns:
                df.reset_index(drop=True, inplace=True)
            elif 'PlanCode' in df.columns:
                df.reset_index(drop=True, inplace=True)
                df.rename(columns={'PlanCode': 'Plan Code'}, inplace=True)
            else:
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Plan Code'}, inplace=True)
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
        
        # สร้าง DataFrame — normalize headers ก่อนเพื่อให้ column names สะอาด
        import re as _re_hdr
        headers = [_re_hdr.sub(r'[\s\n\r\t]+', '', str(h)) for h in data[0]]
        # คืน 'Plan Code' ให้ถูกต้อง (หลัง normalize 'Plan Code' → 'PlanCode')
        headers = ['Plan Code' if h == 'PlanCode' else h for h in headers]
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
            
            # แปลง row เป็น dict — normalize column keys ให้ตรงกับ JSON (ลบ space/newline)
            import re as _re_rd
            row_dict = {_re_rd.sub(r'[\s\n\r\t]+', '', str(ck)): cv for ck, cv in row.to_dict().items()}
            
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
        
        # ป้องกัน duplicate column: PlanCode (inner dict) + index (branch code) จะซ้ำกัน
        # ให้ drop index เสมอ แล้วใช้ PlanCode column แทน
        if 'PlanCode' in df.columns:
            df.reset_index(drop=True, inplace=True)
            df.rename(columns={'PlanCode': 'Plan Code'}, inplace=True)
        elif 'Plan Code' in df.columns:
            df.reset_index(drop=True, inplace=True)
        else:
            df.reset_index(inplace=True)
            df.rename(columns={'index': 'Plan Code'}, inplace=True)
        
        return df
        
    except Exception as e:
        safe_print(f"❌ Error: {e}")
        # ถ้าเกิด error ให้ใช้ข้อมูลเก่า
        if existing_data:
            safe_print(f"📦 ใช้ข้อมูลเก่าจาก JSON")
            df = pd.DataFrame.from_dict(existing_data, orient='index')
            if 'PlanCode' in df.columns:
                df.reset_index(drop=True, inplace=True)
                df.rename(columns={'PlanCode': 'Plan Code'}, inplace=True)
            elif 'Plan Code' in df.columns:
                df.reset_index(drop=True, inplace=True)
            else:
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Plan Code'}, inplace=True)
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

# ════════════════════════════════════════════════════════════════
# PROVINCE ZONE MAP — ตรงกับ zone_viewer.py (ใช้จัดทริป ป้องกันกระโดด)
# ════════════════════════════════════════════════════════════════
PROVINCE_ZONE_MAP: dict = {
    "กรุงเทพมหานคร": "__BKK__", "กรุงเทพฯ": "__BKK__",
    "กทม": "__BKK__", "กทม.": "__BKK__",
    # ปริมณฑล
    "นนทบุรี":"ปริมณฑล_นนทบุรี","ปทุมธานี":"ปริมณฑล_ปทุมธานี",
    "สมุทรปราการ":"ปริมณฑล_สมุทรปราการ","นครปฐม":"ปริมณฑล_นครปฐม",
    "สมุทรสาคร":"ปริมณฑล_สมุทรสาคร","สมุทรสงคราม":"ปริมณฑล_สมุทรสงคราม",
    "พระนครศรีอยุธยา":"ปริมณฑล_อยุธยา","สระบุรี":"ปริมณฑล_สระบุรี",
    "อ่างทอง":"ปริมณฑล_อ่างทอง","สิงห์บุรี":"ปริมณฑล_สิงห์บุรี",
    "ชัยนาท":"ปริมณฑล_ชัยนาท","ลพบุรี":"ปริมณฑล_ลพบุรี",
    # ภาคเหนือ
    "นครสวรรค์":"เหนือ_นครสวรรค์","อุทัยธานี":"เหนือ_อุทัยธานี",
    "กำแพงเพชร":"เหนือ_กำแพงเพชร","ตาก":"เหนือ_ตาก",
    "สุโขทัย":"เหนือ_สุโขทัย","พิษณุโลก":"เหนือ_พิษณุโลก",
    "พิจิตร":"เหนือ_พิจิตร","เพชรบูรณ์":"เหนือ_เพชรบูรณ์",
    "อุตรดิตถ์":"เหนือ_อุตรดิตถ์","แพร่":"เหนือ_แพร่",
    "น่าน":"เหนือ_น่าน","พะเยา":"เหนือ_พะเยา",
    "เชียงราย":"เหนือ_เชียงราย","เชียงใหม่":"เหนือ_เชียงใหม่",
    "ลำพูน":"เหนือ_ลำพูน","ลำปาง":"เหนือ_ลำปาง",
    "แม่ฮ่องสอน":"เหนือ_แม่ฮ่องสอน",
    # ภาคอีสาน
    "หนองบัวลำภู":"อีสาน_หนองบัวลำภู","อุดรธานี":"อีสาน_อุดรธานี",
    "หนองคาย":"อีสาน_หนองคาย","บึงกาฬ":"อีสาน_บึงกาฬ",
    "เลย":"อีสาน_เลย","สกลนคร":"อีสาน_สกลนคร",
    "นครพนม":"อีสาน_นครพนม","มุกดาหาร":"อีสาน_มุกดาหาร",
    "ชัยภูมิ":"อีสาน_ชัยภูมิ","ขอนแก่น":"อีสาน_ขอนแก่น",
    "กาฬสินธุ์":"อีสาน_กาฬสินธุ์","มหาสารคาม":"อีสาน_มหาสารคาม",
    "ร้อยเอ็ด":"อีสาน_ร้อยเอ็ด","นครราชสีมา":"อีสาน_นครราชสีมา",
    "บุรีรัมย์":"อีสาน_บุรีรัมย์","สุรินทร์":"อีสาน_สุรินทร์",
    "ศรีสะเกษ":"อีสาน_ศรีสะเกษ","อุบลราชธานี":"อีสาน_อุบลราชธานี",
    "ยโสธร":"อีสาน_ยโสธร","อำนาจเจริญ":"อีสาน_อำนาจเจริญ",
    # ภาคตะวันออก
    "ฉะเชิงเทรา":"ตะวันออก_ฉะเชิงเทรา","นครนายก":"ตะวันออก_นครนายก",
    "ปราจีนบุรี":"ตะวันออก_ปราจีนบุรี","สระแก้ว":"ตะวันออก_สระแก้ว",
    "ชลบุรี":"ตะวันออก_ชลบุรี","ระยอง":"ตะวันออก_ระยอง",
    "จันทบุรี":"ตะวันออก_จันทบุรี","ตราด":"ตะวันออก_ตราด",
    # ภาคตะวันตก
    "กาญจนบุรี":"ตะวันตก_กาญจนบุรี","ราชบุรี":"ตะวันตก_ราชบุรี",
    "สุพรรณบุรี":"ตะวันตก_สุพรรณบุรี","เพชรบุรี":"ตะวันตก_เพชรบุรี",
    "ประจวบคีรีขันธ์":"ตะวันตก_ประจวบคีรีขันธ์",
    # ภาคใต้
    "ชุมพร":"ใต้_ชุมพร","ระนอง":"ใต้_ระนอง",
    "สุราษฎร์ธานี":"ใต้_สุราษฎร์ธานี","นครศรีธรรมราช":"ใต้_นครศรีธรรมราช",
    "พังงา":"ใต้_พังงา","กระบี่":"ใต้_กระบี่","ภูเก็ต":"ใต้_ภูเก็ต",
    "ตรัง":"ใต้_ตรัง","พัทลุง":"ใต้_พัทลุง","สตูล":"ใต้_สตูล",
    "สงขลา":"ใต้_สงขลา","ปัตตานี":"ใต้_ปัตตานี",
    "ยะลา":"ใต้_ยะลา","นราธิวาส":"ใต้_นราธิวาส",
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
    'ZONE_TAK_ตาก': {
        'provinces': ['ตาก'],
        'highway': '1/105',
        'priority': 10,
        'distance_from_dc_km': 420,
        'description': 'ตาก-แม่สอด สาย 1/105 เหนือตอนล่างฝั่งตะวันตก'
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

# 🏙️ Bangkok geographic center (สำหรับแบ่ง sub-zone)
_BKK_CENTER_LAT = 13.7563
_BKK_CENTER_LON = 100.5018
_BKK_CENTER_RADIUS_KM = 4.5  # รัศมี BKK_CENTER

# ชื่อ sub-zone กรุงเทพ (8 ทิศ + กลาง)
BKK_SUBZONE_NAMES = {
    'BKK_CENTER': 'กรุงเทพ - ใจกลาง (Silom/Sathorn/Siam)',
    'BKK_N':      'กรุงเทพ - เหนือ (ดอนเมือง/ลาดยาว/หลักสี่)',
    'BKK_NE':     'กรุงเทพ - ตะวันออกเฉียงเหนือ (ลาดพร้าว/มีนบุรี)',
    'BKK_E':      'กรุงเทพ - ตะวันออก (วังทองหลาง/ลาดกระบัง)',
    'BKK_SE':     'กรุงเทพ - ตะวันออกเฉียงใต้ (พระโขนง/บางนา)',
    'BKK_S':      'กรุงเทพ - ใต้ (ราษฎร์บูรณะ/บางขุนเทียน)',
    'BKK_SW':     'กรุงเทพ - ตะวันตกเฉียงใต้ (ธนบุรี/หนองแขม)',
    'BKK_W':      'กรุงเทพ - ตะวันตก (ตลิ่งชัน/บางแค)',
    'BKK_NW':     'กรุงเทพ - ตะวันตกเฉียงเหนือ (บางพลัด/บางซื่อ)',
}

def get_bkk_sub_zone(lat, lon):
    """
    จัดกรุงเทพแบ่ง sub-zone จากทิศทาง + ระยะจากใจกลาง
    Returns: 'BKK_CENTER' | 'BKK_N' | 'BKK_NE' | ... | 'BKK_NW'
    """
    if not lat or not lon or lat == 0 or lon == 0:
        return 'BKK_CENTER'
    dist = haversine_distance(_BKK_CENTER_LAT, _BKK_CENTER_LON, lat, lon)
    if dist <= _BKK_CENTER_RADIUS_KM:
        return 'BKK_CENTER'
    bearing = calculate_bearing(_BKK_CENTER_LAT, _BKK_CENTER_LON, lat, lon)
    # 8 sectors (45° each), starting from North
    if bearing < 22.5 or bearing >= 337.5:
        return 'BKK_N'
    elif bearing < 67.5:
        return 'BKK_NE'
    elif bearing < 112.5:
        return 'BKK_E'
    elif bearing < 157.5:
        return 'BKK_SE'
    elif bearing < 202.5:
        return 'BKK_S'
    elif bearing < 247.5:
        return 'BKK_SW'
    elif bearing < 292.5:
        return 'BKK_W'
    else:
        return 'BKK_NW'


def get_prov_zone(province: str, district: str = '') -> str:
    """
    ดึงโซนจัดส่งระดับจังหวัด (ระบบเดียวกับ zone_viewer.py)
    BKK  → BKK_{เขต}    |    จังหวัดอื่น → {ภาค}_{จังหวัด}
    ใช้เป็น primary key สำหรับจัดทริป ป้องกันกระโดดข้ามภาค/จังหวัด
    """
    if not province:
        return 'ไม่ระบุ'
    prov = str(province).strip()
    _alias = {'กรุงเทพฯ': 'กรุงเทพมหานคร', 'กทม': 'กรุงเทพมหานคร',
              'กทม.': 'กรุงเทพมหานคร', 'โคราช': 'นครราชสีมา'}
    prov = _alias.get(prov, prov)
    rz = PROVINCE_ZONE_MAP.get(prov)
    if rz == '__BKK__':
        dist = str(district).strip() if district else ''
        return f'BKK_{dist}' if dist else 'BKK_ไม่ระบุ'
    return rz if rz else f'ไม่ระบุ_{prov}'


def classify_all_branch_zones(master_df=None):
    """
    จัดทุกสาขาใน MASTER_DATA เข้าโซนจัดส่ง (ล้วน geographic — ไม่คำนึง weight/cube)

    Returns:
        dict: {branch_code: zone_name}  (e.g. 'BKK_N', 'ZONE_A_พะเยา', ...)
        dict: zone_summary {zone_name: {'count': N, 'branches': [...]}}
    """
    if master_df is None:
        master_df = MASTER_DATA
    if master_df is None or master_df.empty:
        return {}, {}

    branch_zone_map = {}
    zone_summary = {}

    for _, row in master_df.iterrows():
        code = str(row.get('Plan Code', '')).strip().upper()
        if not code:
            continue

        province = str(row.get('จังหวัด', '') or '').strip()
        district  = str(row.get('อำเภอ', '')  or '').strip()
        subdistrict = str(row.get('ตำบล', '') or '').strip()

        # ──── กรุงเทพมหานคร: แบ่ง sub-zone ────
        _prov_alias = {'กรุงเทพฯ': 'กรุงเทพมหานคร', 'กทม': 'กรุงเทพมหานคร', 'กทม.': 'กรุงเทพมหานคร'}
        _prov_clean = _prov_alias.get(province, province)
        if _prov_clean == 'กรุงเทพมหานคร':
            lat_val = row.get('ละติจูด', 0) or 0
            lon_val = row.get('ลองติจูด', 0) or 0
            try:
                lat_val = float(lat_val)
                lon_val = float(lon_val)
            except (ValueError, TypeError):
                lat_val = lon_val = 0
            zone = get_bkk_sub_zone(lat_val, lon_val)
        else:
            # ──── จังหวัดอื่น: ใช้ LOGISTICS_ZONES ────
            zone = get_logistics_zone(_prov_clean, district, subdistrict)
            if not zone:
                zone = f'UNCLASSIFIED_{_prov_clean}' if _prov_clean else 'UNCLASSIFIED'

        branch_zone_map[code] = zone

        # สะสมสถิติ
        if zone not in zone_summary:
            zone_summary[zone] = {'count': 0, 'branches': [], 'province': _prov_clean}
        zone_summary[zone]['count'] += 1
        zone_summary[zone]['branches'].append(code)

    return branch_zone_map, zone_summary


def _build_zone_color_map(zone_summary):
    """
    สร้าง color map: {zone_name: '#rrggbb'}
    - BKK_* → 9 สีทิศ
    - โซนจังหวัดอื่น → จัดกลุ่มตามภาค แล้วใช้ palette ต่อเนื่อง
    """
    bkk_fixed = {
        'BKK_CENTER': '#C62828',
        'BKK_N':      '#1565C0',
        'BKK_NE':     '#0097A7',
        'BKK_E':      '#2E7D32',
        'BKK_SE':     '#F9A825',
        'BKK_S':      '#E65100',
        'BKK_SW':     '#6A1B9A',
        'BKK_W':      '#AD1457',
        'BKK_NW':     '#4527A0',
    }
    # Region-grouped palettes for province zones
    _region_palettes = {
        'เหนือ':       ['#0D47A1','#1565C0','#1976D2','#1E88E5','#2196F3','#42A5F5','#64B5F6','#90CAF9'],
        'อีสาน':       ['#1B5E20','#2E7D32','#388E3C','#43A047','#4CAF50','#66BB6A','#81C784','#A5D6A7'],
        'ใต้':         ['#006064','#00838F','#00ACC1','#00BCD4','#26C6DA','#4DD0E1','#80DEEA','#B2EBF2'],
        'ตะวันออก':    ['#E65100','#EF6C00','#F57C00','#FB8C00','#FF9800','#FFA726','#FFB74D','#FFCC80'],
        'กลาง':        ['#4A148C','#6A1B9A','#7B1FA2','#8E24AA','#9C27B0','#AB47BC','#BA68C8','#CE93D8'],
        'ตะวันตก':     ['#BF360C','#D84315','#E64A19','#F4511E','#FF5722','#FF7043','#FF8A65','#FFAB91'],
    }
    _fallback_colors = [
        '#607D8B','#78909C','#90A4AE','#B0BEC5',
        '#795548','#8D6E63','#A1887F','#BCAAA4',
    ]

    color_map = {}
    color_map.update(bkk_fixed)

    # group province zones by region
    from collections import defaultdict as _dd
    region_zones = _dd(list)
    for zk, zv in zone_summary.items():
        if zk.startswith('BKK_') or zk.startswith('UNCLASSIFIED'):
            continue
        prov = zv.get('province', '')
        region = get_region_name(prov) if prov else 'ไม่ระบุ'
        region_zones[region].append(zk)

    for region, zlist in region_zones.items():
        palette = _region_palettes.get(region, _fallback_colors)
        for i, zk in enumerate(sorted(zlist)):
            color_map[zk] = palette[i % len(palette)]

    # UNCLASSIFIED → grey
    for zk in zone_summary:
        if zk.startswith('UNCLASSIFIED'):
            color_map[zk] = '#9E9E9E'

    return color_map


def _build_zone_folium_map(master_df, branch_zone_map, color_map):
    """
    สร้าง Folium map แสดงสาขาทุกสาขาระบายสีตามโซน พร้อม Label โซน
    """
    if not FOLIUM_AVAILABLE:
        return None

    # Thailand center
    m = folium.Map(location=[13.0, 101.5], zoom_start=6,
                   tiles='CartoDB positron', control_scale=True)

    # Build lat/lon lookup from master_df
    lat_col = 'ละติจูด' if 'ละติจูด' in master_df.columns else None
    lon_col = 'ลองติจูด' if 'ลองติจูด' in master_df.columns else None
    name_col = 'สาขา' if 'สาขา' in master_df.columns else None
    code_col = 'Plan Code' if 'Plan Code' in master_df.columns else None

    if not (lat_col and lon_col and code_col):
        return m

    # Build coord dict  {code: (lat, lon, name)}
    _coords = {}
    for _, row in master_df.iterrows():
        code = str(row.get(code_col, '')).strip().upper()
        if not code:
            continue
        try:
            lat = float(row.get(lat_col, 0) or 0)
            lon = float(row.get(lon_col, 0) or 0)
        except (ValueError, TypeError):
            lat = lon = 0
        name = str(row.get(name_col, '') or '') if name_col else ''
        _coords[code] = (lat, lon, name)

    # Group branches by zone
    from collections import defaultdict as _ddict
    zone_branches = _ddict(list)
    for code, zone in branch_zone_map.items():
        code_upper = str(code).strip().upper()
        if code_upper in _coords:
            zone_branches[zone].append((code_upper, *_coords[code_upper]))

    # Create a FeatureGroup per zone + compute centroid for label
    zone_centroids = {}  # {zone: (lat, lon, count)}
    for zone, branches in sorted(zone_branches.items()):
        color = color_map.get(zone, '#9E9E9E')
        fg = folium.FeatureGroup(name=f"{zone} ({len(branches)})", show=True)

        lats, lons = [], []
        for code, lat, lon, name in branches:
            if lat == 0 or lon == 0:
                continue
            lats.append(lat)
            lons.append(lon)
            tooltip_html = f"<b>{code}</b><br>{name}<br><i>{zone}</i>"
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                weight=1,
                tooltip=folium.Tooltip(tooltip_html, sticky=False),
            ).add_to(fg)

        fg.add_to(m)

        if lats:
            zone_centroids[zone] = (sum(lats)/len(lats), sum(lons)/len(lons), len(lats), color)

    # Add zone label markers at centroids
    label_fg = folium.FeatureGroup(name="🏷️ Zone Labels", show=True)
    for zone, (clat, clon, cnt, color) in zone_centroids.items():
        short_label = zone.replace('ZONE_', '').replace('_', ' ')
        icon_html = (
            f'<div style="background:{color};color:#fff;border-radius:6px;'
            f'padding:3px 7px;font-size:11px;font-weight:700;white-space:nowrap;'
            f'border:1.5px solid rgba(0,0,0,.3);box-shadow:1px 1px 3px rgba(0,0,0,.25);'
            f'opacity:.92;">{short_label}</div>'
        )
        folium.Marker(
            location=[clat, clon],
            icon=folium.DivIcon(html=icon_html, icon_size=(120, 28), icon_anchor=(60, 14)),
            tooltip=f"{zone} — {cnt} สาขา",
        ).add_to(label_fg)
    label_fg.add_to(m)

    folium.LayerControl(collapsed=False, position='topright').add_to(m)
    return m


def _build_zone_excel(master_df, branch_zone_map, zone_summary, color_map):
    """
    สร้าง Excel หลายชีต:
    - สาขาทั้งหมด_โซน
    - สรุปโซน
    - กรุงเทพ_SubZone
    - หนึ่งชีตต่อภาค (จังหวัดโซน)
    """
    import io as _io
    output = _io.BytesIO()

    lat_col = 'ละติจูด' if 'ละติจูด' in master_df.columns else None
    lon_col = 'ลองติจูด' if 'ลองติจูด' in master_df.columns else None
    name_col = 'สาขา' if 'สาขา' in master_df.columns else None

    # Build main dataframe
    rows = []
    for _, row in master_df.iterrows():
        code = str(row.get('Plan Code', '')).strip().upper()
        if not code:
            continue
        zone = branch_zone_map.get(code, 'UNCLASSIFIED')
        color = color_map.get(zone, '#9E9E9E')
        name = str(row.get(name_col, '') or '') if name_col else ''
        prov = str(row.get('จังหวัด', '') or '')
        dist = str(row.get('อำเภอ', '') or '')
        subdist = str(row.get('ตำบล', '') or '')
        lat = ''
        lon = ''
        if lat_col:
            try: lat = float(row.get(lat_col, 0) or 0)
            except: lat = ''
        if lon_col:
            try: lon = float(row.get(lon_col, 0) or 0)
            except: lon = ''
        region = get_region_name(prov) if prov else ''
        zone_label = BKK_SUBZONE_NAMES.get(zone, zone)
        rows.append({
            'Plan Code': code,
            'ชื่อสาขา': name,
            'จังหวัด': prov,
            'อำเภอ': dist,
            'ตำบล': subdist,
            'ภาค': region,
            'Zone': zone,
            'Zone_Description': zone_label,
            '_hex': color,
            'ละติจูด': lat,
            'ลองติจูด': lon,
        })
    main_df = pd.DataFrame(rows)

    # Summary dataframe
    sum_rows = []
    for zone, zv in sorted(zone_summary.items(), key=lambda x: (-x[1]['count'], x[0])):
        prov = zv.get('province', '')
        region = get_region_name(prov) if prov else ''
        zone_desc = BKK_SUBZONE_NAMES.get(zone, zone)
        sum_rows.append({
            'Zone': zone,
            'คำอธิบาย': zone_desc,
            'จังหวัด': prov,
            'ภาค': region,
            'จำนวนสาขา': zv['count'],
            'สีโซน': color_map.get(zone, '#9E9E9E'),
        })
    sum_df = pd.DataFrame(sum_rows)

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        wb = writer.book

        # ─── Helper: write df to sheet with header format + zone color ───
        def _write_sheet(df, sheet_name, freeze=True, color_col=None):
            export_df = df.drop(columns=[c for c in ['_hex'] if c in df.columns], errors='ignore')
            export_df.to_excel(writer, sheet_name=sheet_name, index=False)
            ws = writer.sheets[sheet_name]
            # Header format
            hdr_fmt = wb.add_format({'bold': True, 'bg_color': '#1B5E20',
                                      'font_color': '#FFFFFF', 'border': 1,
                                      'align': 'center', 'valign': 'vcenter'})
            for col_idx, col_name in enumerate(export_df.columns):
                ws.write(0, col_idx, col_name, hdr_fmt)
                ws.set_column(col_idx, col_idx, max(12, min(40, len(str(col_name)) + 4)))
            if freeze:
                ws.freeze_panes(1, 0)
            # Color rows by zone
            if color_col and color_col in df.columns and '_hex' in df.columns:
                col_idx = list(export_df.columns).index(color_col)
                _fmt_cache = {}
                for row_idx, (_, row) in enumerate(df.iterrows(), start=1):
                    hex_c = str(row.get('_hex', '#FFFFFF')).replace('#', '')
                    if hex_c not in _fmt_cache:
                        _fmt_cache[hex_c] = wb.add_format({
                            'bg_color': f'#{hex_c}', 'font_color': '#FFFFFF',
                            'bold': True, 'border': 1, 'align': 'center'
                        })
                    ws.write(row_idx, col_idx, row.get(color_col, ''), _fmt_cache[hex_c])

        # Sheet 1: ทุกสาขา
        _write_sheet(main_df, 'สาขาทั้งหมด_โซน', color_col='Zone')

        # Sheet 2: สรุปโซน
        _write_sheet(sum_df, 'สรุปโซนทั้งหมด', color_col='Zone')

        # Sheet 3: กรุงเทพ sub-zone detail
        bkk_df = main_df[main_df['Zone'].str.startswith('BKK_', na=False)].copy()
        if not bkk_df.empty:
            _write_sheet(bkk_df, 'กรุงเทพ_SubZone', color_col='Zone')

        # Sheet 4-N: province zones by region
        from collections import defaultdict as _ddef
        region_map_ex = _ddef(list)
        for _, row in main_df.iterrows():
            z = row.get('Zone', '')
            if z.startswith('BKK_') or z.startswith('UNCLASSIFIED'):
                continue
            region_map_ex[row.get('ภาค', 'ไม่ระบุ')].append(row)
        for region_name, region_rows in sorted(region_map_ex.items()):
            sheet_df = pd.DataFrame(region_rows)
            safe_name = f"โซน_{region_name}"[:31]
            _write_sheet(sheet_df, safe_name, color_col='Zone')

        # Sheet last: UNCLASSIFIED
        unc_df = main_df[main_df['Zone'].str.startswith('UNCLASSIFIED', na=False)].copy()
        if not unc_df.empty:
            _write_sheet(unc_df, 'ไม่ระบุโซน')

    output.seek(0)
    return output.getvalue()


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
    if not zone_name:
        return 999
    # LOGISTICS_ZONES format (ZONE_A_พะเยา)
    if zone_name in LOGISTICS_ZONES:
        return LOGISTICS_ZONES[zone_name]['priority']
    # zone_viewer format (เหนือ_จังหวัด_อำเภอ / BKK_เขต)
    _ZV_PRIORITY = {
        'เหนือ': 5, 'อีสาน': 12, 'ใต้': 20, 'ตะวันออก': 16,
        'ตะวันตก': 18, 'ปริมณฑล': 80, 'BKK': 90,
    }
    prefix = zone_name.split('_')[0] if '_' in zone_name else zone_name
    return _ZV_PRIORITY.get(prefix, 50)

def get_zone_highway(zone_name):
    """
    ดึงทางหลวงหลักของโซน
    
    Returns:
        str: เช่น 'สาย 1 (พหลโยธิน)', 'สาย 2 (มิตรภาพ)'
    """
    if not zone_name:
        return ''
    if zone_name in LOGISTICS_ZONES:
        return LOGISTICS_ZONES[zone_name].get('highway', '')
    # zone_viewer format — derive highway from zone prefix
    _ZV_HW = {
        'เหนือ': '1/11', 'อีสาน': '2/24', 'ใต้': '4',
        'ตะวันออก': '3', 'ตะวันตก': '32/4', 'ปริมณฑล': '',
    }
    prefix = zone_name.split('_')[0] if '_' in zone_name else zone_name
    return _ZV_HW.get(prefix, '')

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
        
        # ──────────────────────────────────────────────────────────
        # 🔑 Normalize column names: ลบ space, newline, tab ที่ Sheets ใส่มา
        # (เช่น 'Max       x\nTruckType' → 'MaxTruckType')
        # ──────────────────────────────────────────────────────────
        import re as _re_col
        df_from_sheets.columns = [
            _re_col.sub(r'[\s\n\r\t]+', '', str(c)) for c in df_from_sheets.columns
        ]

        # ทำความสะอาด Plan Code
        if 'Plan Code' in df_from_sheets.columns:
            df_from_sheets['Plan Code'] = df_from_sheets['Plan Code'].astype(str).str.strip().str.upper()
            df_from_sheets = df_from_sheets[df_from_sheets['Plan Code'] != '']
        elif 'PlanCode' in df_from_sheets.columns:   # หลัง normalize อาจกลายเป็น PlanCode
            df_from_sheets.rename(columns={'PlanCode': 'Plan Code'}, inplace=True)
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

# ──────────────────────────────────────────────────────────────────
# 🔑 MASTER_DATA_DICT  — Plan Code (upper) เป็น PK → O(1) lookup
# ── สร้างทุกครั้งที่ MASTER_DATA โหลดใหม่ ──
# ──────────────────────────────────────────────────────────────────
def _build_master_dict(md: 'pd.DataFrame') -> dict:
    """สร้าง dict {plan_code_upper: row_dict} จาก MASTER_DATA"""
    if md is None or md.empty or 'Plan Code' not in md.columns:
        return {}
    result = {}
    truck_cols_priority = [
        'MaxTruckType', 'Max Truck Type', 'MaxVehicle', 'Max Vehicle',
        'รถสูงสุด', 'Max_Truck_Type', 'max_truck', 'MaxTruck',
        'ข้อจำกัดรถ', 'Truck', 'truck_type', 'TruckType',
        'ประเภทรถ', 'Vehicle', 'vehicle_type', 'VehicleType'
    ]
    found_truck_col = next((c for c in truck_cols_priority if c in md.columns), None)
    for _, row in md.iterrows():
        code = str(row.get('Plan Code', '')).strip().upper()
        if not code:
            continue
        entry = {
            'max_truck': '6W',   # default
            '_row': row.to_dict(),
        }
        if found_truck_col:
            raw = str(row.get(found_truck_col, '')).strip().upper()
            if raw in ('4W', '4 W', '4-W'):
                entry['max_truck'] = '4W'
            elif raw in ('JB', 'J B', 'J-B', '4WJ', '4WJ '):
                entry['max_truck'] = 'JB'
            elif raw in ('6W', '6 W', '6-W'):
                entry['max_truck'] = '6W'
        result[code] = entry
    safe_print(f"🔑 MASTER_DATA_DICT: {len(result)} สาขา (PK=Plan Code) | truck col='{found_truck_col}'")
    return result

MASTER_DATA_DICT: dict = _build_master_dict(MASTER_DATA)

# ══════════════════════════════════════════════════════════════════════════════
# 🗺️ BRANCH_ZONES_CACHE — โหลดจาก branch_zones.json ที่ zone_viewer.py สร้าง
# Format: {branch_code_upper: zone_string}  เช่น "NY00" → "เหนือ_เชียงใหม่_เมือง"
# ══════════════════════════════════════════════════════════════════════════════
def _load_branch_zones() -> dict:
    """โหลด branch_zones.json ที่ zone_viewer.py export ไว้"""
    _path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'branch_zones.json')
    if not os.path.exists(_path):
        safe_print("⚠️ ไม่พบ branch_zones.json — ใช้ LOGISTICS_ZONES แทน (รัน zone_viewer.py ก่อน)")
        return {}
    try:
        with open(_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        safe_print(f"🗺️ โหลด branch_zones.json: {len(data):,} สาขา")
        return {str(k).upper(): v for k, v in data.items()}
    except Exception as e:
        safe_print(f"⚠️ โหลด branch_zones.json ล้มเหลว: {e}")
        return {}

BRANCH_ZONES_CACHE: dict = _load_branch_zones()

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
    """ดึงรถใหญ่สุดที่สาขานี้รองรับ
    ใช้ MASTER_DATA_DICT (PK=Plan Code) เพื่อ O(1) lookup
    """
    branch_code_str = str(branch_code).strip().upper()

    # ── 1. Fast path: ค้นจาก MASTER_DATA_DICT โดยตรง ──
    if MASTER_DATA_DICT:
        entry = MASTER_DATA_DICT.get(branch_code_str)

        # ── 2. ถ้าไม่พบ ลอง prefix-strip fallback ──
        if entry is None:
            prefixes = ['PUN-', 'MAX-', 'MM-', 'PT-']
            code_clean = branch_code_str
            for p in prefixes:
                if code_clean.startswith(p):
                    code_clean = code_clean[len(p):]
                    break
            if code_clean != branch_code_str:
                entry = MASTER_DATA_DICT.get(code_clean)
            # ลอง match แบบ strip prefix จากฝั่ง master
            if entry is None:
                for mk, mv in MASTER_DATA_DICT.items():
                    mk_clean = mk
                    for p in prefixes:
                        if mk_clean.startswith(p):
                            mk_clean = mk_clean[len(p):]
                            break
                    if mk_clean == code_clean:
                        entry = mv
                        break

        if entry is not None:
            return entry['max_truck']   # '4W' / 'JB' / '6W'

    # ── 3. Legacy fallback: scan DataFrame (ถ้า dict ยังไม่ build) ──
    if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
        master_codes = MASTER_DATA['Plan Code'].str.strip().str.upper()
        branch_row = MASTER_DATA[master_codes == branch_code_str]
        if not branch_row.empty:
            possible_cols = [
                'MaxTruckType', 'Max Truck Type', 'MaxVehicle', 'Max Vehicle',
                'รถสูงสุด', 'Max_Truck_Type', 'max_truck', 'MaxTruck',
                'ข้อจำกัดรถ', 'Truck', 'truck_type', 'TruckType',
                'ประเภทรถ', 'Vehicle', 'vehicle_type', 'VehicleType'
            ]
            for col in possible_cols:
                if col in branch_row.columns and pd.notna(branch_row.iloc[0][col]):
                    raw = str(branch_row.iloc[0][col]).strip().upper()
                    if raw in ('4W', '4 W', '4-W'):
                        return '4W'
                    elif raw in ('JB', 'J B', 'J-B', '4WJ', '4WJ '):
                        return 'JB'
                    elif raw in ('6W', '6 W', '6-W'):
                        return '6W'

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

def predict_trips(test_df, model_data, punthai_buffer=1.0, maxmart_buffer=1.10, fleet_limits=None):
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
    # 🗺️ ลำดับความสำคัญ:
    #   1. BRANCH_ZONES_CACHE (จาก zone_viewer.py) — ถ้ามี branch_zones.json
    #   2. PROVINCE_ZONE_MAP fallback — ใช้ logic เดียวกับ zone_viewer.py
    def _zone_from_prov_dist(prov: str, dist: str) -> str:
        """Fallback zone ใช้ logic เดียวกับ zone_viewer.py load_and_classify()"""
        prov = str(prov or '').strip()
        for _alias, _full in [("กรุงเทพฯ","กรุงเทพมหานคร"),("กทม","กรุงเทพมหานคร"),("กทม.","กรุงเทพมหานคร"),("โคราช","นครราชสีมา")]:
            if prov == _alias: prov = _full; break
        dist = str(dist or '').strip()
        rz = PROVINCE_ZONE_MAP.get(prov)
        if rz == '__BKK__':
            return f'BKK_{dist}' if dist else 'BKK_ไม่ระบุ'
        if rz:
            parts = rz.split('_', 1)
            prefix, prov_short = parts[0], parts[1] if len(parts) > 1 else rz
            return f'{prefix}_{prov_short}_{dist}' if dist else rz
        return f'ไม่ระบุ_{prov}' if prov else 'ไม่ระบุ'

    def _get_zone_for_row(row):
        code = str(row.get('Code', '')).strip().upper()
        if code and BRANCH_ZONES_CACHE:
            z = BRANCH_ZONES_CACHE.get(code)
            if z:
                return z
        return _zone_from_prov_dist(row['_province'], row['_district'])

    df['_logistics_zone'] = df.apply(_get_zone_for_row, axis=1)
    df['_zone_priority'] = df['_logistics_zone'].apply(get_zone_priority)
    df['_zone_highway'] = df['_logistics_zone'].apply(get_zone_highway)
    # 🎯 Province Zone (zone_viewer.py system) — ใช้ป้องกันกระโดดข้ามจังหวัด
    df['_prov_zone'] = df.apply(
        lambda row: get_prov_zone(row['_province'], row['_district']), axis=1
    )
    
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
    
    # 🎯 Sort: ใช้ PROVINCE ZONE เป็นหลัก (ระบบเดียวกับ zone_viewer.py)
    # หลักการ: สาขาในจังหวัด/ภาคเดียวกันอยู่ติดกัน ป้องกันกระโดดข้ามจังหวัด
    # 1. Province Zone (BKK_เขต / ภาค_จังหวัด) — สาขากลุ่มเดียวกันอยู่ติดกัน
    # 2. ระยะทางจาก DC (ไกลมาใกล้) - LIFO
    # 3. จังหวัด/อำเภอ/ตำบล - จัดกลุ่มในพื้นที่เดียวกัน
    df = df.sort_values(
        [
            '_prov_zone',           # 1. 🗺️ Province Zone — BKK_เขต หรือ ภาค_จังหวัด
            '_distance_from_dc',    # 2. ระยะทางไกลก่อน (LIFO)
            '_province',            # 3. จังหวัดเดียวกัน
            '_district',            # 4. อำเภอเดียวกัน
            '_subdistrict',         # 5. ตำบลเดียวกัน
            '_vehicle_priority'     # 6. ข้อจำกัดรถ (secondary)
        ],
        ascending=[
            True,   # Province Zone เรียง A-Z (จัดกลุ่มจังหวัดเดียวกัน)
            False,  # ไกลมาใกล้ (LIFO)
            True,   # จังหวัดเรียง A-Z
            True,   # อำเภอเรียง A-Z
            True,   # ตำบลเรียง A-Z
            True    # ข้อจำกัดมากก่อน
        ]
    ).reset_index(drop=True)
    
    safe_print(f"📊 DEBUG: Province zones = {df['_prov_zone'].unique().tolist()}")
    
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

    # ─── Runtime Nearby Groups (≤10km) ──────────────────────────────────────
    # สร้าง map สาขา → สาขาที่อยู่ใกล้กัน ≤10km จากข้อมูลที่ upload
    # เพื่อให้สาขาในรัศมี 10km ไปในทริปเดียวกัน ไม่ถูกแยก
    _NEARBY_GROUP_KM = 10.0   # รัศมีรวมกลุ่ม (km)
    _df_codes_upper = {str(c).strip().upper() for c in df['Code'].tolist()}

    # สร้าง _rt_same_loc จาก NEARBY_BRANCHES (pre-computed) + haversine fallback
    _rt_same_loc: dict = {}   # code_upper → [code_upper, ...]

    # Pass 1: ใช้ NEARBY_BRANCHES (เร็ว)
    for _nc in _df_codes_upper:
        if _nc in NEARBY_BRANCHES:
            _nearby_in_run = [
                nb_code for nb_code, nb_dist in NEARBY_BRANCHES[_nc]
                if nb_dist <= _NEARBY_GROUP_KM and nb_code in _df_codes_upper
            ]
            if _nearby_in_run:
                _rt_same_loc[_nc] = [_nc] + _nearby_in_run

    # Pass 2: สาขาที่ไม่อยู่ใน NEARBY_BRANCHES → ใช้ haversine จาก df พิกัด
    _df_coord_map = {}
    for _, _sr in df.iterrows():
        _slat = float(_sr.get('_lat', 0) or 0)
        _slon = float(_sr.get('_lon', 0) or 0)
        if _slat > 0 and _slon > 0:
            _df_coord_map[str(_sr['Code']).strip().upper()] = (_slat, _slon)

    for _nc in _df_codes_upper:
        if _nc not in _rt_same_loc and _nc in _df_coord_map:
            _nlat, _nlon = _df_coord_map[_nc]
            _nearby_hv = [
                _oc for _oc, (_olat, _olon) in _df_coord_map.items()
                if _oc != _nc and haversine_distance(_nlat, _nlon, _olat, _olon) <= _NEARBY_GROUP_KM
            ]
            if _nearby_hv:
                _rt_same_loc[_nc] = [_nc] + _nearby_hv

    _rt_grp_count = sum(1 for v in _rt_same_loc.values() if len(v) > 1)
    safe_print(f"📍 Runtime nearby group (≤{_NEARBY_GROUP_KM:.0f}km): {_rt_grp_count} สาขามีเพื่อนร่วมทริป จาก {len(_df_codes_upper)} สาขา")

    def get_group_branches_rt(code: str) -> list:
        """รวม precomputed group (≤200m) + runtime nearby (≤10km)"""
        code_upper = str(code).strip().upper()
        # precomputed group (≤200m จาก branch_groups.json)
        grp = list(get_group_branches(code_upper))
        grp_upper = {str(c).strip().upper() for c in grp}
        # เพิ่มสาขาในรัศมี 10km (runtime)
        for _rt_c in _rt_same_loc.get(code_upper, []):
            if _rt_c not in grp_upper:
                grp.append(_rt_c)
                grp_upper.add(_rt_c)
        return grp

    # 🧠 AI LEARNING: โหลด pair_freq สำหรับ affinity boost
    _ai_pair_freq = load_trip_history()
    _ai_active = bool(_ai_pair_freq)
    if _ai_active:
        safe_print(f"🧠 AI Learning: โหลด {len(_ai_pair_freq):,} pair records")

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
    _current_zone = ''   # 🎯 Zone-complete: ล็อคโซนที่กำลังจัดอยู่ → จัดให้ครบโซนก่อนขยาย
    
    while unassigned:
        # หาสาขาที่ยังไม่ได้จัด
        unassigned_df = df[df['Code'].isin(unassigned)]
        if unassigned_df.empty:
            break
        
        # 🎯 Zone-complete: อยู่ในโซนปัจจุบันก่อนจนหมด แล้วค่อยย้ายโซนใหม่
        farthest_row = None
        if _current_zone:
            _cz_df = unassigned_df[unassigned_df['_prov_zone'] == _current_zone]
            if not _cz_df.empty:
                _cz_df = _cz_df.sort_values(
                    ['_vehicle_rank', '_distance_from_dc'], ascending=[True, False]
                )
                farthest_row = _cz_df.iloc[0]
        if farthest_row is None:
            # โซนปัจจุบันหมดแล้ว หรือยังไม่ได้เริ่ม → เลือกโซนถัดไป (zone priority)
            unassigned_df = unassigned_df.sort_values(
                ['_zone_priority', '_vehicle_rank', '_distance_from_dc'], 
                ascending=[True, True, False]
            )
            farthest_row = unassigned_df.iloc[0]
            _current_zone = farthest_row.get('_prov_zone', '')   # ล็อคโซนใหม่
        
        # เลือกสาขาแรก (ไกลสุด + ข้อจำกัดมากสุด)
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
        trip_prov_zone = farthest_row.get('_prov_zone', '')              # 🗺️ PROVINCE ZONE (zone_viewer system)
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
        # 🎯 ดึงสาขาทั้งกลุ่ม (≤10km = จุดส่งใกล้เคียง) ของสาขาแรก
        # ใช้ get_group_branches_rt: รวม precomputed(≤200m) + runtime nearby(≤10km)
        start_group_codes = get_group_branches_rt(start_code)
        start_group_unassigned = [c for c in start_group_codes if c in unassigned or c.upper() in [str(x).upper() for x in unassigned]]
        if not start_group_unassigned:
            start_group_unassigned = [start_code]

        # 🔢 เรียงสมาชิกกลุ่ม start จากใกล้ start_code ก่อน (สาขาไกลรอ greedy loop)
        def _dist_from_start(code):
            cu = str(code).strip().upper()
            if cu in NEARBY_BRANCHES:
                pass
            if cu == str(start_code).strip().upper():
                return 0.0
            if cu in NEARBY_BRANCHES:
                for _nb, _nd in NEARBY_BRANCHES[cu]:
                    if _nb == str(start_code).strip().upper():
                        return _nd
            sr = df[df['Code'].apply(lambda x: str(x).upper() == cu)]
            if not sr.empty and start_lat > 0 and start_lon > 0:
                _slat = float(sr.iloc[0].get('_lat', 0) or 0)
                _slon = float(sr.iloc[0].get('_lon', 0) or 0)
                if _slat > 0 and _slon > 0:
                    return haversine_distance(_slat, _slon, start_lat, start_lon)
            return 999.0
        start_group_unassigned.sort(key=_dist_from_start)

        # capacity limit สำหรับ start_group: ใช้ 6W max (รถใหญ่สุด) * buffer
        _sg_is_pt = all(branch_bu_cache.get(str(c).strip().upper(), False) for c in [start_code])
        _sg_buf = punthai_buffer if _sg_is_pt else maxmart_buffer
        _sg_lim = (PUNTHAI_LIMITS if _sg_is_pt else LIMITS)['6W']
        _sg_max_w = _sg_lim['max_w'] * _sg_buf
        _sg_max_c = _sg_lim['max_c'] * _sg_buf
        _sg_max_d = _sg_lim['max_drops']

        # คำนวณ weight/cube รวมทั้งกลุ่ม (พร้อม capacity check สำหรับสมาชิก 10km)
        trip_codes = []
        trip_weight = 0
        trip_cube = 0
        for gc in start_group_unassigned:
            gc_row = df[df['Code'].apply(lambda x: str(x).upper() == str(gc).upper())]
            if not gc_row.empty:
                actual_code = gc_row.iloc[0]['Code']
                _gc_w = gc_row.iloc[0]['Weight']
                _gc_c = gc_row.iloc[0]['Cube']

                # 🚫 Capacity check สำหรับสมาชิกที่ไกลกว่า 0.3km (ไม่ใช่พิกัดเดียวกัน)
                _gc_is_start = (str(gc).upper() == str(start_code).upper())
                if not _gc_is_start:
                    _gc_dist_s = _dist_from_start(str(gc).upper())
                    if _gc_dist_s > 0.3:  # ไกลกว่า 300m → ตรวจ capacity และ region
                        # capacity check
                        if (trip_weight + _gc_w > _sg_max_w or
                                trip_cube + _gc_c > _sg_max_c or
                                len(trip_codes) + 1 > _sg_max_d):
                            safe_print(f"      📦 START-GROUP SKIP (เต็ม): {actual_code} (+{_gc_w:.0f}kg) → รอ greedy")
                            continue  # ไม่เพิ่ม — greedy loop จะหยิบทีหลัง
                        # region guard
                        _sg_prov = str(gc_row.iloc[0].get('_province', '') or '')
                        _sg_region = get_region_name(_sg_prov) if _sg_prov else ''
                        if trip_original_region and trip_original_region not in ('', 'ไม่ระบุ'):
                            if _sg_region and _sg_region not in ('', 'ไม่ระบุ') and _sg_region != trip_original_region:
                                safe_print(f"      🛑 START-GROUP GUARD: ตัด {actual_code} ภาค {_sg_region} ≠ {trip_original_region} (ห่าง {_gc_dist_s:.1f}km)")
                                continue
                trip_codes.append(actual_code)
                trip_weight += _gc_w
                trip_cube += _gc_c
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
        
        # 🛣️ ROAD CORRIDOR (ตามเส้นทางถนนจริง ไม่ใช้มุมเส้นตรง):
        # ใช้ highway number จาก LOGISTICS_ZONES เป็น "ถนนเส้นเดียวกัน" — trip_original_hws lock ไว้แล้ว
        trip_max_dist_dc = float(farthest_row.get('_distance_from_dc', 0) or 0)

        # หา allowed vehicles จาก constraints (จำกัดตาม trip_max_vehicle)
        trip_allowed = get_allowed_from_codes(trip_codes, ['4W', 'JB', '6W'])
        trip_is_punthai = all(branch_bu_cache.get(c, False) for c in trip_codes)
        
        # ─── ระยะ reach สูงสุดที่ยอมขยายออกจากทริป (km) ───
        # ปรับได้: ยิ่งมากยิ่ง "ดึงโซนใกล้เคียง" แต่อาจรวมสาขาไกลเกินไป
        _MAX_EXPAND_KM     = 80   # ขยายสูงสุด 80km จากสาขาใดๆ ในทริป → adjacent zones
        _PREFERRED_NEAR_KM = 40   # zone เดียวกัน+ใกล้กว่านี้ = ลองก่อน

        # 2️⃣ Greedy: หาสาขาใกล้สุดมาเติมจนเต็ม buffer
        while unassigned:
            remaining_df = df[df['Code'].isin(unassigned)].copy()
            if remaining_df.empty:
                break
            
            # ✅ รีเซ็ต same_zone_df ทุก iteration ป้องกัน stale value จาก iteration ก่อน
            same_zone_df = None
            filter_level = ""

            # 🌐 BUILD REACH SET — สาขาทั้งหมดที่อยู่ในระยะ MAX_EXPAND_KM จากสาขาใดๆ ในทริป
            # → ครอบคลุมโซนรอบข้างติดกันโดยอัตโนมัติ (adjacent zones) ไม่กระโดดข้าม
            unassigned_upper_set_vc = {str(c).strip().upper() for c in unassigned}
            reach_codes   = set()   # สาขาในระยะ MAX_EXPAND_KM
            very_close_codes = set()  # < 15km (fast path สำหรับ BKK/ใกล้มาก)
            ultra_close_codes = set() # < 8km  (bypass highway filter)

            # Pre-compute valid coords ของสาขาในทริป
            _trip_coords_reach = []
            for _tc in trip_codes:
                _tr = df[df['Code'] == _tc]
                if not _tr.empty:
                    _tlat = _tr.iloc[0]['_lat']; _tlon = _tr.iloc[0]['_lon']
                    if _tlat and _tlat > 0 and _tlon and _tlon > 0:
                        _trip_coords_reach.append((_tlat, _tlon))

            for _tc in trip_codes:
                _tc_upper = str(_tc).strip().upper()
                if _tc_upper in NEARBY_BRANCHES:
                    for nearby_code, dist in NEARBY_BRANCHES[_tc_upper]:
                        if nearby_code in unassigned_upper_set_vc:
                            if dist <= _MAX_EXPAND_KM:
                                reach_codes.add(nearby_code)
                            if dist < 15.0:
                                very_close_codes.add(nearby_code)
                            if dist < 8.0:
                                ultra_close_codes.add(nearby_code)

            # Fallback: ถ้า NEARBY_BRANCHES ไม่ครอบ MAX_EXPAND_KM → ใช้ haversine
            if _trip_coords_reach:
                for _, _rrow in remaining_df.iterrows():
                    _rc_upper = str(_rrow['Code']).strip().upper()
                    if _rc_upper in reach_codes:
                        continue
                    _rlat = _rrow['_lat']; _rlon = _rrow['_lon']
                    if _rlat and _rlat > 0 and _rlon and _rlon > 0:
                        _min_d = min(haversine_distance(_rlat, _rlon, _tlat, _tlon)
                                     for _tlat, _tlon in _trip_coords_reach)
                        if _min_d <= _MAX_EXPAND_KM:
                            reach_codes.add(_rc_upper)
                        if _min_d < 15.0:
                            very_close_codes.add(_rc_upper)
                        if _min_d < 8.0:
                            ultra_close_codes.add(_rc_upper)

            # 📌 PRIMARY FILTER: จัดโซนเดียวกันให้ครบก่อน แล้วค่อยขยายไปโซนรอบข้าง
            # ขั้นตอน 1️⃣: ลอง same prov_zone + reach → จัดโซนให้ครบก่อน
            if reach_codes and trip_prov_zone:
                _sz_reach_df = remaining_df[
                    remaining_df['Code'].apply(lambda x: str(x).strip().upper() in reach_codes) &
                    (remaining_df['_prov_zone'] == trip_prov_zone)
                ].copy()
                if not _sz_reach_df.empty:
                    same_zone_df = _sz_reach_df
                    filter_level = f"same_zone+reach({_MAX_EXPAND_KM}km)"
            # ขั้นตอน 2️⃣: โซนเดิมหมดแล้ว → ขยายไปโซนรอบข้าง (adjacent zones)
            if same_zone_df is None and reach_codes:
                reach_df = remaining_df[remaining_df['Code'].apply(
                    lambda x: str(x).strip().upper() in reach_codes)].copy()
                if not reach_df.empty:
                    same_zone_df = reach_df
                    filter_level = f"reach({_MAX_EXPAND_KM}km)"
            
            # Fallback: ถ้าไม่มีสาขาในระยะ reach → ใช้ province-zone เดิม
            if same_zone_df is None and trip_prov_zone:
                pzone_df = remaining_df[remaining_df['_prov_zone'] == trip_prov_zone].copy()
                if not pzone_df.empty:
                    same_zone_df = pzone_df
                    filter_level = "province-zone(fallback)"

            # (reach filter computed above — same_zone_df already set if reach_codes found)
            # ถ้า same_zone_df ยังว่าง → fallback เต็มรูปแบบ: province → logistic zone → highway
            if same_zone_df is None:
                # จังหวัดเดียวกัน
                _prov_set = {p for p in [trip_province, trip_original_province] if p}
                if _prov_set:
                    province_df = remaining_df[remaining_df['_province'].isin(_prov_set)].copy()
                    if not province_df.empty:
                        same_zone_df = province_df
                        filter_level = "จังหวัด(fallback)"

            # Province-zone fallback
            if same_zone_df is None and trip_prov_zone:
                pzone_df = remaining_df[remaining_df['_prov_zone'] == trip_prov_zone].copy()
                if not pzone_df.empty:
                    same_zone_df = pzone_df
                    filter_level = "province-zone(fallback)"
                    trip_subdistricts = set()
                    trip_districts = set()

            # Logistics-zone fallback
            if same_zone_df is None and trip_logistics_zone:
                zone_df = remaining_df[remaining_df['_logistics_zone'] == trip_logistics_zone].copy()
                if not zone_df.empty:
                    same_zone_df = zone_df
                    filter_level = "โซน(fallback)"
                    trip_subdistricts = set()
                    trip_districts = set()

            # หมดสาขา reach → ปิดทริป
            if same_zone_df is None:
                safe_print(f"      🛑 ไม่มีสาขาในระยะ {_MAX_EXPAND_KM}km แล้ว → ปิดทริป {trip_counter}")
                break

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
            
            # คำนวณระยะทาง — ใช้ NEARBY_BRANCHES cache ถ้ามี
            # fallback: haversine จาก branch ในทริปที่ใกล้ที่สุด (ไม่ใช่แค่ branch สุดท้าย)
            # → รองรับกรณีที่ branch สุดท้ายอยู่ปลายทาง (เช่น เชียงคำ) แต่ trip มี branch อื่นที่ใกล้กว่า
            _last_code_d = trip_codes[-1]
            _last_row_d = df[df['Code'] == _last_code_d].iloc[0]
            _last_lat_d, _last_lon_d = _last_row_d['_lat'], _last_row_d['_lon']

            # Pre-compute valid coords of all branches in trip (for nearest-branch fallback)
            _trip_valid_coords = []
            for _tc_coord in trip_codes:
                _tr = df[df['Code'] == _tc_coord]
                if not _tr.empty:
                    _tlat = _tr.iloc[0]['_lat']
                    _tlon = _tr.iloc[0]['_lon']
                    if _tlat and _tlat > 0 and _tlon and _tlon > 0:
                        _trip_valid_coords.append((_tlat, _tlon))

            def _dist_for_row(row):
                cu = str(row['Code']).strip().upper()
                if cu in candidate_distances:
                    return candidate_distances[cu]
                # fallback: haversine จาก branch ในทริปที่ใกล้ที่สุด (nearest-branch)
                clat, clon = row['_lat'], row['_lon']
                if clat and clat > 0 and clon and clon > 0 and _trip_valid_coords:
                    return min(haversine_distance(clat, clon, tlat, tlon)
                               for tlat, tlon in _trip_valid_coords)
                return 999

            same_zone_df['_dist_to_trip'] = same_zone_df.apply(_dist_for_row, axis=1)
            
            # เรียงตาม จังหวัดเดียวกันก่อน + priority + distance
            _trip_provs_sort = {p for p in [trip_province, trip_original_province] if p}
            same_zone_df['_same_prov_rank'] = same_zone_df['_province'].apply(
                lambda p: 0 if p in _trip_provs_sort else 1
            )

            # 🧠 AI AFFINITY RANK: สาขาที่เคยอยู่ทริปเดียวกันจะได้ rank ดีกว่า
            if _ai_active and trip_codes:
                def _affinity(cand_code):
                    total = 0
                    for tc in trip_codes:
                        a, b = (tc, cand_code) if tc < cand_code else (cand_code, tc)
                        total += _ai_pair_freq.get(f"{a}|{b}", 0)
                    return total
                same_zone_df['_affinity'] = same_zone_df['Code'].apply(_affinity)
                # rank 0 = มีประวัติคู่กัน, rank 1 = ไม่มีประวัติ
                same_zone_df['_affinity_rank'] = same_zone_df['_affinity'].apply(lambda x: 0 if x > 0 else 1)
                sort_cols = ['_same_prov_rank', '_affinity_rank', '_priority', '_dist_to_trip']
            else:
                sort_cols = ['_same_prov_rank', '_priority', '_dist_to_trip']

            same_zone_df = same_zone_df.sort_values(sort_cols)
            # เช็คว่ายังมีสาขาจังหวัดเดียวกัน **ภายใน 25km** ไหม
            # [FIX] ใช้ distance cap 25km: มีสาขาใกล้จริงๆ ถึงจะบล็อก cross-province
            # ป้องกันกรณีที่สาขาจังหวัดเดียวกันอยู่ไกลมาก (>25km) แต่มีสาขาต่างจังหวัดที่ใกล้กว่า
            _same_prov_close = (
                not same_zone_df[(same_zone_df['_same_prov_rank'] == 0) & (same_zone_df['_dist_to_trip'] <= 25)].empty
            ) if _trip_provs_sort else False
            
            found_candidate = False
            
            for _, candidate_row in same_zone_df.iterrows():
                candidate_code = candidate_row['Code']
                candidate_dist = candidate_row['_dist_to_trip']
                
                # 🛣️ ROAD CORRIDOR CHECK (ตามถนนจริง ไม่ใช้มุมเส้นตรง)
                # เงื่อนไข 1: highway ต้องมี overlap กับทริป (ใช้ถนนเส้นเดียวกัน)
                # เงื่อนไข 2: ไม่ไกลจาก DC เกิน endpoint ของทริป + 30km
                _cand_dist_dc = float(candidate_row.get('_distance_from_dc', 0) or 0)
                _c_hw_corr = str(candidate_row.get('_zone_highway', '') or '')
                _c_hws_corr = set(_c_hw_corr.split('/')) - {''} if _c_hw_corr else set()
                _trip_hws_corr = trip_original_hws  # lock ไว้ตั้งแต่ต้นทริป
                if _trip_hws_corr and _c_hws_corr and _cand_dist_dc > 80:
                    if not _trip_hws_corr.intersection(_c_hws_corr):
                        continue  # ไม่มี highway ร่วมกัน → คนละเส้นทาง → ยกเป็นทริปตัวเอง
                if trip_max_dist_dc > 0 and _cand_dist_dc > trip_max_dist_dc + 30:
                    continue  # ไกลจาก DC เกิน endpoint + 30km → ยกเป็นทริปตัวเอง

                # �🚫 ถ้าไกลเกิน limit → ข้ามไปสาขาถัดไป (อย่า break — priority อื่นอาจใกล้กว่า)
                # ใช้ dynamic distance limit ตาม utilization: รถยิ่งว่าง → radius ยิ่งกว้าง (ไม่ปล่อยว่าง)
                _cand_prov_s6 = str(candidate_row.get('_province', '')).strip()
                _trip_prov_s6 = str(trip_province or trip_original_province or '').strip()
                # คำนวณ utilization ปัจจุบัน (relative to 6W max)
                _util_6w_s6 = max(
                    trip_weight / LIMITS['6W']['max_w'],
                    trip_cube / LIMITS['6W']['max_c']
                )
                if _cand_prov_s6 and _trip_prov_s6 and _cand_prov_s6 == _trip_prov_s6:
                    # จังหวัดเดียวกัน: ขยาย 100km เมื่อรถ < 40%, 65km ปกติ
                    _dist_limit_s6 = 100 if _util_6w_s6 < 0.40 else 65
                else:
                    # ต่างจังหวัด same zone: ขยาย 45km เมื่อรถ < 40%, 25km ปกติ
                    _dist_limit_s6 = 45 if _util_6w_s6 < 0.40 else 25
                if candidate_dist > _dist_limit_s6:
                    continue
                
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
                # 🔒 กรุงเทพฯ isolation (Step 6 greedy): ห้ามกรุงเทพฯ ปนกับจังหวัดอื่น ไม่ว่าจะ zone/highway เดียวกัน
                _BKK = 'กรุงเทพมหานคร'
                if ((_c_prov == _BKK and trip_original_province not in ('', None) and trip_original_province != _BKK) or
                        (trip_original_province == _BKK and _c_prov not in ('', None) and _c_prov != _BKK)):
                    safe_print(f"      🚫 BKK isolation step6: ตัด {candidate_code} ({_c_prov}) ≠ trip ({trip_original_province})")
                    continue
                # 🔒 ZONE_NEARBY strict (Step 6 greedy): ห้ามรวม ZONE_NEARBY ต่างจังหวัด
                _is_trip_nearby_s6 = str(trip_logistics_zone or '').startswith('ZONE_NEARBY_')
                _is_cand_nearby_s6 = str(_c_zone or '').startswith('ZONE_NEARBY_')
                if (_is_trip_nearby_s6 or _is_cand_nearby_s6):
                    _trip_prov_now = trip_province or trip_original_province
                    if _c_prov and _trip_prov_now and _c_prov != _trip_prov_now:
                        safe_print(f"      🚫 NEARBY strict step6: ตัด {candidate_code} ({_c_prov}/{_c_zone}) ≠ trip ({_trip_prov_now}/{trip_logistics_zone})")
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
                    # 🌟 Proximity override: ถ้าสาขาอยู่ใกล้มาก (<10km) และภาคเดียวกัน → ข้ามข้อจำกัด zone
                    # แก้ปัญหา "กระโดดข้ามสาขาใกล้ไปไกล" เพราะสาขาใกล้อยู่ต่าง logistics zone
                    if candidate_dist < 10.0 and _region_compat:
                        pass  # อนุญาต — proximity overrides zone restriction
                    else:
                        safe_print(f"      🚫 step6 skip {candidate_code} ({_c_prov}/{_c_zone}) ≠ trip ({trip_original_province}/{trip_province}/{trip_logistics_zone})")
                        continue   # ลองสาขาถัดไปใน same_zone_df
                # 🔒 Province priority: เน้นจังหวัดเดียวกันก่อนหมด
                # ถ้ายังมีสาขาจังหวัดเดียวกัน **ภายใน 25km** → ข้ามสาขาต่างจังหวัดก่อน
                # [FIX] เพิ่ม distance cap 25km: ไม่บล็อก cross-province ถ้าสาขาจังหวัดเดียวกันอยู่ไกลมาก
                if (_same_prov_close and _trip_provs_sort and
                        _c_prov and _c_prov not in _trip_provs_sort and
                        _c_zone == trip_logistics_zone):  # เฉพาะกรณีผ่านด้วย zone (ไม่ใช่ province)
                    continue  # มีสาขาจังหวัดเดียวกันใกล้ๆ → ข้ามต่างจังหวัด

                # 🎯 ดึงสาขาทั้งกลุ่มพิกัดเดียวกัน (จุดส่งเดียวกัน)
                # ใช้ get_group_branches_rt: รวม precomputed(≤200m) + runtime same-coord
                group_codes = get_group_branches_rt(candidate_code)
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
                        # 🏠 สมาชิกกลุ่ม (≤200m จาก candidate) = จุดส่งเดียวกัน → ไม่แยกทริปเด็ดขาด
                        # ตรวจระยะก่อน — ถ้าอยู่ใกล้ (<0.3km) ให้ข้าม region guard
                        _cg_lat = float(gc_row.iloc[0].get('_lat', 0) or 0)
                        _cg_lon = float(gc_row.iloc[0].get('_lon', 0) or 0)
                        _cd_lat = float(candidate_row.get('_lat', 0) or 0)
                        _cd_lon = float(candidate_row.get('_lon', 0) or 0)
                        _cg_phys_dist = haversine_distance(_cg_lat, _cg_lon, _cd_lat, _cd_lon) if (_cg_lat > 0 and _cg_lon > 0 and _cd_lat > 0 and _cd_lon > 0) else 999
                        if _cg_phys_dist > 0.3:  # ถ้าไกลกว่า 300m → ตรวจ region guard ตามปกติ
                            if trip_original_region and trip_original_region not in ('', 'ไม่ระบุ'):
                                _cg_prov = str(gc_row.iloc[0].get('_province', '') or '')
                                _cg_region = get_region_name(_cg_prov) if _cg_prov else ''
                                if _cg_region and _cg_region not in ('', 'ไม่ระบุ') and _cg_region != trip_original_region:
                                    safe_print(f"      🛑 CAND-GROUP GUARD: ตัด {gc} ภาค {_cg_region} ≠ {trip_original_region} (ห่าง {_cg_phys_dist:.1f}km)")
                                    continue  # ไม่เพิ่มเข้ากลุ่ม — น้ำหนัก/คิวก็ไม่นับ
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
                # 🔋 Force-fill: ถ้ารถว่างมาก (<70%) → ลองเพิ่มสาขาจังหวัดเดียวกัน ไม่จำกัดระยะ
                _ff_lims = PUNTHAI_LIMITS if trip_is_punthai else LIMITS
                _ff_buf = punthai_buffer if trip_is_punthai else maxmart_buffer
                _ff_cur_veh = next((v for v in reversed(['4W', 'JB', '6W']) if v in (trip_allowed or ['6W'])), '6W')
                _ff_cur_lim = _ff_lims.get(_ff_cur_veh, _ff_lims['6W'])
                _ff_max_w = _ff_cur_lim['max_w'] * _ff_buf
                _ff_max_c = _ff_cur_lim['max_c'] * _ff_buf
                _ff_util = max(
                    trip_weight / _ff_max_w if _ff_max_w > 0 else 1,
                    trip_cube / _ff_max_c if _ff_max_c > 0 else 1
                )
                _force_filled = False
                if _ff_util < 0.70 and trip_original_province:
                    _ff_rem = df[df['Code'].isin(unassigned)].copy()
                    _ff_same = _ff_rem[
                        _ff_rem['_province'].apply(lambda p: str(p or '').strip() == str(trip_original_province).strip())
                    ].copy()
                    if not _ff_same.empty:
                        if _trip_valid_coords:
                            _ff_same['_ff_dist'] = _ff_same.apply(
                                lambda row: min(
                                    haversine_distance(float(row['_lat'] or 0), float(row['_lon'] or 0), tlat, tlon)
                                    for tlat, tlon in _trip_valid_coords
                                ) if float(row.get('_lat', 0) or 0) > 0 else 999,
                                axis=1
                            )
                        else:
                            _ff_same['_ff_dist'] = 999
                        _ff_same = _ff_same.sort_values('_ff_dist')
                        for _, _ff_row in _ff_same.iterrows():
                            _ff_code = _ff_row['Code']
                            _ff_prov = str(_ff_row.get('_province', '') or '')
                            # BKK isolation
                            _BKK = 'กรุงเทพมหานคร'
                            if ((_ff_prov == _BKK and trip_original_province != _BKK) or
                                    (trip_original_province == _BKK and _ff_prov and _ff_prov != _BKK)):
                                continue
                            # ZONE_NEARBY: ห้ามต่างจังหวัด
                            _ff_zone = str(_ff_row.get('_logistics_zone', '') or '')
                            _is_trip_nb_ff = str(trip_logistics_zone or '').startswith('ZONE_NEARBY_')
                            _is_cand_nb_ff = _ff_zone.startswith('ZONE_NEARBY_')
                            if (_is_trip_nb_ff or _is_cand_nb_ff) and _ff_prov != trip_original_province:
                                continue
                            # Region check
                            _ff_region = get_region_name(_ff_prov) if _ff_prov else ''
                            if (trip_original_region and trip_original_region not in ('', 'ไม่ระบุ') and
                                    _ff_region and _ff_region not in ('', 'ไม่ระบุ') and
                                    _ff_region != trip_original_region):
                                continue
                            # Vehicle constraint + capacity check
                            _ff_test_codes = trip_codes + [_ff_code]
                            _ff_test_allowed = get_allowed_from_codes(_ff_test_codes, ['4W', 'JB', '6W'])
                            if not _ff_test_allowed:
                                continue
                            _ff_test_w = trip_weight + float(_ff_row.get('Weight', 0) or 0)
                            _ff_test_c = trip_cube + float(_ff_row.get('Cube', 0) or 0)
                            _ff_test_d = len(_ff_test_codes)
                            _ff_veh_ok = None
                            for _ffv in ['4W', 'JB', '6W']:
                                if _ffv not in _ff_test_allowed:
                                    continue
                                _ffvl = _ff_lims.get(_ffv, _ff_lims['6W'])
                                if (_ff_test_w <= _ffvl['max_w'] * _ff_buf and
                                        _ff_test_c <= _ffvl['max_c'] * _ff_buf and
                                        _ff_test_d <= _ffvl.get('max_drops', 999)):
                                    _ff_veh_ok = _ffv
                                    break
                            if not _ff_veh_ok:
                                continue
                            # ✅ เพิ่มสาขา (ไม่จำกัดระยะ)
                            trip_codes.append(_ff_code)
                            if _ff_code in unassigned:
                                unassigned.remove(_ff_code)
                            else:
                                for _u in list(unassigned):
                                    if str(_u).upper() == str(_ff_code).upper():
                                        unassigned.remove(_u)
                                        break
                            trip_weight = _ff_test_w
                            trip_cube = _ff_test_c
                            trip_allowed = _ff_test_allowed
                            trip_is_punthai = all(branch_bu_cache.get(c, False) for c in trip_codes)
                            safe_print(f"      🔋 Force-fill #{trip_counter}: +{_ff_code} ({_ff_prov}) util={_ff_util*100:.0f}%")
                            _force_filled = True
                            break
                if not _force_filled:
                    # ไม่มีสาขาจังหวัดเดียวกันที่เหมาะสม → ปิดทริป
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

                # ขยาย merge radius ตาม utilization + ตาม province
                # จังหวัดเดียวกัน: 150km / 80km  |  ต่างจังหวัด: 80km / 40km
                _b_prov = branch_row.get('_province', '')  # ต้องนิยามก่อนใช้ใน _mg_same_prov
                _mg_util = max(trip_cap['weight'] / trip_cap['max_w'], trip_cap['cube'] / trip_cap['max_c'])
                _mg_same_prov = bool(trip_cap.get('provinces', set()) & {_b_prov}) if _b_prov else False
                if _mg_same_prov:
                    _mg_dist_limit = 150 if _mg_util < 0.60 else 80
                else:
                    _mg_dist_limit = 80 if _mg_util < 0.60 else 40  # ต่างจังหวัด: จำกัดระยะ
                if dist_to_trip > _mg_dist_limit:
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
                # (_b_prov already assigned above for distance check)
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
                # 🔒 กรุงเทพฯ isolation (Step 6.6 merge): ห้ามกรุงเทพฯ ปนกับจังหวัดอื่น
                _BKK = 'กรุงเทพมหานคร'
                _trip_provs_mg = trip_cap.get('provinces', set())
                if ((_b_prov == _BKK and _trip_provs_mg and _BKK not in _trip_provs_mg) or
                        (_BKK in _trip_provs_mg and _b_prov and _b_prov != _BKK)):
                    safe_print(f"      🚫 BKK isolation merge: ตัด {branch_code} ({_b_prov}) ≠ trip {_trip_provs_mg}")
                    continue
                # 🔒 ZONE_NEARBY strict (Step 6.6 merge): ห้ามรวม ZONE_NEARBY ต่างจังหวัด
                _trip_zones_mg = trip_cap.get('logistics_zones', set())
                _is_trip_nearby_mg = any(str(z).startswith('ZONE_NEARBY_') for z in _trip_zones_mg)
                _is_branch_nearby_mg = str(_b_zone or '').startswith('ZONE_NEARBY_')
                if _is_trip_nearby_mg or _is_branch_nearby_mg:
                    if _b_prov and _trip_provs_mg and _b_prov not in _trip_provs_mg:
                        safe_print(f"      🚫 NEARBY strict merge: ตัด {branch_code} ({_b_prov}/{_b_zone}) ≠ trip provinces {_trip_provs_mg}")
                        continue
                _zone_ok = (
                    _b_prov in trip_cap.get('provinces', set()) or
                    _b_zone in trip_cap.get('logistics_zones', set())
                    # 🔒 ลบ highway-only check ออก: ป้องกัน cross-zone merge
                    # (เช่น ZONE_H highway='2/24' merge เข้า ZONE_K highway='24' ผ่าน intersection)
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
                        # 🔒 กรุงเทพฯ isolation (nearby merge)
                        _BKK = 'กรุงเทพมหานคร'
                        _nb_trip_provs = trip_cap.get('provinces', set())
                        if ((_nb_prov == _BKK and _nb_trip_provs and _BKK not in _nb_trip_provs) or
                                (_BKK in _nb_trip_provs and _nb_prov and _nb_prov != _BKK)):
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
    # Step 6.65: 🔗 AGGRESSIVE CONSOLIDATION — รวมทริปที่ยังว่างอยู่
    # หลักการ: "จะตัดใหม่ต้องเต็มก่อน" — รวม 2 ทริปที่ util ต่ำเข้าด้วยกัน
    # ถ้าน้ำหนัก+ปริมาตร+drops รวมกันแล้วยังพอดีรถ
    # ==========================================
    MIN_CONSOLIDATION_UTIL = 0.75  # ทริปที่ util < 75% เป็น candidate consolidation

    safe_print("🔗 Consolidating under-utilized trips (< 75%)...")
    _consol_rounds = 0
    _consol_total = 0
    while _consol_rounds < 30:
        _consol_rounds += 1
        _trips_now = sorted(df[df['Trip'] > 0]['Trip'].unique())

        # Build capacity info for all trips
        _caps_cs = {}
        for _t_cs in _trips_now:
            _c_cs = get_trip_capacity(_t_cs)
            if _c_cs:
                _caps_cs[_t_cs] = _c_cs

        # Find under-utilized trips (sorted: lowest util first)
        _under_cs = sorted(
            [t for t, c in _caps_cs.items()
             if max(c['weight'] / c['max_w'], c['cube'] / c['max_c']) < MIN_CONSOLIDATION_UTIL],
            key=lambda t: max(_caps_cs[t]['weight'] / _caps_cs[t]['max_w'],
                              _caps_cs[t]['cube'] / _caps_cs[t]['max_c'])
        )
        if not _under_cs:
            break

        _merged_cs = False
        for _ta_cs in _under_cs:
            if _ta_cs not in _caps_cs:
                continue
            _ca_cs = _caps_cs[_ta_cs]

            # Try to merge with any compatible trip (prefer lowest-numbered / same zone)
            for _tb_cs in sorted(_caps_cs.keys()):
                if _tb_cs == _ta_cs or _tb_cs not in _caps_cs:
                    continue
                _cb_cs = _caps_cs[_tb_cs]

                _pa_cs = _ca_cs.get('provinces', set())
                _pb_cs = _cb_cs.get('provinces', set())
                _za_cs = _ca_cs.get('logistics_zones', set())
                _zb_cs = _cb_cs.get('logistics_zones', set())
                _ra_cs = _ca_cs.get('regions', set())
                _rb_cs = _cb_cs.get('regions', set())

                # Zone family: ใช้ 2 prefix แรก (เช่น ZONE_BKK, ZONE_H1, ZONE_K)
                def _zfam_cs(z): parts = str(z).split('_'); return '_'.join(parts[:2]) if len(parts) >= 2 else str(z)
                _za_fam_cs = {_zfam_cs(z) for z in _za_cs if z}
                _zb_fam_cs = {_zfam_cs(z) for z in _zb_cs if z}

                # Must share province OR exact zone OR zone family (no random cross-zone merging)
                if not ((_pa_cs & _pb_cs) or (_za_cs & _zb_cs) or (_za_fam_cs & _zb_fam_cs)):
                    continue
                # Must share region (ภาค) — strict: if EITHER side has known region, both must match
                if _ra_cs and _rb_cs and not (_ra_cs & _rb_cs):
                    continue
                if _ra_cs and not _rb_cs and _pa_cs and _pb_cs:
                    # Trip B has no region info but has provinces → compute region from provinces
                    _rb_cs_calc = {get_region_name(str(p)) for p in _pb_cs if p}
                    _rb_cs_calc.discard('ไม่ระบุ'); _rb_cs_calc.discard('')
                    if _rb_cs_calc and not (_ra_cs & _rb_cs_calc):
                        continue
                if _rb_cs and not _ra_cs and _pa_cs and _pb_cs:
                    _ra_cs_calc = {get_region_name(str(p)) for p in _pa_cs if p}
                    _ra_cs_calc.discard('ไม่ระบุ'); _ra_cs_calc.discard('')
                    if _ra_cs_calc and not (_rb_cs & _ra_cs_calc):
                        continue
                # BKK isolation — normalize aliases
                _BKK_cs = 'กรุงเทพมหานคร'
                _BKK_ALIASES_cs = {'กรุงเทพฯ', 'กทม', 'กทม.', 'Bangkok'}
                _pa_has_bkk = bool(_pa_cs & ({_BKK_cs} | _BKK_ALIASES_cs))
                _pb_has_bkk = bool(_pb_cs & ({_BKK_cs} | _BKK_ALIASES_cs))
                if _pa_has_bkk != _pb_has_bkk:
                    continue
                # 📐 Zone isolation by centroid distance: ห้ามรวมทริปที่ centroid ห่างกันเกิน
                # (ต่างจังหวัด: max 120km, จังหวัดเดียวกันไม่จำกัด)
                if not (_pa_cs & _pb_cs):  # ต่างจังหวัด
                    _ca_lat_cs = float(_ca_cs.get('centroid_lat', 0) or 0)
                    _ca_lon_cs = float(_ca_cs.get('centroid_lon', 0) or 0)
                    _cb_lat_cs = float(_cb_cs.get('centroid_lat', 0) or 0)
                    _cb_lon_cs = float(_cb_cs.get('centroid_lon', 0) or 0)
                    if _ca_lat_cs and _cb_lat_cs:
                        _dp_cs2 = radians(_cb_lat_cs - _ca_lat_cs)
                        _dl_cs2 = radians(_cb_lon_cs - _ca_lon_cs)
                        _aa_cs2 = sin(_dp_cs2/2)**2 + cos(radians(_ca_lat_cs))*cos(radians(_cb_lat_cs))*sin(_dl_cs2/2)**2
                        _cdist_cs = 2*6371*atan2(sqrt(_aa_cs2), sqrt(1-_aa_cs2))
                        if _cdist_cs > 120.0:
                            continue
                # ZONE_NEARBY: only same province
                _a_nb_cs = any(str(z).startswith('ZONE_NEARBY_') for z in _za_cs)
                _b_nb_cs = any(str(z).startswith('ZONE_NEARBY_') for z in _zb_cs)
                if (_a_nb_cs or _b_nb_cs) and not (_pa_cs & _pb_cs):
                    continue

                # Check combined load fits in a truck
                _cw_cs = _ca_cs['weight'] + _cb_cs['weight']
                _cc_cs = _ca_cs['cube'] + _cb_cs['cube']
                _cd_cs = _ca_cs['drops'] + _cb_cs['drops']
                _call_cs = get_allowed_from_codes(
                    _ca_cs['codes'] + _cb_cs['codes'], ['4W', 'JB', '6W'])
                if not _call_cs:
                    continue
                _cpunthai_cs = all(branch_bu_cache.get(c, False)
                                   for c in _ca_cs['codes'] + _cb_cs['codes'])
                _cbuf_cs = punthai_buffer if _cpunthai_cs else maxmart_buffer
                _clims_cs = PUNTHAI_LIMITS if _cpunthai_cs else LIMITS

                _fits_veh_cs = None
                for _fv_cs in ['4W', 'JB', '6W']:
                    if _fv_cs not in _call_cs:
                        continue
                    _fl_cs = _clims_cs[_fv_cs]
                    if (_cw_cs <= _fl_cs['max_w'] * _cbuf_cs and
                            _cc_cs <= _fl_cs['max_c'] * _cbuf_cs and
                            _cd_cs <= _fl_cs.get('max_drops', 999)):
                        _fits_veh_cs = _fv_cs
                        break

                if not _fits_veh_cs:
                    continue

                # ✅ Merge _tb_cs into _ta_cs
                _new_util = max(_cw_cs / (_clims_cs[_fits_veh_cs]['max_w'] * _cbuf_cs),
                                _cc_cs / (_clims_cs[_fits_veh_cs]['max_c'] * _cbuf_cs))
                df.loc[df['Trip'] == _tb_cs, 'Trip'] = _ta_cs
                safe_print(f"   🔗 Consolidate Trip {_tb_cs} → Trip {_ta_cs} "
                           f"[{_fits_veh_cs}] {_cd_cs} drops {_cw_cs:.0f}kg "
                           f"→ {_new_util*100:.0f}%")
                _caps_cs[_ta_cs] = get_trip_capacity(_ta_cs)
                del _caps_cs[_tb_cs]
                _consol_total += 1
                _merged_cs = True
                break

            if _merged_cs:
                break

        if not _merged_cs:
            break

    if _consol_total > 0:
        safe_print(f"🔗 Consolidation done: merged {_consol_total} trips")
        # Renumber after consolidation
        _remaining_cs = sorted(df[df['Trip'] > 0]['Trip'].unique())
        _renumber_cs = {old: new for new, old in enumerate(_remaining_cs, start=1)}
        df['Trip'] = df['Trip'].map(lambda x: _renumber_cs.get(x, x) if x > 0 else x)
    else:
        safe_print("🔗 Consolidation: no further merges possible")

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
    # Step 6.8: 🔗 POST-AUDIT CONSOLIDATION — รวมเศษทริปที่เกิดจากการ audit แตก
    # เพราะ Step 6.7 อาจแยกทริปแล้วทิ้ง fragment เล็กๆ ไว้ ต้องรวมกลับ
    # ==========================================
    _pa_total = 0
    _pa_rounds = 0
    while _pa_rounds < 20:
        _pa_rounds += 1
        _pa_caps = {}
        for _t_pa in sorted(df[df['Trip'] > 0]['Trip'].unique()):
            _c_pa = get_trip_capacity(_t_pa)
            if _c_pa:
                _pa_caps[_t_pa] = _c_pa
        _pa_under = sorted(
            [t for t, c in _pa_caps.items()
             if max(c['weight'] / c['max_w'], c['cube'] / c['max_c']) < MIN_CONSOLIDATION_UTIL],
            key=lambda t: max(_pa_caps[t]['weight'] / _pa_caps[t]['max_w'],
                              _pa_caps[t]['cube'] / _pa_caps[t]['max_c'])
        )
        if not _pa_under:
            break
        _pa_merged = False
        for _ta_pa in _pa_under:
            if _ta_pa not in _pa_caps:
                continue
            _ca_pa = _pa_caps[_ta_pa]
            for _tb_pa in sorted(_pa_caps.keys()):
                if _tb_pa == _ta_pa or _tb_pa not in _pa_caps:
                    continue
                _cb_pa = _pa_caps[_tb_pa]
                _pa_a = _ca_pa.get('provinces', set())
                _pb_pa = _cb_pa.get('provinces', set())
                _za_pa = _ca_pa.get('logistics_zones', set())
                _zb_pa = _cb_pa.get('logistics_zones', set())
                _ra_pa = _ca_pa.get('regions', set())
                _rb_pa = _cb_pa.get('regions', set())
                def _zfam_pa(z): _p = str(z).split('_'); return '_'.join(_p[:2]) if len(_p) >= 2 else str(z)
                _za_fam_pa = {_zfam_pa(z) for z in _za_pa if z}
                _zb_fam_pa = {_zfam_pa(z) for z in _zb_pa if z}
                if not ((_pa_a & _pb_pa) or (_za_pa & _zb_pa) or (_za_fam_pa & _zb_fam_pa)):
                    continue
                if _ra_pa and _rb_pa and not (_ra_pa & _rb_pa):
                    continue
                if _ra_pa and not _rb_pa and _pa_a and _pb_pa:
                    _rb_pa_calc = {get_region_name(str(p)) for p in _pb_pa if p}
                    _rb_pa_calc.discard('ไม่ระบุ'); _rb_pa_calc.discard('')
                    if _rb_pa_calc and not (_ra_pa & _rb_pa_calc):
                        continue
                if _rb_pa and not _ra_pa and _pa_a and _pb_pa:
                    _ra_pa_calc = {get_region_name(str(p)) for p in _pa_a if p}
                    _ra_pa_calc.discard('ไม่ระบุ'); _ra_pa_calc.discard('')
                    if _ra_pa_calc and not (_rb_pa & _ra_pa_calc):
                        continue
                _BKK_pa = 'กรุงเทพมหานคร'
                _BKK_ALIASES_pa = {'กรุงเทพฯ', 'กทม', 'กทม.', 'Bangkok'}
                _pa_has_bkk_pa = bool(_pa_a & ({_BKK_pa} | _BKK_ALIASES_pa))
                _pb_has_bkk_pa = bool(_pb_pa & ({_BKK_pa} | _BKK_ALIASES_pa))
                if _pa_has_bkk_pa != _pb_has_bkk_pa:
                    continue
                # 📐 Zone isolation by centroid distance (post-audit consolidation)
                if not (_pa_a & _pb_pa):  # ต่างจังหวัด: max 120km centroid
                    _ca_lat_pa = float(_ca_pa.get('centroid_lat', 0) or 0)
                    _ca_lon_pa = float(_ca_pa.get('centroid_lon', 0) or 0)
                    _cb_lat_pa = float(_cb_pa.get('centroid_lat', 0) or 0)
                    _cb_lon_pa = float(_cb_pa.get('centroid_lon', 0) or 0)
                    if _ca_lat_pa and _cb_lat_pa:
                        _dp_pa2 = radians(_cb_lat_pa - _ca_lat_pa)
                        _dl_pa2 = radians(_cb_lon_pa - _ca_lon_pa)
                        _aa_pa2 = sin(_dp_pa2/2)**2 + cos(radians(_ca_lat_pa))*cos(radians(_cb_lat_pa))*sin(_dl_pa2/2)**2
                        _cdist_pa = 2*6371*atan2(sqrt(_aa_pa2), sqrt(1-_aa_pa2))
                        if _cdist_pa > 120.0:
                            continue
                _a_nb_pa = any(str(z).startswith('ZONE_NEARBY_') for z in _za_pa)
                _b_nb_pa = any(str(z).startswith('ZONE_NEARBY_') for z in _zb_pa)
                if (_a_nb_pa or _b_nb_pa) and not (_pa_a & _pb_pa):
                    continue
                _cw_pa = _ca_pa['weight'] + _cb_pa['weight']
                _cc_pa = _ca_pa['cube'] + _cb_pa['cube']
                _cd_pa = _ca_pa['drops'] + _cb_pa['drops']
                _call_pa = get_allowed_from_codes(
                    _ca_pa['codes'] + _cb_pa['codes'], ['4W', 'JB', '6W'])
                if not _call_pa:
                    continue
                _cpun_pa = all(branch_bu_cache.get(c, False)
                               for c in _ca_pa['codes'] + _cb_pa['codes'])
                _cbuf_pa = punthai_buffer if _cpun_pa else maxmart_buffer
                _clim_pa = PUNTHAI_LIMITS if _cpun_pa else LIMITS
                _fveh_pa = None
                for _fv_pa in ['4W', 'JB', '6W']:
                    if _fv_pa not in _call_pa:
                        continue
                    _fl_pa = _clim_pa[_fv_pa]
                    if (_cw_pa <= _fl_pa['max_w'] * _cbuf_pa and
                            _cc_pa <= _fl_pa['max_c'] * _cbuf_pa and
                            _cd_pa <= _fl_pa.get('max_drops', 999)):
                        _fveh_pa = _fv_pa
                        break
                if not _fveh_pa:
                    continue
                df.loc[df['Trip'] == _tb_pa, 'Trip'] = _ta_pa
                _nutil = max(_cw_pa / (_clim_pa[_fveh_pa]['max_w'] * _cbuf_pa),
                             _cc_pa / (_clim_pa[_fveh_pa]['max_c'] * _cbuf_pa))
                safe_print(f"   🔗 Post-audit merge Trip {_tb_pa} → Trip {_ta_pa} "
                           f"[{_fveh_pa}] {_cd_pa}drops {_cw_pa:.0f}kg → {_nutil*100:.0f}%")
                _pa_caps[_ta_pa] = get_trip_capacity(_ta_pa)
                del _pa_caps[_tb_pa]
                _pa_total += 1
                _pa_merged = True
                break
            if _pa_merged:
                break
        if not _pa_merged:
            break
    if _pa_total > 0:
        safe_print(f"🔗 Post-audit consolidation: merged {_pa_total} trips")
        _pa_rem = sorted(df[df['Trip'] > 0]['Trip'].unique())
        _pa_ren = {old: new for new, old in enumerate(_pa_rem, start=1)}
        df['Trip'] = df['Trip'].map(lambda x: _pa_ren.get(x, x) if x > 0 else x)
    else:
        safe_print("🔗 Post-audit consolidation: nothing to merge")

    # ==========================================
    # Step 6.9: 🚛 Fleet-Target Consolidation
    # รันเฉพาะเมื่อ fleet_limits ถูกตั้ง — พยายามลดทริปให้พอดีกับกองยาน
    # ยอมผ่อนปรน max_drops เมื่อจำเป็น แต่ยังคุม weight/cube/MaxTruckType
    # ==========================================
    if fleet_limits:
        import math as _math_fc
        _fl_target = sum(v for v in fleet_limits.values() if v and v < 999)
        _fl_current_count = len(df[df['Trip'] > 0]['Trip'].unique())
        if _fl_current_count > _fl_target:
            safe_print(f"\n🚛 Step 6.9: Fleet Consolidation เป้าหมาย {_fl_target} ทริป (ปัจจุบัน {_fl_current_count})")

            # คำนวณ drop limit ตามสัดส่วน branches/trucks ของแต่ละประเภทรถ
            _fc_branch_type_count: dict = {'4W': 0, 'JB': 0, '6W': 0}
            for _fc_c in df[df['Trip'] > 0]['Code'].unique():
                _fc_mv = branch_max_vehicle_cache.get(str(_fc_c).strip().upper(), '6W')
                if _fc_mv in _fc_branch_type_count:
                    _fc_branch_type_count[_fc_mv] += 1

            _fc_relaxed_drops: dict = {}
            for _fc_vt, _fc_fl_cnt in fleet_limits.items():
                if not _fc_fl_cnt or _fc_fl_cnt >= 999:
                    continue
                _fc_orig_drops = LIMITS[_fc_vt]['max_drops']
                if _fc_orig_drops >= 999:  # 6W: unlimited
                    _fc_relaxed_drops[_fc_vt] = 999
                    continue
                _fc_bc = _fc_branch_type_count.get(_fc_vt, 0)
                # ผ่อนผัน: ใช้ max(original, ceil(branches/trucks)*1.5) เพื่อเปิดโอกาส merge
                _fc_nd = max(_fc_orig_drops, int(_math_fc.ceil(_fc_bc / max(_fc_fl_cnt, 1) * 1.5)))
                _fc_relaxed_drops[_fc_vt] = _fc_nd
                safe_print(f"   📏 {_fc_vt}: {_fc_bc} branches ÷ {_fc_fl_cnt} trucks → max_drops ผ่อนถึง {_fc_nd} (ปกติ {_fc_orig_drops})")

            _BKK_PROV_SET = {'กรุงเทพมหานคร', 'กรุงเทพฯ', 'กทม', 'กทม.', 'Bangkok'}

            _fc_total_merged = 0
            _fc_pass = True
            while _fc_pass:
                _fc_pass = False
                _fc_trips_now = sorted(df[df['Trip'] > 0]['Trip'].unique())
                if len(_fc_trips_now) <= _fl_target:
                    break

                # สร้าง trip info dict (เรียงจาก utilization ต่ำสุดก่อน = merge ออกก่อน)
                _fc_info: dict = {}
                for _ft in _fc_trips_now:
                    _ftd = df[df['Trip'] == _ft]
                    _ft_codes = _ftd['Code'].unique().tolist()
                    _ft_ranks = [vehicle_priority.get(branch_max_vehicle_cache.get(str(c).strip().upper(), '6W'), 3) for c in _ft_codes]
                    _ft_min_rank = min(_ft_ranks) if _ft_ranks else 3
                    _ft_mv = {1: '4W', 2: 'JB', 3: '6W'}.get(_ft_min_rank, '6W')
                    _ft_is_pun = all(branch_bu_cache.get(c, False) for c in _ft_codes)
                    _ft_lim = PUNTHAI_LIMITS if _ft_is_pun else LIMITS
                    _ft_buf = punthai_buffer if _ft_is_pun else maxmart_buffer
                    _ft_prov = str(_ftd.iloc[0].get('_province', '') or '').strip()
                    _ft_w = float(_ftd['Weight'].sum())
                    _ft_c = float(_ftd['Cube'].sum())
                    _ft_ndr = len(_ftd['Code'].unique())  # drops = unique code count
                    _ft_util = max(_ft_w / (_ft_lim[_ft_mv]['max_w'] * _ft_buf),
                                   _ft_c / (_ft_lim[_ft_mv]['max_c'] * _ft_buf)) if _ft_mv in _ft_lim else 0.0
                    _fc_info[_ft] = {
                        'codes': _ft_codes, 'w': _ft_w, 'c': _ft_c, 'rows': _ft_ndr,
                        'min_rank': _ft_min_rank, 'mv': _ft_mv,
                        'is_pun': _ft_is_pun, 'lim': _ft_lim, 'buf': _ft_buf,
                        'prov': _ft_prov,
                        'is_bkk': _ft_prov in _BKK_PROV_SET,
                        'util': _ft_util,
                        'lat': float(_ftd['_lat'].mean()) if '_lat' in _ftd.columns else 0.0,
                        'lon': float(_ftd['_lon'].mean()) if '_lon' in _ftd.columns else 0.0,
                    }

                # เรียงจาก util ต่ำ → สูง (ลบทริปเล็ก/ว่างก่อน)
                _fc_sorted = sorted(_fc_info.keys(), key=lambda t: _fc_info[t]['util'])

                for _ts_id in _fc_sorted:
                    if _ts_id not in _fc_info:
                        continue
                    _ts = _fc_info[_ts_id]

                    for _tb_id in _fc_sorted:
                        if _tb_id == _ts_id or _tb_id not in _fc_info:
                            continue
                        _tb = _fc_info[_tb_id]

                        # ── Rule 1: BKK isolation ──────────────────────────────
                        if _ts['is_bkk'] != _tb['is_bkk']:
                            continue

                        # ── Rule 2: Non-BKK: ต้องจังหวัดเดียวกัน ──────────────
                        if not _ts['is_bkk'] and _ts['prov'] != _tb['prov']:
                            continue

                        # ── Rule 3: BKK: allow same-province merge freely ──────────
                        # (BKK is one province, all areas ≤40km apart)

                        # ── Capacity check ────────────────────────────────────
                        _comb_w = _ts['w'] + _tb['w']
                        _comb_c = _ts['c'] + _tb['c']
                        # drops = unique code count (not row count)
                        _comb_codes_set = set(_ts['codes']) | set(_tb['codes'])
                        _comb_rows = len(_comb_codes_set)
                        _comb_min_rank = min(_ts['min_rank'], _tb['min_rank'])
                        _comb_is_pun = _ts['is_pun'] and _tb['is_pun']
                        _comb_lim = PUNTHAI_LIMITS if _comb_is_pun else LIMITS
                        _comb_buf = punthai_buffer if _comb_is_pun else maxmart_buffer

                        _fc_fit_veh = None
                        for _fv, _fvr in [('4W', 1), ('JB', 2), ('6W', 3)]:
                            if _fvr > _comb_min_rank:
                                continue
                            _fl_c = _comb_lim[_fv]
                            _rd = _fc_relaxed_drops.get(_fv, _fl_c['max_drops'])
                            if (_comb_w <= _fl_c['max_w'] * _comb_buf and
                                    _comb_c <= _fl_c['max_c'] * _comb_buf and
                                    _comb_rows <= _rd):
                                _fc_fit_veh = _fv
                                break

                        if _fc_fit_veh:
                            df.loc[df['Trip'] == _ts_id, 'Trip'] = _tb_id
                            _tb['codes'] += _ts['codes']
                            _tb['w'] = _comb_w
                            _tb['c'] = _comb_c
                            _tb['rows'] = _comb_rows
                            _tb['min_rank'] = _comb_min_rank
                            _tb['is_pun'] = _comb_is_pun
                            del _fc_info[_ts_id]
                            _fc_total_merged += 1
                            safe_print(f"   🚛 FC: Trip {_ts_id}→{_tb_id} [{_fc_fit_veh}] {_comb_rows}rows {_comb_w:.0f}kg {_comb_c:.1f}m³ [{_ts['prov']}]")
                            _fc_pass = True
                            break
                    if _fc_pass:
                        break

            if _fc_total_merged > 0:
                _fc_final = len(df[df['Trip'] > 0]['Trip'].unique())
                safe_print(f"✅ Fleet Consolidation: รวม {_fc_total_merged} merges → {_fc_final} ทริป")
                _fc_rem = sorted(df[df['Trip'] > 0]['Trip'].unique())
                _fc_ren = {old: new for new, old in enumerate(_fc_rem, start=1)}
                df['Trip'] = df['Trip'].map(lambda x: _fc_ren.get(x, x) if x > 0 else x)
            else:
                safe_print(f"⚠️ Fleet Consolidation: ไม่มี merge ที่ทำได้ คงที่ {_fl_current_count} ทริป")

    # ==========================================
    # Step 7: สร้าง Summary + Central Rule + Punthai Drop Limits
    # ==========================================
    summary_data = []

    # 🚛 Fleet Constraint: ติดตามจำนวนรถแต่ละประเภทที่ใช้ไป
    _fleet_limits = fleet_limits or {'4W': 999, 'JB': 999, '6W': 999}
    fleet_used = {'4W': 0, 'JB': 0, '6W': 0}
    _fleet_rank = {1: '4W', 2: 'JB', 3: '6W'}
    _rank_fleet = {'4W': 1, 'JB': 2, '6W': 3}

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

        # 🚛 Fleet Constraint: ถ้าโควต้ารถประเภทนี้เต็ม → ลอง upgrade ไปรถใหญ่กว่า
        _sv = suggested  # บันทึกรถเดิม
        _sv_rank = _rank_fleet.get(suggested, 3)
        _upgraded_by_fleet = False
        while fleet_used.get(suggested, 0) >= _fleet_limits.get(suggested, 999):
            _next_rank = _sv_rank + 1
            if _next_rank > 3:
                # ไม่มีรถให้ upgrade → ใช้รถเดิม + เตือน
                source += " ⚠️ เกินโควต้า"
                break
            _next_veh = _fleet_rank.get(_next_rank, '6W')
            # เช็คว่าสาขาอนุญาตรถใหญ่กว่าไหม
            if _next_rank <= min_max_size:
                suggested = _next_veh
                _sv_rank = _next_rank
                _upgraded_by_fleet = True
                safe_print(f"      🚛 Fleet upgrade: Trip {trip_num} {_sv}→{suggested} (โควต้า {_sv} เต็ม {fleet_used.get(_sv,0)}/{_fleet_limits.get(_sv,999)})")
            else:
                # สาขาจำกัดไม่ให้ใช้รถใหญ่กว่า → ยังคงใช้รถเดิม + เตือน
                source += " ⚠️ เกินโควต้า"
                break
        if _upgraded_by_fleet:
            source += f" ↑ Fleet({_sv}→{suggested})"
            # คำนวณ utilization ใหม่ด้วยรถที่ upgrade
            if suggested in limits_for_util:
                w_util = (total_w / limits_for_util[suggested]['max_w']) * 100
                c_util = (total_c / limits_for_util[suggested]['max_c']) * 100
        fleet_used[suggested] = fleet_used.get(suggested, 0) + 1

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
            # ป้องกัน duplicate code: ตัด code ซ้ำเพียงครั้งเดียว (แต่ลบทุก row ในทริปนั้น)
            codes_to_remove = []
            _cut_codes_seen = set()
            for _, row in trip_data.iterrows():
                # เช็คทั้ง buffer และ drops
                if current_w <= max_w and current_c <= max_c and current_drops <= max_drops:
                    break  # พอดีแล้ว
                
                code = row['Code']
                if code in _cut_codes_seen:
                    # แถวซ้ำ — อัพเดต current แต่ไม่เพิ่ม overflow ซ้ำ
                    current_w -= row['Weight']
                    current_c -= row['Cube']
                    current_drops -= 1
                    continue
                _cut_codes_seen.add(code)
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
                # ใช้ .sum() เพื่อรองรับ df ที่มีแถว code ซ้ำ
                trip_codes = []
                trip_weight = 0
                trip_cube = 0
                trip_drops = 0
                
                for code in list(remaining_codes):
                    code_row = df[(df['Code'] == code) & (df['Trip'] == 0)]
                    if code_row.empty:
                        # ลองหาโดยไม่กรอง Trip (อาจถูก assign ไปแล้ว)
                        code_row = df[df['Code'] == code]
                    if code_row.empty:
                        remaining_codes.remove(code)
                        continue
                    
                    # ใช้ .sum() รองรับ duplicate rows
                    code_w = float(code_row['Weight'].sum())
                    code_c = float(code_row['Cube'].sum())
                    n_rows = len(code_row)
                    
                    # เช็คว่าเพิ่มได้หรือไม่
                    if (trip_weight + code_w <= max_w and 
                        trip_cube + code_c <= max_c and 
                        trip_drops + n_rows <= max_drops):
                        trip_codes.append(code)
                        trip_weight += code_w
                        trip_cube += code_c
                        trip_drops += n_rows
                        remaining_codes.remove(code)
                    elif trip_drops == 0:
                        # ถ้าสาขาเดียวเกิน buffer ก็ต้องเพิ่มอยู่ดี
                        trip_codes.append(code)
                        trip_weight += code_w
                        trip_cube += code_c
                        trip_drops += n_rows
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
                    # นับแถวจริงจาก df (ไม่ใช้ len(trip_codes) เพราะอาจมี duplicate rows)
                    _ov_actual = df[df['Trip'] == new_trip]
                    _ov_w = _ov_actual['Weight'].sum()
                    _ov_c = _ov_actual['Cube'].sum()
                    is_overflow_punthai = all(
                        str(df[df['Code'] == c]['BU'].values[0] if len(df[df['Code'] == c]) > 0 else '').upper() in ['211', 'PUNTHAI'] 
                        for c in trip_codes
                    )
                    overflow_limits_final = PUNTHAI_LIMITS if is_overflow_punthai else LIMITS
                    overflow_buffer_final = punthai_buffer if is_overflow_punthai else maxmart_buffer
                    buffer_label = f"🅿️ {int(overflow_buffer_final*100)}%" if is_overflow_punthai else f"🅼 {int(overflow_buffer_final*100)}%"
                    
                    summary_data.append({
                        'Trip': new_trip,
                        'Branches': len(_ov_actual),
                        'Weight': _ov_w,
                        'Cube': _ov_c,
                        'Truck': f'{max_veh} 🔪 ตัดออก',
                        'BU_Type': 'punthai' if is_overflow_punthai else 'mixed',
                        'Buffer': buffer_label,
                        'Weight_Use%': (_ov_w / overflow_limits_final[max_veh]['max_w']) * 100,
                        'Cube_Use%': (_ov_c / overflow_limits_final[max_veh]['max_c']) * 100,
                        'Total_Distance': 0
                    })
                    safe_print(f"   ✅ สร้าง Trip {new_trip} ใหม่สำหรับสาขา {max_veh} ({len(_ov_actual)} แถว/{len(trip_codes)} code, {_ov_w:.0f}kg)")
    
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
            _veh_buffer = punthai_buffer if is_punthai else maxmart_buffer
            
            current_w = 0
            current_c = 0
            current_drops = 0
            max_w = limits[max_veh]['max_w'] * _veh_buffer
            max_c = limits[max_veh]['max_c'] * _veh_buffer
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
            _vp_local = {'4W': 1, 'JB': 2, '6W': 3}  # local copy — ไม่ shadow outer vehicle_priority
            min_rank = min(_vp_local.get(v, 3) for v in max_vehicles)
            suggested = {1: '4W', 2: 'JB', 3: '6W'}.get(min_rank, '6W')
            df.loc[df['Trip'] == trip_num, 'Truck'] = f"{suggested} 📋 จัดใหม่"
        
        df['VehicleCheck'] = df.apply(check_vehicle_compliance, axis=1)

    # ==========================================
    # Step 8.8: 🔒 FINAL REGION & BKK ISOLATION AUDIT
    # รันหลังทุก step เพื่อรับประกันไม่มีทริปที่ปนภาค/ปนกรุงเทพฯ
    # ==========================================
    safe_print("\n🔒 Step 8.8: Final Region & BKK Isolation Audit...")
    _BKK_PROV = 'กรุงเทพมหานคร'
    _final_audit_fixed = 0
    _fa_max_trip = df[df['Trip'] > 0]['Trip'].max() if len(df[df['Trip'] > 0]) > 0 else 0

    for _fa_trip in sorted(df[df['Trip'] > 0]['Trip'].unique()):
        _fa_data = df[df['Trip'] == _fa_trip]
        _fa_provs = [str(r.get('_province', '') or '') for _, r in _fa_data.iterrows()]
        _fa_provs_clean = [p for p in _fa_provs if p and p != 'nan']

        # 1️⃣ BKK Isolation: กรุงเทพฯ ห้ามปนกับจังหวัดอื่น
        _fa_has_bkk = _BKK_PROV in _fa_provs_clean
        _fa_has_non_bkk = any(p != _BKK_PROV for p in _fa_provs_clean)
        if _fa_has_bkk and _fa_has_non_bkk:
            # แยกสาขาที่ไม่ใช่กรุงเทพฯ ออก
            _fa_split_codes = [
                r['Code'] for _, r in _fa_data.iterrows()
                if str(r.get('_province', '') or '') != _BKK_PROV
            ]
            if _fa_split_codes:
                _fa_max_trip += 1
                df.loc[df['Code'].isin(_fa_split_codes), 'Trip'] = _fa_max_trip
                safe_print(f"   🔒 BKK AUDIT: Trip {_fa_trip} → แยก {len(_fa_split_codes)} สาขา non-BKK → Trip {_fa_max_trip}")
                _final_audit_fixed += 1
            continue  # ตรวจข้ออื่นบนข้อมูลใหม่ในรอบถัดไป

        # 2️⃣ Region Mixing: ห้ามปนภาค
        _fa_regions: dict = {}
        for _, _far in _fa_data.iterrows():
            _fap = str(_far.get('_province', '') or '')
            _fareg = get_region_name(_fap) if _fap and _fap != 'nan' else ''
            if not _fareg or _fareg == 'ไม่ระบุ':
                _fareg = str(_far.get('_region_name', '') or '')
            if _fareg and _fareg != 'ไม่ระบุ':
                _fa_regions[_fareg] = _fa_regions.get(_fareg, 0) + 1
        if len(_fa_regions) <= 1:
            continue  # clean — no mixing

        # พบการปนภาค → dominant = ภาคที่มีสาขามากสุด
        _fa_region_order = ['เหนือ', 'อีสาน', 'ตะวันออก', 'กลาง', 'ตะวันตก', 'ใต้']
        _fa_dominant = max(
            _fa_regions,
            key=lambda k: (_fa_regions[k], -(_fa_region_order.index(k) if k in _fa_region_order else 99))
        )
        _fa_minority_codes = []
        for _, _far2 in _fa_data.iterrows():
            _fap2 = str(_far2.get('_province', '') or '')
            _fareg2 = get_region_name(_fap2) if _fap2 and _fap2 != 'nan' else ''
            if not _fareg2 or _fareg2 == 'ไม่ระบุ':
                _fareg2 = str(_far2.get('_region_name', '') or '')
            if _fareg2 and _fareg2 != 'ไม่ระบุ' and _fareg2 != _fa_dominant:
                _fa_minority_codes.append(_far2['Code'])
        if _fa_minority_codes:
            _fa_max_trip += 1
            df.loc[df['Code'].isin(_fa_minority_codes), 'Trip'] = _fa_max_trip
            safe_print(f"   🔒 REGION AUDIT: Trip {_fa_trip} ปนภาค {_fa_regions} → แยก {_fa_minority_codes} → Trip {_fa_max_trip}")
            _final_audit_fixed += 1

    if _final_audit_fixed > 0:
        safe_print(f"   ✅ Final Audit: แก้ไข {_final_audit_fixed} ทริป")
        # Renumber trips after final audit
        _fa_rem = sorted(df[df['Trip'] > 0]['Trip'].unique())
        _fa_ren = {old: new for new, old in enumerate(_fa_rem, start=1)}
        df['Trip'] = df['Trip'].map(lambda x: _fa_ren.get(x, x) if x > 0 else x)
        # อัพเดต Truck mapping หลัง renumber
        try:
            for _fa_t in df[df['Trip'] > 0]['Trip'].unique():
                if _fa_t not in trip_truck_map:
                    _fa_codes = df[df['Trip'] == _fa_t]['Code'].tolist()
                    _fa_vp = {'4W': 1, 'JB': 2, '6W': 3}
                    _fa_max_veh_list = [branch_max_vehicle_cache.get(str(c).strip().upper(), '6W') for c in _fa_codes]
                    _fa_min_rank = min(_fa_vp.get(v, 3) for v in _fa_max_veh_list)
                    _fa_truck = {1: '4W', 2: 'JB', 3: '6W'}.get(_fa_min_rank, '6W')
                    df.loc[df['Trip'] == _fa_t, 'Truck'] = f"{_fa_truck} ✂️ audit-split"
        except Exception:
            pass
        # อัพเดต trip_truck_map ใหม่
        trip_truck_map = {}
        for _fa_t2 in df[df['Trip'] > 0]['Trip'].unique():
            _fa_td = df[df['Trip'] == _fa_t2]
            if not _fa_td.empty:
                _raw_trk = str(_fa_td.iloc[0].get('Truck', '6W') or '6W').split()[0]
                trip_truck_map[_fa_t2] = _raw_trk
        df['Truck'] = df['Trip'].map(trip_truck_map)
    else:
        safe_print("   ✅ Final Audit: ไม่พบการปนภาค/BKK")

    # ==========================================
    # Step 8.9: 🚛 Post-Enforcement Fleet Consolidation (second pass)
    # รันหลัง enforcement เพื่อ merge ทริปเล็กที่เกิดจาก overflow
    # ==========================================
    if fleet_limits:
        import math as _math_fc2
        _fl2_target = sum(v for v in fleet_limits.values() if v and v < 999)
        _fl2_current = len(df[df['Trip'] > 0]['Trip'].unique())
        if _fl2_current > _fl2_target:
            safe_print(f"\n🚛 Step 8.9: Post-Enforcement Fleet Consolidation ({_fl2_current} → {_fl2_target} ทริป)")
            _BKK_PROV_SET2 = {'กรุงเทพมหานคร', 'กรุงเทพฯ', 'กทม', 'กทม.', 'Bangkok'}
            # คำนวณ drop limit ผ่อนผันใหม่
            _fc2_all_codes = df[df['Trip'] > 0]['Code'].unique()
            _fc2_bc: dict = {'4W': 0, 'JB': 0, '6W': 0}
            for _c2 in _fc2_all_codes:
                _mv2 = branch_max_vehicle_cache.get(str(_c2).strip().upper(), '6W')
                if _mv2 in _fc2_bc:
                    _fc2_bc[_mv2] += 1
            _fc2_rdrop: dict = {}
            for _vt2, _fl_cnt2 in fleet_limits.items():
                if not _fl_cnt2 or _fl_cnt2 >= 999:
                    continue
                _od2 = LIMITS[_vt2]['max_drops']
                if _od2 >= 999:
                    _fc2_rdrop[_vt2] = 999; continue
                _nd2 = max(_od2, int(_math_fc2.ceil(_fc2_bc.get(_vt2, 0) / max(_fl_cnt2, 1) * 1.5)))
                _fc2_rdrop[_vt2] = _nd2

            _fc2_merged_total = 0
            _fc2_pass = True
            while _fc2_pass:
                _fc2_pass = False
                _fc2_trips_now = sorted(df[df['Trip'] > 0]['Trip'].unique())
                if len(_fc2_trips_now) <= _fl2_target:
                    break
                _fc2_info: dict = {}
                for _ft2 in _fc2_trips_now:
                    _ftd2 = df[df['Trip'] == _ft2]
                    _fc2_codes = _ftd2['Code'].unique().tolist()
                    _fc2_ranks = [vehicle_priority.get(branch_max_vehicle_cache.get(str(c).strip().upper(), '6W'), 3) for c in _fc2_codes]
                    _fc2_mn = min(_fc2_ranks) if _fc2_ranks else 3
                    _fc2_mv2 = {1: '4W', 2: 'JB', 3: '6W'}.get(_fc2_mn, '6W')
                    _fc2_is_pun = all(branch_bu_cache.get(c, False) for c in _fc2_codes)
                    _fc2_lim2 = PUNTHAI_LIMITS if _fc2_is_pun else LIMITS
                    _fc2_buf2 = punthai_buffer if _fc2_is_pun else maxmart_buffer
                    _fc2_w = float(_ftd2['Weight'].sum())
                    _fc2_c = float(_ftd2['Cube'].sum())
                    _fc2_nd2 = len(_fc2_codes)
                    _fc2_u = max(_fc2_w / (_fc2_lim2[_fc2_mv2]['max_w'] * _fc2_buf2),
                                 _fc2_c / (_fc2_lim2[_fc2_mv2]['max_c'] * _fc2_buf2)) if _fc2_mv2 in _fc2_lim2 else 0.0
                    _fc2_prov = str(_ftd2.iloc[0].get('_province', '') or '').strip()
                    _fc2_info[_ft2] = {
                        'codes': _fc2_codes, 'w': _fc2_w, 'c': _fc2_c, 'drops': _fc2_nd2,
                        'min_rank': _fc2_mn, 'mv': _fc2_mv2,
                        'is_pun': _fc2_is_pun, 'lim': _fc2_lim2, 'buf': _fc2_buf2,
                        'prov': _fc2_prov, 'is_bkk': _fc2_prov in _BKK_PROV_SET2,
                        'util': _fc2_u,
                    }
                _fc2_sorted = sorted(_fc2_info.keys(), key=lambda t: _fc2_info[t]['util'])
                for _ts2 in _fc2_sorted:
                    if _ts2 not in _fc2_info: continue
                    _tsi = _fc2_info[_ts2]
                    for _tb2 in _fc2_sorted:
                        if _tb2 == _ts2 or _tb2 not in _fc2_info: continue
                        _tbi = _fc2_info[_tb2]
                        if _tsi['is_bkk'] != _tbi['is_bkk']: continue
                        if not _tsi['is_bkk'] and _tsi['prov'] != _tbi['prov']: continue
                        _cw2 = _tsi['w'] + _tbi['w']
                        _cc2 = _tsi['c'] + _tbi['c']
                        _cd2 = len(set(_tsi['codes']) | set(_tbi['codes']))
                        _cmn2 = min(_tsi['min_rank'], _tbi['min_rank'])
                        _cpun2 = _tsi['is_pun'] and _tbi['is_pun']
                        _clim2 = PUNTHAI_LIMITS if _cpun2 else LIMITS
                        _cbuf2 = punthai_buffer if _cpun2 else maxmart_buffer
                        _fv2 = None
                        for _fvv, _fvr in [('4W', 1), ('JB', 2), ('6W', 3)]:
                            if _fvr > _cmn2: continue
                            _fl2 = _clim2[_fvv]
                            # FC2: ใช้ strict max_drops (ไม่ผ่อน) เพื่อป้องกัน enforcement re-split
                            if (_cw2 <= _fl2['max_w'] * _cbuf2 and
                                    _cc2 <= _fl2['max_c'] * _cbuf2 and
                                    _cd2 <= _fl2['max_drops']):
                                _fv2 = _fvv; break
                        if _fv2:
                            df.loc[df['Trip'] == _ts2, 'Trip'] = _tb2
                            _tbi['codes'] = list(set(_tsi['codes']) | set(_tbi['codes']))
                            _tbi['w'] = _cw2; _tbi['c'] = _cc2; _tbi['drops'] = _cd2
                            _tbi['min_rank'] = _cmn2; _tbi['is_pun'] = _cpun2
                            del _fc2_info[_ts2]
                            _fc2_merged_total += 1
                            safe_print(f"   🚛 FC2: Trip {_ts2}→{_tb2} [{_fv2}] {_cd2}drops {_cw2:.0f}kg {_cc2:.1f}m³ [{_tsi['prov']}]")
                            _fc2_pass = True; break
                    if _fc2_pass: break
            if _fc2_merged_total > 0:
                _fc2_final = len(df[df['Trip'] > 0]['Trip'].unique())
                safe_print(f"✅ FC2: รวม {_fc2_merged_total} merges → {_fc2_final} ทริป")
                _fc2_rem = sorted(df[df['Trip'] > 0]['Trip'].unique())
                _fc2_ren = {old: new for new, old in enumerate(_fc2_rem, start=1)}
                df['Trip'] = df['Trip'].map(lambda x: _fc2_ren.get(x, x) if x > 0 else x)
                # อัพเดต Truck label หลัง FC2
                for _fc2_t in df[df['Trip'] > 0]['Trip'].unique():
                    if _fc2_t not in trip_truck_map:
                        _fc2_td = df[df['Trip'] == _fc2_t]
                        _fc2_mv_list = [branch_max_vehicle_cache.get(str(c).strip().upper(), '6W') for c in _fc2_td['Code'].unique()]
                        _fc2_mr = min(vehicle_priority.get(v, 3) for v in _fc2_mv_list)
                        _fc2_tk = {1: '4W', 2: 'JB', 3: '6W'}.get(_fc2_mr, '6W')
                        df.loc[df['Trip'] == _fc2_t, 'Truck'] = f"{_fc2_tk} 🔀 FC2"
            else:
                safe_print(f"⚠️ FC2: ไม่มี merge เพิ่มเติม")

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
        trip_sort9_keys[trip_num] = (-(trip_max_distances[trip_num]),)  # ระยะทางไกลก่อน (ไม่ใช้อักษร)
    
    # เรียงทริปตาม ระยะทางไกลก่อน (ไม่ใช้ภาค/จังหวัด/อำเภอ)
    sorted_trips = sorted(trip_max_distances.keys(), key=lambda x: trip_sort9_keys.get(x, (0,)))
    
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
    
    # 📋 เรียงลำดับสาขาภายในทริป: ระยะทางจาก DC ไกลก่อน (ไม่ใช้ตัวอักษรจังหวัด/อำเภอ)
    if '_distance_from_dc' in df.columns:
        df = df.sort_values(['Trip', '_distance_from_dc'], ascending=[True, False]).reset_index(drop=True)
    else:
        df = df.sort_values('Trip', ascending=True).reset_index(drop=True)
    
    # ลบคอลัมน์ชั่วคราว (เก็บ _province, _district, _subdistrict, _max_vehicle, _lat, _lon, _distance_from_dc ไว้สำหรับแผนที่)
    cols_to_drop = ['_region_code', '_region_name', '_prov_code', '_dist_code', '_subdist_code', '_route', '_group_key', '_region_order', '_prov_max_dist', '_dist_max_dist', '_region_allowed_vehicles', '_vehicle_priority']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    return df, summary_df, fleet_used
def main():
    st.set_page_config(
        page_title="ระบบจัดเที่ยว",
        page_icon="🚚",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # ── Global white/green theme CSS ──────────────────────────────────────
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important; }

/* Main background */
.stApp { background: #f0fdf4 !important; }
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Header / Title */
h1 { color: #065f46 !important; font-weight: 800 !important; letter-spacing: -0.5px; }
h2, h3, h4 { color: #0f172a !important; font-weight: 700 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #ffffff;
    border: 1.5px solid #d1fae5;
    border-radius: 14px;
    padding: 14px 18px !important;
    box-shadow: 0 2px 10px rgba(5,150,105,.07);
    transition: box-shadow .2s;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 20px rgba(5,150,105,.14);
}
[data-testid="stMetricLabel"] { color: #6b7280 !important; font-size: 12px !important; font-weight: 600 !important; }
[data-testid="stMetricValue"] { color: #065f46 !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"] { font-weight: 600 !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: inherit !important;
    transition: all .18s !important;
    box-shadow: 0 2px 8px rgba(16,185,129,.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(16,185,129,.4) !important;
    filter: brightness(1.05) !important;
}
.stButton > button[kind="secondary"] {
    background: #f1f5f9 !important;
    color: #475569 !important;
    border: 1.5px solid #e2e8f0 !important;
    box-shadow: none !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #e2e8f0 !important;
    transform: translateY(-1px) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #ffffff;
    border-radius: 12px;
    padding: 4px;
    border: 1.5px solid #d1fae5;
    box-shadow: 0 2px 8px rgba(5,150,105,.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    padding: 8px 20px !important;
    transition: all .18s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 8px rgba(16,185,129,.35) !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* Expanders */
details[data-testid="stExpander"] {
    background: #ffffff;
    border: 1.5px solid #d1fae5 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 8px rgba(5,150,105,.06);
    margin-bottom: 8px;
}
details[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #065f46 !important;
    padding: 12px 16px !important;
    border-radius: 12px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 2px dashed #6ee7b7 !important;
    border-radius: 14px !important;
    padding: 12px !important;
    transition: border-color .2s;
}
[data-testid="stFileUploader"]:hover { border-color: #10b981 !important; }

/* Number inputs */
[data-testid="stNumberInput"] input {
    background: #f0fdf4 !important;
    border: 1.5px solid #bbf7d0 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,.12) !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #f0fdf4 !important;
    border: 1.5px solid #bbf7d0 !important;
    border-radius: 10px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1.5px solid #d1fae5 !important;
}

/* Success / Warning / Error / Info banners */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 4px !important;
    font-weight: 500 !important;
}

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(16,185,129,.3) !important;
}
[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(16,185,129,.45) !important;
}

/* Horizontal rule */
hr { border-color: #d1fae5 !important; }

/* Spinner */
[data-testid="stSpinner"] > div { border-top-color: #059669 !important; }

/* Progress bar */
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #059669, #10b981) !important; }

/* Checkbox */
[data-testid="stCheckbox"] label { font-weight: 600 !important; color: #374151 !important; }
input[type="checkbox"]:checked + span { background: #059669 !important; border-color: #059669 !important; }

/* Caption */
.stCaption { color: #6b7280 !important; font-size: 11px !important; }

/* Status bar area */
#status-bar { background: #ffffff; border-radius: 12px; padding: 12px 16px; border: 1.5px solid #d1fae5; }
</style>
""", unsafe_allow_html=True)
    
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
    status_cols = st.columns([3, 1, 1])
    with status_cols[0]:
        if SHEETS_AVAILABLE:
            st.success("📊 **Google Sheets:** เชื่อมต่อสำเร็จ | Auto-sync ทุก 5 นาที")
        else:
            st.warning("📊 **Data Source:** branch_data.json (local cache)")
    with status_cols[1]:
        st.metric("📍 Master Data", f"{len(MASTER_DATA):,} สาขา")
    with status_cols[2]:
        if st.button("🔄 ดึงข้อมูลใหม่", use_container_width=True, help="sync จาก Google Sheets และ rebuild MASTER_DATA_DICT"):
            with st.spinner("⏳ กำลังดึงข้อมูลจาก Google Sheets..."):
                try:
                    # ล้าง cache ทั้งหมด → module-level code จะ re-execute ตอน rerun
                    st.cache_data.clear()
                    # clear trip result เก่าออก (ต้องจัดเที่ยวใหม่กับข้อมูลล่าสุด)
                    for _k in ['trip_result', 'trip_summary', '_imap_html', '_imap_key',
                                'trip_result_excel', '_imap_build_time']:
                        st.session_state.pop(_k, None)
                    st.success("✅ ล้าง cache เรียบร้อย — กำลังโหลดใหม่...")
                except Exception as _re:
                    st.error(f"❌ ดึงข้อมูลไม่สำเร็จ: {_re}")
                finally:
                    st.rerun()
    
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
                tab1, tab2, tab3 = st.tabs([
                    "📦 จัดเที่ยว (ตามน้ำหนัก)",
                    "🗺️ จัดกลุ่มตามภาค",
                    "🏙️ โซนจัดส่งสาขา"
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
                    st.markdown("#### 🚛 จำนวนรถที่มี (ใส่ 0 = ไม่จำกัด)")
                    col_f1, col_f2, col_f3 = st.columns(3)
                    with col_f1:
                        fleet_4w = st.number_input("🚗 4W (คัน)", min_value=0, max_value=99, value=0, step=1, key="fleet_4w",
                                                   help="จำนวนรถ 4W ที่มีทั้งหมด (0 = ไม่จำกัด)")
                    with col_f2:
                        fleet_jb = st.number_input("🚚 JB (คัน)", min_value=0, max_value=99, value=0, step=1, key="fleet_jb",
                                                   help="จำนวนรถ JB ที่มีทั้งหมด (0 = ไม่จำกัด)")
                    with col_f3:
                        fleet_6w = st.number_input("🚛 6W (คัน)", min_value=0, max_value=99, value=0, step=1, key="fleet_6w",
                                                   help="จำนวนรถ 6W ที่มีทั้งหมด (0 = ไม่จำกัด)")
                    fleet_limits_input = {
                        '4W': int(fleet_4w) if fleet_4w > 0 else 999,
                        'JB': int(fleet_jb) if fleet_jb > 0 else 999,
                        '6W': int(fleet_6w) if fleet_6w > 0 else 999,
                    }

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
                            result_df, summary, fleet_used = predict_trips(
                                df_to_process,
                                model_data,
                                punthai_buffer=punthai_buffer_value,
                                maxmart_buffer=maxmart_buffer_value,
                                fleet_limits=fleet_limits_input
                            )

                            elapsed_time = time_module.time() - start_time
                            progress_bar.progress(90)

                            # 💾 เก็บผลลัพธ์ใน session_state เพื่อใช้ตอน export
                            st.session_state['trip_result'] = result_df
                            st.session_state['trip_summary'] = summary
                            st.session_state['fleet_used'] = fleet_used
                            st.session_state['fleet_limits'] = fleet_limits_input
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

                        # ── เรียงลำดับ result_df: ทริป → ระยะทางจาก DC (ไกลก่อน) ──
                        _rd_sort_cols = ['Trip']
                        _rd_sort_asc  = [True]
                        if '_distance_from_dc' in result_df.columns:
                            _rd_sort_cols.append('_distance_from_dc')
                            _rd_sort_asc.append(False)   # ไกลก่อนภายในทริปเดียวกัน
                        result_df = result_df.sort_values(
                            _rd_sort_cols,
                            ascending=_rd_sort_asc,
                            na_position='last'
                        ).reset_index(drop=True)
                        st.session_state['trip_result'] = result_df

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

                        # 🚛 Fleet Usage Summary
                        _fu = st.session_state.get('fleet_used', {})
                        _fl = st.session_state.get('fleet_limits', {})
                        _any_limit = any(v < 999 for v in _fl.values()) if _fl else False
                        if _fu:
                            st.markdown("---")
                            st.markdown("#### 🚛 สรุปการใช้รถ")
                            _fc1, _fc2, _fc3 = st.columns(3)
                            for _col, _vtype, _icon in [(_fc1, '4W', '🚗'), (_fc2, 'JB', '🚚'), (_fc3, '6W', '🚛')]:
                                _used = _fu.get(_vtype, 0)
                                _limit = _fl.get(_vtype, 999) if _fl else 999
                                _limit_str = str(_limit) if _limit < 999 else '∞'
                                _delta = f"/{_limit_str} คัน"
                                _over = _limit < 999 and _used > _limit
                                with _col:
                                    st.metric(f"{_icon} {_vtype}", f"{_used} ทริป", delta=_delta,
                                              delta_color="inverse" if _over else "normal")
                                    if _over:
                                        st.warning(f"⚠️ เกินโควต้า {_vtype}: ใช้ {_used}/{_limit}")
                        elif _any_limit:
                            st.info("ℹ️ ตั้งโควต้ารถไว้แล้ว — จัดทริปใหม่เพื่อดูผล")
                        
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

                        with st.expander("📋 ดูรายละเอียดรายสาขา (เรียงตามทริป → จังหวัด → อำเภอ)"):
                            # จัดเรียงคอลัมน์ที่สำคัญ
                            display_cols = ['Trip', 'Code', 'Name']
                            # จังหวัด
                            if '_province' in result_df.columns:
                                display_cols.append('_province')
                            elif 'Province' in result_df.columns:
                                display_cols.append('Province')
                            # อำเภอ
                            if '_district' in result_df.columns:
                                display_cols.append('_district')
                            elif 'District' in result_df.columns:
                                display_cols.append('District')
                            # ตำบล
                            if '_subdistrict' in result_df.columns:
                                display_cols.append('_subdistrict')
                            if 'Region' in result_df.columns:
                                display_cols.append('Region')
                            display_cols.extend(['Max_Distance_in_Trip', 'Weight', 'Cube', 'Truck', 'VehicleCheck'])
                            
                            # กรองคอลัมน์ที่มีอยู่จริง
                            display_cols = [col for col in dict.fromkeys(display_cols) if col in result_df.columns]
                            display_df = result_df[display_cols].copy()
                            
                            # ตั้งชื่อคอลัมน์ภาษาไทย
                            col_names = {'Trip': 'ทริป', 'Code': 'รหัส', 'Name': 'ชื่อสาขา',
                                       'Province': 'จังหวัด', '_province': 'จังหวัด',
                                       'District': 'อำเภอ', '_district': 'อำเภอ',
                                       '_subdistrict': 'ตำบล',
                                       'Region': 'ภาค', 'Max_Distance_in_Trip': 'ระยะทาง Max(km)', 
                                       'Weight': 'น้ำหนัก(kg)', 'Cube': 'คิว(m³)', 'Truck': 'รถ', 'VehicleCheck': 'ตรวจสอบรถ'}
                            display_df.columns = [col_names.get(c, c) for c in display_cols]
                            
                            # จัดรูปแบบคอลัมน์ระยะทาง
                            _fmt_disp = {}
                            if 'ระยะทาง Max(km)' in display_df.columns: _fmt_disp['ระยะทาง Max(km)'] = '{:.1f}'
                            if 'น้ำหนัก(kg)' in display_df.columns: _fmt_disp['น้ำหนัก(kg)'] = '{:.2f}'
                            if 'คิว(m³)' in display_df.columns: _fmt_disp['คิว(m³)'] = '{:.2f}'
                            st.dataframe(
                                display_df.style.format(_fmt_disp) if _fmt_disp else display_df,
                                width="stretch", 
                                height=500
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
                        _xl_sig = f"v7|{len(result_df)}|{int(result_df['Trip'].max())}|{sorted(result_df['Trip'].unique().tolist())}"
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
                                    if _dist_src_col:
                                        _pmx = float(_tg[_dist_src_col].max() or 0)
                                    else:
                                        _pmx = 0.0
                                    # เรียงตามระยะทางไกลก่อน (ไม่ใช้ภาค/จังหวัด/อำเภอ)
                                    trip_sort_keys[_tnum] = (-_pmx,)

                                sorted_trips = sorted(
                                    [t for t in result_df['Trip'].unique() if t != 0],
                                    key=lambda t: trip_sort_keys.get(t, (0,))
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
                                # เรียงแถว: ทริป → ระยะทางจาก DC ไกลก่อน (ไม่ใช้อักษร)
                                if _dist_src_col:
                                    _rd = _rd.sort_values(['_trip_order', _dist_src_col], ascending=[True, False])
                                else:
                                    _rd = _rd.sort_values(['_trip_order'])

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

                        # ── 🧠 AI LEARNING PANEL ──────────────────────────────────
                        with st.expander("🧠 ระบบ AI เรียนรู้การจัดทริป", expanded=False):
                            _ai_stats = get_trip_learning_stats()
                            _c1, _c2, _c3 = st.columns(3)
                            _c1.metric("Sessions ที่บันทึก", _ai_stats['sessions'])
                            _c2.metric("คู่สาขาที่เรียนรู้", f"{_ai_stats['unique_pairs']:,}")
                            _c3.metric("ทริปในระบบ", f"{sum(s.get('trips',0) for s in (json.load(open(TRIP_HISTORY_FILE,encoding='utf-8')).get('sessions',[]) if os.path.exists(TRIP_HISTORY_FILE) else []))}")

                            if st.button("💾 บันทึกการเรียนรู้จากทริปนี้", type="secondary", use_container_width=True):
                                _n_saved = save_trip_history(assigned_df)
                                st.success(f"✅ บันทึก {_n_saved:,} คู่สาขาสำเร็จ — AI จะใช้ข้อมูลนี้ในการจัดทริปครั้งถัดไป")

                            if _ai_stats['top_pairs']:
                                st.markdown("**คู่สาขาที่บ่อยที่สุด:**")
                                for _pk, _pv in _ai_stats['top_pairs'][:5]:
                                    _pa, _pb = _pk.split('|')
                                    st.caption(f"• {_pa} ↔ {_pb}: {_pv} ครั้ง")
                        # ────────────────────────────────────────────────────────

                        st.markdown("---")
                        
                        # 🗺️ แผนที่เส้นทาง (Interactive - Leaflet.js)
                        with st.expander("🗺️ แผนที่เส้นทาง (Interactive)", expanded=True):
                            try:
                                import importlib as _imp, trip_map_interactive as _tmi
                                _imp.reload(_tmi)
                                _build_imap = _tmi.build_interactive_map_html
                                import streamlit.components.v1 as _cmp2
                                import hashlib as _hl

                                _imap_sig = f"v29|{len(assigned_df)}|{int(assigned_df['Trip'].max())}|{sorted(assigned_df['Trip'].unique().tolist())}"
                                _imap_key = _hl.md5(_imap_sig.encode()).hexdigest()[:12]

                                if st.session_state.get('_imap_key') != _imap_key:
                                    with st.spinner("🗺️ กำลังสร้างแผนที่..."):
                                        _t_map = time_module.time()
                                        _imap_html = _build_imap(
                                            result_df=assigned_df,
                                            summary_df=summary,
                                            limits=LIMITS,
                                            punthai_limits=PUNTHAI_LIMITS,
                                            trip_no_map=trip_no_map,
                                            dc_lat=14.1459, dc_lon=100.6873,
                                            route_cache=ROUTE_CACHE_DATA,
                                        )
                                        st.session_state['_imap_html'] = _imap_html
                                        st.session_state['_imap_key'] = _imap_key
                                        st.session_state['_imap_build_time'] = time_module.time() - _t_map

                                # components.v1.html() — HTML inline ทุกอย่าง (CSS+JS) ไม่ต้อง load ภายนอก
                                _htm = st.session_state['_imap_html']
                                # Sanitize: drop any surrogate characters that break UTF-8 encoding
                                try:
                                    _htm.encode('utf-8')
                                except UnicodeEncodeError:
                                    _htm = _htm.encode('utf-8', errors='replace').decode('utf-8')
                                    st.session_state['_imap_html'] = _htm
                                import re as _re
                                _nb = len(_re.findall(r'"code":', _htm))
                                st.caption(f"🗺️ HTML: {len(_htm)//1024}KB · branches in HTML: {_nb} · _lat col: {'_lat' in assigned_df.columns} · valid coords: {int((assigned_df.get('_lat',0)>0).sum()) if '_lat' in assigned_df.columns else 0}")
                                _cmp2.html(_htm, height=840, scrolling=False)
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

                # ==========================================
                # แท็บ 3: โซนจัดส่งสาขา (geographic zone classification)
                # ==========================================
                with tab3:
                    st.markdown("### 🏙️ การจำแนกโซนจัดส่งของสาขา")
                    st.markdown(
                        "จัดทุกสาขาเข้าโซนจัดส่งตาม **ที่ตั้งทางภูมิศาสตร์** (ไม่คำนึงถึง Weight/Cube)  \n"
                        "กรุงเทพฯ แบ่ง **9 sub-zone** (ใจกลาง + 8 ทิศ)  •  จังหวัดอื่นอิง **LOGISTICS_ZONES** (เส้นทางถนน)"
                    )

                    if st.button("🔍 จำแนกโซนสาขาทั้งหมด", type="primary", key="btn_classify_zones"):
                        with st.spinner("กำลังจำแนกโซน..."):
                            _bz_map, _bz_summary = classify_all_branch_zones(MASTER_DATA)
                            st.session_state['branch_zone_map'] = _bz_map
                            st.session_state['branch_zone_summary'] = _bz_summary
                        st.success(f"✅ จำแนกเสร็จ: {len(_bz_map):,} สาขา / {len(_bz_summary)} โซน")

                    _bz_map = st.session_state.get('branch_zone_map', {})
                    _bz_summary = st.session_state.get('branch_zone_summary', {})

                    if _bz_summary:
                        _bz_colors = _build_zone_color_map(_bz_summary)

                        bkk_zones  = {k: v for k, v in _bz_summary.items() if k.startswith('BKK_')}
                        prov_zones = {k: v for k, v in _bz_summary.items()
                                      if not k.startswith('BKK_') and not k.startswith('UNCLASSIFIED')}
                        unc_zones  = {k: v for k, v in _bz_summary.items() if k.startswith('UNCLASSIFIED')}

                        # ── Metrics ──
                        _zm1, _zm2, _zm3, _zm4 = st.columns(4)
                        _zm1.metric("📍 สาขาทั้งหมด", f"{len(_bz_map):,}")
                        _zm2.metric("🗺️ โซนทั้งหมด", f"{len(_bz_summary)}")
                        _zm3.metric("🏙️ BKK Sub-zones", f"{len(bkk_zones)}")
                        _unc_total = sum(v['count'] for v in unc_zones.values())
                        _zm4.metric("❓ ไม่ระบุโซน", f"{_unc_total}")

                        # ── Map ──
                        st.markdown("---")
                        st.markdown("#### 🗺️ แผนที่แสดงโซนจัดส่ง (สีต่างกันต่างโซน — คลิก layer ด้านขวาเพื่อเปิด/ปิดโซน)")

                        if FOLIUM_AVAILABLE:
                            with st.spinner("กำลังสร้างแผนที่..."):
                                _zone_fmap = _build_zone_folium_map(MASTER_DATA, _bz_map, _bz_colors)
                            if _zone_fmap:
                                folium_static(_zone_fmap, width=1100, height=680)
                            else:
                                st.warning("ไม่สามารถสร้างแผนที่: ตรวจสอบว่า MASTER_DATA มีคอลัมน์ ละติจูด/ลองติจูด")
                        else:
                            st.warning("⚠️ ต้องติดตั้ง folium และ streamlit-folium เพื่อดูแผนที่")

                        # ── Zone Legend ──
                        st.markdown("---")
                        st.markdown("#### 🎨 Legend — สีและโซน")

                        _leg_tabs = st.tabs(["🏙️ กรุงเทพฯ (Sub-zones)", "🗺️ จังหวัด (Logistics Zones)", "❓ ไม่ระบุโซน"])

                        with _leg_tabs[0]:
                            if bkk_zones:
                                _bkk_order = ['BKK_CENTER','BKK_N','BKK_NE','BKK_E','BKK_SE',
                                              'BKK_S','BKK_SW','BKK_W','BKK_NW']
                                _bkk_cols = st.columns(3)
                                for _bi, _bk in enumerate([z for z in _bkk_order if z in bkk_zones]):
                                    _bv = bkk_zones[_bk]
                                    _bc = _bz_colors.get(_bk, '#888')
                                    _desc = BKK_SUBZONE_NAMES.get(_bk, _bk)
                                    with _bkk_cols[_bi % 3]:
                                        st.markdown(
                                            f'<div style="background:{_bc};color:#fff;border-radius:8px;'
                                            f'padding:10px 14px;margin:4px 0;font-size:13px;">'
                                            f'<b>{_bk}</b><br><span style="font-size:11px;">{_desc}</span>'
                                            f'<br><b>{_bv["count"]} สาขา</b></div>',
                                            unsafe_allow_html=True
                                        )
                            else:
                                st.info("ไม่พบข้อมูลกรุงเทพฯ")

                        with _leg_tabs[1]:
                            if prov_zones:
                                # Group by region for display
                                from collections import defaultdict as _ddleg
                                _leg_by_region = _ddleg(list)
                                for _zk, _zv in prov_zones.items():
                                    _pv = _zv.get('province', '')
                                    _rv = get_region_name(_pv) if _pv else 'ไม่ระบุ'
                                    _leg_by_region[_rv].append((_zk, _zv))

                                for _rv in ['เหนือ','อีสาน','ใต้','ตะวันออก','กลาง','ตะวันตก','ไม่ระบุ']:
                                    if _rv not in _leg_by_region:
                                        continue
                                    _rlist = sorted(_leg_by_region[_rv], key=lambda x: -x[1]['count'])
                                    with st.expander(f"📍 ภาค{_rv} ({len(_rlist)} โซน)", expanded=False):
                                        _rcols = st.columns(4)
                                        for _ri, (_zk, _zv) in enumerate(_rlist):
                                            _zc = _bz_colors.get(_zk, '#888')
                                            _zlabel = _zk.replace('ZONE_', '').replace('_', ' ')
                                            with _rcols[_ri % 4]:
                                                st.markdown(
                                                    f'<div style="background:{_zc};color:#fff;border-radius:6px;'
                                                    f'padding:6px 10px;margin:3px 0;font-size:12px;">'
                                                    f'<b>{_zlabel}</b><br>{_zv["count"]} สาขา</div>',
                                                    unsafe_allow_html=True
                                                )
                            else:
                                st.info("ไม่พบโซนจังหวัด")

                        with _leg_tabs[2]:
                            if unc_zones:
                                _unc_rows = [{'โซน (จังหวัดที่ไม่พบใน LOGISTICS_ZONES)': k,
                                              'จำนวนสาขา': v['count']} for k, v in unc_zones.items()]
                                st.dataframe(pd.DataFrame(_unc_rows), hide_index=True, use_container_width=True)
                            else:
                                st.success("✅ ทุกสาขามีโซนครบถ้วน")

                        # ── Zone Summary Table ──
                        st.markdown("---")
                        st.markdown("#### 📋 ตารางสรุปโซนทั้งหมด")
                        _sum_rows = []
                        for _zk, _zv in sorted(_bz_summary.items(), key=lambda x: (-x[1]['count'], x[0])):
                            _pv = _zv.get('province', '')
                            _rv = get_region_name(_pv) if _pv else ''
                            _zdesc = BKK_SUBZONE_NAMES.get(_zk, _zk.replace('ZONE_','').replace('_',' '))
                            _zc = _bz_colors.get(_zk, '#9E9E9E')
                            _sum_rows.append({
                                '🎨': f'<div style="background:{_zc};width:18px;height:18px;border-radius:4px;"></div>',
                                'Zone': _zk,
                                'คำอธิบาย': _zdesc,
                                'จังหวัด': _pv,
                                'ภาค': _rv,
                                'จำนวนสาขา': _zv['count'],
                            })
                        _sum_df = pd.DataFrame(_sum_rows)
                        st.dataframe(
                            _sum_df.drop(columns=['🎨']),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                'จำนวนสาขา': st.column_config.ProgressColumn(
                                    'จำนวนสาขา', format='%d สาขา',
                                    min_value=0,
                                    max_value=int(_sum_df['จำนวนสาขา'].max()) if len(_sum_df) > 0 else 1
                                )
                            }
                        )

                        # ── Downloads ──
                        st.markdown("---")
                        st.markdown("#### 📥 ดาวน์โหลดข้อมูล")
                        _dl1, _dl2 = st.columns(2)

                        with _dl1:
                            # Excel multi-sheet
                            with st.spinner("เตรียมไฟล์ Excel..."):
                                _excel_bytes = _build_zone_excel(MASTER_DATA, _bz_map, _bz_summary, _bz_colors)
                            st.download_button(
                                label="📊 ดาวน์โหลด Excel แยกโซน (หลายชีต)",
                                data=_excel_bytes,
                                file_name=f"branch_zones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                type="primary",
                                key="dl_zone_excel"
                            )

                        with _dl2:
                            # CSV quick export
                            _bz_csv_rows = []
                            _nm_map2 = {}
                            if not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns and 'สาขา' in MASTER_DATA.columns:
                                _nm_map2 = dict(zip(
                                    MASTER_DATA['Plan Code'].astype(str).str.strip(),
                                    MASTER_DATA['สาขา'].fillna('')))
                            for _code, _zone in sorted(_bz_map.items()):
                                _bz_csv_rows.append({
                                    'Plan Code': _code,
                                    'ชื่อสาขา': _nm_map2.get(_code, ''),
                                    'Zone': _zone,
                                    'Zone_Description': BKK_SUBZONE_NAMES.get(_zone, _zone),
                                })
                            _bz_csv = pd.DataFrame(_bz_csv_rows).to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="📄 ดาวน์โหลด CSV (ไฟล์เดียว)",
                                data=_bz_csv,
                                file_name=f"branch_zones_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                key="dl_zone_csv"
                            )
                    else:
                        st.info("กด **🔍 จำแนกโซนสาขาทั้งหมด** เพื่อเริ่มต้น")

if __name__ == "__main__":
    try:
        main()
    finally:
        # บันทึก cache ก่อนปิดโปรแกรม
        if USE_CACHE:
            save_distance_cache(DISTANCE_CACHE)
            save_route_cache(ROUTE_CACHE_DATA)
            safe_print(f"💾 บันทึก cache: {len(DISTANCE_CACHE)} ระยะทาง, {len(ROUTE_CACHE_DATA)} เส้นทาง")

