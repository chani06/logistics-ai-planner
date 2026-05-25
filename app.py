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
import re

# ฟังก์ชัน safe print สำหรับ Windows console
def safe_print(*args, **kwargs):
    """Print with fallback encoding for Windows console + append to UI log buffer"""
    text = ' '.join(str(arg) for arg in args)
    # ── append to session_state log buffer ──
    try:
        import streamlit as _st
        if hasattr(_st, 'session_state'):
            if '_ui_log' not in _st.session_state:
                _st.session_state['_ui_log'] = []
            _st.session_state['_ui_log'].append(text)
    except Exception:
        pass
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

def safe_join(values, sep=', '):
    """Join mixed-type values safely, skipping None/NaN."""
    if values is None:
        return ''
    cleaned = []
    for v in values:
        try:
            if v is None or bool(pd.isna(v)):
                continue
        except Exception:
            pass
        cleaned.append(str(v))
    return sep.join(cleaned)

def _safe_float(v, default=1):
    """แปลงค่าเป็น float อย่างปลอดภัย - ไม่ throw exception สำหรับ string ที่แปลงไม่ได้"""
    try:
        if v is None:
            return default
        val = float(v)
        return val
    except (ValueError, TypeError):
        return default

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

# นับ entry ใหม่ที่ยังไม่ได้ save (dirty counter)
_DIST_CACHE_DIRTY = 0
_DIST_CACHE_SAVE_BATCH = 50  # บันทึกทุก N entries ใหม่

_ROUTE_CACHE_DIRTY = 0
_ROUTE_CACHE_SAVE_BATCH = 10  # route แต่ละ entry ใหญ่กว่า → batch เล็กกว่า

def save_distance_cache(cache_dict, force=False):
    """บันทึก distance cache ลงไฟล์ — เฉพาะเมื่อมีการเปลี่ยนแปลงเท่านั้น"""
    global _DIST_CACHE_DIRTY
    if not force and _DIST_CACHE_DIRTY == 0:
        return  # ไม่มีการเปลี่ยนแปลง ไม่ต้อง save
    try:
        with open(DISTANCE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_dict, f, ensure_ascii=False, indent=2)
        _DIST_CACHE_DIRTY = 0
    except Exception as e:
        safe_print(f"⚠️ ไม่สามารถบันทึก distance cache: {e}")

def load_route_cache():
    """โหลด route cache จากไฟล์"""
    if os.path.exists(ROUTE_CACHE_FILE):
        try:
            with open(ROUTE_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_route_cache(cache_dict, force=False):
    """บันทึก route cache ลงไฟล์ — เฉพาะเมื่อมีการเปลี่ยนแปลง (dirty counter)"""
    global _ROUTE_CACHE_DIRTY
    if not force and _ROUTE_CACHE_DIRTY == 0:
        return
    try:
        with open(ROUTE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_dict, f, ensure_ascii=False, separators=(',', ':'))
        _ROUTE_CACHE_DIRTY = 0
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
    global SHEETS_AVAILABLE, sh, gc

    # ── พยายาม reconnect ถ้า sh หรือ SHEETS_AVAILABLE เป็น False/None ──
    if not SHEETS_AVAILABLE or sh is None:
        try:
            _creds = None
            _scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
            if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
                from oauth2client.service_account import ServiceAccountCredentials as _SAC
                _creds = _SAC.from_json_keyfile_dict(dict(st.secrets['gcp_service_account']), _scope)
            elif os.path.exists('credentials.json'):
                from oauth2client.service_account import ServiceAccountCredentials as _SAC
                _creds = _SAC.from_json_keyfile_name('credentials.json', _scope)
            if _creds:
                import socket as _s2
                _pt = _s2.getdefaulttimeout()
                _s2.setdefaulttimeout(15)
                try:
                    import gspread as _gs
                    gc = _gs.authorize(_creds)
                    sh = gc.open_by_key('12DmIfECwVpsWfl8rl2r1A_LB4_5XMrmnmwlPUHKNU-o')
                    SHEETS_AVAILABLE = True
                    safe_print("✅ Reconnect Google Sheets สำเร็จ")
                finally:
                    _s2.setdefaulttimeout(_pt)
        except Exception as _re:
            safe_print(f"⚠️ Reconnect ล้มเหลว: {_re}")

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
    '4W': {'max_w': 2000, 'max_c': 7.0,  'max_drops': 999},   # Cube ≤ 5.0, W ≤ 1500 (ไม่จำกัดจุด)
    'JB': {'max_w': 3000, 'max_c': 10.0,  'max_drops': 999},   # Cube ≤ 7.0, W ≤ 3000 (ไม่จำกัดจุด)
    '6W': {'max_w': 5500, 'max_c': 25.0, 'max_drops': 999}    # Cube ≤ 20, W ≤ 5800 (ไม่จำกัดจุด)
}

# 🔒 ขีดจำกัดสำหรับ Punthai ล้วน — เฉพาะน้ำหนัก/คิว (ไม่จำกัดจุด)
PUNTHAI_LIMITS = {
    '4W': {'max_w': 2000, 'max_c': 7.0,  'max_drops': 999},   # Cube ≤ 5.0, W ≤ 1500 (ไม่จำกัดจุด)
    'JB': {'max_w': 3000, 'max_c': 10.0,  'max_drops': 999},   # Cube ≤ 7.0, W ≤ 3000 (ไม่จำกัดจุด)
    '6W': {'max_w': 5500, 'max_c': 25.0, 'max_drops': 999}  
}

#  Geographic Clustering Config
MAX_DISTRICT_DISTANCE_KM = 30  # คนละจังหวัด: ห่างกันเกิน 30km ไม่ควรรวมทริป (จังหวัดเดียวกันสามารถ 80km)

# Utilization Config (ใช้ buffer จากหน้าเว็บเท่านั้น ไม่ fix ตายตัว)
MIN_VEHICLE_UTILIZATION = 1.0  # เป้าหมาย: รถต้องเต็ม 100% (แสดง warning ถ้าต่ำกว่า)

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
    dist = haversine_distance(_BKK_CENTER_LAT, _BKK_CENTER_LON, lat, lon, use_osrm_cache=False)
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
    
    distance = haversine_distance(lat1, lon1, lat2, lon2, use_osrm_cache=False)
    
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
@st.cache_data(ttl=300, show_spinner=False)  # Cache 5 นาที (real-time เมื่อ Sheets เปลี่ยน)
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
        safe_print(f"📋 คอลัมน์ทั้งหมด ({len(df_from_sheets.columns)}): {safe_join(df_from_sheets.columns.tolist())}")
        
        # 🔍 Debug: แสดงคอลัมน์ที่เกี่ยวข้องกับรถ (ค้นหาแบบยืดหยุ่น)
        vehicle_cols = [
            'MaxTruckType', 'Max Truck Type', 'MaxVehicle', 'Max Vehicle', 
            'รถสูงสุด', 'Max_Truck_Type', 'max_truck', 'MaxTruck',
            'ข้อจำกัดรถ', 'Truck', 'truck_type', 'TruckType',
            'ประเภทรถ', 'Vehicle', 'vehicle_type', 'VehicleType'
        ]
        found_vehicle_cols = [col for col in vehicle_cols if col in df_from_sheets.columns]
        if found_vehicle_cols:
            safe_print(f"✅ พบคอลัมน์ข้อจำกัดรถ: {safe_join(found_vehicle_cols)}")
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
# โหลดจาก branch_groups.json (สร้างโดย precompute_branch_data.py ด้วย haversine ≤500m + ตำบล/อำเภอ/จังหวัดเดียวกัน)
# ==========================================
@st.cache_data(show_spinner=False)
def load_branch_groups():
    """
    โหลด branch_groups.json ที่สร้างด้วย haversine ≤500m + ตำบล/อำเภอ/จังหวัดเดียวกัน
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
        safe_print("⚠️ ไม่พบ branch_groups.json - รัน python precompute_branch_data.py ก่อน")
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
# �️ PRE-SEED DISTANCE CACHE จาก branch_clusters.json
# inject ระยะทาง pre-computed ทุกคู่ใน NEARBY_BRANCHES → DISTANCE_CACHE
# เพื่อให้ hot-path haversine_distance(use_osrm_cache=False) ได้ cache hit เสมอ
# (รันใน background thread ไม่บล็อก UI)
# ==========================================
def _preseed_distance_cache_from_clusters():
    """
    อ่านระยะทางจาก branch_clusters.json (pre-computed OSRM distances)
    และ inject เข้า DISTANCE_CACHE ทุกคู่ที่ยังไม่มี
    จากนั้น OSRM live สำหรับคู่ที่ไม่มี pre-computed distance (batch ทีละ 20 คู่)
    """
    if not USE_CACHE or not BRANCH_INFO or not NEARBY_BRANCHES:
        return
    import threading
    import time as _time

    def _run():
        global _DIST_CACHE_DIRTY
        _injected = 0
        _to_fetch: list = []  # [(key, lat1, lon1, lat2, lon2)]

        for code, nearby_list in NEARBY_BRANCHES.items():
            code_up = str(code).strip().upper()
            info1 = BRANCH_INFO.get(code_up, {})
            lat1 = info1.get('lat', 0)
            lon1 = info1.get('lon', 0)
            if not lat1 or not lon1:
                continue
            for item in nearby_list:
                if isinstance(item, dict):
                    nb_code = str(item.get('code', '')).strip().upper()
                    pre_dist = item.get('distance', None)
                else:
                    nb_code = str(item).strip().upper()
                    pre_dist = None
                if not nb_code:
                    continue
                info2 = BRANCH_INFO.get(nb_code, {})
                lat2 = info2.get('lat', 0)
                lon2 = info2.get('lon', 0)
                if not lat2 or not lon2:
                    continue
                ck  = f"{lat1:.4f},{lon1:.4f}_{lat2:.4f},{lon2:.4f}"
                ckr = f"{lat2:.4f},{lon2:.4f}_{lat1:.4f},{lon1:.4f}"
                if ck in DISTANCE_CACHE or ckr in DISTANCE_CACHE:
                    continue  # มีแล้ว
                if pre_dist and pre_dist > 0:
                    # inject จาก pre-computed ทันที
                    DISTANCE_CACHE[ck] = round(pre_dist, 2)
                    _DIST_CACHE_DIRTY += 1
                    _injected += 1
                else:
                    # ไม่มี pre_dist → inject haversine×1.35 ทันที (ไม่เรียก OSRM)
                    from math import radians, sin, cos, sqrt, atan2
                    R = 6371.0
                    phi1, phi2 = radians(lat1), radians(lat2)
                    dphi = radians(lat2 - lat1)
                    dlambda = radians(lon2 - lon1)
                    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
                    hav_dist = round(R * 2 * atan2(sqrt(a), sqrt(1-a)) * 1.35, 2)
                    DISTANCE_CACHE[ck] = hav_dist
                    _DIST_CACHE_DIRTY += 1
                    _injected += 1

        # inject DC → ทุกสาขาใน BRANCH_INFO (haversine×1.35, zero-network)
        _dc_injected = 0
        _dc_lat = DC_WANG_NOI_LAT
        _dc_lon = DC_WANG_NOI_LON
        from math import radians as _r, sin as _s, cos as _c, sqrt as _sq, atan2 as _at
        for _bc_code, _bc_info in BRANCH_INFO.items():
            _blat = _bc_info.get('lat', 0)
            _blon = _bc_info.get('lon', 0)
            if not _blat or not _blon:
                continue
            _ck_dc  = f"{_dc_lat:.4f},{_dc_lon:.4f}_{_blat:.4f},{_blon:.4f}"
            _ck_dcr = f"{_blat:.4f},{_blon:.4f}_{_dc_lat:.4f},{_dc_lon:.4f}"
            if _ck_dc in DISTANCE_CACHE or _ck_dcr in DISTANCE_CACHE:
                continue
            _phi1, _phi2 = _r(_dc_lat), _r(_blat)
            _dphi = _r(_blat - _dc_lat)
            _dlam = _r(_blon - _dc_lon)
            _a = _s(_dphi/2)**2 + _c(_phi1)*_c(_phi2)*_s(_dlam/2)**2
            _hav = round(6371.0 * 2 * _at(_sq(_a), _sq(1-_a)) * 1.35, 2)
            DISTANCE_CACHE[_ck_dc] = _hav
            _DIST_CACHE_DIRTY += 1
            _dc_injected += 1

        # save ถ้ามีของใหม่
        if _DIST_CACHE_DIRTY > 0:
            save_distance_cache(DISTANCE_CACHE, force=True)
        safe_print(f"🗄️ Pre-seed cache: nearby {_injected} + DC→สาขา {_dc_injected} คู่ ({len(DISTANCE_CACHE):,} total)")

    t = threading.Thread(target=_run, daemon=True, name="preseed-cache")
    t.start()

_preseed_distance_cache_from_clusters()

# ==========================================
# �🚀 PRE-COMPUTE: Distance Matrix & Nearby Branches
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
                    global _ROUTE_CACHE_DIRTY
                    _ROUTE_CACHE_DIRTY += 1
                    if _ROUTE_CACHE_DIRTY >= _ROUTE_CACHE_SAVE_BATCH:
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
                    global _ROUTE_CACHE_DIRTY
                    _ROUTE_CACHE_DIRTY += 1
                    if _ROUTE_CACHE_DIRTY >= _ROUTE_CACHE_SAVE_BATCH:
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


# ฟังก์ชันคำนวณ haversine แยกไว้ข้างนอกเพื่อลดการคำนวณซ้ำ
def _calculate_haversine_raw(lat1, lon1, lat2, lon2):
    """คำนวณระยะทาง haversine ดิบ (ไม่×1.35)"""
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

# LRU cache สำหรับเก็บผลลัพธ์ล่าสุด (ขนาดเล็กเพื่อ performance)
from functools import lru_cache
@lru_cache(maxsize=1000)
def _haversine_with_cache(lat1, lon1, lat2, lon2):
    """Haversine พร้อม LRU cache สำหรับความเร็ว"""
    return _calculate_haversine_raw(lat1, lon1, lat2, lon2)

def haversine_distance(lat1, lon1, lat2, lon2, use_osrm_cache=True):
    """
    คืนค่าระยะทางถนน (km)
    ลำดับ:
      1. DISTANCE_CACHE → คืนค่าระยะทางถนนจริงทันที (เร็วที่สุด, ทั้งสองโหมด)
      2. Cache miss + use_osrm_cache=True  → OSRM live (6s), cache ผล, fallback haversine×1.35
      3. Cache miss + use_osrm_cache=False → haversine×1.35 ทันที (zero-latency, hot-path)
    """
    # 1. ตรวจ DISTANCE_CACHE ก่อนเสมอ (ทั้งสองโหมดได้ระยะทางจริงถ้ามีแคช)
    # ใช้ tuple เป็น key แทน string เพื่อความเร็ว
    cache_tuple = (round(lat1, 4), round(lon1, 4), round(lat2, 4), round(lon2, 4))
    cache_tuple_reverse = (round(lat2, 4), round(lon2, 4), round(lat1, 4), round(lon1, 4))
    
    # สร้าง string key สำหรับ DISTANCE_CACHE (ต้องใช้ string format เดิมเพื่อความเข้ากันได้)
    cache_key = f"{cache_tuple[0]:.4f},{cache_tuple[1]:.4f}_{cache_tuple[2]:.4f},{cache_tuple[3]:.4f}"
    cache_key_reverse = f"{cache_tuple_reverse[0]:.4f},{cache_tuple_reverse[1]:.4f}_{cache_tuple_reverse[2]:.4f},{cache_tuple_reverse[3]:.4f}"

    if USE_CACHE:
        if cache_key in DISTANCE_CACHE:
            return DISTANCE_CACHE[cache_key]
        if cache_key_reverse in DISTANCE_CACHE:
            return DISTANCE_CACHE[cache_key_reverse]

    # 2a. Cache miss + hot-path → haversine×1.35 ทันที (ไม่ network เด็ดขาด)
    if not use_osrm_cache:
        # ใช้ LRU cache สำหรับ haversine การคำนวณ
        dist = _haversine_with_cache(*cache_tuple)
        return round(dist * 1.35, 2)

    # 2b. Cache miss + precision → OSRM live, cache ไว้
    try:
        _url = (
            f"http://router.project-osrm.org/table/v1/driving/"
            f"{lon1},{lat1};{lon2},{lat2}?annotations=distance"
        )
        _r = requests.get(_url, timeout=6)
        _data = _r.json()
        if _data.get("code") == "Ok":
            _dist_m = _data["distances"][0][1]
            if _dist_m and _dist_m > 0:
                dist_km = round(_dist_m / 1000.0, 2)
                if USE_CACHE:
                    DISTANCE_CACHE[cache_key] = dist_km
                    global _DIST_CACHE_DIRTY
                    _DIST_CACHE_DIRTY += 1
                    if _DIST_CACHE_DIRTY >= _DIST_CACHE_SAVE_BATCH:
                        save_distance_cache(DISTANCE_CACHE)
                return dist_km
    except Exception:
        pass

    # 3. OSRM ล้มเหลว/timeout → haversine×1.35 (fallback)
    # ใช้ LRU cache สำหรับ haversine การคำนวณ
    dist = _haversine_with_cache(*cache_tuple)
    return round(dist * 1.35, 2)

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
        # suppress model version mismatch warning in UI
        return model_data
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def _extract_header_info(file_content):
    """
    อ่าน header row จากไฟล์ Excel ต้นฉบับ
    คืนค่า [(col_name, '#RRGGBB'), ...] — ชื่อและสีพื้นหลังของแต่ละ header cell
    ใช้ openpyxl เพื่อดึงสีที่แท้จริง
    """
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(file_content), data_only=True, read_only=False)
        # เลือก sheet เดียวกับ load_excel
        ws = None
        for sn in wb.sheetnames:
            if 'punthai' in sn.lower() or '2.' in sn.lower():
                ws = wb[sn]
                break
        if ws is None:
            ws = wb.active
        # หา header row (ใช้เงื่อนไขเดียวกับ load_excel)
        hrow = 1
        for ri, row in enumerate(ws.iter_rows(min_row=1, max_row=min(10, ws.max_row)), start=1):
            vals = ' '.join(str(c.value or '').upper() for c in row)
            if sum(kw in vals for kw in ('BRANCH', 'TRIP', 'รหัสสาขา', 'BU')) >= 2:
                hrow = ri
                break
        result = []
        for cell in ws[hrow]:
            if cell.column > ws.max_column:
                break
            name = str(cell.value) if cell.value is not None else ''
            color = '#D9D9D9'  # fallback grey
            try:
                fill = cell.fill
                if fill and fill.fill_type == 'solid' and fill.fgColor:
                    fg = fill.fgColor
                    if fg.type == 'rgb' and fg.rgb and len(fg.rgb) >= 6:
                        rgb_hex = fg.rgb[-6:]   # AARRGGBB → RRGGBB
                        if rgb_hex.upper() not in ('FFFFFF', '000000'):
                            color = '#' + rgb_hex
            except Exception:
                pass
            result.append((name, color))
        wb.close()
        return result
    except Exception as ex:
        safe_print(f"⚠️ _extract_header_info: {ex}")
        return []


def _extract_style_info(file_content):
    """
    อ่าน row height + font จากแถวข้อมูลแรกของไฟล์ต้นฉบับ
    คืน {'row_height': float, 'font_name': str, 'font_size': float}
    """
    result = {'row_height': 15.0, 'font_name': 'Angsana New', 'font_size': 14.0}
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(file_content), data_only=True, read_only=False)
        ws = None
        for sn in wb.sheetnames:
            if 'punthai' in sn.lower() or '2.' in sn.lower():
                ws = wb[sn]
                break
        if ws is None:
            ws = wb.active
        # หา header row
        hrow = 1
        for ri, row in enumerate(ws.iter_rows(min_row=1, max_row=min(10, ws.max_row)), start=1):
            vals = ' '.join(str(c.value or '').upper() for c in row)
            if sum(kw in vals for kw in ('BRANCH', 'TRIP', 'รหัสสาขา', 'BU')) >= 2:
                hrow = ri
                break
        # อ่านแถวข้อมูลแรก (หลัง header)
        data_row_idx = hrow + 1
        rd = ws.row_dimensions.get(data_row_idx)
        if rd and rd.height:
            result['row_height'] = float(rd.height)
        # อ่าน font จาก cell แรกที่มีข้อมูล
        for cell in ws[data_row_idx]:
            if cell.value is not None:
                try:
                    if cell.font:
                        if cell.font.name:
                            result['font_name'] = cell.font.name
                        if cell.font.size:
                            result['font_size'] = float(cell.font.size)
                except Exception:
                    pass
                break
        wb.close()
    except Exception as ex:
        safe_print(f"⚠️ _extract_style_info: {ex}")
    return result


def _extract_dc_row_info(file_content):
    """
    อ่านแถว DC (DC011/PTDC) จากไฟล์ต้นฉบับ
    คืน dict {orig_col_name: value} ของแถวนั้น
    """
    try:
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(file_content), data_only=True)
        ws = None
        for sn in wb.sheetnames:
            if 'punthai' in sn.lower() or '2.' in sn.lower():
                ws = wb[sn]
                break
        if ws is None:
            ws = wb.active
        # หา header row (เงื่อนไขเดียวกับ _extract_header_info)
        hrow = 1
        headers = []
        for ri, row in enumerate(ws.iter_rows(min_row=1, max_row=min(10, ws.max_row)), start=1):
            vals = ' '.join(str(c.value or '').upper() for c in row)
            if sum(kw in vals for kw in ('BRANCH', 'TRIP', 'รหัสสาขา', 'BU')) >= 2:
                hrow = ri
                headers = [str(c.value) if c.value is not None else '' for c in row]
                break
        if not headers:
            headers = [str(c.value) if c.value is not None else ''
                       for c in list(ws.iter_rows(min_row=hrow, max_row=hrow))[0]]
        # ค้นหาแถว DC
        _DC_CODES = {'DC011', 'PTDC', 'PTG DISTRIBUTION CENTER'}
        for row in ws.iter_rows(min_row=hrow + 1, max_row=ws.max_row):
            vals = [c.value for c in row]
            for v in vals[:6]:
                if str(v or '').strip().upper() in _DC_CODES:
                    row_dict = {headers[i]: ('' if vals[i] is None else vals[i])
                                for i in range(min(len(headers), len(vals)))}
                    wb.close()
                    return row_dict
        wb.close()
    except Exception as ex:
        safe_print(f"⚠️ _extract_dc_row_info: {ex}")
    return {}


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
            row_list = [str(v) for v in df_temp.iloc[i]]
            row_upper = ' '.join(row_list).upper()
            match_count = sum([
                'BRANCH' in row_upper,
                'TRIP' in row_upper,
                'รหัสสาขา' in ' '.join(row_list)
            ])
            if match_count >= 2:
                header_row = i
                break
        
        df = pd.read_excel(xls, sheet_name=target_sheet, header=header_row)
        df = df.loc[:, ~df.columns.duplicated()]
        
        return df
    except Exception as e:
        import traceback as _tb
        st.error(f"❌ Error: {e}")
        safe_print(f"❌ load_excel traceback: {_tb.format_exc()}")
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
    # ลำดับ 3 = รหัส WMS
    if len(col_list) > 3:
        rename_map[col_list[3]] = 'WMSCode'
    # ลำดับ 4 = สาขา/ชื่อ (Branch)
    if len(col_list) > 4:
        rename_map[col_list[4]] = 'Name'
    # ลำดับ 5 = TOTALCUBE
    if len(col_list) > 5:
        rename_map[col_list[5]] = 'Cube'
    # ลำดับ 6 = TOTALWGT
    if len(col_list) > 6:
        rename_map[col_list[6]] = 'Weight'
    # ลำดับ 7 = OriginalQTY
    if len(col_list) > 7:
        rename_map[col_list[7]] = 'OriginalQty'
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
        # WMSCode
        elif 'WMS' in col_upper or col_upper in ['WMSCODE', 'BRANCHCODEWMS', 'CODEWMS']:
            rename_map[col] = 'WMSCode'
        # OriginalQty
        elif col_upper in ['ORIGINALQTY', 'ORIGINALQUANTITY', 'ORIGINALQTY', 'ORIG_QTY', 'ORIGQTY'] or 'ORIGINALQ' in col_upper:
            rename_map[col] = 'OriginalQty'
        # Booking
        elif 'BOOKING' in col_upper:
            rename_map[col] = 'Booking'
    
    df = df.rename(columns=rename_map)

    # บันทึก rename_map สำหรับ export (original col name → internal name)
    try:
        st.session_state['_col_rename_map'] = dict(rename_map)
    except Exception:
        pass

    # ลบคอลัมน์ซ้ำ
    df = df.loc[:, ~df.columns.duplicated()]
    
    if 'Code' in df.columns:
        df['Code'] = df['Code'].apply(normalize)
        
        # ตัดสาขาที่ไม่ต้องการออก (รหัส)
        df = df[~df['Code'].isin(EXCLUDE_BRANCHES)]
        
        # ตัดสาขาที่ชื่อมี keyword ที่ไม่ต้องการ
        if 'Name' in df.columns:
            _excl_kws = [re.escape(str(n)) for n in EXCLUDE_NAMES
                         if n is not None and not (isinstance(n, float) and pd.isna(n))]
            if _excl_kws:
                exclude_pattern = '|'.join(_excl_kws)
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

def predict_trips(test_df, model_data, punthai_buffer=1.0, maxmart_buffer=1.10, fleet_limits=None, max_qty_per_trip=0):
    """
    จัดทริปแบบใหม่ - เรียบง่ายและมีประสิทธิภาพ
    
    หลักการ:
    1. เรียงตาม: ภาค → จังหวัด → อำเภอ → ตำบล → Route (ใช้ระยะทางจากพิกัดจริง)
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
    # Step 1: สร้าง location_map จากข้อมูล MASTER_DATA (Google Sheets) + พิกัด
    # ==========================================
    location_map = {}  # {code: {province, district, subdistrict, route, lat, lon, distance_from_dc, region_name}}
    
    for _, row in test_df.iterrows():
        code = str(row.get('Code', '')).strip().upper()
        if not code:
            continue
        
        province = ''
        district = ''
        subdistrict = ''
        route = ''
        
        # ลองหาจาก MASTER_DATA ก่อน (ข้อมูลล่าสุดจาก Sheets)
        if isinstance(model_data, pd.DataFrame) and not model_data.empty and 'Plan Code' in model_data.columns:
            master_row = model_data[model_data['Plan Code'] == code]
            if not master_row.empty:
                province    = str(master_row.iloc[0].get('จังหวัด', '')).strip() if pd.notna(master_row.iloc[0].get('จังหวัด')) else ''
                district    = str(master_row.iloc[0].get('อำเภอ', '')).strip()   if pd.notna(master_row.iloc[0].get('อำเภอ'))   else ''
                subdistrict = str(master_row.iloc[0].get('ตำบล', '')).strip()    if pd.notna(master_row.iloc[0].get('ตำบล'))    else ''
                route       = str(master_row.iloc[0].get('Route', '')).strip()   if pd.notna(master_row.iloc[0].get('Route'))   else ''
        
        # Fallback → Excel upload
        if not province:    province    = str(row.get('Province', '')).strip()    if pd.notna(row.get('Province'))    else ''
        if not district:    district    = str(row.get('District', '')).strip()    if pd.notna(row.get('District'))    else ''
        if not subdistrict: subdistrict = str(row.get('Subdistrict', '')).strip() if pd.notna(row.get('Subdistrict')) else ''
        if not route:       route       = str(row.get('Route', '')).strip()       if pd.notna(row.get('Route'))       else ''
        
        # normalize
        _prov_alias = {'พระนครศรีอยุธยา':'อยุธยา','กรุงเทพฯ':'กรุงเทพมหานคร',
                       'กทม':'กรุงเทพมหานคร','กทม.':'กรุงเทพมหานคร','โคราช':'นครราชสีมา'}
        province    = _prov_alias.get(clean_name(province), clean_name(province))
        district    = clean_name(district)
        subdistrict = clean_name(subdistrict)
        
        # พิกัด
        lat = 0; lon = 0
        for lat_col in ['Latitude', 'latitude', 'ละติจูด', 'lat', 'ละ']:
            if lat_col in row and pd.notna(row[lat_col]):
                try: lat = float(row[lat_col]); break
                except: pass
        for lon_col in ['Longitude', 'longitude', 'ลองจิจูด', 'ลองติจูด', 'lon', 'long', 'ลอง']:
            if lon_col in row and pd.notna(row[lon_col]):
                try: lon = float(row[lon_col]); break
                except: pass
        
        # Fallback พิกัดจาก MASTER_DATA
        if (lat == 0 or lon == 0) and not MASTER_DATA.empty and 'Plan Code' in MASTER_DATA.columns:
            master_row = MASTER_DATA[MASTER_DATA['Plan Code'] == code]
            if not master_row.empty:
                lat = float(master_row.iloc[0].get('ละติจูด', 0))  if pd.notna(master_row.iloc[0].get('ละติจูด'))  else 0
                lon = float(master_row.iloc[0].get('ลองติจูด', 0)) if pd.notna(master_row.iloc[0].get('ลองติจูด')) else 0
        
        # ระยะทางจาก DC (haversine จากพิกัด)
        dist_from_dc = haversine_distance(DC_WANG_NOI_LAT, DC_WANG_NOI_LON, lat, lon, use_osrm_cache=False) if (lat and lon) else 9999
        
        location_map[code] = {
            'province': province,
            'district': district,
            'subdistrict': subdistrict,
            'route': route,
            'lat': lat,
            'lon': lon,
            'distance_from_dc': dist_from_dc,
            'region_name': get_region_name(province)
        }
    
    # ==========================================
    # Step 2: เพิ่มข้อมูลพื้นที่ให้แต่ละสาขา (pd.merge แบบ manual)
    # ==========================================
    df = test_df.copy()
    
    def get_location_info(code):
        code_upper = str(code).strip().upper()
        return location_map.get(code_upper, {
            'province': '', 'district': '', 'subdistrict': '', 'route': '',
            'lat': 0, 'lon': 0,
            'distance_from_dc': 9999,
            'region_name': 'ไม่ระบุ'
        })
    
    # เพิ่มคอลัมน์ข้อมูลพื้นที่
    _loc_cache = {str(c).strip().upper(): get_location_info(c) for c in df['Code'].unique()}
    df['_region_name']      = df['Code'].map(lambda c: _loc_cache.get(str(c).strip().upper(), {}).get('region_name', 'ไม่ระบุ'))
    df['_province']         = df['Code'].map(lambda c: _loc_cache.get(str(c).strip().upper(), {}).get('province', ''))
    df['_district']         = df['Code'].map(lambda c: _loc_cache.get(str(c).strip().upper(), {}).get('district', ''))
    df['_subdistrict']      = df['Code'].map(lambda c: _loc_cache.get(str(c).strip().upper(), {}).get('subdistrict', ''))
    df['_route']            = df['Code'].map(lambda c: _loc_cache.get(str(c).strip().upper(), {}).get('route', ''))
    df['_distance_from_dc'] = df['Code'].map(lambda c: _loc_cache.get(str(c).strip().upper(), {}).get('distance_from_dc', 9999))
    df['_lat']              = df['Code'].map(lambda c: _loc_cache.get(str(c).strip().upper(), {}).get('lat', 0))
    df['_lon']              = df['Code'].map(lambda c: _loc_cache.get(str(c).strip().upper(), {}).get('lon', 0))
    
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
    
    # คำนวณ Subdistrict Max Distance (ตำบลไหนมีจุดไกลสุด — แทน sum_code)
    subdist_max_dist = df.groupby(['_province', '_district', '_subdistrict'])['_distance_from_dc'].max().reset_index()
    subdist_max_dist.columns = ['_province', '_district', '_subdistrict', '_subdist_max_dist']
    df = df.merge(subdist_max_dist, on=['_province', '_district', '_subdistrict'], how='left')
    df['_subdist_max_dist'] = df['_subdist_max_dist'].fillna(9999)
    
    # 🎯 Sort: LIFO (Last In First Out) - ไกลส่งก่อน, ใกล้ส่งทีหลัง
    # Zone → Region → Province Max Dist → District Max Dist → Subdistrict Max Dist → Route → Distance
    df = df.sort_values(
        ['_zone_priority', '_region_order', '_prov_max_dist', '_dist_max_dist', '_subdist_max_dist', '_route', '_distance_from_dc'],
        ascending=[True, True, False, False, False, True, False]
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
        return f"L_{row['_subdistrict']}_{row['_district']}_{row['_province']}"
    
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
        
        # 🚨 STRICT ENFORCEMENT: ส่งคืนข้อจำกัดจริง ไม่คูณด้วย buffer
        # buffer จะถูกใช้เฉพาะตอนตรวจสอบเท่านั้น
        return {
            'max_w': lim.get('max_w', 6000),
            'max_c': lim.get('max_c', 20.0),
            'max_d': lim.get('max_drops', 999)
        }
    
    # Helper function: เลือกรถที่เหมาะสม (STRICT - บังคับข้อจำกัด)
    def select_vehicle_for_load(weight, cube, drops, is_punthai, allowed_vehicles, strict_constraint=True):
        """
        เลือกรถที่เหมาะสมตามโหลดและข้อจำกัด
        
        Logic: ใช้ buffer จากหน้าเว็บ (punthai_buffer, maxmart_buffer)
        - Punthai: buffer = 100% (ห้ามเกิน)
        - Maxmart: buffer = 110% (เกินได้ 10%)
        - strict_constraint=True: ห้ามใช้รถที่ใหญ่กว่า allowed_vehicles (บังคับข้อจำกัด)
        
        🚨 UPDATE: บังคับใช้ข้อจำกัดอย่างเคร่งครัดตามที่ผู้ใช้กำหนด
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
            
            # 🚨 STRICT ENFORCEMENT: ตรวจสอบให้แน่ใจว่าไม่เกินข้อจำกัดตามประเภทรถ
            # ไม่ใช้ buffer สำหรับการตรวจสอบครั้งแรก (บังคับตามข้อจำกัดจริง)
            if (weight <= lim['max_w'] and
                cube <= lim['max_c'] and
                drops <= lim.get('max_drops', 999)):
                return v
            
            # ถ้ายังไม่พบ ลองใช้ buffer (สำหรับ Maxmart เท่านั้น)
            if not is_punthai:
                if (weight <= lim['max_w'] * buffer_mult and
                    cube <= lim['max_c'] * buffer_mult and
                    drops <= lim.get('max_drops', 999)):
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
                dist = haversine_distance(trip_lat_mean, trip_lon_mean, row['_lat'], row['_lon'], use_osrm_cache=False)
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
    # เพื่อให้สาขาในรัศมี 100m ไปในทริปเดียวกัน ไม่ถูกแยก
    # รวมเกณฑ์: ≤100m | ชื่อเดียวกัน | เลขหลังตัวอักษรในรหัสเดียวกัน (≥3 หลัก)
    _NEARBY_GROUP_KM = 0.1   # 100m — พิกัดเดียวกัน (force-group)
    _df_codes_upper = {str(c).strip().upper() for c in df['Code'].tolist()}

    # สร้าง _rt_same_loc จาก NEARBY_BRANCHES (pre-computed) + haversine fallback
    _rt_same_loc: dict = {}   # code_upper → [code_upper, ...]

    # Pass 1: ใช้ NEARBY_BRANCHES (เร็ว) — รัศมี 100m
    for _nc in _df_codes_upper:
        if _nc in NEARBY_BRANCHES:
            _nearby_in_run = [
                nb_code for nb_code, nb_dist in NEARBY_BRANCHES[_nc]
                if nb_dist <= _NEARBY_GROUP_KM and nb_code in _df_codes_upper
            ]
            if _nearby_in_run:
                _rt_same_loc[_nc] = [_nc] + _nearby_in_run

    # Pass 2: สาขาที่ไม่อยู่ใน NEARBY_BRANCHES → ใช้ haversine จาก df พิกัด
    # 🚀 SPEED: vectorized dict comprehension แทน iterrows
    _df_lats_cm = df['_lat'].fillna(0).astype(float).to_numpy()
    _df_lons_cm = df['_lon'].fillna(0).astype(float).to_numpy()
    _df_codes_cm = df['Code'].str.strip().str.upper().tolist()
    _valid_cm = _df_lats_cm > 0
    _df_coord_map = {
        _df_codes_cm[i]: (_df_lats_cm[i], _df_lons_cm[i])
        for i in range(len(_df_codes_cm)) if _valid_cm[i]
    }

    for _nc in _df_codes_upper:
        if _nc not in _rt_same_loc and _nc in _df_coord_map:
            _nlat, _nlon = _df_coord_map[_nc]
            _nearby_hv = [
                _oc for _oc, (_olat, _olon) in _df_coord_map.items()
                if _oc != _nc and haversine_distance(_nlat, _nlon, _olat, _olon, use_osrm_cache=False) <= _NEARBY_GROUP_KM
            ]
            if _nearby_hv:
                _rt_same_loc[_nc] = [_nc] + _nearby_hv

    # Pass 3: ชื่อสาขาเดียวกัน (Name) → force-group
    # 🚀 SPEED: zip-based แทน iterrows
    _rt_name_map: dict = {}   # normalized_name → [code_upper, ...]
    for _n3, _c3 in zip(df['Name'].fillna('').astype(str).str.strip(), df['Code'].str.strip().str.upper()):
        if _n3:
            _rt_name_map.setdefault(_n3, []).append(_c3)
    for _nc, _mates in _rt_name_map.items():
        if len(_mates) > 1:
            for _m in _mates:
                if _m not in _rt_same_loc:
                    _rt_same_loc[_m] = [_m]
                for _other in _mates:
                    if _other != _m and _other not in _rt_same_loc[_m]:
                        _rt_same_loc[_m].append(_other)

    # Pass 4: พิกัดเดียวกัน ≤300 เมตร → group (แทนตัวเลขท้ายสาขา)
    _COORD_300_KM = 0.3  # 300 เมตร
    # สร้าง clusters ด้วย union-find: ทุกคู่ที่ห่าง ≤300m ถูก group ร่วมกัน
    _p4_parent: dict = {_nc: _nc for _nc in _df_codes_upper}
    def _p4_find(x):
        while _p4_parent[x] != x:
            _p4_parent[x] = _p4_parent[_p4_parent[x]]
            x = _p4_parent[x]
        return x
    def _p4_union(a, b):
        ra, rb = _p4_find(a), _p4_find(b)
        if ra != rb:
            _p4_parent[rb] = ra
    # 🚀 ใช้ NEARBY_BRANCHES (pre-computed & cached) แทน O(n²) scan
    # — เร็วกว่ามาก เพราะ NEARBY_BRANCHES คำนวณและ cache ไว้แล้ว
    _df_codes_set_p4 = set(_df_codes_upper)
    for _nc_a in _df_codes_upper:
        for _nb_code_p4, _nb_dist_p4 in NEARBY_BRANCHES.get(_nc_a, []):
            if _nb_dist_p4 <= _COORD_300_KM and _nb_code_p4 in _df_codes_set_p4:
                _p4_union(_nc_a, _nb_code_p4)
    # fallback O(n²) เฉพาะสาขาที่ไม่อยู่ใน NEARBY_BRANCHES (ปกติ = 0)
    _no_nb_p4 = [nc for nc in _df_codes_upper if nc not in NEARBY_BRANCHES and nc in _df_coord_map]
    if _no_nb_p4:
        _all_p4 = [nc for nc in _df_codes_upper if nc in _df_coord_map]
        for _nc_a in _no_nb_p4:
            _la, _lo = _df_coord_map[_nc_a]
            for _nc_b in _all_p4:
                if _nc_b == _nc_a:
                    continue
                _lb, _lob = _df_coord_map[_nc_b]
                if haversine_distance(_la, _lo, _lb, _lob, use_osrm_cache=False) <= _COORD_300_KM:
                    _p4_union(_nc_a, _nc_b)
    # รวม cluster เป็น dict: root → [codes]
    _p4_clusters: dict = {}
    for _nc in _df_codes_upper:
        _root = _p4_find(_nc) if _nc in _p4_parent else _nc
        _p4_clusters.setdefault(_root, []).append(_nc)
    for _root, _cl_codes in _p4_clusters.items():
        if len(_cl_codes) > 1:
            for _nc in _cl_codes:
                if _nc not in _rt_same_loc:
                    _rt_same_loc[_nc] = [_nc]
                for _other in _cl_codes:
                    if _other != _nc and _other not in _rt_same_loc[_nc]:
                        _rt_same_loc[_nc].append(_other)

    # Coord-300m force partners: ห้ามแยกทริปเด็ดขาด (พิกัดห่างกัน ≤300m)
    _suffix_force_partners: dict = {}  # code_upper → [partner_code_upper, ...]
    for _root, _cl_codes in _p4_clusters.items():
        if len(_cl_codes) > 1:
            for _nc in _cl_codes:
                _sfp_list = [c for c in _cl_codes if c != _nc]
                if _sfp_list:
                    _suffix_force_partners[_nc] = _sfp_list

    # 🏗️ Inject coord-300m groups into BRANCH_GROUPS/BRANCH_TO_GROUP (backend group system)
    _sfx_injected = 0
    for _root, _cl_codes in _p4_clusters.items():
        if len(_cl_codes) > 1:
            _sfx_gid = f'COORD300_{_root}'
            _sfx_all: list = list(_cl_codes)
            _sfx_old_gids: set = set()
            for _nc in _cl_codes:
                _og = BRANCH_TO_GROUP.get(_nc)
                if _og and _og != _sfx_gid:
                    _sfx_old_gids.add(_og)
                    for _om in BRANCH_GROUPS.get(_og, []):
                        if _om not in _sfx_all:
                            _sfx_all.append(_om)
            BRANCH_GROUPS[_sfx_gid] = _sfx_all
            for _nc in _sfx_all:
                BRANCH_TO_GROUP[_nc] = _sfx_gid
            for _og in _sfx_old_gids:
                if _og in BRANCH_GROUPS:
                    del BRANCH_GROUPS[_og]
            _sfx_injected += len(_cl_codes)
    if _sfx_injected:
        safe_print(f"🏗️ Coord-300m inject: {_sfx_injected} สาขา → BRANCH_GROUPS backend แล้ว")

    _rt_grp_count = sum(1 for v in _rt_same_loc.values() if len(v) > 1)
    safe_print(f"📍 Runtime nearby group (≤{_NEARBY_GROUP_KM:.0f}km): {_rt_grp_count} สาขามีเพื่อนร่วมทริป จาก {len(_df_codes_upper)} สาขา")
    if _suffix_force_partners:
        safe_print(f"🔗 Suffix-force pairs: {len(_suffix_force_partners)} สาขาที่ถูก lock ไว้กับเพื่อนร่วม suffix")

    def get_group_branches_rt(code: str) -> list:
        """รวม precomputed group (≤200m) + runtime nearby (≤10km)"""
        code_upper = str(code).strip().upper()
        # precomputed group (≤500m + ตำบล/อำเภอ/จังหวัด จาก branch_groups.json)
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

    # ─── 🔀 PRE-MERGE: รวม suffix partners เป็นแถวเดียวก่อนจัดทริป ───────────────
    # เหตุผล: ป้องกันโซ่ greedy แยก suffix-group ออกจากกันโดยเด็ดขาด
    # Primary row = รับ Weight/Cube/Qty สะสม; Shadow rows = เก็บไว้ขยายคืนภายหลัง
    _shadow_rows: dict = {}   # primary_code_upper → [shadow_row_dict, ...]
    _premerge_done = 0
    _df_cu = df['Code'].apply(lambda x: str(x).strip().upper())
    # PRE-MERGE: ใช้ coord-300m clusters แทนตัวเลขท้ายสาขา
    _to_drop_indices: list = []
    for _root_pm, _pmcodes in _p4_clusters.items():
        # กรองเฉพาะ codes ที่อยู่ใน df จริง
        _df_cu_set_pm = set(_df_cu)
        _pmcodes_in_df = [c for c in _pmcodes if c in _df_cu_set_pm]
        if len(_pmcodes_in_df) < 2:
            continue
        # primary = คนที่อยู่ไกล DC มากที่สุด (seed ก่อน)
        _pm_dists = {}
        for _pmr_code in _pmcodes_in_df:
            _pmr = df[_df_cu == _pmr_code]
            if not _pmr.empty:
                _pmr_dict = _pmr.to_dict('records')[0]
                _pm_dists[_pmr_code] = float(_pmr_dict.get('_distance_from_dc', 0) or 0)
        _pm_primary = max(_pm_dists, key=_pm_dists.get)
        _pm_shadows = [c for c in _pmcodes_in_df if c != _pm_primary]
        # รวม weight/cube/qty จาก shadows เข้า primary
        _pm_primary_idx = df[_df_cu == _pm_primary].index
        if _pm_primary_idx.empty:
            continue
        _pm_pri_idx = _pm_primary_idx[0]
        for _pms_code in _pm_shadows:
            _pms_rows = df[_df_cu == _pms_code]
            if _pms_rows.empty:
                continue
            _pms_row = _pms_rows.iloc[0].to_dict()  # plain dict → ป้องกัน Series ambiguity
            # ไม่สะสม Weight/Cube/Qty ใน primary — primary เก็บของตัวเองเท่านั้น
            # (EXPAND คืน shadow rows กลับพร้อม weight เดิม → summary ถูกต้อง)
            # vehicle rank: ใช้ค่าที่ restrictive กว่า (รถเล็กสุดที่ group ต้องการ)
            _pms_vr = int(_pms_row.get('_vehicle_rank', 3))
            _cur_vr = int(df.loc[_pm_pri_idx, '_vehicle_rank'])
            if _pms_vr < _cur_vr:
                df.loc[_pm_pri_idx, '_vehicle_rank'] = _pms_vr
                df.loc[_pm_pri_idx, '_max_vehicle'] = _pms_row.get('_max_vehicle', '6W')
            # เก็บ shadow row ไว้ขยายคืน
            _shadow_rows.setdefault(_pm_primary, []).append(_pms_row)  # already plain dict
            # mark index ให้ drop
            _to_drop_indices.extend(list(_pms_rows.index))
        _premerge_done += len(_pm_shadows)
        safe_print(f"   🔀 PRE-MERGE coord-300m: primary={_pm_primary} รวม {_pm_shadows} ({len(_pm_shadows)} rows)")
    if _to_drop_indices:
        df = df.drop(index=_to_drop_indices).reset_index(drop=True)
        _df_cu = df['Code'].apply(lambda x: str(x).strip().upper())  # refresh
        safe_print(f"🔀 PRE-MERGE เสร็จ: รวม {_premerge_done} shadow rows → {len(_shadow_rows)} primary rows (df เหลือ {len(df)} แถว)")
        # 🔄 REBUILD branch_max_vehicle_cache หลัง PRE-MERGE (primary อาจถูกลด _max_vehicle แล้ว)
        # ถ้าไม่ rebuild → get_allowed_from_codes ยังใช้ค่าเก่าก่อน PRE-MERGE → อนุญาตรถใหญ่เกินจริง
        # 🚀 SPEED: zip แทน iterrows
        for _code_pv, _veh_pv in zip(df['Code'], df['_max_vehicle'].fillna('6W').astype(str)):
            branch_max_vehicle_cache[_code_pv] = _veh_pv if _veh_pv else '6W'
        safe_print(f"🔄 branch_max_vehicle_cache rebuilt: {len(branch_max_vehicle_cache)} entries หลัง PRE-MERGE")
    # ────────────────────────────────────────────────────────────────────────

    # รีเซ็ตทริปทั้งหมด
    df['Trip'] = 0
    trip_counter = 1

    # ─── 🚀 SPEED: สร้าง lookup dict ครั้งเดียว แทน df[df['Code'].apply(...)] ───
    # code_upper → plain dict row (ป้องกัน Series ambiguity จาก duplicate columns)
    _code_row_map: dict = {}   # str_upper → plain dict row
    _code_real_map: dict = {}  # str_upper → actual Code value in df
    for _cr in df.to_dict('records'):  # plain dicts — ไม่มีปัญหา Series ambiguous
        _cu = str(_cr.get('Code', '')).strip().upper()
        if _cu:
            _code_row_map[_cu] = _cr
            _code_real_map[_cu] = _cr.get('Code', '')
    # 🚀 SPEED: iloc index map — ให้ remaining_df/unassigned_df rebuild เป็น O(n_unassigned) แทน O(n_df)
    _code_upper_to_iloc: dict = {str(c).strip().upper(): i for i, c in enumerate(df['Code'])}

    # Coord-300m partners per code (ใช้แทน suffix map ใน group guard + lookahead)
    _code_coord300_partners: dict = {}  # str_upper → frozenset of partner codes ≤300m
    for _root_cp, _cl_cp in _p4_clusters.items():
        if len(_cl_cp) > 1:
            _cl_set = frozenset(_cl_cp)
            for _nc_cp in _cl_cp:
                _code_coord300_partners[_nc_cp] = _cl_set - {_nc_cp}

    # 🚀 SPEED: precompute province → region map (หลีกเลี่ยง get_region_name() ซ้ำๆ ใน loop)
    _prov_region_map: dict = {}
    for _cu2 in _code_row_map:
        _pv = str(_code_row_map[_cu2].get('_province', '') or '')
        if _pv and _pv not in _prov_region_map:
            _prov_region_map[_pv] = get_region_name(_pv)
    # ─────────────────────────────────────────────────────────────────────────

    # สร้าง set ของสาขาที่ยังไม่ได้จัด
    unassigned = set(df['Code'].tolist())

    # 3️⃣ เรียงลำดับ 1 ครั้ง (ไกลสุด→ใกล้สุด ตามโซน) แล้วเดิน pointer + epidemic frontier
    _sorted_start_codes = list(df.sort_values(
        ['_zone_priority', '_distance_from_dc'], ascending=[True, False]
    )['Code'])
    _sorted_start_ptr = 0   # pointer — เลื่อนไปข้างหน้าตลอด ไม่ reset
    _last_trip_all_coords: list = []   # พิกัดของทริปล่าสุด (ใช้หา seed ถัดไปแบบ epidemic)
    _last_trip_region: str = ''        # ภาคของทริปล่าสุด
    _last_trip_subdistricts: set = set()  # ตำบลของทริปล่าสุด (สำหรับ priority same-subdistrict)
    _last_trip_districts: set = set()     # อำเภอของทริปล่าสุด
    _last_trip_zone: str = ''          # logistics zone ของทริปล่าสุด (ใช้ตรวจ zone-exhaustion)
    _EPIDEMIC_NEXT_KM = 60             # ระยะสูงสุดที่ถือว่า frontier ของทริปก่อนยังต่อเนื่องกัน
    _lock_next_district: str = ''      # ล็อคอำเภอสำหรับทริปถัดไป (overflow district isolation)
    _wave_lat: float = 0.0  # 🌊 wave front lat — ตำแหน่งสาขาสุดท้ายของทริปก่อน (ไม่กระโดดไปตั้งต้นไกลใหม่)
    _wave_lon: float = 0.0  # 🌊 wave front lon

    while unassigned:
        _una_idxs = [_code_upper_to_iloc[str(c).strip().upper()] for c in unassigned if str(c).strip().upper() in _code_upper_to_iloc]
        unassigned_df = df.iloc[_una_idxs] if _una_idxs else df.iloc[[]]
        if unassigned_df.empty:
            break

        farthest_row = None

        # 🦠 Epidemic inter-trip: เริ่มทริปถัดไปจาก frontier ของทริปล่าสุด (ภาคเดียวกัน)
        # ถ้ามีสาขาใกล้ frontier ≤ _EPIDEMIC_NEXT_KM → ต่อเนื่อง (epidemic wave)
        # ถ้าไม่มี → เริ่มโซนใหม่จาก pointer (zone jump)
        #
        # 🔒 Zone-Exhaustion: ถ้าโซนล่าสุดยังมีสาขาเหลือ → ห้ามข้ามไปโซนอื่น
        # seed ต้องมาจากโซนเดิมก่อน จนโซนหมดถึงข้ามได้
        _zone_still_has_branches = (
            _last_trip_zone and
            not unassigned_df[unassigned_df['_logistics_zone'] == _last_trip_zone].empty
        )
        if _last_trip_all_coords:
            _ep_best = None
            _ep_best_dist = 999.0
            _ep_best_priority = 9  # 0=ตำบล, 1=อำเภอ, 2=nearest

            # 🚀 SPEED: vectorized haversine แทน iterrows (O(n × trip_coords) → numpy)
            _ep_df = (unassigned_df[unassigned_df['_logistics_zone'] == _last_trip_zone]
                      if _zone_still_has_branches else unassigned_df)
            if not _ep_df.empty:
                _ep_lats_np = _ep_df['_lat'].fillna(0).to_numpy(dtype=float)
                _ep_lons_np = _ep_df['_lon'].fillna(0).to_numpy(dtype=float)
                _ep_valid_m = _ep_lats_np > 0
                _ep_min_d_arr = np.full(len(_ep_df), 999.0)
                if _ep_valid_m.any() and _last_trip_all_coords:
                    _tc_arr_ep = np.array(_last_trip_all_coords, dtype=float)  # (n_trip, 2)
                    _vl_ep = np.radians(_ep_lats_np[_ep_valid_m])[:, np.newaxis]  # (n_v, 1)
                    _vo_ep = np.radians(_ep_lons_np[_ep_valid_m])[:, np.newaxis]
                    _tc_lr_ep = np.radians(_tc_arr_ep[:, 0])  # (n_trip,)
                    _tc_or_ep = np.radians(_tc_arr_ep[:, 1])
                    _a_ep = (np.sin((_vl_ep - _tc_lr_ep) / 2) ** 2 +
                             np.cos(_tc_lr_ep) * np.cos(_vl_ep) *
                             np.sin((_vo_ep - _tc_or_ep) / 2) ** 2)
                    _ep_min_d_arr[_ep_valid_m] = (
                        6371.0 * 2 * np.arctan2(np.sqrt(_a_ep), np.sqrt(1 - _a_ep)) * 1.35
                    ).min(axis=1)
                # วนเฉพาะ candidate ที่อยู่ในระยะ (มักจะน้อยมาก)
                _ep_in_range = np.where(_ep_min_d_arr <= _EPIDEMIC_NEXT_KM)[0]
                _ep_df_ri = _ep_df.reset_index(drop=True)
                _ep_records = _ep_df_ri.to_dict('records')  # plain dicts → ป้องกัน Series ambiguity
                for _i_ep in _ep_in_range:
                    _eprow = _ep_records[_i_ep]
                    _epd = float(_ep_min_d_arr[_i_ep])
                    _ep_cand_region = _prov_region_map.get(str(_eprow.get('_province', '') or ''), '')
                    if (_last_trip_region and _last_trip_region not in ('', 'ไม่ระบุ') and
                            _ep_cand_region and _ep_cand_region not in ('', 'ไม่ระบุ') and
                            _ep_cand_region != _last_trip_region):
                        continue  # ต่างภาค → ข้าม
                    _ep_sub = str(_eprow.get('_subdistrict', '') or '')
                    _ep_dis = str(_eprow.get('_district', '') or '')
                    if _last_trip_subdistricts and _ep_sub and _ep_sub in _last_trip_subdistricts:
                        _ep_prio = 0
                    elif _last_trip_districts and _ep_dis and _ep_dis in _last_trip_districts:
                        _ep_prio = 1
                    else:
                        _ep_prio = 2
                    if (_ep_prio < _ep_best_priority or
                            (_ep_prio == _ep_best_priority and _epd < _ep_best_dist)):
                        _ep_best_priority = _ep_prio
                        _ep_best_dist = _epd
                        _ep_best = _eprow
            if _ep_best is not None:
                farthest_row = _ep_best
            elif _zone_still_has_branches and farthest_row is None:
                # Epidemic ไม่มี candidate ในโซนเดิม (frontier ห่างเกิน _EPIDEMIC_NEXT_KM)
                # แต่ยังมีสาขาในโซนเดิม → บังคับเลือก seed ในโซนเดิมจาก pointer/sort
                _ep_same_zone = unassigned_df[
                    unassigned_df['_logistics_zone'] == _last_trip_zone
                ].sort_values('_distance_from_dc', ascending=False)
                if not _ep_same_zone.empty:
                    farthest_row = _ep_same_zone.iloc[0]
                    safe_print(f"   🔒 Zone-exhaust fallback: {farthest_row['Code']} (โซน {_last_trip_zone} ยังไม่หมด)")

        if farthest_row is None:
            if _wave_lat > 0 and _wave_lon > 0:
                # 🌊 Wave-ripple: seed ถัดไปจาก nearest ไปยัง wave front (ไม่กระโดดไปไกลใหม่)
                _wf_lats = unassigned_df['_lat'].fillna(0).to_numpy(dtype=float)
                _wf_lons = unassigned_df['_lon'].fillna(0).to_numpy(dtype=float)
                _wf_valid = _wf_lats > 0
                if _wf_valid.any():
                    _wf_dphi = np.radians(_wf_lats[_wf_valid] - _wave_lat)
                    _wf_dlam = np.radians(_wf_lons[_wf_valid] - _wave_lon)
                    _wf_a = (np.sin(_wf_dphi / 2) ** 2 +
                             np.cos(np.radians(_wave_lat)) * np.cos(np.radians(_wf_lats[_wf_valid])) *
                             np.sin(_wf_dlam / 2) ** 2)
                    _wf_d = 6371.0 * 2 * np.arctan2(np.sqrt(_wf_a), np.sqrt(1 - _wf_a)) * 1.35
                    _wf_best_i = int(np.argmin(_wf_d))
                    _wf_best_df_i = np.where(_wf_valid)[0][_wf_best_i]
                    farthest_row = unassigned_df.reset_index(drop=True).iloc[_wf_best_df_i]
                    safe_print(f"   🌊 Wave-seed #{trip_counter}: {farthest_row['Code']} (ห่าง wave front {_wf_d[_wf_best_i]:.1f}km)")
            if farthest_row is None:
                # ทริปแรก หรือไม่มี coord → sorted pointer (เริ่มจากไกลสุด)
                while _sorted_start_ptr < len(_sorted_start_codes):
                    _sc = _sorted_start_codes[_sorted_start_ptr]
                    _sorted_start_ptr += 1
                    if _sc in unassigned:
                        _frow_r = _code_row_map.get(str(_sc).strip().upper())
                        if _frow_r is not None:
                            farthest_row = _frow_r
                        break
        if farthest_row is None:
            farthest_row = unassigned_df.sort_values(
                ['_zone_priority', '_distance_from_dc'], ascending=[True, False]
            ).iloc[0]
        
        # ป้องกัน Series ambiguity จาก duplicate columns → แปลงเป็น plain dict ก่อนใช้งาน
        if hasattr(farthest_row, 'to_dict'):
            farthest_row = farthest_row.to_dict()

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
        # 🔒 District isolation: รับค่า lock จากทริปก่อน แล้วรีเซ็ตรอรอบถัดไป
        _current_trip_district_lock = _lock_next_district
        _lock_next_district = ''
        if _current_trip_district_lock:
            safe_print(f"   🔒 District-lock #{trip_counter}: จัดเฉพาะอำเภอ '{_current_trip_district_lock}' (overflow จากทริปก่อน)")
        
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
            _sr = _code_row_map.get(cu)
            if _sr is not None and start_lat > 0 and start_lon > 0:
                _slat = float(_sr.get('_lat', 0) or 0)
                _slon = float(_sr.get('_lon', 0) or 0)
                if _slat > 0 and _slon > 0:
                    return haversine_distance(_slat, _slon, start_lat, start_lon, use_osrm_cache=False)
            return 999.0
        start_group_unassigned.sort(key=_dist_from_start)

        # capacity limit สำหรับ start_group: ใช้ 6W base limit เท่านั้น (100%)
        _sg_is_pt = all(branch_bu_cache.get(str(c).strip().upper(), False) for c in [start_code])
        _sg_buf = 1.0
        _sg_lim = (PUNTHAI_LIMITS if _sg_is_pt else LIMITS)['6W']
        _sg_max_w = _sg_lim['max_w'] * _sg_buf
        _sg_max_c = _sg_lim['max_c'] * _sg_buf
        _sg_max_d = _sg_lim['max_drops']

        # คำนวณ weight/cube รวมทั้งกลุ่ม (พร้อม capacity check สำหรับสมาชิก 10km)
        trip_codes = []
        trip_weight = 0
        trip_cube = 0
        trip_qty = 0  # นับชิ้น (OriginalQty)
        for gc in start_group_unassigned:
            _gc_upper = str(gc).strip().upper()
            _gc_r = _code_row_map.get(_gc_upper)
            if _gc_r is not None:
                actual_code = _code_real_map.get(_gc_upper, gc)
                _gc_w = _safe_float(_gc_r.get('Weight', 0), 0)
                _gc_c = _safe_float(_gc_r.get('Cube', 0), 0)

                _gc_is_start = (_gc_upper == str(start_code).strip().upper())
                if not _gc_is_start:
                    _gc_dist_s = _dist_from_start(_gc_upper)
                    if _gc_dist_s > 0.3:
                        _sg_prov = str(_gc_r.get('_province', '') or '')
                        _sg_region = get_region_name(_sg_prov) if _sg_prov else ''
                        if trip_original_region and trip_original_region not in ('', 'ไม่ระบุ'):
                            if _sg_region and _sg_region not in ('', 'ไม่ระบุ') and _sg_region != trip_original_region:
                                safe_print(f"      🛑 START-GROUP GUARD: ตัด {actual_code} ภาค {_sg_region} ≠ {trip_original_region} (ห่าง {_gc_dist_s:.1f}km)")
                                continue
                        _sg_test_codes_chk = (trip_codes if trip_codes else [start_code]) + [actual_code]
                        _sg_allowed_chk = get_allowed_from_codes(_sg_test_codes_chk, ['4W', 'JB', '6W'])
                        if not _sg_allowed_chk:
                            safe_print(f"      ⛔ START-GROUP VEH: ตัด {actual_code} vehicle constraint ขัดแย้ง (ห่าง {_gc_dist_s:.1f}km)")
                            continue
                        # 🔒 ตรวจ capacity ก่อนเพิ่ม start-group member
                        _sg_test_is_pt = all(branch_bu_cache.get(c, False) for c in _sg_test_codes_chk)
                        _sg_test_lims = PUNTHAI_LIMITS if _sg_test_is_pt else LIMITS
                        _sg_test_buf = punthai_buffer if _sg_test_is_pt else maxmart_buffer
                        _sg_fits = any(
                            (trip_weight + _gc_w) <= _sg_test_lims[v]['max_w'] * _sg_test_buf and
                            (trip_cube + _gc_c) <= _sg_test_lims[v]['max_c'] * _sg_test_buf
                            for v in (_sg_allowed_chk or ['4W', 'JB', '6W'])
                        )
                        if not _sg_fits:
                            safe_print(f"      ⚠️ START-GROUP CAP: ตัด {actual_code} เกิน limit ({trip_weight+_gc_w:.0f}kg/{trip_cube+_gc_c:.2f}m³) (ห่าง {_gc_dist_s:.1f}km)")
                            continue
                        safe_print(f"      🔒 START-GROUP FORCE: {actual_code} (+{_gc_w:.0f}kg,+{_gc_c:.2f}m³) บังคับรวมกลุ่ม")
                trip_codes.append(actual_code)
                trip_weight += _gc_w
                trip_cube += _gc_c
                trip_qty += int(_safe_float(_gc_r.get('OriginalQty', 0), 0))
                if actual_code in unassigned:
                    unassigned.remove(actual_code)
                else:
                    for u in list(unassigned):
                        if str(u).strip().upper() == _gc_upper:
                            unassigned.remove(u)
                            break

        # 🔗 SUFFIX-FORCE (start_group): coord-300m partners + capacity check
        _trip_up_st = {str(t).strip().upper() for t in trip_codes}
        _unassigned_up_st = {str(u).strip().upper() for u in unassigned}
        _trip_is_pt_sf = all(branch_bu_cache.get(str(c).strip().upper(), False) for c in trip_codes)
        for _sfc in list(trip_codes):
            for _sfp in _suffix_force_partners.get(str(_sfc).strip().upper(), []):
                if _sfp in _trip_up_st:
                    continue
                if _sfp in _unassigned_up_st:
                    _sfp_r2 = _code_row_map.get(_sfp)
                    if _sfp_r2 is not None:
                        _sfp_actual = _code_real_map.get(_sfp, _sfp)
                        _sfp_w2 = _safe_float(_sfp_r2.get('Weight', 0), 0)
                        _sfp_c2 = _safe_float(_sfp_r2.get('Cube', 0), 0)
                        _sfp_is_pt2 = branch_bu_cache.get(_sfp, False)
                        _trip_min_priority2 = min(
                            vehicle_priority.get(branch_max_vehicle_cache.get(str(c).strip().upper(), '6W'), 3)
                            for c in trip_codes
                        ) if trip_codes else 3
                        _trip_capacity2 = {
                            'weight': trip_weight,
                            'cube': trip_cube,
                            'drops': len(trip_codes),
                            'min_priority': _trip_min_priority2,
                            'allowed_vehicle': {1: '4W', 2: 'JB', 3: '6W'}.get(_trip_min_priority2, '6W'),
                            'is_punthai': _trip_is_pt_sf,
                        }
                        can_add_sf, reason_sf = can_add_branch_to_trip(_sfp_r2, _trip_capacity2)
                        if not can_add_sf:
                            safe_print(f"      ⚠️ COORD-300m skip {_sfp_actual}: {reason_sf}")
                            continue
                        trip_codes.append(_sfp_actual)
                        _trip_up_st.add(_sfp)
                        trip_weight += _sfp_w2
                        trip_cube += _sfp_c2
                        trip_qty += int(_safe_float(_sfp_r2.get('OriginalQty', 0), 0))
                        _trip_is_pt_sf = _trip_is_pt_sf and _sfp_is_pt2
                        safe_print(f"      🔗 COORD-300m FORCE (start): {_sfp_actual} → บังคับเข้าทริปเดียวกับ {_sfc}")
                        if _sfp_actual in unassigned:
                            unassigned.remove(_sfp_actual)
                        else:
                            for _u in list(unassigned):
                                if str(_u).strip().upper() == _sfp:
                                    unassigned.remove(_u)
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

        # 🚀 Incremental caches สำหรับ inner while greedy loop (reset ทุกทริปใหม่)
        _incr_cand_dists: dict = {}       # code_upper → min dist from NEARBY_BRANCHES
        _incr_reach_nb: set = set()       # reach_codes from NEARBY_BRANCHES
        _incr_cross_nb: set = set()
        _incr_ultra_nb: set = set()
        _incr_trip_coords_all: list = []  # [(lat,lon),...] trip coords so far
        _incr_processed_tc: set = set()   # trip_codes already processed
        _incr_min_dist_all = np.full(len(df), 999.0)  # aligned with df.iloc
        # Pre-compute df lat/lon once (อ้างอิง _df_all_lats/lons ถ้าไม่มีให้สร้าง)
        _df_all_lats_g = df['_lat'].fillna(0).to_numpy(dtype=float)
        _df_all_lons_g = df['_lon'].fillna(0).to_numpy(dtype=float)
        _df_all_valid_g = _df_all_lats_g > 0
        # 🔗 Nearest-neighbor chain: ตามตำแหน่งสาขาสุดท้ายที่เพิ่มเข้าทริป
        _last_added_lat = float(farthest_row.get('_lat', 0) or 0)
        _last_added_lon = float(farthest_row.get('_lon', 0) or 0)

        # 2️⃣ Greedy: หาสาขาใกล้สุดมาเติมจนเต็ม buffer
        while unassigned:
            # 🔒 ถ้า trip เต็มตาม buffer แล้ว → ปิดทริปทันที ไม่รับสาขาอื่น
            _greedy_is_pt = all(branch_bu_cache.get(str(c).strip().upper(), False) for c in trip_codes)
            _greedy_lim = (PUNTHAI_LIMITS if _greedy_is_pt else LIMITS)
            _greedy_buf = punthai_buffer if _greedy_is_pt else maxmart_buffer
            _greedy_allowed = get_allowed_from_codes(trip_codes, ['4W', 'JB', '6W']) or ['6W']
            _greedy_veh = _greedy_allowed[-1]  # รถใหญ่สุดที่อนุญาต
            if (trip_weight >= _greedy_lim[_greedy_veh]['max_w'] * _greedy_buf or
                    trip_cube >= _greedy_lim[_greedy_veh]['max_c'] * _greedy_buf):
                safe_print(f"      🛑 TRIP FULL ({int(_greedy_buf*100)}% buffer): {trip_weight:.0f}kg/{trip_cube:.2f}m³ → ปิดทริป {trip_counter}")
                break
            _rem_idxs = [_code_upper_to_iloc[str(c).strip().upper()] for c in unassigned if str(c).strip().upper() in _code_upper_to_iloc]
            remaining_df = df.iloc[_rem_idxs] if _rem_idxs else df.iloc[[]]  # view — no copy
            if remaining_df.empty:
                break
            
            # ✅ รีเซ็ต same_zone_df ทุก iteration ป้องกัน stale value จาก iteration ก่อน
            same_zone_df = None
            filter_level = ""

            # ─────────────────────────────────────────────────────────────────
            # 📡 หลักการ "แพร่เชื่อกราย ทีละโซน" (Zone-Diffusion)
            #
            #  Level 1 — โซนเดียวกัน: เชื่อมต่อถึงกัน ≤ _CHAIN_KM จากสาขาใดในทริป
            #  Level 2 — ข้ามโซน:     ต้องอยู่ภายใน _CROSS_ZONE_KM และภาคเดียวกัน
            #
            #  ปิดทริปทันทีถ้าไม่มีสาขาผ่านทั้ง 2 ระดับ
            #  (ลบ fallback สาย province / logistics-zone / highway ออก —
            #   การ "กระโดด" ข้ามโซนที่ไม่ได้เชื่อมกันจริงทำให้ทริปกระจาย)
            # ─────────────────────────────────────────────────────────────────
            _CHAIN_KM      = 50   # ระยะสูงสุดเพื่อ "เชื่อมต่อ" ภายในโซน
            _CROSS_ZONE_KM = 20   # ระยะสูงสุดเพื่อข้ามโซน

            same_zone_df = None
            filter_level  = ""

            # ─── คำนวณ reach codes จาก NEARBY_BRANCHES + haversine ───
            unassigned_upper_set_vc = {str(c).strip().upper() for c in unassigned}
            reach_codes       = set()   # ≤ _CHAIN_KM จากสาขาใดในทริป
            cross_zone_codes  = set()   # ≤ _CROSS_ZONE_KM (ข้ามโซนได้)
            ultra_close_codes = set()   # < 8 km (bypass ทุก filter)

            # 🚀 Incremental: อัปเดตเฉพาะ trip_codes ใหม่ (ไม่ rebuild ทั้งหมดทุกรอบ)
            for _tc_i in trip_codes:
                _tc_u_i = str(_tc_i).strip().upper()
                if _tc_u_i in _incr_processed_tc:
                    continue
                _incr_processed_tc.add(_tc_u_i)
                _tr_i = _code_row_map.get(_tc_u_i)
                if _tr_i is not None:
                    _tl_i = float(_tr_i.get('_lat', 0) or 0)
                    _to_i = float(_tr_i.get('_lon', 0) or 0)
                    if _tl_i > 0 and _to_i > 0:
                        _incr_trip_coords_all.append((_tl_i, _to_i))
                        # อัปเดต min_dist สำหรับ df ทั้งหมด (vectorized)
                        _dphi_i2 = np.radians(_df_all_lats_g[_df_all_valid_g] - _tl_i)
                        _dlam_i2 = np.radians(_df_all_lons_g[_df_all_valid_g] - _to_i)
                        _a_i2 = (np.sin(_dphi_i2 / 2) ** 2 +
                                 np.cos(np.radians(_tl_i)) * np.cos(np.radians(_df_all_lats_g[_df_all_valid_g])) *
                                 np.sin(_dlam_i2 / 2) ** 2)
                        _d_i2 = 6371.0 * 2 * np.arctan2(np.sqrt(_a_i2), np.sqrt(1 - _a_i2)) * 1.35
                        _incr_min_dist_all[_df_all_valid_g] = np.minimum(_incr_min_dist_all[_df_all_valid_g], _d_i2)
                if _tc_u_i in NEARBY_BRANCHES:
                    for _nbc_i, _nbd_i in NEARBY_BRANCHES[_tc_u_i]:
                        if _nbd_i <= _CHAIN_KM:
                            _incr_reach_nb.add(_nbc_i)
                        if _nbd_i <= _CROSS_ZONE_KM:
                            _incr_cross_nb.add(_nbc_i)
                        if _nbd_i < 8.0:
                            _incr_ultra_nb.add(_nbc_i)
                        if _nbd_i < _incr_cand_dists.get(_nbc_i, float('inf')):
                            _incr_cand_dists[_nbc_i] = _nbd_i

            _trip_coords_reach = _incr_trip_coords_all  # compat reference

            # Build reach_codes จาก NEARBY_BRANCHES incremental + haversine incremental
            if _rem_idxs:
                _best_r_np = _incr_min_dist_all[np.array(_rem_idxs, dtype=int)]
                _rm_codes_np = remaining_df['Code'].str.strip().str.upper().to_numpy()
                reach_codes.update(_rm_codes_np[_best_r_np <= _CHAIN_KM].tolist())
                cross_zone_codes.update(_rm_codes_np[_best_r_np <= _CROSS_ZONE_KM].tolist())
                ultra_close_codes.update(_rm_codes_np[_best_r_np < 8.0].tolist())
            # เพิ่มจาก NEARBY_BRANCHES (filter เฉพาะ unassigned)
            reach_codes.update(c for c in _incr_reach_nb if c in unassigned_upper_set_vc)
            cross_zone_codes.update(c for c in _incr_cross_nb if c in unassigned_upper_set_vc)
            ultra_close_codes.update(c for c in _incr_ultra_nb if c in unassigned_upper_set_vc)

            # Level 0: ตำบล/อำเภอเดียวกัน → รวมใน reach เสมอ (ไม่มี distance limit)
            if trip_subdistricts and trip_districts:
                _sub0 = remaining_df[
                    remaining_df['_subdistrict'].isin(trip_subdistricts) &
                    remaining_df['_district'].isin(trip_districts)
                ]
                reach_codes.update(str(_rc0).strip().upper() for _rc0 in _sub0['Code'])
            elif trip_districts:
                _dis0 = remaining_df[remaining_df['_district'].isin(trip_districts)]
                reach_codes.update(str(_rc0).strip().upper() for _rc0 in _dis0['Code'])

            # Level 1: 🦠 Epidemic frontier
            if reach_codes:
                _rm_upper_ser = remaining_df['Code'].str.upper().str.strip()
                _sz_df = remaining_df[_rm_upper_ser.isin(reach_codes)].copy()
                if not _sz_df.empty:
                    same_zone_df = _sz_df
                    filter_level = f"epidemic-frontier({_CHAIN_KM}km)"

            # (Level 2 ถูกรวมเข้า Level 1 แล้ว — reach_codes ครอบคลุม ≤_CHAIN_KM ทุกทิศทาง)

            # 🔒 District lock: ทริปนี้ถูก lock ไว้กับอำเภอจากทริปก่อน
            # → กรองเฉพาะสาขาอำเภอนั้น ป้องกันอำเภอต่างกันปนกัน
            if _current_trip_district_lock and same_zone_df is not None:
                _dl_same = same_zone_df[same_zone_df['_district'] == _current_trip_district_lock]
                if not _dl_same.empty:
                    same_zone_df = _dl_same.copy()
                    filter_level += f"+district-lock({_current_trip_district_lock})"
                # ถ้าไม่มีสาขาจากอำเภอที่ lock → ปล่อยผ่าน (อาจปิดทริปในขั้นถัดไป)

            # Level 3: frontier หมด → ดึงสาขาที่ใกล้ที่สุดในโซนเดียวกัน (NEARBY_BRANCHES)
            # ถ้าไม่มีในโซนเดียวกัน → ลองโซนอื่นที่ใกล้ที่สุด (ภาคเดียวกัน)
            # หลักการ: "กระโดด" ไปหา seed ใหม่ในทริปเดิม แทนการปิดแล้วเปิดทริปใหม่
            if same_zone_df is None:
                _jump_found = False
                # รวม NEARBY_BRANCHES ของทุกสาขาในทริป → หาสาขาที่ยังไม่ได้จัด
                _nb_candidates: list = []  # [(dist, code_upper, zone)]
                for _tc in trip_codes:
                    _tc_upper = str(_tc).strip().upper()
                    if _tc_upper in NEARBY_BRANCHES:
                        for _nb_code, _nb_dist in NEARBY_BRANCHES[_tc_upper]:
                            if _nb_code in unassigned_upper_set_vc:
                                _nb_r = _code_row_map.get(_nb_code)
                                if _nb_r is not None:
                                    _nb_zone = str(_nb_r.get('_prov_zone', '') or '')
                                    _nb_prov_nb = str(_nb_r.get('_province', '') or '')
                                    _nb_region = _prov_region_map.get(_nb_prov_nb, get_region_name(_nb_prov_nb))
                                    _nb_candidates.append((_nb_dist, _nb_code, _nb_zone, _nb_region))
                # ลบ duplicates (เก็บ min dist ต่อ code)
                _nb_best: dict = {}
                for _nd, _nc, _nz, _nr in _nb_candidates:
                    if _nc not in _nb_best or _nd < _nb_best[_nc][0]:
                        _nb_best[_nc] = (_nd, _nz, _nr)
                # เรียง: โซนเดียวกันก่อน ภาคเดียวกันก่อน ใกล้สุดก่อน
                def _nb_sort_key(item):
                    _nc2, (_nd2, _nz2, _nr2) = item
                    _same_zone = 0 if (_nz2 == trip_prov_zone) else 1
                    _same_region = 0 if (not trip_original_region or not _nr2 or _nr2 == trip_original_region) else 2
                    return (_same_zone, _same_region, _nd2)
                _nb_sorted = sorted(_nb_best.items(), key=_nb_sort_key)
                for _nc2, (_nd2, _nz2, _nr2) in _nb_sorted:
                    # ตรวจ region compat (เข้มขึ้น: ถ้าไม่รู้ภาคทั้งคู่ → ตรวจ BKK เท่านั้น)
                    if (trip_original_region and trip_original_region not in ('', 'ไม่ระบุ') and
                            _nr2 and _nr2 not in ('', 'ไม่ระบุ') and _nr2 != trip_original_region):
                        continue
                    # 🔒 cluster-jump: ถ้าไม่รู้ภาคทริป แต่รู้ภาค candidate → หา region จาก province
                    if (not trip_original_region or trip_original_region == 'ไม่ระบุ') and _nr2 and _nr2 not in ('', 'ไม่ระบุ'):
                        # ยอมรับ cluster-jump ข้ามภาคได้แค่ ≤ 15km (สาขาใกล้มาก)
                        if _nd2 > 15.0:
                            continue
                    # ตรวจ BKK isolation
                    _nb_r2_dict = _code_row_map.get(_nc2)
                    if _nb_r2_dict is None:
                        continue
                    _nb_prov2 = str(_nb_r2_dict.get('_province', '') or '')
                    _BKK = 'กรุงเทพมหานคร'
                    if ((_nb_prov2 == _BKK and trip_original_province not in ('', None) and trip_original_province != _BKK) or
                            (trip_original_province == _BKK and _nb_prov2 and _nb_prov2 != _BKK)):
                        continue
                    # ✅ ดึง seed ใหม่เข้า same_zone_df (ต้องเป็น DataFrame)
                    _nb_r2_df = remaining_df[remaining_df['Code'].str.upper().str.strip() == _nc2]
                    if _nb_r2_df.empty:
                        continue
                    same_zone_df = _nb_r2_df.copy()
                    filter_level = f"cluster-jump({_nd2:.1f}km,zone={_nz2})"
                    safe_print(f"      🦘 Cluster-jump #{trip_counter}: seed ใหม่ {_nc2} zone={_nz2} ห่าง {_nd2:.1f}km จากทริป")
                    _jump_found = True
                    break
                if not _jump_found:
                    safe_print(f"      🛑 epidemic สิ้นสุด #{trip_counter}: ไม่มีสาขาเชื่อมต่อใน cluster ({len(trip_codes)} สาขา, {trip_weight:.0f}kg) → ปิดทริป")
                    break

            # ─── กรองภาค (ล็อคถ้าไม่รู้ภาค) ────────────────────────────────
            if not trip_original_region or trip_original_region in ('ไม่ระบุ', ''):
                # vectorized: นับภาคจาก province column
                _crn_ser2 = same_zone_df['_province'].map(
                    lambda p: _prov_region_map.get(str(p), get_region_name(str(p))) if p else ''
                )
                _valid_r2 = _crn_ser2[_crn_ser2.notna() & (_crn_ser2 != '') & (_crn_ser2 != 'ไม่ระบุ')]
                _cand_region_counts = _valid_r2.value_counts().to_dict()
                if _cand_region_counts:
                    # ล็อคภาคจาก candidate ที่มีมากที่สุด
                    trip_original_region = max(_cand_region_counts, key=_cand_region_counts.get)
                    safe_print(f"      🔒 ล็อคภาคอัตโนมัติ #{trip_counter}: '{trip_original_region}' (จาก candidates {_cand_region_counts})")
                    _region_filtered = same_zone_df[_crn_ser2 == trip_original_region]
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
                    same_zone_df['_province'].map(lambda p: _prov_region_map.get(str(p), get_region_name(str(p))) if p else '') == trip_original_region
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
            
            # 🎯 คำนวณระยะทาง — ใช้ incremental caches (เร็วกว่า rebuild ทุก iteration)
            unassigned_upper = {str(c).strip().upper() for c in unassigned}
            candidate_distances = {k: v for k, v in _incr_cand_dists.items() if k in unassigned_upper}

            # คำนวณ _dist_to_trip — dict map จาก candidate_distances + fallback จาก _incr_min_dist_all
            _cu_ser  = same_zone_df['Code'].str.strip().str.upper()
            _pre_dist = _cu_ser.map(candidate_distances)  # NaN ถ้าไม่มี
            # fallback: ดึงจาก _incr_min_dist_all (indexed by df.iloc)
            _szdf_cu_list = _cu_ser.tolist()
            _fallback_arr = np.array([
                _incr_min_dist_all[_code_upper_to_iloc[cu]] if cu in _code_upper_to_iloc else 999.0
                for cu in _szdf_cu_list
            ])
            _fallback_ser = pd.Series(_fallback_arr, index=same_zone_df.index)
            same_zone_df['_dist_to_trip'] = _pre_dist.combine_first(_fallback_ser).astype(float)

            # 🔗 คำนวณ _dist_to_last: ระยะจากสาขาสุดท้ายที่เพิ่ม (nearest-neighbor chain)
            if _last_added_lat > 0 and _last_added_lon > 0:
                _szdf_la_nn = same_zone_df['_lat'].fillna(0).to_numpy(dtype=float)
                _szdf_lo_nn = same_zone_df['_lon'].fillna(0).to_numpy(dtype=float)
                _dl_nn = np.radians(_szdf_la_nn - _last_added_lat)
                _do_nn = np.radians(_szdf_lo_nn - _last_added_lon)
                _a_nn = (np.sin(_dl_nn / 2) ** 2 +
                         np.cos(np.radians(_last_added_lat)) * np.cos(np.radians(_szdf_la_nn)) *
                         np.sin(_do_nn / 2) ** 2)
                _d_last_nn = 6371.0 * 2 * np.arctan2(
                    np.sqrt(np.clip(_a_nn, 0, 1)), np.sqrt(np.clip(1 - _a_nn, 0, 1))
                ) * 1.35
                _d_last_nn[_szdf_la_nn <= 0] = 999.0
                same_zone_df['_dist_to_last'] = _d_last_nn
            else:
                same_zone_df['_dist_to_last'] = same_zone_df['_dist_to_trip']

            # 🧠 AI AFFINITY RANK (ยังคงใช้ช่วย tie-break)
            _trip_provs_sort = {p for p in [trip_province, trip_original_province] if p}
            if _ai_active and trip_codes:
                _cand_codes_aff = same_zone_df['Code'].tolist()
                _aff_vals = []
                for _cc_aff in _cand_codes_aff:
                    _tot_aff = 0
                    for _tc_aff in trip_codes:
                        _pa, _pb = (_tc_aff, _cc_aff) if _tc_aff < _cc_aff else (_cc_aff, _tc_aff)
                        _tot_aff += _ai_pair_freq.get(f"{_pa}|{_pb}", 0)
                    _aff_vals.append(_tot_aff)
                same_zone_df['_affinity'] = _aff_vals
                same_zone_df['_affinity_rank'] = (same_zone_df['_affinity'] == 0).astype(int)
                sort_cols = ['_dist_to_last', '_priority', '_affinity_rank', '_dist_to_trip']
            else:
                sort_cols = ['_dist_to_last', '_priority', '_dist_to_trip']

            same_zone_df = same_zone_df.sort_values(sort_cols)
            _same_prov_close = (
                not same_zone_df[
                    same_zone_df['_province'].isin(_trip_provs_sort) &
                    (same_zone_df['_dist_to_last'] <= 25)
                ].empty
            ) if _trip_provs_sort else False

            # 🚀 SPEED: Pre-filter same_zone_df vectorized ก่อน iterrows (ลด rows ที่ต้องวน)
            # Filter 1: distance guard (priority != 1 → ต้องอยู่ใน _CHAIN_KM)
            same_zone_df = same_zone_df[
                (same_zone_df['_priority'] == 1) |
                (same_zone_df['_dist_to_trip'] <= _CHAIN_KM)
            ]
            # Filter 2: DC distance limit
            if trip_max_dist_dc > 0:
                same_zone_df = same_zone_df[
                    same_zone_df['_distance_from_dc'] <= trip_max_dist_dc + 30
                ]
            # Filter 3: Region compat (vectorized ด้วย _prov_region_map dict map — ไม่ใช้ lambda)
            _orig_region_calc = trip_original_region
            if _orig_region_calc and _orig_region_calc not in ('', 'ไม่ระบุ'):
                _szdf_reg = same_zone_df['_province'].map(_prov_region_map).fillna('')
                same_zone_df = same_zone_df[
                    _szdf_reg.isin(('', 'ไม่ระบุ', _orig_region_calc))
                ]
            # Filter 4: BKK isolation
            _BKK_PRE = 'กรุงเทพมหานคร'
            if trip_original_province == _BKK_PRE:
                same_zone_df = same_zone_df[same_zone_df['_province'] == _BKK_PRE]
            elif trip_original_province and trip_original_province not in ('', None):
                same_zone_df = same_zone_df[
                    (same_zone_df['_province'] != _BKK_PRE) |
                    (same_zone_df['_province'].isna())
                ]

            found_candidate = False
            # 🚀 SPEED: to_dict('records') หนึ่งครั้งก่อนลูป (~5x เร็วกว่า iterrows)
            for candidate_row in same_zone_df.to_dict('records'):
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

                # 🚫 Distance guard (ตำบลเดียวกัน priority=1: ไม่มี limit)
                _cand_prio_zone = int(candidate_row.get('_priority', 4))
                if _cand_prio_zone != 1:  # ไม่ใช่ตำบลเดียวกัน → ตรวจ distance
                    if candidate_dist > _CHAIN_KM:
                        continue

                # 🚫 Zone/province/region axis check
                _c_prov   = candidate_row.get('_province', '')
                _c_zone   = candidate_row.get('_logistics_zone', '')
                _c_hw     = candidate_row.get('_zone_highway', '')
                _c_hws    = set(str(_c_hw).split('/')) if _c_hw else set()
                # 🚀 SPEED: ใช้ _prov_region_map แทน get_region_name() ทุก row
                _c_region_calc = _prov_region_map.get(str(_c_prov), get_region_name(str(_c_prov)) if _c_prov else '') if _c_prov else ''
                _orig_region_calc = trip_original_region
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
                    # 🌟 Proximity override: ถ้าสาขาอยู่ใกล้มาก (<8km) + ภาคเดียวกัน + จังหวัดเดียวกัน
                    # (ลดจาก 10km เป็น 8km และเพิ่มเงื่อนไขจังหวัดเดียวกัน เพื่อป้องกันข้ามโซน)
                    _same_prov_override = (_c_prov and trip_original_province and _c_prov == trip_original_province)
                    if candidate_dist < 8.0 and _region_compat and _same_prov_override:
                        pass  # อนุญาต — proximity (same province) overrides zone restriction
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
                group_codes = get_group_branches_rt(candidate_code)
                # กรองเฉพาะสาขาที่ยังไม่ได้จัด — ใช้ upper set แทน linear scan
                _unassigned_upper_grp = {str(u).strip().upper() for u in unassigned}
                group_codes_unassigned = [c for c in group_codes if c.upper() in _unassigned_upper_grp]
                if not group_codes_unassigned:
                    group_codes_unassigned = [candidate_code]

                # คำนวณ weight/cube รวมทั้งกลุ่ม — ใช้ _code_row_map แทน df.apply
                _cd_lat = float(candidate_row.get('_lat', 0) or 0)
                _cd_lon = float(candidate_row.get('_lon', 0) or 0)
                _cd_coord300 = _code_coord300_partners.get(str(candidate_code).strip().upper(), frozenset())
                _cd_name = str(candidate_row.get('Name', '') or '').strip()
                group_weight = 0
                group_cube = 0
                group_codes_valid = []
                for gc in group_codes_unassigned:
                    _gc_upper = str(gc).strip().upper()
                    _gc_r = _code_row_map.get(_gc_upper)
                    if _gc_r is None:
                        continue
                    _cg_lat = float(_gc_r.get('_lat', 0) or 0)
                    _cg_lon = float(_gc_r.get('_lon', 0) or 0)
                    _cg_phys_dist = haversine_distance(_cg_lat, _cg_lon, _cd_lat, _cd_lon, use_osrm_cache=False) if (_cg_lat > 0 and _cg_lon > 0 and _cd_lat > 0 and _cd_lon > 0) else 999
                    # ตรวจเงื่อนไขกลุ่ม: ≤100m, coord-300m, ชื่อเดียวกัน
                    _is_same_sfx = _gc_upper in _cd_coord300
                    _is_same_name = bool(
                        str(_gc_r.get('Name', '') or '').strip() == _cd_name and _cd_name
                    )
                    _is_same_point = _cg_phys_dist <= 0.1  # ≤100m
                    _is_forced_group = _is_same_point or _is_same_sfx or _is_same_name
                    if not _is_forced_group and _cg_phys_dist > 0.3:
                        if trip_original_region and trip_original_region not in ('', 'ไม่ระบุ'):
                            _cg_prov = str(_gc_r.get('_province', '') or '')
                            _cg_region = _prov_region_map.get(_cg_prov, get_region_name(_cg_prov) if _cg_prov else '') if _cg_prov else ''
                            if _cg_region and _cg_region not in ('', 'ไม่ระบุ') and _cg_region != trip_original_region:
                                safe_print(f"      🛑 CAND-GROUP GUARD: ตัด {gc} ภาค {_cg_region} ≠ {trip_original_region} (ห่าง {_cg_phys_dist:.1f}km)")
                                continue
                    group_weight += _safe_float(_gc_r.get('Weight', 0), 0)
                    group_cube += _safe_float(_gc_r.get('Cube', 0), 0)
                    group_codes_valid.append(_code_real_map.get(_gc_upper, gc))

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
                    # ข้อจำกัดรถขัดแย้ง → ข้ามไปสาขาถัดไป (ไม่ปิดทริป)
                    safe_print(f"      ⏭️ skip {candidate_code}: vehicle constraint ขัดแย้ง → ลองถัดไป")
                    continue
                
                # เช็คน้ำหนัก/ปริมาตร/drops รวมทั้งกลุ่ม
                test_weight = trip_weight + group_weight
                test_cube = trip_cube + group_cube
                test_drops = len(test_codes)
                # คำนวณ qty กลุ่มนี้ — ใช้ _code_row_map แทน df.apply
                group_qty = sum(int(_safe_float(_code_row_map.get(str(gc).strip().upper(), {}).get('OriginalQty', 0), 0)) for gc in group_codes_valid)
                test_qty = trip_qty + group_qty
                if max_qty_per_trip > 0 and test_qty > max_qty_per_trip:
                    safe_print(f"      🔢 QTY limit: {test_qty} > {max_qty_per_trip} → ปิดทริป {trip_counter}")
                    break

                # หา buffer ที่ใช้
                test_is_punthai = all(branch_bu_cache.get(c, False) for c in test_codes)
                buffer = punthai_buffer if test_is_punthai else maxmart_buffer
                limits = PUNTHAI_LIMITS if test_is_punthai else LIMITS

                # Lookahead: รวม suffix partner loads เข้า capacity check — ใช้ _code_row_map
                _sfp_lk_w = 0.0
                _sfp_lk_c = 0.0
                _trip_upper_set = {str(t).strip().upper() for t in trip_codes}
                _grp_upper_set  = {str(g).strip().upper() for g in group_codes_valid}
                _unassigned_up2 = {str(u).strip().upper() for u in unassigned}
                for _lk_gc in group_codes_valid:
                    for _lk_sfp in _suffix_force_partners.get(str(_lk_gc).strip().upper(), []):
                        if _lk_sfp in _trip_upper_set or _lk_sfp in _grp_upper_set:
                            continue
                        if _lk_sfp in _unassigned_up2:
                            _lk_r = _code_row_map.get(_lk_sfp)
                            if _lk_r is not None:
                                _sfp_lk_w += _safe_float(_lk_r.get('Weight', 0), 0)
                                _sfp_lk_c += _safe_float(_lk_r.get('Cube', 0), 0)
                _chk_w = test_weight + _sfp_lk_w
                _chk_c = test_cube + _sfp_lk_c

                # 🎯 เลือกรถเล็กสุดที่รับโหลดได้ — น้ำหนัก/คิ้วห้ามเกิน buffer + ข้อจำกัดสาขาห้ามเกิน
                selected_vehicle = None
                for veh in ['4W', 'JB', '6W']:
                    veh_rank = vehicle_rank.get(veh, 3)
                    if veh_rank > group_min_max_rank:
                        break
                    if veh not in test_allowed:
                        continue
                    lim = limits[veh]
                    if (_chk_w <= lim['max_w'] * buffer and
                            _chk_c <= lim['max_c'] * buffer and
                            test_drops <= lim['max_drops']):
                        selected_vehicle = veh
                        break

                if not selected_vehicle:
                    if len(group_codes_valid) > 1:
                        safe_print(f"      🛑 กลุ่ม {len(group_codes_valid)} สาขา โหลดเกินทุกรถที่อนุญาต → ปิดทริป {trip_counter}")
                    else:
                        safe_print(f"      🛑 สาขา {candidate_code} โหลดเกินทุกรถที่อนุญาต → ปิดทริป {trip_counter}")
                    break

                # ✅ เพิ่มสาขาทั้งกลุ่มเข้าทริป
                max_w = limits[selected_vehicle]['max_w'] * buffer
                max_c = limits[selected_vehicle]['max_c'] * buffer
                _unassigned_up3 = {str(u).strip().upper() for u in unassigned}
                for gc in group_codes_valid:
                    if gc not in trip_codes:
                        trip_codes.append(gc)
                    _gc_up = str(gc).strip().upper()
                    if gc in unassigned:
                        unassigned.remove(gc)
                    elif _gc_up in _unassigned_up3:
                        for u in list(unassigned):
                            if str(u).strip().upper() == _gc_up:
                                unassigned.remove(u)
                                break

                trip_weight = test_weight
                trip_cube = test_cube
                trip_qty = test_qty
                trip_allowed = test_allowed
                trip_is_punthai = test_is_punthai
                found_candidate = True
                # 🔗 อัปเดตตำแหน่งสาขาล่าสุด (nearest-neighbor chain)
                if _cd_lat > 0 and _cd_lon > 0:
                    _last_added_lat = _cd_lat
                    _last_added_lon = _cd_lon

                # 🔗 SUFFIX-FORCE (greedy): บังคับ coord-300m partners เข้าทริปเดียวกัน
                # ตรวจ capacity ก่อนเพิ่มแต่ละ partner — ห้ามเกินขีดจำกัดเด็ดขาด
                _trip_upper_sf = {str(t).strip().upper() for t in trip_codes}
                _unassigned_up_sf = {str(u).strip().upper() for u in unassigned}
                for _sfc in list(group_codes_valid):
                    for _sfp in _suffix_force_partners.get(str(_sfc).strip().upper(), []):
                        if _sfp in _trip_upper_sf:
                            continue
                        if _sfp in _unassigned_up_sf:
                            _sfp_r = _code_row_map.get(_sfp)
                            if _sfp_r is not None:
                                _sfp_actual = _code_real_map.get(_sfp, _sfp)
                                _sfp_w = _safe_float(_sfp_r.get('Weight', 0), 0)
                                _sfp_c = _safe_float(_sfp_r.get('Cube', 0), 0)
                                _sfp_is_pt = branch_bu_cache.get(_sfp, False)
                                _trip_min_priority = min(
                                    vehicle_priority.get(branch_max_vehicle_cache.get(str(c).strip().upper(), '6W'), 3)
                                    for c in trip_codes
                                ) if trip_codes else 3
                                _trip_capacity = {
                                    'weight': trip_weight,
                                    'cube': trip_cube,
                                    'drops': len(trip_codes),
                                    'min_priority': _trip_min_priority,
                                    'allowed_vehicle': trip_allowed[-1] if trip_allowed else '6W',
                                    'is_punthai': trip_is_punthai,
                                }
                                can_add_sf, reason_sf = can_add_branch_to_trip(_sfp_r, _trip_capacity)
                                if not can_add_sf:
                                    safe_print(f"      ⚠️ SUFFIX-FORCE skip {_sfp_actual}: {reason_sf}")
                                    continue
                                trip_codes.append(_sfp_actual)
                                _trip_upper_sf.add(_sfp)
                                trip_weight += _sfp_w
                                trip_cube += _sfp_c
                                trip_qty += int(_safe_float(_sfp_r.get('OriginalQty', 0), 0))
                                trip_is_punthai = trip_is_punthai and _sfp_is_pt
                                trip_allowed = get_allowed_from_codes(trip_codes, ['4W', 'JB', '6W'])
                                safe_print(f"      🔗 SUFFIX-FORCE (greedy): {_sfp_actual} → บังคับเข้าทริปเดียวกับ {_sfc}")
                                if _sfp_actual in unassigned:
                                    unassigned.remove(_sfp_actual)
                                else:
                                    for _u in list(unassigned):
                                        if str(_u).strip().upper() == _sfp:
                                            unassigned.remove(_u)
                                            break

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
                
                # เช็คว่าเต็มหรือยัง (>= 100% เท่านั้น ไม่ตัดก่อนเต็ม)
                w_util = trip_weight / max_w
                c_util = trip_cube / max_c
                _qty_full = (max_qty_per_trip > 0 and trip_qty >= max_qty_per_trip)
                if max(w_util, c_util) >= 1.0 or _qty_full:
                    safe_print(f"      ✅ Trip {trip_counter} เต็ม {max(w_util, c_util)*100:.1f}% qty={trip_qty}/{max_qty_per_trip} ({len(trip_codes)} สาขา)")
                    break  # เต็มแล้ว
                
                break  # หาสาขาเพิ่มได้กลุ่ม/สาขา → วนลูปใหม่หา centroid ใหม่
            
            if not found_candidate:
                # epidemic: ทุก candidates ใน frontier ไม่ผ่าน filter
                # ✅ ถ้าทริปยัง util < 70% → force-fill จากจังหวัดเดียวกันก่อนปิด
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
                # ป้องกัน Series ambiguity: แปลง trip_original_province/region เป็น str ก่อนใช้
                _ff_top_prov = str(trip_original_province) if trip_original_province is not None else ''
                _ff_top_reg  = str(trip_original_region)  if trip_original_region  is not None else ''
                if _ff_util < 0.90 and _ff_top_prov:
                    _ff_rem = df[df['Code'].isin(unassigned)].copy()
                    _ff_same = _ff_rem[
                        _ff_rem['_province'].apply(lambda p: str(p or '').strip() == _ff_top_prov.strip())
                    ].copy()
                    if not _ff_same.empty:
                        _trip_valid_coords_ff = [(float(r.get('_lat', 0) or 0), float(r.get('_lon', 0) or 0))
                                                 for r in [_code_row_map.get(str(tc).strip().upper()) for tc in trip_codes]
                                                 if r and float(r.get('_lat', 0) or 0) > 0]
                        # 🚀 vectorized haversine (ป้องกัน Series ambiguity ใน apply lambda)
                        _fflats_np = _ff_same['_lat'].fillna(0).to_numpy(dtype=float)
                        _fflons_np = _ff_same['_lon'].fillna(0).to_numpy(dtype=float)
                        _ff_d_arr = np.full(len(_ff_same), 999.0)
                        if _trip_valid_coords_ff:
                            _ffv_mask = _fflats_np > 0
                            if _ffv_mask.any():
                                _ffc2 = np.array(_trip_valid_coords_ff, dtype=float)
                                _ffla = np.radians(_fflats_np[_ffv_mask])[:, np.newaxis]
                                _fflo = np.radians(_fflons_np[_ffv_mask])[:, np.newaxis]
                                _ffa = (np.sin((_ffla - np.radians(_ffc2[:, 0])) / 2) ** 2 +
                                        np.cos(np.radians(_ffc2[:, 0])) * np.cos(_ffla) *
                                        np.sin((_fflo - np.radians(_ffc2[:, 1])) / 2) ** 2)
                                _ff_d_arr[_ffv_mask] = (6371.0 * 2 * np.arctan2(np.sqrt(_ffa), np.sqrt(1 - _ffa)) * 1.35).min(axis=1)
                        _ff_same = _ff_same.reset_index(drop=True).copy()
                        _ff_same['_ff_dist'] = _ff_d_arr
                        _ff_same = _ff_same.sort_values('_ff_dist')
                        # แปลงเป็น plain dict ก่อน loop → ป้องกัน Series ambiguity ทุกกรณี
                        _ff_records = _ff_same.to_dict('records')
                        for _ff_rec in _ff_records:
                            if _ff_util >= 0.90:
                                break
                            _ff_code = str(_ff_rec.get('Code', '') or '')
                            if not _ff_code:
                                continue
                            _ff_prov = str(_ff_rec.get('_province', '') or '')
                            _BKK = 'กรุงเทพมหานคร'
                            if ((_ff_prov == _BKK and _ff_top_prov != _BKK) or
                                    (_ff_top_prov == _BKK and _ff_prov and _ff_prov != _BKK)):
                                continue
                            _ff_region = get_region_name(_ff_prov) if _ff_prov else ''
                            if (_ff_top_reg and _ff_top_reg not in ('', 'ไม่ระบุ') and
                                    _ff_region and _ff_region not in ('', 'ไม่ระบุ') and
                                    _ff_region != _ff_top_reg):
                                continue
                            _ff_test_codes = trip_codes + [_ff_code]
                            _ff_test_allowed = get_allowed_from_codes(_ff_test_codes, ['4W', 'JB', '6W'])
                            if not _ff_test_allowed:
                                continue
                            _ff_test_w = trip_weight + _safe_float(_ff_rec.get('Weight'), 0)
                            _ff_test_c = trip_cube + _safe_float(_ff_rec.get('Cube'), 0)
                            _ff_veh_ok = None
                            for _ffv in ['4W', 'JB', '6W']:
                                if _ffv not in _ff_test_allowed:
                                    continue
                                _ffvl2 = _ff_lims.get(_ffv, _ff_lims['6W'])
                                if (_ff_test_w <= _ffvl2['max_w'] * _ff_buf and
                                        _ff_test_c <= _ffvl2['max_c'] * _ff_buf):
                                    _ff_veh_ok = _ffv
                                    break
                            if not _ff_veh_ok:
                                continue
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
                            _ff_util = max(_ff_test_w / _ff_max_w if _ff_max_w > 0 else 1,
                                           _ff_test_c / _ff_max_c if _ff_max_c > 0 else 1)
                            safe_print(f"      🔋 Force-fill #{trip_counter}: +{_ff_code} ({_ff_prov}) util={_ff_util*100:.0f}%")
                            _force_filled = True
                if not _force_filled:
                    safe_print(f"      🛑 epidemic candidates ถูกกรองหมด #{trip_counter} ({len(trip_codes)} สาขา) → ปิดทริป")
                break
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
                if _ff_util < 1.0 and trip_original_province:
                    _ff_rem = df[df['Code'].isin(unassigned)].copy()
                    _ff_same = _ff_rem[
                        _ff_rem['_province'].apply(lambda p: str(p or '').strip() == str(trip_original_province).strip())
                    ].copy()
                    if not _ff_same.empty:
                        if _trip_valid_coords:
                            _ff_same['_ff_dist'] = _ff_same.apply(
                                lambda row: min(
                                    haversine_distance(float(row['_lat'] or 0), float(row['_lon'] or 0), tlat, tlon, use_osrm_cache=False)
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
                            _ff_test_w = trip_weight + _safe_float(_ff_row.get('Weight', 0), 0)
                            _ff_test_c = trip_cube + _safe_float(_ff_row.get('Cube', 0), 0)
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
                    pass  # force-fill ไม่ได้ผล → ปิดทริปด้านบนแล้ว

        # 3️⃣ Assign ทริป
        for code in trip_codes:
            df.loc[df['Code'] == code, 'Trip'] = trip_counter
        
        # 🦠 อัปเดต epidemic frontier สำหรับทริปถัดไป
        _last_trip_all_coords = []
        _last_trip_subdistricts = set()
        _last_trip_districts = set()
        for _ltc in trip_codes:
            _ltr2 = _code_row_map.get(str(_ltc).strip().upper())
            if _ltr2 is not None:
                _ltlat = float(_ltr2.get('_lat', 0) or 0)
                _ltlon = float(_ltr2.get('_lon', 0) or 0)
                if _ltlat > 0 and _ltlon > 0:
                    _last_trip_all_coords.append((_ltlat, _ltlon))
                _lt_sub = str(_ltr2.get('_subdistrict', '') or '')
                _lt_dis = str(_ltr2.get('_district', '') or '')
                if _lt_sub: _last_trip_subdistricts.add(_lt_sub)
                if _lt_dis: _last_trip_districts.add(_lt_dis)
        _last_trip_region = trip_original_region
        _last_trip_zone   = trip_logistics_zone   # อัปเดต zone ล่าสุด
        # 🌊 อัปเดต wave front: ตำแหน่งสาขาสุดท้ายของทริปนี้ → seed ทริปถัดไป (ไม่กระโดดไกล)
        if _last_added_lat > 0 and _last_added_lon > 0:
            _wave_lat = _last_added_lat
            _wave_lon = _last_added_lon

        # 🔒 District isolation: ถ้าทริปนี้มีแค่ 1 อำเภอ และยังมีสาขาอำเภอเดิมเหลือ
        # → ล็อคทริปถัดไปให้จัดเฉพาะอำเภอนั้น (ป้องกันอำเภอเดียวกันแตกใส่ทริปอื่น)
        _lock_next_district = ''
        if len(trip_districts) == 1:
            _sole_d = list(trip_districts)[0]
            if _sole_d and df[(df['Code'].isin(unassigned)) & (df['_district'] == _sole_d)].shape[0] > 0:
                _lock_next_district = _sole_d
                safe_print(f"   🔒 District overflow lock: '{_sole_d}' → trip ถัดไปจัดอำเภอนี้ก่อน")
        trip_counter += 1
    
    safe_print(f"🎯 จัดทริปเสร็จ: {trip_counter - 1} ทริป")

    # ==========================================
    # Step 6.4.4: 🔋 FILL-UP PASS — เติมรถที่ยังไม่เต็มด้วยสาขาที่เหลือ
    # สาขาที่ยังไม่ได้จัด (unassigned) → ลองเพิ่มเข้าทริปที่ util < 70%
    # เฉพาะสาขาที่อยู่ในภาค/จังหวัดเดียวกัน และใกล้ทริปนั้น ≤ 60km
    # ==========================================
    safe_print("🔋 Fill-up pass: ตรวจสอบทริปที่ยังไม่เต็ม...")
    _FILLUP_MIN_UTIL = 0.90   # ทริปที่ util < 90% → ลองเติม (เติมจนเขียวก่อนปล่อย)
    _FILLUP_MAX_KM   = 60.0  # รัศมีเพิ่มสาขา (km)
    _fillup_added = 0
    for _ft in df[df['Trip'] > 0]['Trip'].unique():
        _ft_rows = df[df['Trip'] == _ft]
        _ft_codes = _ft_rows['Code'].tolist()
        _ft_is_pt = all(branch_bu_cache.get(str(c).strip().upper(), False) for c in _ft_codes)
        _ft_buf = punthai_buffer if _ft_is_pt else maxmart_buffer
        _ft_lims = PUNTHAI_LIMITS if _ft_is_pt else LIMITS
        _ft_allowed = get_allowed_from_codes(_ft_codes, ['4W', 'JB', '6W'])
        _ft_w = float(_ft_rows['Weight'].sum())
        _ft_c = float(_ft_rows['Cube'].sum())
        # หารถที่เล็กสุดที่รับโหลดได้จริง (ไม่ใช่แค่ smallest allowed)
        _ft_veh = next(reversed(['4W', 'JB', '6W']), '6W')  # default largest
        for _ftv in ['4W', 'JB', '6W']:
            if _ftv not in (_ft_allowed or ['6W']):
                continue
            _ftv_lim = _ft_lims[_ftv]
            if _ft_w <= _ftv_lim['max_w'] and _ft_c <= _ftv_lim['max_c']:
                _ft_veh = _ftv
                break
        _ft_lim = _ft_lims.get(_ft_veh, _ft_lims['6W'])
        _ft_max_w = _ft_lim['max_w'] * _ft_buf
        _ft_max_c = _ft_lim['max_c'] * _ft_buf
        _ft_util = max(_ft_w / _ft_max_w if _ft_max_w > 0 else 1.0, _ft_c / _ft_max_c if _ft_max_c > 0 else 1.0)
        if _ft_util >= _FILLUP_MIN_UTIL:
            continue  # เต็มพอแล้ว → ข้าม
        # รวบรวมพิกัดของทริปนี้
        _ft_coords = [(float(r.get('_lat') or 0), float(r.get('_lon') or 0)) for r in _ft_rows.to_dict('records')
                      if float(r.get('_lat') or 0) > 0]
        # ใช้ _code_row_map แทน iloc[0].get() → ป้องกัน Series ambiguity จาก duplicate columns
        _ft_first_code = str(_ft_rows.iloc[0]['Code']).strip().upper() if len(_ft_rows) > 0 else ''
        _ft_first_meta = _code_row_map.get(_ft_first_code, {})
        _ft_region = str(_ft_first_meta.get('_region_name', '') or '')
        if not _ft_region or _ft_region == 'ไม่ระบุ':
            _ft_prov_tmp = str(_ft_first_meta.get('_province', '') or '')
            _ft_region = _prov_region_map.get(_ft_prov_tmp, get_region_name(_ft_prov_tmp) if _ft_prov_tmp else '')
        _ft_prov = str(_ft_first_meta.get('_province', '') or '')
        # 🔒 ถ้ายังไม่รู้ภาคหลัง fallback → ข้ามทริปนี้ (ไม่เพิ่มสาขาต่างภาคโดยไม่รู้จัก)
        if not _ft_region or _ft_region == 'ไม่ระบุ':
            continue
        # 🚀 SPEED: vectorized haversine + _prov_region_map แทน iterrows
        _fu_unassigned = df[df['Trip'] == 0].copy()
        if _fu_unassigned.empty:
            break
        # กรอง: ภาค/จังหวัดเดียวกัน + ระยะ ≤ 60km
        _fu_cands = []
        if _ft_coords:
            _fu_lats_np = _fu_unassigned['_lat'].fillna(0).to_numpy(dtype=float)
            _fu_lons_np = _fu_unassigned['_lon'].fillna(0).to_numpy(dtype=float)
            _fu_valid_m = _fu_lats_np > 0
            _fu_min_d_arr = np.full(len(_fu_unassigned), 999.0)
            if _fu_valid_m.any():
                _fuc_arr = np.array(_ft_coords, dtype=float)  # (n_coords, 2)
                _fuvl = np.radians(_fu_lats_np[_fu_valid_m])[:, np.newaxis]
                _fuvo = np.radians(_fu_lons_np[_fu_valid_m])[:, np.newaxis]
                _fuc_lr = np.radians(_fuc_arr[:, 0])
                _fuc_or = np.radians(_fuc_arr[:, 1])
                _a_fu = (np.sin((_fuvl - _fuc_lr) / 2) ** 2 +
                         np.cos(_fuc_lr) * np.cos(_fuvl) *
                         np.sin((_fuvo - _fuc_or) / 2) ** 2)
                _fu_min_d_arr[_fu_valid_m] = (
                    6371.0 * 2 * np.arctan2(np.sqrt(_a_fu), np.sqrt(1 - _a_fu)) * 1.35
                ).min(axis=1)
            _fu_in_range = np.where((_fu_valid_m) & (_fu_min_d_arr <= _FILLUP_MAX_KM))[0]
            _fu_df_ri = _fu_unassigned.reset_index(drop=True)
            _fu_records = _fu_df_ri.to_dict('records')  # plain dicts → ป้องกัน Series ambiguity
            for _fu_idx in _fu_in_range:
                _fur = _fu_records[_fu_idx]
                _fu_prov = str(_fur.get('_province', '') or '')
                _fu_reg = _prov_region_map.get(_fu_prov, get_region_name(_fu_prov) if _fu_prov else '')
                if (_ft_region and _ft_region not in ('', 'ไม่ระบุ') and
                        _fu_reg and _fu_reg not in ('', 'ไม่ระบุ') and
                        _fu_reg != _ft_region):
                    continue
                _fu_cands.append((_fu_min_d_arr[_fu_idx], _fur.get('Code', ''),
                                  _safe_float(_fur.get('Weight', 0), 0),
                                  _safe_float(_fur.get('Cube', 0), 0)))
        _fu_cands.sort(key=lambda x: x[0])  # ใกล้สุดก่อน
        for _fu_d, _fu_code, _fu_w, _fu_c in _fu_cands:
            if _ft_util >= 1.0:
                break
            # ตรวจ vehicle constraint
            _fu_test_codes = _ft_codes + [_fu_code]
            _fu_test_allowed = get_allowed_from_codes(_fu_test_codes, ['4W', 'JB', '6W'])
            if not _fu_test_allowed:
                continue
            _fu_test_w = _ft_w + _fu_w
            _fu_test_c = _ft_c + _fu_c
            _fu_veh_ok = None
            for _fuv in ['4W', 'JB', '6W']:
                if _fuv not in _fu_test_allowed: continue
                _fuvl = _ft_lims.get(_fuv, _ft_lims['6W'])
                if (_fu_test_w <= _fuvl['max_w'] and
                        _fu_test_c <= _fuvl['max_c']):
                    _fu_veh_ok = _fuv
                    break
            if not _fu_veh_ok:
                continue
            # ✅ เพิ่มเข้าทริป
            df.loc[df['Code'] == _fu_code, 'Trip'] = _ft
            _ft_codes.append(_fu_code)
            _ft_w = _fu_test_w
            _ft_c = _fu_test_c
            _ft_util = max(_ft_w / _ft_max_w if _ft_max_w > 0 else 1.0,
                           _ft_c / _ft_max_c if _ft_max_c > 0 else 1.0)
            _fillup_added += 1
            safe_print(f"   🔋 Fill-up trip {_ft}: +{_fu_code} ({_fu_d:.1f}km) util={_ft_util*100:.0f}%")
    safe_print(f"🔋 Fill-up pass: เพิ่ม {_fillup_added} สาขาเข้าทริปที่ไม่เต็ม")

    # ==========================================
    # Step 6.4.4c: 🔗 SMALL-TRIP MERGE PASS
    # ทริปที่ util < 50% → หาทริปใกล้เคียงในภาคเดียวกัน + จำนวนจุดรวมไม่เกิน limit
    # → รวมเป็นทริปเดียว (ประหยัดรถ)
    # ==========================================
    safe_print("🔗 Small-trip merge pass: ตรวจทริปที่ใช้งานน้อย...")
    _MERGE_MAX_UTIL  = 0.85   # ทริปที่ util < 85% → ลองรวมกับทริปใกล้เคียง
    _MERGE_CENTROID_KM = 200.0  # รัศมีค้นหาทริปคู่ merge (km)
    _merge_count = 0

    def _trip_centroid(trip_num):
        _rows = df[df['Trip'] == trip_num]
        _lats = _rows['_lat'].fillna(0).astype(float)
        _lons = _rows['_lon'].fillna(0).astype(float)
        _valid = _lats > 0
        if not _valid.any():
            return 0.0, 0.0
        return float(_lats[_valid].mean()), float(_lons[_valid].mean())

    def _trip_util(trip_num):
        _rows = df[df['Trip'] == trip_num]
        _codes = _rows['Code'].tolist()
        _is_pt = all(branch_bu_cache.get(str(c).strip().upper(), False) for c in _codes)
        _buf  = punthai_buffer if _is_pt else maxmart_buffer
        _lims = PUNTHAI_LIMITS if _is_pt else LIMITS
        _allowed = get_allowed_from_codes(_codes, ['4W', 'JB', '6W'])
        _w = float(_rows['Weight'].sum())
        _c = float(_rows['Cube'].sum())
        _veh = '6W'
        for v in ['4W', 'JB', '6W']:
            if v not in (_allowed or ['6W']): continue
            vl = _lims[v]
            if _w <= vl['max_w'] * _buf and _c <= vl['max_c'] * _buf:
                _veh = v; break
        vl = _lims.get(_veh, _lims['6W'])
        mw = vl['max_w'] * _buf; mc = vl['max_c'] * _buf
        return max(_w / mw if mw > 0 else 1, _c / mc if mc > 0 else 1)

    _merge_changed = True
    while _merge_changed:
        _merge_changed = False
        _all_trips = sorted(df[df['Trip'] > 0]['Trip'].unique())
        # สร้าง centroid + util + region สำหรับทุกทริป
        _trip_meta = {}
        for _tn in _all_trips:
            _r = df[df['Trip'] == _tn]
            _clat, _clon = _trip_centroid(_tn)
            _util = _trip_util(_tn)
            # ใช้ _code_row_map แทน iloc[0].get() → ป้องกัน Series ambiguity จาก duplicate columns
            _first_code = str(_r.iloc[0]['Code']).strip().upper() if len(_r) > 0 else ''
            _first_meta = _code_row_map.get(_first_code, {})
            _reg = str(_first_meta.get('_region_name', '') or '')
            if not _reg or _reg == 'ไม่ระบุ':
                _pv = str(_first_meta.get('_province', '') or '')
                _reg = _prov_region_map.get(_pv, get_region_name(_pv) if _pv else '')
            _prov = str(_first_meta.get('_province', '') or '')
            _is_pt = all(branch_bu_cache.get(str(c).strip().upper(), False) for c in _r['Code'].tolist())
            _buf = punthai_buffer if _is_pt else maxmart_buffer
            _lims = PUNTHAI_LIMITS if _is_pt else LIMITS
            _allowed = get_allowed_from_codes(_r['Code'].tolist(), ['4W', 'JB', '6W'])
            _w = float(_r['Weight'].sum()); _c = float(_r['Cube'].sum())
            _trip_meta[_tn] = {'clat': _clat, 'clon': _clon, 'util': _util,
                               'region': _reg, 'prov': _prov, 'w': _w, 'c': _c,
                               'allowed': _allowed, 'buf': _buf, 'lims': _lims, 'is_pt': _is_pt}

        # หาทริปที่ util < 50% (เรียงจากน้อยสุด)
        _small_trips = sorted(
            [t for t in _all_trips if _trip_meta[t]['util'] < _MERGE_MAX_UTIL],
            key=lambda t: _trip_meta[t]['util']
        )
        for _ta in _small_trips:
            if _ta not in _trip_meta: continue
            _ma = _trip_meta[_ta]
            if _ma['clat'] == 0: continue
            # หาทริปคู่ที่ใกล้ที่สุดในภาคเดียวกัน
            _best_tb, _best_dist = None, 999.0
            for _tb in _all_trips:
                if _tb == _ta: continue
                if _tb not in _trip_meta: continue
                _mb = _trip_meta[_tb]
                if _mb['clat'] == 0: continue
                # ต้องเป็นภาคเดียวกัน
                if (_ma['region'] and _mb['region'] and
                        _ma['region'] not in ('', 'ไม่ระบุ') and
                        _mb['region'] not in ('', 'ไม่ระบุ') and
                        _ma['region'] != _mb['region']):
                    continue
                # ตรวจ BKK isolation — ใช้ prov จาก _trip_meta (plain str แล้ว)
                _ra_prov = _ma.get('prov', '')
                _rb_prov = _mb.get('prov', '')
                _BKK = 'กรุงเทพมหานคร'
                if ((_ra_prov == _BKK and _rb_prov and _rb_prov != _BKK) or
                        (_rb_prov == _BKK and _ra_prov and _ra_prov != _BKK)):
                    continue
                # ตรวจว่ารวมแล้วจะ fit หรือไม่
                _merged_codes = df[df['Trip'].isin([_ta, _tb])]['Code'].tolist()
                _merged_allowed = get_allowed_from_codes(_merged_codes, ['4W', 'JB', '6W'])
                if not _merged_allowed: continue
                _merged_w = _ma['w'] + _mb['w']
                _merged_c = _ma['c'] + _mb['c']
                _merged_is_pt = _ma['is_pt'] and _mb['is_pt']
                _merged_buf = punthai_buffer if _merged_is_pt else maxmart_buffer
                _merged_lims = PUNTHAI_LIMITS if _merged_is_pt else LIMITS
                _fits = False
                for _mv in ['4W', 'JB', '6W']:
                    if _mv not in _merged_allowed: continue
                    _mvl = _merged_lims[_mv]
                    if (_merged_w <= _mvl['max_w'] * _merged_buf and
                            _merged_c <= _mvl['max_c'] * _merged_buf):
                        _fits = True; break
                if not _fits: continue
                # คำนวณระยะระหว่าง centroid
                _d = haversine_distance(_ma['clat'], _ma['clon'], _mb['clat'], _mb['clon'], use_osrm_cache=False)
                if _d < _best_dist and _d <= _MERGE_CENTROID_KM:
                    _best_dist = _d; _best_tb = _tb

            if _best_tb is not None:
                # รวม: ย้ายสาขาจาก _ta → _best_tb (เก็บเลข trip ที่น้อยกว่า)
                _keep = min(_ta, _best_tb); _drop = max(_ta, _best_tb)
                df.loc[df['Trip'] == _drop, 'Trip'] = _keep
                safe_print(f"   🔗 Merge: Trip {_drop} → Trip {_keep} (dist={_best_dist:.1f}km, util_a={_ma['util']*100:.0f}%)")
                _merge_count += 1
                _merge_changed = True
                break  # rebuild metadata แล้ว loop ใหม่

    safe_print(f"🔗 Small-trip merge: รวม {_merge_count} ครั้ง")

    # ==========================================
    # Step 6.4.4b: 📍 SAME-COORDINATE FORCE MERGE
    # สาขาพิกัดเดียวกัน (≤50m) ในต่างทริป → รวมทริปเข้าด้วยกัน (ยอมเกิน capacity)
    # รวมถึงสาขาชื่อเดียวกันที่อยู่ห่างกัน ≤50m
    # ==========================================
    _SAME_COORD_KM = 0.05   # 50 เมตร
    safe_print("📍 SAME-COORDINATE FORCE MERGE: ตรวจสาขาพิกัดเดียวกันต่างทริป...")
    _samecoord_merged = 0
    _sc_changed = True
    while _sc_changed:
        _sc_changed = False
        # สร้าง {trip: [(code, lat, lon, name), ...]} — 🚀 vectorized แทน iterrows
        _trip_coord_map2: dict = {}
        _scd_df = df[df['Trip'] > 0][['Trip', 'Code', '_lat', '_lon', 'Name']].copy()
        _scd_lats = _scd_df['_lat'].fillna(0).astype(float)
        _scd_lons = _scd_df['_lon'].fillna(0).astype(float)
        _scd_valid = (_scd_lats > 0).to_numpy()
        for _ii in np.where(_scd_valid)[0]:
            _scd_row = _scd_df.iloc[_ii]
            _sc_t2 = int(_scd_row['Trip'])
            if _sc_t2 not in _trip_coord_map2:
                _trip_coord_map2[_sc_t2] = []
            _trip_coord_map2[_sc_t2].append((
                str(_scd_row['Code']),
                float(_scd_lats.iloc[_ii]),
                float(_scd_lons.iloc[_ii]),
                str(_scd_row.get('Name', '') or '').strip()
            ))
        _trip_list2 = sorted(_trip_coord_map2.keys())
        for _i2, _ta2 in enumerate(_trip_list2):
            if _sc_changed: break
            for _tb2 in _trip_list2[_i2+1:]:
                if _sc_changed: break
                for (_ca2, _lat_a2, _lon_a2, _nm_a2) in _trip_coord_map2.get(_ta2, []):
                    if _sc_changed: break
                    for (_cb2, _lat_b2, _lon_b2, _nm_b2) in _trip_coord_map2.get(_tb2, []):
                        _d_sc2 = haversine_distance(_lat_a2, _lon_a2, _lat_b2, _lon_b2, use_osrm_cache=False)
                        # พิกัดใกล้กัน ≤50m หรือ ชื่อสาขาเดียวกัน + ≤200m
                        _name_match = (_nm_a2 and _nm_b2 and _nm_a2 == _nm_b2 and _d_sc2 <= 0.2)
                        if _d_sc2 <= _SAME_COORD_KM or _name_match:
                            # 🔒 ตรวจ region ก่อนรวม — ห้ามข้ามภาค
                            _ta2_first_code = str(df[df['Trip'] == _ta2].iloc[0]['Code']).strip().upper()
                            _tb2_first_code = str(df[df['Trip'] == _tb2].iloc[0]['Code']).strip().upper()
                            _ta2_first = _code_row_map.get(_ta2_first_code, {})
                            _tb2_first = _code_row_map.get(_tb2_first_code, {})
                            _reg_a2 = _prov_region_map.get(str(_ta2_first.get('_province','') or ''), '')
                            _reg_b2 = _prov_region_map.get(str(_tb2_first.get('_province','') or ''), '')
                            if (_reg_a2 and _reg_b2 and
                                    _reg_a2 not in ('', 'ไม่ระบุ') and _reg_b2 not in ('', 'ไม่ระบุ') and
                                    _reg_a2 != _reg_b2):
                                safe_print(f"   🚫 same-coord skip region: {_ca2}({_reg_a2}) ≠ {_cb2}({_reg_b2})")
                                continue
                            # 🔒 ตรวจ capacity ก่อนรวม — ห้ามเกินเด็ดขาด
                            _ta2_data = df[df['Trip'] == _ta2]
                            _tb2_data = df[df['Trip'] == _tb2]
                            _merged_w = _ta2_data['Weight'].sum() + _tb2_data['Weight'].sum()
                            _merged_c = _ta2_data['Cube'].sum() + _tb2_data['Cube'].sum()
                            _all_codes_sc = _ta2_data['Code'].tolist() + _tb2_data['Code'].tolist()
                            _is_pt_sc = all(branch_bu_cache.get(str(c).strip().upper(), False) for c in _all_codes_sc)
                            _buf_sc = punthai_buffer if _is_pt_sc else maxmart_buffer
                            _lims_sc = PUNTHAI_LIMITS if _is_pt_sc else LIMITS
                            _allowed_sc = get_allowed_from_codes(_all_codes_sc, ['4W', 'JB', '6W'])
                            _fits_sc = any(
                                _merged_w <= _lims_sc[v]['max_w'] * _buf_sc and
                                _merged_c <= _lims_sc[v]['max_c'] * _buf_sc
                                for v in (['4W', 'JB', '6W'] if not _allowed_sc else _allowed_sc)
                            )
                            if not _fits_sc:
                                safe_print(f"   ⚠️ same-coord skip: {_ca2}↔{_cb2} รวมแล้วเกิน limit ({_merged_w:.0f}kg/{_merged_c:.2f}cbm) → skip")
                                continue
                            _len_a2 = len(_trip_coord_map2.get(_ta2, []))
                            _len_b2 = len(_trip_coord_map2.get(_tb2, []))
                            _base_sc2  = _ta2 if _len_a2 >= _len_b2 else _tb2
                            _other_sc2 = _tb2 if _base_sc2 == _ta2 else _ta2
                            df.loc[df['Trip'] == _other_sc2, 'Trip'] = _base_sc2
                            _samecoord_merged += 1
                            safe_print(f"   📍 รวม trip {_other_sc2} → {_base_sc2} ({_ca2}↔{_cb2} ห่าง {_d_sc2*1000:.0f}m name={_name_match})")
                            _sc_changed = True
                            break
    if _samecoord_merged:
        safe_print(f"✅ same-coord merge: รวม {_samecoord_merged} ทริป")

    # ถ้าทริปใด มีแค่ 1 unique Code และสาขาอยู่ห่างจาก solo-trip อื่น ≤ 500m → รวม
    # ยอมเกิน limit ได้เพราะสาขาอยู่จุดเดียวกัน (ต้องส่งพร้อมกัน)
    # ==========================================
    _SOLO_MERGE_KM = 0.5   # 500 เมตร
    safe_print(f"🔀 ตรวจสอบ same-location solo trips (≤{_SOLO_MERGE_KM*1000:.0f}m)...")
    _sc_merged = 0
    _sc_iters = 0
    while _sc_iters < 100:
        _sc_iters += 1
        _changed = False
        # สร้าง {trip_num: list of rows}
        _solo_info: dict = {}   # {trip_num: (code, lat, lon, region, province, district, subdistrict)}
        for _tnum_sc in df[df['Trip'] > 0]['Trip'].unique():
            _tdf = df[df['Trip'] == _tnum_sc]
            _ucodes = _tdf['Code'].astype(str).str.strip().str.upper().unique().tolist()
            if len(_ucodes) == 1:   # solo trip
                _r0_code = _ucodes[0]
                _r0 = _code_row_map.get(_r0_code, {})
                _lat0  = float(_r0.get('_lat', 0) or 0)
                _lon0  = float(_r0.get('_lon', 0) or 0)
                _reg0  = str(_r0.get('_region_name', '') or '')
                _prov0 = str(_r0.get('_province', '') or '')
                _dist0 = str(_r0.get('_district', '') or '')
                _sub0  = str(_r0.get('_subdistrict', '') or '')
                _solo_info[_tnum_sc] = (_ucodes[0], _lat0, _lon0, _reg0, _prov0, _dist0, _sub0)
        if len(_solo_info) < 2:
            break
        _solo_list = sorted(_solo_info.items())   # [(trip_num, (...)), ...]
        _merged_this_round: set = set()
        for _i, (_ta, (_ca, _lata, _lona, _rega, _pova, _disa, _suba)) in enumerate(_solo_list):
            if _ta in _merged_this_round:
                continue
            for _tb, (_cb, _latb, _lonb, _regb, _povb, _disb, _subb) in _solo_list[_i+1:]:
                if _tb in _merged_this_round:
                    continue
                # ตรวจ ตำบล อำเภอ จังหวัด ภาค ต้องตรงกัน (ถ้ามีค่า)
                def _ne(a, b): return a and b and a not in ('', 'ไม่ระบุ') and b not in ('', 'ไม่ระบุ') and a != b
                if _ne(_rega, _regb): continue
                if _ne(_pova, _povb): continue
                if _ne(_disa, _disb): continue
                if _ne(_suba, _subb): continue
                # ตรวจระยะ
                if _lata <= 0 or _lona <= 0 or _latb <= 0 or _lonb <= 0:
                    continue
                _d = haversine_distance(_lata, _lona, _latb, _lonb, use_osrm_cache=False)
                if _d <= _SOLO_MERGE_KM:
                    # 🔒 ตรวจ capacity ก่อนรวม — ห้ามเกินเด็ดขาด
                    _tda = df[df['Trip'] == _ta]
                    _tdb = df[df['Trip'] == _tb]
                    _solo_w = _tda['Weight'].sum() + _tdb['Weight'].sum()
                    _solo_c = _tda['Cube'].sum() + _tdb['Cube'].sum()
                    _solo_codes = _tda['Code'].tolist() + _tdb['Code'].tolist()
                    _is_pt_sl = all(branch_bu_cache.get(str(c).strip().upper(), False) for c in _solo_codes)
                    _buf_sl = punthai_buffer if _is_pt_sl else maxmart_buffer
                    _lims_sl = PUNTHAI_LIMITS if _is_pt_sl else LIMITS
                    _allowed_sl = get_allowed_from_codes(_solo_codes, ['4W', 'JB', '6W'])
                    
                    # 🚨 STRICT ENFORCEMENT: ตรวจสอบให้แน่ใจว่าไม่เกินข้อจำกัดจริงก่อน
                    _fits_strict = any(
                        _solo_w <= _lims_sl[v]['max_w'] and
                        _solo_c <= _lims_sl[v]['max_c']
                        for v in (['4W', 'JB', '6W'] if not _allowed_sl else _allowed_sl)
                    )
                    
                    # สำหรับ Maxmart: ตรวจสอบกับ buffer (แต่ Punthai ต้องเคร่งครัดเสมอ)
                    _fits_buffer = False
                    if not _is_pt_sl:
                        _fits_buffer = any(
                            _solo_w <= _lims_sl[v]['max_w'] * _buf_sl and
                            _solo_c <= _lims_sl[v]['max_c'] * _buf_sl
                            for v in (['4W', 'JB', '6W'] if not _allowed_sl else _allowed_sl)
                        )
                    
                    if not _fits_strict and (not _fits_buffer):
                        safe_print(f"   ⚠️ solo-merge skip: {_ca}↔{_cb} รวมแล้วเกิน limit ({_solo_w:.0f}kg/{_solo_c:.2f}cbm) → skip")
                        continue
                    # รวม trip_num ใหญ่ → เล็ก
                    _base_t2  = min(_ta, _tb)
                    _other_t2 = max(_ta, _tb)
                    df.loc[df['Trip'] == _other_t2, 'Trip'] = _base_t2
                    _merged_this_round.add(_other_t2)
                    _sc_merged += 1
                    safe_print(f"   🔀 รวม trip {_other_t2}({_cb}) → {_base_t2}({_ca}) ห่าง {_d*1000:.0f}m [{_suba}/{_disa}/{_pova}]")
                    _changed = True
                    break   # หาพาร์ทเนอร์ให้ _ta ได้แล้ว → ไปตัวถัดไป
        if not _changed:
            break
    if _sc_merged:
        safe_print(f"✅ same-location merge: รวม {_sc_merged} ทริป")


    # ==========================================
    # เรียงทริปชั่วคราว: zone_priority → region_order → max_distance (ไกลก่อน)
    _t5_sort = {}
    for trip_num in df[df['Trip'] > 0]['Trip'].unique():
        _td5 = df[df['Trip'] == trip_num]
        _zp5 = int(_td5['_zone_priority'].mode()[0]) if '_zone_priority' in _td5.columns and not _td5['_zone_priority'].mode().empty else 99
        _pv5 = _td5['_province'].mode()[0] if '_province' in _td5.columns and not _td5['_province'].mode().empty else ''
        _ro5 = REGION_ORDER.get(get_region_name(str(_pv5)), 99) if _pv5 else 99
        _md5 = _td5['_distance_from_dc'].max() if '_distance_from_dc' in _td5.columns else 0
        _t5_sort[trip_num] = (_zp5, _ro5, -(_md5 or 0))
    sorted_trips = sorted(_t5_sort.keys(), key=lambda x: _t5_sort[x])
    trip_mapping = {old_num: new_num for new_num, old_num in enumerate(sorted_trips, start=1)}
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
        
        # 🚨 STRICT ENFORCEMENT: ใช้ข้อจำกัดจริง ไม่คูณด้วย buffer
        max_w = limits[allowed_vehicle]['max_w']
        max_c = limits[allowed_vehicle]['max_c']
        max_drops = limits[allowed_vehicle]['max_drops']
        
        # 💾 เก็บ buffer ไว้สำหรับการตรวจสอบเพิ่มเติม (ถ้าต้องการ)
        buffered_max_w = limits[allowed_vehicle]['max_w'] * buffer
        buffered_max_c = limits[allowed_vehicle]['max_c'] * buffer
        
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

        # 🚨 STRICT ENFORCEMENT: ตรวจสอบให้แน่ใจว่าไม่เกินข้อจำกัดจริง
        # ตรวจสอบกับข้อจำกัดจริงก่อน (ไม่ใช้ buffer)
        if new_w > eff_limits['max_w']:
            return False, f"น้ำหนักเกินข้อจำกัด ({effective_vehicle}): {new_w} > {eff_limits['max_w']}"
        if new_c > eff_limits['max_c']:
            return False, f"ปริมาตรเกินข้อจำกัด ({effective_vehicle}): {new_c} > {eff_limits['max_c']}"
        if new_drops > eff_limits['max_drops']:
            return False, f"Drop เกินข้อจำกัด ({effective_vehicle}): {new_drops} > {eff_limits['max_drops']}"

        branch_is_punthai = branch_bu_cache.get(branch_code, False)
        is_punthai = trip_capacity.get('is_punthai', False) and branch_is_punthai
        eff_limits = (PUNTHAI_LIMITS if is_punthai else LIMITS)[effective_vehicle]
        eff_buffer = punthai_buffer if is_punthai else maxmart_buffer

        # สำหรับ Maxmart: ตรวจสอบกับ buffer (แต่ Punthai ต้องเคร่งครัดเสมอ)
        if not is_punthai:
            if new_w > eff_limits['max_w'] * eff_buffer:
                return False, f"น้ำหนักเกิน buffer ({effective_vehicle}): {new_w} > {eff_limits['max_w'] * eff_buffer}"
            if new_c > eff_limits['max_c'] * eff_buffer:
                return False, f"ปริมาตรเกิน buffer ({effective_vehicle}): {new_c} > {eff_limits['max_c'] * eff_buffer}"

        return True, "OK"
    
    def get_nearby_branches(branch_row, all_branches_df, max_dist_km=6.0):
        """หาสาขาที่อยู่ใกล้กัน — 🚀 vectorized numpy (แทน iterrows)"""
        if all_branches_df.empty:
            return []
        branch_lat = float(branch_row.get('_lat', 0) or 0)
        branch_lon = float(branch_row.get('_lon', 0) or 0)
        branch_subdistrict = branch_row.get('_subdistrict', '')
        branch_code = branch_row['Code']
        nearby_set = set()
        # 1. ตำบลเดียวกัน
        if branch_subdistrict:
            _sub_mask = (all_branches_df['_subdistrict'] == branch_subdistrict) & (all_branches_df['Code'] != branch_code)
            nearby_set.update(all_branches_df.loc[_sub_mask, 'Code'].tolist())
        # 2. ห่างกัน < max_dist_km (vectorized haversine)
        if branch_lat > 0 and branch_lon > 0:
            _nb_lats = all_branches_df['_lat'].fillna(0).to_numpy(dtype=float)
            _nb_lons = all_branches_df['_lon'].fillna(0).to_numpy(dtype=float)
            _nb_codes = all_branches_df['Code'].tolist()
            _nb_valid = _nb_lats > 0
            if _nb_valid.any():
                _dphi_nb = np.radians(_nb_lats[_nb_valid] - branch_lat)
                _dlam_nb = np.radians(_nb_lons[_nb_valid] - branch_lon)
                _a_nb = (np.sin(_dphi_nb / 2) ** 2 +
                         np.cos(np.radians(branch_lat)) * np.cos(np.radians(_nb_lats[_nb_valid])) *
                         np.sin(_dlam_nb / 2) ** 2)
                _d_nb = 6371.0 * 2 * np.arctan2(np.sqrt(_a_nb), np.sqrt(1 - _a_nb))
                _near_idxs = np.where(_nb_valid)[0][_d_nb <= max_dist_km]
                for _i_nb in _near_idxs:
                    if _nb_codes[_i_nb] != branch_code:
                        nearby_set.add(_nb_codes[_i_nb])
        return list(nearby_set)
    
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
            # 🚀 SPEED: vectorized haversine แทน apply row-by-row
            _ntd_lats = next_trip_data['_lat'].fillna(0).to_numpy(dtype=float)
            _ntd_lons = next_trip_data['_lon'].fillna(0).to_numpy(dtype=float)
            _ntd_valid = _ntd_lats > 0
            _ntd_dists = np.full(len(next_trip_data), 999.0)
            _clat_r, _clon_r = trip_cap['centroid_lat'], trip_cap['centroid_lon']
            if _ntd_valid.any() and _clat_r > 0 and _clon_r > 0:
                _dphi_n = np.radians(_ntd_lats[_ntd_valid] - _clat_r)
                _dlam_n = np.radians(_ntd_lons[_ntd_valid] - _clon_r)
                _a_n = (np.sin(_dphi_n / 2) ** 2 +
                        np.cos(np.radians(_clat_r)) * np.cos(np.radians(_ntd_lats[_ntd_valid])) *
                        np.sin(_dlam_n / 2) ** 2)
                _ntd_dists[_ntd_valid] = 6371.0 * 2 * np.arctan2(np.sqrt(_a_n), np.sqrt(1 - _a_n)) * 1.35
            next_trip_data = next_trip_data.copy()
            next_trip_data['_dist_to_current'] = _ntd_dists
            
            # เรียงตามระยะใกล้สุดก่อน
            next_trip_data = next_trip_data.sort_values('_dist_to_current')
            
            # เก็บสาขาที่ย้ายแล้วเพื่อไม่ให้ซ้ำ
            already_moved = set()
            
            # ดึงสาขาที่ใกล้และเข้ากันได้
            # 🚀 SPEED: to_dict('records') แทน iterrows (~5x เร็วกว่า)
            for branch_row in next_trip_data.to_dict('records'):
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
                
                # อัปเดต trip_cap เพราะอาจมีการเพิ่มสาขาแล้ว (ใช้ incremental — ไม่ re-scan df)
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
                # 🔒 ตรวจภาคก่อน — 🚀 ใช้ _prov_region_map แทน get_region_name()
                _b_region = _prov_region_map.get(str(_b_prov), '') if _b_prov else ''
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
                    # 🚀 Incremental update trip_cap (ไม่ re-scan df)
                    _add_w66 = float(branch_row.get('Weight', 0) or 0)
                    _add_c66 = float(branch_row.get('Cube', 0) or 0)
                    _add_pv66 = str(branch_row.get('_province', '') or '')
                    _add_zn66 = str(branch_row.get('_logistics_zone', '') or '')
                    _add_hw66 = str(branch_row.get('_zone_highway', '') or '')
                    _add_rg66 = _prov_region_map.get(_add_pv66, '')
                    trip_cap['weight'] += _add_w66
                    trip_cap['cube']   += _add_c66
                    trip_cap['drops']  += 1
                    trip_cap['codes'].append(branch_code)
                    if _add_pv66: trip_cap['provinces'].add(_add_pv66)
                    if _add_zn66: trip_cap['logistics_zones'].add(_add_zn66)
                    if _add_hw66: trip_cap['highways'].update(_add_hw66.split('/'))
                    if _add_rg66 and _add_rg66 != 'ไม่ระบุ': trip_cap['regions'].add(_add_rg66)
                    # เช็ค vehicle downgrade (ระวัง: สาขาใหม่อาจจำกัดรถเล็กลง)
                    _add_vr66 = vehicle_priority.get(branch_max_vehicle_cache.get(branch_code, '6W'), 3)
                    if _add_vr66 < trip_cap.get('min_priority', 3):
                        trip_cap['min_priority'] = _add_vr66
                        trip_cap['allowed_vehicle'] = {1: '4W', 2: 'JB', 3: '6W'}.get(_add_vr66, '6W')
                        _is_pt66 = trip_cap.get('is_punthai', False)
                        _buf66 = punthai_buffer if _is_pt66 else maxmart_buffer
                        _lims66 = PUNTHAI_LIMITS if _is_pt66 else LIMITS
                        trip_cap['max_w'] = _lims66[trip_cap['allowed_vehicle']]['max_w'] * _buf66
                        trip_cap['max_c'] = _lims66[trip_cap['allowed_vehicle']]['max_c'] * _buf66
                    if not branch_bu_cache.get(branch_code, False) and trip_cap.get('is_punthai', False):
                        trip_cap['is_punthai'] = False
                        _buf66b = maxmart_buffer
                        _lims66b = LIMITS
                        trip_cap['max_w'] = _lims66b[trip_cap['allowed_vehicle']]['max_w'] * _buf66b
                        trip_cap['max_c'] = _lims66b[trip_cap['allowed_vehicle']]['max_c'] * _buf66b
                    
                    # 🔗 หาสาขาใกล้เคียง (ตำบลเดียวกัน หรือ ห่าง < 6 km) แล้วย้ายมาด้วย
                    nearby_codes = get_nearby_branches(branch_row, next_trip_data[~next_trip_data['Code'].isin(already_moved)])
                    
                    for nearby_code in nearby_codes:
                        if nearby_code in already_moved:
                            continue
                        
                        # เช็คว่าเต็มหรือยัง (ใช้ incremental trip_cap — ไม่ re-scan df)
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
                        # 🔒 region check ก่อน — 🚀 ใช้ _prov_region_map
                        _nb_region = _prov_region_map.get(str(_nb_prov), '') if _nb_prov else ''
                        _trip_regions_nb = trip_cap.get('regions', set())
                        _nb_region_ok = (
                            not _nb_region or _nb_region == 'ไม่ระบุ' or   # candidate ไม่ทราบภาค
                            not _trip_regions_nb or                           # ทริปยังไม่รู้ภาค → ใช้ BKK-isolation เป็น guard
                            _nb_region in _trip_regions_nb                    # ภาคตรงกัน
                        )
                        if not _nb_region_ok:
                            continue
                        # 🔒 เพิ่ม: ถ้าทริปรู้ภาคแล้ว และ candidate ไม่รู้ภาค → ตรวจระยะห่าง ≤ 20km
                        if _trip_regions_nb and (not _nb_region or _nb_region == 'ไม่ระบุ'):
                            _nbx_lat2 = float(nearby_row.get('_lat', 0) or 0)
                            _nbx_lon2 = float(nearby_row.get('_lon', 0) or 0)
                            _txn_lat2 = float(trip_cap.get('centroid_lat', 0) or 0)
                            _txn_lon2 = float(trip_cap.get('centroid_lon', 0) or 0)
                            if _nbx_lat2 > 0 and _txn_lat2 > 0:
                                _dp8 = radians(_nbx_lat2 - _txn_lat2); _dl8 = radians(_nbx_lon2 - _txn_lon2)
                                _a8 = sin(_dp8/2)**2 + cos(radians(_txn_lat2))*cos(radians(_nbx_lat2))*sin(_dl8/2)**2
                                if 2*6371*atan2(sqrt(_a8), sqrt(1-_a8)) > 20.0:
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
                            # 🚀 Incremental update trip_cap (ไม่ re-scan df)
                            _add_wnb = float(nearby_row.get('Weight', 0) or 0)
                            _add_cnb = float(nearby_row.get('Cube', 0) or 0)
                            _add_pvnb = str(nearby_row.get('_province', '') or '')
                            _add_znnb = str(nearby_row.get('_logistics_zone', '') or '')
                            _add_hwnb = str(nearby_row.get('_zone_highway', '') or '')
                            _add_rgnb = _prov_region_map.get(_add_pvnb, '')
                            trip_cap['weight'] += _add_wnb
                            trip_cap['cube']   += _add_cnb
                            trip_cap['drops']  += 1
                            trip_cap['codes'].append(nearby_code)
                            if _add_pvnb: trip_cap['provinces'].add(_add_pvnb)
                            if _add_znnb: trip_cap['logistics_zones'].add(_add_znnb)
                            if _add_hwnb: trip_cap['highways'].update(_add_hwnb.split('/'))
                            if _add_rgnb and _add_rgnb != 'ไม่ระบุ': trip_cap['regions'].add(_add_rgnb)
                            _add_vrnb = vehicle_priority.get(branch_max_vehicle_cache.get(nearby_code, '6W'), 3)
                            if _add_vrnb < trip_cap.get('min_priority', 3):
                                trip_cap['min_priority'] = _add_vrnb
                                trip_cap['allowed_vehicle'] = {1: '4W', 2: 'JB', 3: '6W'}.get(_add_vrnb, '6W')
                                _is_ptnb = trip_cap.get('is_punthai', False)
                                _bufnb = punthai_buffer if _is_ptnb else maxmart_buffer
                                _limsnb = PUNTHAI_LIMITS if _is_ptnb else LIMITS
                                trip_cap['max_w'] = _limsnb[trip_cap['allowed_vehicle']]['max_w'] * _bufnb
                                trip_cap['max_c'] = _limsnb[trip_cap['allowed_vehicle']]['max_c'] * _bufnb
                            if not branch_bu_cache.get(nearby_code, False) and trip_cap.get('is_punthai', False):
                                trip_cap['is_punthai'] = False
                                trip_cap['max_w'] = LIMITS[trip_cap['allowed_vehicle']]['max_w'] * maxmart_buffer
                                trip_cap['max_c'] = LIMITS[trip_cap['allowed_vehicle']]['max_c'] * maxmart_buffer
        
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
    MIN_CONSOLIDATION_UTIL = 1.0  # รวมทริปที่ยังไม่เต็ม 100% เสมอ (ไม่ปล่อยให้หลุด)
    _consol_rounds = 0
    _consol_total = 0
    # 🚀 SPEED: สร้าง _caps_cs ครั้งเดียว แล้วอัปเดต incremental (แทน rebuild ทุก round)
    _trips_now_cs = sorted(df[df['Trip'] > 0]['Trip'].unique())
    _caps_cs = {}
    for _t_cs in _trips_now_cs:
        _c_cs = get_trip_capacity(_t_cs)
        if _c_cs:
            _caps_cs[_t_cs] = _c_cs
    while _consol_rounds < 30:
        _consol_rounds += 1

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
        # 🚀 Vectorized: คำนวณ region ทุกแถวพร้อมกัน
        _aud_prov_ser = _aud_data['_province'].fillna('').astype(str)
        _aud_reg_ser  = _aud_prov_ser.map(_prov_region_map).fillna('')
        # fallback จาก _region_name column
        _aud_reg_ser  = _aud_reg_ser.where(_aud_reg_ser != '',
                           _aud_data['_region_name'].fillna('').astype(str))
        _aud_reg_valid = _aud_reg_ser[(_aud_reg_ser != '') & (_aud_reg_ser != 'ไม่ระบุ')]
        _aud_regions = _aud_reg_valid.value_counts().to_dict()
        if len(_aud_regions) <= 1:
            continue  # ไม่มีการปนภาค
        _dominant = max(_aud_regions, key=lambda k: (_aud_regions[k], ['เหนือ','อีสาน','ตะวันออก','กลาง','ตะวันตก','ใต้'].index(k) if k in ['เหนือ','อีสาน','ตะวันออก','กลาง','ตะวันตก','ใต้'] else 99))
        # minority mask — vectorized
        _min_mask = (_aud_reg_ser != '') & (_aud_reg_ser != 'ไม่ระบุ') & (_aud_reg_ser != _dominant)
        _minority_codes = _aud_data.loc[_min_mask[_min_mask].index, 'Code'].tolist()
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
        # ⚠️ ใช้ _vehicle_rank จาก df โดยตรง (PRE-MERGE อาจอัปเดต rank ของ primary ให้รัดกุมขึ้นแล้ว)
        # ห้ามใช้ get_max_vehicle_for_branch อีกครั้ง เพราะจะอ่านจาก MASTER_DATA_DICT ซึ่งไม่รู้ shadow constraint
        if '_vehicle_rank' in trip_data.columns and not trip_data['_vehicle_rank'].isna().all():
            min_max_size = int(trip_data['_vehicle_rank'].min())
        else:
            max_vehicles = [get_max_vehicle_for_branch(c) for c in trip_codes]
            min_max_size = min(vehicle_priority.get(v, 3) for v in max_vehicles)
        max_allowed_vehicle = {1: '4W', 2: 'JB', 3: '6W'}.get(min_max_size, '6W')
        
        # ตรวจ BU ของทริป (🚀 vectorized — ไม่ใช้ iterrows)
        is_punthai_only_trip = trip_data['BU'].astype(str).str.strip().str.upper().isin(['211', 'PUNTHAI']).all()
        
        buffer = punthai_buffer if is_punthai_only_trip else maxmart_buffer
        buffer_pct = int(buffer * 100)
        buffer_label = f"🅿️ {buffer_pct}%" if is_punthai_only_trip else f"🅼 {buffer_pct}%"
        trip_type = 'punthai' if is_punthai_only_trip else 'maxmart'
        
        # 🎯 เลือกรถตามภาค + ข้อจำกัดสาขาจาก master
        # เหนือ/ใต้ → ใช้ max_allowed_vehicle (รถใหญ่สุดที่ master อนุญาต) — เส้นทางไกล ไม่ downgrade
        # ภาคอื่น  → เล็กสุดที่รับโหลดได้ (4W → JB → 6W) — ไม่เกิน master limit
        limits_to_check = PUNTHAI_LIMITS if is_punthai_only_trip else LIMITS
        is_long_haul = str(trip_region) in ('เหนือ', 'ใต้')
        suggested = max_allowed_vehicle  # fallback = รถใหญ่สุดที่ master อนุญาต
        if is_long_haul:
            # เหนือ/ใต้: ใช้ max_allowed_vehicle ตรงๆ (ไม่ downgrade — เส้นทางไกล)
            source = "🚛 ไกล (เหนือ/ใต้)" if min_max_size >= 3 else "📋 จำกัดสาขา (เหนือ/ใต้)"
        else:
            # ภาคอื่น: เล็กสุดที่รับโหลดได้ ไม่เกินข้อจำกัด master
            source = "📋 จำกัดสาขา" if min_max_size < 3 else "🤖 อัตโนมัติ"
            for _veh in ['4W', 'JB', '6W']:
                _vr = vehicle_priority.get(_veh, 3)
                if _vr > min_max_size:
                    break  # ห้ามเกินข้อจำกัด master
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
        
        # คำนวณ utilization - วัดเทียบกับขีดจำกัดจริง (รวม buffer) → 100% = เต็มขีดจำกัด
        max_util_threshold = 100  # 100% ของขีดจำกัดจริง (buffer รวมอยู่แล้วในตัวหาร)
        limits_for_util = PUNTHAI_LIMITS if is_punthai_only_trip else LIMITS
        if suggested in limits_for_util:
            w_util = (total_w / (limits_for_util[suggested]['max_w'] * buffer)) * 100
            c_util = (total_c / (limits_for_util[suggested]['max_c'] * buffer)) * 100
            max_util = max(w_util, c_util)
            # ถ้าเกิน 100% ของขีดจำกัด (รวม buffer) → ต้องเพิ่มขนาดรถ
            if max_util > max_util_threshold:
                if suggested == '4W' and min_max_size >= 2:
                    jb_util = max(
                        (total_w / (limits_for_util['JB']['max_w'] * buffer)),
                        (total_c / (limits_for_util['JB']['max_c'] * buffer))
                    ) * 100
                    if jb_util <= max_util_threshold:
                        suggested = 'JB'
                        source += " → JB"
                        w_util = (total_w / (limits_for_util['JB']['max_w'] * buffer)) * 100
                        c_util = (total_c / (limits_for_util['JB']['max_c'] * buffer)) * 100
                    elif min_max_size >= 3:
                        suggested = '6W'
                        source += " → 6W"
                        w_util = (total_w / (limits_for_util['6W']['max_w'] * buffer)) * 100
                        c_util = (total_c / (limits_for_util['6W']['max_c'] * buffer)) * 100
                    else:
                        source += " ⚠️ เกินแต่สาขาจำกัด"
                elif suggested == 'JB' and min_max_size >= 3:
                    suggested = '6W'
                    source += " → 6W"
                    w_util = (total_w / (limits_for_util['6W']['max_w'] * buffer)) * 100
                    c_util = (total_c / (limits_for_util['6W']['max_c'] * buffer)) * 100
                else:
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
            # คำนวณระยะทางรวม: ลอง ROUTE_CACHE ก่อน; ถ้าไม่มีใช้ haversine×1.35 ประมาณ (ไม่เรียก network)
            _wp_td = [[DC_WANG_NOI_LAT, DC_WANG_NOI_LON]] + [[la, lo] for la, lo in branch_coords] + [[DC_WANG_NOI_LAT, DC_WANG_NOI_LON]]
            _ck_td = "|".join([f"{la:.4f},{lo:.4f}" for la, lo in _wp_td])
            if USE_CACHE and _ck_td in ROUTE_CACHE_DATA:
                _rc_td = ROUTE_CACHE_DATA[_ck_td]
                total_distance = _rc_td.get('distance', 0)
            else:
                # ประมาณจาก haversine×1.35 (zero-network) DC→b1→b2→...→DC
                _pts = _wp_td
                total_distance = sum(
                    haversine_distance(_pts[_pi][0], _pts[_pi][1], _pts[_pi+1][0], _pts[_pi+1][1], use_osrm_cache=False)
                    for _pi in range(len(_pts) - 1)
                )

        # � HARD Fleet Enforcement: ห้ามเกินโควต้าเด็ดขาด — upgrade ทิ้ง branch constraint ถ้าจำเป็น
        _sv = suggested  # บันทึกรถเดิม
        _sv_rank = _rank_fleet.get(suggested, 3)
        _upgraded_by_fleet = False
        while fleet_used.get(suggested, 0) >= _fleet_limits.get(suggested, 999):
            _next_rank = _sv_rank + 1
            if _next_rank > 3:
                # ไม่มีรถใหญ่กว่า 6W แล้ว → overflow จริงๆ ไม่มีทางเลือก
                source += " ⚠️ เกินโควต้า (ไม่มีรถเหลือ)"
                break
            _next_veh = _fleet_rank.get(_next_rank, '6W')
            _prev_suggested = suggested
            # บังคับ upgrade — fleet enforcement override branch constraint เสมอ
            suggested = _next_veh
            _sv_rank = _next_rank
            _upgraded_by_fleet = True
            if _next_rank > min_max_size:
                source += f" ⚠️(branch จำกัด {max_allowed_vehicle} แต่ fleet บังคับ {suggested})"
            safe_print(f"      🚛 Fleet HARD: Trip {trip_num} {_prev_suggested}→{suggested} "
                       f"(โควต้า {_prev_suggested} เต็ม {fleet_used.get(_prev_suggested,0)}/{_fleet_limits.get(_prev_suggested,999)})")
        if _upgraded_by_fleet:
            source += f" ↑ Fleet({_sv}→{suggested})"
            # คำนวณ utilization ใหม่ด้วยรถที่ upgrade
            if suggested in limits_for_util:
                w_util = (total_w / (limits_for_util[suggested]['max_w'] * buffer)) * 100
                c_util = (total_c / (limits_for_util[suggested]['max_c'] * buffer)) * 100
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
        # ⚠️ ใช้ _vehicle_rank จาก df (PRE-MERGE อาจอัปเดต primary's rank แล้ว)
        vehicle_priority_map = {'4W': 1, 'JB': 2, '6W': 3}
        if '_vehicle_rank' in trip_data.columns and not trip_data['_vehicle_rank'].isna().all():
            min_max_size = int(trip_data['_vehicle_rank'].min())
        else:
            max_vehicles = [get_max_vehicle_for_branch(c) for c in trip_codes]
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
        
        # 🎯 เลือกรถตามภาค + ข้อจำกัดสาขาจาก master
        # เหนือ/ใต้ → max_allowed_v (ยึดตาม master) — ไม่ downgrade
        # ภาคอื่น  → เล็กสุดที่รับโหลดได้ — ไม่เกิน master limit
        trip_region_75 = trip_data['_region_name'].iloc[0] if '_region_name' in trip_data.columns else 'ไม่ระบุ'
        is_long_haul_75 = str(trip_region_75) in ('เหนือ', 'ใต้')
        correct_vehicle = max_allowed_v  # fallback = max ตาม master
        if is_long_haul_75:
            # เหนือ/ใต้: ใช้ max_allowed_v (เส้นทางไกล ไม่ downgrade)
            pass
        else:
            # ภาคอื่น: เล็กสุดที่รับโหลดได้ — น้ำหนัก/คิ้วห้ามเกิน
            # Pass 1: ไม่เกินข้อจำกัด master
            for _veh in ['4W', 'JB', '6W']:
                _vr = vehicle_priority_map.get(_veh, 3)
                if _vr > min_max_size:
                    break  # ห้ามเกินข้อจำกัด master
                _lim = limits[_veh]
                if (total_w <= _lim['max_w'] * buffer and
                        total_c <= _lim['max_c'] * buffer and
                        len(trip_codes) <= _lim['max_drops']):
                    correct_vehicle = _veh
                    break  # เล็กสุดที่รับโหลดได้
        max_w = limits[correct_vehicle]['max_w'] * buffer
        max_c = limits[correct_vehicle]['max_c'] * buffer
        max_drops = limits[correct_vehicle]['max_drops']
        
        w_util = (total_w / max_w) * 100
        c_util = (total_c / max_c) * 100
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
            
            max_w = limits[truck_str]['max_w']
            max_c = limits[truck_str]['max_c']
            
            # คำนวณน้ำหนัก/คิวปัจจุบัน
            current_w = trip_data['Weight'].sum()
            current_c = trip_data['Cube'].sum()
            current_drops = len(trip_data)
            
            # ตัดสาขาออกจนกว่าจะไม่เกิน (weight, cube, และ drops)
            # 🔗 รักษา coord-300m group: ตัดเป็น atomic unit (ตัดทั้งกลุ่มหรือไม่ตัดเลย)
            _trip75_upper = {str(c).strip().upper() for c in trip_codes}
            _processed75 = set()
            _cut_units = []  # list of (max_dist, actual_codes_list, total_w, total_c, total_drops)
            for _, _r75 in trip_data.drop_duplicates(subset='Code').iterrows():
                _cu75 = str(_r75['Code']).strip().upper()
                if _cu75 in _processed75:
                    continue
                # ดึง coord-300m partners ที่อยู่ในทริปเดียวกัน
                _ptrs75 = [p for p in _suffix_force_partners.get(_cu75, []) if p in _trip75_upper and p not in _processed75]
                _unit_uppers = [_cu75] + _ptrs75
                for _uu in _unit_uppers:
                    _processed75.add(_uu)
                _unit_actuals = [_code_real_map.get(_uu, _uu) for _uu in _unit_uppers]
                _unit_rows = trip_data[trip_data['Code'].isin(_unit_actuals)]
                _unit_w = _unit_rows['Weight'].sum()
                _unit_c = _unit_rows['Cube'].sum()
                _unit_drops = _unit_rows['Code'].nunique()
                _unit_dist = float(_unit_rows['_distance_from_dc'].max()) if '_distance_from_dc' in _unit_rows.columns else 0.0
                _cut_units.append((_unit_dist, _unit_actuals, _unit_w, _unit_c, _unit_drops))
            # เรียงตาม max distance จาก DC (ไกลสุดก่อน)
            _cut_units.sort(key=lambda x: x[0], reverse=True)

            codes_to_remove = []
            for _ud, _ucodes, _uw, _uc_val, _udrops in _cut_units:
                if current_w <= max_w and current_c <= max_c and current_drops <= max_drops:
                    break
                for _uc_code in _ucodes:
                    codes_to_remove.append(_uc_code)
                    overflow_branches.append(_uc_code)
                current_w -= _uw
                current_c -= _uc_val
                current_drops -= _udrops
                if len(_ucodes) == 1:
                    safe_print(f"      🔪 ตัด {_ucodes[0]} ออก (ไกลสุด {_ud:.1f} km)")
                else:
                    safe_print(f"      🔪 ตัด coord-300m group {_ucodes} ออก (ไกลสุด {_ud:.1f} km)")
            
            # ลบสาขาออกจากทริป (Trip = 0)
            for code in codes_to_remove:
                df.loc[df['Code'] == code, 'Trip'] = 0
            
            # อัพเดต summary
            if codes_to_remove:
                new_trip_data = df[df['Trip'] == trip_num]
                new_w = new_trip_data['Weight'].sum()
                new_c = new_trip_data['Cube'].sum()
                new_w_util = (new_w / limits[truck_str]['max_w']) * 100
                new_c_util = (new_c / limits[truck_str]['max_c']) * 100
                
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
                        # สาขาเดียวเกิน max_veh limit → ลอง upgrade vehicle ก่อน
                        _ov_upgraded = None
                        _veh_order = ['4W', 'JB', '6W']
                        _ov_code_rank = {'4W': 1, 'JB': 2, '6W': 3}
                        _ov_br_rank = _ov_code_rank.get(get_max_vehicle_for_branch(code), 3)
                        for _try_veh in _veh_order:
                            if _ov_code_rank.get(_try_veh, 3) < _ov_code_rank.get(max_veh, 1):
                                continue  # ข้ามรถที่เล็กกว่า max_veh
                            if _ov_code_rank.get(_try_veh, 3) > _ov_br_rank:
                                continue  # ข้ามรถที่ใหญ่กว่าข้อจำกัดสาขา
                            _try_lim = overflow_limits[_try_veh]
                            if (code_w <= _try_lim['max_w'] * overflow_buffer and
                                    code_c <= _try_lim['max_c'] * overflow_buffer):
                                _ov_upgraded = _try_veh
                                break
                        if _ov_upgraded and _ov_upgraded != max_veh:
                            # ย้ายไปกลุ่มรถที่ใหญ่กว่า แล้วหยุดทริปนี้ (ไม่รวมสาขานี้)
                            if _ov_upgraded not in overflow_by_max_vehicle:
                                overflow_by_max_vehicle[_ov_upgraded] = []
                            overflow_by_max_vehicle[_ov_upgraded].insert(0, code)
                            remaining_codes.remove(code)
                            safe_print(f"   🔼 overflow upgrade: {code} {max_veh}→{_ov_upgraded} ({code_w:.0f}kg)")
                            break
                        # ถ้าไม่มีรถใหญ่กว่าที่รับได้ → เพิ่มอยู่ดี (unavoidable)
                        trip_codes.append(code)
                        trip_weight += code_w
                        trip_cube += code_c
                        trip_drops += n_rows
                        remaining_codes.remove(code)
                        safe_print(f"   ⚠️ unavoidable overflow: {code} {code_w:.0f}kg > {max_veh} limit")
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
                    overflow_buffer_final = 1.0
                    buffer_label = f"🅿️ {int(overflow_buffer_final*100)}%" if is_overflow_punthai else f"🅼 {int(overflow_buffer_final*100)}%"
                    
                    summary_data.append({
                        'Trip': new_trip,
                        'Branches': len(_ov_actual),
                        'Weight': _ov_w,
                        'Cube': _ov_c,
                        'Truck': f'{max_veh} 🔪 ตัดออก',
                        'BU_Type': 'punthai' if is_overflow_punthai else 'mixed',
                        'Buffer': buffer_label,
                        'Weight_Use%': (_ov_w / (overflow_limits_final[max_veh]['max_w'] * overflow_buffer_final)) * 100,
                        'Cube_Use%': (_ov_c / (overflow_limits_final[max_veh]['max_c'] * overflow_buffer_final)) * 100,
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
    
    # � SPEED: vectorized vehicle compliance check (แทน df.apply row-by-row)
    def _compute_vehicle_check(df_in):
        _vr = {'4W': 1, 'JB': 2, '6W': 3}
        _truck_s = df_in['Truck'].fillna('').astype(str).str.split().str[0].replace('4WJ', 'JB')
        _maxv_s  = df_in['_max_vehicle'].fillna('6W').astype(str)
        _rank_t  = _truck_s.map(_vr).fillna(3).astype(int)
        _rank_m  = _maxv_s.map(_vr).fillna(3).astype(int)
        _res = pd.Series('✅ ใช้ได้', index=df_in.index, dtype=object)
        _res[df_in['Trip'] == 0] = '⚠️ ไม่ได้จัด'
        _over = (_rank_t > _rank_m) & (df_in['Trip'] != 0)
        if _over.any():
            _res[_over] = '❌ เกินข้อจำกัด (Max: ' + _maxv_s[_over] + ', ใช้: ' + _truck_s[_over] + ')'
        return _res

    # (compat: keep check_vehicle_compliance for any code that still references it)
    def check_vehicle_compliance(row):
        if row['Trip'] == 0:
            return '⚠️ ไม่ได้จัด'
        _vr = {'4W': 1, 'JB': 2, '6W': 3}
        max_allowed = row['_max_vehicle']
        truck_assigned = str(row.get('Truck', '')).split()[0] if pd.notna(row.get('Truck')) else ''
        if truck_assigned == '4WJ':
            truck_assigned = 'JB'
        if max_allowed not in _vr or truck_assigned not in _vr:
            return '✅ ใช้ได้'
        if _vr[truck_assigned] <= _vr[max_allowed]:
            return '✅ ใช้ได้'
        return f'❌ เกินข้อจำกัด (Max: {max_allowed}, ใช้: {truck_assigned})'

    df['VehicleCheck'] = _compute_vehicle_check(df)
    
    # ==========================================
    # 🚨 Step 8.5: บังคับแก้ไขสาขาที่เกินข้อจำกัดรถ (Enforce Vehicle Constraints)
    # ==========================================
    safe_print("\n📋 Step 8.5: บังคับข้อจำกัดรถ...")
    vehicle_violations = df[df['VehicleCheck'].str.contains('❌', na=False)]

    if len(vehicle_violations) > 0:
        safe_print(f"   ⚠️ พบ {len(vehicle_violations)} สาขาที่ใช้รถเกินข้อจำกัด → บังคับ recalculate Truck ทุกทริปที่มีปัญหา")
        _85_vrank = {'4W': 1, 'JB': 2, '6W': 3}
        _85_rvrank = {1: '4W', 2: 'JB', 3: '6W'}
        # รวม trips ที่มี violation
        _violating_trips = vehicle_violations['Trip'].unique()
        for _vt in _violating_trips:
            _vt_data = df[df['Trip'] == _vt]
            # คำนวณ min rank จาก _vehicle_rank ของทุกสาขาในทริปนั้น
            if '_vehicle_rank' in _vt_data.columns and not _vt_data['_vehicle_rank'].isna().all():
                _vt_min_rank = int(_vt_data['_vehicle_rank'].min())
            else:
                _vt_maxveh = [branch_max_vehicle_cache.get(str(c).strip().upper(), '6W')
                              for c in _vt_data['Code']]
                _vt_min_rank = min(_85_vrank.get(v, 3) for v in _vt_maxveh)
            _vt_correct = _85_rvrank.get(_vt_min_rank, '6W')
            _vt_cur = str(_vt_data.iloc[0].get('Truck', '6W') or '6W').split()[0]
            # บังคับ Truck ให้ตรงกับ max_allowed (ไม่มีข้อยกเว้น)
            df.loc[df['Trip'] == _vt, 'Truck'] = f"{_vt_correct} 📋 จำกัดสาขา"
            safe_print(f"      🔒 Trip {_vt}: {_vt_cur} → {_vt_correct} (max allowed by branch constraint)")
        df['VehicleCheck'] = _compute_vehicle_check(df)
        remaining_violations = df[df['VehicleCheck'].str.contains('❌', na=False)]
        if remaining_violations.empty:
            safe_print(f"   ✅ Step 8.5: แก้ไขครบแล้ว — ไม่มี violation เหลือ")
        else:
            safe_print(f"   ⚠️ Step 8.5: ยังมี {len(remaining_violations)} violation (อาจเกิดจาก _vehicle_rank ไม่ sync)")
    else:
        safe_print("   ✅ Step 8.5: ไม่พบ violation")

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
        # 🚀 vectorized — ไม่ใช้ iterrows
        _fa_provs_clean = [p for p in _fa_data['_province'].fillna('').astype(str).tolist() if p and p != 'nan']

        # 1️⃣ BKK Isolation: กรุงเทพฯ ห้ามปนกับจังหวัดอื่น
        _fa_has_bkk = _BKK_PROV in _fa_provs_clean
        _fa_has_non_bkk = any(p != _BKK_PROV for p in _fa_provs_clean)
        if _fa_has_bkk and _fa_has_non_bkk:
            # แยกสาขาที่ไม่ใช่กรุงเทพฯ ออก
            _fa_split_codes = _fa_data[_fa_data['_province'].fillna('').astype(str) != _BKK_PROV]['Code'].tolist()
            if _fa_split_codes:
                _fa_max_trip += 1
                df.loc[df['Code'].isin(_fa_split_codes), 'Trip'] = _fa_max_trip
                safe_print(f"   🔒 BKK AUDIT: Trip {_fa_trip} → แยก {len(_fa_split_codes)} สาขา non-BKK → Trip {_fa_max_trip}")
                _final_audit_fixed += 1
            continue  # ตรวจข้ออื่นบนข้อมูลใหม่ในรอบถัดไป

        # 2️⃣ Region Mixing: ห้ามปนภาค (🚀 vectorized — ไม่ใช้ iterrows)
        _fa_prov_ser = _fa_data['_province'].fillna('').astype(str)
        _fa_reg_ser  = _fa_prov_ser.map(_prov_region_map).fillna('')
        # fallback: ใช้ _region_name column
        _fa_reg_ser  = _fa_reg_ser.where(_fa_reg_ser != '', _fa_data['_region_name'].fillna('').astype(str))
        _fa_reg_valid = _fa_reg_ser[(_fa_reg_ser != '') & (_fa_reg_ser != 'ไม่ระบุ')]
        _fa_regions = _fa_reg_valid.value_counts().to_dict()
        if len(_fa_regions) <= 1:
            continue  # clean — no mixing

        # พบการปนภาค → dominant = ภาคที่มีสาขามากสุด
        _fa_region_order = ['เหนือ', 'อีสาน', 'ตะวันออก', 'กลาง', 'ตะวันตก', 'ใต้']
        _fa_dominant = max(
            _fa_regions,
            key=lambda k: (_fa_regions[k], -(_fa_region_order.index(k) if k in _fa_region_order else 99))
        )
        _fa_minority_mask = _fa_reg_valid.index.isin(
            _fa_reg_valid[_fa_reg_valid != _fa_dominant].index
        ) if len(_fa_reg_valid) > 0 else pd.Series(dtype=bool)
        _fa_minority_codes = _fa_data.loc[
            _fa_data.index.isin(
                _fa_reg_ser[(_fa_reg_ser != '') & (_fa_reg_ser != 'ไม่ระบุ') & (_fa_reg_ser != _fa_dominant)].index
            )
        ]['Code'].tolist()
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
    # Step 8.9: Catch-all — สาขาที่ยังไม่ได้จัดทริป (Trip=0)
    # รองรับ Z*, LUBE, SUPPLY, USE, สาขาไม่มีพิกัด ฯลฯ
    # ==========================================
    _catchall_remaining = df[df['Trip'] == 0].copy()
    if len(_catchall_remaining) > 0:
        safe_print(f"\n⚠️  Step 8.9: พบ {len(_catchall_remaining)} สาขายังไม่ได้จัดทริป → จัดทริปเดี่ยว...")
        _ca_max_trip = int(df[df['Trip'] > 0]['Trip'].max()) if len(df[df['Trip'] > 0]) > 0 else 0
        for _, _ca_row in _catchall_remaining.iterrows():
            _ca_max_trip += 1
            _ca_code = _ca_row['Code']
            df.loc[df['Code'] == _ca_code, 'Trip'] = _ca_max_trip
            _ca_veh = branch_max_vehicle_cache.get(str(_ca_code).strip().upper(), '6W')
            df.loc[df['Code'] == _ca_code, 'Truck'] = f"{_ca_veh} ⚙️ จัดเดี่ยว"
            safe_print(f"   ➕ {_ca_code} → Trip {_ca_max_trip} ({_ca_veh})")
        safe_print(f"   ✅ Step 8.9: จัดทริปเพิ่ม {len(_catchall_remaining)} สาขา")

    # ==========================================
    # Step 8.92: 🚫 HARD VEHICLE CONSTRAINT ENFORCEMENT
    # บังคับ Truck ทุกทริปให้ ≤ max_allowed ของสาขาที่เข้มงวดที่สุดในทริป
    # รันหลังทุก step ก่อน EXPAND — ไม่มีข้อยกเว้น
    # ==========================================
    safe_print("\n🚫 Step 8.92: Hard Vehicle Constraint Enforcement...")
    _veh_fix_count = 0
    _veh_rank_map = {'4W': 1, 'JB': 2, '6W': 3}
    _rank_veh_map = {1: '4W', 2: 'JB', 3: '6W'}
    for _hvt in df[df['Trip'] > 0]['Trip'].unique():
        _hvd = df[df['Trip'] == _hvt]
        # หา min rank จาก _vehicle_rank ของทุกสาขาในทริป
        if '_vehicle_rank' in _hvd.columns and not _hvd['_vehicle_rank'].isna().all():
            _hv_min_rank = int(_hvd['_vehicle_rank'].min())
        else:
            _hv_max_vehicles = [branch_max_vehicle_cache.get(str(c).strip().upper(), '6W') for c in _hvd['Code']]
            _hv_min_rank = min(_veh_rank_map.get(v, 3) for v in _hv_max_vehicles)
        _hv_max_allowed = _rank_veh_map.get(_hv_min_rank, '6W')
        # ตรวจ Truck ปัจจุบัน
        _hv_cur_truck = str(_hvd.iloc[0].get('Truck', '6W') or '6W').split()[0]
        _hv_cur_rank = _veh_rank_map.get(_hv_cur_truck, 3)
        if _hv_cur_rank > _hv_min_rank:
            # Truck ปัจจุบันใหญ่เกินข้อจำกัด → บังคับลดให้ตรง
            _hv_new_truck = f"{_hv_max_allowed} 🚫 บังคับ"
            df.loc[df['Trip'] == _hvt, 'Truck'] = _hv_new_truck
            safe_print(f"   🚫 Trip {_hvt}: {_hv_cur_truck} → {_hv_max_allowed} (max allowed by branch constraint)")
            _veh_fix_count += 1
    if _veh_fix_count:
        safe_print(f"   ✅ Step 8.92: บังคับแก้ {_veh_fix_count} ทริป")
    else:
        safe_print(f"   ✅ Step 8.92: ไม่พบ violation")

    # ==========================================
    # Step 8.95: 🔀 EXPAND SHADOW ROWS (suffix pre-merge ↔ expand)
    # คืน shadow rows กลับเข้า df โดยใช้ Trip เดียวกับ primary
    # ต้องทำก่อน Step 9 (renumber) เพื่อให้ summary ถูกต้อง
    # ==========================================
    if _shadow_rows:
        _expand_rows: list = []
        for _pri_code, _shad_list in _shadow_rows.items():
            # หา Trip ของ primary
            _pri_cu = str(_pri_code).strip().upper()
            _pri_match = df[df['Code'].apply(lambda x: str(x).strip().upper()) == _pri_cu]
            if _pri_match.empty:
                continue
            _pri_trip = int(_pri_match.iloc[0]['Trip'])
            _pri_truck = str(_pri_match.iloc[0].get('Truck', '') or '')
            for _shad in _shad_list:
                _shad_row = dict(_shad)
                _shad_row['Trip']  = _pri_trip
                _shad_row['Truck'] = _pri_truck
                _expand_rows.append(_shad_row)
                safe_print(f"   🔀 EXPAND: {_shad_row.get('Code')} → Trip {_pri_trip} (primary={_pri_code})")
        if _expand_rows:
            _expand_df = pd.DataFrame(_expand_rows)
            # ทำให้ column ตรงกัน
            for _ec in df.columns:
                if _ec not in _expand_df.columns:
                    _expand_df[_ec] = None
            _expand_df = _expand_df[df.columns]
            df = pd.concat([df, _expand_df], ignore_index=True)
            safe_print(f"🔀 EXPAND เสร็จ: คืน {len(_expand_rows)} shadow rows → df มี {len(df)} แถว")
            # 🔄 รีเฟรช VehicleCheck หลัง EXPAND เพื่อตรวจ shadow rows ใหม่
            if 'Truck' in df.columns and '_max_vehicle' in df.columns:
                df['VehicleCheck'] = _compute_vehicle_check(df)

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
        # sort key: zone_priority → region_order → -(max_distance)
        # ดึง zone_priority และ region_order จาก dominant branches ของทริป
        _zp9 = 99
        if '_zone_priority' in trip_data.columns:
            _zp9 = int(trip_data['_zone_priority'].mode()[0]) if not trip_data['_zone_priority'].mode().empty else 99
        trip_sort9_keys[trip_num] = (_zp9, _rorder9, -(trip_max_distances[trip_num]))
    
    # เรียงทริปตาม zone_priority → region_order → ระยะทางไกลก่อน
    sorted_trips = sorted(trip_max_distances.keys(), key=lambda x: trip_sort9_keys.get(x, (99, 99, 0)))
    
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
        _cons_buf = punthai_buffer if is_punthai else maxmart_buffer
        
        summary_data_new.append({
            'Trip': trip_num,
            'Branches': len(trip_codes_list),
            'Weight': total_w,
            'Cube': total_c,
            'Truck': truck,
            'BU_Type': 'punthai' if is_punthai else 'maxmart',
            'Buffer': f"🅿️ {int(punthai_buffer*100)}%" if is_punthai else f"🅼 {int(maxmart_buffer*100)}%",
            'Weight_Use%': (total_w / (max_w * _cons_buf)) * 100,
            'Cube_Use%': (total_c / (max_c * _cons_buf)) * 100,
            'Total_Distance': max_dist if pd.notna(max_dist) else 0
        })
    
    summary_df = pd.DataFrame(summary_data_new)
    summary_df = summary_df.sort_values('Trip').reset_index(drop=True)
    
    if sorted_trips:
        safe_print(f"   ✅ เรียงใหม่: {len(sorted_trips)} ทริป (Trip 1 = ไกลสุด {trip_max_distances[sorted_trips[0]]:.0f} km)")
    else:
        safe_print("   ✅ เรียงใหม่: 0 ทริป")
    
    # 📋 เรียงลำดับสาขาภายในทริป: ไกลสุดก่อน แล้ว nearest-neighbor ต่อเนื่อง
    # 1) หาสาขาที่ไกล DC มากสุด → ตั้งต้น
    # 2) จากนั้น nearest-neighbor ไปสาขาที่ใกล้ current position ที่สุดทีละขั้น
    if '_lat' in df.columns and '_lon' in df.columns:
        import math as _math_nn
        def _haversine_nn(lat1, lon1, lat2, lon2):
            if lat2 <= 0 or lon2 <= 0:
                return 9999.0
            _dlat = _math_nn.radians(lat2 - lat1)
            _dlon = _math_nn.radians(lon2 - lon1)
            _a = (_math_nn.sin(_dlat / 2) ** 2 +
                  _math_nn.cos(_math_nn.radians(lat1)) *
                  _math_nn.cos(_math_nn.radians(lat2)) *
                  _math_nn.sin(_dlon / 2) ** 2)
            return 6371.0 * 2 * _math_nn.asin(_math_nn.sqrt(max(0.0, min(1.0, _a))))

        def _nn_order_trip_stops(trip_df):
            """Start from farthest-from-DC branch, then nearest-neighbor chain"""
            if len(trip_df) <= 1:
                return trip_df.index.tolist()
            _lats = trip_df['_lat'].to_numpy(dtype=float)
            _lons = trip_df['_lon'].to_numpy(dtype=float)
            _idxs = trip_df.index.tolist()
            _n = len(_idxs)
            _visited = [False] * _n
            _order = []
            # Step 1: หาสาขาที่ไกล DC มากสุด
            _dc_lat0 = float(DC_WANG_NOI_LAT)
            _dc_lon0 = float(DC_WANG_NOI_LON)
            _far_i = max(range(_n), key=lambda i: _haversine_nn(_dc_lat0, _dc_lon0, _lats[i], _lons[i]))
            _visited[_far_i] = True
            _order.append(_idxs[_far_i])
            _cur_lat, _cur_lon = _lats[_far_i], _lons[_far_i]
            # Step 2: nearest-neighbor chain
            for _ in range(_n - 1):
                _best_i, _best_d = -1, float('inf')
                for _i in range(_n):
                    if _visited[_i]:
                        continue
                    _d = _haversine_nn(_cur_lat, _cur_lon, _lats[_i], _lons[_i])
                    if _d < _best_d:
                        _best_d = _d
                        _best_i = _i
                _visited[_best_i] = True
                _order.append(_idxs[_best_i])
                if _lats[_best_i] > 0 and _lons[_best_i] > 0:
                    _cur_lat, _cur_lon = _lats[_best_i], _lons[_best_i]
            return _order

        _nn_frames = []
        for _nn_t in sorted(df['Trip'].unique()):
            _nn_td = df[df['Trip'] == _nn_t]
            if _nn_t <= 0:
                _nn_frames.append(_nn_td)
            else:
                _nn_ord = _nn_order_trip_stops(_nn_td)
                _nn_frames.append(_nn_td.loc[_nn_ord])
        df = pd.concat(_nn_frames).reset_index(drop=True)
    else:
        df = df.sort_values('Trip', ascending=True).reset_index(drop=True)
    
    # ลบคอลัมน์ชั่วคราว (เก็บ _province, _district, _subdistrict, _max_vehicle, _lat, _lon, _distance_from_dc ไว้สำหรับแผนที่)
    cols_to_drop = ['_region_name', '_route', '_group_key', '_region_order', '_prov_max_dist', '_dist_max_dist', '_subdist_max_dist', '_region_allowed_vehicles', '_vehicle_priority']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # POST-ASSIGNMENT FIX: รวมสาขาที่จุดส่งเดียวกันแต่ถูกแยกทริป
    # นิยาม "กลุ่มรถ" = ตรงเงื่อนไขใดข้อหนึ่ง:
    #   1. พิกัดห่างกัน <= 100m
    #   2. เลขท้ายหลังอักษรเดียวกัน >= 3 หลัก (PT001 + JB001 -> "001")
    #   3. ชื่อสาขาเหมือนกัน (Name column)
    import re as _diag_re
    from collections import Counter as _FixCounter
    _fix_rounds = 0
    _total_fix_merges = 0
    _vrank_fix = {'4W': 1, 'JB': 2, '6W': 3}
    _rvrank_fix = {1: '4W', 2: 'JB', 3: '6W'}
    _SAME_DELIVERY_KM = 0.1  # <= 100m = จุดส่งเดียวกัน

    def _fix_merge_trips(ta, tb, reason_label):
        _la = len(df[df['Trip'] == ta])
        _lb = len(df[df['Trip'] == tb])
        _base_t = ta if _la >= _lb else tb
        _other_t = tb if _base_t == ta else ta
        _all_c = df[df['Trip'].isin([_base_t, _other_t])]['Code'].tolist()
        # 🔒 ตรวจ capacity ก่อนรวม — ห้ามเกินเด็ดขาด
        _fix_w = df[df['Trip'].isin([_base_t, _other_t])]['Weight'].sum()
        _fix_c = df[df['Trip'].isin([_base_t, _other_t])]['Cube'].sum()
        _is_pt_fx = all(branch_bu_cache.get(str(c).strip().upper(), False) for c in _all_c)
        _buf_fx = punthai_buffer if _is_pt_fx else maxmart_buffer
        _lims_fx = PUNTHAI_LIMITS if _is_pt_fx else LIMITS
        _allowed_fx = get_allowed_from_codes(_all_c, ['4W', 'JB', '6W'])
        _fits_fx = any(
            _fix_w <= _lims_fx[v]['max_w'] * _buf_fx and
            _fix_c <= _lims_fx[v]['max_c'] * _buf_fx
            for v in (['4W', 'JB', '6W'] if not _allowed_fx else _allowed_fx)
        )
        if not _fits_fx:
            safe_print(f"   ⚠️ fix-merge skip: trip {ta}+{tb} รวมแล้วเกิน limit ({_fix_w:.0f}kg/{_fix_c:.2f}cbm) → skip")
            return None  # ไม่รวม
        _all_r = [_vrank_fix.get(branch_max_vehicle_cache.get(str(c).strip().upper(), '6W'), 3) for c in _all_c]
        _new_r = min(_all_r) if _all_r else 3
        _new_trk = _rvrank_fix.get(_new_r, '6W')
        df.loc[df['Trip'] == _other_t, 'Trip'] = _base_t
        df.loc[df['Trip'] == _base_t, 'Truck'] = f"{_new_trk} {reason_label}"
        safe_print(f"   {reason_label}: Trip {_other_t} -> {_base_t} truck={_new_trk}")
        return _base_t

    for _fix_iter in range(30):
        _assigned = df[df['Trip'] > 0].copy()
        _merged_this_round = False
        if _assigned.empty:
            break

        # เงื่อนไข 1: <= 500m — 🚀 SPEED: vectorized pairwise แทน O(n²) double iterrows
        if '_lat' in _assigned.columns and '_lon' in _assigned.columns:
            _coords = _assigned[(_assigned['_lat'] > 0) & (_assigned['_lon'] > 0)].copy()
            _coords['_cu'] = _coords['Code'].str.strip().str.upper()
            # สร้าง arrays
            _c_lats = _coords['_lat'].to_numpy(dtype=float)
            _c_lons = _coords['_lon'].to_numpy(dtype=float)
            _c_trips = _coords['Trip'].to_numpy(dtype=int)
            _c_codes = _coords['_cu'].to_numpy()
            _n = len(_c_lats)
            # numpy broadcasting — คำนวณ distance ทุก pair พร้อมกัน
            if _n > 1:
                _R = 6371.0
                _lat_r = np.radians(_c_lats)
                _lon_r = np.radians(_c_lons)
                _dlat = _lat_r[:, np.newaxis] - _lat_r  # (n, n)
                _dlon = _lon_r[:, np.newaxis] - _lon_r
                _a_mx = (np.sin(_dlat / 2) ** 2 +
                         np.cos(_lat_r[:, np.newaxis]) * np.cos(_lat_r) *
                         np.sin(_dlon / 2) ** 2)
                _d_mx = _R * 2 * np.arctan2(np.sqrt(_a_mx), np.sqrt(1 - _a_mx))
                # เช็คเฉพาะ upper-triangle + ต่างทริป + ระยะ <= threshold
                _close = np.argwhere(
                    (_d_mx <= _SAME_DELIVERY_KM) &
                    (np.arange(_n)[:, np.newaxis] < np.arange(_n)) &
                    (_c_trips[:, np.newaxis] != _c_trips)
                )
                if len(_close) > 0:
                    _pi, _pj = _close[0]
                    _d_val = float(_d_mx[_pi, _pj])
                    _fm_r = _fix_merge_trips(int(_c_trips[_pi]), int(_c_trips[_pj]),
                                     f"distance({_d_val*1000:.0f}m)")
                    if _fm_r is not None:
                        _merged_this_round = True
                        _total_fix_merges += 1
        if _merged_this_round:
            _fix_rounds += 1
            continue

        # เงื่อนไข 2: พิกัดห่างกัน ≤ 300 เมตร (แทนตัวเลขท้ายสาขา)
        if '_lat' in df.columns and '_lon' in df.columns:
            _assigned2 = df[df['Trip'] > 0].copy()
            _a2_coords = _assigned2[(_assigned2['_lat'] > 0) & (_assigned2['_lon'] > 0)].copy()
            if len(_a2_coords) > 1:
                _a2_lats = _a2_coords['_lat'].to_numpy(dtype=float)
                _a2_lons = _a2_coords['_lon'].to_numpy(dtype=float)
                _a2_trips = _a2_coords['Trip'].to_numpy(dtype=int)
                _a2_n = len(_a2_lats)
                _a2_lat_r = np.radians(_a2_lats)
                _a2_lon_r = np.radians(_a2_lons)
                _a2_dlat = _a2_lat_r[:, np.newaxis] - _a2_lat_r
                _a2_dlon = _a2_lon_r[:, np.newaxis] - _a2_lon_r
                _a2_a = (np.sin(_a2_dlat / 2) ** 2 +
                         np.cos(_a2_lat_r[:, np.newaxis]) * np.cos(_a2_lat_r) *
                         np.sin(_a2_dlon / 2) ** 2)
                _a2_d = 6371.0 * 2 * np.arctan2(np.sqrt(_a2_a), np.sqrt(1 - _a2_a))
                _a2_close = np.argwhere(
                    (_a2_d <= 0.3) &
                    (np.arange(_a2_n)[:, np.newaxis] < np.arange(_a2_n)) &
                    (_a2_trips[:, np.newaxis] != _a2_trips)
                )
                if len(_a2_close) > 0:
                    _a2_pi, _a2_pj = _a2_close[0]
                    _d2_val = float(_a2_d[_a2_pi, _a2_pj])
                    _fm_r2 = _fix_merge_trips(int(_a2_trips[_a2_pi]), int(_a2_trips[_a2_pj]),
                                     f"coord300m({_d2_val*1000:.0f}m)")
                    if _fm_r2 is not None:
                        _merged_this_round = True
                        _total_fix_merges += 1
        if _merged_this_round:
            _fix_rounds += 1
            continue

        # เงื่อนไข 3: ชื่อสาขาเหมือนกัน
        if 'Name' in df.columns:
            _assigned3 = df[df['Trip'] > 0].copy()
            # 🚀 SPEED: groupby แทน iterrows
            _name_trips: dict = {}
            _a3_names = _assigned3['Name'].fillna('').astype(str).str.strip()
            _a3_trips = _assigned3['Trip'].astype(int)
            for _nm, _t3 in zip(_a3_names, _a3_trips):
                if not _nm:
                    continue
                _name_trips.setdefault(_nm, set()).add(_t3)
            for _nm, _trip_set in _name_trips.items():
                if len(_trip_set) > 1:
                    _sorted_t = sorted(_trip_set, key=lambda t: -len(df[df['Trip'] == t]))
                    _base_t3 = _sorted_t[0]
                    for _other_t3 in _sorted_t[1:]:
                        _fm_r3 = _fix_merge_trips(_base_t3, _other_t3, f"name({_nm[:20]})")
                        if _fm_r3 is not None:
                            _merged_this_round = True
                            _total_fix_merges += 1
                    if _merged_this_round:
                        break

        if not _merged_this_round:
            break
        _fix_rounds += 1

    if _total_fix_merges > 0:
        df['VehicleCheck'] = _compute_vehicle_check(df)
        safe_print(f"\n POST-ASSIGNMENT FIX: {_total_fix_merges} trips merged in {_fix_rounds} rounds")
    else:
        safe_print("POST-ASSIGNMENT FIX: no split same-location branches found")

    # --- diagnostic report ---
    _assigned_final = df[df['Trip'] > 0].copy()
    _split_issues: list = []
    if not _assigned_final.empty:
        # 🚀 SPEED: str vectorized แทน apply(lambda)
        _code_trip_map_f = dict(zip(
            _assigned_final['Code'].str.strip().str.upper(),
            _assigned_final['Trip']
        ))
        _diag_sfx2: dict = {}
        for _dc2 in _code_trip_map_f:
            _ms2 = _diag_re.search(r'\d{3,}$', _dc2)
            if _ms2:
                _diag_sfx2.setdefault(_ms2.group(), []).append(_dc2)
        for _sfx2, _sfx_codes2 in _diag_sfx2.items():
            if len(_sfx_codes2) > 1:
                _trips2 = {c: _code_trip_map_f[c] for c in _sfx_codes2}
                if len(set(_trips2.values())) > 1:
                    _split_issues.append(f"  suffix={_sfx2}: {_trips2}")

    if _split_issues:
        safe_print(f"\n{'='*60}")
        safe_print(f"DIAGNOSTIC (after fix): still {len(_split_issues)} suffix groups split")
        for _si in _split_issues:
            safe_print(_si)
        safe_print(f"{'='*60}\n")
    else:
        safe_print("DIAGNOSTIC: no suffix-split branches remain")

    # ==========================================
    # FLEET CONSOLIDATION: ลดจำนวนทริปต่อประเภทรถให้ไม่เกิน fleet_limits
    # ทำหลัง POST-ASSIGNMENT FIX เพื่อรวมทริปที่ underloaded เข้าด้วยกัน
    # ==========================================
    _fl_real = {k: v for k, v in (fleet_limits or {}).items() if 0 < v < 999}
    if _fl_real:
        safe_print(f"\n🔀 FLEET CONSOLIDATION: fleet_limits={_fl_real}")

        def _trip_natural_vehicle_fc(trip_num):
            """หารถเล็กสุดที่รับโหลดได้ (ไม่นับ fleet limit)"""
            td = df[df['Trip'] == trip_num]
            if td.empty:
                return '6W', 3
            tw = float(td['Weight'].sum())
            tc = float(td['Cube'].sum())
            vr_min = int(td['_vehicle_rank'].min()) if '_vehicle_rank' in td.columns and not td['_vehicle_rank'].isna().all() else 3
            is_pt = all(branch_bu_cache.get(c, False) for c in td['Code'].tolist())
            lims = PUNTHAI_LIMITS if is_pt else LIMITS
            buf = punthai_buffer if is_pt else maxmart_buffer
            for veh_fc in ['4W', 'JB', '6W']:
                vr_fc = vehicle_priority.get(veh_fc, 3)
                if vr_fc > vr_min:
                    break
                lim_fc = lims[veh_fc]
                if tw <= lim_fc['max_w'] * buf and tc <= lim_fc['max_c'] * buf:
                    return veh_fc, vr_fc
            return {1: '4W', 2: 'JB', 3: '6W'}.get(vr_min, '6W'), vr_min

        _fc_all_t = sorted([t for t in df['Trip'].unique() if t > 0])
        _fc_veh_map = {t: _trip_natural_vehicle_fc(t) for t in _fc_all_t}  # t → (veh, vr)

        def _fc_util(trip_num):
            td = df[df['Trip'] == trip_num]
            if td.empty:
                return 0.0
            tw = float(td['Weight'].sum())
            tc = float(td['Cube'].sum())
            veh_fc, _ = _fc_veh_map.get(trip_num, ('6W', 3))
            is_pt_fc = all(branch_bu_cache.get(c, False) for c in td['Code'].tolist())
            lim_fc = (PUNTHAI_LIMITS if is_pt_fc else LIMITS).get(veh_fc, LIMITS['6W'])
            buf_fc = punthai_buffer if is_pt_fc else maxmart_buffer
            return max(tw / (lim_fc['max_w'] * buf_fc + 1e-9), tc / (lim_fc['max_c'] * buf_fc + 1e-9))

        _fc_skip = set()   # trips ที่ถูก absorb แล้ว
        _fc_merged_total = 0

        for _fc_veh in ['4W', 'JB', '6W']:
            if _fc_veh not in _fl_real:
                continue
            _fc_quota = _fl_real[_fc_veh]
            # ทริปที่ควรใช้ vehicle นี้ (ตาม load/constraint)
            _fc_trips_v = [t for t in _fc_all_t if t not in _fc_skip and _fc_veh_map.get(t, ('6W',))[0] == _fc_veh]

            if len(_fc_trips_v) <= _fc_quota:
                safe_print(f"   {_fc_veh}: {len(_fc_trips_v)} trips ≤ quota {_fc_quota} ✓")
                continue

            _fc_excess = len(_fc_trips_v) - _fc_quota
            safe_print(f"   {_fc_veh}: {len(_fc_trips_v)} trips > quota {_fc_quota} → merge {_fc_excess}")

            # เรียง underloaded ก่อน
            _fc_trips_sorted = sorted(_fc_trips_v, key=_fc_util)
            _fc_merged_this = 0

            for _fc_src in _fc_trips_sorted:
                if _fc_merged_this >= _fc_excess:
                    break
                if _fc_src in _fc_skip:
                    continue
                _fc_src_td = df[df['Trip'] == _fc_src]
                if _fc_src_td.empty:
                    continue
                _fc_src_w = float(_fc_src_td['Weight'].sum())
                _fc_src_c = float(_fc_src_td['Cube'].sum())
                _fc_src_vr = int(_fc_src_td['_vehicle_rank'].min()) if '_vehicle_rank' in _fc_src_td.columns and not _fc_src_td['_vehicle_rank'].isna().all() else 3
                _fc_src_reg = str(_fc_src_td['_region_name'].iloc[0]) if '_region_name' in _fc_src_td.columns else ''
                _fc_src_is_pt = all(branch_bu_cache.get(c, False) for c in _fc_src_td['Code'].tolist())

                _fc_best_tgt = None
                _fc_best_util = 999.0

                for _fc_tgt in _fc_trips_v:
                    if _fc_tgt == _fc_src or _fc_tgt in _fc_skip:
                        continue
                    _fc_tgt_td = df[df['Trip'] == _fc_tgt]
                    if _fc_tgt_td.empty:
                        continue
                    _fc_tgt_reg = str(_fc_tgt_td['_region_name'].iloc[0]) if '_region_name' in _fc_tgt_td.columns else ''
                    # ภาคต้องเดียวกัน
                    if (_fc_src_reg and _fc_tgt_reg and
                            _fc_src_reg not in ('', 'ไม่ระบุ') and
                            _fc_tgt_reg not in ('', 'ไม่ระบุ') and
                            _fc_src_reg != _fc_tgt_reg):
                        continue
                    _fc_tgt_is_pt = all(branch_bu_cache.get(c, False) for c in _fc_tgt_td['Code'].tolist())
                    _fc_comb_is_pt = _fc_src_is_pt and _fc_tgt_is_pt
                    _fc_tgt_vr = int(_fc_tgt_td['_vehicle_rank'].min()) if '_vehicle_rank' in _fc_tgt_td.columns and not _fc_tgt_td['_vehicle_rank'].isna().all() else 3
                    _fc_comb_vr = max(_fc_src_vr, _fc_tgt_vr)
                    _fc_comb_veh = {1: '4W', 2: 'JB', 3: '6W'}.get(_fc_comb_vr, '6W')
                    # ห้ามรวมแล้วต้องใช้รถใหญ่กว่า _fc_veh (จะไม่ลด quota)
                    if vehicle_priority.get(_fc_comb_veh, 3) > vehicle_priority.get(_fc_veh, 3):
                        continue
                    _fc_lim = (PUNTHAI_LIMITS if _fc_comb_is_pt else LIMITS).get(_fc_comb_veh, LIMITS['6W'])
                    _fc_buf = punthai_buffer if _fc_comb_is_pt else maxmart_buffer
                    _fc_tgt_w = float(_fc_tgt_td['Weight'].sum())
                    _fc_tgt_c = float(_fc_tgt_td['Cube'].sum())
                    _fc_new_w = _fc_src_w + _fc_tgt_w
                    _fc_new_c = _fc_src_c + _fc_tgt_c
                    if (_fc_new_w <= _fc_lim['max_w'] * _fc_buf and
                            _fc_new_c <= _fc_lim['max_c'] * _fc_buf):
                        _fc_u = max(_fc_new_w / (_fc_lim['max_w'] * _fc_buf + 1e-9),
                                    _fc_new_c / (_fc_lim['max_c'] * _fc_buf + 1e-9))
                        if _fc_u < _fc_best_util:
                            _fc_best_util = _fc_u
                            _fc_best_tgt = _fc_tgt

                if _fc_best_tgt is not None:
                    df.loc[df['Trip'] == _fc_src, 'Trip'] = _fc_best_tgt
                    _fc_skip.add(_fc_src)
                    _fc_merged_this += 1
                    _fc_merged_total += 1
                    safe_print(f"      🔀 {_fc_veh}: Trip {_fc_src} → Trip {_fc_best_tgt} (util {_fc_best_util*100:.0f}%)")
                else:
                    # Try to find a partial merge opportunity - split the source trip and merge part of it
                    _fc_partial_success = False
                    if _fc_src_td.shape[0] > 1:  # Only if there are multiple branches
                        _fc_src_codes = _fc_src_td['Code'].tolist()
                        _fc_src_weights = _fc_src_td['Weight'].tolist()
                        _fc_src_cubes = _fc_src_td['Cube'].tolist()
                        
                        # Try to find a subset of branches that can be merged
                        for i in range(1, len(_fc_src_codes)):
                            for _fc_tgt in _fc_trips_v:
                                if _fc_tgt == _fc_src or _fc_tgt in _fc_skip:
                                    continue
                                _fc_tgt_td = df[df['Trip'] == _fc_tgt]
                                if _fc_tgt_td.empty:
                                    continue
                                _fc_tgt_reg = str(_fc_tgt_td['_region_name'].iloc[0]) if '_region_name' in _fc_tgt_td.columns else ''
                                # ภาคต้องเดียวกัน
                                if (_fc_src_reg and _fc_tgt_reg and
                                        _fc_src_reg not in ('', 'ไม่ระบุ') and
                                        _fc_tgt_reg not in ('', 'ไม่ระบุ') and
                                        _fc_src_reg != _fc_tgt_reg):
                                    continue
                                _fc_tgt_is_pt = all(branch_bu_cache.get(c, False) for c in _fc_tgt_td['Code'].tolist())
                                _fc_comb_is_pt = _fc_src_is_pt and _fc_tgt_is_pt
                                _fc_tgt_vr = int(_fc_tgt_td['_vehicle_rank'].min()) if '_vehicle_rank' in _fc_tgt_td.columns and not _fc_tgt_td['_vehicle_rank'].isna().all() else 3
                                _fc_comb_vr = max(_fc_src_vr, _fc_tgt_vr)
                                _fc_comb_veh = {1: '4W', 2: 'JB', 3: '6W'}.get(_fc_comb_vr, '6W')
                                # ห้ามรวมแล้วต้องใช้รถใหญ่กว่า _fc_veh (จะไม่ลด quota)
                                if vehicle_priority.get(_fc_comb_veh, 3) > vehicle_priority.get(_fc_veh, 3):
                                    continue
                                _fc_lim = (PUNTHAI_LIMITS if _fc_comb_is_pt else LIMITS).get(_fc_comb_veh, LIMITS['6W'])
                                _fc_buf = punthai_buffer if _fc_comb_is_pt else maxmart_buffer
                                _fc_tgt_w = float(_fc_tgt_td['Weight'].sum())
                                _fc_tgt_c = float(_fc_tgt_td['Cube'].sum())
                                
                                # Try each subset of branches from source trip
                                import itertools
                                for subset_size in range(1, min(i, len(_fc_src_codes))):
                                    for subset in itertools.combinations(range(len(_fc_src_codes)), subset_size):
                                        _fc_subset_w = sum(_fc_src_weights[j] for j in subset)
                                        _fc_subset_c = sum(_fc_src_cubes[j] for j in subset)
                                        _fc_new_w = _fc_subset_w + _fc_tgt_w
                                        _fc_new_c = _fc_subset_c + _fc_tgt_c
                                        
                                        if (_fc_new_w <= _fc_lim['max_w'] * _fc_buf and
                                                _fc_new_c <= _fc_lim['max_c'] * _fc_buf):
                                            # Found a partial merge
                                            _fc_u = max(_fc_new_w / (_fc_lim['max_w'] * _fc_buf + 1e-9),
                                                        _fc_new_c / (_fc_lim['max_c'] * _fc_buf + 1e-9))
                                            
                                            # Move the subset of branches to target trip
                                            for j in subset:
                                                _fc_code = _fc_src_codes[j]
                                                df.loc[df['Code'] == _fc_code, 'Trip'] = _fc_tgt
                                            
                                            _fc_partial_success = True
                                            _fc_merged_this += 1
                                            _fc_merged_total += 1
                                            safe_print(f"      🔀 PARTIAL {_fc_veh}: {subset_size} สาขาจาก Trip {_fc_src} → Trip {_fc_tgt} (util {_fc_u*100:.0f}%)")
                                            break
                                    if _fc_partial_success:
                                        break
                                if _fc_partial_success:
                                    break
                    
                    if _fc_partial_success:
                        # Update source trip data after partial merge
                        _fc_src_td = df[df['Trip'] == _fc_src]
                        if not _fc_src_td.empty:
                            _fc_src_w = float(_fc_src_td['Weight'].sum())
                            _fc_src_c = float(_fc_src_td['Cube'].sum())
                    
                    if not _fc_partial_success:
                        safe_print(f"      ⚠️ {_fc_veh}: Trip {_fc_src} ไม่พบทริปรับ (load เต็ม/ต่างภาค)")

            safe_print(f"   {_fc_veh}: merged {_fc_merged_this}/{_fc_excess}")

            # ─── UPGRADE-MERGE PASS: merge คู่ที่รวมกันได้ใน vehicle ใหญ่กว่า ───
            # เช่น 2 JB trips ที่ full → merge เป็น 1 trip ใช้ 6W แทน
            # → ลด JB count 2 เพิ่ม 6W count 1 (net -1 trip)
            _veh_order = ['4W', 'JB', '6W']
            _fc_veh_idx = _veh_order.index(_fc_veh) if _fc_veh in _veh_order else -1
            _fc_trips_after = [t for t in _fc_all_t if t not in _fc_skip and _fc_veh_map.get(t, ('6W',))[0] == _fc_veh]
            _fc_still_excess = len(_fc_trips_after) - _fc_quota
            if _fc_still_excess > 0 and _fc_veh_idx < len(_veh_order) - 1:
                _fc_upgrade_veh = _veh_order[_fc_veh_idx + 1]  # รถที่ใหญ่กว่า (เช่น JB→6W)
                _fc_upgrade_quota = _fl_real.get(_fc_upgrade_veh, 999)
                _fc_upgrade_cur = len([t for t in _fc_all_t if t not in _fc_skip and _fc_veh_map.get(t, ('6W',))[0] == _fc_upgrade_veh])
                _fc_upgrade_room = _fc_upgrade_quota - _fc_upgrade_cur
                if _fc_upgrade_room > 0:
                    safe_print(f"   ↑ upgrade-merge pass: {_fc_veh}→{_fc_upgrade_veh} (quota room={_fc_upgrade_room})")
                    _fc_up_sorted = sorted(_fc_trips_after, key=_fc_util)  # underloaded first
                    _fc_up_merged = 0
                    _fc_up_skip = set()
                    for _fc_up_i, _fc_up_a in enumerate(_fc_up_sorted):
                        if _fc_up_a in _fc_up_skip or _fc_up_a in _fc_skip:
                            continue
                        if _fc_up_merged >= _fc_still_excess:
                            break
                        if _fc_up_merged >= _fc_upgrade_room:
                            break
                        _fc_up_a_td = df[df['Trip'] == _fc_up_a]
                        if _fc_up_a_td.empty:
                            continue
                        _fc_up_a_w = float(_fc_up_a_td['Weight'].sum())
                        _fc_up_a_c = float(_fc_up_a_td['Cube'].sum())
                        _fc_up_a_vr = int(_fc_up_a_td['_vehicle_rank'].min()) if '_vehicle_rank' in _fc_up_a_td.columns and not _fc_up_a_td['_vehicle_rank'].isna().all() else 3
                        _fc_up_a_reg = str(_fc_up_a_td['_region_name'].iloc[0]) if '_region_name' in _fc_up_a_td.columns else ''
                        _fc_up_a_is_pt = all(branch_bu_cache.get(c, False) for c in _fc_up_a_td['Code'].tolist())
                        _fc_up_best_b = None
                        _fc_up_best_u = 999.0
                        for _fc_up_b in _fc_up_sorted[_fc_up_i + 1:]:
                            if _fc_up_b in _fc_up_skip or _fc_up_b in _fc_skip:
                                continue
                            _fc_up_b_td = df[df['Trip'] == _fc_up_b]
                            if _fc_up_b_td.empty:
                                continue
                            _fc_up_b_reg = str(_fc_up_b_td['_region_name'].iloc[0]) if '_region_name' in _fc_up_b_td.columns else ''
                            # ภาคต้องเดียวกัน
                            if (_fc_up_a_reg and _fc_up_b_reg and
                                    _fc_up_a_reg not in ('', 'ไม่ระบุ') and
                                    _fc_up_b_reg not in ('', 'ไม่ระบุ') and
                                    _fc_up_a_reg != _fc_up_b_reg):
                                continue
                            _fc_up_b_vr = int(_fc_up_b_td['_vehicle_rank'].min()) if '_vehicle_rank' in _fc_up_b_td.columns and not _fc_up_b_td['_vehicle_rank'].isna().all() else 3
                            # ตรวจว่า branch constraint ยอม _fc_upgrade_veh ไหม
                            _fc_up_comb_vr = max(_fc_up_a_vr, _fc_up_b_vr)
                            _fc_up_upgrade_vr = vehicle_priority.get(_fc_upgrade_veh, 3)
                            if _fc_up_comb_vr > _fc_up_upgrade_vr:
                                continue  # branch ไม่อนุญาตรถใหญ่
                            _fc_up_b_is_pt = all(branch_bu_cache.get(c, False) for c in _fc_up_b_td['Code'].tolist())
                            _fc_up_comb_is_pt = _fc_up_a_is_pt and _fc_up_b_is_pt
                            _fc_up_lim = (PUNTHAI_LIMITS if _fc_up_comb_is_pt else LIMITS).get(_fc_upgrade_veh, LIMITS['6W'])
                            _fc_up_buf = punthai_buffer if _fc_up_comb_is_pt else maxmart_buffer
                            _fc_up_b_w = float(_fc_up_b_td['Weight'].sum())
                            _fc_up_b_c = float(_fc_up_b_td['Cube'].sum())
                            _fc_up_new_w = _fc_up_a_w + _fc_up_b_w
                            _fc_up_new_c = _fc_up_a_c + _fc_up_b_c
                            if (_fc_up_new_w <= _fc_up_lim['max_w'] * _fc_up_buf and
                                    _fc_up_new_c <= _fc_up_lim['max_c'] * _fc_up_buf):
                                _fc_up_u = max(_fc_up_new_w / (_fc_up_lim['max_w'] * _fc_up_buf + 1e-9),
                                               _fc_up_new_c / (_fc_up_lim['max_c'] * _fc_up_buf + 1e-9))
                                if _fc_up_u < _fc_up_best_u:
                                    _fc_up_best_u = _fc_up_u
                                    _fc_up_best_b = _fc_up_b
                        if _fc_up_best_b is not None:
                            # merge _fc_up_a → _fc_up_best_b และเปลี่ยน vehicle เป็น upgrade
                            df.loc[df['Trip'] == _fc_up_a, 'Trip'] = _fc_up_best_b
                            df.loc[df['Trip'] == _fc_up_best_b, 'Truck'] = f"{_fc_upgrade_veh} ↑upgrade-merge"
                            _fc_skip.add(_fc_up_a)
                            _fc_up_skip.add(_fc_up_a)
                            _fc_up_skip.add(_fc_up_best_b)
                            _fc_up_merged += 1
                            _fc_merged_total += 1
                            safe_print(f"      ↑ {_fc_veh}→{_fc_upgrade_veh}: Trip {_fc_up_a} → {_fc_up_best_b} (util {_fc_up_best_u*100:.0f}%)")
                        else:
                            # Try partial merge for upgrade pass as well
                            _fc_up_partial_success = False
                            if _fc_up_a_td.shape[0] > 1:  # Only if there are multiple branches
                                _fc_up_a_codes = _fc_up_a_td['Code'].tolist()
                                _fc_up_a_weights = _fc_up_a_td['Weight'].tolist()
                                _fc_up_a_cubes = _fc_up_a_td['Cube'].tolist()
                                
                                # Try to find a subset of branches that can be merged with another trip
                                for _fc_up_b in _fc_up_sorted[_fc_up_i + 1:]:
                                    if _fc_up_b in _fc_up_skip or _fc_up_b in _fc_skip:
                                        continue
                                    _fc_up_b_td = df[df['Trip'] == _fc_up_b]
                                    if _fc_up_b_td.empty:
                                        continue
                                    _fc_up_b_reg = str(_fc_up_b_td['_region_name'].iloc[0]) if '_region_name' in _fc_up_b_td.columns else ''
                                    # ภาคต้องเดียวกัน
                                    if (_fc_up_a_reg and _fc_up_b_reg and
                                            _fc_up_a_reg not in ('', 'ไม่ระบุ') and
                                            _fc_up_b_reg not in ('', 'ไม่ระบุ') and
                                            _fc_up_a_reg != _fc_up_b_reg):
                                        continue
                                    _fc_up_b_vr = int(_fc_up_b_td['_vehicle_rank'].min()) if '_vehicle_rank' in _fc_up_b_td.columns and not _fc_up_b_td['_vehicle_rank'].isna().all() else 3
                                    # ตรวจว่า branch constraint ยอม _fc_upgrade_veh ไหม
                                    _fc_up_comb_vr = max(_fc_up_a_vr, _fc_up_b_vr)
                                    _fc_up_upgrade_vr = vehicle_priority.get(_fc_upgrade_veh, 3)
                                    if _fc_up_comb_vr > _fc_up_upgrade_vr:
                                        continue  # branch ไม่อนุญาตรถใหญ่
                                    _fc_up_b_is_pt = all(branch_bu_cache.get(c, False) for c in _fc_up_b_td['Code'].tolist())
                                    _fc_up_comb_is_pt = _fc_up_a_is_pt and _fc_up_b_is_pt
                                    _fc_up_lim = (PUNTHAI_LIMITS if _fc_up_comb_is_pt else LIMITS).get(_fc_upgrade_veh, LIMITS['6W'])
                                    _fc_up_buf = punthai_buffer if _fc_up_comb_is_pt else maxmart_buffer
                                    _fc_up_b_w = float(_fc_up_b_td['Weight'].sum())
                                    _fc_up_b_c = float(_fc_up_b_td['Cube'].sum())
                                    
                                    # Try each subset of branches from trip A
                                    import itertools
                                    for subset_size in range(1, len(_fc_up_a_codes)):
                                        for subset in itertools.combinations(range(len(_fc_up_a_codes)), subset_size):
                                            _fc_subset_w = sum(_fc_up_a_weights[j] for j in subset)
                                            _fc_subset_c = sum(_fc_up_a_cubes[j] for j in subset)
                                            _fc_new_w = _fc_subset_w + _fc_up_b_w
                                            _fc_new_c = _fc_subset_c + _fc_up_b_c
                                            
                                            if (_fc_new_w <= _fc_up_lim['max_w'] * _fc_up_buf and
                                                    _fc_new_c <= _fc_up_lim['max_c'] * _fc_up_buf):
                                                # Found a partial merge for upgrade
                                                _fc_up_u = max(_fc_new_w / (_fc_up_lim['max_w'] * _fc_up_buf + 1e-9),
                                                               _fc_new_c / (_fc_up_lim['max_c'] * _fc_up_buf + 1e-9))
                                                
                                                # Move the subset of branches to target trip
                                                for j in subset:
                                                    _fc_code = _fc_up_a_codes[j]
                                                    df.loc[df['Code'] == _fc_code, 'Trip'] = _fc_up_b
                                                
                                                # Update target trip vehicle
                                                df.loc[df['Trip'] == _fc_up_b, 'Truck'] = f"{_fc_upgrade_veh} ↑partial-upgrade"
                                                
                                                _fc_up_partial_success = True
                                                _fc_up_merged += 1
                                                _fc_merged_total += 1
                                                safe_print(f"      ↑ PARTIAL {_fc_veh}→{_fc_upgrade_veh}: {subset_size} สาขาจาก Trip {_fc_up_a} → Trip {_fc_up_b} (util {_fc_up_u*100:.0f}%)")
                                                break
                                        if _fc_up_partial_success:
                                            break
                                    if _fc_up_partial_success:
                                        break
                            
                            if _fc_up_partial_success:
                                # Update trip A data after partial merge
                                _fc_up_a_td = df[df['Trip'] == _fc_up_a]
                                if not _fc_up_a_td.empty:
                                    _fc_up_a_w = float(_fc_up_a_td['Weight'].sum())
                                    _fc_up_a_c = float(_fc_up_a_td['Cube'].sum())
                            
                            if not _fc_up_partial_success:
                                safe_print(f"      ↑ skip Trip {_fc_up_a}: ไม่พบคู่ที่พอดี {_fc_upgrade_veh}")
                    if _fc_up_merged > 0:
                        # refresh _fc_veh_map สำหรับ trips ที่ถูก upgrade
                        for _t_up in df['Trip'].unique():
                            if _t_up > 0 and _t_up not in _fc_veh_map:
                                _fc_veh_map[_t_up] = _trip_natural_vehicle_fc(_t_up)
                        safe_print(f"   ↑ upgrade-merge: {_fc_up_merged} trips {_fc_veh}→{_fc_upgrade_veh}")


        _fc_remaining = [t for t in _fc_all_t if t not in _fc_skip]
        safe_print(f"   Fleet consolidation done: merged {_fc_merged_total}, remaining {len(_fc_remaining)} trips")
        # Rebuild VehicleCheck หลัง consolidation
        df['VehicleCheck'] = _compute_vehicle_check(df)
    else:
        safe_print("ℹ️ Fleet consolidation: ไม่มี fleet_limits จริง → ข้าม")

    # ==========================================
    # FINAL SUMMARY REBUILD: สร้าง summary_df ใหม่จาก df ล่าสุด
    # (หลัง POST-ASSIGNMENT FIX ที่อาจ merge trips เปลี่ยนน้ำหนัก/จำนวนสาขา)
    # ==========================================
    safe_print("\n📊 Final summary rebuild...")
    _final_summary_data = []
    for _fs_trip in sorted(df[df['Trip'] > 0]['Trip'].unique()):
        _fs_td = df[df['Trip'] == _fs_trip]
        _fs_w = float(_fs_td['Weight'].sum())
        _fs_c = float(_fs_td['Cube'].sum())
        _fs_codes = _fs_td['Code'].tolist()
        _fs_max_dist = _fs_td['_distance_from_dc'].max() if '_distance_from_dc' in _fs_td.columns else 0
        _fs_truck = str(_fs_td['Truck'].iloc[0]) if 'Truck' in _fs_td.columns and len(_fs_td) > 0 else '6W'
        _fs_truck_str = _fs_truck.split()[0] if pd.notna(_fs_truck) else '6W'
        _fs_is_pt = all(branch_bu_cache.get(c, False) for c in _fs_codes)
        _fs_limits = PUNTHAI_LIMITS if _fs_is_pt else LIMITS
        _fs_buf = punthai_buffer if _fs_is_pt else maxmart_buffer
        _fs_max_w = _fs_limits.get(_fs_truck_str, _fs_limits['6W'])['max_w']
        _fs_max_c = _fs_limits.get(_fs_truck_str, _fs_limits['6W'])['max_c']
        _final_summary_data.append({
            'Trip': _fs_trip,
            'Branches': len(_fs_codes),
            'Weight': _fs_w,
            'Cube': _fs_c,
            'Truck': _fs_truck,
            'BU_Type': 'punthai' if _fs_is_pt else 'maxmart',
            'Buffer': f"🅿️ {int(punthai_buffer*100)}%" if _fs_is_pt else f"🅼 {int(maxmart_buffer*100)}%",
            'Weight_Use%': (_fs_w / (_fs_max_w * _fs_buf)) * 100 if _fs_max_w > 0 else 0,
            'Cube_Use%': (_fs_c / (_fs_max_c * _fs_buf)) * 100 if _fs_max_c > 0 else 0,
            'Total_Distance': float(_fs_max_dist) if pd.notna(_fs_max_dist) else 0,
        })
    if _final_summary_data:
        summary_df = pd.DataFrame(_final_summary_data).sort_values('Trip').reset_index(drop=True)
        safe_print(f"   ✅ Final summary: {len(summary_df)} trips rebuilt")
    else:
        safe_print("   ⚠️ Final summary: 0 trips")

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
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ═══════════════════════════════════════════════════════════════
   RESET & BASE — FORCE LIGHT MODE
═══════════════════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }
html, body {
    font-family: 'Sarabun', 'Inter', system-ui, sans-serif !important;
    background: #f0faf4 !important;
    color: #0f172a !important;
    font-size: 16px !important;
}
[class*="css"] {
    font-family: 'Sarabun', 'Inter', system-ui, sans-serif !important;
}
/* Nuke every possible dark container */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stBottomBlockContainer"],
section[tabindex="0"],
.main, section.main, .block-container,
div[data-layout="wide"] {
    background-color: #f0faf4 !important;
    color: #0f172a !important;
}
.stApp {
    background: #f0faf4 !important;
    background-image:
        radial-gradient(ellipse 70% 40% at 50% -5%, rgba(16,185,129,0.10), transparent),
        radial-gradient(ellipse 50% 35% at 90% 70%, rgba(6,182,212,0.05), transparent) !important;
}
.main .block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
    padding-bottom: 4rem !important;
    max-width: 1480px !important;
    background-color: transparent !important;
}

/* ═══════════════════════════════════════════════════════════════
   TOP NAV BAR
═══════════════════════════════════════════════════════════════ */
.app-navbar {
    background: #ffffff;
    border-bottom: 3px solid #10b981;
    padding: 0 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 64px;
    margin: -1rem -1rem 0 -1rem;
    position: sticky;
    top: 0;
    z-index: 999;
    box-shadow: 0 2px 16px rgba(16,185,129,0.10);
}
.app-navbar-brand {
    display: flex; align-items: center; gap: 14px;
    color: #064e3b !important;
    font-size: 1.15rem;
    font-weight: 800;
    letter-spacing: -0.3px;
    text-decoration: none;
}
.app-navbar-brand .brand-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem;
    box-shadow: 0 4px 12px rgba(16,185,129,0.3);
    flex-shrink: 0;
}
.app-navbar-brand .brand-name { line-height: 1.1; }
.app-navbar-brand .brand-name small { display: block; font-size: 0.65rem; font-weight: 500; color: #6b7280; letter-spacing: 0.5px; }
.app-navbar-status {
    display: flex; align-items: center; gap: 10px;
    font-size: 0.8rem; color: #047857;
    font-weight: 500;
}
.nav-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    padding: 4px 12px; border-radius: 20px;
    font-size: 0.76rem; font-weight: 600;
    color: #065f46; white-space: nowrap;
}
.nav-sync-btn {
    display: inline-flex; align-items: center; gap: 6px;
    background: #059669; color: #ffffff !important;
    border: none; border-radius: 20px;
    padding: 6px 16px; font-size: 0.78rem; font-weight: 700;
    cursor: pointer; transition: background .2s, box-shadow .2s;
    box-shadow: 0 2px 8px rgba(5,150,105,0.3);
    text-decoration: none;
}
.nav-sync-btn:hover { background: #047857; box-shadow: 0 4px 12px rgba(5,150,105,0.4); }
/* push stButton inside navbar area */
.navbar-sync-wrapper { position: relative; margin: -64px 0 0 0; height: 64px; display: flex; align-items: center; justify-content: flex-end; padding-right: 2rem; pointer-events: none; }
.navbar-sync-wrapper > * { pointer-events: all; }
.nav-dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; display: inline-block; box-shadow: 0 0 6px rgba(74,222,128,0.9); animation: pulse-dot 2s infinite; flex-shrink: 0; }
.nav-dot.offline { background: #fbbf24; box-shadow: 0 0 6px rgba(251,191,36,0.8); animation: none; }
.nav-badge-warn { background: #fffbeb !important; border-color: #fde68a !important; color: #92400e !important; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
/* ── Page title ── */
.page-section-title {
    font-size: 1.4rem; font-weight: 800; color: #064e3b;
    margin: 1.2rem 0 0.8rem;
    display: flex; align-items: center; gap: 10px;
}
.page-section-title small { font-size: 0.82rem; font-weight: 500; color: #6b7280; }
/* ── KPI cards ── */
.kpi-card {
    background: #ffffff;
    border: 1.5px solid #d1fae5;
    border-radius: 16px;
    padding: 1rem 1.25rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(5,150,105,0.06);
    transition: border-color .2s, box-shadow .2s;
}
.kpi-card:hover { border-color: #6ee7b7; box-shadow: 0 4px 16px rgba(16,185,129,0.12); }
.kpi-value { font-size: 1.55rem; font-weight: 800; color: #059669; line-height: 1.1; }
.kpi-label { font-size: 0.72rem; color: #6b7280; font-weight: 500; margin-top: 4px; }

/* ═══════════════════════════════════════════════════════════════
   SECTION CARD & DIVIDER
═══════════════════════════════════════════════════════════════ */
.glass-card {
    background: #ffffff;
    border: 1.5px solid #d1fae5;
    border-radius: 20px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 2px 8px rgba(5,150,105,0.06), 0 1px 3px rgba(15,23,42,0.05);
    transition: border-color .2s, box-shadow .2s;
}
.glass-card:hover {
    border-color: #6ee7b7;
    box-shadow: 0 6px 24px rgba(16,185,129,0.12);
}
.divider-label {
    display: flex; align-items: center; gap: 12px;
    color: #059669; font-size: 0.78rem; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin: 2rem 0 1.25rem 0;
}
.divider-label::before, .divider-label::after {
    content: ''; flex: 1; height: 1.5px;
    background: linear-gradient(90deg, #d1fae5, transparent);
}
.divider-label::before { background: linear-gradient(90deg, transparent, #d1fae5); }

/* ═══════════════════════════════════════════════════════════════
   TYPOGRAPHY
═══════════════════════════════════════════════════════════════ */
h1 { color: #064e3b !important; font-weight: 900 !important; font-size: 2rem !important; }
h2 { color: #065f46 !important; font-weight: 800 !important; font-size: 1.5rem !important; }
h3 { color: #047857 !important; font-weight: 700 !important; font-size: 1.2rem !important; }
h4 { color: #0f172a !important; font-weight: 700 !important; font-size: 1.05rem !important; }
h5, h6 { color: #1e293b !important; font-weight: 600 !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #064e3b !important;
}
[data-testid="stMarkdownContainer"] p {
    color: #1e293b !important;
    font-size: 0.95rem !important;
    line-height: 1.65 !important;
}
[data-testid="stMarkdownContainer"] li {
    color: #374151 !important;
    font-size: 0.95rem !important;
}
/* All text elements */
label, span, div, p, td, th, li {
    color: #0f172a;
}

/* ═══════════════════════════════════════════════════════════════
   METRIC CARDS
═══════════════════════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1.5px solid #d1fae5 !important;
    border-radius: 18px !important;
    padding: 20px 24px !important;
    box-shadow: 0 2px 8px rgba(5,150,105,0.07) !important;
    transition: all .22s cubic-bezier(.4,0,.2,1) !important;
    position: relative !important;
    overflow: hidden !important;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, #059669, #10b981, #34d399);
    border-radius: 18px 18px 0 0;
}
[data-testid="metric-container"]:hover {
    border-color: #6ee7b7 !important;
    box-shadow: 0 8px 28px rgba(16,185,129,0.15) !important;
    transform: translateY(-3px) !important;
}
[data-testid="stMetricLabel"] {
    color: #059669 !important;
    font-size: 0.78rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] {
    color: #064e3b !important;
    font-weight: 900 !important;
    font-size: 2rem !important;
    letter-spacing: -1px !important;
}
[data-testid="stMetricDelta"] { font-weight: 700 !important; font-size: 0.85rem !important; }

/* ═══════════════════════════════════════════════════════════════
   BUTTONS
═══════════════════════════════════════════════════════════════ */
.stButton > button {
    background: #ffffff !important;
    color: #374151 !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-family: inherit !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.3rem !important;
    transition: all .18s cubic-bezier(.4,0,.2,1) !important;
    box-shadow: 0 1px 3px rgba(15,23,42,0.08) !important;
}
.stButton > button:hover {
    background: #f0fdf4 !important;
    border-color: #10b981 !important;
    color: #065f46 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(16,185,129,0.15) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    padding: 0.7rem 1.75rem !important;
    box-shadow: 0 4px 20px rgba(5,150,105,0.35) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
    box-shadow: 0 8px 28px rgba(5,150,105,0.45) !important;
    transform: translateY(-2px) !important;
}

/* ═══════════════════════════════════════════════════════════════
   TABS
═══════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 3px;
    background: #ecfdf5;
    border-radius: 14px;
    padding: 5px;
    border: 1.5px solid #d1fae5;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    font-weight: 700 !important;
    color: #6b7280 !important;
    padding: 10px 26px !important;
    font-size: 0.92rem !important;
    transition: all .18s !important;
    border: none !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 12px rgba(5,150,105,0.35) !important;
    font-weight: 800 !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
[data-baseweb="tab-border"] { display: none !important; }

/* ═══════════════════════════════════════════════════════════════
   EXPANDERS
═══════════════════════════════════════════════════════════════ */
details[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 14px !important;
    margin-bottom: 10px;
    box-shadow: 0 1px 4px rgba(15,23,42,0.05);
    transition: all .2s;
}
details[data-testid="stExpander"][open] {
    border-color: #6ee7b7 !important;
    box-shadow: 0 3px 14px rgba(16,185,129,0.10) !important;
}
details[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #1e293b !important;
    padding: 15px 20px !important;
    border-radius: 14px !important;
    font-size: 0.95rem !important;
    cursor: pointer;
}
details[data-testid="stExpander"] summary:hover { color: #059669 !important; }

/* ═══════════════════════════════════════════════════════════════
   HERO UPLOAD CARD
═══════════════════════════════════════════════════════════════ */
.hero-upload-card {
    background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 60%, #ffffff 100%);
    border: 2px solid #a7f3d0;
    border-radius: 24px;
    padding: 2rem 2.5rem 1.5rem;
    margin: 1rem 0 1.5rem;
    box-shadow: 0 4px 24px rgba(16,185,129,0.08);
}
.hero-upload-title {
    font-size: 1.3rem; font-weight: 800; color: #064e3b;
    margin-bottom: 0.25rem;
}
.hero-upload-sub {
    font-size: 0.88rem; color: #6b7280; margin-bottom: 1.2rem;
}
/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 2px dashed #6ee7b7 !important;
    border-radius: 16px !important;
    padding: 1.2rem !important;
    transition: all .25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #10b981 !important;
    background: #f0fdf4 !important;
    box-shadow: 0 0 0 4px rgba(16,185,129,0.1) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: #6b7280 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] small { color: #9ca3af !important; }

/* ═══════════════════════════════════════════════════════════════
   INPUTS
═══════════════════════════════════════════════════════════════ */
[data-testid="stNumberInput"] > div,
[data-testid="stTextInput"] > div > div {
    background: #ffffff !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 12px !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: transparent !important;
    border: none !important;
    color: #0f172a !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    caret-color: #10b981;
}
[data-testid="stNumberInput"] > div:focus-within,
[data-testid="stTextInput"] > div > div:focus-within {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.12) !important;
    background: #f0fdf4 !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 12px !important;
    color: #0f172a !important;
}
/* Input labels */
[data-testid="stNumberInput"] label,
[data-testid="stTextInput"] label,
[data-testid="stSelectbox"] label,
.stSlider label {
    color: #059669 !important;
    font-size: 0.82rem !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* ═══════════════════════════════════════════════════════════════
   DATAFRAME
═══════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 14px !important;
    overflow: hidden !important;
    border: 1.5px solid #d1fae5 !important;
    box-shadow: 0 2px 8px rgba(5,150,105,0.06) !important;
}
[data-testid="stDataFrame"] thead th {
    background: #ecfdf5 !important;
    color: #065f46 !important;
    font-weight: 800 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    border-bottom: 1.5px solid #a7f3d0 !important;
}
[data-testid="stDataFrame"] tbody tr:hover td { background: #f0fdf4 !important; }

/* ═══════════════════════════════════════════════════════════════
   ALERTS — LIGHT MODE COLORS
═══════════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}
[data-testid="stAlert"][data-type="success"],
div.stAlert[data-baseweb="notification"].st-success {
    background: #f0fdf4 !important;
    border: 1.5px solid #a7f3d0 !important;
    border-left: 4px solid #059669 !important;
    color: #065f46 !important;
}
[data-testid="stAlert"][data-type="warning"] {
    background: #fffbeb !important;
    border: 1.5px solid #fde68a !important;
    border-left: 4px solid #d97706 !important;
    color: #92400e !important;
}
[data-testid="stAlert"][data-type="error"] {
    background: #fef2f2 !important;
    border: 1.5px solid #fecaca !important;
    border-left: 4px solid #ef4444 !important;
    color: #991b1b !important;
}
[data-testid="stAlert"][data-type="info"] {
    background: #eff6ff !important;
    border: 1.5px solid #bfdbfe !important;
    border-left: 4px solid #3b82f6 !important;
    color: #1e40af !important;
}

/* ═══════════════════════════════════════════════════════════════
   DOWNLOAD BUTTON
═══════════════════════════════════════════════════════════════ */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 800 !important;
    font-size: 0.95rem !important;
    border-radius: 12px !important;
    box-shadow: 0 3px 10px rgba(5,150,105,0.28) !important;
}
[data-testid="stDownloadButton"] > button:hover {
    filter: brightness(1.07) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 22px rgba(5,150,105,0.38) !important;
}

/* ═══════════════════════════════════════════════════════════════
   PROGRESS & SPINNER
═══════════════════════════════════════════════════════════════ */
[data-testid="stProgress"] > div {
    background: #d1fae5 !important;
    border-radius: 99px !important;
    overflow: hidden;
}
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #059669, #10b981, #34d399) !important;
    border-radius: 99px !important;
    box-shadow: 0 0 8px rgba(16,185,129,0.4);
}
[data-testid="stSpinner"] > div { border-top-color: #10b981 !important; }

/* ═══════════════════════════════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f0fdf4; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #a7f3d0; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6ee7b7; }

/* ═══════════════════════════════════════════════════════════════
   MISC
═══════════════════════════════════════════════════════════════ */
[data-testid="stCheckbox"] label { font-weight: 700 !important; color: #1e293b !important; font-size: 0.95rem !important; }
.stCaption { color: #6b7280 !important; font-size: 0.78rem !important; }
hr { border: none !important; border-top: 1.5px solid #d1fae5 !important; margin: 1.5rem 0 !important; }
[data-testid="stRadio"] label { font-weight: 600 !important; color: #1e293b !important; font-size: 0.95rem !important; }

/* ═══════════════════════════════════════════════════════════════
   HIDE STREAMLIT BRANDING
═══════════════════════════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stAppViewContainer"] > section > div:first-child { padding-top: 0 !important; }
.stDeployButton { display: none; }
</style>
""", unsafe_allow_html=True)


    # ป้องกันการโหลดโซนซ้ำ - ใช้ session state
    if 'zones_loaded' not in st.session_state:
        st.session_state.zones_loaded = False
    if 'cache_stats_shown' not in st.session_state:
        st.session_state.cache_stats_shown = False
    
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
    
    # ── Top Navigation Bar ──────────────────────────────────────────────────
    # check precompute status
    _precompute_running = False
    if st.session_state.get('_precompute_pid'):
        import subprocess as _sp2
        _pid = st.session_state['_precompute_pid']
        try:
            if os.name == 'nt':
                _r = _sp2.run(['tasklist', '/FI', f'PID eq {_pid}', '/NH', '/FO', 'CSV'],
                              capture_output=True, text=True)
                _precompute_running = str(_pid) in _r.stdout
            else:
                os.kill(_pid, 0)
                _precompute_running = True
        except Exception:
            _precompute_running = False
        if not _precompute_running:
            st.session_state.pop('_precompute_pid', None)

    # ── Top Navbar ──────────────────────────────────────────────────────────
    _sheets_ok    = SHEETS_AVAILABLE
    _dot_color    = '#4ade80' if _sheets_ok else '#fbbf24'
    _sheets_label = 'Google Sheets' if _sheets_ok else 'ออฟไลน์'
    _sheets_sub   = 'เชื่อมต่อแล้ว' if _sheets_ok else 'ออฟไลน์'
    _branch_count = len(MASTER_DATA) if not MASTER_DATA.empty else 0
    _today_str    = datetime.now().strftime('%d %b %Y')
    _rebuild_badge = '<span class="nav-badge nav-badge-warn">⏳ กำลัง Rebuild...</span>' if _precompute_running else ''
    _dot_span = f'<span style="width:8px;height:8px;border-radius:50%;background:{_dot_color};display:inline-block;flex-shrink:0;margin-right:4px"></span>'
    _badge1 = f'<span class="nav-badge">{_dot_span} {_sheets_label}: {_sheets_sub}</span>'
    _badge2 = f'<span class="nav-badge">📍 {_branch_count:,} สาขา</span>'
    _navbar_html = f'<div class="app-navbar"><div class="app-navbar-brand"><div class="brand-icon">🚚</div><div class="brand-name">Route <strong>Optimizer</strong><small>ระบบจัดเที่ยวอัจฉริยะ</small></div></div><div class="app-navbar-status">{_rebuild_badge}{_badge1}{_badge2}</div></div>'
    st.markdown(_navbar_html, unsafe_allow_html=True)

    # ── Page title ─────────────────────────────────────────────────────────
    _cache_dist = len(DISTANCE_CACHE)
    _cache_rt   = len(ROUTE_CACHE_DATA)
    st.markdown(f'<div class="page-section-title"><span>📦</span> จัดเส้นทางจัดส่ง <small>· {_today_str}</small></div>', unsafe_allow_html=True)

    # ── KPI row: 3 stats + sync button ──────────────────────────────────────
    _kc1, _kc2, _kc3, _kc4 = st.columns(4)
    with _kc1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{_branch_count:,}</div><div class="kpi-label">🏪 สาขาในระบบ</div></div>', unsafe_allow_html=True)
    with _kc2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{_cache_dist:,}</div><div class="kpi-label">📍 ระยะทางแคช</div></div>', unsafe_allow_html=True)
    with _kc3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{_cache_rt:,}</div><div class="kpi-label">🛣️ เส้นทางแคช</div></div>', unsafe_allow_html=True)
    with _kc4:
        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
        if st.button("🔄 ซิงค์ข้อมูล", use_container_width=True, type="primary",
                     help="ดึงข้อมูลจาก Google Sheets + rebuild distances"):
            with st.spinner("⏳ กำลังดึงข้อมูล..."):
                try:
                    st.cache_data.clear()
                    for _k in ['trip_result', 'trip_summary', '_imap_html', '_imap_key',
                                'trip_result_excel', '_imap_build_time']:
                        st.session_state.pop(_k, None)
                    import subprocess as _sp, sys as _sys
                    _precompute_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'precompute_branch_data.py')
                    if os.path.exists(_precompute_script):
                        _proc = _sp.Popen(
                            [_sys.executable, _precompute_script],
                            cwd=os.path.dirname(os.path.abspath(__file__)),
                            stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                            creationflags=_sp.CREATE_NO_WINDOW if os.name == 'nt' else 0
                        )
                        st.session_state['_precompute_pid'] = _proc.pid
                except Exception as _re:
                    st.error(f"❌ {_re}")
                finally:
                    st.rerun()
        if _precompute_running:
            st.caption("⏳ กำลัง rebuild...")

    # โหลดโมเดล
    model_data = load_model()
    if not model_data:
        st.error("❌ ไม่พบข้อมูลโมเดล กรุณาเทรนโมเดลก่อนใช้งาน")
        st.stop()

    # ── Upload card ─────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero-upload-card">
  <div class="hero-upload-title">📂 นำเข้าไฟล์ออเดอร์</div>
  <div class="hero-upload-sub">รองรับไฟล์ Excel (.xlsx) ที่มีรายการสาขาและน้ำหนัก / คิว</div>
</div>""", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "upload", type=['xlsx'],
        help="อัปโหลดไฟล์ Excel ที่มีรายการสาขาและออเดอร์",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        # เก็บไฟล์ต้นฉบับไว้ใน session_state เพื่อใช้ตอน export
        uploaded_file_content = uploaded_file.read()
        # เคลียร์ผลลัพธ์เก่าถ้าไฟล์เปลี่ยน
        _prev_file_id = st.session_state.get('_uploaded_file_id')
        _curr_file_id = (uploaded_file.name, uploaded_file.size)
        if _prev_file_id != _curr_file_id:
            for _k in ('trip_result', 'trip_summary', 'fleet_used', 'fleet_limits', 'trip_buffers', '_trip_result_fresh', '_trip_elapsed'):
                st.session_state.pop(_k, None)
            st.session_state['_uploaded_file_id'] = _curr_file_id
        st.session_state['original_file_content'] = uploaded_file_content
        # เคลียร์ log buffer ก่อนโหลด
        st.session_state['_ui_log'] = []
        
        with st.spinner("⏳ กำลังอ่านข้อมูล..."):
            df = load_excel(uploaded_file_content)
            # ดึงชื่อและสีหัวคอลัมน์จากไฟล์ต้นฉบับก่อน rename
            _orig_hdr = _extract_header_info(uploaded_file_content)
            if _orig_hdr:
                st.session_state['_orig_headers'] = _orig_hdr
            _orig_style = _extract_style_info(uploaded_file_content)
            st.session_state['_orig_style_info'] = _orig_style
            _orig_dc_raw = _extract_dc_row_info(uploaded_file_content)
            if _orig_dc_raw:
                st.session_state['_orig_dc_row_raw'] = _orig_dc_raw
            df = process_dataframe(df)
        
        # ►►► แสดง log หลังโหลด
        _logs = st.session_state.get('_ui_log', [])
        if _logs:
            with st.expander("📝 Log การโหลดไฟล์", expanded=False):
                st.code('\n'.join(_logs), language=None)
        
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
            
            st.markdown('<div class="divider-label">⚙️ การจัดการ</div>', unsafe_allow_html=True)

            # แท็บหลัก
            tab1, tab2, tab3 = st.tabs([
                "📦 จัดเที่ยว",
                "🗺️ จัดกลุ่มตามภาค",
                "🏙️ โซนจัดส่ง"
            ])
                
            # ==========================================
            # แท็บ 1: จัดเที่ยว (ตามน้ำหนัก)
            # ==========================================
            with tab1:
                # เพิ่ม Region ถ้ายังไม่มี
                if 'Region' not in df.columns and 'Province' in df.columns:
                    df['Region'] = df['Province'].apply(get_region_name)
                
                # ==========================================
                # ตัวเลือกการตั้งค่า — ใช้ st.form กัน re-run เมื่อเปลี่ยนค่า input
                # ==========================================
                with st.form(key="_trip_settings_form"):
                    st.markdown('<div class="divider-label">⚙️ ตั้งค่าการจัดทริป</div>', unsafe_allow_html=True)
                    # กรอก Buffer แยกตามประเภท
                    col_buf1, col_buf2 = st.columns(2)
                    with col_buf1:
                        punthai_buffer = st.number_input(
                            "🅿️ Punthai Buffer %",
                            min_value=80,
                            max_value=120,
                            value=int(st.session_state.get('_punthai_buf_pct', 100)),
                            step=5,
                            key="punthai_buffer_input",
                        )
                    with col_buf2:
                        maxmart_buffer = st.number_input(
                            "🅼 Maxmart/ผสม Buffer %",
                            min_value=80,
                            max_value=150,
                            value=int(st.session_state.get('_maxmart_buf_pct', 100)),
                            step=5,
                            key="maxmart_buffer_input",
                        )
                    st.markdown('<div class="divider-label">🚛 กำหนดจำนวนรถ</div>', unsafe_allow_html=True)
                    col_f1, col_f2, col_f3 = st.columns(3)
                    with col_f1:
                        fleet_4w = st.number_input("🚗 4W (คัน)", min_value=0, max_value=99,
                                                   value=int(st.session_state.get('fleet_4w', 0)),
                                                   step=1, key="fleet_4w",
                                                   help="จำนวนรถ 4W ที่มีทั้งหมด (0 = ไม่จำกัด)")
                    with col_f2:
                        fleet_jb = st.number_input("🚚 JB (คัน)", min_value=0, max_value=99,
                                                   value=int(st.session_state.get('fleet_jb', 0)),
                                                   step=1, key="fleet_jb",
                                                   help="จำนวนรถ JB ที่มีทั้งหมด (0 = ไม่จำกัด)")
                    with col_f3:
                        fleet_6w = st.number_input("🚛 6W (คัน)", min_value=0, max_value=99,
                                                   value=int(st.session_state.get('fleet_6w', 0)),
                                                   step=1, key="fleet_6w",
                                                   help="จำนวนรถ 6W ที่มีทั้งหมด (0 = ไม่จำกัด)")
                    st.markdown('<div class="divider-label">⏰ เวลาและวันที่โหลดสินค้า</div>', unsafe_allow_html=True)
                    _ld_col1, _ld_col2 = st.columns(2)
                    with _ld_col1:
                        st.time_input(
                            "🕐 เวลาเริ่มโหลดแรก",
                            value=datetime.strptime("00:00", "%H:%M").time(),
                            key="load_start_time",
                            help="เลือกเวลาเริ่มต้นโหลดสินค้า (24 ชั่วโมง)",
                            step=1800,
                        )
                    with _ld_col2:
                        st.date_input(
                            "📅 วันที่โหลด",
                            value=datetime.now().date(),
                            key="load_date_input",
                            format="DD/MM/YYYY",
                            help="เลือกวันที่โหลดสินค้า",
                        )
                    st.caption("⏸️ เวลาพักและรถฮับ: ข้ามไป 1 ชม. อัตโนมัติ — คำนวณตาม Original QTY ในทริป")
                    # 🚀 ปุ่ม submit อยู่ใน form → กดปุ่มนี้เท่านั้นถึงจะ re-run (เปลี่ยนค่า input ไม่ re-run)
                    _tsf_submitted = st.form_submit_button("🚀 เริ่มจัดเที่ยว", type="primary", use_container_width=True)

                # ── ค่าที่ derived จาก form (accessible หลัง form close) ──
                st.session_state['_punthai_buf_pct'] = int(punthai_buffer)
                st.session_state['_maxmart_buf_pct'] = int(maxmart_buffer)
                punthai_buffer_value = punthai_buffer / 100.0
                maxmart_buffer_value = maxmart_buffer / 100.0
                st.caption(f"📏 Punthai: max C={5.0*punthai_buffer_value:.2f} / max W={1500*punthai_buffer_value:.0f} | Maxmart: max C={5.0*maxmart_buffer_value:.2f} / max W={1500*maxmart_buffer_value:.0f}")
                fleet_limits_input = {
                    '4W': int(fleet_4w) if fleet_4w > 0 else 999,
                    'JB': int(fleet_jb) if fleet_jb > 0 else 999,
                    '6W': int(fleet_6w) if fleet_6w > 0 else 999,
                }
                max_qty_per_trip = 0  # ไม่จำกัดจำนวนชิ้น

                st.markdown('<div class="divider-label">📋 ข้อจำกัดรถจาก Master Data</div>', unsafe_allow_html=True)
                
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
                
                # ── จัดทริปเมื่อ form submit ──
                if _tsf_submitted:
                    # เคลียร์ log เก่า
                    st.session_state['_ui_log'] = []
                    st.session_state['_trip_log'] = []
                    # สร้าง status container แบบ popup
                    with st.status("🚀 กำลังประมวลผล...", expanded=True) as status:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        log_area = st.empty()

                        status_text.write("⏳ กำลังเตรียมข้อมูล...")
                        progress_bar.progress(10)

                        df_to_process = df.copy()
                        progress_bar.progress(20)

                        import time as time_module, threading as _threading
                        start_time = time_module.time()

                        # ── รัน predict_trips ใน thread แยก เพื่อให้ log แสดง live ──
                        _result_box = {'result': None, 'error': None, 'done': False}

                        def _run_predict():
                            try:
                                _result_box['result'] = predict_trips(
                                    df_to_process, model_data,
                                    punthai_buffer=punthai_buffer_value,
                                    maxmart_buffer=maxmart_buffer_value,
                                    fleet_limits=fleet_limits_input,
                                    max_qty_per_trip=int(max_qty_per_trip)
                                )
                            except Exception as _ex:
                                _result_box['error'] = _ex
                            finally:
                                _result_box['done'] = True

                        _t = _threading.Thread(target=_run_predict, daemon=True)
                        _t.start()

                        # ── polling loop: อัปเดต log ทุก 0.4s ──
                        _tick = 0
                        while not _result_box['done']:
                            time_module.sleep(0.4)
                            _tick += 1
                            _logs_now = st.session_state.get('_ui_log', [])
                            if _logs_now:
                                _last_line = _logs_now[-1]
                                status_text.write(f"⏳ {_last_line[:120]}")
                                log_area.code('\n'.join(_logs_now[-30:]), language=None)
                            progress_bar.progress(min(88, 20 + _tick * 2))

                        _t.join()

                        if _result_box['error']:
                            status.update(label=f"❌ เกิดข้อผิดพลาด", state="error", expanded=True)
                            st.error(f"❌ {_result_box['error']}")
                            import traceback as _tb2
                            st.code(_tb2.format_exc(), language='text')
                            st.stop()

                        result_df, summary, fleet_used = _result_box['result']
                        elapsed_time = time_module.time() - start_time

                        # snapshot log หลังเสร็จ
                        _collected_log = list(st.session_state.get('_ui_log', []))
                        st.session_state['_trip_log'] = _collected_log
                        if _collected_log:
                            log_area.code('\n'.join(_collected_log[-30:]), language=None)

                        # 💾 เก็บผลลัพธ์ใน session_state
                        st.session_state['trip_result'] = result_df
                        st.session_state['trip_summary'] = summary
                        st.session_state['fleet_used'] = fleet_used
                        st.session_state['fleet_limits'] = fleet_limits_input
                        st.session_state['trip_buffers'] = {
                            'punthai': punthai_buffer_value,
                            'maxmart': maxmart_buffer_value
                        }
                        st.session_state['_trip_result_fresh'] = True
                        st.session_state['_trip_elapsed'] = elapsed_time

                        progress_bar.progress(100)
                        status_text.write(f"✅ จัดทริปเสร็จสิ้น! (ใช้เวลา {elapsed_time:.1f} วินาที)")
                        status.update(label=f"✅ ประมวลผลเสร็จสมบูรณ์! ({elapsed_time:.1f}s)", state="complete", expanded=False)

                        # 🗺️ Pre-cache เส้นทาง OSRM ทุกทริปใน background thread
                        # เพื่อให้แผนที่แสดงเส้นจริงทันทีโดยไม่ต้องรอ API ขณะ render
                        def _precache_routes(df_snapshot):
                            import threading
                            _dc_lat = DC_WANG_NOI_LAT
                            _dc_lon = DC_WANG_NOI_LON
                            _trip_ids = sorted(df_snapshot[df_snapshot['Trip'] > 0]['Trip'].unique())
                            _cached_count = 0
                            for _tid in _trip_ids:
                                _tdata = df_snapshot[df_snapshot['Trip'] == _tid]
                                _pts = []
                                for _, _r in _tdata.iterrows():
                                    _la = float(_r.get('_lat', 0) or 0)
                                    _lo = float(_r.get('_lon', 0) or 0)
                                    if _la > 0 and _lo > 0:
                                        _pts.append([_la, _lo])
                                if not _pts:
                                    continue
                                _wp = [[_dc_lat, _dc_lon]] + _pts + [[_dc_lat, _dc_lon]]
                                _ck = "|".join([f"{la:.4f},{lo:.4f}" for la, lo in _wp])
                                if USE_CACHE and _ck in ROUTE_CACHE_DATA:
                                    continue  # มีแล้ว ข้าม
                                get_multi_point_route_osrm(_wp)  # จะ cache ผลอัตโนมัติ
                                _cached_count += 1
                            if _cached_count > 0:
                                save_route_cache(ROUTE_CACHE_DATA, force=True)
                                safe_print(f"🗺️ Pre-cached {_cached_count} trip routes → route_cache.json")

                        import threading as _thr
                        _rt = _thr.Thread(
                            target=_precache_routes,
                            args=(result_df.copy(),),
                            daemon=True,
                            name="precache-routes"
                        )
                        _rt.start()
                    # ✅ ไม่ต้อง st.rerun() — ผลลัพธ์แสดงใน run เดียวกันได้เลย
                
                # 📊 แสดงผลลัพธ์ถ้ามีข้อมูลใน session_state
                if 'trip_result' in st.session_state and 'trip_summary' in st.session_state:
                    result_df = st.session_state['trip_result']
                    summary = st.session_state['trip_summary']

                    # ── แสดง log การจัดทริป ──
                    _trip_log = st.session_state.get('_trip_log', [])
                    if _trip_log:
                        _elapsed = st.session_state.get('_trip_elapsed', 0)
                        with st.expander(f"📋 Log การจัดทริป ({len(_trip_log)} บรรทัด · {_elapsed:.1f}s)", expanded=False):
                            st.code('\n'.join(_trip_log), language=None)

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

                    # ตรวจสอบสาขาที่ไม่ได้จัดทริป (Trip = 0) — ควรเป็น 0 หลัง Step 8.9
                    unassigned_count = len(result_df[result_df['Trip'] == 0])
                    if unassigned_count > 0:
                        st.warning(f"⚠️ มี {unassigned_count} สาขาที่ไม่ได้จัดทริป (Trip = 0)")
                        with st.expander(f"🔍 ดูรายละเอียดสาขาที่ไม่ได้จัดทริป ({unassigned_count} สาขา)"):
                            _ua_cols = [c for c in ['Code', 'Name', 'BU', 'Weight', 'Cube', 'Province'] if c in result_df.columns]
                            st.dataframe(result_df[result_df['Trip'] == 0][_ua_cols].reset_index(drop=True), hide_index=True)
                    
                    # กรองเฉพาะสาขาที่จัดทริปแล้ว สำหรับการแสดงผล
                    assigned_df = result_df[result_df['Trip'] > 0].copy()
                    
                    # แสดง balloons เฉพาะครั้งแรกที่ผลลัพธ์ใหม่ (ไม่เป็นทุก rerender)
                    if st.session_state.get('_trip_result_fresh', False):
                        st.balloons()
                        st.session_state['_trip_result_fresh'] = False
                    st.success(f"✅ **จัดทริปเสร็จสมบูรณ์!** รวม **{len(summary)}** ทริป ({len(assigned_df)} สาขา)")
                    
                    st.markdown('<div class="divider-label">📊 สรุปผลการจัดทริป</div>', unsafe_allow_html=True)
                    
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
                        st.markdown('<div class="divider-label">🚛 การใช้รถ</div>', unsafe_allow_html=True)
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

                    st.markdown('<div class="divider-label">🚛 รายละเอียดแต่ละทริป</div>', unsafe_allow_html=True)
                    
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
                    # ซ่อน BU_Type (ใช้ภายในเท่านั้น ไม่แสดงใน UI)
                    _summary_display = summary.drop(columns=[c for c in ['BU_Type'] if c in summary.columns])
                    if format_dict:
                        styled_df = _summary_display.style.format({k:v for k,v in format_dict.items() if k in _summary_display.columns})
                        if gradient_cols:
                            # สีแดง = < 90%, เหลือง = ~90-100%, เขียว = 100%+
                            styled_df = styled_df.background_gradient(
                                subset=[c for c in gradient_cols if c in _summary_display.columns],
                                cmap='RdYlGn',
                                vmin=0,
                                vmax=90
                            )
                        st.dataframe(styled_df, width="stretch", height=400)
                    else:
                        st.dataframe(_summary_display, width="stretch", height=400)

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
                    # อ่าน load_start_time (datetime.time จาก st.time_input)
                    _t_val = st.session_state.get('load_start_time')
                    if hasattr(_t_val, 'hour'):
                        _load_start_min = _t_val.hour * 60 + _t_val.minute
                    else:
                        try:
                            _lh2, _lm3 = map(int, str(_t_val or '00:00').split(':'))
                            _load_start_min = _lh2 * 60 + _lm3
                        except Exception:
                            _load_start_min = 0
                    # อ่าน load_date_input (datetime.date จาก st.date_input)
                    _d_val = st.session_state.get('load_date_input')
                    if hasattr(_d_val, 'strftime'):
                        _load_date_sig = _d_val.strftime('%d/%m/%Y')
                    else:
                        _load_date_sig = str(_d_val or datetime.now().strftime('%d/%m/%Y'))
                    _xl_sig = f"v9|{len(result_df)}|{int(result_df['Trip'].max())}|{sorted(result_df['Trip'].unique().tolist())}|{_load_start_min}|{_load_date_sig}"
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
                            vehicle_counts = {'4W': 0, 'JB': 0, '6W': 0}
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

                            trip_vehicle_map = {}   # _tnum → '4W'/'JB'/'6W'
                            _global_trip_seq = 0
                            for _tnum in sorted_trips:
                                _global_trip_seq += 1
                                _ts = summary[summary['Trip'] == _tnum]
                                _vt = '6W'
                                if len(_ts) > 0:
                                    _vi = _ts.iloc[0]['Truck']
                                    _vt = _vi.split()[0] if _vi else '6W'
                                    if _vt in ('4WJ',): _vt = 'JB'
                                    if _vt not in vehicle_counts: _vt = '6W'
                                    vehicle_counts[_vt] = vehicle_counts.get(_vt, 0) + 1
                                trip_vehicle_map[_tnum] = _vt
                                trip_no_map[_tnum] = f"{_vt}{_global_trip_seq:03d}"
                            # ── 5.5 helper functions สำหรับ load schedule (ใช้ใน section 7.5) ──
                            _PT_RATE_PURE = 25000   # P ล้วน: 25,000 ชิ้น/ชม.
                            _MM_RATE_PURE = 35000   # M ล้วน: 35,000 ชิ้น/ชม.
                            _MIX_RATE     = 39000  # P+M คละ: 15,000+25,000 = 40,000 ชิ้น/ชม. (2 ช่องพร้อมกัน)
                            _BREAK_STARTS = {7*60, 13*60, 19*60, 1*60}
                            _BREAK_DUR    = 60
                            _HUB_STARTS   = {8*60, 10*60, 12*60, 14*60}
                            _HUB_DUR      = 60
                            _DOORS_6W     = [2, 9]
                            _DOORS_SMALL  = [d for d in range(1, 10) if d not in _DOORS_6W]

                            def _skip_blocked(t_min):
                                for _ in range(20):
                                    t_mod = t_min % 1440
                                    changed = False
                                    for _bs in _BREAK_STARTS:
                                        if _bs <= t_mod < _bs + _BREAK_DUR:
                                            t_min += (_bs + _BREAK_DUR) - t_mod
                                            changed = True
                                            break
                                    if changed: continue
                                    for _hs in _HUB_STARTS:
                                        if _hs <= t_mod < _hs + _HUB_DUR:
                                            t_min += (_hs + _HUB_DUR) - t_mod
                                            changed = True
                                            break
                                    if not changed:
                                        break
                                return t_min

                            def _fmt_time(t_min):
                                t_mod = int(t_min) % 1440
                                return f"{t_mod//60:02d}:{t_mod%60:02d}"

                            def _fmt_date(t_min, base_date_str):
                                try:
                                    from datetime import datetime as _dt, timedelta as _td
                                    _bd = _dt.strptime(base_date_str.strip(), '%d/%m/%Y')
                                    _days = int(t_min) // 1440
                                    return (_bd + _td(days=_days)).strftime('%d/%m/%Y')
                                except Exception:
                                    return base_date_str

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

                            # ── 7.5 คำนวณ load schedule (เวลาโหลด/วันที่/ประตู) ──
                            # อ่านวันที่ (datetime.date หรือ string)
                            _d_val2 = st.session_state.get('load_date_input')
                            if hasattr(_d_val2, 'strftime'):
                                _base_date = _d_val2.strftime('%d/%m/%Y')
                            else:
                                _base_date = str(_d_val2 or datetime.now().strftime('%d/%m/%Y'))
                            _cur_min = _load_start_min
                            _cur_min = _skip_blocked(_cur_min)

                            trip_load_date: dict = {}
                            trip_load_time: dict = {}
                            trip_door:      dict = {}
                            # Door: 6W มี counter ของตัวเอง | 4W+JB ใช้ counter 'small' ร่วมกัน
                            _door_idx: dict = {'small': 0, '6W': 0}

                            # ── batch scheduling: ทุกประเภทรถรวมกัน (shared counter) ──
                            # qty สะสมรวมทุกทริปทุกประเภท — เมื่อเต็มชั่วโมง → flush
                            _max_qty_sched = int(max_qty_per_trip) if max_qty_per_trip and int(max_qty_per_trip) > 0 else 0

                            def _batch_rate_from_flags(has_pt, has_mm):
                                """เลือกอัตราโหลดตามประเภทสินค้าใน batch"""
                                if has_pt and has_mm:
                                    return _MIX_RATE       # P+M คละ: 40,000 ชิ้น/ชม.
                                elif has_pt:
                                    return _PT_RATE_PURE   # P ล้วน:  25,000 ชิ้น/ชม.
                                else:
                                    return _MM_RATE_PURE   # M ล้วน:  35,000 ชิ้น/ชม.

                            _user_qty_limit = int(max_qty_per_trip) if max_qty_per_trip and int(max_qty_per_trip) > 0 else 0

                            def _get_batch_limit(has_pt, has_mm):
                                if _user_qty_limit > 0:
                                    return _user_qty_limit
                                return _batch_rate_from_flags(has_pt, has_mm)

                            # shared batch state (รวมทุกประเภทรถ)
                            _sb = {'qty': 0.0, 'pt_qty': 0.0, 'mm_qty': 0.0,
                                   'start': _cur_min, 'has_pt': False, 'has_mm': False}

                            for _tnum in sorted_trips:
                                _vt_s = trip_vehicle_map.get(_tnum, '6W')
                                _rows_s = _trip_rows.get(_tnum, [])
                                # นับ PT qty และ MM qty แยกกันใน trip
                                _pt_qty_s = sum(_safe_float(_r.get('OriginalQty', 1)) for _r in _rows_s
                                                if str(_r.get('BU', '')).upper() in ('211', 'PUNTHAI'))
                                _mm_qty_s = sum(_safe_float(_r.get('OriginalQty', 1)) for _r in _rows_s
                                                if str(_r.get('BU', '')).upper() not in ('211', 'PUNTHAI'))
                                _trip_qty_s = _pt_qty_s + _mm_qty_s
                                if _trip_qty_s <= 0:
                                    _trip_qty_s = float(len(_rows_s) * 10)
                                    _pt_qty_s   = _trip_qty_s
                                _is_pt_s = _pt_qty_s > 0 and _mm_qty_s == 0
                                _is_mm_s = _mm_qty_s > 0 and _pt_qty_s == 0

                                # ตรวจ overflow (shared counter ทุกประเภทรวมกัน)
                                _new_pt = _sb['pt_qty'] + _pt_qty_s
                                _new_mm = _sb['mm_qty'] + _mm_qty_s
                                if _user_qty_limit > 0:
                                    _overflow = _sb['qty'] > 0 and _sb['qty'] + _trip_qty_s > _user_qty_limit
                                else:
                                    _load_hr = (_new_pt / _PT_RATE_PURE if _PT_RATE_PURE > 0 else 0) + \
                                               (_new_mm / _MM_RATE_PURE if _MM_RATE_PURE > 0 else 0)
                                    _overflow = _sb['qty'] > 0 and _load_hr > 1.0
                                if _overflow:
                                    # flush shared batch → เลื่อน 1 ชั่วโมง
                                    _sb['start']  = _skip_blocked(_sb['start'] + 60)
                                    _sb['qty']    = 0.0
                                    _sb['pt_qty'] = 0.0
                                    _sb['mm_qty'] = 0.0
                                    _sb['has_pt'] = False
                                    _sb['has_mm'] = False

                                # ใส่เวลา (shared)
                                trip_load_date[_tnum] = _fmt_date(_sb['start'], _base_date)
                                trip_load_time[_tnum] = _fmt_time(_sb['start'])

                                # สะสม qty (shared)
                                _sb['qty']    += _trip_qty_s
                                _sb['pt_qty'] += _pt_qty_s
                                _sb['mm_qty'] += _mm_qty_s
                                _sb['has_pt']  = _sb['has_pt'] or _is_pt_s
                                _sb['has_mm']  = _sb['has_mm'] or _is_mm_s

                                # ประตู: 6W วนของตัวเอง | 4W+JB ใช้ counter ร่วมกัน (_DOORS_SMALL)
                                if _vt_s == '6W':
                                    _door = _DOORS_6W[_door_idx['6W'] % len(_DOORS_6W)]
                                    _door_idx['6W'] += 1
                                else:
                                    # 4W และ JB แชร์ประตูชุดเดียวกัน
                                    _door = _DOORS_SMALL[_door_idx['small'] % len(_DOORS_SMALL)]
                                    _door_idx['small'] += 1
                                trip_door[_tnum] = _door

                            # ── 8. failed_trips — สีแดงเมื่อ util ต่ำกว่า 90% ──
                            _UTIL_RED_THRESHOLD = 90  # % — ต่ำกว่านี้ = แดง
                            failed_trips = set()
                            for _t in sorted_trips:
                                _rows_t = _trip_rows.get(_t, [])
                                if not _rows_t: continue
                                _is_pt = all(str(_r.get('BU', '')).upper() in ('211', 'PUNTHAI') for _r in _rows_t)
                                _vt2 = trip_vehicle_map.get(_t, '6W')
                                _lim = (PUNTHAI_LIMITS if _is_pt else LIMITS).get(_vt2, LIMITS['6W'])
                                _tw  = sum(_safe_float(_r.get('Weight', 0), 0) for _r in _rows_t)
                                _tc  = sum(_safe_float(_r.get('Cube',   0), 0) for _r in _rows_t)
                                if (_tw / _lim['max_w'] * 100) < _UTIL_RED_THRESHOLD and (_tc / _lim['max_c'] * 100) < _UTIL_RED_THRESHOLD:
                                    failed_trips.add(_t)

                            # ── 9. xlsxwriter write ──
                            _output = io.BytesIO()
                            try:
                                _wb_xl = _xlw.Workbook(_output, {'in_memory': True, 'constant_memory': True})
                                _ws_xl = _wb_xl.add_worksheet('2.Punthai')

                                # ── ดึงสไตล์ต้นฉบับ (font/row height) ──
                                _ostyle = st.session_state.get('_orig_style_info', {})
                                _ofont  = _ostyle.get('font_name', 'Angsana New')
                                _ofsize = _ostyle.get('font_size', 14.0)
                                _orh    = _ostyle.get('row_height', 15.0)
                                def _f(d):
                                    """เพิ่ม font ต้นฉบับเข้า format dict"""
                                    return {**d, 'font_name': _ofont, 'font_size': _ofsize}

                                _hdr_fmt = _wb_xl.add_format(_f({'bold':True,'border':1,'bg_color':'#D9D9D9','align':'center'}))
                                _yfmt    = _wb_xl.add_format(_f({'bg_color':'#FFE699','border':1}))
                                _wfmt    = _wb_xl.add_format(_f({'bg_color':'#FFFFFF','border':1}))
                                _yfmt_r  = _wb_xl.add_format(_f({'bg_color':'#FFE699','border':1,'font_color':'#FF0000'}))
                                _wfmt_r  = _wb_xl.add_format(_f({'bg_color':'#FFFFFF','border':1,'font_color':'#FF0000'}))
                                _ynfmt   = _wb_xl.add_format(_f({'bg_color':'#FFE699','border':1,'num_format':'#,##0.00'}))
                                _wnfmt   = _wb_xl.add_format(_f({'bg_color':'#FFFFFF','border':1,'num_format':'#,##0.00'}))
                                _ynfmt_r = _wb_xl.add_format(_f({'bg_color':'#FFE699','border':1,'num_format':'#,##0.00','font_color':'#FF0000'}))
                                _wnfmt_r = _wb_xl.add_format(_f({'bg_color':'#FFFFFF','border':1,'num_format':'#,##0.00','font_color':'#FF0000'}))
                                # DC summary row formats
                                _dc_fmt  = _wb_xl.add_format(_f({'bg_color':'#BDD7EE','border':1,'bold':True}))
                                _dc_nfmt = _wb_xl.add_format(_f({'bg_color':'#BDD7EE','border':1,'bold':True,'num_format':'#,##0.00'}))
                                # helper: 0-indexed col → Excel letter (A, B, ..., Z, AA, ...)
                                def _col_letter(n):
                                    s = ''
                                    n += 1
                                    while n > 0:
                                        n, r = divmod(n - 1, 26)
                                        s = chr(65 + r) + s
                                    return s

                                # ── column plan จากหัวคอลัมน์ต้นฉบับ ──
                                # คอลัมน์ที่ระบบเพิ่มเองหรือ internal — ห้ามซ้ำใน extra_export_cols
                                _FIXED_EXPORT_COLS = {
                                    'BU','Code','WMSCode','Name','Cube','Weight','OriginalQty','Trip',
                                    'Truck','MaxVehicle','VehicleCheck','Region','Distance_from_DC',
                                    'Province','District','Subdistrict','TripNo','Booking',
                                    'Latitude','Longitude','Max_Distance_in_Trip',
                                    'Route','Reference',   # ไม่ให้ extra_cols เพิ่ม Route จาก _rd
                                }
                                _extra_export_cols = [
                                    c for c in _rd.columns
                                    if c not in _FIXED_EXPORT_COLS
                                    and not str(c).startswith('_')
                                    and c.lower() not in {'หมายเหตุ', 'remark'}
                                ]

                                # โหลด original header info
                                _orig_hdr_info  = st.session_state.get('_orig_headers', [])   # [(orig_name, color), ...]
                                _rename_map_sv  = st.session_state.get('_col_rename_map', {}) # orig → internal
                                # DC row: map original col names → internal keys
                                _dc_row_raw = st.session_state.get('_orig_dc_row_raw', {})
                                _dc_row = {_rename_map_sv.get(k, k): v for k, v in _dc_row_raw.items()}

                                # format cache สำหรับสีหัวคอลัมน์
                                _hdr_fmt_cache = {}
                                def _get_hdr_fmt(c):
                                    if c not in _hdr_fmt_cache:
                                        _hdr_fmt_cache[c] = _wb_xl.add_format(_f({'bold':True,'border':1,'bg_color':c,'align':'center','font_color':'#000000'}))
                                    return _hdr_fmt_cache[c]

                                # default widths ต่อ internal name
                                _DEF_W = {'__SEP__':6,'BU':6,'Code':12,'WMSCode':12,'Name':30,
                                          'Cube':11,'Weight':11,'OriginalQty':12,
                                          '__SUB_GEN__':14,'__DIS_GEN__':14,'__PROV_GEN__':16,'__PROV_BLANK__':16,
                                          '__TRIP__':6,'__TRIPNO__':10,
                                          '__LOAD_DATE__':14,'__LOAD_TIME__':14,'__DOOR__':8,'__REMARK__':28}

                                # สร้าง column plan: [(display_name, color, internal_key), ...]
                                _GEO_COLOR  = '#E2EFDA'
                                _TRIP_COLOR = '#BDD7EE'
                                # คอลัมน์ geo จากต้นฉบับ → ข้าม (จะแทรกหลัง Name แทน)
                                _SKIP_IKEYS = {'Route','Reference','_rt',
                                               'Subdistrict','ตำบล','District','อำเภอ','Province','จังหวัด',
                                               '__TRIP__','__TRIPNO__',
                                               '__PROV_BLANK__','__SUB_GEN__','__DIS_GEN__','__PROV_GEN__',
                                               '__LOAD_DATE__','__LOAD_TIME__','__DOOR__',
                                               'remark','Remark','REMARK','หมายเหตุ'}
                                # Trip / Trip no / วันที่โหลด / เวลาโหลด / ประตู → แทนด้วย generated ในตำแหน่งเดิม
                                _SCH_COLOR  = '#FCE4D6'   # สีส้มอ่อน — วันที่/เวลา/ประตู
                                _REPLACE_MAP = {
                                    'Sep.':            ('__SEP__',       '#D9D9D9'),
                                    'Sep':             ('__SEP__',       '#D9D9D9'),
                                    'SEP':             ('__SEP__',       '#D9D9D9'),
                                    'sep.':            ('__SEP__',       '#D9D9D9'),
                                    'ลำดับ':          ('__SEP__',       '#D9D9D9'),
                                    'Trip':            ('__TRIP__',      _TRIP_COLOR),
                                    'T':               ('__TRIP__',      _TRIP_COLOR),
                                    'Trip no':         ('__TRIPNO__',    _TRIP_COLOR),
                                    'Trip No':         ('__TRIPNO__',    _TRIP_COLOR),
                                    'TripNo':          ('__TRIPNO__',    _TRIP_COLOR),
                                    'TripNumber':      ('__TRIPNO__',    _TRIP_COLOR),
                                    'วันที่โหลด':    ('__LOAD_DATE__', _SCH_COLOR),
                                    'LoadDate':        ('__LOAD_DATE__', _SCH_COLOR),
                                    'เวลาโหลด(ประมาณ)': ('__LOAD_TIME__', _SCH_COLOR),
                                    'เวลาโหลด':     ('__LOAD_TIME__', _SCH_COLOR),
                                    'LoadTime':        ('__LOAD_TIME__', _SCH_COLOR),
                                    'ประตู':          ('__DOOR__',      _SCH_COLOR),
                                    'Door':            ('__DOOR__',      _SCH_COLOR),
                                }
                                _col_plan = []
                                _orig_has_trip    = False
                                _orig_has_tripno  = False
                                _orig_has_loaddate= False
                                _orig_has_loadtime= False
                                _orig_has_door    = False
                                _orig_has_sep     = False
                                _geo_inserted    = False   # แทรก geo หลัง Name แค่ครั้งเดียว
                                if _orig_hdr_info:
                                    for _oname, _oclr in _orig_hdr_info:
                                        _ikey = _rename_map_sv.get(_oname, _oname)
                                        if _ikey in _SKIP_IKEYS:
                                            continue
                                        if _ikey in _REPLACE_MAP:
                                            _gen_ikey, _gen_clr = _REPLACE_MAP[_ikey]
                                            _col_plan.append((_oname, _gen_clr, _gen_ikey))
                                            if _gen_ikey == '__SEP__':
                                                _orig_has_sep = True
                                            elif _gen_ikey == '__TRIP__':
                                                _orig_has_trip = True
                                            elif _gen_ikey == '__TRIPNO__':
                                                _orig_has_tripno = True
                                            elif _gen_ikey == '__LOAD_DATE__':
                                                _orig_has_loaddate = True
                                            elif _gen_ikey == '__LOAD_TIME__':
                                                _orig_has_loadtime = True
                                            elif _gen_ikey == '__DOOR__':
                                                _orig_has_door = True
                                        else:
                                            _col_plan.append((_oname, _oclr, _ikey))
                                            # แทรก ตำบล/อำเภอ/จังหวัด(generated) หลัง Name
                                            if _ikey == 'Name' and not _geo_inserted:
                                                _geo_inserted = True
                                                _col_plan.append(('ตำบล',    _GEO_COLOR, '__SUB_GEN__'))
                                                _col_plan.append(('อำเภอ',   _GEO_COLOR, '__DIS_GEN__'))
                                                _col_plan.append(('จังหวัด', _GEO_COLOR, '__PROV_GEN__'))
                                else:
                                    # fallback ถ้าไม่มีข้อมูลต้นฉบับ
                                    _col_plan = [
                                        ('Sep.',        '#D9D9D9', '__SEP__'),
                                        ('BU',          '#D9D9D9', 'BU'),
                                        ('รหัสสาขา',    '#D9D9D9', 'Code'),
                                        ('รหัส WMS',    '#D9D9D9', 'WMSCode'),
                                        ('สาขา',        '#D9D9D9', 'Name'),
                                        ('ตำบล',        _GEO_COLOR, '__SUB_GEN__'),
                                        ('อำเภอ',       _GEO_COLOR, '__DIS_GEN__'),
                                        ('จังหวัด',     _GEO_COLOR, '__PROV_GEN__'),
                                        ('Total Cube',  '#D9D9D9', 'Cube'),
                                        ('Total Wgt',   '#D9D9D9', 'Weight'),
                                        ('Original QTY','#D9D9D9', 'OriginalQty'),
                                    ]

                                # ต่อด้วย extra cols จากต้นฉบับที่ยังไม่อยู่ใน plan
                                _plan_ikeys  = {k for _, _, k in _col_plan}
                                _plan_dnames = {n for n, _, _ in _col_plan}
                                for _ec in _extra_export_cols:
                                    if _ec not in _plan_ikeys and _ec not in _plan_dnames:
                                        _col_plan.append((_ec, '#D9D9D9', _ec))

                                # เพิ่ม T + Trip no ท้ายสุด เฉพาะกรณีต้นฉบับไม่มี
                                if not _orig_has_sep:
                                    _col_plan.insert(0, ('Sep.', '#D9D9D9', '__SEP__'))
                                if not _orig_has_trip:
                                    _col_plan.append(('T',           _TRIP_COLOR, '__TRIP__'))
                                if not _orig_has_tripno:
                                    _col_plan.append(('Trip no',     _TRIP_COLOR, '__TRIPNO__'))
                                # วันที่โหลด / เวลาโหลด / ประตู — ต่อท้าย ถ้าต้นฉบับไม่มี
                                if not _orig_has_loaddate:
                                    _col_plan.append(('วันที่โหลด',       _SCH_COLOR, '__LOAD_DATE__'))
                                if not _orig_has_loadtime:
                                    _col_plan.append(('เวลาโหลด(ประมาณ)', _SCH_COLOR, '__LOAD_TIME__'))
                                if not _orig_has_door:
                                    _col_plan.append(('ประตู',            _SCH_COLOR, '__DOOR__'))
                                # หมายเหตุ: แทรกทันทีหลัง ประตู (__DOOR__) — ถ้าไม่มีประตู → ต่อท้ายก่อนจังหวัด
                                _door_idx = next((i for i, (_, _, k) in enumerate(_col_plan) if k == '__DOOR__'), None)
                                if _door_idx is not None:
                                    _col_plan.insert(_door_idx + 1, ('หมายเหตุ', _SCH_COLOR, '__REMARK__'))
                                else:
                                    _col_plan.append(('หมายเหตุ', _SCH_COLOR, '__REMARK__'))
                                # จังหวัด (ว่าง) — ท้ายสุดเสมอ
                                _col_plan.append(('จังหวัด', _GEO_COLOR, '__PROV_BLANK__'))

                                # ── Title row (row 0): แผนงาน + วันที่ (อิงจาก load_date_input) ──
                                _title_fmt  = _wb_xl.add_format(_f({'bold':True,'font_size':_ofsize,'align':'left'}))
                                _title_rfmt = _wb_xl.add_format(_f({'bold':True,'font_size':_ofsize,'num_format':'#,##0','font_color':'#FF0000','align':'right'}))
                                try:
                                    from datetime import datetime as _dt2, timedelta as _td2
                                    _bd2 = _dt2.strptime(_base_date, '%d/%m/%Y')
                                    _ord_d = _bd2.strftime('%d/%m/%y')
                                    _pik_d = (_bd2 + _td2(days=1)).strftime('%d/%m/%y')
                                except Exception:
                                    _ord_d = _base_date; _pik_d = _base_date
                                _title_text = f"แผนงานรอบสั่งวันที่ {_ord_d} รอบหยิบวันที่ {_pik_d}"
                                _ws_xl.write(0, 1, _title_text, _title_fmt)
                                # SUM จำนวนชิ้นทั้งหมด (สีแดง) ในหัวแถว
                                _qty_ci = next((i for i, (_, _, k) in enumerate(_col_plan) if k == 'OriginalQty'), None)
                                if _qty_ci is not None:
                                    _qty_ltr = _col_letter(_qty_ci)
                                    _ws_xl.write_formula(0, _qty_ci, f'=SUM({_qty_ltr}3:{_qty_ltr}9999)', _title_rfmt, 0)
                                _ws_xl.set_row(0, 20)

                                # ── Header row (row 1) ──
                                for _ci, (_dname, _dcolor, _ikey) in enumerate(_col_plan):
                                    _ws_xl.write(1, _ci, _dname, _get_hdr_fmt(_dcolor))
                                _ws_xl.set_row(1, 18)
                                for _ci, (_, _, _ikey) in enumerate(_col_plan):
                                    _cw = _DEF_W.get(_ikey, max(12, len(str(_ikey))+2))
                                    _ws_xl.set_column(_ci, _ci, _cw)

                                # format สำหรับแถวว่างท้ายทริป
                                _sep_fmt = _wb_xl.add_format({'border':0})

                                # ── คำนวณ constraint ที่ถึงก่อนต่อทริป (น้ำหนัก/คิว) ──
                                trip_remark = {}
                                for _t_rm in sorted_trips:
                                    _rows_rm = _trip_rows.get(_t_rm, [])
                                    _vt_rm = trip_vehicle_map.get(_t_rm, '6W')
                                    _is_pt_rm = all(str(_r.get('BU','')).strip() in ('211','PUNTHAI') for _r in _rows_rm) if _rows_rm else False
                                    _lim_rm = (PUNTHAI_LIMITS if _is_pt_rm else LIMITS).get(_vt_rm, LIMITS['6W'])
                                    _tw_rm = sum(_safe_float(_r.get('Weight',0), 0) for _r in _rows_rm)
                                    _tc_rm = sum(_safe_float(_r.get('Cube',0), 0) for _r in _rows_rm)
                                    _wu_rm = _tw_rm / _lim_rm['max_w'] if _lim_rm['max_w'] > 0 else 0
                                    _cu_rm = _tc_rm / _lim_rm['max_c'] if _lim_rm['max_c'] > 0 else 0
                                    if _wu_rm >= _cu_rm:
                                        trip_remark[_t_rm] = f"น้ำหนัก {int(_tw_rm):,}/{int(_lim_rm['max_w']):,}kg ({_wu_rm*100:.0f}%)"
                                    else:
                                        trip_remark[_t_rm] = f"คิว {_tc_rm:.2f}/{_lim_rm['max_c']:.1f}m\u00b3 ({_cu_rm*100:.0f}%)"

                                # เรียงทริปตามเวลาโหลด (เพื่อให้ Excel เรียงตามลำดับโหลดจริง)
                                def _dt_sort_key(t):
                                    _d = trip_load_date.get(t, '31/12/9999')
                                    _tm = trip_load_time.get(t, '99:99')
                                    try:
                                        _p = _d.split('/')
                                        _ds = f"{_p[2]}/{_p[1]}/{_p[0]}"  # DD/MM/YYYY → YYYY/MM/DD
                                    except Exception:
                                        _ds = '9999/99/99'
                                    return (_ds, _tm)
                                export_sorted_trips = sorted(sorted_trips, key=_dt_sort_key)

                                use_yellow = True
                                _row_xl = 2
                                _row_seq = 1   # Sep. sequential ต่อแถว (รวม DC row)

                                for _tnum in export_sorted_trips:
                                    _rows = _trip_rows.get(_tnum, [])
                                    _tno  = trip_no_map.get(_tnum, '')
                                    _is_f = _tnum in failed_trips
                                    _tf = (_yfmt_r if _is_f else _yfmt) if use_yellow else (_wfmt_r if _is_f else _wfmt)
                                    _nf = (_ynfmt_r if _is_f else _ynfmt) if use_yellow else (_wnfmt_r if _is_f else _wnfmt)
                                    use_yellow = not use_yellow
                                    _tnum_int = int(_tnum)
                                    _tno_str  = str(_tno)
                                    _first_row_of_trip = True
                                    _trip_start_row = _row_xl  # เก็บแถวเริ่มต้น (0-indexed) สำหรับสูตร SUM
                                    for _rec in _rows:
                                        for _ci, (_, _, _ikey) in enumerate(_col_plan):
                                            _is_num = _ikey in ('Cube', 'Weight')
                                            _dfmt = _nf if _is_num else _tf
                                            if _ikey == '__SEP__':
                                                _val = _row_seq  # sequential ทุกแถว
                                            elif _ikey == '__PROV_BLANK__':
                                                _val = ''   # ว่างไว้ให้ user กรอกเอง
                                            elif _ikey == '__TRIP__':
                                                _val = _tnum_int
                                            elif _ikey == '__TRIPNO__':
                                                _val = _tno_str
                                            elif _ikey == 'BU':
                                                _val = _rec.get('BU', 211)
                                            elif _ikey == 'Code':
                                                _val = str(_rec.get('Code', ''))
                                            elif _ikey == 'WMSCode':
                                                _val = str(_rec.get('WMSCode', _rec.get('Code', '')))
                                            elif _ikey == 'Name':
                                                _val = str(_rec.get('Name', ''))
                                            elif _ikey == 'Cube':
                                                _val = round(_safe_float(_rec.get('Cube', 0), 0), 2)
                                            elif _ikey == 'Weight':
                                                _val = round(_safe_float(_rec.get('Weight', 0), 0), 2)
                                            elif _ikey == 'OriginalQty':
                                                _val = int(_safe_float(_rec.get('OriginalQty', 0), 0))
                                            elif _ikey == '__SUB_GEN__':
                                                _val = str(_rec.get('_sp_eff','') or _rec.get('_sp','') or _rec.get('Subdistrict','') or _rec.get('ตำบล',''))
                                            elif _ikey == '__DIS_GEN__':
                                                _val = str(_rec.get('_sd_eff','') or _rec.get('_sd','') or _rec.get('District','') or _rec.get('อำเภอ',''))
                                            elif _ikey == '__PROV_GEN__':
                                                _val = str(_rec.get('_sv_eff','') or _rec.get('_sv','') or _rec.get('Province','') or _rec.get('จังหวัด',''))
                                            elif _ikey == '__PROV_BLANK__':
                                                _val = ''   # จังหวัดจากต้นฉบับ — ว่างไว้ให้กรอกเอง
                                            elif _ikey == '__LOAD_DATE__':
                                                _val = trip_load_date.get(_tnum, '')  # ทุกแถว
                                            elif _ikey == '__LOAD_TIME__':
                                                _val = trip_load_time.get(_tnum, '')  # ทุกแถว
                                            elif _ikey == '__DOOR__':
                                                _val = trip_door.get(_tnum, '')  # ทุกแถว
                                            elif _ikey == '__REMARK__':
                                                _val = trip_remark.get(_tnum, '') if _first_row_of_trip else ''
                                            else:
                                                # คอลัมน์จากต้นฉบับ — ใช้ค่าตรงๆ จากไฟล์
                                                _val = _rec.get(_ikey, '')
                                                if _val is None or (isinstance(_val, float) and _val != _val):
                                                    _val = ''
                                            _ws_xl.write(_row_xl, _ci, _val, _dfmt)
                                        _ws_xl.set_row(_row_xl, _orh)
                                        _row_xl += 1
                                        _row_seq += 1
                                        _first_row_of_trip = False

                                    # ── จุดสุดท้าย: DC return row (ท้ายทริป) ──
                                    for _ci, (_, _, _ikey) in enumerate(_col_plan):
                                        if _ikey == '__SEP__':
                                            _dcval = _row_seq
                                        elif _ikey == 'BU':
                                            _dcval = 'PROJECT'
                                        elif _ikey == 'Code':
                                            _dcval = 'DC011'
                                        elif _ikey == 'WMSCode':
                                            _dcval = 'DC011'
                                        elif _ikey == 'Name':
                                            _dcval = 'บ.พีทีจี เอ็นเนอยี จำกัด (มหาชน) (DCวังน้อย)'
                                        elif _ikey in ('Cube', 'Weight'):
                                            _dcval = 0.0
                                        elif _ikey == 'OriginalQty':
                                            _dcval = 0
                                        elif _ikey == '__LOAD_TIME__':
                                            _dcval = trip_load_time.get(_tnum, '')
                                        elif _ikey == '__DOOR__':
                                            _dcval = trip_door.get(_tnum, '')
                                        elif _ikey == '__TRIP__':
                                            _dcval = _tnum_int
                                        elif _ikey == '__TRIPNO__':
                                            _dcval = _tno_str
                                        elif _ikey == '__LOAD_DATE__':
                                            _dcval = trip_load_date.get(_tnum, '')
                                        else:
                                            _dcval = ''
                                        _is_dc_num = _ikey in ('Cube', 'Weight')
                                        _ws_xl.write(_row_xl, _ci, _dcval, _nf if _is_dc_num else _tf)
                                    _ws_xl.set_row(_row_xl, _orh)
                                    _row_xl += 1
                                    _row_seq += 1

                                _wb_xl.close()
                                _output.seek(0)

                            except Exception as _xe:
                                st.warning(f"⚠️ xlsxwriter error: {_xe} — fallback to basic")
                                _output = io.BytesIO()
                                with pd.ExcelWriter(_output, engine='xlsxwriter') as _writer:
                                    _exp = _rd.drop(columns=[c for c in ['_key_u','_sp','_sd','_sv','_rt','_trip_order','_sp_eff','_sd_eff','_sv_eff','_rorder'] if c in _rd.columns], errors='ignore').copy()
                                    _exp['Trip_No'] = _exp['Trip'].map(lambda x: trip_no_map.get(x, ''))
                                    _exp.to_excel(_writer, sheet_name='รายละเอียดทริป', index=False)
                                    summary.drop(columns=[c for c in ['BU_Type'] if c in summary.columns]).to_excel(_writer, sheet_name='สรุปทริป', index=False)

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
                            import importlib as _imp, trip_map_interactive as _tmi
                            import time as _time_mod
                            import streamlit.components.v1 as _cmp2
                            import hashlib as _hl
                            _build_imap = _tmi.build_interactive_map_html

                            _trip_bufs = st.session_state.get('trip_buffers', {'punthai': 1.0, 'maxmart': 1.0})
                            _imap_pt_buf = _trip_bufs.get('punthai', 1.0)
                            _imap_mm_buf = _trip_bufs.get('maxmart', 1.0)

                            _imap_sig = f"v30|{len(assigned_df)}|{int(assigned_df['Trip'].max())}|{sorted(assigned_df['Trip'].unique().tolist())}|pt{_imap_pt_buf}|mm{_imap_mm_buf}"
                            _imap_key = _hl.md5(_imap_sig.encode()).hexdigest()[:12]

                            if st.session_state.get('_imap_key') != _imap_key:
                                with st.spinner("🗺️ กำลังสร้างแผนที่..."):
                                    _t_map = _time_mod.time()
                                    _imap_html = _build_imap(
                                        result_df=assigned_df,
                                        summary_df=summary,
                                        limits=LIMITS,
                                        punthai_limits=PUNTHAI_LIMITS,
                                        punthai_buffer=_imap_pt_buf,
                                        maxmart_buffer=_imap_mm_buf,
                                        trip_no_map=trip_no_map,
                                        dc_lat=14.1459, dc_lon=100.6873,
                                        route_cache=ROUTE_CACHE_DATA,
                                    )
                                    st.session_state['_imap_html'] = _imap_html
                                    st.session_state['_imap_key'] = _imap_key
                                    st.session_state['_imap_build_time'] = _time_mod.time() - _t_map

                            _htm = st.session_state.get('_imap_html', '')
                            if not _htm:
                                st.warning("⚠️ ยังไม่มีข้อมูลแผนที่ กดจัดเที่ยวก่อน")
                            else:
                                # Sanitize surrogates
                                try:
                                    _htm.encode('utf-8')
                                except UnicodeEncodeError:
                                    _htm = _htm.encode('utf-8', errors='replace').decode('utf-8')
                                    st.session_state['_imap_html'] = _htm
                                import re as _re
                                _nb = len(_re.findall(r'"code":', _htm))
                                _build_t = st.session_state.get('_imap_build_time', 0)
                                st.caption(f"🗺️ HTML: {len(_htm)//1024} KB · {_nb} สาขา · build: {_build_t:.1f}s")
                                _cmp2.html(_htm, height=860, scrolling=False)
                        except Exception as _e:
                            import traceback as _tb
                            st.error(f"❌ Interactive map error: {_e}")
                            st.code(_tb.format_exc(), language='text')
                            st.info(f"📋 columns: {list(assigned_df.columns)} | rows: {len(assigned_df)} | trips: {sorted(assigned_df['Trip'].unique().tolist())}")

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

                                                    # ใช้ ROUTE_CACHE_DATA (global file-backed) โดยตรงผ่าน get_multi_point_route_osrm
                                                    real_route_coords, total_trip_distance = get_multi_point_route_osrm(waypoints)

                                                    # fallback: ถ้า OSRM ล้มเหลว (coords = waypoints เหมือนกัน) → คำนวณ distance จาก cache
                                                    if total_trip_distance == 0 or real_route_coords == waypoints:
                                                        _fb_dist = 0
                                                        for _wi in range(len(waypoints) - 1):
                                                            lat1, lon1 = waypoints[_wi]
                                                            lat2, lon2 = waypoints[_wi + 1]
                                                            _fb_dist += haversine_distance(lat1, lon1, lat2, lon2, use_osrm_cache=False)
                                                        total_trip_distance = _fb_dist
                                                        # เส้นเชื่อม DC → waypoints แบบ straight-segment (fallback)
                                                        real_route_coords = waypoints

                                                    # 🏭 หมุด DC ออกเดิน (ลำดับ 0)
                                                    _dc_start_label = f'<div style="background-color:{trip_color};color:#fff;border-radius:12px;min-width:50px;height:24px;text-align:center;line-height:24px;font-weight:bold;font-size:10px;border:2px solid #000;box-shadow:2px 2px 6px rgba(0,0,0,0.5);padding:0 4px;">T{trip_id}(0)</div>'
                                                    folium.Marker(
                                                        location=[DC_LAT, DC_LON],
                                                        popup=folium.Popup(f"<b>🏭 DC ออกเดิน — Trip {trip_id}</b><br>ลำดับ: 0<br>ระยะทางรวม: {total_trip_distance:.1f} km", max_width=250),
                                                        tooltip=f"Trip {trip_id} - 0. DC (ออกเดิน)",
                                                        icon=folium.DivIcon(html=_dc_start_label)
                                                    ).add_to(fg)
                                                
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

                                                    # 🏭 หมุด DC กลับ (ลำดับสุดท้าย = len(points)+1)
                                                    _dc_return_label = f'<div style="background-color:{trip_color};color:#fff;border-radius:12px;min-width:50px;height:24px;text-align:center;line-height:24px;font-weight:bold;font-size:10px;border:2px solid #000;box-shadow:2px 2px 6px rgba(0,0,0,0.5);padding:0 4px;">T{trip_id}({len(points)+1})</div>'
                                                    folium.Marker(
                                                        location=[DC_LAT, DC_LON],
                                                        popup=folium.Popup(f"<b>🏭 DC กลับ — Trip {trip_id}</b><br>ลำดับ: {len(points)+1}<br>ระยะทางรวม: {total_trip_distance:.1f} km", max_width=250),
                                                        tooltip=f"Trip {trip_id} - {len(points)+1}. DC (กลับ)",
                                                        icon=folium.DivIcon(html=_dc_return_label)
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
                                                # ใช้ OSRM multi-point route เพื่อระยะทางถนนจริง
                                                _wp = [[DC_LAT, DC_LON]] + points
                                                _cache_k = "|".join([f"{la:.4f},{lo:.4f}" for la, lo in _wp])
                                                if USE_CACHE and _cache_k in ROUTE_CACHE_DATA:
                                                    _rc = ROUTE_CACHE_DATA[_cache_k]
                                                    route_distance = _rc.get('distance', 0)
                                                else:
                                                    _, route_distance = get_multi_point_route_osrm(_wp)
                                                # inter_branch = OSRM ไม่รวม DC leg แรก → ประมาณจาก DISTANCE_CACHE
                                                for j in range(len(points) - 1):
                                                    seg = haversine_distance(points[j][0], points[j][1], points[j+1][0], points[j+1][1])
                                                    if seg < 9999:
                                                        inter_branch_distance += seg
                                            
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
                st.markdown('<div class="divider-label">🗺️ สาขาแยกตามภาค</div>', unsafe_allow_html=True)
                
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
                st.markdown('<div class="divider-label">🔗 กลุ่มสาขาที่เคยไปด้วยกัน</div>', unsafe_allow_html=True)
                
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
                st.markdown('<div class="divider-label">🏙️ โซนจัดส่งของสาขา</div>', unsafe_allow_html=True)
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
        # บันทึก cache ก่อนปิดโปรแกรม (เฉพาะเมื่อมีการเปลี่ยนแปลง)
        if USE_CACHE:
            save_distance_cache(DISTANCE_CACHE, force=True)
            save_route_cache(ROUTE_CACHE_DATA, force=True)
            safe_print(f"💾 บันทึก cache: {len(DISTANCE_CACHE)} ระยะทาง, {len(ROUTE_CACHE_DATA)} เส้นทาง")

