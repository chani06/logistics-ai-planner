"""
Logistics Planner - Simple & Fast Version
ลดความซับซ้อน เน้นความเร็ว
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import io
from math import radians, sin, cos, sqrt, atan2

# ==========================================
# CONFIG - Simple Version
# ==========================================
st.set_page_config(page_title="🚛 Trip Planner (Fast)", layout="wide")

# ขีดจำกัดรถ
LIMITS = {
    '6W': {'max_c': 20, 'min_c': 18, 'max_w': 7000, 'max_drops': float('inf')},
    'JB': {'max_c': 7, 'max_w': 3500, 'max_drops': 7},
    '4W': {'max_c': 5, 'max_w': 2500, 'max_drops': 12}
}

# พิกัด DC วังน้อย
DC_LAT = 14.179394
DC_LON = 100.648149

# ระยะทาง threshold
NEAR_DC_THRESHOLD = 150  # km - nearby
FAR_DC_THRESHOLD = 290   # km - upcountry

# ตัดออก
EXCLUDE_BRANCHES = ['DC011', 'PTDC', 'PTG DISTRIBUTION CENTER']

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def haversine(lat1, lon1, lat2, lon2):
    """คำนวณระยะทาง Haversine (km)"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def get_distance_from_dc(lat, lon):
    """ระยะห่างจาก DC (km)"""
    if pd.isna(lat) or pd.isna(lon):
        return 0
    return haversine(DC_LAT, DC_LON, lat, lon)

def recommend_vehicle(total_cube, total_weight, branch_count, distance_from_dc):
    """แนะนำประเภทรถ"""
    # เช็คเกิน 6W
    if total_cube > LIMITS['6W']['max_c'] * 1.05 or total_weight > LIMITS['6W']['max_w'] * 1.05:
        return '6W', 'เกินขีดจำกัด'
    
    # ห่างจาก DC มาก → 6W
    if distance_from_dc > FAR_DC_THRESHOLD:
        return '6W', f'ห่าง DC > {FAR_DC_THRESHOLD} km'
    
    # เช็ค 4W
    if total_cube <= LIMITS['4W']['max_c'] * 1.05 and total_weight <= LIMITS['4W']['max_w'] * 1.05:
        if branch_count <= LIMITS['4W']['max_drops']:
            return '4W', 'พอดี 4W'
    
    # เช็ค JB
    if total_cube <= LIMITS['JB']['max_c'] * 1.05 and total_weight <= LIMITS['JB']['max_w'] * 1.05:
        if branch_count <= LIMITS['JB']['max_drops']:
            return 'JB', 'พอดี JB'
    
    # Default: 6W
    return '6W', 'เกิน JB/4W'

# ==========================================
# CORE ALGORITHM - Simple & Fast
# ==========================================
def process_trips_simple(df):
    """ประมวลผลทริป - แบบง่ายและเร็ว"""
    start_time = datetime.now()
    
    # ตัด DC ออก
    df = df[~df['Code'].isin(EXCLUDE_BRANCHES)].copy()
    
    if len(df) == 0:
        return df, {}
    
    # คำนวณระยะห่างจาก DC
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        df['Distance_DC'] = df.apply(
            lambda r: get_distance_from_dc(r.get('Latitude', 0), r.get('Longitude', 0)), 
            axis=1
        )
    else:
        df['Distance_DC'] = 0
    
    # เรียงตามระยะห่างจาก DC (ไกล → ใกล้)
    df = df.sort_values('Distance_DC', ascending=False).reset_index(drop=True)
    
    # จัดกลุ่มสาขา → ทริป
    df['Trip'] = 0
    trip_vehicles = {}
    current_trip = 1
    
    assigned = set()
    
    for idx, row in df.iterrows():
        if row['Code'] in assigned:
            continue
        
        # เริ่มทริปใหม่
        trip_branches = [row['Code']]
        trip_cube = row.get('Cube', 0)
        trip_weight = row.get('Weight', 0)
        trip_distance = row.get('Distance_DC', 0)
        assigned.add(row['Code'])
        
        # หา neighbors ที่อยู่ใกล้
        for idx2, row2 in df.iterrows():
            if row2['Code'] in assigned:
                continue
            
            # เช็คว่าใกล้กันพอไหม (ระยะห่างจาก DC ใกล้กัน)
            distance_diff = abs(row2.get('Distance_DC', 0) - trip_distance)
            
            if distance_diff <= 50:  # ระยะห่างไม่เกิน 50 km
                new_cube = trip_cube + row2.get('Cube', 0)
                new_weight = trip_weight + row2.get('Weight', 0)
                new_count = len(trip_branches) + 1
                
                # เช็คว่าเพิ่มได้ไหม
                if new_cube <= LIMITS['6W']['max_c'] and new_weight <= LIMITS['6W']['max_w']:
                    if new_count <= 12:  # ไม่เกิน 12 สาขา
                        trip_branches.append(row2['Code'])
                        trip_cube = new_cube
                        trip_weight = new_weight
                        assigned.add(row2['Code'])
        
        # กำหนดทริป
        df.loc[df['Code'].isin(trip_branches), 'Trip'] = current_trip
        
        # แนะนำรถ
        vehicle, reason = recommend_vehicle(trip_cube, trip_weight, len(trip_branches), trip_distance)
        trip_vehicles[current_trip] = vehicle
        
        current_trip += 1
    
    elapsed = (datetime.now() - start_time).total_seconds()
    st.success(f"✅ ประมวลผลเสร็จใน {elapsed:.1f} วินาที | {current_trip-1} ทริป")
    
    return df, trip_vehicles

# ==========================================
# STREAMLIT UI
# ==========================================
def main():
    st.title("🚛 Trip Planner - Fast Version v3.0 ⚡")
    st.caption("Simple & Fast - เน้นความเร็วในการประมวลผล")
    
    # Upload file
    uploaded_file = st.file_uploader("📁 อัพโหลดไฟล์ Excel", type=['xlsx', 'xls'])
    
    if uploaded_file:
        try:
            # อ่านไฟล์
            df = pd.read_excel(uploaded_file)
            st.info(f"📊 ข้อมูล: {len(df)} แถว, {len(df.columns)} คอลัมน์")
            
            # แสดงตัวอย่าง
            with st.expander("📋 ดูข้อมูลตัวอย่าง"):
                st.dataframe(df.head(20))
            
            # ปุ่มประมวลผล
            if st.button("🚀 ประมวลผลทริป", type="primary"):
                with st.spinner("⏳ กำลังประมวลผล..."):
                    result_df, trip_vehicles = process_trips_simple(df)
                
                if len(result_df) > 0:
                    # แสดงสรุป
                    st.subheader("📊 สรุปผลการจัดทริป")
                    
                    trips = result_df[result_df['Trip'] > 0]['Trip'].unique()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("จำนวนทริป", len(trips))
                    with col2:
                        count_6w = sum(1 for v in trip_vehicles.values() if v == '6W')
                        st.metric("รถ 6W", count_6w)
                    with col3:
                        count_small = sum(1 for v in trip_vehicles.values() if v in ['4W', 'JB'])
                        st.metric("รถ 4W/JB", count_small)
                    
                    # เพิ่ม Vehicle column
                    result_df['Vehicle'] = result_df['Trip'].map(trip_vehicles)
                    
                    # แสดงผลลัพธ์
                    st.subheader("📋 รายละเอียดทริป")
                    
                    for trip_num in sorted(trips):
                        trip_data = result_df[result_df['Trip'] == trip_num]
                        vehicle = trip_vehicles.get(trip_num, '4W')
                        total_cube = trip_data['Cube'].sum() if 'Cube' in trip_data.columns else 0
                        total_weight = trip_data['Weight'].sum() if 'Weight' in trip_data.columns else 0
                        
                        with st.expander(f"🚛 Trip {trip_num} | {vehicle} | {len(trip_data)} สาขา | {total_cube:.1f} คิว | {total_weight:.0f} kg"):
                            st.dataframe(trip_data[['Code', 'Name', 'Cube', 'Weight', 'Distance_DC']].reset_index(drop=True))
                    
                    # Download
                    st.subheader("📥 ดาวน์โหลดผลลัพธ์")
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        result_df.to_excel(writer, sheet_name='Result', index=False)
                    
                    st.download_button(
                        label="⬇️ ดาวน์โหลด Excel",
                        data=output.getvalue(),
                        file_name=f"trip_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
