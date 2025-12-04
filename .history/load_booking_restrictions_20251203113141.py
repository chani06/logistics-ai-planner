# -*- coding: utf-8 -*-
"""โหลดข้อจำกัดรถจาก Booking History (แทนที่ Punthai)"""
import pandas as pd

@st.cache_data
def load_booking_history_restrictions():
    """โหลดประวัติการจัดส่งจาก Booking History - ข้อมูลจริงจำนวนมาก"""
    try:
        file_path = 'Dc/ประวัติงานจัดส่ง DC วังน้อย(1).xlsx'
        df = pd.read_excel(file_path)
        
        # แปลงประเภทรถ
        vehicle_mapping = {
            '4 ล้อ จัมโบ้ ตู้ทึบ': 'JB',
            '6 ล้อ ตู้ทึบ': '6W',
            '4 ล้อ ตู้ทึบ': '4W'
        }
        df['Vehicle_Type'] = df['ประเภทรถ'].map(vehicle_mapping)
        
        # วิเคราะห์ความสัมพันธ์สาขา-รถ
        branch_vehicle_history = {}
        booking_groups = df.groupby('Booking No')
        
        for booking_no, booking_data in booking_groups:
            vehicle_types = booking_data['Vehicle_Type'].dropna().unique()
            if len(vehicle_types) > 0:
                vehicle = booking_data['Vehicle_Type'].mode()[0] if len(booking_data['Vehicle_Type'].mode()) > 0 else vehicle_types[0]
                for branch_code in booking_data['รหัสสาขา'].dropna().unique():
                    if branch_code not in branch_vehicle_history:
                        branch_vehicle_history[branch_code] = []
                    branch_vehicle_history[branch_code].append(vehicle)
        
        # สร้าง restrictions
        branch_restrictions = {}
        vehicle_sizes = {'4W': 1, 'JB': 2, '6W': 3}
        
        for branch_code, vehicle_list in branch_vehicle_history.items():
            vehicles_used = set(vehicle_list)
            vehicle_counts = pd.Series(vehicle_list).value_counts().to_dict()
            
            if len(vehicles_used) == 1:
                # STRICT
                vehicle = list(vehicles_used)[0]
                branch_restrictions[str(branch_code)] = {
                    'max_vehicle': vehicle,
                    'allowed': [vehicle],
                    'history': vehicle_counts,
                    'total_bookings': len(vehicle_list),
                    'restriction_type': 'STRICT',
                    'source': 'BOOKING_HISTORY'
                }
            else:
                # FLEXIBLE
                max_vehicle = max(vehicles_used, key=lambda v: vehicle_sizes.get(v, 0))
                branch_restrictions[str(branch_code)] = {
                    'max_vehicle': max_vehicle,
                    'allowed': list(vehicles_used),
                    'history': vehicle_counts,
                    'total_bookings': len(vehicle_list),
                    'restriction_type': 'FLEXIBLE',
                    'source': 'BOOKING_HISTORY'
                }
        
        stats = {
            'total_branches': len(branch_restrictions),
            'strict_restrictions': len([b for b, r in branch_restrictions.items() if r['restriction_type'] == 'STRICT']),
            'flexible': len([b for b, r in branch_restrictions.items() if r['restriction_type'] == 'FLEXIBLE']),
            'total_bookings': len(booking_groups),
            'total_records': len(df)
        }
        
        return {
            'branch_restrictions': branch_restrictions,
            'stats': stats
        }
    except Exception as e:
        print(f"Error loading booking history: {e}")
        return {'branch_restrictions': {}, 'stats': {}}
