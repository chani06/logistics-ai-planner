"""
ทดสอบการกระจาย % ด้วยการจำลองทริปแบบง่าย
ไม่ใช้ AI model แต่จัดทริปด้วยกฎเบื้องต้น
"""

import sys
import os

# แก้ปัญหา Unicode
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path

# ข้อมูลรถ
LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5.0, 'max_branches': 12},
    'JB': {'max_w': 3500, 'max_c': 8.0, 'max_branches': 12},
    '6W': {'max_w': 5500, 'max_c': 20.0, 'max_branches': 999}
}

def simple_trip_planning(df, target_util=0.95, max_util=1.3):
    """
    จัดทริปแบบง่าย: เติมสาขาเข้าทริปจนถึง target utilization
    """
    # เรียงตามปริมาตรมากไปน้อย
    df = df.sort_values('Cube', ascending=False).reset_index(drop=True)
    
    trips = []
    current_trip = []
    current_weight = 0
    current_cube = 0
    trip_num = 1
    
    for idx, row in df.iterrows():
        code = row['Code']
        weight = row['Weight']
        cube = row['Cube']
        
        # ลองเติมเข้าทริปปัจจุบัน
        new_weight = current_weight + weight
        new_cube = current_cube + cube
        
        # เช็คว่าเกิน 6W หรือไม่ (รถใหญ่สุด)
        w_util_6w = new_weight / LIMITS['6W']['max_w']
        c_util_6w = new_cube / LIMITS['6W']['max_c']
        max_util_6w = max(w_util_6w, c_util_6w)
        
        # ถ้าเกินขีดจำกัด หรือถึงเป้าหมายแล้ว -> เริ่มทริปใหม่
        if max_util_6w > max_util or (len(current_trip) > 0 and max_util_6w > target_util):
            # บันทึกทริปปัจจุบัน
            if current_trip:
                for c in current_trip:
                    trips.append({'Code': c[0], 'Weight': c[1], 'Cube': c[2], 'Trip': trip_num})
                trip_num += 1
            
            # เริ่มทริปใหม่
            current_trip = [(code, weight, cube)]
            current_weight = weight
            current_cube = cube
        else:
            # เติมเข้าทริปปัจจุบัน
            current_trip.append((code, weight, cube))
            current_weight = new_weight
            current_cube = new_cube
    
    # บันทึกทริปสุดท้าย
    if current_trip:
        for c in current_trip:
            trips.append({'Code': c[0], 'Weight': c[1], 'Cube': c[2], 'Trip': trip_num})
    
    return pd.DataFrame(trips)


def calculate_utilization(weight, cube, vehicle):
    """คำนวณ % การใช้รถ"""
    if vehicle not in LIMITS:
        return 0, 0, 0
    
    w_util = (weight / LIMITS[vehicle]['max_w']) * 100
    c_util = (cube / LIMITS[vehicle]['max_c']) * 100
    max_util = max(w_util, c_util)
    
    return w_util, c_util, max_util


def main():
    print("\n" + "="*80)
    print("ทดสอบการกระจาย Utilization ด้วยไฟล์ Punthai")
    print("="*80)
    
    # โหลดไฟล์
    test_file = Path('Dc/test.xlsx')
    
    if not test_file.exists():
        print(f"ERROR: ไม่พบไฟล์ {test_file}")
        return 1
    
    print(f"\nโหลดไฟล์: {test_file}")
    
    try:
        df = pd.read_excel(test_file, sheet_name='2.Punthai', header=1)
        
        # กรองข้อมูล
        df = df[pd.notna(df['BranchCode'])].copy()
        df = df[df['TOTALCUBE'] > 0].copy()
        
        print(f"สาขาทั้งหมด: {len(df)}")
        print(f"น้ำหนักรวม: {df['TOTALWGT'].sum():,.1f} kg")
        print(f"ปริมาตรรวม: {df['TOTALCUBE'].sum():,.2f} m³")
        
        # เตรียมข้อมูล
        input_df = pd.DataFrame({
            'Code': df['BranchCode'].values,
            'Weight': df['TOTALWGT'].values,
            'Cube': df['TOTALCUBE'].values
        })
        
        print("\nจัดทริปแบบง่าย (เป้าหมาย 95-130%)...")
        result_df = simple_trip_planning(input_df, target_util=0.95, max_util=1.3)
        
        num_trips = result_df['Trip'].nunique()
        print(f"จำนวนทริป: {num_trips}")
        
        # วิเคราะห์แต่ละทริป
        print("\n" + "="*80)
        print("วิเคราะห์ Utilization แต่ละทริป")
        print("="*80)
        
        trip_stats = []
        
        for trip_num in sorted(result_df['Trip'].unique()):
            trip_data = result_df[result_df['Trip'] == trip_num]
            
            total_w = trip_data['Weight'].sum()
            total_c = trip_data['Cube'].sum()
            branches = len(trip_data)
            
            # คำนวณ util สำหรับแต่ละรถ
            util_4w = calculate_utilization(total_w, total_c, '4W')
            util_jb = calculate_utilization(total_w, total_c, 'JB')
            util_6w = calculate_utilization(total_w, total_c, '6W')
            
            # เลือกรถที่เหมาะสม
            best_vehicle = None
            best_util = 0
            
            for vehicle, (w, c, m) in [('4W', util_4w), ('JB', util_jb), ('6W', util_6w)]:
                if 95 <= m <= 130 and branches <= LIMITS[vehicle]['max_branches']:
                    if best_vehicle is None or abs(m - 112.5) < abs(best_util - 112.5):
                        best_vehicle = vehicle
                        best_util = m
            
            if best_vehicle is None:
                for vehicle, (w, c, m) in [('4W', util_4w), ('JB', util_jb), ('6W', util_6w)]:
                    if branches <= LIMITS[vehicle]['max_branches']:
                        if best_vehicle is None or m > best_util:
                            best_vehicle = vehicle
                            best_util = m
            
            trip_stats.append({
                'trip': trip_num,
                'branches': branches,
                'weight': total_w,
                'cube': total_c,
                'vehicle': best_vehicle,
                'util': best_util
            })
        
        # แสดงผล
        print(f"\n{'Trip':<6} {'สาขา':<6} {'รถ':<6} {'น้ำหนัก':<10} {'ปริมาตร':<10} {'%ใช้':<8} {'สถานะ':<15}")
        print("-"*80)
        
        optimal_count = 0
        under_count = 0
        over_count = 0
        
        for stat in trip_stats[:30]:
            if stat['util'] < 75:
                status = "รถเหลือมาก"
                under_count += 1
            elif stat['util'] < 95:
                status = "รถเหลือ"
                under_count += 1
            elif stat['util'] <= 130:
                status = "เหมาะสม"
                optimal_count += 1
            elif stat['util'] <= 140:
                status = "เต็มเกินไป"
                over_count += 1
            else:
                status = "เกินขีดจำกัด"
                over_count += 1
            
            print(f"{stat['trip']:<6} {stat['branches']:<6} {stat['vehicle']:<6} "
                  f"{stat['weight']:<10.1f} {stat['cube']:<10.2f} "
                  f"{stat['util']:<8.1f} {status:<15}")
        
        if len(trip_stats) > 30:
            print(f"... และอีก {len(trip_stats) - 30} ทริป")
            
            for stat in trip_stats[30:]:
                if stat['util'] < 95:
                    under_count += 1
                elif stat['util'] <= 130:
                    optimal_count += 1
                else:
                    over_count += 1
        
        # สรุป
        total = len(trip_stats)
        optimal_pct = (optimal_count / total) * 100 if total > 0 else 0
        
        print("\n" + "="*80)
        print("สรุป")
        print("="*80)
        print(f"ทริปเหมาะสม (95-130%): {optimal_count}/{total} ({optimal_pct:.1f}%)")
        print(f"ทริปต่ำ (<95%): {under_count}/{total} ({under_count/total*100:.1f}%)")
        print(f"ทริปสูง (>130%): {over_count}/{total} ({over_count/total*100:.1f}%)")
        
        # ตามประเภทรถ
        print("\nตามประเภทรถ:")
        for vehicle in ['4W', 'JB', '6W']:
            vehicle_trips = [s for s in trip_stats if s['vehicle'] == vehicle]
            if vehicle_trips:
                count = len(vehicle_trips)
                avg = np.mean([s['util'] for s in vehicle_trips])
                opt = sum(1 for s in vehicle_trips if 95 <= s['util'] <= 130)
                print(f"  {vehicle}: {count} ทริป, เฉลี่ย {avg:.1f}%, เหมาะสม {opt}/{count} ({opt/count*100:.1f}%)")
        
        # ผลการทดสอบ
        print("\n" + "="*80)
        over_140 = sum(1 for s in trip_stats if s['util'] > 140)
        
        if optimal_pct >= 70 and over_140 == 0:
            print("PASS: ผ่านการทดสอบ!")
            print(f"  OK: {optimal_pct:.1f}% อยู่ในช่วงเหมาะสม (เป้าหมาย >=70%)")
            print(f"  OK: ไม่มีทริปเกิน 140%")
            return_code = 0
        else:
            print("FAIL: ไม่ผ่านการทดสอบ")
            if optimal_pct < 70:
                print(f"  ERROR: {optimal_pct:.1f}% อยู่ในช่วงเหมาะสม (เป้าหมาย >=70%)")
            if over_140 > 0:
                print(f"  ERROR: มี {over_140} ทริปที่เกิน 140%")
            return_code = 1
        
        print("="*80)
        
        # บันทึก
        output_file = 'test_result_simple.xlsx'
        result_df.to_excel(output_file, index=False)
        print(f"\nบันทึกผลลัพธ์: {output_file}\n")
        
        return return_code
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
