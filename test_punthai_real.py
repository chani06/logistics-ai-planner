"""
ทดสอบระบบด้วยไฟล์ Punthai จริง
รันระบบ predict_trips แล้ววิเคราะห์ utilization
"""

import sys
import os

# แก้ปัญหา Unicode บน Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from pathlib import Path

# Import functions from app
try:
    from app import predict_trips, LIMITS, load_booking_history_restrictions
    print("OK: โหลด functions จาก app.py สำเร็จ")
except Exception as e:
    print(f"ERROR: ไม่สามารถโหลด app.py: {e}")
    print("   กรุณารันสคริปต์นี้ในโฟลเดอร์เดียวกับ app.py")
    sys.exit(1)


def calculate_utilization(weight, cube, vehicle):
    """คำนวณ % การใช้รถ"""
    if vehicle not in LIMITS:
        return 0, 0, 0
    
    w_util = (weight / LIMITS[vehicle]['max_w']) * 100
    c_util = (cube / LIMITS[vehicle]['max_c']) * 100
    max_util = max(w_util, c_util)
    
    return w_util, c_util, max_util


def main():
    print("\n" + "🚛" * 40)
    print(" " * 10 + "ทดสอบระบบด้วยไฟล์ Punthai (Dc/test.xlsx)")
    print(" " * 15 + "เป้าหมาย Utilization: 95-130%")
    print("🚛" * 40)
    
    # โหลดไฟล์ทดสอบ
    test_file = Path('Dc/test.xlsx')
    
    if not test_file.exists():
        print(f"\n❌ ไม่พบไฟล์: {test_file}")
        return 1
    
    print(f"\n📂 โหลดไฟล์: {test_file}")
    print("   ชีต: 2.Punthai")
    
    try:
        # อ่านข้อมูล
        df = pd.read_excel(test_file, sheet_name='2.Punthai', header=1)
        
        # เลือกคอลัมน์ที่จำเป็น
        # BranchCode, TOTALCUBE, TOTALWGT
        required_cols = ['BranchCode', 'TOTALCUBE', 'TOTALWGT']
        
        for col in required_cols:
            if col not in df.columns:
                print(f"❌ ไม่พบคอลัมน์: {col}")
                print(f"   คอลัมน์ที่มี: {df.columns.tolist()}")
                return 1
        
        # กรองข้อมูลที่มีค่า
        df = df[pd.notna(df['BranchCode'])].copy()
        df = df[df['TOTALCUBE'] > 0].copy()
        
        print(f"✅ โหลดข้อมูล: {len(df)} สาขา")
        print(f"   น้ำหนักรวม: {df['TOTALWGT'].sum():,.1f} kg")
        print(f"   ปริมาตรรวม: {df['TOTALCUBE'].sum():,.2f} m³")
        
        # เตรียมข้อมูลสำหรับ predict_trips
        test_df = pd.DataFrame({
            'Code': df['BranchCode'].values,
            'Weight': df['TOTALWGT'].values,
            'Cube': df['TOTALCUBE'].values,
            'Trip': 0  # ยังไม่ได้จัดทริป
        })
        
        print("\n🤖 รันระบบวางแผนทริป...")
        
        # โหลดข้อมูลประวัติ
        try:
            model_data = {
                'model': None,
                'trip_pairs': set(),
                'restrictions': load_booking_history_restrictions()
            }
            print("✅ โหลดข้อมูลข้อจำกัดสาขา")
        except Exception as e:
            print(f"⚠️  ไม่สามารถโหลดข้อมูลข้อจำกัด: {e}")
            model_data = {'model': None, 'trip_pairs': set(), 'restrictions': {}}
        
        # รันระบบ
        result_df, diagnostics = predict_trips(test_df, model_data)
        
        if result_df is None or len(result_df) == 0:
            print("❌ ระบบไม่สามารถสร้างทริปได้")
            print("   ตรวจสอบว่า predict_trips ทำงานถูกต้อง")
            return 1
        
        # วิเคราะห์ผลลัพธ์
        print("\n" + "=" * 80)
        print("📊 ผลการวางแผนทริป")
        print("=" * 80)
        
        trips = result_df[result_df['Trip'] > 0]
        num_trips = trips['Trip'].nunique()
        
        print(f"จำนวนทริป: {num_trips}")
        print(f"จำนวนสาขา: {len(trips)}")
        
        # ตรวจสอบว่ามีทริปหรือไม่
        if num_trips == 0 or len(trips) == 0:
            print("\n❌ ไม่มีทริปที่ถูกสร้าง!")
            print(f"   สาขาทั้งหมด: {len(result_df)}")
            print(f"   สาขาที่มี Trip > 0: {len(trips)}")
            print(f"\n   ตัวอย่างข้อมูล result_df:")
            print(result_df[['Code', 'Weight', 'Cube', 'Trip']].head(10))
            print(f"\n   ค่า Trip ที่พบ: {result_df['Trip'].unique()}")
            return 1
        
        # วิเคราะห์แต่ละทริป
        trip_stats = []
        
        for trip_num in sorted(trips['Trip'].unique()):
            trip_data = trips[trips['Trip'] == trip_num]
            
            total_w = trip_data['Weight'].sum()
            total_c = trip_data['Cube'].sum()
            branch_count = len(trip_data)
            
            # คำนวณ utilization สำหรับรถแต่ละประเภท
            util_4w = calculate_utilization(total_w, total_c, '4W')
            util_jb = calculate_utilization(total_w, total_c, 'JB')
            util_6w = calculate_utilization(total_w, total_c, '6W')
            
            # หารถที่เหมาะสม (95-130%)
            best_vehicle = None
            best_util = 0
            
            for vehicle, (w_u, c_u, max_u) in [('4W', util_4w), ('JB', util_jb), ('6W', util_6w)]:
                if 95 <= max_u <= 130 and branch_count <= LIMITS[vehicle]['max_branches']:
                    if best_vehicle is None or abs(max_u - 112.5) < abs(best_util - 112.5):
                        best_vehicle = vehicle
                        best_util = max_u
            
            if best_vehicle is None:
                for vehicle, (w_u, c_u, max_u) in [('4W', util_4w), ('JB', util_jb), ('6W', util_6w)]:
                    if branch_count <= LIMITS[vehicle]['max_branches']:
                        if best_vehicle is None or max_u > best_util:
                            best_vehicle = vehicle
                            best_util = max_u
            
            trip_stats.append({
                'trip': trip_num,
                'branches': branch_count,
                'weight': total_w,
                'cube': total_c,
                'vehicle': best_vehicle,
                'util': best_util,
                '4w': util_4w[2],
                'jb': util_jb[2],
                '6w': util_6w[2]
            })
        
        # แสดงผลแต่ละทริป
        print(f"\n{'Trip':<6} {'สาขา':<6} {'รถ':<6} {'น้ำหนัก':<10} {'ปริมาตร':<10} {'%ใช้':<8} {'สถานะ':<20}")
        print("-" * 80)
        
        optimal_count = 0
        under_count = 0
        over_count = 0
        
        for stat in trip_stats[:20]:  # แสดง 20 ทริปแรก
            status = ""
            if stat['util'] < 75:
                status = "⚠️  รถเหลือมาก"
                under_count += 1
            elif stat['util'] < 95:
                status = "⚠️  รถเหลือ"
                under_count += 1
            elif stat['util'] <= 130:
                status = "✅ เหมาะสม"
                optimal_count += 1
            elif stat['util'] <= 140:
                status = "⚠️  เต็มเกินไป"
                over_count += 1
            else:
                status = "❌ เกินขีดจำกัด"
                over_count += 1
            
            print(f"{stat['trip']:<6} {stat['branches']:<6} {stat['vehicle']:<6} "
                  f"{stat['weight']:<10.1f} {stat['cube']:<10.2f} "
                  f"{stat['util']:<8.1f} {status:<20}")
        
        if len(trip_stats) > 20:
            print(f"... และอีก {len(trip_stats) - 20} ทริป")
        
        # สรุปผล
        print("\n" + "=" * 80)
        print("📈 สรุปการกระจาย Utilization")
        print("=" * 80)
        
        for stat in trip_stats[20:]:
            if stat['util'] < 95:
                under_count += 1
            elif stat['util'] <= 130:
                optimal_count += 1
            else:
                over_count += 1
        
        total = len(trip_stats)
        
        # ป้องกัน division by zero
        if total == 0:
            print("❌ ไม่มีทริปให้วิเคราะห์")
            return 1
        
        optimal_pct = (optimal_count / total) * 100
        
        print(f"✅ ทริปเหมาะสม (95-130%): {optimal_count}/{total} ({optimal_pct:.1f}%)")
        print(f"⚠️  ทริปต่ำ (<95%): {under_count}/{total} ({under_count/total*100:.1f}%)")
        print(f"⚠️  ทริปสูง (>130%): {over_count}/{total} ({over_count/total*100:.1f}%)")
        
        # แยกตามประเภทรถ
        print("\n" + "=" * 80)
        print("📊 สรุปตามประเภทรถ")
        print("=" * 80)
        
        for vehicle in ['4W', 'JB', '6W']:
            vehicle_trips = [s for s in trip_stats if s['vehicle'] == vehicle]
            if vehicle_trips:
                count = len(vehicle_trips)
                avg_util = np.mean([s['util'] for s in vehicle_trips])
                optimal = sum(1 for s in vehicle_trips if 95 <= s['util'] <= 130)
                print(f"{vehicle}: {count} ทริป, เฉลี่ย {avg_util:.1f}%, เหมาะสม {optimal}/{count} ({optimal/count*100:.1f}%)")
        
        # ผลการทดสอบ
        print("\n" + "=" * 80)
        if optimal_pct >= 70:
            print(f"🎉 ผ่านการทดสอบ!")
            print(f"   ✅ {optimal_pct:.1f}% ของทริปอยู่ในช่วงเหมาะสม (เป้าหมาย ≥70%)")
        else:
            print(f"⚠️  ไม่ผ่านการทดสอบ")
            print(f"   ❌ {optimal_pct:.1f}% ของทริปอยู่ในช่วงเหมาะสม (เป้าหมาย ≥70%)")
        
        over_140 = sum(1 for s in trip_stats if s['util'] > 140)
        if over_140 == 0:
            print("   ✅ ไม่มีทริปที่เกิน 140%")
        else:
            print(f"   ❌ มี {over_140} ทริปที่เกิน 140% (ต้องแยกทริป)")
        
        print("=" * 80 + "\n")
        
        # บันทึกผลลัพธ์
        output_file = 'test_result_utilization.xlsx'
        result_df.to_excel(output_file, index=False)
        print(f"💾 บันทึกผลลัพธ์: {output_file}")
        
        return 0 if optimal_pct >= 70 and over_140 == 0 else 1
        
    except Exception as e:
        print(f"\n❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
