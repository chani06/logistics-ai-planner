"""
ทดสอบการกระจาย % การใช้รถ (Utilization)
เป้าหมาย: 95-130% (ต่ำสุด 75%, ห้ามเกิน 140%)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ข้อมูลรถ
LIMITS = {
    '4W': {'max_w': 2500, 'max_c': 5.0, 'max_branches': 12},
    'JB': {'max_w': 3500, 'max_c': 8.0, 'max_branches': 12},
    '6W': {'max_w': 5500, 'max_c': 20.0, 'max_branches': 999}
}

def calculate_utilization(weight, cube, vehicle):
    """คำนวณ % การใช้รถ"""
    if vehicle not in LIMITS:
        return 0, 0, 0
    
    w_util = (weight / LIMITS[vehicle]['max_w']) * 100
    c_util = (cube / LIMITS[vehicle]['max_c']) * 100
    max_util = max(w_util, c_util)
    
    return w_util, c_util, max_util


def analyze_trip_file(file_path, sheet_name=None, header_row=0):
    """วิเคราะห์ไฟล์ผลลัพธ์ทริป"""
    
    print("\n" + "=" * 80)
    if sheet_name:
        print(f"📂 วิเคราะห์ไฟล์: {file_path.name} (ชีต: {sheet_name})")
    else:
        print(f"📂 วิเคราะห์ไฟล์: {file_path.name}")
    print("=" * 80)
    
    try:
        # อ่านไฟล์
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
        else:
            df = pd.read_excel(file_path)
        
        # ตรวจสอบคอลัมน์ที่จำเป็น - รองรับหลายชื่อ
        trip_col = None
        weight_col = None
        cube_col = None
        
        # หา Trip column
        for col in df.columns:
            col_lower = str(col).lower()
            if 'trip' in col_lower and trip_col is None:
                trip_col = col
            if 'weight' in col_lower or 'wgt' in col_lower or 'น้ำหนัก' in col_lower:
                weight_col = col
            if 'cube' in col_lower or 'ปริมาตร' in col_lower:
                cube_col = col
        
        if not all([trip_col, weight_col, cube_col]):
            missing = []
            if not trip_col: missing.append('Trip')
            if not weight_col: missing.append('Weight/WGT')
            if not cube_col: missing.append('Cube')
            print(f"❌ ไม่พบคอลัมน์: {missing}")
            print(f"   คอลัมน์ที่มี: {df.columns.tolist()[:10]}")
            return None
        
        # กรองเฉพาะทริปที่มีข้อมูล (Trip > 0)
        df = df[pd.notna(df[trip_col]) & (df[trip_col] > 0)].copy()
        
        if len(df) == 0:
            print("❌ ไม่มีข้อมูลทริป")
            return None
        
        print(f"\n📊 จำนวนทริปทั้งหมด: {df[trip_col].nunique()}")
        print(f"📦 จำนวนสาขาทั้งหมด: {len(df)}")
        
        # วิเคราะห์แต่ละทริป
        trip_stats = []
        
        for trip_num in sorted(df[trip_col].unique()):
            trip_data = df[df[trip_col] == trip_num]
            
            total_w = trip_data[weight_col].sum()
            total_c = trip_data[cube_col].sum()
            branch_count = len(trip_data)
            
            # ลองคำนวณ % สำหรับรถแต่ละประเภท
            util_4w = calculate_utilization(total_w, total_c, '4W')
            util_jb = calculate_utilization(total_w, total_c, 'JB')
            util_6w = calculate_utilization(total_w, total_c, '6W')
            
            # หารถที่เหมาะสมที่สุด (95-130%)
            best_vehicle = None
            best_util = 0
            
            for vehicle, (w_u, c_u, max_u) in [('4W', util_4w), ('JB', util_jb), ('6W', util_6w)]:
                if 95 <= max_u <= 130 and branch_count <= LIMITS[vehicle]['max_branches']:
                    if best_vehicle is None or abs(max_u - 112.5) < abs(best_util - 112.5):
                        best_vehicle = vehicle
                        best_util = max_u
            
            # ถ้าไม่มีรถที่พอดี เลือกรถที่ใกล้เคียงที่สุด
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
                '4w_w': util_4w[0],
                '4w_c': util_4w[1],
                '4w_max': util_4w[2],
                'jb_w': util_jb[0],
                'jb_c': util_jb[1],
                'jb_max': util_jb[2],
                '6w_w': util_6w[0],
                '6w_c': util_6w[1],
                '6w_max': util_6w[2],
                'best_util': best_util
            })
        
        stats_df = pd.DataFrame(trip_stats)
        
        return stats_df
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        return None


def analyze_utilization_distribution(stats_df):
    """วิเคราะห์การกระจาย % การใช้รถ"""
    
    print("\n" + "=" * 80)
    print("📊 การวิเคราะห์การกระจาย Utilization")
    print("=" * 80)
    
    # แยกตามประเภทรถ
    vehicle_summary = {}
    
    for vehicle in ['4W', 'JB', '6W']:
        vehicle_trips = stats_df[stats_df['vehicle'] == vehicle]
        
        if len(vehicle_trips) == 0:
            continue
        
        utils = vehicle_trips['best_util'].values
        
        vehicle_summary[vehicle] = {
            'count': len(vehicle_trips),
            'min': utils.min(),
            'max': utils.max(),
            'mean': utils.mean(),
            'median': np.median(utils),
            'under_75': np.sum(utils < 75),
            'under_95': np.sum((utils >= 75) & (utils < 95)),
            'optimal_95_130': np.sum((utils >= 95) & (utils <= 130)),
            'over_130': np.sum(utils > 130),
            'over_140': np.sum(utils > 140)
        }
    
    # แสดงผล
    print(f"\n{'รถ':<6} {'จำนวน':<8} {'ต่ำสุด':<10} {'สูงสุด':<10} {'เฉลี่ย':<10} {'มัธยฐาน':<10}")
    print("-" * 60)
    
    for vehicle, summary in vehicle_summary.items():
        print(f"{vehicle:<6} {summary['count']:<8} {summary['min']:<10.1f} {summary['max']:<10.1f} "
              f"{summary['mean']:<10.1f} {summary['median']:<10.1f}")
    
    print("\n" + "=" * 80)
    print("📈 การกระจายตามช่วง %")
    print("=" * 80)
    
    all_pass = True
    
    for vehicle, summary in vehicle_summary.items():
        print(f"\n🚛 {vehicle} ({summary['count']} ทริป):")
        
        # < 75% (ต่ำเกินไป)
        if summary['under_75'] > 0:
            pct = (summary['under_75'] / summary['count']) * 100
            print(f"   ⚠️  < 75%: {summary['under_75']:>3} ทริป ({pct:>5.1f}%) - รถเหลือมาก ควรลดขนาด")
            all_pass = False
        
        # 75-94% (พอใช้ได้แต่ไม่เหมาะ)
        if summary['under_95'] > 0:
            pct = (summary['under_95'] / summary['count']) * 100
            print(f"   ⚠️  75-94%: {summary['under_95']:>3} ทริป ({pct:>5.1f}%) - ใช้ได้แต่ยังมีที่ว่าง")
        
        # 95-130% (เป้าหมาย ✅)
        if summary['optimal_95_130'] > 0:
            pct = (summary['optimal_95_130'] / summary['count']) * 100
            print(f"   ✅ 95-130%: {summary['optimal_95_130']:>3} ทริป ({pct:>5.1f}%) - เหมาะสม!")
        
        # 131-140% (เกินเล็กน้อย)
        over_130_not_140 = summary['over_130'] - summary['over_140']
        if over_130_not_140 > 0:
            pct = (over_130_not_140 / summary['count']) * 100
            print(f"   ⚠️  131-140%: {over_130_not_140:>3} ทริป ({pct:>5.1f}%) - เต็มเกินไป")
            all_pass = False
        
        # > 140% (เกินมาก ต้องแยก)
        if summary['over_140'] > 0:
            pct = (summary['over_140'] / summary['count']) * 100
            print(f"   ❌ > 140%: {summary['over_140']:>3} ทริป ({pct:>5.1f}%) - เกินขีดจำกัด ต้องแยกทริป!")
            all_pass = False
    
    # สรุปภาพรวม
    total_trips = sum(s['count'] for s in vehicle_summary.values())
    total_optimal = sum(s['optimal_95_130'] for s in vehicle_summary.values())
    total_under_75 = sum(s['under_75'] for s in vehicle_summary.values())
    total_over_140 = sum(s['over_140'] for s in vehicle_summary.values())
    
    print("\n" + "=" * 80)
    print("📊 สรุปภาพรวม")
    print("=" * 80)
    print(f"ทริปทั้งหมด: {total_trips}")
    print(f"✅ ทริปที่เหมาะสม (95-130%): {total_optimal} ({(total_optimal/total_trips*100):.1f}%)")
    print(f"⚠️  ทริปที่รถเหลือมาก (<75%): {total_under_75} ({(total_under_75/total_trips*100):.1f}%)")
    print(f"❌ ทริปที่เกินขีดจำกัด (>140%): {total_over_140} ({(total_over_140/total_trips*100):.1f}%)")
    
    # เป้าหมาย: อย่างน้อย 70% ต้องอยู่ในช่วง 95-130%
    optimal_pct = (total_optimal / total_trips) * 100
    
    print("\n" + "=" * 80)
    if optimal_pct >= 70:
        print(f"🎉 ผ่าน: {optimal_pct:.1f}% ของทริปอยู่ในช่วงเหมาะสม (เป้าหมาย ≥70%)")
    else:
        print(f"⚠️  ไม่ผ่าน: {optimal_pct:.1f}% ของทริปอยู่ในช่วงเหมาะสม (เป้าหมาย ≥70%)")
        all_pass = False
    
    if total_over_140 == 0:
        print("✅ ผ่าน: ไม่มีทริปที่เกิน 140%")
    else:
        print(f"❌ ไม่ผ่าน: มี {total_over_140} ทริปที่เกิน 140% (ต้องแยกทริป)")
        all_pass = False
    
    print("=" * 80)
    
    return all_pass, vehicle_summary


def show_problem_trips(stats_df):
    """แสดงทริปที่มีปัญหา"""
    
    print("\n" + "=" * 80)
    print("🔍 รายละเอียดทริปที่มีปัญหา")
    print("=" * 80)
    
    # ทริปที่รถเหลือมาก (<75%)
    under_utilized = stats_df[stats_df['best_util'] < 75].copy()
    if len(under_utilized) > 0:
        print(f"\n⚠️  ทริปที่รถเหลือมาก (<75%): {len(under_utilized)} ทริป")
        print(f"{'Trip':<6} {'รถ':<6} {'สาขา':<6} {'น้ำหนัก':<10} {'ปริมาตร':<10} {'%ใช้รถ':<10}")
        print("-" * 60)
        for _, row in under_utilized.head(10).iterrows():
            print(f"{row['trip']:<6} {row['vehicle']:<6} {row['branches']:<6} "
                  f"{row['weight']:<10.1f} {row['cube']:<10.2f} {row['best_util']:<10.1f}")
        if len(under_utilized) > 10:
            print(f"... และอีก {len(under_utilized) - 10} ทริป")
    
    # ทริปที่เกินขีดจำกัด (>140%)
    over_utilized = stats_df[stats_df['best_util'] > 140].copy()
    if len(over_utilized) > 0:
        print(f"\n❌ ทริปที่เกินขีดจำกัด (>140%): {len(over_utilized)} ทริป")
        print(f"{'Trip':<6} {'รถ':<6} {'สาขา':<6} {'น้ำหนัก':<10} {'ปริมาตร':<10} {'%ใช้รถ':<10}")
        print("-" * 60)
        for _, row in over_utilized.head(10).iterrows():
            print(f"{row['trip']:<6} {row['vehicle']:<6} {row['branches']:<6} "
                  f"{row['weight']:<10.1f} {row['cube']:<10.2f} {row['best_util']:<10.1f}")
        if len(over_utilized) > 10:
            print(f"... และอีก {len(over_utilized) - 10} ทริป")


def main():
    """รันการทดสอบทั้งหมด"""
    
    print("\n" + "🚛" * 40)
    print(" " * 15 + "ทดสอบการกระจาย % การใช้รถ (Utilization)")
    print(" " * 20 + "เป้าหมาย: 95-130% (≥70% ของทริป)")
    print("🚛" * 40)
    
    # ตรวจสอบไฟล์ทดสอบเฉพาะ
    test_file = Path('Dc/test.xlsx')
    
    if test_file.exists():
        print(f"\n✅ พบไฟล์ทดสอบ: {test_file}")
        print("   ชีต: 2.Punthai")
        print("   เริ่มจากแถว: 2 (header)")
        
        # วิเคราะห์ไฟล์
        stats_df = analyze_trip_file(test_file, sheet_name='2.Punthai', header_row=1)
        
        if stats_df is not None:
            passed, summary = analyze_utilization_distribution(stats_df)
            show_problem_trips(stats_df)
            
            print("\n" + "=" * 80)
            if passed:
                print("🎉 ผ่านการทดสอบ!")
                print("   ✅ อย่างน้อย 70% ของทริปอยู่ในช่วง 95-130%")
                print("   ✅ ไม่มีทริปที่เกิน 140%")
            else:
                print("⚠️  ไม่ผ่านการทดสอบ")
                print("   กรุณาปรับปรุงการจัดทริปให้อยู่ในช่วง 95-130%")
            print("=" * 80 + "\n")
            
            return 0 if passed else 1
        else:
            print("\n❌ ไม่สามารถวิเคราะห์ไฟล์ได้")
            return 1
    else:
        print(f"\n❌ ไม่พบไฟล์: {test_file}")
        print("   กรุณาตรวจสอบว่าไฟล์อยู่ในโฟลเดอร์ Dc/")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
