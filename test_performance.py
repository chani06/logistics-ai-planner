"""
Performance Testing Script for Logistics Trip Planning
ทดสอบประสิทธิภาพการจัดทริป
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Import functions from main app
sys.path.insert(0, str(Path(__file__).parent))

print("🚀 กำลังโหลดโมดูลหลัก...")
from app import predict_trips, load_model, MASTER_DATA

def create_test_data(num_branches=100):
    """สร้างข้อมูลทดสอบ"""
    print(f"\n📊 สร้างข้อมูลทดสอบ {num_branches} สาขา...")
    
    # ดึง sample จาก MASTER_DATA
    if not MASTER_DATA.empty and len(MASTER_DATA) >= num_branches:
        # ใช้คอลัมน์ Plan Code แทน Code
        test_df = MASTER_DATA.head(num_branches).copy()
        
        # Rename Plan Code → Code
        if 'Plan Code' in test_df.columns:
            test_df = test_df.rename(columns={'Plan Code': 'Code'})
        
        # เพิ่มคอลัมน์ที่จำเป็น
        test_df['Weight'] = np.random.uniform(100, 1500, num_branches)
        test_df['Cube'] = np.random.uniform(0.5, 5.0, num_branches)
        
        # ใช้ BU จาก Master Data ถ้ามี
        if 'BU' not in test_df.columns:
            test_df['BU'] = np.random.choice(['211', 'MAXMART'], num_branches, p=[0.3, 0.7])
        
        return test_df
    else:
        # สร้างข้อมูล mock ถ้าไม่มี MASTER_DATA
        data = {
            'Code': [f'T{i:04d}' for i in range(num_branches)],
            'Name': [f'สาขาทดสอบ {i+1}' for i in range(num_branches)],
            'Weight': np.random.uniform(100, 1500, num_branches),
            'Cube': np.random.uniform(0.5, 5.0, num_branches),
            'Province': np.random.choice(['กรุงเทพ', 'นนทบุรี', 'ปทุมธานี', 'สมุทรปราการ'], num_branches),
            'District': [f'อำเภอ {i%10}' for i in range(num_branches)],
            'Subdistrict': [f'ตำบล {i%20}' for i in range(num_branches)],
            'BU': np.random.choice(['211', 'MAXMART'], num_branches, p=[0.3, 0.7]),
        }
        return pd.DataFrame(data)

def run_performance_test(test_sizes=[50, 100, 200, 500]):
    """รันการทดสอบประสิทธิภาพ"""
    print("\n" + "="*80)
    print("🧪 เริ่มทดสอบประสิทธิภาพระบบจัดทริป")
    print("="*80)
    
    # โหลดโมเดล
    print("\n📦 โหลดโมเดลและข้อมูล Master...")
    model_data = load_model()
    print(f"✅ โหลดสำเร็จ: {len(MASTER_DATA)} สาขาใน Master Data")
    
    results = []
    
    for size in test_sizes:
        print(f"\n{'─'*80}")
        print(f"📏 ทดสอบขนาด: {size} สาขา")
        print(f"{'─'*80}")
        
        # สร้างข้อมูลทดสอบ
        test_df = create_test_data(size)
        
        # วัดเวลา
        start_time = time.time()
        
        try:
            result_df, summary = predict_trips(
                test_df, 
                model_data,
                punthai_buffer=1.0,
                maxmart_buffer=1.10
            )
            
            elapsed_time = time.time() - start_time
            
            # วิเคราะห์ผลลัพธ์
            total_trips = len(summary)
            assigned_branches = len(result_df[result_df['Trip'] > 0])
            unassigned_branches = len(result_df[result_df['Trip'] == 0])
            avg_branches_per_trip = assigned_branches / total_trips if total_trips > 0 else 0
            
            # แสดงผล
            print(f"\n✅ สำเร็จ!")
            print(f"⏱️  เวลาประมวลผล: {elapsed_time:.2f} วินาที")
            print(f"🚚 จำนวนทริป: {total_trips}")
            print(f"📍 สาขาที่จัดได้: {assigned_branches}/{size} ({assigned_branches/size*100:.1f}%)")
            if unassigned_branches > 0:
                print(f"⚠️  สาขาที่ไม่ได้จัด: {unassigned_branches}")
            print(f"📊 เฉลี่ยสาขา/ทริป: {avg_branches_per_trip:.1f}")
            
            # เก็บผลลัพธ์
            results.append({
                'size': size,
                'time': elapsed_time,
                'trips': total_trips,
                'assigned': assigned_branches,
                'unassigned': unassigned_branches,
                'avg_per_trip': avg_branches_per_trip,
                'speed': size / elapsed_time  # สาขาต่อวินาที
            })
            
        except Exception as e:
            print(f"\n❌ ข้อผิดพลาด: {e}")
            import traceback
            traceback.print_exc()
    
    # สรุปผล
    print("\n" + "="*80)
    print("📊 สรุปผลการทดสอบ")
    print("="*80)
    
    if results:
        results_df = pd.DataFrame(results)
        print("\n" + results_df.to_string(index=False))
        
        # คำนวณค่าเฉลี่ย
        avg_speed = results_df['speed'].mean()
        print(f"\n🏆 ประสิทธิภาพเฉลี่ย: {avg_speed:.1f} สาขา/วินาที")
        
        # ประมาณการสำหรับจำนวนสาขามาก
        for target_size in [1000, 5000, 10000]:
            estimated_time = target_size / avg_speed
            print(f"   • {target_size:,} สาขา → ประมาณ {estimated_time:.1f} วินาที ({estimated_time/60:.1f} นาที)")
    
    print("\n✅ ทดสอบเสร็จสิ้น!")
    print("="*80 + "\n")

def quick_test():
    """ทดสอบแบบเร็ว (50 สาขา)"""
    print("\n⚡ Quick Test: 50 สาขา")
    run_performance_test([50])

def full_test():
    """ทดสอบแบบเต็ม (หลายขนาด)"""
    print("\n🔬 Full Test: หลายขนาด")
    run_performance_test([50, 100, 200, 500])

def stress_test():
    """ทดสอบภาระหนัก"""
    print("\n💪 Stress Test: ขนาดใหญ่")
    run_performance_test([500, 1000])

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        if test_type == 'quick':
            quick_test()
        elif test_type == 'full':
            full_test()
        elif test_type == 'stress':
            stress_test()
        else:
            print(f"❌ Unknown test type: {test_type}")
            print("Usage: python test_performance.py [quick|full|stress]")
    else:
        # Default: quick test
        quick_test()
