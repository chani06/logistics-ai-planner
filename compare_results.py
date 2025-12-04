"""
เปรียบเทียบผลลัพธ์ AI กับแผน Punthai
คำนวณค่า Accuracy และ Metrics ต่างๆ
"""

import pandas as pd
import numpy as np
from itertools import combinations
from collections import defaultdict

def load_data():
    """โหลดข้อมูล AI Result และ Punthai"""
    print("📂 โหลดข้อมูล...")
    
    # โหลดผลลัพธ์ AI (Export จาก Streamlit)
    try:
        ai_file = 'Dc/AI_Result_Output.xlsx'
        df_ai = pd.read_excel(ai_file)
        print(f"   ✅ AI Result: {len(df_ai)} สาขา, {df_ai['Trip'].nunique()} ทริป")
    except:
        print(f"   ❌ ไม่พบไฟล์ {ai_file}")
        print("   💡 กรุณา Export ผลลัพธ์จาก Streamlit ก่อน")
        return None, None
    
    # โหลด Punthai
    try:
        punthai_file = 'Dc/Punthai_reference.xlsx'
        df_punthai = pd.read_excel(punthai_file)
        print(f"   ✅ Punthai: {len(df_punthai)} สาขา, {df_punthai['Trip'].nunique()} ทริป")
    except:
        print(f"   ❌ ไม่พบไฟล์ {punthai_file}")
        return None, None
    
    return df_ai, df_punthai

def calculate_trip_matching_accuracy(df_ai, df_punthai):
    """
    คำนวณ Trip Matching Accuracy
    ตรวจสอบว่าสาขาที่ควรอยู่ด้วยกัน อยู่ด้วยกันจริงหรือไม่
    """
    print("\n🔍 Trip Matching Accuracy")
    print("-" * 60)
    
    # สร้าง dict: code -> trip
    punthai_trips = dict(zip(df_punthai['Code'], df_punthai['Trip']))
    ai_trips = dict(zip(df_ai['Code'], df_ai['Trip']))
    
    # หา common branches
    common_codes = set(punthai_trips.keys()) & set(ai_trips.keys())
    print(f"   สาขาที่เหมือนกัน: {len(common_codes)}/{len(punthai_trips)}")
    
    # สร้าง pairs ที่ควรอยู่ด้วยกัน (จาก Punthai)
    punthai_pairs = set()
    for trip in df_punthai['Trip'].unique():
        codes = df_punthai[df_punthai['Trip'] == trip]['Code'].tolist()
        if len(codes) > 1:
            punthai_pairs.update(combinations(sorted(codes), 2))
    
    # สร้าง pairs จาก AI
    ai_pairs = set()
    for trip in df_ai['Trip'].unique():
        codes = df_ai[df_ai['Trip'] == trip]['Code'].tolist()
        if len(codes) > 1:
            ai_pairs.update(combinations(sorted(codes), 2))
    
    # หา pairs ที่ตรงกัน
    correct_pairs = punthai_pairs & ai_pairs
    
    accuracy = len(correct_pairs) / len(punthai_pairs) * 100 if punthai_pairs else 0
    
    print(f"   Punthai Pairs: {len(punthai_pairs)}")
    print(f"   AI Pairs: {len(ai_pairs)}")
    print(f"   Correct Pairs: {len(correct_pairs)}")
    print(f"   ✅ Accuracy: {accuracy:.2f}%")
    
    return accuracy

def calculate_vehicle_accuracy(df_ai, df_punthai):
    """คำนวณความถูกต้องของการเลือกรถ"""
    print("\n🚛 Vehicle Assignment Accuracy")
    print("-" * 60)
    
    if 'Vehicle_Type' not in df_ai.columns or 'Vehicle_Type' not in df_punthai.columns:
        print("   ⚠️  ไม่มีข้อมูล Vehicle_Type")
        return 0
    
    # เปรียบเทียบแบบ branch-level
    merged = df_ai[['Code', 'Vehicle_Type']].merge(
        df_punthai[['Code', 'Vehicle_Type']], 
        on='Code', 
        suffixes=('_ai', '_punthai')
    )
    
    correct = (merged['Vehicle_Type_ai'] == merged['Vehicle_Type_punthai']).sum()
    accuracy = correct / len(merged) * 100
    
    print(f"   สาขาที่ใช้รถถูกต้อง: {correct}/{len(merged)}")
    print(f"   ✅ Accuracy: {accuracy:.2f}%")
    
    # แสดงรายละเอียดการใช้รถ
    print(f"\n   การใช้รถ:")
    for vehicle in ['4W', 'JB', '6W']:
        ai_count = (df_ai['Vehicle_Type'] == vehicle).sum()
        punthai_count = (df_punthai['Vehicle_Type'] == vehicle).sum()
        print(f"   - {vehicle}: AI={ai_count}, Punthai={punthai_count}")
    
    return accuracy

def calculate_branch_count_mae(df_ai, df_punthai):
    """คำนวณ MAE ของจำนวนสาขาต่อทริป"""
    print("\n📊 Branch Count per Trip (MAE)")
    print("-" * 60)
    
    ai_counts = df_ai.groupby('Trip').size().values
    punthai_counts = df_punthai.groupby('Trip').size().values
    
    # ถ้าจำนวนทริปไม่เท่ากัน ให้ใช้ค่าเฉลี่ย
    if len(ai_counts) != len(punthai_counts):
        print(f"   จำนวนทริปไม่เท่ากัน: AI={len(ai_counts)}, Punthai={len(punthai_counts)}")
        mae = abs(np.mean(ai_counts) - np.mean(punthai_counts))
    else:
        mae = np.mean(np.abs(ai_counts - punthai_counts))
    
    print(f"   AI เฉลี่ย: {np.mean(ai_counts):.1f} สาขา/ทริป")
    print(f"   Punthai เฉลี่ย: {np.mean(punthai_counts):.1f} สาขา/ทริป")
    print(f"   ✅ MAE: {mae:.2f} สาขา")
    
    return mae

def main():
    print("=" * 80)
    print("🧪 เปรียบเทียบผลลัพธ์ AI vs Punthai")
    print("=" * 80)
    
    # โหลดข้อมูล
    df_ai, df_punthai = load_data()
    if df_ai is None or df_punthai is None:
        return
    
    # คำนวณ metrics
    trip_accuracy = calculate_trip_matching_accuracy(df_ai, df_punthai)
    vehicle_accuracy = calculate_vehicle_accuracy(df_ai, df_punthai)
    branch_mae = calculate_branch_count_mae(df_ai, df_punthai)
    
    # สรุปผลรวม
    print("\n" + "=" * 80)
    print("📈 สรุปผลการทดสอบ")
    print("=" * 80)
    print(f"   1. Trip Matching Accuracy: {trip_accuracy:.2f}%")
    print(f"   2. Vehicle Accuracy: {vehicle_accuracy:.2f}%")
    print(f"   3. Branch Count MAE: {branch_mae:.2f} สาขา")
    
    # คะแนนรวม
    overall_score = (trip_accuracy + vehicle_accuracy) / 2
    print(f"\n   🎯 Overall Score: {overall_score:.2f}%")
    
    # ประเมินผล
    if overall_score >= 90:
        grade = "🌟 ยอดเยี่ยม (Excellent)"
    elif overall_score >= 80:
        grade = "✅ ดีมาก (Very Good)"
    elif overall_score >= 70:
        grade = "👍 ดี (Good)"
    elif overall_score >= 60:
        grade = "⚠️  ปานกลาง (Fair)"
    else:
        grade = "❌ ต้องปรับปรุง (Needs Improvement)"
    
    print(f"   {grade}")
    print("=" * 80)

if __name__ == "__main__":
    main()
