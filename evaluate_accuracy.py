import os
import pandas as pd
import numpy as np
from datetime import datetime

def save_accuracy_log(metrics, log_file='accuracy_history.csv'):
    """
    บันทึกผลการประเมินลงในไฟล์ CSV เพื่อติดตามการพัฒนา (Tracking Improvement)
    """
    record = {
        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Total_Actual_Branches': metrics['Total_Actual_Branches'],
        'Total_System_Branches': metrics['Total_System_Branches'],
        'Matched_Branches': metrics['Matched_Branches'],
        'Missing_Branches': metrics['Missing_Branches'],
        'Extra_Branches': metrics['Extra_Branches'],
        'Vehicle_Accuracy_Pct': metrics['Vehicle_Accuracy_Pct'],
        'Avg_Routing_Similarity': metrics['Avg_Routing_Similarity']
    }
    
    df_record = pd.DataFrame([record])
    
    if os.path.exists(log_file):
        df_record.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df_record.to_csv(log_file, mode='w', header=True, index=False)
        
    return log_file

def evaluate_routing_accuracy(actual_df, system_df, act_branch_col='Branch', sys_branch_col='Branch', actual_trip_col='Actual_Trip', actual_vehicle_col='Actual_Vehicle', sys_trip_col='Trip', sys_vehicle_col='Vehicle'):
    """
    เปรียบเทียบความถูกต้องอ้างอิงระดับรายสาขา (Branch-centric)
    """
    # เตรียมข้อมูล Actual
    act_df = actual_df[[act_branch_col, actual_trip_col, actual_vehicle_col]].copy()
    act_df.rename(columns={act_branch_col: 'Branch', actual_trip_col: 'Actual_Trip', actual_vehicle_col: 'Actual_Vehicle'}, inplace=True)
    act_df['Branch'] = act_df['Branch'].astype(str).str.strip()
    act_df = act_df.dropna(subset=['Actual_Trip', 'Branch'])
    act_df = act_df[act_df['Actual_Trip'].astype(str).str.strip() != '']
    
    # หาก 1 สาขามีหลายทริปในไฟล์เดียวกัน (เช่น ส่งเช้า-บ่าย) ให้รวบเป็นบรรทัดแรกไปก่อน เพื่อความง่ายในการเทสเบื้องต้น
    act_df = act_df.drop_duplicates(subset=['Branch'], keep='first')
    
    # เตรียมข้อมูล System
    sys_df = system_df[[sys_branch_col, sys_trip_col, sys_vehicle_col]].copy()
    sys_df.rename(columns={sys_branch_col: 'Branch', sys_trip_col: 'System_Trip', sys_vehicle_col: 'System_Vehicle'}, inplace=True)
    sys_df['Branch'] = sys_df['Branch'].astype(str).str.strip()
    sys_df = sys_df.dropna(subset=['System_Trip', 'Branch'])
    sys_df = sys_df[sys_df['System_Trip'] != 0] # 0 = ไม่ได้จัดทริป
    sys_df = sys_df[sys_df['System_Trip'].astype(str).str.strip() != '']
    
    sys_df = sys_df.drop_duplicates(subset=['Branch'], keep='first')
    
    # รวมข้อมูลโดยยึดสาขาเป็นหลัก (Full Outer Join)
    merged_df = pd.merge(act_df, sys_df, on='Branch', how='outer')
    
    # เติมค่าว่าง
    merged_df.fillna({'Actual_Trip': 'None', 'Actual_Vehicle': 'None', 'System_Trip': 'None', 'System_Vehicle': 'None'}, inplace=True)
    
    # ประเมินสถานะ
    def check_status(row):
        if row['Actual_Trip'] == 'None':
            return 'Extra in System'
        elif row['System_Trip'] == 'None':
            return 'Missing in System'
        else:
            return 'Matched'
            
    merged_df['Status'] = merged_df.apply(check_status, axis=1)
    
    # เทียบประเภทรถ
    merged_df['Vehicle_Match'] = (merged_df['Actual_Vehicle'].astype(str).str.strip() == merged_df['System_Vehicle'].astype(str).str.strip()) & (merged_df['Status'] == 'Matched')
    
    # สร้าง Map เพื่อนร่วมทริป เพื่อเทียบลักษณะทริป
    act_trip_map = act_df.groupby('Actual_Trip')['Branch'].apply(set).to_dict()
    sys_trip_map = sys_df.groupby('System_Trip')['Branch'].apply(set).to_dict()
    
    def calculate_trip_similarity(row):
        if row['Status'] != 'Matched':
            return 0.0
        act_group = act_trip_map.get(row['Actual_Trip'], set())
        sys_group = sys_trip_map.get(row['System_Trip'], set())
        
        intersection = len(act_group.intersection(sys_group))
        union = len(act_group.union(sys_group))
        if union == 0:
            return 0.0
        return round(intersection / union, 4)
        
    merged_df['Group_Similarity'] = merged_df.apply(calculate_trip_similarity, axis=1)
    
    # จัดเรียงคอลัมน์ให้ดูง่าย
    merged_df = merged_df[['Branch', 'Actual_Trip', 'System_Trip', 'Actual_Vehicle', 'System_Vehicle', 'Status', 'Vehicle_Match', 'Group_Similarity']]
    merged_df = merged_df.sort_values(by=['Status', 'Branch'])
    
    # คำนวณ Metrics สรุป
    total_branches_actual = len(act_df['Branch'].unique())
    total_branches_system = len(sys_df['Branch'].unique())
    
    matched_branches = len(merged_df[merged_df['Status'] == 'Matched'])
    missing_branches = len(merged_df[merged_df['Status'] == 'Missing in System'])
    extra_branches = len(merged_df[merged_df['Status'] == 'Extra in System'])
    
    vehicle_matched_count = len(merged_df[merged_df['Vehicle_Match'] == True])
    
    # หาค่าเฉลี่ย Group Similarity (ความเหมือนของการจัดกลุ่ม)
    avg_similarity = merged_df[merged_df['Status'] == 'Matched']['Group_Similarity'].mean()
    if pd.isna(avg_similarity):
        avg_similarity = 0.0
    
    metrics = {
        'Total_Actual_Branches': total_branches_actual,
        'Total_System_Branches': total_branches_system,
        'Matched_Branches': matched_branches,
        'Missing_Branches': missing_branches,
        'Extra_Branches': extra_branches,
        'Vehicle_Match_Count': vehicle_matched_count,
        'Vehicle_Accuracy_Pct': round((vehicle_matched_count / matched_branches) * 100, 2) if matched_branches > 0 else 0.0,
        'Avg_Routing_Similarity': round(avg_similarity * 100, 2)
    }
    
    return metrics, merged_df
