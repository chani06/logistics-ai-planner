#!/usr/bin/env python
"""
สคริปต์สำหรับลบฟังก์ชันที่ไม่ได้ใช้งานออกจาก app.py
"""

import re

def remove_function(content, function_name, next_function_name=None):
    """ลบฟังก์ชันออกจากโค้ด"""
    # หา pattern ของฟังก์ชัน
    if next_function_name:
        # ลบจากฟังก์ชันปัจจุบันจนถึงก่อนฟังก์ชันถัดไป
        pattern = rf'(def {function_name}\([^)]*\):.*?)(?=def {next_function_name}\()'
    else:
        # ลบจากฟังก์ชันปัจจุบันจนถึงบรรทัดว่าง 2 บรรทัด
        pattern = rf'(def {function_name}\([^)]*\):.*?)(?=\n\n[a-zA-Z@])'
    
    result = re.sub(pattern, '', content, flags=re.DOTALL)
    return result

def main():
    # อ่านไฟล์
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Original size:", len(content), "characters")
    
    # รายการฟังก์ชันที่ต้องลบ (เรียงตามลำดับในไฟล์)
    functions_to_remove = [
        ('is_punthai_only', 'get_punthai_drop_limit'),
        ('get_punthai_drop_limit', 'load_booking_history_restrictions'),
        ('can_fit_truck', 'calculate_optimal_vehicle_split'),
        ('calculate_optimal_vehicle_split', 'can_branch_use_vehicle'),
        ('can_branch_use_vehicle', 'get_max_vehicle_for_branch_old'),
        ('get_max_vehicle_for_branch_old', 'get_most_used_vehicle_for_branch'),
        ('get_most_used_vehicle_for_branch', 'is_similar_name'),
        ('is_similar_name', 'get_region_name'),
    ]
    
    # ลบฟังก์ชันทีละตัว
    for func_name, next_func in functions_to_remove:
        print(f"Removing {func_name}()...")
        content = remove_function(content, func_name, next_func)
    
    print("New size:", len(content), "characters")
    print("Saved:", len(content) - len(content), "characters")
    
    # บันทึกไฟล์
    with open('app_cleaned.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("\nCleaned file saved as app_cleaned.py")
    print("Please review the changes before replacing app.py")

if __name__ == '__main__':
    main()
