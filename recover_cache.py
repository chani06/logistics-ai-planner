"""สร้าง distance_cache.json ใหม่จาก backup หรือข้อมูลที่มี"""
import json
import re

print("🔧 กู้คืน distance_cache.json...")

# อ่านไฟล์แบบ line-by-line และสร้าง dict ใหม่
cache_dict = {}
error_count = 0

with open('distance_cache.json', 'r', encoding='utf-8') as f:
    content = f.read()

# ใช้ regex หาทุกคู่ key-value ที่สมบูรณ์
pattern = r'"([^"]+)":\s*([0-9.]+)'
matches = re.findall(pattern, content)

print(f"📊 พบข้อมูล: {len(matches)} รายการ")

for key, value in matches:
    try:
        cache_dict[key] = float(value)
    except ValueError:
        error_count += 1

print(f"✅ กู้คืนได้: {len(cache_dict):,} รายการ")
if error_count > 0:
    print(f"⚠️ ข้ามรายการที่เสีย: {error_count} รายการ")

# บันทึกเป็น JSON ใหม่
with open('distance_cache_recovered.json', 'w', encoding='utf-8') as f:
    json.dump(cache_dict, f, ensure_ascii=False, indent=2)

print("✅ บันทึก: distance_cache_recovered.json")

# แทนที่ไฟล์เดิม
import shutil
import os
if os.path.exists('distance_cache_backup.json'):
    print("⚠️ มี backup อยู่แล้ว")
else:
    shutil.copy('distance_cache.json', 'distance_cache_backup.json')
    print("✅ สำรอง: distance_cache_backup.json")

shutil.copy('distance_cache_recovered.json', 'distance_cache.json')
print("✅ แทนที่: distance_cache.json")
print(f"🎉 เสร็จสิ้น! มีข้อมูล {len(cache_dict):,} รายการ")
