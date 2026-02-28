"""แก้ไข distance_cache.json ที่เสียหาย"""
import json

print("🔧 แก้ไข distance_cache.json...")

# อ่านไฟล์ทีละบรรทัด
with open('distance_cache.json', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"📄 อ่านไฟล์: {len(lines)} บรรทัด")

# หาบรรทัดที่มีปัญหา (จบด้วย ": " หรือ ", \n" แต่ไม่มีค่า)
fixed_lines = []
skip_next = False
removed = 0

for i, line in enumerate(lines):
    if skip_next:
        skip_next = False
        continue
    
    # ตรวจสอบว่าบรรทัดนี้จบด้วย ": " (ไม่มีค่า)
    if line.strip().endswith(': '):
        print(f"⚠️ บรรทัด {i+1}: {line.strip()}")
        removed += 1
        continue
    
    # ตรวจสอบว่าบรรทัดก่อนหน้า } มี comma หรือไม่
    if line.strip() == '}' and fixed_lines and fixed_lines[-1].strip().endswith(','):
        # ลบ comma ออกจากบรรทัดก่อนหน้า
        fixed_lines[-1] = fixed_lines[-1].rstrip(',\n') + '\n'
    
    fixed_lines.append(line)

print(f"✅ ลบบรรทัดที่มีปัญหา: {removed} บรรทัด")
print(f"📝 เหลือ: {len(fixed_lines)} บรรทัด")

# บันทึกไฟล์ใหม่
with open('distance_cache_fixed.json', 'w', encoding='utf-8') as f:
    f.writelines(fixed_lines)

# ทดสอบว่าเป็น JSON ที่ถูกต้องหรือไม่
try:
    with open('distance_cache_fixed.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"✅ ไฟล์ใหม่ถูกต้อง: {len(data):,} รายการ")
    
    # แทนที่ไฟล์เดิม
    import shutil
    shutil.copy('distance_cache.json', 'distance_cache_backup.json')
    shutil.copy('distance_cache_fixed.json', 'distance_cache.json')
    print("✅ บันทึกเรียบร้อย: distance_cache.json")
    print("✅ สำรองไฟล์เดิม: distance_cache_backup.json")
except json.JSONDecodeError as e:
    print(f"❌ ยังมีปัญหา: {e}")
    print("💡 ลองตรวจสอบด้วยตาที่บรรทัดที่มีปัญหา")
