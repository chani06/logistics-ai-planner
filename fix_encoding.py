"""แก้ไข encoding ของไฟล์ให้เป็นภาษาไทยที่ถูกต้อง"""

# อ่านไฟล์ที่มี encoding ผิด
with open('app_backup_original.py', 'rb') as f:
    content_bytes = f.read()

# ลองแปลงจาก Windows-1252/Latin-1 เป็น UTF-8
try:
    # อ่านเป็น Windows-1252 แล้วแปลงเป็น UTF-8
    content_text = content_bytes.decode('windows-1252')
    
    # บันทึกเป็น UTF-8
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content_text)
    
    print("✅ แก้ไข encoding สำเร็จ!")
    print(f"   ไฟล์: app.py")
    print(f"   ขนาด: {len(content_text):,} ตัวอักษร")
    
except Exception as e:
    print(f"❌ เกิดข้อผิดพลาด: {e}")
