# 🔐 Google Sheets API Setup Guide

## วิธีสร้าง credentials.json

### 1. สร้าง Google Cloud Project

1. ไปที่ [Google Cloud Console](https://console.cloud.google.com/)
2. สร้าง Project ใหม่ หรือเลือก Project ที่มีอยู่
3. จดชื่อ Project ID

### 2. Enable APIs

1. ไปที่ **APIs & Services** > **Library**
2. ค้นหาและ Enable:
   - ✅ **Google Sheets API**
   - ✅ **Google Drive API**

### 3. สร้าง Service Account

1. ไปที่ **APIs & Services** > **Credentials**
2. คลิก **Create Credentials** > **Service Account**
3. ตั้งชื่อ Service Account (เช่น "data-robot")
4. Skip optional steps และ **Done**

### 4. สร้าง Key

1. คลิกที่ Service Account ที่สร้าง
2. ไปที่แท็บ **Keys**
3. คลิก **Add Key** > **Create new key**
4. เลือก **JSON**
5. ไฟล์จะถูก download

### 5. Setup ในโปรเจค

1. Rename ไฟล์ที่ download เป็น `credentials.json`
2. วางไฟล์ในโฟลเดอร์ `app/`
3. Copy email จากไฟล์ (client_email)

### 6. Share Google Sheets

1. เปิด Google Sheets ที่ต้องการใช้
2. คลิก **Share**
3. Paste email จาก `client_email` (ในไฟล์ credentials.json)
4. ให้สิทธิ์ **Editor**
5. ยกเลิก "Notify people"
6. คลิก **Share**

## 📝 ตัวอย่างโครงสร้างไฟล์

ดูตัวอย่างใน `credentials_template.json` แล้วแทนที่:
- `YOUR_PROJECT_ID` → Project ID จาก Google Cloud
- `YOUR_PRIVATE_KEY_ID` → จากไฟล์ที่ download
- `YOUR_PRIVATE_KEY_HERE` → Private key (เก็บรักษา BEGIN/END)
- `YOUR_SERVICE_ACCOUNT` → Service account name
- `YOUR_CLIENT_ID` → Client ID จากไฟล์

## ⚠️ ความปลอดภัย

- ❌ **ห้าม** commit `credentials.json` เข้า git
- ✅ ไฟล์นี้ถูก ignore โดย `.gitignore`
- ✅ เก็บไฟล์นี้ local เท่านั้น
- ✅ ใช้ template สำหรับ share

## 🧪 ทดสอบการเชื่อมต่อ

```bash
# ทดสอบ sync
python sync_now.py

# หรือ
python test_sheets_sync.py
```

## 🚀 ใช้งาน

เมื่อตั้งค่าเสร็จ เปิด Streamlit:
```bash
streamlit run app.py
```

ระบบจะ auto-sync จาก Google Sheets เมื่อโหลดหน้าเว็บ
