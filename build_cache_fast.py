"""สร้าง distance_cache.json จาก branch_data.json แบบเร็ว"""
import json
from math import radians, sin, cos, sqrt, atan2

print("🚀 สร้าง distance_cache.json...")

# โหลด branch_data
with open('branch_data.json', 'r', encoding='utf-8') as f:
    branch_data = json.load(f)

print(f"📊 โหลด {len(branch_data):,} สาขา")

# DC coordinates
DC_LAT, DC_LON = 14.179394, 100.648149

distance_cache = {}
count = 0

# คำนวณระยะทาง DC -> สาขาทั้งหมด
for code, data in branch_data.items():
    lat = data.get('ละ') or data.get('ละติจูด')
    lon = data.get('ลอง') or data.get('ลองติจูด')
    
    if lat and lon:
        try:
            lat_f = float(lat)
            lon_f = float(lon)
            if lat_f > 0 and lon_f > 0:
                # Haversine
                dlat = radians(lat_f - DC_LAT)
                dlon = radians(lon_f - DC_LON)
                a = sin(dlat/2)**2 + cos(radians(DC_LAT)) * cos(radians(lat_f)) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1-a))
                dist = 6371 * c
                
                key = f"{DC_LAT:.4f},{DC_LON:.4f}_{lat_f:.4f},{lon_f:.4f}"
                distance_cache[key] = dist
                count += 1
                
                if count % 1000 == 0:
                    print(f"   ⏳ {count:,}/{len(branch_data):,}...")
        except:
            pass

# บันทึก
with open('distance_cache.json', 'w', encoding='utf-8') as f:
    json.dump(distance_cache, f, ensure_ascii=False)

print(f"✅ สำเร็จ! บันทึก {len(distance_cache):,} รายการ")
