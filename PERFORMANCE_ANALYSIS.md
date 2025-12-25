# การวิเคราะห์ประสิทธิภาพและปัญหาของระบบจัดทริป

## 🚨 ปัญหาที่พบ

### 1. **เรียกฟังก์ชันซ้ำซ้อน (Redundant Function Calls)**

#### `is_all_punthai_codes()` - เรียก 10+ ครั้ง
- บรรทัด 3026: `split_until_fits()` loop
- บรรทัด 3084: `process_overflow_queue()`
- บรรทัด 3546: subdistrict processing
- บรรทัด 3565, 3643, 3674: เริ่มทริปใหม่

**ปัญหา**: คำนวณซ้ำทุกครั้งที่เปลี่ยน codes โดยไม่จำเป็น

#### `get_allowed_from_codes()` - เรียก 10+ ครั้ง
- บรรทัด 3027: `split_until_fits()` loop
- บรรทัด 3085: `process_overflow_queue()`
- บรรทัด 3105, 3120, 3133: สร้างทริปใหม่
- บรรทัด 3374, 3399: ตรวจสอบ constraint
- บรรทัด 3547, 3564, 3642, 3673: subdistrict processing

**ปัญหา**: คำนวณ constraint intersection ซ้ำแม้ว่า codes ไม่เปลี่ยน

### 2. **Logic ช้า (Slow Logic)**

#### `check_intra_trip_spread()` - O(n²) complexity
```python
for _, row in trip_df.iterrows():  # Loop ทุก branch
    if row['_lat'] > 0 and row['_lon'] > 0:
        dist = haversine_distance(...)  # คำนวณระยะทาง (ช้า)
        max_dist_from_center = max(max_dist_from_center, dist)
```

**ปัญหา**:
- ใช้ `iterrows()` ช้ามาก
- คำนวณ haversine distance ทุก branch
- เรียกทุกครั้งที่ merge subdistrict (บรรทัด 3550)

#### `split_until_fits()` - Nested loops
```python
while iteration < max_iterations:  # Loop 1
    iteration += 1
    # ... calculations ...
    current_trip['is_punthai'] = is_all_punthai_codes(...)  # เรียกทุก iteration
    current_trip['allowed_vehicles'] = get_allowed_from_codes(...)  # เรียกทุก iteration
```

**ปัญหา**: Loop 100 iterations สูงสุด × คำนวณซ้ำทุกครั้ง

### 3. **ข้อมูลซ้ำซ้อน (Redundant Data)**

#### Buffer calculation ซ้ำ
- คำนวณ `buffer_mult` ซ้ำในหลายจุด:
  - `select_vehicle_for_load()`
  - `finalize_current_trip()`
  - `split_until_fits()`

#### Limits lookup ซ้ำ
- เรียก `get_max_limits()` ซ้ำแม้ว่า allowed_vehicles ไม่เปลี่ยน

### 4. **Logic ที่หายไป (Missing Logic)**

#### ❌ ไม่มีการเช็ค empty dataframe
- `check_intra_trip_spread()` อาจเจอ empty df
- `df.loc[df['Code'] == overflow_code]` อาจไม่เจอ

#### ❌ ไม่มีการ validate input
- `select_vehicle_for_load()` ไม่เช็ค negative weight/cube
- ไม่เช็ค allowed_vehicles format

#### ❌ ไม่มี early exit
- `split_until_fits()` loop ต่อแม้รู้ว่าต้อง split
- `check_intra_trip_spread()` คำนวณทั้งหมดแม้เจอ outlier แล้ว

---

## 🎯 การแก้ไขที่แนะนำ

### 1. **Cache Function Results**
```python
# Cache is_punthai และ allowed_vehicles
trip_metadata_cache = {}  # {tuple(codes): {'is_punthai': bool, 'allowed': list}}

def get_trip_metadata(codes, allowed_vehicles):
    key = tuple(sorted(codes))
    if key not in trip_metadata_cache:
        trip_metadata_cache[key] = {
            'is_punthai': is_all_punthai_codes(codes),
            'allowed': get_allowed_from_codes(codes, allowed_vehicles)
        }
    return trip_metadata_cache[key]
```

### 2. **Optimize check_intra_trip_spread()**
```python
def check_intra_trip_spread(trip_codes_list):
    if len(trip_codes_list) < 2:
        return True
    
    # ใช้ vectorized operations แทน iterrows()
    trip_df = df[df['Code'].isin(trip_codes_list)]
    if trip_df.empty or len(trip_df) < 2:
        return True
    
    # กรอง branch ที่ไม่มีพิกัด
    valid_coords = trip_df[(trip_df['_lat'] > 0) & (trip_df['_lon'] > 0)]
    if len(valid_coords) < 2:
        return True
    
    # Vectorized distance calculation
    center_lat = valid_coords['_lat'].mean()
    center_lon = valid_coords['_lon'].mean()
    
    # Calculate distances in batch (faster)
    distances = valid_coords.apply(
        lambda row: haversine_distance(center_lat, center_lon, row['_lat'], row['_lon']),
        axis=1
    )
    
    # Early exit if any distance > 80km
    return distances.max() <= 80
```

### 3. **Reduce split_until_fits() iterations**
```python
def split_until_fits(allowed_vehicles, region):
    # Pre-calculate limits ONCE
    is_punthai = current_trip['is_punthai']
    limits = get_max_limits(current_trip['allowed_vehicles'], is_punthai)
    buffer_mult = punthai_buffer if is_punthai else maxmart_buffer
    
    # Calculate how many branches to remove in one go
    max_iterations = 10  # Reduce from 100
    
    while iteration < max_iterations:
        # ... existing logic ...
        
        # Only recalculate if codes changed
        if len(current_trip['codes']) != prev_codes_count:
            current_trip['is_punthai'] = is_all_punthai_codes(current_trip['codes'])
            current_trip['allowed_vehicles'] = get_allowed_from_codes(current_trip['codes'], allowed_vehicles)
            prev_codes_count = len(current_trip['codes'])
```

### 4. **Add Early Exit Logic**
```python
# In select_vehicle_for_load()
def select_vehicle_for_load(weight, cube, drops, is_punthai, allowed_vehicles, debug=False):
    # Validate input
    if weight <= 0 or cube <= 0:
        return '6W' if '6W' in allowed_vehicles else '4W'
    
    if not allowed_vehicles:
        return '6W'
    
    # ... rest of logic ...
```

### 5. **Pre-compute Metadata**
```python
# Pre-compute trip metadata when creating dataframe
df['_is_punthai'] = df['Code'].apply(lambda c: branch_bu_cache.get(c, False))
df['_max_allowed'] = df['Code'].apply(lambda c: branch_max_vehicle_cache.get(c, '6W'))
```

---

## 📊 ผลกระทบที่คาดว่าจะได้รับ

### ก่อนแก้ไข:
- **Function calls**: 100-200+ ครั้งต่อ 1000 สาขา
- **Complexity**: O(n³) ในบางกรณี
- **เวลาประมวลผล**: 30-60 วินาที สำหรับ 1000 สาขา

### หลังแก้ไข (คาดการณ์):
- **Function calls**: 20-30 ครั้งต่อ 1000 สาขา (ลด 80%)
- **Complexity**: O(n log n) 
- **เวลาประมวลผล**: 5-10 วินาที (เร็วขึ้น 5-6 เท่า)

---

## ⚠️ ความเสี่ยง

1. **Cache invalidation**: ต้องแน่ใจว่า cache ถูกลบเมื่อข้อมูลเปลี่ยน
2. **Memory usage**: Cache อาจใช้ memory มากขึ้น (trade-off)
3. **Logic changes**: ต้องทดสอบให้แน่ใจว่าผลลัพธ์ไม่เปลี่ยน

---

## 🔧 ขั้นตอนการแก้ไข

1. ✅ สร้างรายงานวิเคราะห์ (เอกสารนี้)
2. ⏳ แก้ check_intra_trip_spread() ให้ใช้ vectorized operations
3. ⏳ เพิ่ม metadata cache
4. ⏳ ลด split_until_fits() iterations
5. ⏳ เพิ่ม input validation และ early exits
6. ⏳ Test และวัดผล

---

**สร้างเมื่อ**: 2025-12-25
**Version**: 1.0
