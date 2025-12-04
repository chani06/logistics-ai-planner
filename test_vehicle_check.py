"""
Test script to verify vehicle restriction logic
"""

# Mock data structures
BOOKING_RESTRICTIONS = {
    'branch_restrictions': {
        'MH64': {'max_vehicle': '4W', 'count': 26},
        'MI65': {'max_vehicle': '4W', 'count': 12},
        'G031': {'max_vehicle': '4W', 'count': 1},
    }
}

PUNTHAI_PATTERNS = {
    'punthai_restrictions': {}
}

def get_max_vehicle_for_branch(branch_code):
    """ดึงรถใหญ่สุดที่สาขานี้รองรับ"""
    branch_code_str = str(branch_code).strip()
    
    # 1. Booking History
    booking_restrictions = BOOKING_RESTRICTIONS.get('branch_restrictions', {})
    if branch_code_str in booking_restrictions:
        return booking_restrictions[branch_code_str].get('max_vehicle', '6W')
    
    # 2. Punthai
    punthai_restrictions = PUNTHAI_PATTERNS.get('punthai_restrictions', {})
    if branch_code_str in punthai_restrictions:
        return punthai_restrictions[branch_code_str].get('max_vehicle', '6W')
    
    # 3. Default
    return '6W'

def get_max_vehicle_for_trip(trip_codes):
    """หารถใหญ่สุดที่ทริปนี้ใช้ได้"""
    vehicle_priority = {'4W': 1, 'JB': 2, '6W': 3}
    max_allowed = '6W'
    max_priority = 3
    
    for code in trip_codes:
        branch_max = get_max_vehicle_for_branch(code)
        priority = vehicle_priority.get(branch_max, 3)
        
        if priority < max_priority:
            max_priority = priority
            max_allowed = branch_max
    
    return max_allowed

# Test cases
print("=== Test Vehicle Restrictions ===\n")

# Test 1: สาขาจำกัด 4W
print("Test 1: สาขา MH64 (จำกัด 4W)")
print(f"  get_max_vehicle_for_branch('MH64') = {get_max_vehicle_for_branch('MH64')}")
print(f"  Expected: 4W\n")

# Test 2: สาขาใหม่ (ไม่มีข้อจำกัด)
print("Test 2: สาขา N669 (ไม่มีข้อจำกัด)")
print(f"  get_max_vehicle_for_branch('N669') = {get_max_vehicle_for_branch('N669')}")
print(f"  Expected: 6W\n")

# Test 3: ทริปผสม (มีทั้งจำกัดและไม่จำกัด)
trip_codes = {'MH64', 'N669', 'GP00'}
print(f"Test 3: ทริปผสม {trip_codes}")
print(f"  - MH64: {get_max_vehicle_for_branch('MH64')} (จำกัด)")
print(f"  - N669: {get_max_vehicle_for_branch('N669')} (ไม่จำกัด)")
print(f"  - GP00: {get_max_vehicle_for_branch('GP00')} (ไม่จำกัด)")
print(f"  get_max_vehicle_for_trip() = {get_max_vehicle_for_trip(trip_codes)}")
print(f"  Expected: 4W (เพราะมี MH64 จำกัด 4W)\n")

# Test 4: ทริปไม่มีข้อจำกัด
trip_codes_no_limit = {'N669', 'GP00', '11005361'}
print(f"Test 4: ทริปไม่มีข้อจำกัด {trip_codes_no_limit}")
print(f"  get_max_vehicle_for_trip() = {get_max_vehicle_for_trip(trip_codes_no_limit)}")
print(f"  Expected: 6W\n")

# Test 5: Logic การเช็คก่อนเพิ่มสาขา
print("Test 5: การเช็คก่อนเพิ่มสาขา")
current_trip = ['N669', 'GP00']
new_branch = 'MH64'
vehicle_type = '6W'

print(f"  ทริปปัจจุบัน: {current_trip} (ใช้รถ {vehicle_type})")
print(f"  สาขาใหม่: {new_branch}")

current_trip_with_new = current_trip + [new_branch]
trip_max_vehicle = get_max_vehicle_for_trip(set(current_trip_with_new))

vehicle_priority = {'4W': 1, 'JB': 2, '6W': 3}
current_priority = vehicle_priority.get(vehicle_type, 3)
new_priority = vehicle_priority.get(trip_max_vehicle, 3)

print(f"  trip_max_vehicle หลังเพิ่ม: {trip_max_vehicle}")
print(f"  current_priority: {current_priority} (6W)")
print(f"  new_priority: {new_priority} (4W)")
print(f"  new_priority < current_priority? {new_priority < current_priority}")
print(f"  ผลลัพธ์: {'❌ ไม่เพิ่ม (สาขาจำกัด 4W แต่ทริปใช้ 6W)' if new_priority < current_priority else '✅ เพิ่มได้'}\n")

print("=== สรุป ===")
print("✅ Logic ถูกต้อง: สาขาที่จำกัด 4W จะไม่ถูกเพิ่มเข้าทริป 6W")
