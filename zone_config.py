# -*- coding: utf-8 -*-
"""
Zone Configuration - ระบบจัดการโซนโลจิสติกส์
แบ่งโซนละเอียดถึงระดับตำบล สำหรับการจัดทริป

หลักการ:
1. เลือกโซนหลักก่อน (ระดับจังหวัด/กลุ่มจังหวัด)
2. ถ้ามีโซนย่อย (ระดับอำเภอ) ให้ใช้โซนย่อย
3. ถ้ามีโซนย่อยสุด (ระดับตำบล) ให้ใช้โซนย่อยสุด
"""

# ==========================================
# HIGHWAY ROUTES
# ==========================================
HIGHWAY_ROUTES = {
    # สายหลัก
    'สาย1_พหลโยธิน': {
        'highway_no': '1',
        'description': 'กทม → เชียงราย',
        'provinces': ['สระบุรี', 'ลพบุรี', 'นครสวรรค์', 'กำแพงเพชร', 'ตาก', 'ลำปาง', 'ลำพูน', 'เชียงใหม่', 'เชียงราย'],
    },
    'สาย2_มิตรภาพ': {
        'highway_no': '2',
        'description': 'สระบุรี → หนองคาย',
        'provinces': ['นครราชสีมา', 'ขอนแก่น', 'อุดรธานี', 'หนองคาย', 'เลย', 'หนองบัวลำภู'],
    },
    'สาย3_สุขุมวิท': {
        'highway_no': '3',
        'description': 'กทม → ตราด',
        'provinces': ['ชลบุรี', 'ระยอง', 'จันทบุรี', 'ตราด'],
    },
    'สาย4_เพชรเกษม': {
        'highway_no': '4',
        'description': 'กทม → สงขลา',
        'provinces': ['เพชรบุรี', 'ประจวบคีรีขันธ์', 'ชุมพร', 'สุราษฎร์ธานี', 'นครศรีธรรมราช', 'สงขลา'],
    },
    # สายรอง
    'สาย9_กาญจนาภิเษก': {
        'highway_no': '9',
        'description': 'รอบนอกกทม',
        'provinces': ['กรุงเทพมหานคร', 'นนทบุรี', 'ปทุมธานี', 'สมุทรปราการ'],
    },
    'สาย11_เอเชียสายเก่า': {
        'highway_no': '11',
        'description': 'นครสวรรค์ → แพร่',
        'provinces': ['นครสวรรค์', 'พิจิตร', 'พิษณุโลก', 'อุตรดิตถ์', 'แพร่'],
    },
    'สาย24_เดชอุดม': {
        'highway_no': '24',
        'description': 'โคราช → อุบล',
        'provinces': ['นครราชสีมา', 'บุรีรัมย์', 'สุรินทร์', 'ศรีสะเกษ', 'อุบลราชธานี'],
    },
    'สาย32_สายเอเชีย': {
        'highway_no': '32',
        'description': 'กทม → นครสวรรค์',
        'provinces': ['พระนครศรีอยุธยา', 'อ่างทอง', 'สิงห์บุรี', 'ชัยนาท', 'นครสวรรค์'],
    },
    'สาย35_บรมราชชนนี': {
        'highway_no': '35',
        'description': 'กทม → นครปฐม',
        'provinces': ['กรุงเทพมหานคร', 'นนทบุรี', 'นครปฐม', 'สมุทรสาคร'],
    },
}

# ==========================================
# LOGISTICS ZONES - โซนหลัก
# ==========================================
ZONE_MAIN = {
    # ภาคเหนือ
    'เหนือ_ไกลสุด': {
        'zone_id': 'NORTH_FAR',
        'provinces': ['พะเยา', 'เชียงราย', 'แม่ฮ่องสอน'],
        'highway': 'สาย1',
        'priority': 1,
        'distance_km': 700,
    },
    'เหนือ_บน': {
        'zone_id': 'NORTH_UPPER',
        'provinces': ['เชียงใหม่', 'ลำปาง', 'ลำพูน', 'น่าน', 'แพร่'],
        'highway': 'สาย1/11/101',
        'priority': 2,
        'distance_km': 550,
    },
    'เหนือ_กลาง': {
        'zone_id': 'NORTH_MID',
        'provinces': ['พิษณุโลก', 'อุตรดิตถ์', 'สุโขทัย', 'เพชรบูรณ์'],
        'highway': 'สาย11',
        'priority': 3,
        'distance_km': 400,
    },
    'เหนือ_ล่าง': {
        'zone_id': 'NORTH_LOWER',
        'provinces': ['นครสวรรค์', 'พิจิตร', 'อุทัยธานี', 'กำแพงเพชร', 'ตาก'],
        'highway': 'สาย1/32',
        'priority': 4,
        'distance_km': 300,
    },
    
    # ภาคอีสาน
    'อีสาน_บน': {
        'zone_id': 'ISAN_UPPER',
        'provinces': ['อุดรธานี', 'หนองคาย', 'บึงกาฬ', 'หนองบัวลำภู', 'เลย', 'สกลนคร', 'นครพนม', 'มุกดาหาร'],
        'highway': 'สาย2',
        'priority': 5,
        'distance_km': 550,
    },
    'อีสาน_กลาง': {
        'zone_id': 'ISAN_MID',
        'provinces': ['ขอนแก่น', 'มหาสารคาม', 'กาฬสินธุ์', 'ร้อยเอ็ด'],
        'highway': 'สาย2',
        'priority': 6,
        'distance_km': 450,
    },
    'อีสาน_ประตู': {
        'zone_id': 'ISAN_GATE',
        'provinces': ['นครราชสีมา', 'ชัยภูมิ'],
        'highway': 'สาย2',
        'priority': 7,
        'distance_km': 260,
    },
    'อีสาน_ใต้': {
        'zone_id': 'ISAN_LOWER',
        'provinces': ['บุรีรัมย์', 'สุรินทร์', 'ศรีสะเกษ', 'อุบลราชธานี', 'ยโสธร', 'อำนาจเจริญ'],
        'highway': 'สาย24',
        'priority': 8,
        'distance_km': 500,
    },
    
    # ภาคตะวันออก
    'ตะวันออก_EEC': {
        'zone_id': 'EAST_EEC',
        'provinces': ['ชลบุรี', 'ระยอง'],
        'highway': 'สาย3/331',
        'priority': 9,
        'distance_km': 120,
    },
    'ตะวันออก_ไกล': {
        'zone_id': 'EAST_FAR',
        'provinces': ['จันทบุรี', 'ตราด'],
        'highway': 'สาย3',
        'priority': 10,
        'distance_km': 300,
    },
    'ตะวันออก_ใกล้': {
        'zone_id': 'EAST_NEAR',
        'provinces': ['ฉะเชิงเทรา', 'ปราจีนบุรี', 'นครนายก', 'สระแก้ว'],
        'highway': 'สาย305/304',
        'priority': 11,
        'distance_km': 100,
    },
    
    # ภาคใต้
    'ใต้_ไกลสุด': {
        'zone_id': 'SOUTH_FAR',
        'provinces': ['สงขลา', 'ปัตตานี', 'ยะลา', 'นราธิวาส'],
        'highway': 'สาย4',
        'priority': 12,
        'distance_km': 900,
    },
    'ใต้_กลาง': {
        'zone_id': 'SOUTH_MID',
        'provinces': ['นครศรีธรรมราช', 'พัทลุง', 'ตรัง'],
        'highway': 'สาย4',
        'priority': 13,
        'distance_km': 700,
    },
    'ใต้_อันดามัน': {
        'zone_id': 'SOUTH_ANDAMAN',
        'provinces': ['ภูเก็ต', 'กระบี่', 'พังงา', 'สตูล'],
        'highway': 'สาย401/402',
        'priority': 14,
        'distance_km': 800,
    },
    'ใต้_บน': {
        'zone_id': 'SOUTH_UPPER',
        'provinces': ['สุราษฎร์ธานี', 'ชุมพร', 'ระนอง'],
        'highway': 'สาย4',
        'priority': 15,
        'distance_km': 450,
    },
    'ใต้_ประตู': {
        'zone_id': 'SOUTH_GATE',
        'provinces': ['เพชรบุรี', 'ประจวบคีรีขันธ์'],
        'highway': 'สาย4',
        'priority': 16,
        'distance_km': 250,
    },
    
    # ภาคตะวันตก
    'ตะวันตก_ไกล': {
        'zone_id': 'WEST_FAR',
        'provinces': ['ตาก', 'กาญจนบุรี'],
        'highway': 'สาย321',
        'priority': 17,
        'distance_km': 350,
    },
    'ตะวันตก_ใกล้': {
        'zone_id': 'WEST_NEAR',
        'provinces': ['ราชบุรี', 'สุพรรณบุรี'],
        'highway': 'สาย321/340',
        'priority': 18,
        'distance_km': 120,
    },
    
    # กทม/ปริมณฑล
    'กทม_ปริมณฑล': {
        'zone_id': 'BANGKOK_METRO',
        'provinces': ['กรุงเทพมหานคร', 'นนทบุรี', 'ปทุมธานี', 'สมุทรปราการ', 'สมุทรสาคร', 'นครปฐม'],
        'highway': 'สาย9/35',
        'priority': 99,
        'distance_km': 40,
    },
    'อยุธยา_DC': {
        'zone_id': 'AYUTTHAYA_DC',
        'provinces': ['พระนครศรีอยุธยา'],
        'highway': 'สาย1/32',
        'priority': 100,
        'distance_km': 20,
    },
}

# ==========================================
# LOGISTICS ZONES - โซนย่อย (ระดับอำเภอ)
# ==========================================
ZONE_SUB_DISTRICT = {
    # นครราชสีมา - แบ่งย่อย
    'โคราช_เมือง': {
        'parent_zone': 'อีสาน_ประตู',
        'province': 'นครราชสีมา',
        'districts': ['เมืองนครราชสีมา', 'ปักธงชัย', 'สีดา'],
        'priority': 7.1,
    },
    'โคราช_เหนือ': {
        'parent_zone': 'อีสาน_ประตู',
        'province': 'นครราชสีมา',
        'districts': ['พิมาย', 'ห้วยแถลง', 'โชคชัย', 'แก้งสนามนาง'],
        'priority': 7.2,
    },
    'โคราช_ตะวันออก': {
        'parent_zone': 'อีสาน_ประตู',
        'province': 'นครราชสีมา',
        'districts': ['บัวใหญ่', 'ครบุรี', 'สีคิ้ว', 'โนนสูง', 'โนนแดง'],
        'priority': 7.3,
    },
    'โคราช_ใต้': {
        'parent_zone': 'อีสาน_ประตู',
        'province': 'นครราชสีมา',
        'districts': ['ปากช่อง', 'วังน้ำเขียว', 'คง', 'ชุมพวง'],
        'priority': 7.4,
    },
    
    # ขอนแก่น - แบ่งย่อย
    'ขอนแก่น_เมือง': {
        'parent_zone': 'อีสาน_กลาง',
        'province': 'ขอนแก่น',
        'districts': ['เมืองขอนแก่น', 'น้ำพอง', 'บ้านไผ่'],
        'priority': 6.1,
    },
    'ขอนแก่น_เหนือ': {
        'parent_zone': 'อีสาน_กลาง',
        'province': 'ขอนแก่น',
        'districts': ['กระนวน', 'พระยืน', 'หนองเรือ', 'อุบลรัตน์'],
        'priority': 6.2,
    },
    'ขอนแก่น_ใต้': {
        'parent_zone': 'อีสาน_กลาง',
        'province': 'ขอนแก่น',
        'districts': ['บ้านฝาง', 'ชนบท', 'พล', 'แวงใหญ่'],
        'priority': 6.3,
    },
    
    # พิษณุโลก - แบ่งย่อย
    'พิษณุโลก_เมือง': {
        'parent_zone': 'เหนือ_กลาง',
        'province': 'พิษณุโลก',
        'districts': ['เมืองพิษณุโลก'],
        'subdistricts': ['วัดจันทร์', 'ในเมือง', 'หัวรอ', 'บ้านคลอง'],
        'priority': 3.1,
    },
    'พิษณุโลก_มหาวิทยาลัย': {
        'parent_zone': 'เหนือ_กลาง',
        'province': 'พิษณุโลก',
        'districts': ['เมืองพิษณุโลก'],
        'subdistricts': ['ท่าโพธิ์', 'อรัญญิก', 'แม่กา'],
        'priority': 3.2,
    },
    'พิษณุโลก_ตะวันออก': {
        'parent_zone': 'เหนือ_กลาง',
        'province': 'พิษณุโลก',
        'districts': ['วังทอง', 'พรหมพิราม', 'เนินมะปราง', 'บางระกำ'],
        'priority': 3.3,
    },
    
    # พิจิตร - แบ่งย่อย
    'พิจิตร_สายหลัก': {
        'parent_zone': 'เหนือ_ล่าง',
        'province': 'พิจิตร',
        'districts': ['เมืองพิจิตร', 'สากเหล็ก', 'สามง่าม'],
        'priority': 4.1,
    },
    'พิจิตร_ตะวันออก': {
        'parent_zone': 'เหนือ_ล่าง',
        'province': 'พิจิตร',
        'districts': ['ตะพานหิน', 'ทับคล้อ', 'ดงเจริญ'],
        'priority': 4.2,
    },
    
    # กรุงเทพ - แบ่งย่อย
    'กทม_เหนือ': {
        'parent_zone': 'กทม_ปริมณฑล',
        'province': 'กรุงเทพมหานคร',
        'districts': ['หลักสี่', 'ดอนเมือง', 'สายไหม', 'จตุจักร'],
        'priority': 99.1,
    },
    'กทม_กลาง': {
        'parent_zone': 'กทม_ปริมณฑล',
        'province': 'กรุงเทพมหานคร',
        'districts': ['ปทุมวัน', 'วัฒนา', 'คลองเตย', 'บางรัก'],
        'priority': 99.2,
    },
    'กทม_ใต้': {
        'parent_zone': 'กทม_ปริมณฑล',
        'province': 'กรุงเทพมหานคร',
        'districts': ['บางแค', 'ราษฎร์บูรณะ', 'ทุ่งครุ', 'จอมทอง'],
        'priority': 99.3,
    },
    'กทม_ตะวันออก': {
        'parent_zone': 'กทม_ปริมณฑล',
        'province': 'กรุงเทพมหานคร',
        'districts': ['บางกะปิ', 'บึงกุ่ม', 'สะพานสูง', 'ลาดกระบัง'],
        'priority': 99.4,
    },
}

# ==========================================
# LOGISTICS ZONES - โซนย่อยสุด (ระดับตำบล)
# ==========================================
ZONE_SUB_SUBDISTRICT = {
    # พิษณุโลก เมือง - แบ่งละเอียดสุด
    'พิษณุโลก_ในเมือง_ตลาด': {
        'parent_zone': 'พิษณุโลก_เมือง',
        'province': 'พิษณุโลก',
        'district': 'เมืองพิษณุโลก',
        'subdistricts': ['ในเมือง', 'วัดจันทร์'],
        'priority': 3.11,
        'description': 'ย่านตลาด-ใจกลางเมือง',
    },
    'พิษณุโลก_ในเมือง_หัวรอ': {
        'parent_zone': 'พิษณุโลก_เมือง',
        'province': 'พิษณุโลก',
        'district': 'เมืองพิษณุโลก',
        'subdistricts': ['หัวรอ', 'บ้านคลอง', 'บึงพระ'],
        'priority': 3.12,
        'description': 'ย่านหัวรอ-ชุมชน',
    },
    
    # นครราชสีมา เมือง - แบ่งละเอียดสุด
    'โคราช_ในเมือง_ตลาด': {
        'parent_zone': 'โคราช_เมือง',
        'province': 'นครราชสีมา',
        'district': 'เมืองนครราชสีมา',
        'subdistricts': ['ในเมือง', 'ปากธงชัย', 'โพธิ์กลาง'],
        'priority': 7.11,
        'description': 'ใจกลางเมืองโคราช',
    },
    
    # ขอนแก่น เมือง - แบ่งละเอียดสุด
    'ขอนแก่น_ในเมือง_CBD': {
        'parent_zone': 'ขอนแก่น_เมือง',
        'province': 'ขอนแก่น',
        'district': 'เมืองขอนแก่น',
        'subdistricts': ['ในเมือง', 'บ้านเป็ด', 'บ้านค้อ'],
        'priority': 6.11,
        'description': 'ใจกลางเมืองขอนแก่น',
    },
}

# ==========================================
# NO CROSS ZONE RULES
# ==========================================
NO_CROSS_ZONE_PAIRS = [
    ('พะเยา', 'เชียงใหม่'),  # ข้ามเทือกเขา
    ('น่าน', 'พะเยา'),  # หุบเขา
    ('เพชรบูรณ์', 'ชัยภูมิ'),  # ข้ามเขา
    ('เพชรบูรณ์', 'เลย'),  # ข้ามเขา
    ('หนองคาย', 'อุบลราชธานี'),  # คนละฝั่งโขง
    ('กระบี่', 'สุราษฎร์ธานี'),  # ฝั่งอันดามัน vs อ่าวไทย
    ('ภูเก็ต', 'นครศรีธรรมราช'),  # ฝั่งอันดามัน vs อ่าวไทย
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_main_zone(province):
    """
    หาโซนหลักจากจังหวัด
    
    Args:
        province (str): ชื่อจังหวัด
    
    Returns:
        tuple: (zone_id, zone_info) หรือ (None, None) ถ้าไม่พบ
    """
    if not province:
        return None, None
    
    province = str(province).strip()
    
    for zone_name, zone_info in ZONE_MAIN.items():
        if province in zone_info['provinces']:
            return zone_name, zone_info
    
    return None, None


def get_sub_zone(province, district):
    """
    หาโซนย่อยจากจังหวัด + อำเภอ
    
    Args:
        province (str): ชื่อจังหวัด
        district (str): ชื่ออำเภอ
    
    Returns:
        tuple: (zone_id, zone_info) หรือ (None, None) ถ้าไม่พบ
    """
    if not province or not district:
        return None, None
    
    province = str(province).strip()
    district = str(district).strip()
    
    for zone_name, zone_info in ZONE_SUB_DISTRICT.items():
        if zone_info['province'] == province and district in zone_info.get('districts', []):
            return zone_name, zone_info
    
    return None, None


def get_sub_subzone(province, district, subdistrict):
    """
    หาโซนย่อยสุดจากจังหวัด + อำเภอ + ตำบล
    
    Args:
        province (str): ชื่อจังหวัด
        district (str): ชื่ออำเภอ
        subdistrict (str): ชื่อตำบล
    
    Returns:
        tuple: (zone_id, zone_info) หรือ (None, None) ถ้าไม่พบ
    """
    if not province or not district or not subdistrict:
        return None, None
    
    province = str(province).strip()
    district = str(district).strip()
    subdistrict = str(subdistrict).strip()
    
    for zone_name, zone_info in ZONE_SUB_SUBDISTRICT.items():
        if (zone_info['province'] == province and 
            zone_info['district'] == district and 
            subdistrict in zone_info.get('subdistricts', [])):
            return zone_name, zone_info
    
    return None, None


def get_logistics_zone(province, district='', subdistrict=''):
    """
    หาโซนโลจิสติกส์แบบ Hierarchical
    ลำดับ: โซนย่อยสุด (ตำบล) → โซนย่อย (อำเภอ) → โซนหลัก (จังหวัด)
    
    Args:
        province (str): ชื่อจังหวัด
        district (str): ชื่ออำเภอ (optional)
        subdistrict (str): ชื่อตำบล (optional)
    
    Returns:
        tuple: (zone_id, priority, description)
    """
    # Priority 1: โซนหลักก่อน (ระดับจังหวัด)
    main_zone_id, main_zone_info = get_main_zone(province)
    if not main_zone_id:
        return None, 999, 'ไม่พบโซน'
    
    # Priority 2: โซนย่อย (ระดับอำเภอ) ถ้ามี
    if district:
        sub_zone_id, sub_zone_info = get_sub_zone(province, district)
        if sub_zone_id:
            # Priority 3: โซนย่อยสุด (ระดับตำบล) ถ้ามี
            if subdistrict:
                subsub_zone_id, subsub_zone_info = get_sub_subzone(province, district, subdistrict)
                if subsub_zone_id:
                    return subsub_zone_id, subsub_zone_info['priority'], subsub_zone_info.get('description', '')
            
            # ใช้โซนย่อย
            return sub_zone_id, sub_zone_info['priority'], f"{province}-{district}"
    
    # ใช้โซนหลัก
    return main_zone_id, main_zone_info['priority'], main_zone_info['zone_id']


def get_zone_priority(zone_id):
    """
    ดึงค่า Priority ของโซน
    
    Returns:
        float: Priority (เลขน้อย = ไกล, เลขมาก = ใกล้)
    """
    # ค้นหาใน ZONE_SUB_SUBDISTRICT
    if zone_id in ZONE_SUB_SUBDISTRICT:
        return ZONE_SUB_SUBDISTRICT[zone_id]['priority']
    
    # ค้นหาใน ZONE_SUB_DISTRICT
    if zone_id in ZONE_SUB_DISTRICT:
        return ZONE_SUB_DISTRICT[zone_id]['priority']
    
    # ค้นหาใน ZONE_MAIN
    if zone_id in ZONE_MAIN:
        return ZONE_MAIN[zone_id]['priority']
    
    return 999  # ไม่พบ


def can_combine_zones(zone1, zone2):
    """
    เช็คว่า 2 โซนสามารถรวมทริปได้หรือไม่
    
    Returns:
        bool: True ถ้ารวมได้
    """
    if not zone1 or not zone2:
        return False
    
    # ถ้าโซนเดียวกัน → รวมได้
    if zone1 == zone2:
        return True
    
    # ถ้าเป็นโซนย่อยของโซนหลักเดียวกัน → รวมได้
    zone1_info = ZONE_SUB_DISTRICT.get(zone1) or ZONE_SUB_SUBDISTRICT.get(zone1)
    zone2_info = ZONE_SUB_DISTRICT.get(zone2) or ZONE_SUB_SUBDISTRICT.get(zone2)
    
    if zone1_info and zone2_info:
        parent1 = zone1_info.get('parent_zone')
        parent2 = zone2_info.get('parent_zone')
        if parent1 and parent2 and parent1 == parent2:
            return True
    
    # ถ้าแตกต่างกันมาก → ห้ามรวม
    priority_diff = abs(get_zone_priority(zone1) - get_zone_priority(zone2))
    if priority_diff > 5:  # ห่างกันเกิน 5 level
        return False
    
    return True


def is_cross_zone_violation(province1, province2):
    """
    เช็คว่าห้ามข้ามโซนหรือไม่
    
    Returns:
        bool: True ถ้าห้ามข้าม
    """
    if not province1 or not province2:
        return False
    
    return (province1, province2) in NO_CROSS_ZONE_PAIRS or (province2, province1) in NO_CROSS_ZONE_PAIRS


if __name__ == '__main__':
    # ทดสอบ
    print("=== ทดสอบระบบโซน ===\n")
    
    # Test 1: โซนหลัก
    zone, priority, desc = get_logistics_zone('นครราชสีมา')
    print(f"นครราชสีมา: {zone} (Priority: {priority}) - {desc}")
    
    # Test 2: โซนย่อย
    zone, priority, desc = get_logistics_zone('นครราชสีมา', 'เมืองนครราชสีมา')
    print(f"นครราชสีมา/เมือง: {zone} (Priority: {priority}) - {desc}")
    
    # Test 3: โซนย่อยสุด
    zone, priority, desc = get_logistics_zone('พิษณุโลก', 'เมืองพิษณุโลก', 'ในเมือง')
    print(f"พิษณุโลก/เมือง/ในเมือง: {zone} (Priority: {priority}) - {desc}")
    
    # Test 4: Can combine
    print(f"\nโคราช_เมือง + โคราช_เหนือ: {can_combine_zones('โคราช_เมือง', 'โคราช_เหนือ')}")
    print(f"โคราช_เมือง + ขอนแก่น_เมือง: {can_combine_zones('โคราช_เมือง', 'ขอนแก่น_เมือง')}")
