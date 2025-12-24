"""
ตรวจสอบ DC วังน้อยใน JSON
"""
import json

with open('branch_data.json', encoding='utf-8') as f:
    data = json.load(f)

# ลองหาทุก key ที่มี DC หรือ วังน้อย
print("Keys ที่มี 'DC' หรือ 'wang':")
dc_keys = [k for k in data.keys() if 'DC' in k.upper() or 'WANG' in k.upper() or '8NV' in k.upper()]
for key in dc_keys[:20]:  # แสดง 20 รายการแรก
    branch = data[key]
    print(f"  {key}: {branch.get('สาขา', 'N/A')} | MaxTruckType: {branch.get('MaxTruckType', 'N/A')}")

# ลองค้นหาด้วย case variants
variants = ['8NVDC011', '8nvDC011', '8nvdc011', '8NVDC011', 'DC011']
print("\nค้นหา DC วังน้อยด้วย variants:")
for v in variants:
    if v in data:
        print(f"  พบ: {v}")
        dc = data[v]
        print(f"    สาขา: {dc.get('สาขา', 'N/A')}")
        print(f"    MaxTruckType: {dc.get('MaxTruckType', 'N/A')}")
        break
else:
    print("  ไม่พบทุก variants")

# สรุป MaxTruckType
print("\nสรุป MaxTruckType:")
truck_counts = {}
for branch in data.values():
    truck = branch.get('MaxTruckType', 'N/A')
    truck_counts[truck] = truck_counts.get(truck, 0) + 1

for truck, count in sorted(truck_counts.items()):
    print(f"  {truck}: {count:,} สาขา")
