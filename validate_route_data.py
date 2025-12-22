import pandas as pd
import numpy as np

# --- 1. VEHICLE SPECS FROM EXCEL (MOCKED) ---
# To use real Excel: 
# df_info = pd.read_excel('Autoplan.xlsx', sheet_name='Info')
df_info = pd.DataFrame({
    'Vehicle_Type': ['4W', 'JB', '6W'],
    'Max_Weight_kg': [2500, 3500, 6000],
    'Max_Cube': [5.0, 7.0, 20.0],
    'Max_Drops_Mixed': [12, 12, 999],
    'Max_Drops_Punthai': [5, 10, 999]
})
VEHICLE_SPECS = df_info.set_index('Vehicle_Type').T.to_dict()

# --- 2. ZONE MAPPING ---
PROVINCE_TO_ZONE = {
    'Bangkok': 'CENTRAL', 'Nonthaburi': 'CENTRAL', 'Pathum Thani': 'CENTRAL',
    'Khon Kaen': 'NE', 'Udon Thani': 'NE', 'Chiang Mai': 'NORTH', 'Lampang': 'NORTH',
    # ...add all provinces as needed
}
CENTRAL_ZONE = 'CENTRAL'

# --- 3. MOCKED DELIVERY DATA ---
data = [
    # Zone, Prov, Dist, Subdist, Store, Weight, Cube, Prov_Dist_km, Dist_Subdist_km
    ['CENTRAL', 'Bangkok', 'Bang Kapi', 'Hua Mak', 'StoreA', 500, 1.2, 50, 10],
    ['CENTRAL', 'Bangkok', 'Bang Kapi', 'Khlong Chan', 'StoreB', 600, 1.0, 50, 8],
    ['NE', 'Khon Kaen', 'Mueang', 'Nai Mueang', 'StoreC', 1200, 2.5, 400, 30],
    ['NE', 'Khon Kaen', 'Mueang', 'Ban Pet', 'StoreD', 1000, 2.0, 400, 25],
    ['NORTH', 'Chiang Mai', 'Mueang', 'Chang Phueak', 'StoreE', 800, 1.8, 700, 20],
    # ...add more rows as needed
]
df = pd.DataFrame(data, columns=['Zone', 'Prov', 'Dist', 'Subdist', 'Store', 'Weight', 'Cube', 'Prov_Dist_km', 'Dist_Subdist_km'])

# --- DATA VALIDATION ---
def validate_data(df_info, df_delivery, province_to_zone):
    print('--- VEHICLE SPECS ---')
    print(df_info)
    print('\n--- DELIVERY DATA (HEAD) ---')
    print(df_delivery.head())
    print('\n--- CHECK FOR NaN ---')
    print(df_delivery.isna().sum())
    print('\n--- WEIGHT & CUBE STATS ---')
    print(df_delivery[['Weight', 'Cube']].describe())
    print('\n--- PROVINCE ZONE MAPPING CHECK ---')
    unmapped = set(df_delivery['Prov']) - set(province_to_zone.keys())
    if unmapped:
        print('Unmapped provinces:', unmapped)
    else:
        print('All provinces mapped to zones.')

validate_data(df_info, df, PROVINCE_TO_ZONE)
