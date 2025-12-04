import pandas as pd

# Read the Punthai planning file
file_path = 'Dc/แผนงาน Punthai Maxmart รอบสั่ง 24หยิบ 25พฤศจิกายน 2568 To.เฟิ(1) - สำเนา.xlsx'
df = pd.read_excel(file_path, sheet_name='2.Punthai', header=1)

print('=' * 60)
print('ANALYZING PUNTHAI MAXMART PLANNING FILE')
print('=' * 60)

print('\n📋 COLUMN NAMES:')
for i, col in enumerate(df.columns, 1):
    print(f'{i:2d}. {col}')

print('\n' + '=' * 60)
print('📊 TRIP ANALYSIS')
print('=' * 60)

if 'Trip' in df.columns:
    df_clean = df[df['Trip'].notna()].copy()
    print(f'\nTotal records with Trip: {len(df_clean)}')
    print(f'Unique trips: {df_clean["Trip"].nunique()}')
    
    # Group by trip
    agg_dict = {
        'BranchCode': 'count'
    }
    
    if 'Weight' in df.columns:
        agg_dict['Weight'] = 'sum'
    if 'Cube' in df.columns:
        agg_dict['Cube'] = 'sum'
    if 'จังหวัด' in df.columns:
        agg_dict['จังหวัด'] = lambda x: list(x.dropna().unique())
    
    trip_summary = df_clean.groupby('Trip').agg(agg_dict).reset_index()
    trip_summary.columns = ['Trip', 'Branches', 'Total_Weight', 'Total_Cube', 'Provinces'] if len(trip_summary.columns) == 5 else list(trip_summary.columns)
    
    print('\n🚛 TRIP SUMMARY (First 15 trips):')
    print(trip_summary.head(15).to_string())
    
    print('\n📈 STATISTICS:')
    print(f'  Avg branches per trip: {trip_summary["Branches"].mean():.1f}')
    if 'Total_Weight' in trip_summary.columns:
        print(f'  Avg weight per trip: {trip_summary["Total_Weight"].mean():.1f} kg')
    if 'Total_Cube' in trip_summary.columns:
        print(f'  Avg cube per trip: {trip_summary["Total_Cube"].mean():.2f} m³')
    
    # Sample trips detail
    print('\n' + '=' * 60)
    print('🔍 SAMPLE TRIP DETAILS')
    print('=' * 60)
    
    sample_trips = sorted(df_clean['Trip'].unique())[:3]
    for trip_num in sample_trips:
        trip_data = df_clean[df_clean['Trip'] == trip_num]
        print(f'\nTrip {int(trip_num)}:')
        print(f'  Branches: {len(trip_data)}')
        if 'BranchCode' in trip_data.columns:
            print(f'  Codes: {", ".join(trip_data["BranchCode"].astype(str).tolist())}')
        if 'จังหวัด' in trip_data.columns:
            provinces = trip_data['จังหวัด'].dropna().unique()
            print(f'  Provinces: {", ".join(provinces)}')
        if 'Weight' in trip_data.columns:
            print(f'  Total weight: {trip_data["Weight"].sum():.2f} kg')
        if 'Cube' in trip_data.columns:
            print(f'  Total cube: {trip_data["Cube"].sum():.4f} m³')

print('\n' + '=' * 60)
print('✅ ANALYSIS COMPLETE')
print('=' * 60)
