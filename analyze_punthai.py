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
    
    # Use actual column names
    weight_col = 'TOTALWGT' if 'TOTALWGT' in df.columns else ('Weight' if 'Weight' in df.columns else None)
    cube_col = 'TOTALCUBE' if 'TOTALCUBE' in df.columns else ('Cube' if 'Cube' in df.columns else None)
    
    if weight_col:
        agg_dict[weight_col] = 'sum'
    if cube_col:
        agg_dict[cube_col] = 'sum'
    
    trip_summary = df_clean.groupby('Trip').agg(agg_dict).reset_index()
    
    # Rename columns properly
    new_cols = ['Trip', 'Branches']
    if weight_col:
        new_cols.append('Total_Weight')
    if cube_col:
        new_cols.append('Total_Cube')
    
    trip_summary.columns = new_cols
    
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
    print('🔍 SAMPLE TRIP DETAILS (excluding DC011)')
    print('=' * 60)
    
    # Filter out DC011
    df_clean_no_dc = df_clean[df_clean['BranchCode'] != 'DC011'].copy()
    
    sample_trips = sorted(df_clean_no_dc['Trip'].unique())[:10]
    for trip_num in sample_trips:
        trip_data = df_clean_no_dc[df_clean_no_dc['Trip'] == trip_num]
        
        # Get Trip no
        trip_no = 'N/A'
        if 'Trip no' in trip_data.columns:
            trip_no_values = trip_data['Trip no'].dropna().unique()
            if len(trip_no_values) > 0:
                trip_no = str(trip_no_values[0])
        
        print(f'\n🚛 Trip {int(trip_num)} (Trip no: {trip_no}):')
        print(f'  Branches: {len(trip_data)}')
        if 'BranchCode' in trip_data.columns:
            codes = [str(c) for c in trip_data["BranchCode"].tolist() if str(c) != 'nan']
            if codes:
                print(f'  Codes: {", ".join(codes[:10])}{"..." if len(codes) > 10 else ""}')
        if weight_col and weight_col in trip_data.columns:
            print(f'  Total weight: {trip_data[weight_col].sum():.2f} kg')
        if cube_col and cube_col in trip_data.columns:
            print(f'  Total cube: {trip_data[cube_col].sum():.4f} m³')
        
        # Check vehicle utilization
        if weight_col and cube_col:
            w = trip_data[weight_col].sum()
            c = trip_data[cube_col].sum()
            
            # Check against vehicle limits
            limits = {
                '4W': {'w': 2500, 'c': 5},
                'JB': {'w': 3500, 'c': 8},
                '6W': {'w': 5800, 'c': 22}
            }
            
            for vehicle, lim in limits.items():
                w_pct = (w / lim['w']) * 100
                c_pct = (c / lim['c']) * 100
                max_pct = max(w_pct, c_pct)
                if max_pct <= 105:
                    print(f'  → Fits {vehicle}: {w_pct:.1f}% weight, {c_pct:.1f}% cube')
                    break

print('\n' + '=' * 60)
print('✅ ANALYSIS COMPLETE')
print('=' * 60)
