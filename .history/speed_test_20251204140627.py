"""Test processing speed"""
import datetime
import pandas as pd
import sys

print("Starting speed test...")

t1 = datetime.datetime.now()
# Import app
import app
t2 = datetime.datetime.now()
print(f"Import time: {(t2-t1).total_seconds():.2f}s")

# Check if we have test data
import os
test_file = "file_check_output.txt"
if not os.path.exists(test_file):
    print("No test data found")
    sys.exit(1)

# Load data
df = pd.read_csv(test_file, sep='\t')
print(f"Loaded {len(df)} rows")

# Run prediction
t3 = datetime.datetime.now()
result_df, vehicles = app.predict_trips(df)
t4 = datetime.datetime.now()

print(f"Predict time: {(t4-t3).total_seconds():.2f}s")
print(f"Total trips: {result_df['Trip'].nunique()}")
print(f"Vehicles: {vehicles}")
print(f"✅ DONE")
