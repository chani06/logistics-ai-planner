"""Test processing speed"""
import datetime
import pandas as pd
import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['STREAMLIT_RUNNING'] = 'false'
os.environ['DEBUG_TIMING'] = '1'  # Enable timing debug

print("Starting speed test...")

t1 = datetime.datetime.now()
# Import app
import app
t2 = datetime.datetime.now()
print(f"Import time: {(t2-t1).total_seconds():.2f}s")

# Load model
model_data = app.load_model()
if model_data is None:
    print("No model found")
    sys.exit(1)
print("Model loaded")

# Check if we have test data
test_file = "punthai_test_data.xlsx"
if not os.path.exists(test_file):
    print("No test data found")
    sys.exit(1)

# Load data
df = pd.read_excel(test_file)
print(f"Loaded {len(df)} rows")

# Run prediction - skip post-processing phases
t3 = datetime.datetime.now()

# Just test clustering phase (skip heavy phases)
app.SKIP_HEAVY_PHASES = True
result_df, vehicles = app.predict_trips(df, model_data)
t4 = datetime.datetime.now()

print(f"Predict time: {(t4-t3).total_seconds():.2f}s")
print(f"Total trips: {result_df['Trip'].nunique()}")
print(f"✅ DONE")
