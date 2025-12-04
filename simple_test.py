"""
Simple test for debugging - minimal processing
"""
import time
from app import load_master_data, load_booking_history_restrictions, predict_trips

def simple_test():
    print("🚀 Simple Test - Minimal Processing")
    
    # Load minimal data
    import pandas as pd
    
    # Create simple test data
    test_data = {
        'Code': ['N967', 'NY01', 'NZ07', 'M195'],
        'Weight': [100, 120, 80, 300],
        'Cube': [1.0, 1.2, 0.8, 4.5]
    }
    
    test_df = pd.DataFrame(test_data)
    
    print(f"Test data: {len(test_df)} branches")
    
    start_time = time.time()
    
    # Simple model_data with dummy model
    from sklearn.tree import DecisionTreeClassifier
    dummy_model = DecisionTreeClassifier()
    
    model_data = {
        'model': dummy_model,
        'trip_pairs': {},
        'branch_info': {},
        'trip_vehicles': {},
        'branch_vehicles': {}
    }
    
    try:
        result_df, summary_df = predict_trips(test_df, model_data)
        
        elapsed = time.time() - start_time
        print(f"⚡ Completed in {elapsed:.1f} seconds")
        
        if result_df is not None:
            print("\n📊 Results:")
            for _, row in result_df.iterrows():
                print(f"{row['Code']}: Trip {row.get('Trip', 'N/A')}, Vehicle: {row.get('Truck', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()