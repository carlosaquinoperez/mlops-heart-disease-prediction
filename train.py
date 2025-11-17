import pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from helpers import to_dicts

print("--- [train.py] Script started ---")

# --- 1. Define Features ---
categorical_features = [
    'gender', 'cholesterol', 'glucose', 'is_smoker', 'is_alcoholic', 'is_active'
]
numerical_features = [
    'age_years', 'height_cm', 'weight_kg', 'systolic_bp', 'diastolic_bp'
]

def load_and_prepare_data(filepath):
    """Loads, cleans, and splits the data."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, sep=';')
    
    # Rename columns
    column_mapping = {
        'age': 'age_days', 'height': 'height_cm', 'weight': 'weight_kg',
        'ap_hi': 'systolic_bp', 'ap_lo': 'diastolic_bp', 'gluc': 'glucose',
        'smoke': 'is_smoker', 'alco': 'is_alcoholic', 'active': 'is_active',
        'cardio': 'target'
    }
    df = df.rename(columns=column_mapping)
    
    # Feature Engineering (age in years)
    df['age_years'] = (df['age_days'] / 365.25).round().astype(int)
    
    # Drop unused columns
    df = df.drop(columns=['id', 'age_days'])
    
    print("Splitting data (80% train/val, 20% test)...")
    # We split off a test set, but we train the *final* model on the rest
    df_full_train, _ = train_test_split(df, test_size=0.2, random_state=1)
    
    return df_full_train

def create_pipeline():
    """Defines and returns the scikit-learn pipeline."""
    print("Creating preprocessing and modeling pipeline...")
    
    preprocessor = ColumnTransformer(
        [
            ('num_scaler', StandardScaler(), numerical_features),
            ('cat_encoder', make_pipeline(
                FunctionTransformer(to_dicts),
                DictVectorizer(sparse=False)
            ), categorical_features)
        ],
        remainder='drop'
    )
    
    final_pipeline = make_pipeline(
        preprocessor,
        LogisticRegression(solver='lbfgs', max_iter=1000, random_state=1)
    )
    
    return final_pipeline

# --- 3. Main Execution ---
if __name__ == "__main__":
    
    # Define file paths
    input_file = './data/cardio_train.csv'
    output_file = 'model.bin'
    
    # Load and prepare data
    df_full_train = load_and_prepare_data(input_file)
    
    # Get features (X) and target (y)
    y_full_train = df_full_train['target'].values
    X_full_train = df_full_train.drop(columns=['target'])
    
    # Create the pipeline
    pipeline = create_pipeline()
    
    # Train the pipeline
    print("Training the full pipeline...")
    pipeline.fit(X_full_train, y_full_train)
    print("Pipeline training complete!")
    
    # Save the pipeline
    print(f"Saving final pipeline to: {output_file}")
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)
    
    print(f"--- [train.py] Script finished ---")