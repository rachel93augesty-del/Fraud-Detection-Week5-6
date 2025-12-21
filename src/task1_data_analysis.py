# src/task1_data_analysis.py
# -------------------------------------------------
# Task 1: Data Analysis and Preprocessing
# -------------------------------------------------

from http.client import PROCESSING
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import warnings
# Add these imports at the TOP of task1_analysis.py
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")

# =================================================
# 1. DATA CLEANING
# =================================================

def clean_creditcard_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Credit Card Fraud dataset:
    - Remove duplicates
    - Ensure correct data types
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)
    df["Class"] = df["Class"].astype(int)
    return df


def clean_fraud_data(
    fraud_df: pd.DataFrame,
    ip_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean Fraud_Data.csv and IpAddress_to_Country.csv
    """
    fraud_df = fraud_df.copy()
    ip_df = ip_df.copy()

    # Remove duplicates
    fraud_df.drop_duplicates(inplace=True)

    # Convert datetime columns
    fraud_df["signup_time"] = pd.to_datetime(fraud_df["signup_time"])
    fraud_df["purchase_time"] = pd.to_datetime(fraud_df["purchase_time"])

    # Convert IP to integer
    fraud_df["ip_address"] = fraud_df["ip_address"].astype(int)

    # Convert categoricals
    for col in ["browser", "source", "sex"]:
        fraud_df[col] = fraud_df[col].astype("category")

    # IP-Country dataset
    ip_df.drop_duplicates(inplace=True)
    ip_df["lower_bound_ip_address"] = ip_df["lower_bound_ip_address"].astype(int)
    ip_df["upper_bound_ip_address"] = ip_df["upper_bound_ip_address"].astype(int)
    ip_df["country"] = ip_df["country"].astype("category")

    return fraud_df, ip_df

# =================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =================================================

def plot_class_distribution(df: pd.DataFrame, target_col: str, title: str):
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def univariate_analysis(df: pd.DataFrame, numeric_cols: list):
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()


def bivariate_analysis(df: pd.DataFrame, feature: str, target: str, kind: str = "box"):
    plt.figure(figsize=(6, 4))
    if kind == "box":
        sns.boxplot(x=target, y=feature, data=df)
    elif kind == "violin":
        sns.violinplot(x=target, y=feature, data=df)
    elif kind == "bar":
        sns.barplot(x=target, y=feature, data=df)
    plt.title(f"{feature} vs {target}")
    plt.tight_layout()
    plt.show()
    
# =================================================
# 3. GEOLOCATION INTEGRATION (Fraud dataset)
# =================================================

def map_ip_to_country(fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
    fraud_df = fraud_df.copy()
    ip_df_sorted = ip_df.sort_values("lower_bound_ip_address")
    fraud_df = pd.merge_asof(
        fraud_df.sort_values("ip_address"),
        ip_df_sorted,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward"
    )
    fraud_df = fraud_df[fraud_df["ip_address"] <= fraud_df["upper_bound_ip_address"]]
    return fraud_df

def fraud_by_country(df: pd.DataFrame):
    country_fraud = df.groupby("country")["class"].mean().sort_values(ascending=False)
    return country_fraud
# =================================================
# # 6. FRAUD BY COUNTRY ANALYSIS
# =================================================
def fraud_by_country_analysis(
    df: pd.DataFrame,
    target_col: str = 'class',
    country_col: str = 'country',
    min_transactions: int = 10,
    plot: bool = True
) -> pd.DataFrame:
    """
    Compute fraud statistics by country:
    - Total transactions
    - Fraudulent transactions
    - Fraud rate
    - Optionally plot bar chart for countries with at least `min_transactions`
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    if country_col not in df.columns:
        raise ValueError("Country column not found. Please run geolocation mapping first.")

    # Aggregate per country
    country_stats = (
        df.groupby(country_col)
        .agg(
            total_transactions=pd.NamedAgg(column=target_col, aggfunc='count'),
            fraud_count=pd.NamedAgg(column=target_col, aggfunc='sum')
        )
        .reset_index()
    )
    country_stats['fraud_rate'] = country_stats['fraud_count'] / country_stats['total_transactions']

    # Filter by minimum transactions
    country_stats_filtered = country_stats[country_stats['total_transactions'] >= min_transactions]

    # Optional plot
    if plot:
        plt.figure(figsize=(12,6))
        sns.barplot(
            data=country_stats_filtered.sort_values('fraud_rate', ascending=False),
            x='fraud_rate', y=country_col,
            palette='Reds_r'
        )
        plt.xlabel('Fraud Rate')
        plt.ylabel('Country')
        plt.title('Fraud Rate by Country (Min Transactions >= {})'.format(min_transactions))
        plt.tight_layout()
        plt.show()

    return country_stats_filtered.sort_values('fraud_rate', ascending=False)

## =================================================
# 4. FEATURE ENGINEERING (Fraud dataset)
# =================================================
# =================================================
# 0. LOAD DATA FUNCTION
# =================================================

def load_fraud_data(project_root: str) -> pd.DataFrame:
    """
    Load raw Fraud_Data.csv from the 'data/raw' folder
    """
    data_path = os.path.join(project_root, "data", "raw", "Fraud_Data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found at {data_path}")
    
    df = pd.read_csv(data_path)
    return df


def feature_engineering_fraud(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering for Fraud_Data.csv
    - Transaction count per user
    - Transaction velocity: last 24h and last 7d
    - Time-based features: hour_of_day, day_of_week
    - Time since signup in hours
    """
    df = df.copy()

    # Convert to datetime
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['signup_time'] = pd.to_datetime(df['signup_time'])

    # Sort for operations
    df = df.sort_values(['user_id', 'purchase_time'])

    # Transaction count per user
    df['transaction_count'] = df.groupby('user_id').cumcount() + 1

    # FIXED: Transactions in last 24h and 7d
    # We need to set the index to purchase_time for each group
    def calculate_rolling_counts(group):
        # Set purchase_time as index temporarily
        temp = group.set_index('purchase_time')
        # Calculate rolling counts
        temp['transactions_last_24h'] = temp.index.to_series().rolling('1D').count()
        temp['transactions_last_7d'] = temp.index.to_series().rolling('7D').count()
        return temp.reset_index()
    
    # Apply to each user group
    df = df.groupby('user_id', group_keys=False).apply(calculate_rolling_counts)
    
    # Time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek

    # Time since signup in hours
    df['time_since_signup_hours'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600

    return df

# ============================================================================
# DATA TRANSFORMATION FUNCTIONS (Step 5) AND CLASS IMBALAANCE (STEP 6)
# ============================================================================

# ============================================================================
# CELL 1: IMPORT LIBRARIES AND DEFINE ALL FUNCTIONS
# ============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully")

# ========== DATA TRANSFORMATION FUNCTIONS ==========

def transform_fraud_data(df):
    """
    Transform fraud dataset:
    - StandardScaler for numerical features
    - One-Hot Encoding for categorical features
    """
    if df is None or df.empty:
        print("❌ No data to transform")
        return None
    
    df_copy = df.copy()
    
    print("\nFraud Data Transformation:")
    print("-" * 40)
    
    # Keep target column
    target = None
    for col in ['isFraud', 'Class', 'fraud']:
        if col in df_copy.columns:
            target = col
            print(f"Target column: {target}")
            break
    
    # Remove high-cardinality columns to avoid memory issues
    high_card_cols = ['device_id', 'ip_address', 'user_id', 'signup_time', 'purchase_time']
    cols_to_drop = [col for col in high_card_cols if col in df_copy.columns]
    if cols_to_drop:
        df_copy = df_copy.drop(columns=cols_to_drop)
        print(f"Dropped high-cardinality columns: {cols_to_drop}")
    
    # One-Hot Encoding for categorical columns
    cat_cols = df_copy.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print(f"One-Hot Encoding {len(cat_cols)} categorical columns: {list(cat_cols)}")
        df_copy = pd.get_dummies(df_copy, columns=cat_cols, drop_first=True)
    
    # Scale numerical columns (exclude target)
    num_cols = df_copy.select_dtypes(include=[np.number]).columns
    if target and target in num_cols:
        num_cols = num_cols.drop(target)
    
    if len(num_cols) > 0:
        print(f"Scaling {len(num_cols)} numerical columns with StandardScaler")
        scaler = StandardScaler()
        df_copy[num_cols] = scaler.fit_transform(df_copy[num_cols])
    
    print(f"✓ Final shape: {df_copy.shape}")
    return df_copy

def transform_credit_data(df):
    """
    Transform credit dataset:
    - MinMaxScaler for numerical features
    - One-Hot Encoding for categorical features
    """
    if df is None or df.empty:
        print("❌ No data to transform")
        return None
    
    df_copy = df.copy()
    
    print("\nCredit Data Transformation:")
    print("-" * 40)
    
    # Keep target column
    target = None
    for col in ['Class', 'default', 'target']:
        if col in df_copy.columns:
            target = col
            print(f"Target column: {target}")
            break
    
    # One-Hot Encoding for categorical columns
    cat_cols = df_copy.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print(f"One-Hot Encoding {len(cat_cols)} categorical columns: {list(cat_cols)}")
        df_copy = pd.get_dummies(df_copy, columns=cat_cols, drop_first=True)
    
    # Scale numerical columns (exclude target)
    num_cols = df_copy.select_dtypes(include=[np.number]).columns
    if target and target in num_cols:
        num_cols = num_cols.drop(target)
    
    if len(num_cols) > 0:
        print(f"Scaling {len(num_cols)} numerical columns with MinMaxScaler")
        scaler = MinMaxScaler()
        df_copy[num_cols] = scaler.fit_transform(df_copy[num_cols])
    
    print(f"✓ Final shape: {df_copy.shape}")
    return df_copy

# ========== CLASS IMBALANCE FUNCTIONS ==========

def handle_fraud_imbalance(df):
    """
    Handle fraud data imbalance using SMOTE
    """
    if df is None or df.empty:
        print("❌ No data to balance")
        return None
    
    # Find target column
    target = None
    for col in ['isFraud', 'Class', 'fraud']:
        if col in df.columns:
            target = col
            break
    
    if not target:
        print("❌ No target column found")
        return df
    
    print(f"\nFraud Data - Class Imbalance Handling (SMOTE):")
    print("-" * 50)
    
    print("Class Distribution BEFORE:")
    before_counts = df[target].value_counts()
    for cls, count in before_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Class {cls}: {count:,} samples ({percentage:.2f}%)")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    balanced_df = pd.DataFrame(X_res, columns=X.columns)
    balanced_df[target] = y_res
    
    print("\nClass Distribution AFTER (SMOTE):")
    after_counts = balanced_df[target].value_counts()
    for cls, count in after_counts.items():
        percentage = (count / len(balanced_df)) * 100
        print(f"  Class {cls}: {count:,} samples ({percentage:.2f}%)")
    
    print(f"\nOriginal shape: {df.shape}")
    print(f"Balanced shape: {balanced_df.shape}")
    
    return balanced_df

def handle_credit_imbalance(df):
    """
    Handle credit data imbalance using RandomUnderSampler
    """
    if df is None or df.empty:
        print("❌ No data to balance")
        return None
    
    # Find target column
    target = None
    for col in ['Class', 'default', 'target']:
        if col in df.columns:
            target = col
            break
    
    if not target:
        print("❌ No target column found")
        return df
    
    print(f"\nCredit Data - Class Imbalance Handling (RandomUnderSampler):")
    print("-" * 50)
    
    print("Class Distribution BEFORE:")
    before_counts = df[target].value_counts()
    for cls, count in before_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Class {cls}: {count:,} samples ({percentage:.2f}%)")
    
    X = df.drop(columns=[target])
    y = df[target]
    
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    
    balanced_df = pd.DataFrame(X_res, columns=X.columns)
    balanced_df[target] = y_res
    
    print("\nClass Distribution AFTER (RandomUnderSampler):")
    after_counts = balanced_df[target].value_counts()
    for cls, count in after_counts.items():
        percentage = (count / len(balanced_df)) * 100
        print(f"  Class {cls}: {count:,} samples ({percentage:.2f}%)")
    
    print(f"\nOriginal shape: {df.shape}")
    print(f"Balanced shape: {balanced_df.shape}")
    
    return balanced_df

print("✅ All functions defined successfully")