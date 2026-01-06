import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, send_file, redirect, url_for, flash, jsonify, session
import io
import json
from werkzeug.utils import secure_filename
from datetime import datetime, date
import re
import warnings
import tempfile
import shutil

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Vercel-specific configuration
# Use /tmp directory for serverless functions (readable and writable)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['PROCESSED_FOLDER'] = '/tmp/processed'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB for Vercel
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'autoeda-secret-key-2024')

# Ensure tmp directories exist (Vercel allows writing to /tmp)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Also create static/plots in /tmp for Vercel compatibility
PLOTS_DIR = '/tmp/static/plots'
os.makedirs(PLOTS_DIR, exist_ok=True)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app.json_encoder = CustomJSONEncoder

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_column(df, col):
    """Preprocess column before type detection"""
    if df[col].dtype == 'object':
        # Remove leading/trailing whitespace
        df[col] = df[col].astype(str).str.strip()

        # Convert empty strings and common missing value indicators to NaN
        missing_indicators = ['', 'nan', 'null', 'none', 'na', 'n/a', 'nil', 'missing', '-', '--', '---', 'null',
                              'undefined', 'nan', 'NaN', 'NAN', 'Nan', 'NULL', 'Null', 'NONE', 'None', 'NA', 'N/A']
        df[col] = df[col].replace(missing_indicators, np.nan)

        # Handle case where entire column might be strings like "NaN"
        df[col] = df[col].replace('NaN', np.nan)

    return df[col]


def detect_column_types(df):
    """Advanced column type detection with comprehensive preprocessing"""
    column_info = {}
    cleaning_log = []

    for col in df.columns:
        # Preprocess the column
        df[col] = preprocess_column(df, col)

        # Skip if all values are NaN after preprocessing
        if df[col].isna().all():
            column_info[col] = 'text'
            cleaning_log.append(f"'{col}': All values missing, marked as text")
            continue

        original_non_null = df[col].notna().sum()
        col_lower = col.lower()

        # ====== 1. DATE DETECTION (with multiple formats) ======
        date_formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%d %m %Y',
            '%Y/%m/%d', '%Y-%m-%d', '%Y.%m.%d', '%Y %m %d',
            '%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y', '%m %d %Y'
        ]

        date_keywords = ['date', 'time', 'datetime', 'timestamp', 'joined', 'hire',
                         'start', 'end', 'birth', 'created', 'updated', 'join']

        if any(keyword in col_lower for keyword in date_keywords):
            best_format = None
            best_success = 0

            for date_format in date_formats:
                try:
                    converted = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    success_rate = converted.notna().mean()

                    if success_rate > best_success:
                        best_success = success_rate
                        best_format = date_format

                    if success_rate > 0.8:  # High confidence
                        df[col] = converted
                        column_info[col] = 'date'
                        date_conversions = converted.notna().sum() - (original_non_null - df[col].isna().sum())
                        if date_conversions > 0:
                            cleaning_log.append(
                                f"'{col}': {date_conversions} values converted to date (format: {date_format})")
                        break
                except:
                    continue

            # If no format worked perfectly, try infer_datetime_format
            if 'date' not in column_info.get(col, ''):
                try:
                    converted = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if converted.notna().mean() > 0.6:
                        df[col] = converted
                        column_info[col] = 'date'
                        cleaning_log.append(f"'{col}': Auto-detected as date format")
                except:
                    pass

        # ====== 2. NUMERIC DETECTION (skip if already date) ======
        if col not in column_info or column_info[col] != 'date':
            numeric_keywords = ['age', 'salary', 'income', 'price', 'cost', 'amount', 'value',
                                'score', 'rating', 'percentage', 'percent', 'rate', 'count',
                                'years', 'months', 'days', 'hours', 'minutes', 'seconds',
                                'experience', 'performance', 'level', 'id', 'no', 'number',
                                'amount', 'price', 'cost', 'fee', 'charge', 'payment',
                                'height', 'weight', 'distance', 'length', 'width', 'depth',
                                'quantity', 'total', 'sum', 'average', 'mean', 'median']

            # Enhanced numeric conversion
            if any(keyword in col_lower for keyword in numeric_keywords) or col_lower.endswith('id'):
                try:
                    # Remove commas, currency symbols, etc.
                    temp_series = df[col].copy()

                    # Handle common numeric issues
                    if temp_series.dtype == 'object':
                        # Remove currency symbols, commas, spaces
                        temp_series = temp_series.astype(str).str.replace(r'[$,£€¥₹]', '', regex=True)
                        temp_series = temp_series.str.replace(',', '')
                        temp_series = temp_series.str.replace(' ', '')

                        # Convert common text to NaN
                        text_nan = ['unknown', 'not rated', 'n/a', 'na', 'null', 'none', 'missing', '-', '--']
                        temp_series = temp_series.replace(text_nan, np.nan)

                        # Try to convert
                        numeric_converted = pd.to_numeric(temp_series, errors='coerce')
                    else:
                        numeric_converted = pd.to_numeric(df[col], errors='coerce')

                    numeric_success_rate = numeric_converted.notna().mean()

                    if numeric_success_rate > 0.3:  # Lower threshold for numeric
                        df[col] = numeric_converted
                        column_info[col] = 'numeric'

                        # Count conversions
                        non_numeric_count = (numeric_converted.isna().sum() - df[col].isna().sum())
                        if non_numeric_count > 0:
                            cleaning_log.append(f"'{col}': {non_numeric_count} non-numeric values converted to NaN")
                        continue
                except Exception as e:
                    cleaning_log.append(f"'{col}': Error in numeric detection: {str(e)}")

            # General numeric detection
            try:
                temp_series = df[col].copy()
                if temp_series.dtype == 'object':
                    # Clean the series
                    temp_series = temp_series.astype(str).str.replace(r'[$,£€¥₹,]', '', regex=True)
                    temp_series = temp_series.str.replace(' ', '')
                    temp_series = temp_series.replace(['unknown', 'null', 'none', 'nan', 'NaN'], np.nan)

                numeric_converted = pd.to_numeric(temp_series, errors='coerce')
                numeric_success_rate = numeric_converted.notna().mean()

                if numeric_success_rate > 0.7:  # High threshold for general detection
                    df[col] = numeric_converted
                    column_info[col] = 'numeric'
                    non_numeric_count = (numeric_converted.isna().sum() - df[col].isna().sum())
                    if non_numeric_count > 0:
                        cleaning_log.append(f"'{col}': {non_numeric_count} non-numeric values converted to NaN")
                    continue
            except:
                pass

        # ====== 3. BOOLEAN DETECTION ======
        if df[col].dtype == 'object':
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 5:
                # Check for boolean-like values
                bool_like = all(str(v).lower() in ['true', 'false', 'yes', 'no', '1', '0', 'y', 'n', 't', 'f']
                                for v in unique_values if pd.notna(v))
                if bool_like:
                    column_info[col] = 'boolean'
                    # Convert to proper boolean
                    mapping = {
                        'true': True, 'false': False,
                        'yes': True, 'no': False,
                        'y': True, 'n': False,
                        't': True, 'f': False,
                        '1': True, '0': False
                    }
                    df[col] = df[col].astype(str).str.lower().map(mapping)
                    cleaning_log.append(f"'{col}': Converted to boolean")
                    continue

        # ====== 4. CATEGORICAL/TEXT DETECTION ======
        if col not in column_info:
            # Check unique values ratio
            unique_count = df[col].nunique()
            total_count = df[col].notna().sum()

            if total_count > 0:
                unique_ratio = unique_count / total_count

                if unique_ratio < 0.3 and unique_count < 50:
                    column_info[col] = 'categorical'

                    # Standardize categorical values
                    if df[col].dtype == 'object':
                        # Trim and title case for categorical
                        df[col] = df[col].astype(str).str.strip().str.title()
                        cleaning_log.append(f"'{col}': Standardized as categorical (title case)")
                else:
                    column_info[col] = 'text'

                    # Clean text columns
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.strip()
            else:
                column_info[col] = 'text'

    return df, column_info, cleaning_log


def validate_and_correct_numeric(df, col, column_types):
    """Validate numeric columns and correct invalid values"""
    corrections = []

    if column_types.get(col) == 'numeric':
        # Ensure it's numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')

        # Detect and handle invalid values
        if col.lower() in ['age', 'years']:
            # Age validation: 0-120
            invalid_mask = (df[col] < 0) | (df[col] > 120)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                corrections.append(f"{invalid_count} invalid ages (<0 or >120) set to NaN")
                df.loc[invalid_mask, col] = np.nan

        elif 'salary' in col.lower() or 'income' in col.lower() or 'amount' in col.lower():
            # Salary validation: positive values
            invalid_mask = df[col] < 0
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                corrections.append(f"{invalid_count} negative values set to NaN")
                df.loc[invalid_mask, col] = np.nan

        elif 'id' in col.lower() or col.lower().endswith('id'):
            # ID validation: positive integers
            invalid_mask = df[col] < 0
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                corrections.append(f"{invalid_count} negative IDs set to NaN")
                df.loc[invalid_mask, col] = np.nan

        # Detect outliers using IQR
        valid_values = df[col].dropna()
        if len(valid_values) >= 10:  # Only if we have enough data
            Q1 = valid_values.quantile(0.25)
            Q3 = valid_values.quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:
                lower_bound = Q1 - 3 * IQR  # 3x IQR for extreme outliers
                upper_bound = Q3 + 3 * IQR

                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()

                if outlier_count > 0:
                    corrections.append(f"{outlier_count} extreme outliers detected")
                    # Don't auto-correct outliers, just report them

    return df, corrections


def clean_dataset(df, column_types, user_overrides=None):
    """Comprehensive cleaning for any messy dataset"""
    df_cleaned = df.copy()
    cleaning_report = {
        'duplicates_removed': 0,
        'missing_values_filled': 0,
        'columns_converted': {},
        'invalid_values_corrected': {},
        'cleaning_log': [],
        'data_validation': {},
        'outliers_detected': {}
    }

    # ====== 1. APPLY USER OVERRIDES ======
    if user_overrides:
        for col, new_type in user_overrides.items():
            if col in df_cleaned.columns:
                original_type = column_types.get(col, 'unknown')
                column_types[col] = new_type

                if new_type == 'numeric' and original_type != 'numeric':
                    # Convert to numeric
                    original_non_numeric = (pd.to_numeric(df_cleaned[col], errors='coerce').isna().sum() -
                                            df_cleaned[col].isna().sum())
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    cleaning_report['columns_converted'][
                        col] = f'User override: Converted from {original_type} to numeric'
                    if original_non_numeric > 0:
                        cleaning_report['invalid_values_corrected'][col] = original_non_numeric

                elif new_type == 'date' and original_type != 'date':
                    try:
                        df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce', infer_datetime_format=True)
                        cleaning_report['columns_converted'][
                            col] = f'User override: Converted from {original_type} to date'
                    except:
                        cleaning_report['cleaning_log'].append(f"'{col}': Failed to convert to date")

                elif new_type == 'categorical':
                    cleaning_report['columns_converted'][col] = f'User override: Set as categorical'

    # ====== 2. VALIDATE AND CORRECT DATA ======
    for col in df_cleaned.columns:
        col_type = column_types.get(col, 'text')

        # Validate numeric columns
        if col_type == 'numeric':
            df_cleaned, corrections = validate_and_correct_numeric(df_cleaned, col, column_types)
            if corrections:
                cleaning_report['data_validation'][col] = corrections

        # Clean text/categorical columns
        elif col_type in ['categorical', 'text']:
            if df_cleaned[col].dtype == 'object':
                # Standardize text
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()

                # For categorical, use title case
                if col_type == 'categorical':
                    df_cleaned[col] = df_cleaned[col].str.title()

                # Handle empty strings
                df_cleaned[col] = df_cleaned[col].replace(['', 'nan', 'NaN', 'null', 'None'], np.nan)

        # Format date columns
        elif col_type == 'date':
            try:
                # Ensure datetime
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                # Format to DD/MM/YYYY
                df_cleaned[col] = df_cleaned[col].dt.strftime('%d/%m/%Y')
                # Convert NaT to empty string
                df_cleaned[col] = df_cleaned[col].replace('NaT', '')
            except:
                cleaning_report['cleaning_log'].append(f"'{col}': Error formatting dates")

    # ====== 3. REMOVE DUPLICATES ======
    initial_rows = len(df_cleaned)

    # First, clean up whitespace in all columns for duplicate detection
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()

    df_cleaned = df_cleaned.drop_duplicates()
    cleaning_report['duplicates_removed'] = initial_rows - len(df_cleaned)

    # ====== 4. HANDLE MISSING VALUES ======
    for col in df_cleaned.columns:
        col_type = column_types.get(col, 'text')
        missing_count = df_cleaned[col].isnull().sum()

        if missing_count > 0:
            if col_type == 'numeric':
                # Fill with median
                if not df_cleaned[col].empty:
                    valid_values = df_cleaned[col].dropna()
                    if len(valid_values) > 0:
                        fill_value = valid_values.median()
                    else:
                        fill_value = 0

                    df_cleaned[col].fillna(fill_value, inplace=True)
                    cleaning_report['missing_values_filled'] += missing_count
                    cleaning_report['cleaning_log'].append(
                        f"'{col}': {missing_count} missing values filled with median ({fill_value:.2f})")

            elif col_type in ['categorical', 'text']:
                # Fill with mode
                if not df_cleaned[col].empty:
                    mode_values = df_cleaned[col].dropna().mode()
                    if not mode_values.empty:
                        fill_value = mode_values[0]
                    else:
                        fill_value = 'Unknown' if col_type == 'categorical' else ''

                    df_cleaned[col].fillna(fill_value, inplace=True)
                    cleaning_report['missing_values_filled'] += missing_count
                    cleaning_report['cleaning_log'].append(
                        f"'{col}': {missing_count} missing values filled with '{fill_value}'")

            elif col_type == 'date':
                # For dates, fill with mode or leave empty
                if not df_cleaned[col].empty:
                    # Convert back to datetime for mode calculation
                    temp_dates = pd.to_datetime(df_cleaned[col], errors='coerce')
                    mode_dates = temp_dates.dropna().mode()

                    if not mode_dates.empty:
                        fill_value = mode_dates[0].strftime('%d/%m/%Y')
                        df_cleaned[col].fillna(fill_value, inplace=True)
                        cleaning_report['missing_values_filled'] += missing_count

    # ====== 5. FINAL DATA TYPE ENFORCEMENT ======
    for col, col_type in column_types.items():
        if col in df_cleaned.columns:
            if col_type == 'numeric':
                # Ensure proper numeric type
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                # Round to 2 decimal places for display
                if df_cleaned[col].dtype in ['float64', 'float32']:
                    df_cleaned[col] = df_cleaned[col].round(2)

            elif col_type == 'boolean':
                # Ensure boolean type
                df_cleaned[col] = df_cleaned[col].astype('bool')

    # ====== 6. OUTLIER DETECTION ======
    numeric_cols = [col for col, col_type in column_types.items()
                    if col_type == 'numeric' and col in df_cleaned.columns]

    for col in numeric_cols:
        if col in df_cleaned.columns and pd.api.types.is_numeric_dtype(df_cleaned[col]):
            numeric_data = pd.to_numeric(df_cleaned[col], errors='coerce')
            Q1 = numeric_data.quantile(0.25)
            Q3 = numeric_data.quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = numeric_data[(numeric_data < lower_bound) | (numeric_data > upper_bound)]
                if len(outliers) > 0:
                    cleaning_report['outliers_detected'][col] = {
                        'count': len(outliers),
                        'percentage': (len(outliers) / len(numeric_data)) * 100,
                        'range': f"{lower_bound:.2f} to {upper_bound:.2f}",
                        'values': outliers.tolist()[:10]  # First 10 outlier values
                    }

    return df_cleaned, column_types, cleaning_report


def generate_summary(df, column_types, cleaning_report):
    """Generate comprehensive summary"""
    summary = {
        'dataset_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        },
        'cleaning_summary': cleaning_report,
        'column_analysis': [],
        'numeric_stats': {},
        'categorical_stats': {},
        'missing_values': {}
    }

    # Column analysis
    for col in df.columns:
        col_type = column_types.get(col, 'unknown')
        col_info = {
            'name': col,
            'type': col_type,
            'dtype': str(df[col].dtype),
            'unique_values': df[col].nunique(),
            'missing_values': int(df[col].isnull().sum()),
            'missing_percentage': float((df[col].isnull().sum() / len(df)) * 100)
        }

        if col_type == 'numeric':
            numeric_data = pd.to_numeric(df[col], errors='coerce')
            if numeric_data.notna().any():
                col_info.update({
                    'mean': float(numeric_data.mean()),
                    'median': float(numeric_data.median()),
                    'std': float(numeric_data.std()) if len(numeric_data) > 1 else 0.0,
                    'min': float(numeric_data.min()),
                    'max': float(numeric_data.max()),
                    'q1': float(numeric_data.quantile(0.25)),
                    'q3': float(numeric_data.quantile(0.75))
                })
        elif col_type == 'categorical':
            if not df[col].empty:
                mode_series = df[col].mode()
                col_info['mode'] = str(mode_series[0]) if not mode_series.empty else None
                col_info['top_values'] = df[col].value_counts().head(5).to_dict()

        for key, value in col_info.items():
            if isinstance(value, (np.integer, np.int64)):
                col_info[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                col_info[key] = float(value)
            elif pd.isna(value):
                col_info[key] = None

        summary['column_analysis'].append(col_info)
        summary['missing_values'][col] = col_info['missing_values']

    # Numeric statistics
    numeric_cols = [col for col, col_type in column_types.items()
                    if col_type == 'numeric' and col in df.columns]

    if numeric_cols:
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        if not numeric_df.empty:
            stats = numeric_df.describe(percentiles=[.25, .5, .75])
            summary['numeric_stats'] = {
                col: {
                    stat: float(value) if not pd.isna(value) else None
                    for stat, value in col_stats.items()
                }
                for col, col_stats in stats.to_dict().items()
            }

    # Categorical statistics
    categorical_cols = [col for col, col_type in column_types.items()
                        if col_type == 'categorical' and col in df.columns]

    for col in categorical_cols:
        if col in df.columns:
            value_counts = df[col].value_counts().head(10)
            summary['categorical_stats'][col] = {
                'unique_count': int(df[col].nunique()),
                'top_values': {str(k): int(v) for k, v in value_counts.to_dict().items()}
            }

    return summary


def generate_visualizations(df, column_types, filename):
    plot_paths = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Use Vercel-compatible plots directory
    plots_dir = PLOTS_DIR
    os.makedirs(plots_dir, exist_ok=True)

    numeric_cols = [col for col, col_type in column_types.items()
                    if col_type == 'numeric' and col in df.columns]

    numeric_df = None
    if numeric_cols:
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        numeric_df = numeric_df.dropna(axis=1, how='all')

    categorical_cols = [col for col, col_type in column_types.items()
                        if col_type == 'categorical' and col in df.columns]

    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    if numeric_df is not None and not numeric_df.empty:
        valid_numeric_cols = [col for col in numeric_df.columns if numeric_df[col].notna().any()]
        if valid_numeric_cols:
            try:
                # Histograms
                n_cols = min(3, len(valid_numeric_cols))
                n_rows = int(np.ceil(len(valid_numeric_cols) / n_cols))

                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
                axes = axes.flatten() if len(valid_numeric_cols) > 1 else [axes]

                for idx, col in enumerate(valid_numeric_cols[:len(axes)]):
                    col_data = numeric_df[col].dropna()
                    if len(col_data) > 0:
                        axes[idx].hist(col_data, bins=20, alpha=0.7, color='#2563eb', edgecolor='black')
                        axes[idx].set_title(f'{col}', fontsize=12)
                        axes[idx].set_xlabel('Value')
                        axes[idx].set_ylabel('Frequency')

                for idx in range(len(valid_numeric_cols), len(axes)):
                    axes[idx].set_visible(False)

                plt.suptitle('Numeric Distributions', fontsize=14)
                plt.tight_layout()
                hist_path = os.path.join(plots_dir, f'hist_{timestamp}.png')
                plt.savefig(hist_path, dpi=100, bbox_inches='tight')
                plt.close()
                plot_paths['histogram'] = f'/tmp/static/plots/hist_{timestamp}.png'
            except Exception as e:
                print(f"Error generating histograms: {e}")

        # Correlation heatmap
        if len(valid_numeric_cols) > 1:
            try:
                correlation = numeric_df[valid_numeric_cols].corr()
                if not correlation.empty and correlation.notna().any().any():
                    plt.figure(figsize=(10, 8))
                    mask = np.triu(np.ones_like(correlation, dtype=bool))
                    sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f',
                                cmap='coolwarm', center=0, square=True,
                                cbar_kws={"shrink": .8}, linewidths=1)
                    plt.title('Correlation Heatmap', fontsize=14, pad=20)
                    heatmap_path = os.path.join(plots_dir, f'heatmap_{timestamp}.png')
                    plt.savefig(heatmap_path, dpi=100, bbox_inches='tight')
                    plt.close()
                    plot_paths['heatmap'] = f'/tmp/static/plots/heatmap_{timestamp}.png'
            except Exception as e:
                print(f"Error generating heatmap: {e}")

        # Boxplots
        if len(valid_numeric_cols) > 0 and len(valid_numeric_cols) <= 10:
            try:
                plt.figure(figsize=(12, 6))
                numeric_df[valid_numeric_cols].boxplot()
                plt.title('Boxplot - Outlier Detection', fontsize=14, pad=20)
                plt.xticks(rotation=45)
                plt.tight_layout()
                boxplot_path = os.path.join(plots_dir, f'boxplot_{timestamp}.png')
                plt.savefig(boxplot_path, dpi=100, bbox_inches='tight')
                plt.close()
                plot_paths['boxplot'] = f'/tmp/static/plots/boxplot_{timestamp}.png'
            except Exception as e:
                print(f"Error generating boxplot: {e}")

    # Bar charts for categorical
    if categorical_cols:
        try:
            valid_categorical_cols = [col for col in categorical_cols if col in df.columns and df[col].nunique() > 0]
            if valid_categorical_cols:
                n_cats = min(4, len(valid_categorical_cols))
                fig, axes = plt.subplots(1, n_cats, figsize=(n_cats * 5, 4))
                if n_cats == 1:
                    axes = [axes]

                for idx, col in enumerate(valid_categorical_cols[:n_cats]):
                    if col in df.columns:
                        top_values = df[col].value_counts().head(5)
                        if not top_values.empty:
                            colors = plt.cm.Blues(np.linspace(0.5, 0.9, len(top_values)))
                            axes[idx].bar(range(len(top_values)), top_values.values, color=colors, alpha=0.8)
                            col_name = col[:15] + '...' if len(col) > 15 else col
                            axes[idx].set_title(f'{col_name}', fontsize=11)
                            axes[idx].set_xticks(range(len(top_values)))
                            axes[idx].set_xticklabels([str(x)[:10] + '...' if len(str(x)) > 10 else str(x)
                                                       for x in top_values.index], rotation=45, ha='right')
                            axes[idx].set_ylabel('Count')

                plt.suptitle('Top Categories', fontsize=14)
                plt.tight_layout()
                barchart_path = os.path.join(plots_dir, f'barchart_{timestamp}.png')
                plt.savefig(barchart_path, dpi=100, bbox_inches='tight')
                plt.close()
                plot_paths['barchart'] = f'/tmp/static/plots/barchart_{timestamp}.png'
        except Exception as e:
            print(f"Error generating barchart: {e}")

    if not plot_paths:
        try:
            plt.figure(figsize=(8, 4))
            plt.text(0.5, 0.5, 'No visualizations available\nDataset may not have suitable columns',
                     ha='center', va='center', fontsize=12)
            plt.axis('off')
            no_plot_path = os.path.join(plots_dir, f'noplot_{timestamp}.png')
            plt.savefig(no_plot_path, dpi=100, bbox_inches='tight')
            plt.close()
            plot_paths['noplot'] = f'/tmp/static/plots/noplot_{timestamp}.png'
        except Exception as e:
            print(f"Error generating placeholder plot: {e}")

    return plot_paths


def get_preview_data(df, column_types, filename):
    preview_data = {
        'filename': filename,
        'shape': list(df.shape),
        'columns': list(df.columns.tolist()),
        'column_types': column_types,
        'sample': []
    }

    sample_df = df.head(10)

    for _, row in sample_df.iterrows():
        sample_row = {}
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                sample_row[col] = None
            elif isinstance(value, (np.integer, np.int64)):
                sample_row[col] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                sample_row[col] = float(value)
            elif isinstance(value, (datetime, pd.Timestamp)):
                sample_row[col] = value.strftime('%d/%m/%Y')
            else:
                sample_row[col] = str(value)
        preview_data['sample'].append(sample_row)

    return preview_data


# ====== ROUTES ======
@app.route('/')
def index():
    session_keys = list(session.keys())
    for key in session_keys:
        if key != '_flashes':
            session.pop(key, None)
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected', 'error')
        return redirect('/')

    file = request.files['file']

    if file.filename == '':
        flash('No file selected', 'error')
        return redirect('/')

    if not allowed_file(file.filename):
        flash('Please upload a CSV file only', 'error')
        return redirect('/')

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            flash(f'Error reading CSV file: {str(e)}', 'error')
            return redirect('/')

        if df.empty:
            flash('Uploaded CSV file is empty', 'error')
            return redirect('/')

        df.columns = df.columns.str.strip()
        df, column_types, cleaning_log = detect_column_types(df)
        preview_data = get_preview_data(df, column_types, filename)

        session['uploaded_filename'] = filename
        session['filepath'] = filepath
        session['column_types'] = column_types
        session['cleaning_log'] = cleaning_log

        return render_template('preview.html', preview=preview_data)

    except pd.errors.EmptyDataError:
        flash('The CSV file appears to be empty', 'error')
        return redirect('/')
    except pd.errors.ParserError:
        flash('Invalid CSV format. Please check your file.', 'error')
        return redirect('/')
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect('/')


@app.route('/process', methods=['POST'])
def process_data():
    try:
        if 'filepath' not in session:
            flash('No file uploaded. Please upload a file first.', 'error')
            return redirect('/')

        filepath = session['filepath']
        filename = session['uploaded_filename']
        original_column_types = session.get('column_types', {})

        df = pd.read_csv(filepath, low_memory=False)
        df.columns = df.columns.str.strip()

        user_overrides = {}
        for col in df.columns:
            override_key = f'type_{col}'
            if override_key in request.form and request.form[override_key]:
                user_overrides[col] = request.form[override_key]

        df, detected_types, _ = detect_column_types(df)
        column_types = {**detected_types, **user_overrides}
        df_cleaned, column_types, cleaning_report = clean_dataset(df, column_types, user_overrides)
        summary = generate_summary(df_cleaned, column_types, cleaning_report)
        plot_paths = generate_visualizations(df_cleaned, column_types, filename)

        if not isinstance(plot_paths, dict):
            plot_paths = {}

        cleaned_filename = f"cleaned_{filename}"
        cleaned_filepath = os.path.join(app.config['PROCESSED_FOLDER'], cleaned_filename)

        # Prepare final CSV
        df_to_save = df_cleaned.copy()

        # Format for CSV export
        for col in df_to_save.columns:
            col_type = column_types.get(col, 'text')

            if col_type == 'numeric':
                df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce')
                # Remove decimal if integer
                if df_to_save[col].dropna().apply(lambda x: x.is_integer() if isinstance(x, float) else True).all():
                    df_to_save[col] = df_to_save[col].astype('Int64')

            # Convert NaN to empty string for clean CSV
            df_to_save[col] = df_to_save[col].fillna('')

        os.makedirs(os.path.dirname(cleaned_filepath), exist_ok=True)
        df_to_save.to_csv(cleaned_filepath, index=False)

        # Success message
        success_msg = f"✅ Dataset cleaned successfully! "
        success_msg += f"{cleaning_report['duplicates_removed']} duplicates removed. "
        success_msg += f"{cleaning_report['missing_values_filled']} missing values filled."

        if cleaning_report.get('invalid_values_corrected'):
            invalid_total = sum(cleaning_report['invalid_values_corrected'].values())
            success_msg += f" {invalid_total} invalid values corrected."

        if cleaning_report.get('outliers_detected'):
            outlier_total = sum(info['count'] for info in cleaning_report['outliers_detected'].values())
            success_msg += f" {outlier_total} outliers detected."

        flash(success_msg, 'success')

        return render_template('results.html',
                               summary=summary,
                               plot_paths=plot_paths,
                               cleaned_filename=cleaned_filename,
                               column_types=column_types,
                               cleaning_report=cleaning_report)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in process_data: {e}")
        print(f"Error details: {error_details}")

        flash(f'Error processing data: {str(e)}', 'error')
        return redirect('/')


@app.route('/download/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)

        if not os.path.exists(filepath):
            flash('File not found', 'error')
            return redirect('/')

        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'error')
        return redirect('/')


@app.route('/download_plot/<plot_type>')
def download_plot(plot_type):
    try:
        plot_dir = PLOTS_DIR
        if not os.path.exists(plot_dir):
            flash('Plots directory not found', 'error')
            return redirect('/')

        plot_files = [f for f in os.listdir(plot_dir) if plot_type in f]

        if not plot_files:
            flash('Plot not found', 'error')
            return redirect('/')

        latest_plot = sorted(plot_files)[-1]
        plot_path = os.path.join(plot_dir, latest_plot)

        return send_file(
            plot_path,
            as_attachment=True,
            download_name=f"{plot_type}_{datetime.now().strftime('%Y%m%d')}.png"
        )
    except Exception as e:
        flash(f'Error downloading plot: {str(e)}', 'error')
        return redirect('/')


@app.route('/clear_session')
def clear_session():
    session.clear()
    flash('Session cleared', 'info')
    return redirect('/')


@app.route('/debug_info')
def debug_info():
    info = {
        'session_keys': list(session.keys()),
        'upload_folder_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'processed_folder_exists': os.path.exists(app.config['PROCESSED_FOLDER']),
        'plots_folder_exists': os.path.exists(PLOTS_DIR),
        'upload_folder_files': os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(
            app.config['UPLOAD_FOLDER']) else [],
        'processed_folder_files': os.listdir(app.config['PROCESSED_FOLDER']) if os.path.exists(
            app.config['PROCESSED_FOLDER']) else [],
        'plots_folder_files': os.listdir(PLOTS_DIR) if os.path.exists(PLOTS_DIR) else []
    }
    return jsonify(info)


# Vercel serverless handler
@app.route('/static/plots/<path:filename>')
def serve_plot(filename):
    """Serve plots from /tmp directory"""
    try:
        plot_path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(plot_path):
            return send_file(plot_path)
        else:
            return "Plot not found", 404
    except:
        return "Error serving plot", 500


# Handler for Vercel
def handler(event, context):
    return app(event, context)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)