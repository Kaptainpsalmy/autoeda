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
import base64
from werkzeug.utils import secure_filename
from datetime import datetime, date
import re
import warnings
import gc

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Vercel configuration
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['PROCESSED_FOLDER'] = '/tmp/processed'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max for Vercel
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'autoeda-secret-key-2024')

# Ensure tmp directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)


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


# ====== ALL YOUR ORIGINAL FUNCTIONS ======
def preprocess_column(df, col):
    """Preprocess column before type detection"""
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.strip()
        missing_indicators = ['', 'nan', 'null', 'none', 'na', 'n/a', 'nil', 'missing', '-', '--', '---',
                              'undefined', 'nan', 'NaN', 'NAN', 'Nan', 'NULL', 'Null', 'NONE', 'None', 'NA', 'N/A']
        df[col] = df[col].replace(missing_indicators, np.nan)
        df[col] = df[col].replace('NaN', np.nan)
    return df[col]


def detect_column_types(df):
    """Advanced column type detection with comprehensive preprocessing"""
    column_info = {}
    cleaning_log = []

    for col in df.columns:
        df[col] = preprocess_column(df, col)

        if df[col].isna().all():
            column_info[col] = 'text'
            cleaning_log.append(f"'{col}': All values missing, marked as text")
            continue

        original_non_null = df[col].notna().sum()
        col_lower = col.lower()

        # DATE DETECTION
        date_formats = [
            '%d/%m/%Y', '%d-%m-%Y', '%d.%m.%Y', '%d %m %Y',
            '%Y/%m/%d', '%Y-%m-%d', '%Y.%m.%d', '%Y %m %d',
            '%m/%d/%Y', '%m-%d-%Y', '%m.%d.%Y', '%m %d %Y'
        ]

        date_keywords = ['date', 'time', 'datetime', 'timestamp', 'joined', 'hire',
                         'start', 'end', 'birth', 'created', 'updated', 'join']

        if any(keyword in col_lower for keyword in date_keywords):
            for date_format in date_formats:
                try:
                    converted = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    if converted.notna().mean() > 0.8:
                        df[col] = converted
                        column_info[col] = 'date'
                        break
                except:
                    continue

            if 'date' not in column_info.get(col, ''):
                try:
                    converted = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    if converted.notna().mean() > 0.6:
                        df[col] = converted
                        column_info[col] = 'date'
                except:
                    pass

        # NUMERIC DETECTION (skip if already date)
        if col not in column_info or column_info[col] != 'date':
            numeric_keywords = ['age', 'salary', 'income', 'price', 'cost', 'amount', 'value',
                                'score', 'rating', 'percentage', 'percent', 'rate', 'count',
                                'years', 'months', 'days', 'hours', 'minutes', 'seconds',
                                'experience', 'performance', 'level', 'id', 'no', 'number',
                                'height', 'weight', 'distance', 'length', 'width', 'depth',
                                'quantity', 'total', 'sum', 'average', 'mean', 'median']

            if any(keyword in col_lower for keyword in numeric_keywords) or col_lower.endswith('id'):
                try:
                    temp_series = df[col].copy()
                    if temp_series.dtype == 'object':
                        temp_series = temp_series.astype(str).str.replace(r'[$,£€¥₹]', '', regex=True)
                        temp_series = temp_series.str.replace(',', '')
                        temp_series = temp_series.str.replace(' ', '')
                        temp_series = temp_series.replace(
                            ['unknown', 'not rated', 'n/a', 'na', 'null', 'none', 'missing', '-', '--'], np.nan)
                        numeric_converted = pd.to_numeric(temp_series, errors='coerce')
                    else:
                        numeric_converted = pd.to_numeric(df[col], errors='coerce')

                    if numeric_converted.notna().mean() > 0.3:
                        df[col] = numeric_converted
                        column_info[col] = 'numeric'
                        continue
                except Exception as e:
                    cleaning_log.append(f"'{col}': Error in numeric detection: {str(e)}")

            # General numeric detection
            try:
                temp_series = df[col].copy()
                if temp_series.dtype == 'object':
                    temp_series = temp_series.astype(str).str.replace(r'[$,£€¥₹,]', '', regex=True)
                    temp_series = temp_series.str.replace(' ', '')
                    temp_series = temp_series.replace(['unknown', 'null', 'none', 'nan', 'NaN'], np.nan)

                numeric_converted = pd.to_numeric(temp_series, errors='coerce')
                if numeric_converted.notna().mean() > 0.7:
                    df[col] = numeric_converted
                    column_info[col] = 'numeric'
                    continue
            except:
                pass

        # BOOLEAN DETECTION
        if df[col].dtype == 'object':
            unique_values = df[col].dropna().unique()
            if len(unique_values) <= 5:
                bool_like = all(str(v).lower() in ['true', 'false', 'yes', 'no', '1', '0', 'y', 'n', 't', 'f']
                                for v in unique_values if pd.notna(v))
                if bool_like:
                    column_info[col] = 'boolean'
                    mapping = {
                        'true': True, 'false': False,
                        'yes': True, 'no': False,
                        'y': True, 'n': False,
                        't': True, 'f': False,
                        '1': True, '0': False
                    }
                    df[col] = df[col].astype(str).str.lower().map(mapping)
                    continue

        # CATEGORICAL/TEXT DETECTION
        if col not in column_info:
            unique_count = df[col].nunique()
            total_count = df[col].notna().sum()

            if total_count > 0:
                unique_ratio = unique_count / total_count
                if unique_ratio < 0.3 and unique_count < 50:
                    column_info[col] = 'categorical'
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.strip().str.title()
                else:
                    column_info[col] = 'text'
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.strip()
            else:
                column_info[col] = 'text'

    return df, column_info, cleaning_log


def validate_and_correct_numeric(df, col, column_types):
    """Validate numeric columns and correct invalid values"""
    corrections = []

    if column_types.get(col) == 'numeric':
        df[col] = pd.to_numeric(df[col], errors='coerce')

        if col.lower() in ['age', 'years']:
            invalid_mask = (df[col] < 0) | (df[col] > 120)
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                corrections.append(f"{invalid_count} invalid ages (<0 or >120) set to NaN")
                df.loc[invalid_mask, col] = np.nan

        elif 'salary' in col.lower() or 'income' in col.lower() or 'amount' in col.lower():
            invalid_mask = df[col] < 0
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                corrections.append(f"{invalid_count} negative values set to NaN")
                df.loc[invalid_mask, col] = np.nan

        elif 'id' in col.lower() or col.lower().endswith('id'):
            invalid_mask = df[col] < 0
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                corrections.append(f"{invalid_count} negative IDs set to NaN")
                df.loc[invalid_mask, col] = np.nan

        # Detect outliers
        valid_values = df[col].dropna()
        if len(valid_values) >= 10:
            Q1 = valid_values.quantile(0.25)
            Q3 = valid_values.quantile(0.75)
            IQR = Q3 - Q1

            if IQR > 0:
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    corrections.append(f"{outlier_count} extreme outliers detected")

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

    # Apply user overrides
    if user_overrides:
        for col, new_type in user_overrides.items():
            if col in df_cleaned.columns:
                original_type = column_types.get(col, 'unknown')
                column_types[col] = new_type

                if new_type == 'numeric' and original_type != 'numeric':
                    original_non_numeric = (pd.to_numeric(df_cleaned[col], errors='coerce').isna().sum() -
                                            df_cleaned[col].isna().sum())
                    df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                    cleaning_report['columns_converted'][
                        col] = f'User override: Converted from {original_type} to numeric'
                    if original_non_numeric > 0:
                        cleaning_report['invalid_values_corrected'][col] = original_non_numeric

                elif new_type == 'date' and original_type != 'date':
                    try:
                        df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                        cleaning_report['columns_converted'][
                            col] = f'User override: Converted from {original_type} to date'
                    except:
                        cleaning_report['cleaning_log'].append(f"'{col}': Failed to convert to date")

    # Validate and correct data
    for col in df_cleaned.columns:
        col_type = column_types.get(col, 'text')

        if col_type == 'numeric':
            df_cleaned, corrections = validate_and_correct_numeric(df_cleaned, col, column_types)
            if corrections:
                cleaning_report['data_validation'][col] = corrections

        elif col_type in ['categorical', 'text']:
            if df_cleaned[col].dtype == 'object':
                df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
                if col_type == 'categorical':
                    df_cleaned[col] = df_cleaned[col].str.title()
                df_cleaned[col] = df_cleaned[col].replace(['', 'nan', 'NaN', 'null', 'None'], np.nan)

        elif col_type == 'date':
            try:
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                df_cleaned[col] = df_cleaned[col].dt.strftime('%d/%m/%Y')
                df_cleaned[col] = df_cleaned[col].replace('NaT', '')
            except:
                cleaning_report['cleaning_log'].append(f"'{col}': Error formatting dates")

    # Remove duplicates
    initial_rows = len(df_cleaned)
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
    df_cleaned = df_cleaned.drop_duplicates()
    cleaning_report['duplicates_removed'] = initial_rows - len(df_cleaned)

    # Handle missing values
    for col in df_cleaned.columns:
        col_type = column_types.get(col, 'text')
        missing_count = df_cleaned[col].isnull().sum()

        if missing_count > 0:
            if col_type == 'numeric':
                if not df_cleaned[col].empty:
                    valid_values = df_cleaned[col].dropna()
                    fill_value = valid_values.median() if len(valid_values) > 0 else 0
                    df_cleaned[col].fillna(fill_value, inplace=True)
                    cleaning_report['missing_values_filled'] += missing_count

            elif col_type in ['categorical', 'text']:
                if not df_cleaned[col].empty:
                    mode_values = df_cleaned[col].dropna().mode()
                    fill_value = mode_values[
                        0] if not mode_values.empty else 'Unknown' if col_type == 'categorical' else ''
                    df_cleaned[col].fillna(fill_value, inplace=True)
                    cleaning_report['missing_values_filled'] += missing_count

    # Final data type enforcement
    for col, col_type in column_types.items():
        if col in df_cleaned.columns:
            if col_type == 'numeric':
                df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
                if df_cleaned[col].dtype in ['float64', 'float32']:
                    df_cleaned[col] = df_cleaned[col].round(2)

    # Outlier detection
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
                        'values': outliers.tolist()[:5]  # Limit to 5 values
                    }

    return df_cleaned, column_types, cleaning_report


def generate_summary(df, column_types, cleaning_report):
    """Generate comprehensive summary"""
    summary = {
        'dataset_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
        },
        'cleaning_summary': cleaning_report,
        'column_analysis': [],
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

        summary['column_analysis'].append(col_info)
        summary['missing_values'][col] = col_info['missing_values']

    return summary


def generate_visualizations(df, column_types, filename):
    """Generate visualizations and return as base64 strings"""
    plots = {}

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

    # 1. Histograms (limit to 2 columns for Vercel)
    if numeric_df is not None and not numeric_df.empty:
        valid_numeric_cols = [col for col in numeric_df.columns if numeric_df[col].notna().any()]
        if valid_numeric_cols:
            try:
                # Limit to first 2 columns for Vercel
                cols_to_plot = valid_numeric_cols[:2]
                n_cols = min(2, len(cols_to_plot))

                fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 4))
                if n_cols == 1:
                    axes = [axes]

                for idx, col in enumerate(cols_to_plot):
                    col_data = numeric_df[col].dropna()
                    if len(col_data) > 0:
                        axes[idx].hist(col_data, bins=15, alpha=0.7, color='#2563eb', edgecolor='black')
                        axes[idx].set_title(f'{col}', fontsize=12)
                        axes[idx].set_xlabel('Value')
                        axes[idx].set_ylabel('Frequency')

                plt.suptitle('Numeric Distributions', fontsize=14)
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')  # Lower DPI for Vercel
                plt.close()
                buf.seek(0)
                plots['histogram'] = base64.b64encode(buf.getvalue()).decode('utf-8')
            except Exception as e:
                print(f"Error generating histograms: {e}")

    # 2. Correlation heatmap (limit to 5 columns)
    if numeric_df is not None and len(valid_numeric_cols) > 1:
        try:
            # Limit to 5 columns for performance
            cols_for_heatmap = valid_numeric_cols[:5]
            correlation = df[cols_for_heatmap].corr()
            if not correlation.empty and correlation.notna().any().any():
                plt.figure(figsize=(8, 6))
                mask = np.triu(np.ones_like(correlation, dtype=bool))
                sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f',
                            cmap='coolwarm', center=0, square=True,
                            cbar_kws={"shrink": .8}, linewidths=0.5)
                plt.title('Correlation Heatmap', fontsize=14, pad=20)

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                plots['heatmap'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error generating heatmap: {e}")

    # 3. Bar charts for categorical (limit to 2 columns)
    if categorical_cols:
        try:
            valid_categorical_cols = [col for col in categorical_cols if col in df.columns and df[col].nunique() > 0]
            if valid_categorical_cols:
                # Limit to 2 columns
                cols_to_plot = valid_categorical_cols[:2]
                n_cats = min(2, len(cols_to_plot))

                fig, axes = plt.subplots(1, n_cats, figsize=(n_cats * 5, 4))
                if n_cats == 1:
                    axes = [axes]

                for idx, col in enumerate(cols_to_plot):
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

                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                plots['barchart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error generating barchart: {e}")

    return plots


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


# ====== ROUTES WITH MEMORY OPTIMIZATION ======
@app.route('/')
def index():
    session.clear()
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
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

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read CSV with optimization for Vercel
        try:
            # Read only first 5000 rows for preview on Vercel
            df = pd.read_csv(filepath, nrows=5000, low_memory=False)
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

        # Clean up memory
        gc.collect()

        return render_template('preview.html', preview=preview_data)

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

        # Read with optimization for Vercel
        try:
            # Read with limited rows for Vercel (adjust based on your needs)
            df = pd.read_csv(filepath, nrows=20000, low_memory=False)
        except Exception as e:
            flash(f'Error reading CSV file: {str(e)}', 'error')
            return redirect('/')

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

        # Generate plots with optimization
        plots = generate_visualizations(df_cleaned, column_types, filename)

        cleaned_filename = f"cleaned_{filename}"
        cleaned_filepath = os.path.join(app.config['PROCESSED_FOLDER'], cleaned_filename)

        # Prepare final CSV
        df_to_save = df_cleaned.copy()
        for col in df_to_save.columns:
            col_type = column_types.get(col, 'text')
            if col_type == 'numeric':
                df_to_save[col] = pd.to_numeric(df_to_save[col], errors='coerce')
            df_to_save[col] = df_to_save[col].fillna('')

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

        session['cleaned_filepath'] = cleaned_filepath

        # Clean up memory
        gc.collect()

        return render_template('results.html',
                               summary=summary,
                               plots=plots,
                               cleaned_filename=cleaned_filename,
                               column_types=column_types,
                               cleaning_report=cleaning_report)

    except MemoryError:
        flash(
            'The dataset is too large for processing on Vercel. Please try with a smaller file (<10MB, <20,000 rows).',
            'error')
        return redirect('/')
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in process_data: {e}")
        flash(f'Error processing data: {str(e)}', 'error')
        return redirect('/')


@app.route('/download/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)

        if not os.path.exists(filepath):
            if 'cleaned_filepath' in session and os.path.exists(session['cleaned_filepath']):
                filepath = session['cleaned_filepath']
            else:
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


@app.route('/clear_session')
def clear_session():
    session.clear()
    flash('Session cleared', 'info')
    return redirect('/')


# Error handlers
@app.errorhandler(500)
def internal_error(error):
    flash('Server error. The file might be too large for Vercel. Try a smaller file (<10MB, <20,000 rows).', 'error')
    return redirect('/')


@app.errorhandler(413)
def too_large(error):
    flash('File too large. Maximum file size is 5MB for Vercel deployment.', 'error')
    return redirect('/')


# Vercel handler
def handler(event, context):
    return app(event, context)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)  # debug=False for Vercel