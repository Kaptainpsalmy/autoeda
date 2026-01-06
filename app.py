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
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Simple configuration
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'autoeda-secret-key-2024')

# Create temp directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def simple_detect_types(df):
    """Simple column type detection"""
    column_types = {}
    for col in df.columns:
        # Try numeric first
        try:
            numeric = pd.to_numeric(df[col], errors='coerce')
            if numeric.notna().mean() > 0.5:
                column_types[col] = 'numeric'
                df[col] = numeric
                continue
        except:
            pass

        # Try date
        try:
            date_col = pd.to_datetime(df[col], errors='coerce')
            if date_col.notna().mean() > 0.5:
                column_types[col] = 'date'
                df[col] = date_col
                continue
        except:
            pass

        # Check if categorical
        unique_ratio = df[col].nunique() / len(df[col])
        if unique_ratio < 0.3 and df[col].nunique() < 50:
            column_types[col] = 'categorical'
        else:
            column_types[col] = 'text'

    return df, column_types


def simple_clean(df, column_types):
    """Simple cleaning"""
    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    for col in df.columns:
        col_type = column_types.get(col, 'text')
        if df[col].isna().any():
            if col_type == 'numeric':
                df[col].fillna(df[col].median(), inplace=True)
            elif col_type in ['categorical', 'text']:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else ''
                df[col].fillna(mode_val, inplace=True)
            elif col_type == 'date':
                df[col].fillna('', inplace=True)

    return df


def generate_simple_summary(df, column_types):
    """Generate simple summary"""
    summary = {
        'dataset_info': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
        },
        'column_analysis': []
    }

    for col in df.columns:
        col_info = {
            'name': col,
            'type': column_types.get(col, 'text'),
            'unique_values': int(df[col].nunique()),
            'missing_values': int(df[col].isna().sum()),
            'missing_percentage': float((df[col].isna().sum() / len(df)) * 100) if len(df) > 0 else 0
        }

        if col_info['type'] == 'numeric':
            col_info.update({
                'mean': float(df[col].mean()) if not df[col].empty else 0,
                'median': float(df[col].median()) if not df[col].empty else 0,
                'min': float(df[col].min()) if not df[col].empty else 0,
                'max': float(df[col].max()) if not df[col].empty else 0
            })

        summary['column_analysis'].append(col_info)

    return summary


def generate_simple_plots(df, column_types):
    """Generate simple plots as base64"""
    plots = {}

    # Get numeric columns
    numeric_cols = [col for col, col_type in column_types.items()
                    if col_type == 'numeric' and col in df.columns]

    # Generate histogram for first numeric column
    if numeric_cols:
        try:
            plt.figure(figsize=(8, 5))
            df[numeric_cols[0]].hist(bins=20, alpha=0.7, color='#2563eb', edgecolor='black')
            plt.title(f'Distribution of {numeric_cols[0]}')
            plt.xlabel(numeric_cols[0])
            plt.ylabel('Frequency')
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            plots['histogram'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error generating histogram: {e}")

    # Generate correlation heatmap if multiple numeric columns
    if len(numeric_cols) > 1:
        try:
            plt.figure(figsize=(8, 6))
            correlation = df[numeric_cols].corr()
            sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm')
            plt.title('Correlation Heatmap')
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            plots['heatmap'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error generating heatmap: {e}")

    # Generate bar chart for first categorical column
    categorical_cols = [col for col, col_type in column_types.items()
                        if col_type == 'categorical' and col in df.columns and df[col].nunique() < 10]

    if categorical_cols:
        try:
            plt.figure(figsize=(8, 5))
            df[categorical_cols[0]].value_counts().head(10).plot(kind='bar', color='#2563eb', alpha=0.7)
            plt.title(f'Top values in {categorical_cols[0]}')
            plt.xlabel(categorical_cols[0])
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            plots['barchart'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error generating barchart: {e}")

    return plots


@app.route('/')
def index():
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

        # Read CSV with limited rows for preview
        try:
            df = pd.read_csv(filepath, nrows=1000)  # Limit rows for preview
        except Exception as e:
            flash(f'Error reading CSV file: {str(e)}', 'error')
            return redirect('/')

        if df.empty:
            flash('Uploaded CSV file is empty', 'error')
            return redirect('/')

        df.columns = df.columns.str.strip()
        df, column_types = simple_detect_types(df)

        # Store in session
        session['uploaded_filename'] = filename
        session['filepath'] = filepath
        session['column_types'] = column_types

        # Create preview data
        preview_data = {
            'filename': filename,
            'shape': [len(df), len(df.columns)],
            'columns': list(df.columns),
            'column_types': column_types,
            'sample': df.head(10).to_dict('records')
        }

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
        column_types = session.get('column_types', {})

        # Read the full CSV file
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            flash(f'Error reading CSV file: {str(e)}', 'error')
            return redirect('/')

        df.columns = df.columns.str.strip()

        # Detect types and clean
        df, column_types = simple_detect_types(df)
        df = simple_clean(df, column_types)

        # Generate summary and plots
        summary = generate_simple_summary(df, column_types)
        plots = generate_simple_plots(df, column_types)

        # Save cleaned file
        cleaned_filename = f"cleaned_{filename}"
        cleaned_filepath = os.path.join('/tmp', cleaned_filename)
        df.to_csv(cleaned_filepath, index=False)
        session['cleaned_filepath'] = cleaned_filepath

        flash(f'✅ Dataset cleaned successfully! {len(df)} rows processed.', 'success')

        return render_template('results.html',
                               summary=summary,
                               plots=plots,
                               cleaned_filename=cleaned_filename,
                               column_types=column_types)

    except Exception as e:
        flash(f'Error processing data: {str(e)}', 'error')
        return redirect('/')


@app.route('/download/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join('/tmp', filename)

        if not os.path.exists(filepath):
            # Try to get from session
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


# Error handler
@app.errorhandler(500)
def internal_error(error):
    flash('Internal server error. Please try again with a smaller file.', 'error')
    return redirect('/')


@app.errorhandler(413)
def too_large(error):
    flash('File too large. Maximum file size is 5MB.', 'error')
    return redirect('/')


# Vercel handler
def handler(event, context):
    return app(event, context)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)