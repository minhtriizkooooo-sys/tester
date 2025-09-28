from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import io
import base64
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    logo_error = False
    error = None
    warning = None
    original_table = None
    anomaly_table = None
    csv_data = None
    bar_chart = None
    charts = []
    class_options = []
    subject_options = []
    classes = None

    if request.method == 'POST':
        file = request.files.get('file')
        z_threshold = float(request.form.get('z_threshold', 2.0))
        selected_classes = request.form.getlist('classes')
        selected_subjects = request.form.getlist('subjects')

        if not file:
            error = "Vui lòng upload file CSV."
            return render_template('index.html', error=error, logo_error=logo_error)

        # Đọc file CSV
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except:
            file.seek(0)
            df = pd.read_csv(file, encoding='latin1')
            warning = "File không phải UTF-8, đã dùng latin1."

        # Chuẩn hóa tên cột
        df.columns = df.columns.str.strip().str.replace(' ', '').str.capitalize()

        # Kiểm tra cột lớp
        class_col = [c for c in df.columns if c.lower() == 'lop']
        if not class_col:
            error = "Không tìm thấy cột 'Lop'."
            return render_template('index.html', error=error, logo_error=logo_error)
        df['Lop'] = df[class_col[0]]

        # Kiểm tra cột học sinh
        student_col = [c for c in df.columns if c.lower() in ['mahs', 'id', 'studentid']]
        if not student_col:
            error = "Không tìm thấy cột 'MaHS'."
            return render_template('index.html', error=error, logo_error=logo_error)
        df['MaHS'] = df[student_col[0]]

        # Chọn các cột môn học
        subject_cols = [c for c in df.columns if c not in ['MaHS', 'Lop']]
        if len(subject_cols) == 0:
            error = "Không tìm thấy cột điểm môn học."
            return render_template('index.html', error=error, logo_error=logo_error)

        # Cung cấp danh sách lớp và môn cho form
        class_options = sorted(df['Lop'].unique())
        subject_options = subject_cols

        # Sử dụng lớp và môn được chọn, hoặc mặc định tất cả nếu không chọn
        classes = selected_classes if selected_classes else class_options
        subjects = selected_subjects if selected_subjects else subject_options
        df_filtered = df[df['Lop'].isin(classes)].copy()

        if df_filtered.empty:
            warning = "Không có dữ liệu cho lớp được chọn."
            return render_template('index.html', warning=warning, logo_error=logo_error, class_options=class_options, subject_options=subject_options)

        # Tính Z-score riêng từng môn
        for subj in subjects:
            df_filtered[f'Z_{subj}'] = stats.zscore(df_filtered[subj].fillna(0))
            df_filtered[f'Highlight_{subj}'] = df_filtered[f'Z_{subj}'].abs() > z_threshold

        # Bảng dữ liệu
        original_table = df_filtered[['MaHS', 'Lop'] + subjects].to_html(index=False, classes='table', border=1)

        # Tạo bảng bất thường
        anomaly_cols = ['MaHS', 'Lop'] + [subj for subj in subjects]
        anomalies = df_filtered.copy()
        anomalies = anomalies[anomalies[[f'Highlight_{subj}' for subj in subjects]].any(axis=1)]
        anomaly_table = anomalies[anomaly_cols].to_html(index=False, classes='table', border=1)

        # Chuẩn bị CSV để download
        csv_buffer = io.StringIO()
        anomalies.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_data = base64.b64encode(csv_buffer.getvalue().encode('utf-8')).decode('utf-8')

        # Biểu đồ cột
        class_summary = df_filtered.groupby('Lop').size().reset_index(name='Tổng học sinh')
        anomaly_count = anomalies.groupby('Lop').size().reset_index(name='Học sinh bất thường')
        summary = pd.merge(class_summary, anomaly_count, on='Lop', how='left').fillna(0)
        fig_col = px.bar(summary, x='Lop', y=['Tổng học sinh', 'Học sinh bất thường'],
                         barmode='group', color_discrete_map={'Tổng học sinh': '#4CAF50', 'Học sinh bất thường': '#FF5252'},
                         labels={'value': 'Số học sinh', 'Lop': 'Lớp'}, title="Tổng học sinh & Học sinh bất thường theo lớp")
        bar_chart = fig_col.to_json()

        # Biểu đồ scatter và histogram
        for subj in subjects:
            fig_scat = px.scatter(df_filtered, x='MaHS', y=subj, color=f'Z_{subj}',
                                  color_continuous_scale='RdYlGn_r',
                                  size=df_filtered[f'Z_{subj}'].abs(),
                                  size_max=20,
                                  hover_data={'MaHS': True, subj: True, f'Z_{subj}': True})
            fig_hist = px.histogram(df_filtered, x=subj, nbins=20, color=f'Highlight_{subj}',
                                    color_discrete_map={True: '#FF0000', False: '#4CAF50'},
                                    labels={'count': 'Số học sinh'})
            charts.append((subj, fig_scat.to_json(), fig_hist.to_json()))

        return render_template('index.html', logo_error=logo_error, warning=warning, original_table=original_table,
                               anomaly_table=anomaly_table, csv_data=csv_data, bar_chart=bar_chart, charts=charts,
                               class_options=class_options, subject_options=subject_options, classes=classes)

    return render_template('index.html', logo_error=logo_error, class_options=class_options, subject_options=subject_options)

@app.route('/download', methods=['POST'])
def download():
    csv_data = base64.b64decode(request.form.get('csv_data')).decode('utf-8')
    return send_file(
        io.BytesIO(csv_data.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='Students_Anomalies.csv'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))