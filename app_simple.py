from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

app = Flask(__name__)

def create_sample_data():
    """샘플 데이터 생성 함수"""
    print("샘플 데이터를 생성합니다...")
    
    # 일별 데이터 생성
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    daily_data = []
    
    for date in dates:
        if date.weekday() < 5:  # 평일만
            base_volume = 1000
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            volume = int(base_volume * seasonal_factor * np.random.uniform(0.8, 1.2))
            
            daily_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'weekday': date.strftime('%a'),
                'volume': volume,
                'bw_pages': int(volume * 0.7),
                'color_pages': int(volume * 0.3),
                'total': volume
            })
    
    df_daily = pd.DataFrame(daily_data)
    
    # 월별 데이터 생성
    monthly_data = []
    for year in [2022, 2023, 2024]:
        for month in range(1, 13):
            if year == 2024 and month > datetime.now().month:
                break
                
            base_monthly_volume = 25000
            seasonal_factor = 1 + 0.4 * np.sin(2 * np.pi * month / 12)
            volume = int(base_monthly_volume * seasonal_factor * np.random.uniform(0.9, 1.1))
            
            monthly_data.append({
                'month': f'{year}년{month:02d}월',
                'volume': volume,
                'year': year,
                'month_num': month
            })
    
    df_monthly = pd.DataFrame(monthly_data)
    
    return df_daily, df_monthly

# 전역 변수로 데이터 저장
df_daily, df_monthly = create_sample_data()

@app.route('/')
def index():
    """메인 대시보드 페이지"""
    return render_template('simple_dashboard.html')

@app.route('/api/weekly_comparison')
def get_weekly_comparison():
    """주간 발주량 비교 데이터"""
    try:
        latest_date = pd.to_datetime(df_daily['date'].max())
        this_week_dates = []
        d = latest_date
        
        while len(this_week_dates) < 5:
            if d.weekday() < 5:
                this_week_dates.append(d.date())
            d -= timedelta(days=1)
        this_week_dates.reverse()
        
        last_week_dates = [d - timedelta(days=7) for d in this_week_dates]
        
        df_daily['date_only'] = pd.to_datetime(df_daily['date']).dt.date
        this_week_data = df_daily[df_daily['date_only'].isin(this_week_dates)]
        last_week_data = df_daily[df_daily['date_only'].isin(last_week_dates)]
        
        comparison_data = []
        for i, date in enumerate(this_week_dates):
            this_week_vol = this_week_data[this_week_data['date_only'] == date]['volume'].iloc[0] if len(this_week_data[this_week_data['date_only'] == date]) > 0 else 0
            last_week_vol = last_week_data[last_week_data['date_only'] == date]['volume'].iloc[0] if len(last_week_data[last_week_data['date_only'] == date]) > 0 else 0
            
            comparison_data.append({
                'weekday': date.strftime('%a'),
                'this_week': this_week_vol,
                'last_week': last_week_vol
            })
        
        return jsonify({
            'data': comparison_data,
            'latest_date': latest_date.strftime('%Y-%m-%d')
        })
        
    except Exception as e:
        print(f"주간 비교 데이터 생성 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/monthly_comparison')
def get_monthly_comparison():
    """월간 발주량 비교 데이터"""
    try:
        df_monthly_actual = df_monthly[df_monthly['volume'] > 0].copy()
        df_monthly_sorted = df_monthly_actual.sort_values('month', ascending=False).head(12)
        
        monthly_data = []
        for _, row in df_monthly_sorted.iterrows():
            monthly_data.append({
                'month': row['month'],
                'volume': row['volume'],
                'year': row['year']
            })
        
        return jsonify({'data': monthly_data})
        
    except Exception as e:
        print(f"월간 비교 데이터 생성 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/yearly_trends')
def get_yearly_trends():
    """연도별 월간 추이 데이터"""
    try:
        df_monthly_actual = df_monthly[df_monthly['volume'] > 0].copy()
        
        yearly_data = {}
        for _, row in df_monthly_actual.iterrows():
            year = row['year']
            month = row['month_num']
            volume = row['volume']
            
            if year not in yearly_data:
                yearly_data[year] = {}
            yearly_data[year][month] = volume
        
        return jsonify({'data': yearly_data})
        
    except Exception as e:
        print(f"연도별 추이 데이터 생성 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/summary')
def get_summary():
    """요약 통계 데이터"""
    try:
        total_volume = df_daily['volume'].sum()
        avg_volume = df_daily['volume'].mean()
        total_days = len(df_daily)
        
        return jsonify({
            'total_volume': int(total_volume),
            'avg_volume': round(avg_volume, 2),
            'total_days': total_days
        })
        
    except Exception as e:
        print(f"요약 통계 생성 오류: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Flask 기반 발주량 대시보드를 시작합니다...")
    print("📊 웹 브라우저에서 http://127.0.0.1:5000 로 접속하세요")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
