import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
from sqlalchemy import create_engine
import xgboost as xgb
import numpy as np
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '123456')  # 建议通过环境变量 DB_PASS 注入，不要硬编码真实密码
DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_NAME = os.getenv('DB_NAME', 'bishe_db')

engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4')


def query_db(sql):
    try:
        df = pd.read_sql(sql, engine)
        return df.to_dict(orient='records')
    except Exception as e:
        print(f"数据库查询报错: {e}")
        return []


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/kpi', methods=['GET'])
def get_kpi():
    sales = query_db("SELECT SUM(total_sales) as total_sales FROM category_stats")
    users = query_db("SELECT COUNT(customer_id) as total_users FROM customer_analysis")
    orders = query_db("SELECT SUM(total_quantity) as total_orders FROM category_stats")
    return jsonify({"code": 200, "data": {
        "total_sales": float(sales[0].get('total_sales', 0)) if sales else 0,
        "total_users": int(users[0].get('total_users', 0)) if users else 0,
        "total_orders": int(orders[0].get('total_orders', 0)) if orders else 0
    }})


@app.route('/api/trend', methods=['GET'])
def get_trend():
    return jsonify({"code": 200, "data": query_db("SELECT month, sales FROM monthly_sales ORDER BY month ASC")})


@app.route('/api/rank', methods=['GET'])
def get_rank():
    return jsonify(
        {"code": 200, "data": query_db("SELECT category, total_sales FROM category_stats ORDER BY total_sales DESC")})


@app.route('/api/gender', methods=['GET'])
def get_gender():
    return jsonify(
        {"code": 200, "data": query_db("SELECT gender, COUNT(*) as count FROM customer_analysis GROUP BY gender")})


@app.route('/api/age', methods=['GET'])
def get_age():
    return jsonify(
        {"code": 200, "data": query_db("SELECT age_group, user_count FROM age_groups ORDER BY age_group ASC")})


@app.route('/api/payment', methods=['GET'])
def get_payment():
    return jsonify({"code": 200, "data": query_db(
        "SELECT payment_method, usage_count FROM payment_stats ORDER BY usage_count DESC")})


# ==========================================
# 🌟 真实在线推演接口 (已彻底净化并同步衍生特征)
# ==========================================
@app.route('/api/predict_sample', methods=['GET'])
def predict_sample():
    try:
        data_path = os.path.join(BASE_DIR, 'data', 'processed', 'model_ready_data.csv')
        if not os.path.exists(data_path):
            data_path = os.path.join(BASE_DIR, 'data', 'model_ready_data.csv')

        model_path = os.path.join(BASE_DIR, 'models', 'xgboost_churn_model.json')

        if not os.path.exists(data_path) or not os.path.exists(model_path):
            return jsonify({"code": 500, "message": "未找到特征数据或模型文件"}), 500

        df = pd.read_csv(data_path)

        # 🔴 【核心修复】：在此处同步补全 03_model.py 中所做的深度衍生特征工程
        # 必须让接口推理时的数据字段与训练时保持绝对一致，否则 XGBoost 会报错找不到列
        if 'Monetary_Sum' in df.columns and 'Frequency' in df.columns:
            df['Avg_Order_Value'] = df['Monetary_Sum'] / (df['Frequency'] + 1)
        if 'Monetary_Sum' in df.columns:
            df['Log_Monetary_Sum'] = np.log1p(df['Monetary_Sum'].clip(lower=0))
        if 'Monetary_Max' in df.columns:
            df['Log_Monetary_Max'] = np.log1p(df['Monetary_Max'].clip(lower=0))

        # 强制转换为浮点型防止类型报错
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)

        # 分层抽样：模拟抓取一批新客户
        churn_pool = df[df['Churn_Label'] == 1]
        retention_pool = df[df['Churn_Label'] == 0]
        n_churn = min(7, len(churn_pool))
        n_retention = min(8, len(retention_pool))

        sample_df = pd.concat([churn_pool.sample(n=n_churn), retention_pool.sample(n=n_retention)]).sample(frac=1)

        booster = xgb.Booster()
        booster.load_model(model_path)

        # 剔除无关字段与上帝标签，严格按照纯正模型所需的字段送入推理
        features_for_pred = sample_df.drop(columns=['customer_id', 'Churn_Label', 'Recency', 'Engagement_Score'],
                                           errors='ignore')

        # 确保列顺序与训练时100%对齐
        features_for_pred = features_for_pred[booster.feature_names]

        dmatrix = xgb.DMatrix(features_for_pred)
        probabilities = booster.predict(dmatrix)

        result_data = []
        for i in range(len(sample_df)):
            row = sample_df.iloc[i]
            cust_id = row['customer_id'] if 'customer_id' in row else f"CUST-{np.random.randint(1000, 9999)}"
            recency_val = int(row['Recency']) if 'Recency' in row else int(np.random.randint(10, 150))
            money_val = float(row['Monetary_Sum']) if 'Monetary_Sum' in row else 0.0
            risk_val = int(probabilities[i] * 100)

            result_data.append({
                "id": str(cust_id)[:10],
                "recency": f"{recency_val}天",
                "money": f"¥ {money_val:,.0f}",
                "money_val": money_val,
                "risk": f"{risk_val}%",
                "risk_val": risk_val
            })

        return jsonify({"code": 200, "data": result_data})

    except Exception as e:
        print(f"XGBoost在线推演报错: {e}")
        return jsonify({"code": 500, "message": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 抖音电商数据中台 Flask 后端 (纯净真实版) 已启动...")
    print("👉 访问大屏：http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)