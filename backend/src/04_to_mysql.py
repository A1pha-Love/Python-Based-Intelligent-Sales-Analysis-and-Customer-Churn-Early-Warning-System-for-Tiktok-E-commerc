import pandas as pd
from sqlalchemy import create_engine
import os

# ==========================================
# 1. 配置数据库连接
# ==========================================
DB_USER = os.getenv('DB_USER', 'root')
DB_PASS = os.getenv('DB_PASS', '123456')  # 建议通过环境变量 DB_PASS 注入，不要硬编码真实密码
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_NAME = os.getenv('DB_NAME', 'bishe_db')  # 你的数据库名

# ==========================================
# 2. 自动获取路径
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
FILE_TRANS = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_transactions.csv')


def save_to_mysql():
    print("-" * 30)
    print("Step 4: 数据入库 (6张表豪华版)")
    print("-" * 30)

    # --- 连接数据库 ---
    connection_str = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8"
    try:
        engine = create_engine(connection_str)
        print("✅ 数据库连接成功！")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # --- 读取数据 ---
    if not os.path.exists(FILE_TRANS):
        print(f"❌ 找不到文件: {FILE_TRANS}")
        return

    print("正在读取 CSV...")
    df = pd.read_csv(FILE_TRANS)

    # 修复列名和总金额
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    if 'total_amount' not in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['total_amount'] = df['quantity'] * df['price']

    # 确保日期格式正确
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])

    # ==========================================
    # 1. [transactions] - 交易流水明细表 (基石)
    # ==========================================
    print("正在写入表 1/6: [transactions] ...")
    df.to_sql('transactions', engine, if_exists='replace', index=False)

    # ==========================================
    # 2. [customer_analysis] - 用户画像表 (人)
    # ==========================================
    print("正在写入表 2/6: [customer_analysis] ...")
    df_customers = df.groupby('customer_id').agg({
        'invoice_no': 'nunique',
        'total_amount': 'sum',
        'age': 'first',
        'gender': 'first',
        'category': lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0]
    }).reset_index()
    df_customers.columns = ['customer_id', 'buy_count', 'total_money', 'age', 'gender', 'fav_category']
    df_customers.to_sql('customer_analysis', engine, if_exists='replace', index=False)

    # ==========================================
    # 3. [monthly_sales] - 月度趋势表 (时间)
    #    用途: 画折线图，看每个月的业绩走势
    # ==========================================
    print("正在写入表 3/6: [monthly_sales] ...")
    # 提取月份 (格式如 2022-01)
    df['month_str'] = df['invoice_date'].dt.strftime('%Y-%m')
    df_monthly = df.groupby('month_str').agg({
        'total_amount': 'sum',  # 月销售额
        'invoice_no': 'nunique',  # 月订单数
        'customer_id': 'nunique'  # 月活跃人数
    }).reset_index()
    df_monthly.columns = ['month', 'sales', 'order_count', 'active_users']
    df_monthly.to_sql('monthly_sales', engine, if_exists='replace', index=False)

    # ==========================================
    # 4. [category_stats] - 品类表现表 (货)
    #    用途: 画柱状图，看哪个品类最赚钱
    # ==========================================
    print("正在写入表 4/6: [category_stats] ...")
    df_category = df.groupby('category').agg({
        'total_amount': 'sum',
        'quantity': 'sum',
        'price': 'mean'  # 平均单价
    }).reset_index()
    df_category.columns = ['category', 'total_sales', 'total_quantity', 'avg_price']
    df_category.to_sql('category_stats', engine, if_exists='replace', index=False)

    # ==========================================
    # 5. [age_groups] - 年龄段分布表 (画像)
    #    用途: 画直方图，分析主要消费人群是青年还是老年
    # ==========================================
    print("正在写入表 5/6: [age_groups] ...")
    # 将年龄切分为组: 10-20, 20-30...
    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ['<20', '20-30', '30-40', '40-50', '50-60', '>60']
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

    df_age = df.groupby('age_group').agg({
        'customer_id': 'nunique',  # 人数
        'total_amount': 'sum'  # 贡献金额
    }).reset_index()
    df_age.columns = ['age_group', 'user_count', 'total_sales']
    df_age.to_sql('age_groups', engine, if_exists='replace', index=False)

    # ==========================================
    # 6. [payment_stats] - 支付偏好表 (渠道)
    #    用途: 画环形图，分析支付方式占比
    # ==========================================
    print("正在写入表 6/6: [payment_stats] ...")
    df_payment = df.groupby('payment_method').agg({
        'invoice_no': 'count',  # 使用次数
        'total_amount': 'sum'  # 交易金额
    }).reset_index()
    df_payment.columns = ['payment_method', 'usage_count', 'total_sales']
    df_payment.to_sql('payment_stats', engine, if_exists='replace', index=False)

    print("\n🎉 6张表全部创建成功！请去 Navicat 刷新查看！")


if __name__ == '__main__':
    save_to_mysql()