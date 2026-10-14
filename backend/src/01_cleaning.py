import pandas as pd
import os

# ==========================================
# 1. 自动获取路径 (相对路径)
# ==========================================
# 获取当前脚本所在的文件夹 (src)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (BiShe)
BASE_DIR = os.path.dirname(CURRENT_DIR)

# 拼接文件路径
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'customer_shopping_data.csv')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'processed', 'cleaned_transactions.csv')


def run_cleaning():
    print("-" * 30)
    print("Step 1: 启动数据清洗程序")
    print("-" * 30)

    # 1. 检查文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误：在以下路径找不到数据文件：\n{INPUT_FILE}")
        print("请确认你已经把 customer_shopping_data.csv 放入 data/raw/ 文件夹中。")
        return

    # 2. 读取数据
    print("正在读取数据...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    original_rows = len(df)
    print(f"原始数据量: {original_rows} 行")

    # 3. 清洗逻辑
    # 3.1 标准化列名
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 3.2 转换日期 (最关键一步)
    # errors='coerce' 会把无法转换的乱码变成空值(NaT)
    if 'invoice_date' in df.columns:
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True, errors='coerce')
        df.dropna(subset=['invoice_date'], inplace=True)  # 删除日期错误的行

    # 3.3 转换数值类型
    for col in ['quantity', 'price', 'age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3.4 过滤异常值 (比如数量为负数)
    if 'quantity' in df.columns:
        df = df[df['quantity'] > 0]
    if 'price' in df.columns:
        df = df[df['price'] > 0]

    # 3.5 缺失值处理
    if 'customer_id' in df.columns:
        df.dropna(subset=['customer_id'], inplace=True)

    # 4. 保存清洗后的数据
    # encoding='utf-8-sig' 是为了防止 Excel 打开中文乱码
    try:
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"✅ 清洗完成！")
        print(f"有效数据: {len(df)} 行 (删除了 {original_rows - len(df)} 行)")
        print(f"文件已保存至: {OUTPUT_FILE}")
    except Exception as e:
        print(f"❌ 保存失败: {e}")


if __name__ == '__main__':
    run_cleaning()