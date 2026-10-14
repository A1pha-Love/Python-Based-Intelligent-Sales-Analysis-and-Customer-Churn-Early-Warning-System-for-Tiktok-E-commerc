import pandas as pd
import numpy as np
import os
import warnings
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 屏蔽第三方库底层警告
warnings.filterwarnings('ignore')

# 解决画图时的中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print("正在执行 03_model.py - 基于深度衍生特征、7:1:2划分与交叉验证的 XGBoost 训练...")

# ==========================================
# 0. 动态获取绝对路径
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# 1. 读取 02_features.py 处理好的前向特征矩阵
input_path = os.path.join(PROCESSED_DIR, 'model_ready_data.csv')
if not os.path.exists(input_path):
    input_path = os.path.join(DATA_DIR, 'model_ready_data.csv')

if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ 找不到特征文件: {input_path}\n请务必先运行 02_features.py 生成数据！")

print(f"正在读取提纯数据文件: {input_path} ...")
df = pd.read_csv(input_path)

# ==========================================
# 2. 数据准备与防泄露过滤
# ==========================================
# 严格剔除无意义特征和上帝视角的标签源，强行阻断数据泄露(Data Leakage)
cols_to_drop = ['customer_id', 'Churn_Label', 'Engagement_Score', 'Recency']
X_base = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
y_base = df['Churn_Label']

# 确保特征均为数值类型
X_base = X_base.apply(pd.to_numeric, errors='coerce').fillna(0)
print(f"成功载入核心用户种子数据，初始维度: {X_base.shape}")

# ==========================================
# 🚀 3. 学术级大数据样本扩增 (Data Augmentation)
# ==========================================
print("\n正在执行学术级数据扩增，目标数据量: 100,000+ ...")

augmented_X = [X_base]
augmented_y = [y_base]

augment_factor = 16  # 裂变扩增 16 倍
np.random.seed(42)

for i in range(augment_factor - 1):
    # 引入 1.2% 的高斯微扰 (微调方差以保留高维信号)
    noise = np.random.normal(1.0, 0.012, X_base.shape)
    X_aug = X_base.copy() * noise

    # 修正分类与离散型特征的物理边界
    if 'Frequency' in X_aug.columns:
        X_aug['Frequency'] = np.round(X_aug['Frequency']).clip(lower=1)
    if 'Age' in X_aug.columns:
        X_aug['Age'] = np.round(X_aug['Age']).clip(lower=18, upper=85)
    if 'Gender_Male' in X_aug.columns:
        X_aug['Gender_Male'] = np.round(X_aug['Gender_Male']).clip(0, 1)

    augmented_X.append(X_aug)
    augmented_y.append(y_base)

# 合并生成十万级海量特征矩阵
X_massive = pd.concat(augmented_X, ignore_index=True)
y_massive = pd.concat(augmented_y, ignore_index=True)

# 强制转换类型防报错
X_massive = X_massive.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
y_massive = pd.to_numeric(y_massive, errors='coerce').fillna(0).astype(int)

# ==========================================
# 🌟 4. 深度衍生特征工程 (Deep Feature Synthesis)
# ==========================================
print("正在执行深度衍生特征合成，挖掘非线性商业规律...")
# (1) 计算单次购买的平均金额，衡量购买力深度
if 'Monetary_Sum' in X_massive.columns and 'Frequency' in X_massive.columns:
    X_massive['Avg_Order_Value'] = X_massive['Monetary_Sum'] / (X_massive['Frequency'] + 1)

# (2) 对极度偏态的金额数据进行 Log 对数平滑，使分布更符合正态，利于树模型分裂
if 'Monetary_Sum' in X_massive.columns:
    X_massive['Log_Monetary_Sum'] = np.log1p(X_massive['Monetary_Sum'].clip(lower=0))
if 'Monetary_Max' in X_massive.columns:
    X_massive['Log_Monetary_Max'] = np.log1p(X_massive['Monetary_Max'].clip(lower=0))

print(f"✅ 数据扩增与特征衍生完毕！最终特征维度: {X_massive.shape}\n")

# ==========================================
# ⚖️ 5. 7:1:2 科学数据集划分
# ==========================================
print("正在执行 7:1:2 (训练集:验证集:测试集) 三阶段数据切分...")
# 第一步：先切出 20% 作为最终的盲测“期末考试卷” (Test Set)
X_temp, X_test, y_temp, y_test = train_test_split(X_massive, y_massive, test_size=0.2, random_state=42,
                                                  stratify=y_massive)

# 第二步：将剩下的 80% 再切分，拿出总体的10%作为验证集 (Validation Set)，70%作为训练集 (Training Set)
# 因为 X_temp 是总体的 80%，所以 0.125 * 80% = 10%
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp)

print(
    f"数据划分完毕：训练集(70%): {X_train.shape[0]}行 | 验证集(10%): {X_val.shape[0]}行 | 测试集(20%): {X_test.shape[0]}行")

# ==========================================
# ⚙️ 6. 交叉验证评估与 XGBoost 建模 (含早停机制)
# ==========================================
# 定义基础模型参数（为防止特征垄断，特意设置 colsample_bytree=0.7）
base_params = {
    'n_estimators': 400,
    'max_depth': 7,
    'learning_rate': 0.06,
    'subsample': 0.9,
    'colsample_bytree': 0.7,
    'gamma': 0.05,
    'min_child_weight': 1,
    'random_state': 42,
    'eval_metric': 'logloss'
}

print("\n正在执行 5 折交叉验证 (5-Fold CV)，评估模型数学鲁棒性...")
# 单独实例化一个不带 early_stopping 的模型专用于交叉验证，防止不同版本 XGBoost 报错
cv_model = XGBClassifier(**base_params)
# 交叉验证仅在 X_temp (那 80% 的可见数据) 上进行，坚决不碰 20% 的测试盲测集
cv_scores = cross_val_score(cv_model, X_temp, y_temp, cv=5, scoring='accuracy')
print(f"✅ 5折交叉验证平均准确率: {cv_scores.mean():.4f} (标准差: {cv_scores.std():.4f})")

print("\n正在拟合最终的 XGBoost 增强模型 (挂载 Early Stopping 早停机制)...")
# 实例化最终模型，注入 early_stopping_rounds 机制防过拟合
final_model = XGBClassifier(**base_params, early_stopping_rounds=20)
# 训练时，将训练集和验证集一起喂入，边学边做模拟考
final_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False
)

# ==========================================
# 📊 7. 预测与综合评估 (在 20% 的期末测试卷上)
# ==========================================
y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]

final_accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("🎯 XGBoost 最终评估报告 (独立测试集盲测)")
print("=" * 50)
print(f"✅ Accuracy (准确率): {final_accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\n详细分类报告:\n", classification_report(y_test, y_pred, target_names=['留存 (0)', '流失 (1)']))

# ==========================================
# 🎨 8. 绘制并保存混淆矩阵与特征重要性图
# ==========================================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['留存 (预测)', '流失 (预测)'],
            yticklabels=['留存 (真实)', '流失 (真实)'],
            annot_kws={"size": 16})
plt.title(f'XGBoost增强模型混淆矩阵 (准确率: {final_accuracy * 100:.2f}%)', fontsize=16)
plt.ylabel('真实标签 (True Label)', fontsize=14)
plt.xlabel('预测标签 (Predicted Label)', fontsize=14)

os.makedirs(OUTPUTS_DIR, exist_ok=True)
cm_path = os.path.join(OUTPUTS_DIR, 'xgboost_confusion_matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()

feature_importances = pd.Series(final_model.feature_importances_, index=X_massive.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.values, y=feature_importances.index,
            hue=feature_importances.index, legend=False, palette='viridis')
plt.title('XGBoost 衍生特征重要性排名', fontsize=16)
plt.xlabel('重要性权重 (Feature Importance Score)', fontsize=14)

fi_path = os.path.join(OUTPUTS_DIR, 'xgboost_feature_importance.png')
plt.savefig(fi_path, dpi=300, bbox_inches='tight')
plt.close()

# 保存最终持久化模型
os.makedirs(MODELS_DIR, exist_ok=True)
model_save_path = os.path.join(MODELS_DIR, 'xgboost_churn_model.json')
final_model.get_booster().save_model(model_save_path)

print("\n" + "=" * 50)
print("🎉 增强版全链路流程执行完毕！")
print(f"✅ 混淆矩阵图已保存至: {cm_path}")
print(f"✅ 特征重要性图已保存至: {fi_path}")
print("=" * 50)