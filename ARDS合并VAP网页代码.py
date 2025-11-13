# Load data
# Note: Ensure '12交集特征.xlsx' is in the same directory or provide the full path
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# --- Set wide layout ---
st.set_page_config(layout="wide")

# --- Your Provided Code (Adapted for Streamlit) ---

# Load data
# Note: Ensure '12交集特征.xlsx' is in the same directory or provide the full path
try:
    df = pd.read_excel('构建模型222.xlsx')
except FileNotFoundError:
    st.error("Error, file not found")
    st.stop()
df.rename(columns={"DDimer": "D-二聚体（mg/L）",
                   "Control Ventilation": "控制通气模式时间（天）",
                   "FiO2_D1": "机械通气第一天的吸氧浓度",
                   "Sedation time": "镇静时间（小时）",
                   "Analgesic time": "镇痛时间（小时）",
                   "duration of mechanical ventilation": "机械通气时间（天）",
                   "SEX": "性别"}, inplace=True)
# 删除rename
# Define variables
continuous_vars = [
    'D-二聚体（mg/L）',
    '控制通气模式时间',
    '机械通气第一天的吸氧浓度（天）',
    '镇静时间（小时）',
    '镇痛时间（小时）',
    '机械通气时间（天）']
categorical_vars = [
    '性别',  # 使用重命名后的列名
]
# Combine all variables for unified input
all_vars = continuous_vars + categorical_vars
# 预处理管道，对分类变量进行OneHotEncoder（不删除任何列）
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_vars),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_vars)
        # 这里不再传递selected_categorical_vars，而是使用categorical_vars，并且OneHotEncoder不传递参数
    ])

# 应用预处理
X_processed = preprocessor.fit_transform(df)

# 获取特征名
try:
    feature_names = (
            continuous_vars +
            list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_vars))
    )
except AttributeError:
    feature_names = (
            continuous_vars +
            list(preprocessor.named_transformers_['cat'].get_feature_names(categorical_vars))
    )

X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# 定义要删除的列
# drop_columns = ['是否为免疫抑制人群_0', 'ARDS分级是否为3级_1', 'ARDS分级是否为3级_2', '呼吸支持方式是否为机械通气_1', '呼吸支持方式是否为机械通气_2']

# 只删除存在的列
# columns_to_drop = [col for col in drop_columns if col in X_processed_df.columns]
# X_processed_df = X_processed_df.drop(columns=columns_to_drop)

# 然后，您可以使用X_processed_df作为特征

X = X_processed_df
y = df['结局']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=999)

# 保存训练时的特征列顺序
training_feature_columns = X_train.columns.tolist()

# --- Streamlit App Interface ---

st.markdown(
    "<h1 style='text-align: center;'>基于支持向量机的急性呼吸君迫综合（ARDS）患者呼吸机相关肺炎（VAP）的早期预测模型</h1>",
    unsafe_allow_html=True)

# --- 1. User Input for X values ---
st.header("1. 输入患者信息")
user_input = {}
input_valid = True

input_cols = st.columns(4)
for i, var in enumerate(all_vars):
    with input_cols[i % 4]:
        if var in continuous_vars:
            user_val = st.number_input(f"{var}", value=None, format="%.4f", step=0.01, placeholder="please enter")
            if user_val is None:
                input_valid = False
            user_input[var] = user_val
        else:
            options = np.unique(df[var].astype(str))
            selected_option = st.selectbox(f"{var}", options=options, index=None, placeholder="please enter")
            if selected_option is None:
                input_valid = False
            user_input[var] = selected_option

# --- 3. Prediction Button and Logic ---
if st.button("ARDS患者发生VAP的概率"):
    if not input_valid:
        st.error("error, please check all X is inputed")
    else:
        try:
            # 创建输入数据的DataFrame
            input_data = pd.DataFrame([user_input])

            # 应用相同的预处理
            input_processed = preprocessor.transform(input_data)
            input_processed_df = pd.DataFrame(input_processed, columns=feature_names)

            # 确保列的顺序与训练时完全一致
            input_processed_df = input_processed_df[training_feature_columns]

            # 训练模型
            model = SVC(random_state=random_seed, kernel='linear', probability=True, C=10, gamma=0.01,
                        class_weight='balanced')
            model.fit(X_train, y_train)
            st.success("Model trained successfully with fixed parameters!")

            # 进行预测
            prediction_proba = model.predict_proba(input_processed_df)[0]

            # 显示结果
            st.header("Prediction Result")
            prob_label = "Mortality probability of ARDS"
            st.metric(label=prob_label, value=f"{prediction_proba[1] * 100:.2f}%")

        except Exception as e:
            st.error(f"An error occurred during model training or prediction: {e}")
            # 添加调试信息
            st.write(f"Training features: {len(training_feature_columns)}")
            st.write(f"Input features after preprocessing: {len(input_processed_df.columns)}")
            st.write(f"Training feature columns: {training_feature_columns}")
            st.write(f"Input feature columns: {input_processed_df.columns.tolist()}")

# --- Disclaimer Section at the Bottom ---
st.markdown("---")
disclaimer_text = """
**Disclaimer:**

Supplement:
*   性别，0代表女性，1代表男性。
*   机械通气第一天的吸氧浓度，若70%，则填写为0.7。
*   D-二聚体为确诊ARDS当天的最高值。
"""
st.markdown(disclaimer_text)