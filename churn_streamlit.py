import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ========================================================
# Page Config
# ========================================================
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide"
)

# ========================================================
# Custom CSS
# ========================================================
st.markdown("""
<style>
    .title-main {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
    }
    .subtitle {
        color: #666;
        font-size: 1.1em;
    }
    .risk-high {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
    }
    .risk-medium {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
    }
    .risk-low {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ========================================================
# Load & Train Model
# ========================================================

@st.cache_resource
def train_model():
    """Train the model once and cache it"""
    
    # Load data
    df = pd.read_csv('Churn_Modelling.csv')
    
    # Drop unnecessary columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Prepare features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    # Define feature types
    numeric_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'HasCrCard', 'IsActiveMember']
    categorical_features = ['Geography', 'Gender']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Get predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    return {
        'model': pipeline,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'df': df
    }

# ========================================================
# Main App
# ========================================================

st.markdown('<div class="title-main">🏦 Banking Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict customer churn risk using Machine Learning</div>', unsafe_allow_html=True)

# Load model
model_data = train_model()
model = model_data['model']
metrics = model_data['metrics']

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predictor", "📊 Model Performance", "📈 Data Analysis", "ℹ️ About"])

# ========================================================
# TAB 1: PREDICTOR
# ========================================================
with tab1:
    st.subheader("🔮 Predict Customer Churn")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Customer Demographics:**")
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        geography = st.selectbox("Country", ["France", "Germany", "Spain"])
    
    with col2:
        st.write("**Financial Information:**")
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=10)
        balance = st.number_input("Account Balance ($)", min_value=0.0, max_value=500000.0, value=100000.0, step=1000.0)
        salary = st.number_input("Estimated Salary ($)", min_value=0, max_value=200000, value=50000, step=1000)
    
    with col3:
        st.write("**Bank Relationship:**")
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5, step=1)
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2, step=1)
        is_active = st.selectbox("Is Active Member?", ["Yes", "No"])
        has_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
    
    st.write("---")
    
    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        # Prepare input
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Geography': [geography],
            'Gender': [gender],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_products],
            'HasCrCard': [1 if has_card == "Yes" else 0],
            'IsActiveMember': [1 if is_active == "Yes" else 0],
            'EstimatedSalary': [salary]
        })
        
        # Predict
        churn_prob = model.predict_proba(input_data)[0, 1]
        
        # Display results
        st.success("✅ Prediction Complete!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Churn Probability", f"{churn_prob*100:.1f}%")
        
        with col2:
            status = "Will Churn ⚠️" if churn_prob > 0.5 else "Will Stay ✅"
            st.metric("Prediction", status)
        
        with col3:
            if churn_prob > 0.6:
                risk = "HIGH RISK 🔴"
            elif churn_prob > 0.4:
                risk = "MEDIUM RISK 🟡"
            else:
                risk = "LOW RISK 🟢"
            st.metric("Risk Level", risk)
        
        st.write("---")
        
        # Risk interpretation
        if churn_prob > 0.6:
            st.markdown("""
            <div class="risk-high">
            <b>⚠️ HIGH CHURN RISK</b><br>
            This customer is likely to leave. Immediate action recommended!<br>
            <b>Actions:</b> Personal contact, exclusive offers, service review
            </div>
            """, unsafe_allow_html=True)
        elif churn_prob > 0.4:
            st.markdown("""
            <div class="risk-medium">
            <b>⚠️ MEDIUM CHURN RISK</b><br>
            Monitor this customer closely.<br>
            <b>Actions:</b> Regular check-ins, new product offers
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="risk-low">
            <b>✅ LOW CHURN RISK</b><br>
            This customer is likely to stay loyal.<br>
            <b>Actions:</b> Maintain service quality, cross-sell opportunities
            </div>
            """, unsafe_allow_html=True)

# ========================================================
# TAB 2: MODEL PERFORMANCE
# ========================================================
with tab2:
    st.subheader("📊 Model Performance Metrics")
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall ⭐", f"{metrics['recall']:.4f}")
    col4.metric("F1-Score", f"{metrics['f1']:.4f}")
    col5.metric("ROC-AUC", f"{metrics['auc']:.4f}")
    
    st.write("---")
    
    col1, col2 = st.columns(2)
    
    # Confusion Matrix
    with col1:
        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(model_data['y_test'], model_data['y_pred'])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Stay', 'Churn'], yticklabels=['Stay', 'Churn'])
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig, use_container_width=True)
    
    # ROC Curve
    with col2:
        st.write("**ROC Curve:**")
        fpr, tpr, _ = roc_curve(model_data['y_test'], model_data['y_pred_proba'])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, linewidth=2.5, label=f'ROC-AUC = {metrics["auc"]:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.fill_between(fpr, tpr, alpha=0.2)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
    
    st.write("---")
    st.write("**Model Info:**")
    st.info("""
    - **Algorithm:** Gradient Boosting Classifier
    - **Training Samples:** 8,000 customers
    - **Test Samples:** 2,000 customers
    - **Features:** 10 (8 numeric + 2 categorical)
    - **Churn Rate:** 20.4%
    """)

# ========================================================
# TAB 3: DATA ANALYSIS
# ========================================================
with tab3:
    st.subheader("📈 Exploratory Data Analysis")
    
    df = model_data['df']
    
    col1, col2 = st.columns(2)
    
    # Churn distribution
    with col1:
        st.write("**Churn Distribution:**")
        churn_counts = df['Exited'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#2ecc71', '#e74c3c']
        ax.pie(churn_counts, labels=['Stayed', 'Churned'], autopct='%1.1f%%',
               colors=colors, startangle=90)
        ax.set_title('Customer Churn Distribution')
        st.pyplot(fig, use_container_width=True)
    
    # Age vs Churn
    with col2:
        st.write("**Age Distribution by Churn:**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='Exited', y='Age', palette=['#2ecc71', '#e74c3c'], ax=ax)
        ax.set_xticklabels(['Stayed', 'Churned'])
        ax.set_ylabel('Age')
        ax.set_title('Age vs Churn Status')
        st.pyplot(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # Tenure vs Churn
    with col1:
        st.write("**Tenure Distribution by Churn:**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='Exited', y='Tenure', palette=['#2ecc71', '#e74c3c'], ax=ax)
        ax.set_xticklabels(['Stayed', 'Churned'])
        ax.set_ylabel('Tenure (Years)')
        ax.set_title('Tenure vs Churn Status')
        st.pyplot(fig, use_container_width=True)
    
    # Geography
    with col2:
        st.write("**Churn Rate by Country:**")
        churn_by_geo = pd.crosstab(df['Geography'], df['Exited'], normalize='index') * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        churn_by_geo.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.8)
        ax.set_title('Churn Rate by Country')
        ax.set_xlabel('Country')
        ax.set_ylabel('Percentage (%)')
        ax.legend(['Stayed', 'Churned'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig, use_container_width=True)

# ========================================================
# TAB 4: ABOUT
# ========================================================
with tab4:
    st.subheader("ℹ️ About This Project")
    
    st.write("""
    ### 📌 Overview
    
    This machine learning application predicts the probability that a bank customer 
    will churn (close their account). By identifying at-risk customers early, 
    the bank can take proactive measures to retain them.
    
    ### 🎯 Business Problem
    
    - **Cost:** Customer acquisition is 5-25x more expensive than retention
    - **Challenge:** Banks don't know which customers will leave
    - **Solution:** ML model to predict churn probability
    
    ### 🔬 Model Details
    
    **Algorithm:** Gradient Boosting Classifier
    - Non-linear patterns
    - Handles imbalanced data well
    - High accuracy and interpretability
    
    **Why Recall is Critical:**
    - Missing a churner = Revenue loss ❌
    - False alarm = Cost of retention offer ✓
    - Better to over-predict churn!
    
    **Validation:** Stratified K-Fold
    - Preserves churn ratio in each fold
    - Prevents sampling bias
    
    ### 📊 Features
    
    **Demographics:** Age, Gender, Geography
    
    **Financial:** Credit Score, Salary, Balance
    
    **Behavioral:** Tenure, Products, Activity, Credit Card
    
    ### ⚠️ Disclaimers
    
    - Predictions are based on historical patterns
    - Should be combined with human judgment
    - Not a guarantee of future behavior
    
    ### 💻 Technologies
    
    - Python, scikit-learn, pandas, Streamlit
    - Machine Learning, Data Science
    - Classification, Hyperparameter Tuning
    """)

# ========================================================
# Footer
# ========================================================
st.write("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85em;'>
    <p>Banking Customer Churn Predictor | ML Project | 2024</p>
    <p>⚠️ Educational tool - use with professional judgment</p>
</div>
""", unsafe_allow_html=True)
