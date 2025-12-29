import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score 
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, 
confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay 
import joblib 
import warnings 
warnings.filterwarnings('ignore') 
 
 
USE_SYNTHETIC = False 
DATA_PATH = r"C:\Users\91978\Desktop\python 
program\Python\Python313\train_u6lujuX_CVtuZ9i (1).csv" 
RANDOM_STATE = 42 
TEST_SIZE = 0.2 
TARGET = 'Loan_Status' 
 
 
 
def make_synthetic_loan_data(n=1000): 
    np.random.seed(RANDOM_STATE) 
    df = pd.DataFrame({ 
        'Gender': np.random.choice(['Male', 'Female'], n), 
        'Married': np.random.choice(['Yes', 'No'], n), 
        'Dependents': np.random.choice(['0','1','2','3+'], n), 
        'Education': np.random.choice(['Graduate','Not Graduate'], n), 
        'Self_Employed': np.random.choice(['Yes','No'], n), 
        'ApplicantIncome': np.random.randint(1500, 25000, n), 
        'CoapplicantIncome': np.random.randint(0, 10000, n), 
        'LoanAmount': np.random.randint(50, 700, n), 
        'Loan_Amount_Term': np.random.choice([360, 120, 180, 240], n), 
        'Credit_History': np.random.choice([1.0, 0.0], n), 
        'Property_Area': np.random.choice(['Urban','Semiurban','Rural'], n), 
        'Loan_Status': np.random.choice(['Y','N'], n) 
    }) 
    return df 
 
 
if USE_SYNTHETIC: 
    df = make_synthetic_loan_data(n=2000) 
    print("Using synthetic data with shape:", df.shape) 
else: 
    if not os.path.exists(DATA_PATH): 
        raise FileNotFoundError(f"DATA_PATH {DATA_PATH} not found. Set 
USE_SYNTHETIC=True") 
    df = pd.read_csv(DATA_PATH) 
    print("Loaded data from:", DATA_PATH, "| Shape:", df.shape) 
 
# Drop Loan_ID (identifier) 
if 'Loan_ID' in df.columns: 
    df = df.drop(columns=['Loan_ID']) 
 
 
print("\n--- Head ---\n", df.head()) 
print("\n--- Missing Values ---\n", df.isnull().sum()) 
 
 
def quick_eda(dataframe): 
    if TARGET in dataframe.columns: 
        print("\nTarget value counts:") 
        print(dataframe[TARGET].value_counts()) 
    numeric = dataframe.select_dtypes(include=[np.number]).columns.tolist() 
    numeric = [c for c in numeric if c != TARGET] 
    cat_cols = dataframe.select_dtypes(include=['object','category']).columns.tolist() 
     
     
    if numeric: 
        dataframe[numeric].hist(bins=15, figsize=(12,8)) 
        plt.tight_layout() 
        plt.show() 
    # Categorical counts 
    for c in cat_cols: 
        plt.figure(figsize=(5,3)) 
        sns.countplot(x=c, data=dataframe, order=dataframe[c].value_counts().index) 
        plt.title(c) 
        plt.xticks(rotation=45) 
        plt.tight_layout() 
        plt.show() 
 
quick_eda(df) 
 
df['TotalIncome'] = df['ApplicantIncome'].fillna(0) + df['CoapplicantIncome'].fillna(0) 
df['DebtToIncome'] = df['LoanAmount'] / df['TotalIncome'].replace(0, np.nan) 
df['ApplicantIncome_log'] = np.log1p(df['ApplicantIncome'].fillna(0)) 
df['LoanAmount_log'] = np.log1p(df['LoanAmount'].fillna(0)) 
if 'Dependents' in df.columns: 
    df['Dependents'] = df['Dependents'].replace('3+', '3').astype('object') 
 
 
X = df.drop(columns=[TARGET]) 
y = df[TARGET].copy() 
if y.dtype=='object' or y.dtype.name=='category': 
    le_target = LabelEncoder() 
    y = le_target.fit_transform(y) 
    print("Encoded target classes:", le_target.classes_) 
 
 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE) 
 
 
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist() 
categorical_features = X.select_dtypes(include=['object','category']).columns.tolist() 
print("\nNumeric:", numeric_features) 
print("Categorical:", categorical_features) 
 
 
numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), 
                               ('scaler', StandardScaler())]) 
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), 
                                   ('onehot', OneHotEncoder(handle_unknown='ignore', 
sparse_output=False))]) 
preprocessor = ColumnTransformer([ 
    ('num', numeric_transformer, numeric_features), 
    ('cat', categorical_transformer, categorical_features) 
]) 
 
 
models = { 
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), 
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE), 
    "RandomForest": RandomForestClassifier(n_estimators=200, 
random_state=RANDOM_STATE, n_jobs=-1, class_weight='balanced') 
} 
 
 
def evaluate_model(pipe, X_test, y_test): 
    preds = pipe.predict(X_test) 
    try: 
        proba = pipe.predict_proba(X_test)[:,1] 
    except: 
        proba = None 
    acc = accuracy_score(y_test, preds) 
    prec = precision_score(y_test, preds, zero_division=0) 
    rec = recall_score(y_test, preds, zero_division=0) 
    f1 = f1_score(y_test, preds, zero_division=0) 
    roc = roc_auc_score(y_test, proba) if proba is not None else None 
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | ROC-AUC: 
{roc if roc else 'N/A'}") 
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds)) 
    print("Classification Report:\n", classification_report(y_test, preds, zero_division=0)) 
    return {'accuracy':acc, 'precision':prec, 'recall':rec, 'f1':f1, 'roc_auc':roc} 
 
 
trained_pipelines = {} 
scores = {} 
for name, model in models.items(): 
    print(f"\n--- Training {name} ---") 
    pipe = Pipeline([('preprocessor', preprocessor), ('classifier', model)]) 
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE) 
    try: 
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1) 
        print("CV F1-scores:", cv_scores.round(4), "Mean:", cv_scores.mean()) 
    except Exception as e: 
        print("CV failed:", e) 
    pipe.fit(X_train, y_train) 
    sc = evaluate_model(pipe, X_test, y_test) 
    scores[name] = sc 
    trained_pipelines[name] = pipe 
 
 
best_model_name = max(scores.items(), key=lambda x:x[1]['f1'])[0] 
best_pipeline = trained_pipelines[best_model_name] 
print("Best model:", best_model_name) 
 
 
try: 
    y_proba = best_pipeline.predict_proba(X_test)[:,1] 
    RocCurveDisplay.from_predictions(y_test, y_proba) 
    plt.title(f"ROC Curve - {best_model_name}") 
    plt.show() 
except: 
    pass 
 
 
def get_feature_names(preprocessor_obj): 
    feature_names = [] 
    num_features = preprocessor_obj.transformers_[0][2] 
    feature_names.extend(num_features) 
    cat_transformer = preprocessor_obj.transformers_[1][1] 
    cat_features = preprocessor_obj.transformers_[1][2] 
    ohe = cat_transformer.named_steps['onehot'] 
    ohe_cols = ohe.get_feature_names_out(cat_features) 
    feature_names.extend(list(ohe_cols)) 
    return feature_names 
 
if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'): 
    feat_names = get_feature_names(best_pipeline.named_steps['preprocessor']) 
    importances = best_pipeline.named_steps['classifier'].feature_importances_ 
    fi = pd.Series(importances, index=feat_names).sort_values(ascending=False) 
    print("Top Feature Importances:\n", fi.head(15)) 
    plt.figure(figsize=(10,6)) 
    fi.head(20).plot(kind='barh') 
    plt.gca().invert_yaxis() 
    plt.title("Feature Importance") 
    plt.show() 
 
# ---------- Save best model ---------- 
joblib.dump(best_pipeline, f"loan_model_{best_model_name}.joblib") 
print(f"Saved best model: loan_model_{best_model_name}.joblib") 
 
 
def predict_single(sample_dict, pipeline=best_pipeline): 
    """ 
    Predicts Loan Approval for a single input sample. 
     
    sample_dict: dictionary with keys as feature names (excluding target) 
    Example: 
    sample = { 
        'Gender':'Male','Married':'Yes','Dependents':'0','Education':'Graduate', 
        'Self_Employed':'No','ApplicantIncome':50000,'CoapplicantIncome':0, 
        'LoanAmount':150,'Loan_Amount_Term':360,'Credit_History':1.0,'Property_Area':'Urban' 
    } 
    """ 
    df_sample = pd.DataFrame([sample_dict]) 
     
     
    df_sample['TotalIncome'] = df_sample['ApplicantIncome'].fillna(0) + 
df_sample.get('CoapplicantIncome',0).fillna(0) 
    df_sample['DebtToIncome'] = df_sample['LoanAmount'] / df_sample['TotalIncome'].replace(0, 
np.nan) 
    df_sample['ApplicantIncome_log'] = np.log1p(df_sample['ApplicantIncome'].fillna(0)) 
    df_sample['LoanAmount_log'] = np.log1p(df_sample['LoanAmount'].fillna(0)) 
    if 'Dependents' in df_sample.columns: 
        df_sample['Dependents'] = df_sample['Dependents'].replace('3+', '3').astype('object') 
     
     
    pred = pipeline.predict(df_sample)[0] 
    try: 
        proba = pipeline.predict_proba(df_sample)[0][1] 
    except: 
        proba = None 
    return pred, proba 
 
 
example = { 
    'Gender':'Male','Married':'Yes','Dependents':'0','Education':'Graduate', 
    'Self_Employed':'No','ApplicantIncome':45000,'CoapplicantIncome':5000, 
    'LoanAmount':120,'Loan_Amount_Term':360,'Credit_History':1.0,'Property_Area':'Urban' 
} 
 
pred, proba = predict_single(example) 
print("\nExample Prediction ->", "Approved (1)" if pred==1 else "Rejected (0)", "| 
Prob(Approved):", proba)