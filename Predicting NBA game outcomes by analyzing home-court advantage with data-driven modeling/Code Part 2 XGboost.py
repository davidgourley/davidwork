# Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve

import kagglehub
path = kagglehub.dataset_download("wyattowalsh/basketball")
print("Path to dataset files:", path)
dataset = pd.read_csv(f"{path}/csv/game.csv")  
print("Shape of the dataset:", dataset.shape)

# extract relevant columns
selected_columns = ['team_id_home', 'team_name_home', 'pts_home', 'wl_home', 'team_id_away', 'team_name_away', 'pts_away', 'wl_away', 'game_date']
dataset = dataset[selected_columns]

print(dataset.head())
print(dataset.tail())

null_values = dataset.isnull().sum()
print("Null values:")
print(null_values)

# remove
dataset = dataset.dropna(subset=['wl_home', 'wl_away'])

#feature engineering!!
label_encoder = LabelEncoder()
dataset['wl_home_binary'] = label_encoder.fit_transform(dataset['wl_home'])
dataset['wl_away_binary'] = label_encoder.fit_transform(dataset['wl_away'])

dataset['game_date'] = pd.to_datetime(dataset['game_date'])
dataset['year'] = dataset['game_date'].dt.year
dataset['month'] = dataset['game_date'].dt.month

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

dataset_no_outliers = remove_outliers(dataset, 'pts_home')
dataset_no_outliers = remove_outliers(dataset_no_outliers, 'pts_away')

features = ['pts_home', 'pts_away', 'year', 'month']
target = 'wl_home_binary'

X = dataset_no_outliers[features]
y = dataset_no_outliers[target]

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# TRAIN
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train_scaled, y_train)

# EVLUATION
y_pred = xgb.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of XGBoost Classifier:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
y_pred_prob = xgb.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# feature importance - not necessary
feature_importances = pd.Series(xgb.feature_importances_, index=features)
feature_importances.sort_values().plot(kind='barh', title='Feature Importance in XGBoost')
plt.show()
