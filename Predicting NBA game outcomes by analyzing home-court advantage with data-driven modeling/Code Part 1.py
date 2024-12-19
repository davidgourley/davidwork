import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import kagglehub

path = kagglehub.dataset_download("wyattowalsh/basketball")
print("Path to dataset files:", path)
dataset = pd.read_csv(f"{path}/csv/game.csv") 
print("Shape of the dataset:", dataset.shape)

# relevant columns
selected_columns = ['team_id_home', 'team_name_home', 'pts_home', 'wl_home', 'team_id_away', 'team_name_away', 'pts_away', 'wl_away', 'game_date']
dataset = dataset[selected_columns]
dataset.tail()
dataset.head()

null_values = dataset.isnull().sum()
print("Null values:")
print(null_values)

# remove rows w nulls
dataset = dataset.dropna(subset=['wl_home', 'wl_away'])

# find outliers with boxplot
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
sns.boxplot(y=dataset['pts_home'])
plt.title('Home Team Points - Boxplot')
plt.subplot(1, 2, 2)
sns.boxplot(y=dataset['pts_away'])
plt.title('Away Team Points - Boxplot')
plt.show()

# same as above but w scatterplot
plt.figure(figsize=(11, 7))
plt.scatter(dataset['pts_home'], dataset['pts_away'])
plt.xlabel('Home Team Points')
plt.ylabel('Away Team Points')
plt.title('Home vs Away Team Points - Scatterplot')
plt.show()

# remove outliers using iqr
def remove_outliers(df, column):
     Q1 = df[column].quantile(0.25)
     Q3 = df[column].quantile(0.75)
     IQR = Q3 - Q1
     lower_bound = Q1 - 1.5 * IQR
     upper_bound = Q3 + 1.5 * IQR
     return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

dataset_no_outliers = remove_outliers(dataset, 'pts_home')
dataset_no_outliers = remove_outliers(dataset_no_outliers, 'pts_away')

# compare before & after
plt.figure(figsize=(14, 6))

plt.subplot(2, 2, 1)
sns.boxplot(y=dataset['pts_home'])
plt.title('Boxplot - Home Team Points (Old)')

plt.subplot(2, 2, 2)
sns.boxplot(y=dataset['pts_away'])
plt.title('Boxplot - Away Team Points (Old)')

#

plt.subplot(2, 2, 3)
sns.boxplot(y=dataset_no_outliers['pts_home'])
plt.title('Boxplot - Home Team Points (No Outliers)')

plt.subplot(2, 2, 4)
sns.boxplot(y=dataset_no_outliers['pts_away'])
plt.title('Boxplot - Away Team Points (No Outliers)')

plt.tight_layout()
plt.show()

# before & after - scatterplot
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(dataset['pts_home'], dataset['pts_away'], alpha=0.5)
plt.xlabel('Home Team Points')
plt.ylabel('Away Team Points')
plt.title('Scatterplot of Home vs Away Team Points (Old)')

plt.subplot(1, 2, 2)
plt.scatter(dataset_no_outliers['pts_home'], dataset_no_outliers['pts_away'], alpha=0.5)
plt.xlabel('Home Team Points')
plt.ylabel('Away Team Points')
plt.title('Scatterplot of Home vs Away Team Points (No Outliers)')

plt.tight_layout()
plt.show()
dataset.info()

# win-loss distrib for home / away teams
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.countplot(data=dataset, x='wl_home')
plt.title('Win-Loss Distribution for Home Teams')
plt.xlabel('Win (W) or Loss (L)')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
sns.countplot(data=dataset, x='wl_away')
plt.title('Win-Loss Distribution for Away Teams')
plt.xlabel('Win (W) or Loss (L)')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
#above code shows win differences at home or away

#

# histograms w/KDE for pts scored by home vs away teams
plt.figure(figsize=(14, 6))

# histogram for home pts w/ KDE overlay
plt.subplot(1, 2, 1)
sns.histplot(dataset['pts_home'], bins=20, kde=True, color='blue', alpha=1)
plt.title('Points Scored Distribution for Home Teams')
plt.xlabel('Points')
plt.ylabel('Frequency')

# histogram for away pts w/ KDE overlay
plt.subplot(1, 2, 2)
sns.histplot(dataset['pts_away'], bins=20, kde=True, color='green', alpha=1)
plt.title('Points Scored Distribution for Away Teams')
plt.xlabel('Points')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

# histogram w/ KDE overlay for pts by home teams
sns.histplot(dataset['pts_home'], bins=20, kde=True, color='blue', label='Home', alpha=0.8)

# histogram w/ KDE overlay for pts by away teams
sns.histplot(dataset['pts_away'], bins=20, kde=True, color='green', label='Away', alpha=0.8)

plt.title('Points scored distribution for home vs away teams')
plt.xlabel('Points')
plt.ylabel('Frequency')
plt.legend()
plt.show()
#shows how many pts teams usually score home or away

# avg pts scored by home and away teams
avg_pts_home = dataset['pts_home'].mean()
avg_pts_away = dataset['pts_away'].mean()

plt.figure(figsize=(8, 6))
sns.barplot(x=['Home', 'Away'], y=[avg_pts_home, avg_pts_away], palette='viridis')
plt.title('Average points scored by home and away teams')
plt.xlabel('Team')
plt.ylabel('Average Points')

plt.show()

# compare home vs away pts
plt.figure(figsize=(8, 6))
sns.scatterplot(data=dataset, x='pts_home', y='pts_away', alpha=0.5)
plt.title('Home vs. Away Points Comparison')
plt.xlabel('Home Points')
plt.ylabel('Away Points')

plt.show()

dataset['game_date'] = pd.to_datetime(dataset['game_date'])
dataset['year'] = dataset['game_date'].dt.year
dataset['month'] = dataset['game_date'].dt.month

# find monthly win-loss ratios for home & away
monthly_home_wins = dataset[dataset['wl_home'] == 'W'].groupby(['year', 'month']).size()
monthly_home_losses = dataset[dataset['wl_home'] == 'L'].groupby(['year', 'month']).size()
monthly_away_wins = dataset[dataset['wl_away'] == 'W'].groupby(['year', 'month']).size()
monthly_away_losses = dataset[dataset['wl_away'] == 'L'].groupby(['year', 'month']).size()

monthly_win_loss = pd.DataFrame({
    'home_wins': monthly_home_wins,
    'home_losses': monthly_home_losses,
    'away_wins': monthly_away_wins,
    'away_losses': monthly_away_losses
}).fillna(0).reset_index()

#win ratios
monthly_win_loss['home_win_ratio'] = monthly_win_loss['home_wins'] / (monthly_win_loss['home_wins'] + monthly_win_loss['home_losses'])
monthly_win_loss['away_win_ratio'] = monthly_win_loss['away_wins'] / (monthly_win_loss['away_wins'] + monthly_win_loss['away_losses'])

# find avg points per month for home & away
monthly_avg_points = dataset.groupby(['year', 'month']).agg({
    'pts_home': 'mean',
    'pts_away': 'mean'
}).reset_index()

# merge win-loss ratios & average pts
monthly_stats = pd.merge(monthly_win_loss, monthly_avg_points, on=['year', 'month'])

# plot win-loss ratios
plt.figure(figsize=(14, 6))
plt.plot(monthly_stats['month'], monthly_stats['home_win_ratio'], label='Home Win Ratio', marker='o')
plt.plot(monthly_stats['month'], monthly_stats['away_win_ratio'], label='Away Win Ratio', marker='o')
plt.title('Monthly Win Ratios for Home and Away Games')
plt.xlabel('Month')
plt.ylabel('Win Ratio')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.show()

# plot avg points
plt.figure(figsize=(14, 6))
plt.plot(monthly_stats['month'], monthly_stats['pts_home'], label='Home Team Average Points', marker='o')
plt.plot(monthly_stats['month'], monthly_stats['pts_away'], label='Away Team Average Points', marker='o')
plt.title('Monthly Average Points for Home and Away Teams')
plt.xlabel('Month')
plt.ylabel('Average Points')
plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.show()


#

#

#

# preprocess data
dataset['game_date'] = pd.to_datetime(dataset['game_date'])
dataset = dataset.dropna(subset=['wl_home', 'wl_away'])

dataset['wl_home_binary'] = dataset['wl_home'].apply(lambda x: 1 if x == 'W' else 0)

# features
X = dataset[['pts_home', 'pts_away']]
y = dataset['wl_home_binary']

# split data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# evaluate
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# roc curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# precision recall curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

#

#

#

# evaluation
y_pred = model.predict(X_test)  # Logistic regression model prediction
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Logistic Regression:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()