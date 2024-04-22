import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Load data
df = pd.read_csv('./Tesla.csv')
st.set_option('deprecation.showPyplotGlobalUse', False)
# Plot Tesla Close price
st.write("### Tesla Close price")
plt.figure(figsize=(15,5))
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(df['Close'], color='red')
ax.set_title('Tesla Close price', fontsize=15)
ax.set_ylabel('Price in dollars')
st.pyplot(fig)

# Preprocessing
df = df.drop(['Adj Close'], axis=1)
splitted = df['Date'].str.split('/', expand=True)
df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Visualizations
st.write("### Distribution Plots")
plt.figure(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
    plt.subplot(2,3,i+1)
    sns.displot(df[col], color="red")
st.pyplot()

st.write("### Box Plots")
plt.figure(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close', 'Volume']):
    plt.subplot(2,3,i+1)
    sns.boxplot(df[col], color="green")
st.pyplot()

st.write("### Bar Plots (Yearly Mean)")
df['Open'] = pd.to_numeric(df['Open'], errors='coerce') # Convert 'Open' column to numeric
df['High'] = pd.to_numeric(df['High'], errors='coerce') # Convert 'High' column to numeric
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')   # Convert 'Low' column to numeric
df['Close'] = pd.to_numeric(df['Close'], errors='coerce') # Convert 'Close' column to numeric
df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True) 

data_grouped = df.groupby('year')[['Open', 'High', 'Low', 'Close']].mean()
# data_grouped = df.groupby('year').mean()
plt.figure(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
    plt.subplot(2,2,i+1)
    data_grouped[col].plot.bar()
st.pyplot()

# Heatmap
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
st.write("### Correlation Heatmap")
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix > 0.9, annot=True, cbar=False)
st.pyplot()

# Feature Selection
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Train-test split
X_train, X_valid, Y_train, Y_valid = train_test_split(features, target, test_size=0.1, random_state=2022)

# Model training and evaluation
st.write("### Model Training and Evaluation")
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
for i, model in enumerate(models):
    model.fit(X_train, Y_train)
    st.write(f"Model {i+1}: {model}")
    train_auc = metrics.roc_auc_score(Y_train, model.predict_proba(X_train)[:,1])
    valid_auc = metrics.roc_auc_score(Y_valid, model.predict_proba(X_valid)[:,1])
    st.write(f"Training Accuracy: {train_auc}")
    st.write(f"Validation Accuracy: {valid_auc}")
    st.write('---')

# Confusion Matrix
st.write("### Confusion Matrix (Logistic Regression Model)")
cm = confusion_matrix(Y_valid, models[0].predict(X_valid))

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
st.pyplot()
