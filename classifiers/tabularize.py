import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("fraud.csv")

print(f"Initial shape: {df.shape}")
print(df.head())

cols_to_encode = ['city', 'state', 'sic', 'incorp_state', 'filing_type']
df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

# df['filing_date'] = pd.to_datetime(df['filing_date'], format='%d-%m-%Y', errors='coerce')
df['filing_date'] = pd.to_datetime(df['filing_date'], format='%Y-%m-%d', errors='coerce')
df['reporting_date'] = pd.to_datetime(df['reporting_date'], format='%Y-%m-%d', errors='coerce')
df['dateTime'] = df['dateTime'].str.split('T').str[0]
df['dateTime'] = pd.to_datetime(df['dateTime'], format='%Y-%m-%d', errors='coerce')
df['fraud_start'] = pd.to_datetime(df['fraud_start'], errors='coerce')
df['fraud_end'] = pd.to_datetime(df['fraud_end'], errors='coerce')
df['revoked'] = pd.to_datetime(df['revoked'], format='%m-%Y', errors='coerce')
df['fye'] = pd.to_datetime(df['fye'], format='%Y-%m-%d', errors='coerce')

breakpoint()

cols =['cik', 'name', 'url', 'mda', 'dateTime', 
       'respondents', 'fraud_start', 'fraud_end', 'revoked',
       'certainty_start', 'certainty_end', '17a', '17a2', '17a3', '17b', '5a',
       '5b1', '5c', '10b', '13a', '12b20', '12b25', '13a1', '13a10', '13a11',
       '13a13', '13a14', '13a16', '13b2A', '13b2B', '13a15', '13b5', '14a',
       '14c', '19a', '30A', '100a2', '100b', '105c7B', 'corruption', 'amis',
       'fsf']

df.drop(cols, axis=1, inplace=True)

breakpoint()

date_cols = df.select_dtypes(include=['datetime64']).columns

for col in date_cols:
    df[f'{col}_year'] = df[col].dt.year
    df[f'{col}_month'] = df[col].dt.month
    df[f'{col}_day'] = df[col].dt.day
    df.drop(columns=[col], inplace=True)

df.drop('fye_year', axis=1, inplace=True)

print(f"Shape after preprocessing: {df.shape}")
print(df.head())

df.drop(['word_count', 'char_count', 'word_density'], axis=1, inplace=True)

df.to_csv('/Users/malla/Uni/MA/fraud-analysis/df_tab.csv', index=False)

# X = df.drop('fraudulent', axis=1)
# y = df['fraudulent']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# rf = RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42)

# rf.fit(X_train, y_train)

# y_pred = rf.predict(X_test)

# y_probs = rf.predict_proba(X_test)[:, 1]

# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred))
# auc = roc_auc_score(y_test, y_probs)
# print(f"ROC AUC = {auc:.3f}")