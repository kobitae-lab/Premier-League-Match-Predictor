import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

df_model = pd.read_pickle("df_model.pkl")

feature_cols = [

    'RollingGoalDif_Home', 'RollingGoalDif_Away',
    'RollingShotDif_Home', 'RollingShotDif_Away'
]

feature_cols_old = [
    'RollingGoalsFor_Home', 'RollingGoalsAgainst_Home', 
    'RollingShotsFor_Home', 'RollingShotsAgainst_Home',
    'RollingGoalsFor_Away', 'RollingGoalsAgainst_Away', 
    'RollingShotsFor_Away', 'RollingShotsAgainst_Away'
]

x = df_model[feature_cols]
y = df_model['FTR']

mask = x.notna().all(axis=1)
x = x[mask]
y = y[mask]

# Randomly split data 80 / 20
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#model = RandomForestClassifier(n_estimators=100, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)


print(classification_report(Y_test, Y_pred))