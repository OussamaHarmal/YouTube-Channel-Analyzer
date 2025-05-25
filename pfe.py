import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from math import sqrt
import pickle


data = pd.read_csv(r"C:\Users\dell\Desktop\YouTube-Performance-Project\youtube_channel_real_performance_analytics.csv")


data['Video Publish Time'] = pd.to_datetime(data['Video Publish Time'], errors='coerce')
data['Day of Week'] = data['Day of Week'].astype('category')
data['CTR (%)'] = data['Views'] / data['Impressions'] * 100
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

sns.pairplot(data[['Views', 'Impressions', 'CTR (%)', 'Estimated Revenue (USD)', 'Watch Time (hours)','Video Duration']],
    height=1.8, aspect=1)
plt.suptitle("Pairplot of Key Features", y=0.95)
plt.show()

corr = data[['Views', 'Impressions', 'CTR (%)', 'Estimated Revenue (USD)', 'Watch Time (hours)','Average View Duration','Video Duration','Revenue per 1000 Views (USD)','Shares']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

best_model = None
best_rmse = float('inf')
best_model_name = ""

features = ['Video Duration','Shares', 'Impressions']
target = 'Views'

X = data[features]
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(alpha=0.1),
    "XGBoost": XGBRegressor(n_estimators=400, max_depth=100,learning_rate=0.01, subsample=0.5,random_state=42)
}
dataaaa=[]
for name, model in models.items():
    print(f"\n=== {name} ===")
    if name in ["Linear Regression", "Lasso"]:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    rmse = sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    dataaaa.append({"Model": name, "RMSE": rmse, "R2": r2})
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_model_name = name


    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, preds, alpha=0.6, color='green', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal')
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions RF")
    plt.title("Prédictions vs Réels - "+name)
    plt.legend()
    plt.show()
    
with open("best_model_views.pkl", "wb") as f:
    pickle.dump(best_model, f)

print(f"le modele {best_model_name} a ete sauvegarde") 
results_df = pd.DataFrame(dataaaa)
plt.figure(figsize=(8, 5))
sns.barplot(x="RMSE", y="Model", data=results_df, palette="viridis")
plt.title("Comparaison des modèles par RMSE")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="R2", y="Model", data=results_df, palette="viridis")
plt.title("Comparaison des modèles par R2")
plt.tight_layout()
plt.show()

features_rf = ['Watch Time (hours)', 'Views', 'CTR (%)', 'Revenue per 1000 Views (USD)']
target_rf = 'Estimated Revenue (USD)'

X_rf = data[features_rf]
y_rf = data[target_rf]

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled2 = scaler.fit_transform(X_train_rf)
X_test_scaled2 = scaler.transform(X_test_rf)

modeles = {
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(random_state=42)
}

result = []

for names, models in modeles.items():
    print(f"\n=== {names} ===")
    if names == "Lasso":
        models.fit(X_train_scaled2, y_train_rf)
        y_pred = models.predict(X_test_scaled2)
    else:
        models.fit(X_train_rf, y_train_rf)
        y_pred = models.predict(X_test_rf)
        importance_rf=models.feature_importances_

    rmse = sqrt(mean_squared_error(y_test_rf, y_pred))
    r2 = r2_score(y_test_rf, y_pred)
    result.append({"Model": names, "RMSE": rmse, "R2": r2})

    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    if rmse < best_rmse:
        best_rmse = rmse
        best_model_revenue = models
        best_model_name = names
    
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test_rf, y_pred, alpha=0.6, color='green', label='Predictions')
    plt.plot([y_test_rf.min(), y_test_rf.max()], [y_test_rf.min(), y_test_rf.max()], 'r--', label='Ideal')
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title(f"Prédictions vs Réels - {names}")
    plt.legend()
    plt.grid(True)
    plt.show()

with open("best_model_revenue.pkl", "wb") as f:
    pickle.dump(best_model_revenue, f)

print(f"le modele {best_model_name} a ete sauvegarde")

resultat = pd.DataFrame(result)

plt.figure(figsize=(8, 5))
sns.barplot(x="RMSE", y="Model", data=resultat, palette="viridis")
plt.title("Comparaison des modèles par RMSE")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="R2", y="Model", data=resultat , palette="viridis")
plt.title("Comparaison des modèles par R2")
plt.tight_layout()
plt.show()




features_xgb = ['Average View Duration', 'CTR (%)', 'Impressions', 'Video Duration']
target_xgb = 'Watch Time (hours)'

X_xgb = data[features_xgb]
y_xgb = data[target_xgb]

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, random_state=42)

modele_watch_time = {
    "XGBoost Regressor": XGBRegressor(random_state=42),
    "CatBoost Regressor": CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=0)
}

resultat_watch = []
best_rmse_watch = float('inf')
best_model_watch_time = None
best_model_watch_name = ""

for name, modelu in modele_watch_time.items():
    print(f"------- {name} ----------")
    
    modelu.fit(X_train_xgb, y_train_xgb)
    predictions = modelu.predict(X_test_xgb)
    
    rmse = sqrt(mean_squared_error(y_test_xgb, predictions))
    r2 = r2_score(y_test_xgb, predictions)
    
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
    
    if rmse < best_rmse_watch:
        best_rmse_watch = rmse
        best_model_watch_time = modelu
        best_model_watch_name = name
    
    resultat_watch.append({
        "Model": name,
        "RMSE": rmse,
        "R2": r2
    })    
    
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test_xgb, predictions, alpha=0.6, color='green', label="Prédictions")
    plt.plot([y_test_xgb.min(), y_test_xgb.max()], [y_test_xgb.min(), y_test_xgb.max()], 'r--', label='Idéal')
    plt.title("Prédictions vs Réelles - " + name)
    plt.xlabel("Valeurs Réelles")
    plt.ylabel("Prédictions")
    plt.legend()
    plt.show()

with open("best_model_watch_time.pkl", "wb") as f:
    pickle.dump(best_model_watch_time, f)

print(f"le modele {best_model_name} a ete sauvegarde")
resu = pd.DataFrame(resultat_watch)

plt.figure(figsize=(8, 5))
sns.barplot(x="RMSE", y="Model", data=resu, palette="viridis")
plt.title("Comparaison des modèles par RMSE")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x="R2", y="Model", data=resu , palette="viridis")
plt.title("Comparaison des modèles par R2")
plt.tight_layout()
plt.show()

    
importance_df = pd.DataFrame({
    'Feature': features_rf,
    'Importance': importance_rf
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.title("Feature Importances - Random Forest (Sorted)")
plt.tight_layout()
plt.show()

