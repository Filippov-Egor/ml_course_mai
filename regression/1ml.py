import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

RANDOM_STATE = 42
TEST_SIZE = 0.2
OUT_DIR = Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 120


red = pd.read_csv(OUT_DIR / "winequality-red.csv", sep=";")
white = pd.read_csv(OUT_DIR / "winequality-white.csv", sep=";")
red["type"] = "red"
white["type"] = "white"
df = pd.concat([red, white], ignore_index=True)

print("=" * 70)
print("РАЗМЕР ДАТАСЕТА")
print("=" * 70)
print(f"Red:   {red.shape[0]} строк")
print(f"White: {white.shape[0]} строк")
print(f"Всего: {df.shape[0]} строк, {df.shape[1]} столбцов")
print()
print("Типы данных:")
print(df.dtypes)
print()
print("Пропуски:")
print(df.isnull().sum())
print()
print("Описательная статистика:")
print(df.describe().T.round(3))
print()


print("=" * 70)
print("РАСПРЕДЕЛЕНИЕ QUALITY")
print("=" * 70)
print(df["quality"].value_counts().sort_index())
print()

# графики
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(data=df, x="quality", ax=axes[0], palette="viridis")
axes[0].set_title("Распределение quality (все вина)")
sns.countplot(data=df, x="quality", hue="type", ax=axes[1], palette="Set2")
axes[1].set_title("Распределение quality по типу вина")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_quality_distribution.png", bbox_inches="tight")
plt.close()

# корреляции (только числовые)
num_cols = [c for c in df.columns if c != "type"]
corr = df[num_cols].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, linewidths=0.5)
plt.title("Корреляционная матрица признаков")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_correlation_heatmap.png", bbox_inches="tight")
plt.close()

print("Корреляция признаков с quality (по убыванию |r|):")
quality_corr = corr["quality"].drop("quality").abs().sort_values(ascending=False)
print(corr["quality"].drop("quality").reindex(quality_corr.index).round(3))
print()

# Boxplots ключевых признаков vs quality
key_features = ["alcohol", "volatile acidity", "density", "chlorides", "sulphates"]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()
for i, feat in enumerate(key_features):
    sns.boxplot(data=df, x="quality", y=feat, ax=axes[i], palette="viridis")
    axes[i].set_title(f"{feat} vs quality")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_boxplots_key_features.png", bbox_inches="tight")
plt.close()

# Выводы
print("=" * 70)
print("КРАТКИЕ ВЫВОДЫ ПО EDA")
print("=" * 70)
print("""
1. Датасет содержит 6497 образцов (1599 красных + 4898 белых).
   Пропусков нет. Все признаки непрерывные, quality — целое от 3 до 9.

2. Распределение quality сильно несбалансировано: большинство вин имеют
   оценку 5–6–7. Крайние значения (3, 4, 8, 9) встречаются редко.

3. Наиболее сильные корреляции с quality:
   - alcohol (+0.44) — чем выше содержание алкоголя, тем выше оценка.
   - density (−0.31) — более плотные вина чаще получают более низкую оценку
   - volatile acidity (−0.27) — летучая кислотность (уксусная) негативно влияет на качество
   - chlorides (−0.20) — избыток хлоридов ухудшает вкус.

4. Тип вина (red/white) влияет на распределения многих признаков
   (особенно residual sugar, total sulfur dioxide).
""")


feature_cols = [c for c in df.columns if c not in ("quality", "type")]
X = df[feature_cols]
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=None
)

print("=" * 70)
print("РАЗБИЕНИЕ НА ВЫБОРКИ")
print("=" * 70)
print(f"Обучающая выборка: {X_train.shape[0]} объектов ({100*(1-TEST_SIZE):.0f}%)")
print(f"Тестовая выборка:  {X_test.shape[0]} объектов ({100*TEST_SIZE:.0f}%)")
print(f"random_state = {RANDOM_STATE}")


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Baseline: константа = среднее quality на train (оптимально для RMSE)
baseline_const = y_train.mean()
y_pred_baseline = np.full_like(y_test, baseline_const, dtype=float)

baseline_rmse = rmse(y_test, y_pred_baseline)
baseline_mae = mae(y_test, y_pred_baseline)

print("=" * 70)
print("BASELINE")
print("=" * 70)
print(f"Константа (mean quality на train): {baseline_const:.4f}")
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Baseline MAE:  {baseline_mae:.4f}")
print("""
Почему среднее:
- Для RMSE оптимальная константа — среднее арифметическое целевой
  переменной на обучающей выборке (минимизирует сумму квадратов ошибок).
- Для MAE оптимальной была бы медиана, но основная метрика — RMSE.
""")


results = {}

# Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

results["Linear Regression"] = {
    "rmse": rmse(y_test, y_pred_lr),
    "mae": mae(y_test, y_pred_lr),
    "model": lr,
    "pred": y_pred_lr,
}

print("=" * 70)
print("LINEAR REGRESSION")
print("=" * 70)
print(f"RMSE: {results['Linear Regression']['rmse']:.4f}")
print(f"MAE:  {results['Linear Regression']['mae']:.4f}")

# Коэффициенты (на масштабированных данных — сравнимы)
coef_df = pd.DataFrame({
    "feature": feature_cols,
    "coefficient": lr.coef_,
    "abs_coef": np.abs(lr.coef_),
}).sort_values("abs_coef", ascending=False)

print("\nКоэффициенты (после StandardScaler, отсортированы по |coef|):")
print(coef_df.round(4).to_string(index=False))
print("""
Интерпретация:
- После стандартизации коэффициенты показывают изменение quality
  при изменении признака на 1 стандартное отклонение.
- Самые влиятельные: alcohol (+), volatile acidity (−), residual sugar (+),
  free sulfur dioxide (+), density (−), sulphates (+).
- Без масштабирования сравнивать |coef| нельзя, т.к. признаки в разных
  единицах (alcohol ~9–14, density ~0.99, chlorides ~0.04).
""")

# Random Forest
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

results["Random Forest"] = {
    "rmse": rmse(y_test, y_pred_rf),
    "mae": mae(y_test, y_pred_rf),
    "model": rf,
    "pred": y_pred_rf,
}

print("=" * 70)
print("RANDOM FOREST")
print("=" * 70)
print(f"RMSE: {results['Random Forest']['rmse']:.4f}")
print(f"MAE:  {results['Random Forest']['mae']:.4f}")

rf_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nFeature importance (Random Forest):")
print(rf_imp.round(4).to_string(index=False))

# Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

results["Gradient Boosting"] = {
    "rmse": rmse(y_test, y_pred_gb),
    "mae": mae(y_test, y_pred_gb),
    "model": gb,
    "pred": y_pred_gb,
}

print("=" * 70)
print("GRADIENT BOOSTING")
print("=" * 70)
print(f"RMSE: {results['Gradient Boosting']['rmse']:.4f}")
print(f"MAE:  {results['Gradient Boosting']['mae']:.4f}")

gb_imp = pd.DataFrame({
    "feature": feature_cols,
    "importance": gb.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nFeature importance (Gradient Boosting):")
print(gb_imp.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 6. Итоговая таблица
# ---------------------------------------------------------------------------
print("=" * 70)
print("ИТОГОВАЯ ТАБЛИЦА")
print("=" * 70)

table = pd.DataFrame({
    "Модель": ["Baseline", "Linear Regression", "Random Forest", "Gradient Boosting"],
    "RMSE": [
        baseline_rmse,
        results["Linear Regression"]["rmse"],
        results["Random Forest"]["rmse"],
        results["Gradient Boosting"]["rmse"],
    ],
    "MAE": [
        baseline_mae,
        results["Linear Regression"]["mae"],
        results["Random Forest"]["mae"],
        results["Gradient Boosting"]["mae"],
    ],
})
print(table.round(4).to_string(index=False))
table.to_csv(OUT_DIR / "results_table.csv", index=False)

print("""
Все модели превзошли baseline → они извлекают полезную информацию из признаков.

Сравнение:
- Linear Regression даёт заметное улучшение, но ограничена линейностью
  и не учитывает взаимодействия.
- Random Forest и Gradient Boosting существенно лучше благодаря
  нелинейности и взаимодействию признаков.
- Различия связаны с: (1) несбалансированностью quality,
  (2) наличием нелинейных зависимостей (alcohol, volatile acidity),
  (3) мультиколлинеарностью, которую деревья обрабатывают лучше.
""")


fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# LR abs coef
coef_df_sorted = coef_df.sort_values("abs_coef", ascending=True)
axes[0].barh(coef_df_sorted["feature"], coef_df_sorted["abs_coef"], color="steelblue")
axes[0].set_title("Linear Regression\n(|коэффициенты| после scaling)")
axes[0].set_xlabel("|coefficient|")

# RF
rf_imp_sorted = rf_imp.sort_values("importance", ascending=True)
axes[1].barh(rf_imp_sorted["feature"], rf_imp_sorted["importance"], color="forestgreen")
axes[1].set_title("Random Forest\n(feature importance)")
axes[1].set_xlabel("importance")

# GB
gb_imp_sorted = gb_imp.sort_values("importance", ascending=True)
axes[2].barh(gb_imp_sorted["feature"], gb_imp_sorted["importance"], color="darkorange")
axes[2].set_title("Gradient Boosting\n(feature importance)")
axes[2].set_xlabel("importance")

plt.tight_layout()
plt.savefig(OUT_DIR / "04_feature_importance.png", bbox_inches="tight")
plt.close()

print("=" * 70)
print("FEATURE IMPORTANCE — СРАВНЕНИЕ")
print("=" * 70)
print("""
Общие выводы:
- Во всех моделях alcohol — один из самых важных признаков
  (подтверждает корреляционный анализ).
- volatile acidity стабильно высоко важен (негативное влияние).
- density, sulphates, residual sugar также часто в топе.
- Random Forest и Gradient Boosting дают похожие рейтинги,
  но GB может сильнее фокусироваться на нескольких ключевых признаках.
- Линейная модель после scaling выделяет те же направления,
  но относительные веса могут отличаться из-за линейного предположения.

Можно ли утверждать причинность на основании feature importance?
Нет. Feature importance (и корреляции) показывают ассоциацию,
а не причинно-следственную связь. Изменение признака может быть
следствием другого скрытого фактора, либо корреляция может быть
случайной / опосредованной. Для причинных выводов нужны
эксперименты или методы causal inference.
""")


# Выбираем лучшую по RMSE
best_name = min(results, key=lambda k: results[k]["rmse"])
best_pred = results[best_name]["pred"]
best_rmse = results[best_name]["rmse"]

print("=" * 70)
print(f"АНАЛИЗ ОШИБОК — лучшая модель: {best_name} (RMSE={best_rmse:.4f})")
print("=" * 70)

errors = y_test.values - best_pred
abs_errors = np.abs(errors)

error_df = pd.DataFrame({
    "true": y_test.values,
    "pred": best_pred,
    "error": errors,
    "abs_error": abs_errors,
}, index=y_test.index)

print("Статистика ошибок:")
print(error_df[["error", "abs_error"]].describe().round(3))
print()

# Распределение ошибок
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(errors, bins=40, kde=True, ax=axes[0], color="steelblue")
axes[0].axvline(0, color="red", linestyle="--")
axes[0].set_title("Распределение ошибок (true − pred)")
axes[0].set_xlabel("error")

sns.scatterplot(x=y_test, y=best_pred, alpha=0.4, ax=axes[1])
axes[1].plot([3, 9], [3, 9], "r--", label="идеал")
axes[1].set_xlabel("True quality")
axes[1].set_ylabel("Predicted quality")
axes[1].set_title("True vs Predicted")
axes[1].legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "05_error_analysis.png", bbox_inches="tight")
plt.close()

# Ошибки по значениям quality
print("Средняя абсолютная ошибка по истинным значениям quality:")
mae_by_q = error_df.groupby("true")["abs_error"].agg(["mean", "count"])
print(mae_by_q.round(3))
print()

# Крупнейшие ошибки
print("Топ-10 объектов с наибольшими абсолютными ошибками:")
top_err = error_df.nlargest(10, "abs_error")
print(top_err.round(3))
print()

# Присоединяем признаки для интерпретации
top_err_full = top_err.join(X_test)
print("Признаки объектов с крупнейшими ошибками (первые 5):")
print(top_err_full[feature_cols + ["true", "pred", "abs_error"]].head().round(3).T)
print("""
Возможные причины крупных ошибок:
- Редкие значения quality (3, 4, 8, 9) — мало примеров в обучении,
  модель «не видела» похожих объектов.
- Выбросы по отдельным признакам (очень высокий residual sugar,
  аномальная volatile acidity и т.п.).
- Субъективность экспертной оценки quality: два вина с похожими
  физико-химическими свойствами могут получить разные баллы.
- Модель усредняет предсказания → для крайних quality
  предсказания смещаются к центру распределения (regression to the mean).

Почему хуже на редких quality:
- Мало обучающих примеров → деревья/линейная модель не могут
  надёжно оценить функцию в этих областях.
- Loss (MSE) доминируется частыми классами 5–6–7, поэтому
  оптимизация «жертвует» точностью на хвостах.
""")

np.savez(
    OUT_DIR / "predictions.npz",
    y_test=y_test.values,
    baseline=y_pred_baseline,
    lr=y_pred_lr,
    rf=y_pred_rf,
    gb=y_pred_gb,
)
