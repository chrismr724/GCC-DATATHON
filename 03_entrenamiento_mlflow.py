# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Entrenamiento con MLflow (3 algoritmos × 3 resistencias)
# MAGIC - Modelos: **ElasticNet** (lineal regularizado, con escalado), **RandomForest** y **HistGradientBoosting**.
# MAGIC - Búsqueda de hiperparámetros: `RandomizedSearchCV` con KFold (solo sobre train). El test (20 %) se toca una sola vez.
# MAGIC - Cada modelo × target = 1 run hijo con parámetros, R², MAE, MAPE, gráficos y modelo registrado en Unity Catalog.
# MAGIC - Al final: tablas para el dashboard y **un modelo multi-salida** (calidad cruda → Strength1/7/28).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy.stats import loguniform, uniform
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

setup_mlflow()
client = MlflowClient()

# COMMAND ----------

# MAGIC %md ## Datos: Gold para entrenar, Silver para explicar con variables crudas

# COMMAND ----------

gold = read_table(T_GOLD)
silver = read_table(T_SILVER).set_index("row_id")
FEATURES = [c for c in gold.columns if c not in META_COLS + TARGETS]
assert not set(TARGETS) & set(FEATURES)

# Reconstruimos el Preprocessor con las filas de train y verificamos que reproduce la capa Gold
train_ids = gold.loc[gold["split"] == "train", "row_id"]
prep = Preprocessor().fit(silver.loc[train_ids].reset_index(drop=True))
rebuilt = prep.transform(silver.loc[gold["row_id"]].reset_index(drop=True))[FEATURES]
assert np.allclose(rebuilt.values, gold[FEATURES].values, equal_nan=True), "Gold no coincide con el Preprocessor"
print(f"{len(FEATURES)} features | train={len(train_ids)} | test={(gold['split']=='test').sum()}")

# COMMAND ----------

# MAGIC %md ## Modelos y espacios de búsqueda

# COMMAND ----------

MODELS = {
    "elasticnet": (
        Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=50000, random_state=RANDOM_STATE))]),
        {"model__alpha": loguniform(1e-3, 10), "model__l1_ratio": uniform(0.05, 0.95)},
    ),
    "random_forest": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        {"n_estimators": [200, 300, 500], "max_depth": [None, 6, 10, 14],
         "min_samples_leaf": [1, 2, 4, 8], "max_features": [0.3, 0.5, 0.7, 1.0]},
    ),
    "hist_gradient_boosting": (
        HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        {"learning_rate": loguniform(0.01, 0.2), "max_iter": [200, 400, 800], "max_leaf_nodes": [8, 15, 31],
         "min_samples_leaf": [5, 10, 20, 40], "l2_regularization": loguniform(1e-3, 10), "max_depth": [None, 4, 6]},
    ),
}


def log_sklearn(model, signature, example, registered_name):
    try:   # MLflow 3.x
        return mlflow.sklearn.log_model(model, name="model", signature=signature,
                                        input_example=example, registered_model_name=registered_name,
                                        serialization_format="cloudpickle")
    except TypeError:  # MLflow 2.x
        return mlflow.sklearn.log_model(model, artifact_path="model", signature=signature,
                                        input_example=example, registered_model_name=registered_name,
                                        serialization_format="cloudpickle")


def metrics_dict(y, p):
    return {"r2": r2_score(y, p), "mae": mean_absolute_error(y, p), "mape_pct": 100 * mean_absolute_percentage_error(y, p)}

# COMMAND ----------

# MAGIC %md ## Entrenamiento

# COMMAND ----------

cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results, pred_rows, imp_rows, fitted = [], [], [], {}

with mlflow.start_run(run_name="train_3_modelos_x_3_targets") as parent:
    mlflow.log_params({"random_state": RANDOM_STATE, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS,
                       "n_iter_search": N_ITER_SEARCH, "n_features": len(FEATURES)})
    mlflow.log_text("\n".join(FEATURES), "features.txt")

    for target in TARGETS:
        tr = gold[(gold["split"] == "train") & gold[target].notna()]
        te = gold[(gold["split"] == "test") & gold[target].notna()]
        Xtr, ytr, Xte, yte = tr[FEATURES], tr[target], te[FEATURES], te[target]
        raw_te = silver.loc[te["row_id"]].reset_index()

        for mtype, (estimator, space) in MODELS.items():
            with mlflow.start_run(run_name=f"{mtype}__{target}", nested=True) as run:
                search = RandomizedSearchCV(estimator, space, n_iter=N_ITER_SEARCH, cv=cv,
                                            scoring="neg_mean_absolute_error", random_state=RANDOM_STATE,
                                            n_jobs=-1, refit=True)
                search.fit(Xtr, ytr)
                best = search.best_estimator_
                m_te = metrics_dict(yte, best.predict(Xte))
                m_tr = metrics_dict(ytr, best.predict(Xtr))
                cv_mae = float(-search.best_score_)

                mlflow.log_params({"model_type": mtype, "target": target, "n_features": len(FEATURES),
                                   "train_size": len(Xtr), "test_size": len(Xte), "random_state": RANDOM_STATE,
                                   **{f"best_{k}": v for k, v in search.best_params_.items()}})
                mlflow.log_metrics({"r2": m_te["r2"], "mae": m_te["mae"], "mape_pct": m_te["mape_pct"],
                                    "cv_mae": cv_mae, "train_r2": m_tr["r2"], "train_mae": m_tr["mae"]})

                # Real vs predicho y residuales
                pte = best.predict(Xte)
                fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
                ax[0].scatter(yte, pte, s=14, alpha=0.7)
                lim = [min(yte.min(), pte.min()), max(yte.max(), pte.max())]
                ax[0].plot(lim, lim, "r--"); ax[0].set_xlabel("Real"); ax[0].set_ylabel("Predicho")
                ax[0].set_title(f"{mtype} · {target} · R²={m_te['r2']:.3f}")
                ax[1].scatter(pte, yte - pte, s=14, alpha=0.7); ax[1].axhline(0, color="r", ls="--")
                ax[1].set_xlabel("Predicho"); ax[1].set_ylabel("Residual (real − predicho)")
                ax[1].set_title("Residuales")
                fig.tight_layout()
                mlflow.log_figure(fig, "real_vs_predicho_y_residuales.png"); plt.close(fig)

                # Importancia por permutación sobre variables crudas (test)
                imp = raw_permutation_importance(best, prep, raw_te, yte.values)
                mlflow.log_text(imp.to_csv(index=False), "feature_importance.csv")
                fig, ax = plt.subplots(figsize=(7, 4.5))
                ax.barh(imp["feature"][::-1], imp["importance_mae"][::-1], xerr=imp["importance_std"][::-1], color="#4C72B0")
                ax.set_xlabel("Aumento del MAE al barajar la variable"); ax.set_title(f"Importancia · {mtype} · {target}")
                fig.tight_layout()
                mlflow.log_figure(fig, "feature_importance.png"); plt.close(fig)

                # Modelo registrado en Unity Catalog (con firma y ejemplo de entrada)
                reg_name = f"{CATALOG}.gold.{mtype}_{target.lower()}"
                sig = infer_signature(Xtr, best.predict(Xtr))
                info = log_sklearn(best, sig, Xtr.head(3), reg_name)

                results.append({"model_type": mtype, "target": target, **m_te, "cv_mae": cv_mae,
                                "train_r2": m_tr["r2"], "n_train": len(Xtr), "n_test": len(Xte),
                                "n_features": len(FEATURES), "run_id": run.info.run_id,
                                "registered_model": reg_name,
                                "model_version": str(getattr(info, "registered_model_version", "") or "")})
                fitted[(mtype, target)] = best
                imp_rows.append(imp.assign(model_type=mtype, target=target))

    # ---- Campeón por target: menor MAE de validación cruzada (no se elige mirando test) ----
    res = pd.DataFrame(results)
    res["is_champion"] = False
    for target in TARGETS:
        idx = res[res["target"] == target]["cv_mae"].idxmin()
        res.loc[idx, "is_champion"] = True
    champions = {r.target: r.model_type for r in res[res["is_champion"]].itertuples()}
    mlflow.log_dict(champions, "champions.json")
    for r in res[res["is_champion"]].itertuples():
        try:
            if r.model_version:
                client.set_registered_model_alias(r.registered_model, "champion", r.model_version)
        except Exception as e:
            print("alias no asignado:", e)

    # ---- Tabla comparativa 3 modelos × 3 targets ----
    comp = res[["target", "model_type", "r2", "mae", "mape_pct", "cv_mae", "is_champion"]].sort_values(["target", "mape_pct"])
    show(comp.round(4))
    mlflow.log_text(comp.to_csv(index=False), "tabla_comparativa.csv")

# COMMAND ----------

# MAGIC %md ## Tablas para el dashboard (métricas, predicciones, importancias, explicaciones locales)

# COMMAND ----------

champion_keys = {(r.model_type, r.target) for r in res[res["is_champion"]].itertuples()}
imp_all = pd.concat(imp_rows, ignore_index=True)
imp_all["is_champion"] = [(m, t) in champion_keys for m, t in zip(imp_all["model_type"], imp_all["target"])]

# Predicciones de todas las filas (split indica si es train o test; las métricas honestas son solo de test)
raw_all = silver.loc[gold["row_id"]].reset_index()
X_all = gold[FEATURES]
for (mtype, target), model in fitted.items():
    pred = model.predict(X_all)
    y = gold[target].values
    mean_test = gold.loc[(gold["split"] == "test") & gold[target].notna(), target].mean()
    d = pd.DataFrame({"row_id": gold["row_id"], "split": gold["split"], CAT_COL: gold[CAT_COL],
                      "model_type": mtype, "target": target, "y_true": y, "y_pred": pred})
    d["error"] = d["y_true"] - d["y_pred"]
    d["abs_error"] = d["error"].abs()
    d["ape_pct"] = 100 * d["abs_error"] / d["y_true"].abs()
    d["sq_error"] = d["error"] ** 2
    d["sq_dev"] = (d["y_true"] - mean_test) ** 2
    d["is_champion"] = (mtype, target) in champion_keys
    pred_rows.append(d.dropna(subset=["y_true"]))
preds = pd.concat(pred_rows, ignore_index=True)

# Métricas por ProductType (test)
by_type = []
for (mtype, target, pt), g in preds[preds["split"] == "test"].groupby(["model_type", "target", CAT_COL]):
    if len(g) >= 3:
        by_type.append({"model_type": mtype, "target": target, CAT_COL: pt, "n_test": len(g),
                        "r2": r2_score(g["y_true"], g["y_pred"]),
                        "mae": g["abs_error"].mean(), "mape_pct": g["ape_pct"].mean()})

# Explicaciones locales: contribución de cada variable cruda en cada registro (modelo campeón por target)
contrib = []
for target, mtype in champions.items():
    c = occlusion_contributions(fitted[(mtype, target)], prep, raw_all)
    contrib.append(c.assign(target=target, model_type=mtype))
contrib = pd.concat(contrib, ignore_index=True)

write_table(res, T_METRICS)
write_table(preds, T_PREDS)
write_table(imp_all, T_IMPORTANCE)
write_table(contrib, T_CONTRIB)
write_table(pd.DataFrame(by_type), T_METRICS + "_by_producttype")

# COMMAND ----------

# MAGIC %md ## Modelo multi-entrada / multi-salida
# MAGIC Un único modelo registrado que recibe **variables de calidad crudas** (`ProductType` + 12 numéricas) y devuelve
# MAGIC `Strength1_pred`, `Strength7_pred` y `Strength28_pred` usando el campeón de cada horizonte.

# COMMAND ----------

class CementoMultiOutput(mlflow.pyfunc.PythonModel):
    def __init__(self, models, prep, champions):
        self.models, self.prep, self.champions = models, prep, champions

    def predict(self, context, model_input, params=None):
        raw = model_input.reindex(columns=[CAT_COL] + NUM_COLS).copy()
        raw[CAT_COL] = raw[CAT_COL].map(normalize_product_type)
        X = self.prep.transform(raw)
        return pd.DataFrame({f"{t}_pred": m.predict(X) for t, m in self.models.items()}, index=model_input.index)


multi = CementoMultiOutput({t: fitted[(m, t)] for t, m in champions.items()}, prep, champions)
example = silver.loc[gold.loc[gold["split"] == "test", "row_id"]].reset_index()[[CAT_COL] + NUM_COLS].head(5)
sig = infer_signature(example, multi.predict(None, example))

with mlflow.start_run(run_name="modelo_multi_salida"):
    mlflow.log_dict(champions, "champions.json")
    try:
        info = mlflow.pyfunc.log_model(name="model", python_model=multi, signature=sig, input_example=example,
                                       registered_model_name=MODEL_MULTI)
    except TypeError:
        info = mlflow.pyfunc.log_model(artifact_path="model", python_model=multi, signature=sig, input_example=example,
                                       registered_model_name=MODEL_MULTI)
    version = str(getattr(info, "registered_model_version", "") or "")
    if version:
        client.set_registered_model_alias(MODEL_MULTI, "champion", version)

print("Campeones:", champions)
print(multi.predict(None, example).round(2))
