# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Análisis exploratorio (EDA)
# MAGIC Estadísticas, nulos, distribuciones, outliers (IQR), correlaciones con las 3 resistencias, análisis por `ProductType`,
# MAGIC scatter de las variables más correlacionadas y VIF. Las figuras se guardan también en MLflow (run `EDA`).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import matplotlib
import matplotlib.pyplot as plt
import mlflow

setup_mlflow()
silver = read_table(T_SILVER)
bronze = read_table(T_BRONZE)
bronze_num = bronze[[CAT_COL] + NUM_COLS + TARGETS].copy()
for c in NUM_COLS + TARGETS:
    bronze_num[c] = to_numeric_series(bronze_num[c])

run = mlflow.start_run(run_name="EDA")

def save(fig, name):
    mlflow.log_figure(fig, f"eda/{name}.png")
    plt.show()
    plt.close(fig)

# COMMAND ----------

# MAGIC %md ## 1. Estadísticas descriptivas y nulos (datos crudos de Bronze)

# COMMAND ----------

desc = bronze_num[NUM_COLS + TARGETS].describe().T
desc["nulls"] = bronze_num[NUM_COLS + TARGETS].isna().sum()
desc["nulls_pct"] = (100 * desc["nulls"] / len(bronze_num)).round(2)
show(desc.round(3).reset_index())

fig, ax = plt.subplots(figsize=(9, 4))
desc["nulls_pct"].sort_values().plot.barh(ax=ax, color="#4C72B0")
ax.set_xlabel("% de nulos"); ax.set_title("Valores nulos por columna (Bronze)")
save(fig, "nulos_por_columna")

print("Registros por ProductType:")
print(bronze_num[CAT_COL].map(normalize_product_type).value_counts())

# COMMAND ----------

# MAGIC %md ## 2. Distribuciones (histogramas y boxplots) con datos Silver

# COMMAND ----------

cols = NUM_COLS + TARGETS
n_cols = 4
n_rows = int(np.ceil(len(cols) / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
for ax, c in zip(axes.ravel(), cols):
    ax.hist(silver[c].dropna(), bins=30, color="#4C72B0", edgecolor="white")
    ax.set_title(c, fontsize=9)
for ax in axes.ravel()[len(cols):]:
    ax.axis("off")
fig.suptitle("Histogramas", y=1.01)
fig.tight_layout()
save(fig, "histogramas")

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
for ax, c in zip(axes.ravel(), cols):
    ax.boxplot(silver[c].dropna(), vert=False)
    ax.set_title(c, fontsize=9)
for ax in axes.ravel()[len(cols):]:
    ax.axis("off")
fig.suptitle("Boxplots", y=1.01)
fig.tight_layout()
save(fig, "boxplots")

# COMMAND ----------

# MAGIC %md ## 3. Outliers (IQR 1.5×) en datos crudos

# COMMAND ----------

def iqr_outliers(s):
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

out_tbl = pd.DataFrame({"column": cols, "outliers_iqr": [iqr_outliers(bronze_num[c]) for c in cols]})
out_tbl["pct"] = (100 * out_tbl["outliers_iqr"] / len(bronze_num)).round(2)
show(out_tbl.sort_values("outliers_iqr", ascending=False))

# COMMAND ----------

# MAGIC %md ## 4. Correlaciones predictores ↔ resistencias

# COMMAND ----------

pear = silver[NUM_COLS + TARGETS].corr(method="pearson")
spear = silver[NUM_COLS + TARGETS].corr(method="spearman")

fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(pear.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(pear))); ax.set_xticklabels(pear.columns, rotation=90, fontsize=8)
ax.set_yticks(range(len(pear))); ax.set_yticklabels(pear.index, fontsize=8)
for i in range(len(pear)):
    for j in range(len(pear)):
        ax.text(j, i, f"{pear.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Matriz de correlación (Pearson)")
save(fig, "matriz_correlacion")

corr_long = pd.DataFrame([
    {"feature": f, "target": t, "pearson": pear.loc[f, t], "spearman": spear.loc[f, t]}
    for f in NUM_COLS for t in TARGETS
])
write_table(corr_long, T_CORR)
show(corr_long.pivot(index="feature", columns="target", values="pearson").round(3).reset_index())

# COMMAND ----------

# MAGIC %md ## 5. Análisis por ProductType

# COMMAND ----------

by_type = silver.groupby(CAT_COL)[TARGETS].agg(["count", "mean", "std"]).round(2)
by_type.columns = [f"{a}_{b}" for a, b in by_type.columns]
show(by_type.reset_index())

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
types = sorted(silver[CAT_COL].unique())
for ax, t in zip(axes, TARGETS):
    ax.boxplot([silver.loc[silver[CAT_COL] == p, t].dropna() for p in types])
    ax.set_xticklabels(types)
    ax.set_title(f"{t} por ProductType")
fig.tight_layout()
save(fig, "resistencias_por_producto")

# ¿Cuáles predictores difieren más entre productos? (razón varianza entre grupos / total)
between = {}
for c in NUM_COLS:
    g = silver.groupby(CAT_COL)[c]
    between[c] = float(((g.mean() - silver[c].mean()) ** 2 * g.count()).sum() / ((silver[c] - silver[c].mean()) ** 2).sum())
show(pd.Series(between, name="eta2_entre_productos").sort_values(ascending=False).round(3).reset_index())

# COMMAND ----------

# MAGIC %md ## 6. Scatter de las variables más correlacionadas con cada resistencia

# COMMAND ----------

for t in TARGETS:
    top = pear.loc[NUM_COLS, t].abs().sort_values(ascending=False).head(4).index
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, f in zip(axes, top):
        for p in types:
            m = silver[CAT_COL] == p
            ax.scatter(silver.loc[m, f], silver.loc[m, t], s=10, alpha=0.6, label=p)
        ax.set_xlabel(f); ax.set_ylabel(t)
        ax.set_title(f"r = {pear.loc[f, t]:.2f}", fontsize=10)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Variables más correlacionadas con {t}")
    fig.tight_layout()
    save(fig, f"scatter_{t}")

# COMMAND ----------

# MAGIC %md ## 7. Multicolinealidad (VIF)

# COMMAND ----------

vif = read_table(T_VIF)
show(vif)
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#C44E52" if v > 10 else "#4C72B0" for v in vif["VIF"][::-1]]
ax.barh(vif["feature"][::-1], vif["VIF"][::-1].clip(upper=100), color=colors)
ax.axvline(10, ls="--", color="gray")
ax.set_title("VIF por predictor (rojo = mayor a 10, recortado a 100)")
save(fig, "vif")

mlflow.end_run()
print("✅ EDA terminado. Figuras en el experimento de MLflow, run 'EDA'.")
