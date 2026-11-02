# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Pipeline medallón (Bronze → Silver → Gold)
# MAGIC **Entrada:** `/Volumes/datathon/gcc/raw/datathon_data_gcc.csv` · **Salida:** 3 tablas Delta gobernadas por Unity Catalog.
# MAGIC
# MAGIC | Capa | Qué hace | Decisión documentada |
# MAGIC |---|---|---|
# MAGIC | Bronze | CSV tal cual (todo texto) + metadatos de ingesta | No se transforma nada para conservar trazabilidad |
# MAGIC | Silver | Tipos, `ProductType`, duplicados, nulos, outliers (IQR), VIF | Outliers **se acotan** (winsorizing 3×IQR), no se eliminan: son solo 791 filas |
# MAGIC | Gold | Split 80/20 estratificado, imputación (solo con train), features, one-hot | Escalado dentro del `Pipeline` del modelo lineal, no en la tabla |

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

setup_uc()

# COMMAND ----------

# MAGIC %md ## Bronze · ingesta cruda

# COMMAND ----------

import os
from datetime import datetime

assert os.path.exists(CSV_PATH), (
    f"No encuentro {CSV_PATH}. Sube el CSV: Catalog > {CATALOG} > {RAW_SCHEMA} > {VOLUME} > Upload to this volume."
)

raw = pd.read_csv(CSV_PATH, dtype=str)  # todo como texto: 'tal cual'
raw.columns = [re.sub(r"\W+", "_", c.strip()) for c in raw.columns]  # Delta no admite espacios/símbolos en nombres

expected = [CAT_COL] + NUM_COLS + TARGETS
missing_cols = [c for c in expected if c not in raw.columns]
assert not missing_cols, f"Faltan columnas esperadas en el CSV: {missing_cols}. Ajusta las listas de 00_config."

raw["_ingest_ts"] = datetime.utcnow().isoformat(timespec="seconds")
raw["_source_file"] = CSV_PATH
write_table(raw, T_BRONZE)

# COMMAND ----------

# MAGIC %md ## Silver · limpieza y validación

# COMMAND ----------

bronze = read_table(T_BRONZE)
df = bronze[[c for c in bronze.columns if not c.startswith("_")]].copy()
summary = {"filas_bronze": len(df)}

# 1) ProductType normalizado y categórico
df[CAT_COL] = df[CAT_COL].map(normalize_product_type)

# 2) Tipos: todo lo numérico a float; lo no convertible pasa a NaN (y se cuenta)
coerced = {}
for c in NUM_COLS + TARGETS:
    before = df[c].astype("string").str.strip().replace("", pd.NA).notna().sum()
    df[c] = to_numeric_series(df[c])
    coerced[c] = int(before - df[c].notna().sum())
summary["valores_no_numericos_a_nulo"] = sum(coerced.values())

# 3) Resistencias <= 0 no son físicas -> nulo (se conserva la fila para los otros targets)
invalid_strength = {t: int((df[t] <= 0).sum()) for t in TARGETS}
for t in TARGETS:
    df.loc[df[t] <= 0, t] = np.nan
summary["resistencias_invalidas_a_nulo"] = sum(invalid_strength.values())

# 4) Duplicados exactos
n0 = len(df)
df = df.drop_duplicates(subset=[CAT_COL] + NUM_COLS + TARGETS).reset_index(drop=True)
summary["duplicados_eliminados"] = n0 - len(df)

# 5) Filas sin ninguna resistencia no sirven para entrenar ningún target
n0 = len(df)
df = df.dropna(subset=TARGETS, how="all").reset_index(drop=True)
summary["filas_sin_ningun_target_eliminadas"] = n0 - len(df)

# 6) Outliers por IQR dentro de cada ProductType: se marcan (1.5×IQR) y se acotan (3×IQR) solo en predictores.
#    Los targets NO se modifican; solo se reportan.
def iqr_bounds(s, k):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

flags = pd.DataFrame(0, index=df.index, columns=NUM_COLS)
capped = {c: 0 for c in NUM_COLS}
for c in NUM_COLS:
    for _, idx in df.groupby(CAT_COL).groups.items():
        s = df.loc[idx, c]
        if s.notna().sum() < 8 or s.nunique() < 3:
            continue
        lo, hi = iqr_bounds(s, 1.5)
        flags.loc[idx, c] = ((s < lo) | (s > hi)).astype(int)
        lo3, hi3 = iqr_bounds(s, 3.0)
        capped[c] += int(((s < lo3) | (s > hi3)).sum())
        df.loc[idx, c] = s.clip(lo3, hi3)
df["n_outlier_flags"] = flags.sum(axis=1)
target_outliers = {}
for t in TARGETS:
    flag_t = 0
    for _, idx in df.groupby(CAT_COL).groups.items():
        s = df.loc[idx, t].dropna()
        if len(s) >= 8:
            lo, hi = iqr_bounds(s, 1.5)
            flag_t += int(((s < lo) | (s > hi)).sum())
    target_outliers[t] = flag_t

# 7) Identificador estable de fila
df.insert(0, "row_id", np.arange(1, len(df) + 1, dtype="int64"))
summary["filas_silver"] = len(df)

silver = df[["row_id", CAT_COL] + NUM_COLS + TARGETS + ["n_outlier_flags"]]

# Reporte de calidad por columna (documenta las decisiones)
quality = pd.DataFrame({
    "column": [CAT_COL] + NUM_COLS + TARGETS,
    "nulls": [int(silver[c].isna().sum()) for c in [CAT_COL] + NUM_COLS + TARGETS],
})
quality["nulls_pct"] = (100 * quality["nulls"] / len(silver)).round(2)
quality["non_numeric_coerced"] = quality["column"].map(coerced).fillna(0).astype(int)
quality["outlier_flags_iqr1_5"] = quality["column"].map({**flags.sum().to_dict(), **target_outliers}).fillna(0).astype(int)
quality["values_capped_iqr3"] = quality["column"].map(capped).fillna(0).astype(int)
quality["invalid_set_to_null"] = quality["column"].map(invalid_strength).fillna(0).astype(int)

write_table(silver, T_SILVER)
write_table(quality, T_QUALITY)
write_table(pd.DataFrame({"metric": list(summary), "value": list(summary.values())}), T_QUALITY_SUMMARY)
show(quality)

# COMMAND ----------

# MAGIC %md ### Multicolinealidad (VIF)
# MAGIC VIF > 10 indica redundancia fuerte. **Decisión:** no se eliminan variables; los árboles son robustos a la colinealidad y
# MAGIC el modelo lineal usa regularización (ElasticNet). Aun así se documenta para el análisis de negocio.

# COMMAND ----------

X_vif = silver[NUM_COLS].fillna(silver[NUM_COLS].median())
vif = compute_vif(X_vif)
vif["alto_vif_mayor_10"] = vif["VIF"] > 10
write_table(vif, T_VIF)
show(vif)

# COMMAND ----------

# MAGIC %md ## Gold · features y targets listos para ML

# COMMAND ----------

from sklearn.model_selection import train_test_split

silver = read_table(T_SILVER)

try:
    tr_idx, te_idx = train_test_split(silver.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=silver[CAT_COL])
except ValueError:  # algún ProductType con una sola fila
    tr_idx, te_idx = train_test_split(silver.index, test_size=TEST_SIZE, random_state=RANDOM_STATE)

silver["split"] = "train"
silver.loc[te_idx, "split"] = "test"

# La imputación y las medianas se ajustan SOLO con train (sin fuga de información hacia test)
prep = Preprocessor().fit(silver.loc[tr_idx])
X = prep.transform(silver)

gold = pd.concat([silver[["row_id", "split", CAT_COL]], X, silver[TARGETS]], axis=1)
assert not set(TARGETS) & set(X.columns), "Un target se coló como feature"
write_table(gold, T_GOLD)

print(f"Features: {X.shape[1]} | train: {len(tr_idx)} | test: {len(te_idx)}")
print(gold.groupby(["split", CAT_COL]).size().unstack(0))
