# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Configuración y utilidades
# MAGIC Datathon MICAI 2026 · Reto 2 (GCC) · Predicción de resistencias del cemento.
# MAGIC
# MAGIC Este notebook **no se ejecuta solo**: los demás lo cargan con `%run ./00_config`.
# MAGIC Contiene nombres de tablas, listas de columnas, helpers de I/O y la clase `Preprocessor`
# MAGIC (la misma lógica de features se usa en Gold, en el entrenamiento y en el modelo registrado).

# COMMAND ----------

import re
import numpy as np
import pandas as pd

# ---------- Unity Catalog ----------
CATALOG = "datathon"            # catálogo (si no puedes crearlo en Free Edition, cambia a tu catálogo por defecto)
RAW_SCHEMA, VOLUME = "gcc", "raw"   # volumen con el CSV: datathon.gcc.raw
CSV_NAME = "datathon_data_gcc.csv"
CSV_PATH = f"/Volumes/{CATALOG}/{RAW_SCHEMA}/{VOLUME}/{CSV_NAME}"

T_BRONZE = f"{CATALOG}.bronze.datathon_data_bronze"
T_SILVER = f"{CATALOG}.silver.datathon_data_silver"
T_GOLD = f"{CATALOG}.gold.datathon_data_gold"
T_QUALITY = f"{CATALOG}.silver.datathon_quality_report"
T_QUALITY_SUMMARY = f"{CATALOG}.silver.datathon_quality_summary"
T_VIF = f"{CATALOG}.silver.datathon_vif_report"
T_CORR = f"{CATALOG}.gold.eda_correlations"
T_METRICS = f"{CATALOG}.gold.model_metrics"
T_PREDS = f"{CATALOG}.gold.model_predictions"
T_IMPORTANCE = f"{CATALOG}.gold.feature_importance"
T_CONTRIB = f"{CATALOG}.gold.local_contributions"
MODEL_MULTI = f"{CATALOG}.gold.cemento_resistencias_multi"

# ---------- Reproducibilidad / costo ----------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ITER_SEARCH = 20     # iteraciones de RandomizedSearchCV por modelo y target (baja a 10 si te quedas sin cómputo)
CV_FOLDS = 5

# ---------- Esquema del dataset ----------
CAT_COL = "ProductType"
PHASE_COLS = [f"Mineral_Phase_{c}" for c in "ABCDEFGH"]
PHYS_COLS = ["Physical_Metric_A", "Physical_Metric_B"]
CHEM_COLS = ["Chemical_Composition_A", "Chemical_Composition_B"]
NUM_COLS = PHYS_COLS + PHASE_COLS + CHEM_COLS          # 12 predictores numéricos
TARGETS = ["Strength1", "Strength7", "Strength28"]     # NUNCA se usan como features
META_COLS = ["row_id", "split", CAT_COL]


# COMMAND ----------

# ---------- Helpers de I/O (Spark <-> pandas). Con 791 filas, pandas es más barato que Spark. ----------
def setup_uc():
    """Crea catálogo, esquemas y volumen si no existen (idempotente)."""
    try:
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
    except Exception as e:
        print(f"⚠️ No se pudo crear el catálogo '{CATALOG}': {e}\n   Cambia CATALOG en 00_config al catálogo que sí tengas.")
    spark.sql(f"USE CATALOG {CATALOG}")
    for s in [RAW_SCHEMA, "bronze", "silver", "gold"]:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{s}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{RAW_SCHEMA}.{VOLUME}")


def read_table(fq_name):
    return spark.table(fq_name).toPandas()


def write_table(pdf, fq_name):
    """Escribe un DataFrame de pandas como tabla Delta (NaN -> NULL, esquema explícito)."""
    from pyspark.sql import types as T
    fields = []
    for c, dt in pdf.dtypes.items():
        if pd.api.types.is_bool_dtype(dt):
            t = T.BooleanType()
        elif pd.api.types.is_integer_dtype(dt):
            t = T.LongType()
        elif pd.api.types.is_float_dtype(dt):
            t = T.DoubleType()
        elif pd.api.types.is_datetime64_any_dtype(dt):
            t = T.TimestampType()
        else:
            t = T.StringType()
        fields.append(T.StructField(str(c), t, True))

    def clean(v):
        if v is None or v is pd.NA or v is pd.NaT:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        return v.item() if isinstance(v, np.generic) else v

    rows = [tuple(clean(v) for v in r) for r in pdf.itertuples(index=False, name=None)]
    sdf = spark.createDataFrame(rows, schema=T.StructType(fields))
    sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(fq_name)
    print(f"✅ {fq_name}: {len(pdf):,} filas x {pdf.shape[1]} columnas")


def setup_mlflow(name="datathon_gcc_resistencias"):
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    user = spark.sql("select current_user()").first()[0]
    mlflow.set_experiment(f"/Users/{user}/{name}")


def show(x):
    try:
        display(x)  # noqa: F821  (existe en Databricks)
    except NameError:
        print(x)


# COMMAND ----------

# ---------- Limpieza de tipos ----------
def to_numeric_series(s):
    """Convierte texto a número tolerando espacios y coma decimal ('12,5'). Lo no numérico -> NaN."""
    s = s.astype("string").str.strip()
    is_dec_comma = s.str.fullmatch(r"-?\d+,\d+").fillna(False).astype(bool)
    s = s.where(~is_dec_comma, s.str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce").astype("float64")


def normalize_product_type(v):
    """'type1', 'TYPE 1', ' Type  1 ' -> 'Type 1'. Nulos -> 'Unknown'."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() == "":
        return "Unknown"
    m = re.search(r"(\d+)", str(v))
    return f"Type {int(m.group(1))}" if m else str(v).strip().title()


def compute_vif(X):
    """VIF = diagonal de la inversa de la matriz de correlación (sin depender de statsmodels)."""
    X = X.loc[:, X.std(ddof=0) > 0]
    Z = (X - X.mean()) / X.std(ddof=0)
    R = np.corrcoef(Z.values, rowvar=False)
    vif = np.diag(np.linalg.pinv(R))
    return pd.DataFrame({"feature": X.columns, "VIF": vif}).sort_values("VIF", ascending=False).reset_index(drop=True)


# COMMAND ----------

# ---------- Preprocesamiento + Feature Engineering (se ajusta SOLO con train) ----------
class Preprocessor:
    """
    raw (ProductType + 12 variables de calidad) -> matriz de features para los modelos.
      1. Imputa nulos con la mediana por ProductType (fallback: mediana global), todo calculado en train.
      2. Features derivadas: suma de fases, participación de cada fase, razones Chem A/B y Phys A/B, nº de nulos.
      3. One-hot de ProductType (nombres sin espacios para Delta).
    Se guarda dentro del modelo pyfunc, así el modelo registrado acepta datos crudos de calidad.
    """

    def fit(self, df):
        self.medians_by_type_ = df.groupby(CAT_COL)[NUM_COLS].median()
        self.global_median_ = df[NUM_COLS].median()
        self.categories_ = sorted(df[CAT_COL].dropna().unique().tolist())
        feats = self._build(df)
        self.fe_median_ = feats.median()
        self.feature_names_ = list(feats.columns)
        return self

    def _impute(self, df):
        out = df[NUM_COLS].astype("float64").copy()
        for c in NUM_COLS:
            by_type = df[CAT_COL].map(self.medians_by_type_[c])
            out[c] = out[c].fillna(by_type).fillna(self.global_median_[c])
        return out

    def _build(self, df):
        n_missing = df[NUM_COLS].isna().sum(axis=1).astype("float64")
        X = self._impute(df)
        X["n_missing"] = n_missing
        phase_sum = X[PHASE_COLS].sum(axis=1)
        X["Mineral_Sum"] = phase_sum
        denom = phase_sum.where(phase_sum.abs() > 1e-9)
        for c in PHASE_COLS:
            X[f"Share_{c}"] = X[c] / denom
        for name, (a, b) in {"Chem_Ratio_A_B": CHEM_COLS, "Phys_Ratio_A_B": PHYS_COLS}.items():
            d = X[b].where(X[b].abs() > 1e-9)
            X[name] = X[a] / d
        for cat in getattr(self, "categories_", sorted(df[CAT_COL].dropna().unique())):
            X["PT_" + re.sub(r"\W+", "_", str(cat))] = (df[CAT_COL] == cat).astype("float64")
        return X.replace([np.inf, -np.inf], np.nan)

    def transform(self, df):
        X = self._build(df).fillna(self.fe_median_)
        return X.reindex(columns=self.feature_names_).astype("float64")


# COMMAND ----------

# ---------- Interpretabilidad a nivel de variable de calidad (no de feature derivada) ----------
def raw_permutation_importance(model, prep, raw_df, y, n_repeats=10, seed=RANDOM_STATE):
    """Importancia por permutación de cada variable CRUDA: cuánto sube el MAE al barajarla."""
    from sklearn.metrics import mean_absolute_error
    rng = np.random.RandomState(seed)
    base = mean_absolute_error(y, model.predict(prep.transform(raw_df)))
    rows = []
    for col in [CAT_COL] + NUM_COLS:
        deltas = []
        for _ in range(n_repeats):
            r = raw_df.copy()
            r[col] = rng.permutation(r[col].values)
            deltas.append(mean_absolute_error(y, model.predict(prep.transform(r))) - base)
        rows.append((col, float(np.mean(deltas)), float(np.std(deltas))))
    out = pd.DataFrame(rows, columns=["feature", "importance_mae", "importance_std"])
    pos = out["importance_mae"].clip(lower=0)
    out["importance_pct"] = 100 * pos / pos.sum() if pos.sum() > 0 else 0.0
    return out.sort_values("importance_mae", ascending=False).reset_index(drop=True)


def occlusion_contributions(model, prep, raw_df):
    """
    Explicación local: para cada registro y variable, contribución = pred(real) - pred(variable fijada a la mediana de train).
    > 0 significa que esa variable empuja la resistencia por ENCIMA de lo típico; < 0, por debajo.
    """
    raw_df = raw_df.reset_index(drop=True)
    base = model.predict(prep.transform(raw_df))
    parts = []
    for col in NUM_COLS:
        r = raw_df.copy()
        r[col] = prep.global_median_[col]
        parts.append(pd.DataFrame({
            "row_id": raw_df["row_id"].values if "row_id" in raw_df else np.arange(len(raw_df)),
            "feature": col,
            "value": raw_df[col].values,
            "contribution": base - model.predict(prep.transform(r)),
        }))
    return pd.concat(parts, ignore_index=True)
