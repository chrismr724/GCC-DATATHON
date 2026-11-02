# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Simulador: calidad → resistencias
# MAGIC Responde la pregunta del reto: **dado un conjunto de variables de calidad, ¿qué resistencia se obtiene y qué variables
# MAGIC fueron más determinantes?** Cambia los widgets de arriba (o el diccionario de la celda 3) y ejecuta de nuevo.
# MAGIC
# MAGIC Usa el modelo registrado `datathon.gold.cemento_resistencias_multi@champion`.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import mlflow
setup_mlflow()

loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_MULTI}@champion")
inner = loaded.unwrap_python_model()          # CementoMultiOutput (modelos campeones + Preprocessor)
silver = read_table(T_SILVER)

# COMMAND ----------

# MAGIC %md ## Entradas (valores por defecto = mediana histórica)

# COMMAND ----------

defaults = silver[NUM_COLS].median()
types = sorted(silver[CAT_COL].unique())

try:  # widgets de Databricks
    dbutils.widgets.dropdown(CAT_COL, types[0], types)
    for c in NUM_COLS:
        dbutils.widgets.text(c, f"{defaults[c]:.4f}")
    inputs = {CAT_COL: dbutils.widgets.get(CAT_COL), **{c: float(dbutils.widgets.get(c)) for c in NUM_COLS}}
except NameError:  # fuera de Databricks
    inputs = {CAT_COL: types[0], **defaults.to_dict()}

row = pd.DataFrame([inputs])

# COMMAND ----------

# MAGIC %md ## Predicción

# COMMAND ----------

pred = loaded.predict(row)
show(pred.round(2))

# COMMAND ----------

# MAGIC %md ## ¿Qué variables empujaron cada resistencia hacia arriba o hacia abajo?
# MAGIC Contribución = predicción real − predicción con esa variable fijada en su mediana histórica.

# COMMAND ----------

import matplotlib.pyplot as plt

row_n = row.copy()
row_n[CAT_COL] = row_n[CAT_COL].map(normalize_product_type)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, t in zip(axes, TARGETS):
    c = occlusion_contributions(inner.models[t], inner.prep, row_n.assign(row_id=0)).sort_values("contribution")
    ax.barh(c["feature"], c["contribution"], color=np.where(c["contribution"] >= 0, "#55A868", "#C44E52"))
    ax.set_title(f"{t}: {pred[f'{t}_pred'].iloc[0]:.1f}")
    ax.set_xlabel("Contribución a la predicción")
fig.tight_layout()
plt.show()
