# Databricks notebook source
# MAGIC %md
# MAGIC # 99 · Ejecución de punta a punta (sin intervención manual)
# MAGIC Corre en orden: Bronze→Silver→Gold, EDA y entrenamiento/registro. Para evaluación, ejecuta solo este notebook
# MAGIC o crea un **Job** con estas 3 tareas encadenadas (serverless). Requisito previo: el CSV ya está en el volumen
# MAGIC `datathon.gcc.raw` (ver README, paso 1).

# COMMAND ----------

steps = [
    ("./01_pipeline_medallon", 1800),
    ("./02_eda", 1800),
    ("./03_entrenamiento_mlflow", 3600),
]
for path, timeout in steps:
    print(f"▶ {path}")
    dbutils.notebook.run(path, timeout)
    print(f"✔ {path}")

# COMMAND ----------

# MAGIC %md
# MAGIC Siguiente: crea el dashboard con los datasets de `05_dashboard_datasets.sql` y prueba `04_simulador`.
