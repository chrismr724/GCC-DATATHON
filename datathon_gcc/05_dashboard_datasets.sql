-- Databricks SQL · Datasets para el AI/BI Dashboard "Resistencias del cemento"
-- Cada bloque es un dataset (Dashboard > Data > Create from SQL). Copia el bloque y ponle el nombre indicado.
-- Filtro global: añade un widget de filtro sobre el campo ProductType y vincúlalo a los datasets que lo incluyen.

-- ============================================================================
-- ds_resumen_kpis  ·  Contadores superiores
-- ============================================================================
SELECT
  COUNT(*)                         AS registros,
  ROUND(AVG(Strength1), 2)         AS media_strength1,
  ROUND(AVG(Strength7), 2)         AS media_strength7,
  ROUND(AVG(Strength28), 2)        AS media_strength28,
  ProductType
FROM datathon.silver.datathon_data_silver
GROUP BY ProductType;

-- ============================================================================
-- ds_distribucion  ·  Histograma de las 3 resistencias (20 bins por resistencia)
-- Widget: barras · X = bin_center · Y = SUM(n) · Color = ProductType · Facet/filtro = target
-- ============================================================================
WITH l AS (
  SELECT ProductType, target, value
  FROM (
    SELECT ProductType,
           stack(3, 'Strength1', Strength1, 'Strength7', Strength7, 'Strength28', Strength28) AS (target, value)
    FROM datathon.silver.datathon_data_silver
  )
  WHERE value IS NOT NULL
),
b AS (
  SELECT *, MIN(value) OVER (PARTITION BY target) AS mn, MAX(value) OVER (PARTITION BY target) AS mx
  FROM l
)
SELECT ProductType, target,
       ROUND(mn + (FLOOR(LEAST((value - mn) / NULLIF(mx - mn, 0) * 20, 19.999)) + 0.5) * (mx - mn) / 20, 2) AS bin_center,
       COUNT(*) AS n
FROM b
GROUP BY ProductType, target, 3;

-- ============================================================================
-- ds_comparacion_modelos  ·  3 modelos × 3 targets (métricas globales en test)
-- Widget: tabla o barras agrupadas (X = target, Y = r2 | mae | mape_pct, color = model_type)
-- ============================================================================
SELECT target, model_type, ROUND(r2, 4) AS r2, ROUND(mae, 3) AS mae, ROUND(mape_pct, 3) AS mape_pct,
       ROUND(cv_mae, 3) AS cv_mae, is_champion, n_train, n_test
FROM datathon.gold.model_metrics
ORDER BY target, mape_pct;

-- ============================================================================
-- ds_metricas_por_producto  ·  Métricas del test por ProductType (reacciona al filtro)
-- ============================================================================
SELECT target, model_type, ProductType, n_test, ROUND(r2, 3) AS r2, ROUND(mae, 3) AS mae, ROUND(mape_pct, 3) AS mape_pct
FROM datathon.gold.model_metrics_by_producttype;

-- ============================================================================
-- ds_importancia  ·  Variables de calidad más importantes del modelo campeón de cada resistencia
-- Widget: barras horizontales · X = importance_pct · Y = feature · filtro = target
-- ============================================================================
SELECT target, model_type, feature,
       ROUND(importance_mae, 4) AS aumento_mae,
       ROUND(importance_pct, 2) AS importance_pct
FROM datathon.gold.feature_importance
WHERE is_champion
ORDER BY target, importance_pct DESC;

-- ============================================================================
-- ds_scatter_calidad  ·  Variable de calidad vs resistencia (formato largo: filtro "feature" + eje X = x_value)
-- Widget: scatter · X = x_value · Y = Strength28 (o 1 / 7) · color = ProductType · filtro = feature
-- ============================================================================
SELECT row_id, ProductType, feature, x_value, Strength1, Strength7, Strength28
FROM (
  SELECT row_id, ProductType, Strength1, Strength7, Strength28,
         stack(12,
           'Physical_Metric_A', Physical_Metric_A, 'Physical_Metric_B', Physical_Metric_B,
           'Mineral_Phase_A', Mineral_Phase_A, 'Mineral_Phase_B', Mineral_Phase_B,
           'Mineral_Phase_C', Mineral_Phase_C, 'Mineral_Phase_D', Mineral_Phase_D,
           'Mineral_Phase_E', Mineral_Phase_E, 'Mineral_Phase_F', Mineral_Phase_F,
           'Mineral_Phase_G', Mineral_Phase_G, 'Mineral_Phase_H', Mineral_Phase_H,
           'Chemical_Composition_A', Chemical_Composition_A, 'Chemical_Composition_B', Chemical_Composition_B
         ) AS (feature, x_value)
  FROM datathon.silver.datathon_data_silver
)
WHERE x_value IS NOT NULL;

-- ============================================================================
-- ds_correlaciones  ·  Heatmap variable × resistencia
-- Widget: heatmap · X = target · Y = feature · color = pearson
-- ============================================================================
SELECT feature, target, ROUND(pearson, 3) AS pearson, ROUND(spearman, 3) AS spearman
FROM datathon.gold.eda_correlations;

-- ============================================================================
-- ds_pred_vs_real  ·  Predicho vs real del modelo campeón (solo test = evaluación honesta)
-- Widget: scatter · X = y_true · Y = y_pred · color = ProductType · filtro = target
-- Métricas reactivas al filtro de ProductType (expresiones del widget):
--   MAE  = AVG(abs_error)      MAPE = AVG(ape_pct)      R² = 1 - SUM(sq_error) / SUM(sq_dev)
-- ============================================================================
SELECT row_id, ProductType, target, model_type, y_true, y_pred, error, abs_error, ape_pct, sq_error, sq_dev
FROM datathon.gold.model_predictions
WHERE is_champion AND split = 'test';

-- ============================================================================
-- ds_explicar_registro  ·  "Dado un conjunto de datos de calidad, ¿qué resistencia y qué variables fueron determinantes?"
-- Parámetro: :row_id (número). Widget: barras · X = contribution · Y = feature · filtro = target
-- ============================================================================
SELECT c.target, c.feature, ROUND(c.value, 3) AS valor, ROUND(c.contribution, 3) AS contribution
FROM datathon.gold.local_contributions c
WHERE c.row_id = :row_id;

-- ============================================================================
-- ds_prediccion_registro  ·  Resistencia predicha vs real para ese mismo registro (parámetro :row_id)
-- ============================================================================
SELECT target, model_type, split, ROUND(y_pred, 2) AS predicha, ROUND(y_true, 2) AS real, ROUND(abs_error, 2) AS error_abs
FROM datathon.gold.model_predictions
WHERE is_champion AND row_id = :row_id;

-- ============================================================================
-- ds_alertas  ·  Semáforo de desviaciones
-- Umbral de ejemplo = percentil 10 histórico de cada resistencia. REEMPLAZA por la especificación real
-- (p. ej. 42.5 MPa a 28 días) cuando la tengas: cambia la columna `umbral`.
-- Widget: tabla con la columna `estado` · contador = COUNT_IF(estado LIKE '%Riesgo%')
-- Para alertas automáticas: Create alert sobre este dataset (SQL Alerts) con la condición riesgos > 0.
-- ============================================================================
WITH p AS (
  SELECT row_id, ProductType, target, y_pred, y_true, split,
         PERCENTILE_APPROX(y_true, 0.10) OVER (PARTITION BY target) AS umbral
  FROM datathon.gold.model_predictions
  WHERE is_champion
)
SELECT row_id, ProductType, target, split, ROUND(y_pred, 2) AS predicha, ROUND(umbral, 2) AS umbral,
       ROUND(y_pred - umbral, 2) AS margen,
       CASE WHEN y_pred < umbral THEN '🔴 Riesgo'
            WHEN y_pred < umbral * 1.05 THEN '🟡 Cerca del límite'
            ELSE '🟢 OK' END AS estado
FROM p;
