# Datathon MICAI 2026 · Reto 2 GCC · Predicción de resistencias del cemento

Solución de punta a punta en Databricks: **calidad del cemento → Strength1, Strength7, Strength28**, con pipeline medallón en Unity Catalog, EDA, 3 algoritmos × 3 targets en MLflow, modelo multi-salida registrado y dashboard AI/BI.

> **Estado de la prueba:** el código se probó localmente con un CSV sintético con el mismo esquema que describe el README del reto (nulos, duplicados, outliers, `ProductType` sucio). **No se ha ejecutado en tu workspace ni con los datos reales de GCC**, así que las métricas reales las verás tú al correrlo. Si algo falla en Free Edition (por ejemplo, permisos para crear el catálogo), el mensaje de error indica qué cambiar en `00_config`.

## Archivos

| Archivo | Qué hace |
|---|---|
| `00_config.py` | Nombres de tablas, columnas, helpers y la clase `Preprocessor` (misma lógica de features en Gold, entrenamiento y modelo registrado) |
| `01_pipeline_medallon.py` | Bronze → Silver → Gold, reporte de calidad y VIF |
| `02_eda.py` | EDA completo; figuras guardadas en MLflow (run `EDA`) |
| `03_entrenamiento_mlflow.py` | ElasticNet, RandomForest y HistGradientBoosting × 3 targets, registro en Unity Catalog, tablas del dashboard, modelo multi-salida |
| `04_simulador.py` | Widgets para ingresar variables de calidad y ver resistencias + variables determinantes |
| `05_dashboard_datasets.sql` | 11 datasets listos para el AI/BI Dashboard |
| `99_run_all.py` | Ejecuta 01 → 02 → 03 sin intervención manual |

## Cómo correrlo (Databricks Free Edition, cómputo serverless)

1. **Datos:** Catalog → crea el catálogo `datathon` (o usa el tuyo y cámbialo en `CATALOG` de `00_config`), esquema `gcc`, volumen `raw`. Sube `datathon_data_gcc.csv` al volumen.
2. **Importa los notebooks** (Workspace → Import, los `.py` se importan como notebooks) en una misma carpeta.
3. Ejecuta **`99_run_all`** (o cada notebook en orden 01 → 02 → 03). El primer notebook crea esquemas `bronze`, `silver`, `gold`.
4. Abre `04_simulador` para probar entradas.
5. Crea el dashboard (sección siguiente).

**Cuidar el cómputo diario:** en `00_config`, `N_ITER_SEARCH = 20` controla la búsqueda de hiperparámetros (es el paso más caro). Con 10 baja el costo a la mitad con poca pérdida de precisión. Si una cuenta llega al límite, otro integrante puede importar los notebooks y re-ejecutar: todo se recrea desde el CSV.

## Decisiones técnicas (para documentar y defender en el Q&A)

| Tema | Decisión | Por qué |
|---|---|---|
| Bronze | Todo como texto + `_ingest_ts`, `_source_file` | Trazabilidad: se puede reconstruir Silver desde cero |
| Tipos | Conversión numérica tolerante (coma decimal, espacios); lo no numérico pasa a nulo y se cuenta | Nada se pierde en silencio, queda en `datathon_quality_report` |
| `ProductType` | Normalizado a `Type N` | Evita categorías duplicadas por formato |
| Duplicados | Se eliminan duplicados exactos (predictores + targets) | |
| Filas sin ninguna resistencia | Se eliminan; con alguna resistencia se conservan | Cada target se entrena con sus filas disponibles |
| Outliers | Marcados con 1.5×IQR y **acotados** a 3×IQR por `ProductType` (solo predictores); targets no se tocan | Con 791 filas, eliminar pierde información; los extremos reales del proceso se conservan acotados |
| Nulos | Mediana por `ProductType`, calculada **solo con train**; se agrega `n_missing` como feature | Evita fuga de información hacia test |
| Multicolinealidad | VIF calculado y guardado; **no se eliminan variables** | Los árboles son robustos y ElasticNet regulariza; la decisión queda documentada |
| Features | Suma de fases, participación de cada fase, razones Chem A/B y Phys A/B, one-hot de `ProductType` | Genérico porque los nombres están anonimizados (ver mejora 1) |
| Split | 80/20 estratificado por `ProductType`, `random_state=42` | Reproducible |
| Validación | KFold 5 solo sobre train; el campeón se elige por **MAE de CV**, no por test | Test se usa una sola vez, sin sesgo de selección |
| Escalado | Dentro del `Pipeline` del modelo lineal | Los árboles no lo necesitan |
| Importancias | Permutación sobre variables **crudas** (no features derivadas) | Interpretable para negocio |
| Explicación por registro | Contribución = predicción real − predicción con la variable fijada en su mediana | Responde "¿qué variables fueron determinantes?" para cualquier fila |

## Dashboard AI/BI

Crea un dashboard nuevo, agrega los datasets de `05_dashboard_datasets.sql` (uno por bloque) y construye estas páginas:

| Página | Widgets | Dataset |
|---|---|---|
| **Resumen** | Contadores de registros y medias; histogramas de las 3 resistencias | `ds_resumen_kpis`, `ds_distribucion` |
| **Modelos** | Tabla/barras de R², MAE y MAPE (3 modelos × 3 targets); métricas por producto | `ds_comparacion_modelos`, `ds_metricas_por_producto` |
| **Variables clave** | Barras de importancia del campeón por resistencia; heatmap de correlaciones | `ds_importancia`, `ds_correlaciones` |
| **Calidad vs resistencia** | Scatter con filtro `feature`, color por `ProductType` | `ds_scatter_calidad` |
| **Predicho vs real** | Scatter y/o MAE, MAPE y R² (ver expresiones en el SQL) | `ds_pred_vs_real` |
| **Explicador** | Parámetro `row_id`: resistencia predicha vs real y barras de contribución por variable | `ds_prediccion_registro`, `ds_explicar_registro` |
| **Alertas** | Semáforo 🔴🟡🟢 por lote y contador de riesgos; alerta SQL opcional | `ds_alertas` |

- **Filtro global por `ProductType`:** agrega un widget de filtro con ese campo y vincúlalo a todos los datasets que lo tienen.
- **Simulación "qué pasa si":** el dashboard no ejecuta modelos; para eso usa `04_simulador` (widgets) y muéstralo en vivo en la demo.
- **Atajo con Genie Code**, prompt sugerido: *"Crea un AI/BI dashboard con los datasets que están en 05_dashboard_datasets.sql, con páginas Resumen, Modelos, Variables clave, Calidad vs resistencia, Predicho vs real, Explicador y Alertas, con un filtro global por ProductType."* Revisa siempre el resultado.
- El umbral de alerta de ejemplo es el percentil 10 histórico. **Cámbialo por la especificación real** de GCC si la tienen.

## Cómo mejorar la precisión (en orden de impacto esperado)

1. **Features con sentido físico.** Si GCC te dice qué es cada variable (por ejemplo, fases C3S, C2S, C3A, C4AF; Blaine; SO3), agrega módulos clásicos (LSF, SM, AM) y razones entre fases en `Preprocessor._build`. Suele mover más el error que cambiar de algoritmo.
2. **Ablación de features:** compara base vs. base + derivadas con la misma validación y quédate con lo que mejore el MAE de CV.
3. **Más búsqueda** (`N_ITER_SEARCH` 40 o 60) si tienes cómputo; probar LightGBM/XGBoost como cuarto candidato.
4. **Ensamble** de los 3 modelos (promedio ponderado por 1/MAE de CV).
5. **Reentrenar con todos los datos** (train + test) al final para la versión de producción, reportando antes las métricas honestas del test.
6. **Modelos por `ProductType`** si las métricas por producto del dashboard muestran grandes diferencias.

## Criterio 1 de la rúbrica: valor de negocio (15 %)

Ata la precisión del modelo a dinero con un modelo simple, con supuestos que debes **sustituir por datos de GCC**:

- **Margen de seguridad:** hoy se produce con un sobrediseño `m` (MPa por encima de la especificación) por incertidumbre. Si el modelo reduce el error de predicción de `E0` a `E1`, el margen puede bajar en aproximadamente `(E0 − E1)`.
- **Ahorro por tonelada** = `reducción de margen (MPa) × ahorro por MPa (% de clinker o aditivo) × costo del insumo ($/t) × toneladas/año`.
- **Menos rechazos:** `lotes en riesgo detectados a tiempo × costo por lote fuera de especificación`.
- **Tiempo de reacción:** Strength28 tarda 28 días en medirse; predecir desde el día 1 permite corregir semanas antes.

Pregunta a los organizadores: costo del clinker o aditivo por tonelada, producción anual, tamaño típico de lote y el **benchmark de precisión** de R1, R7 y R28 (la rúbrica lo menciona pero no lo da).

## Preparación para el Q&A (todos deben poder responder)

- ¿Por qué split estratificado y por qué el campeón se elige por CV y no por test?
- ¿Por qué se acotan outliers en vez de eliminarlos?
- ¿Cómo evitaron fuga de información? (imputación con train; targets nunca como features; verificación en `03` de que Gold coincide con el `Preprocessor`)
- ¿Qué variables dominan cada horizonte y tiene sentido físico-químico?
- ¿Qué haría el modelo con un `ProductType` nuevo o con valores fuera de rango?
- ¿Dónde está el límite del modelo? (791 filas, un solo split de test; incertidumbre de R² por `ProductType` con pocos registros)
