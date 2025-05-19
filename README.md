# ISD-Algorithm: Desagregación Iterativa con Desplazamiento

## Descripción

Implementación en Python del Algoritmo de Desagregación Iterativa con Desplazamiento (ISD), diseñado para desagregar múltiples series temporales de baja frecuencia con muestreo irregular y posiblemente superpuestas, a una única serie temporal de alta frecuencia (diaria).

## Características

* **Desagregación multi-fuente**: Procesa múltiples series temporales de baja frecuencia con intervalos irregulares.

* **Consistencia temporal**: Mantiene la consistencia entre niveles de agregación, garantizando que la suma de valores desagregados coincida con las observaciones originales.

* **Modelado flexible**: Utiliza variables exógenas correlacionadas para mejorar la precisión de la desagregación.

* **API científica**: Interfaz de programación compatible con el ecosistema científico de Python (NumPy, Pandas, SciPy).

* **Documentación completa**: Implementación bien documentada basada en el artículo académico original.

## Instalación

    pip install isd-algorithm

## Uso rápido

```python
import pandas as pd
import numpy as np
from isd.core.models import LowFrequencySeries, ISDAlgorithm

# 1. Preparar series de baja frecuencia
series_a = LowFrequencySeries(
    name="Serie_A",
    observations=[
        (150, pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-31')),
        (180, pd.Timestamp('2023-02-01'), pd.Timestamp('2023-02-28')),
    ]
)

# 2. Preparar variables exógenas diarias
dates = pd.date_range(start='2023-01-01', end='2023-02-28', freq='D')
temperature = pd.Series(
    np.sin(np.linspace(0, 2*np.pi, len(dates))) * 15 + 40,  # Temperatura sintética
    index=dates
)
exog_vars = pd.DataFrame({
    'temp': temperature,
}, index=dates)

# 3. Crear y ejecutar el algoritmo ISD
isd = ISDAlgorithm(
    lf_series=[series_a],
    exogenous_vars=exog_vars,
    n_lr_models=5,
    n_disagg_cycles=5,
    alpha=0.1
)

# 4. Desagregar y obtener resultados
daily_series = isd.disaggregate(verbose=True)

# 5. Visualizar resultados
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(daily_series.index, daily_series.values, '-o', label='Desagregado (ISD)')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.title('Serie temporal desagregada')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
Ejemplo avanzadoPara un ejemplo más completo que muestra cómo desagregar múltiples series temporales con intervalos superpuestos, consulta el notebook de ejemplo en la documentación.Parámetros recomendadosEl algoritmo ISD tiene tres parámetros principales que controlan su comportamiento:n_lr_models: Número de modelos de regresión lineal entrenadosn_disagg_cycles: Número de ciclos de desagregación por modeloalpha: Ponderación para la redistribución del errorRecomendaciones según el número de series de entrada:3 o más series: n_lr_models=10, n_disagg_cycles=10, alpha=0.05Menos de 3 series: Se requieren más iteraciones; prueba con n_lr_models=15, n_disagg_cycles=15, alpha=0.1RequisitosPython 3.7+NumPyPandasSciPyMatplotlib (opcional, para visualización)Cómo citarSi utilizas esta librería en tu investigación, por favor cita el artículo original:@article{quinn2024iterative,
    title={An Iterative Shifting Disaggregation Algorithm for Multi-Source, Irregularly Sampled, and Overlapped Time Series},
    author={Quinn, Colin O and Brown, Ronald H and Corliss, George F and Povinelli, Richard J},
    journal={Sensors},
    year={2025},
    publisher={MDPI},
    doi={10.3390/s25030895}
}
