"""
Implementación mejorada del módulo principal con estándares PEP8 y documentación científica adecuada.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats

# Configuración de logging
logger = logging.getLogger(__name__)


@dataclass
class LowFrequencySeries:
    """
    Clase para representar una serie de tiempo de baja frecuencia.
    
    Parameters
    ----------
    name : str
        Nombre identificativo de la serie.
    observations : List[Tuple[float, pd.Timestamp, pd.Timestamp]]
        Lista de observaciones en forma de tuplas (valor, fecha_inicio, fecha_fin).
        Cada tupla representa una medición agregada sobre un intervalo temporal.
    
    Notes
    -----
    Esta clase proporciona métodos para acceder a los días cubiertos por la serie
    y para obtener la observación correspondiente a un día específico.
    """
    name: str
    observations: List[Tuple[float, pd.Timestamp, pd.Timestamp]]
    
    def get_days_covered(self) -> List[pd.Timestamp]:
        """
        Obtiene todos los días cubiertos por esta serie de baja frecuencia.
        
        Returns
        -------
        List[pd.Timestamp]
            Lista ordenada de timestamps diarios cubiertos por la serie.
        """
        all_days = []
        for _, start_date, end_date in self.observations:
            days = pd.date_range(start=start_date, end=end_date)
            all_days.extend(days)
        return sorted(set(all_days))
    
    def get_observation_for_day(
        self, day: pd.Timestamp
    ) -> Optional[Tuple[float, int]]:
        """
        Obtiene la observación que cubre un día específico.
        
        Parameters
        ----------
        day : pd.Timestamp
            El día para el cual se busca la observación.
            
        Returns
        -------
        Optional[Tuple[float, int]]
            Tupla con (valor_observación, duración_en_días) o None si el día 
            no está cubierto por ninguna observación.
        """
        for value, start_date, end_date in self.observations:
            if start_date <= day <= end_date:
                duration = (end_date - start_date).days + 1
                return (value, duration)
        return None


class ISDAlgorithm:
    """
    Implementación del Algoritmo de Disagregación Iterativa con Desplazamiento (ISD).
    
    El algoritmo ISD desagrega múltiples series temporales de baja frecuencia con 
    intervalos superpuestos a una serie de alta frecuencia (diaria), utilizando 
    variables correlacionadas independientes para mejorar la precisión.
    
    Parameters
    ----------
    lf_series : List[LowFrequencySeries]
        Lista de series de baja frecuencia para desagregar.
    exogenous_vars : pd.DataFrame
        DataFrame con variables exógenas a frecuencia diaria.
        El índice debe ser un DatetimeIndex.
    n_lr_models : int, optional (default=10)
        Número de modelos de regresión lineal a entrenar.
        Valores recomendados: 10 si hay 3+ series, más para menos series.
    n_disagg_cycles : int, optional (default=10)
        Número de ciclos de desagregación por modelo.
        Valores recomendados: 10 si hay 3+ series, más para menos series.
    alpha : float, optional (default=0.05)
        Ponderación que controla la redistribución del error.
        Valores recomendados: 0.05 si hay 3+ series, ajustar para menos series.
    handle_missing : str, optional (default='zero')
        Método para manejar datos faltantes: 'zero' (inicializar con ceros) o
        'estimate' (estimar usando regresión).
    
    Notes
    -----
    El algoritmo opera en un proceso iterativo de dos fases:
    1. Fase de predicción: Entrena modelos de regresión lineal utilizando variables
       exógenas para estimar patrones diarios.
    2. Fase de actualización: Redistribuye las observaciones de baja frecuencia entre
       los períodos de alta frecuencia, manteniendo la coherencia temporal.
    
    References
    ----------
    Quinn, C.O., Brown, R.H., Corliss, G.F., & Povinelli, R.J. (2025). 
    An Iterative Shifting Disaggregation Algorithm for Multi-Source,
    Irregularly Sampled, and Overlapped Time Series. Sensors, 25(3), 895.
    https://doi.org/10.3390/s25030895
    """
    
    def __init__(
        self, 
        lf_series: List[LowFrequencySeries], 
        exogenous_vars: pd.DataFrame,
        n_lr_models: int = 10, 
        n_disagg_cycles: int = 10, 
        alpha: float = 0.05,
        handle_missing: str = 'zero'
    ):
        """Inicializa el algoritmo ISD."""
        self.lf_series = lf_series
        self.exogenous_vars = exogenous_vars
        self.n_lr_models = n_lr_models
        self.n_disagg_cycles = n_disagg_cycles
        self.alpha = alpha
        self.handle_missing = handle_missing
        
        # Validación de parámetros
        self._validate_parameters()
        
        # Determinar el período completo de días para la desagregación
        all_days = []
        for series in self.lf_series:
            all_days.extend(series.get_days_covered())
        self.days = sorted(set(all_days))
        
        # Asegurarse de que las variables exógenas cubren todo el período
        self._validate_exogenous_coverage()
            
        # Diccionario para almacenar las series diarias ingenuas
        self.naive_daily_series = {}
        
        # Resultado final: serie diaria desagregada
        self.y_hat = None
        
        # Métricas de ajuste
        self.regression_metrics = []
    
    def _validate_parameters(self) -> None:
        """
        Valida los parámetros de entrada del algoritmo.
        
        Raises
        ------
        ValueError
            Si algún parámetro no cumple con las restricciones requeridas.
        """
        if not self.lf_series:
            raise ValueError("Se requiere al menos una serie de baja frecuencia.")
        
        if self.n_lr_models <= 0:
            raise ValueError("n_lr_models debe ser un entero positivo.")
            
        if self.n_disagg_cycles <= 0:
            raise ValueError("n_disagg_cycles debe ser un entero positivo.")
            
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError("alpha debe estar en el rango (0, 1).")
        
        if self.handle_missing not in ['zero', 'estimate']:
            raise ValueError(
                "handle_missing debe ser 'zero' o 'estimate', "
                f"no '{self.handle_missing}'"
            )
    
    def _validate_exogenous_coverage(self) -> None:
        """
        Verifica que las variables exógenas cubran todo el período requerido.
        
        Raises
        ------
        ValueError
            Si hay días faltantes en las variables exógenas.
        """
        missing_days = set(self.days) - set(self.exogenous_vars.index)
        if missing_days:
            raise ValueError(
                f"Las variables exógenas no cubren todos los días necesarios. "
                f"Faltan {len(missing_days)} días."
            )
    
    def naive_disaggregate(self) -> Dict[str, pd.Series]:
        """
        Realiza la desagregación ingenua inicial de cada serie de baja frecuencia.
        
        Este método divide cada observación de baja frecuencia uniformemente entre
        los días que abarca, siguiendo la ecuación: 
        a_{i,j} = A_{i,d} / T^A_i, donde j = d - T^A_i + 1, ..., d
        
        Returns
        -------
        Dict[str, pd.Series]
            Diccionario con las series diarias ingenuas para cada serie de entrada.
        
        Notes
        -----
        Para datos faltantes, se puede inicializar con ceros o estimarlos según
        el valor del parámetro handle_missing.
        """
        naive_daily_series = {}
        
        for series in self.lf_series:
            # Inicializar serie diaria con ceros
            daily_values = pd.Series(0.0, index=self.days)
            
            # Desagregar cada observación de baja frecuencia
            for value, start_date, end_date in series.observations:
                days_in_interval = pd.date_range(start=start_date, end=end_date)
                duration = len(days_in_interval)
                
                # Valor diario = valor observación / número de días
                daily_value = value / duration
                
                # Asignar el mismo valor diario a todos los días del intervalo
                for day in days_in_interval:
                    daily_values[day] = daily_value
            
            # Manejar valores faltantes si es necesario
            if self.handle_missing == 'estimate' and daily_values.isna().any():
                self._estimate_missing_values(daily_values, series.name)
            
            naive_daily_series[series.name] = daily_values
        
        self.naive_daily_series = naive_daily_series
        return naive_daily_series
    
    def _estimate_missing_values(
        self, daily_values: pd.Series, series_name: str
    ) -> None:
        """
        Estima valores faltantes en la serie desagregada ingenua.
        
        Parameters
        ----------
        daily_values : pd.Series
            Serie de valores diarios con posibles valores faltantes.
        series_name : str
            Nombre de la serie para registro de log.
        """
        # Identificar días con valores faltantes
        missing_days = daily_values[daily_values.isna()].index
        if len(missing_days) == 0:
            return
            
        logger.info(
            f"Estimando {len(missing_days)} valores faltantes para {series_name}"
        )
        
        # Método simple: usar la media de los días disponibles
        # En una implementación más sofisticada, se podría usar regresión
        mean_value = daily_values.mean()
        daily_values.fillna(mean_value, inplace=True)
    
    def aggregate_series(self) -> pd.Series:
        """
        Agrega todas las series diarias ingenuas para obtener la estimación inicial.
        
        Esta operación sigue la ecuación:
        y_hat[d] = a[d] + b[d] + c[d] + ...
        
        Returns
        -------
        pd.Series
            Serie agregada inicial (y_hat) que representa la estimación diaria.
        """
        if not self.naive_daily_series:
            self.naive_disaggregate()
        
        # Inicializar serie agregada con ceros
        y_hat = pd.Series(0.0, index=self.days)
        
        # Sumar todas las series diarias ingenuas
        for series_name, daily_values in self.naive_daily_series.items():
            y_hat = y_hat.add(daily_values)
        
        self.y_hat = y_hat
        return y_hat
    
    def construct_design_matrix(self) -> np.ndarray:
        """
        Construye la matriz de diseño con las variables exógenas.
        
        Forma una matriz de dimensiones (ND x P+1), donde ND es el número de días
        y P+1 incluye una columna de unos para el intercepto.
        
        Returns
        -------
        np.ndarray
            Matriz de diseño X_matriz para el modelo de regresión.
        """
        # Extraer variables exógenas para los días relevantes
        X_exog = self.exogenous_vars.loc[self.days].values
        
        # Añadir columna de unos para el intercepto
        n_days = len(self.days)
        X_matriz = np.hstack((np.ones((n_days, 1)), X_exog))
        
        return X_matriz
    
    def train_regression_model(
        self, X_matriz: np.ndarray, y_hat: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Entrena un modelo de regresión lineal.
        
        Implementa la ecuación de mínimos cuadrados:
        β̂ = (X^T X)^(-1) X^T y
        
        Parameters
        ----------
        X_matriz : np.ndarray
            Matriz de diseño con variables predictoras.
        y_hat : np.ndarray
            Vector de valores objetivo (serie agregada).
            
        Returns
        -------
        Tuple[np.ndarray, Dict[str, float]]
            Coeficientes beta del modelo y métricas de ajuste.
        """
        # Implementación de mínimos cuadrados
        X_transpose = X_matriz.T
        XTX = X_transpose.dot(X_matriz)
        XTX_inv = np.linalg.inv(XTX)
        beta = XTX_inv.dot(X_transpose).dot(y_hat)
        
        # Calcular métricas de ajuste
        y_pred = X_matriz.dot(beta)
        residuals = y_hat - y_pred
        
        # Métricas de regresión
        n, p = X_matriz.shape
        ssr = np.sum(residuals**2)
        sst = np.sum((y_hat - np.mean(y_hat))**2)
        r_squared = 1 - (ssr / sst)
        adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
        rmse = np.sqrt(ssr / n)
        
        metrics = {
            'r_squared': r_squared,
            'adj_r_squared': adj_r_squared,
            'rmse': rmse,
            'nobs': n
        }
        
        self.regression_metrics.append(metrics)
        return beta, metrics
    
    def compute_predicted_profile(
        self, X_matriz: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """
        Calcula el perfil predicho usando los coeficientes del modelo.
        
        Implementa la ecuación:
        y_tilde = X_matriz @ beta
        
        Parameters
        ----------
        X_matriz : np.ndarray
            Matriz de diseño.
        beta : np.ndarray
            Coeficientes de regresión.
            
        Returns
        -------
        np.ndarray
            Perfil predicho y_tilde.
        """
        # y_tilde = X_matriz @ beta (multiplicación matricial)
        y_tilde = X_matriz.dot(beta)
        
        return y_tilde
    
    def remove_interval_contribution(
        self, 
        y_hat: pd.Series, 
        interval: Tuple[pd.Timestamp, pd.Timestamp], 
        series_name: str
    ) -> pd.Series:
        """
        Elimina la contribución de un intervalo específico del y_hat actual.
        
        Parameters
        ----------
        y_hat : pd.Series
            Serie agregada actual.
        interval : Tuple[pd.Timestamp, pd.Timestamp]
            Tupla (fecha_inicio, fecha_fin) del intervalo.
        series_name : str
            Nombre de la serie a la que pertenece el intervalo.
            
        Returns
        -------
        pd.Series
            Serie y_hat sin la contribución del intervalo.
        """
        start_date, end_date = interval
        days_in_interval = pd.date_range(start=start_date, end=end_date)
        
        # Crear copia de y_hat
        y_hat_minus_interval = y_hat.copy()
        
        # Restar la contribución ingenua del intervalo
        for day in days_in_interval:
            if day in self.naive_daily_series[series_name].index:
                y_hat_minus_interval[day] -= self.naive_daily_series[series_name][day]
        
        return y_hat_minus_interval
    
    def calculate_interval_error(
        self, 
        y_tilde: np.ndarray, 
        y_hat_minus_interval: pd.Series,
        interval: Tuple[pd.Timestamp, pd.Timestamp]
    ) -> pd.Series:
        """
        Calcula el error para un intervalo específico y aplica restricción no negativa.
        
        Implementa la ecuación:
        error_interval_k_days[d] = max(0, y_tilde[d] - y_hat_minus_interval_k[d])
        
        Parameters
        ----------
        y_tilde : np.ndarray
            Perfil predicho.
        y_hat_minus_interval : pd.Series
            Serie y_hat sin la contribución del intervalo.
        interval : Tuple[pd.Timestamp, pd.Timestamp]
            Tupla (fecha_inicio, fecha_fin) del intervalo.
            
        Returns
        -------
        pd.Series
            Serie de errores para los días del intervalo.
        """
        start_date, end_date = interval
        days_in_interval = pd.date_range(start=start_date, end=end_date)
        
        # Inicializar serie de errores
        error_interval = pd.Series(0.0, index=days_in_interval)
        
        # Calcular error para cada día en el intervalo
        for i, day in enumerate(days_in_interval):
            day_idx = self.days.index(day)
            error = y_tilde[day_idx] - y_hat_minus_interval[day]
            
            # Aplicar restricción no negativa
            error_interval[day] = max(0, error)
        
        return error_interval
    
    def compute_interval_adjustment(
        self, 
        error_interval: pd.Series, 
        naive_values: pd.Series,
        interval: Tuple[pd.Timestamp, pd.Timestamp]
    ) -> pd.Series:
        """
        Calcula los valores ajustados (z) para un intervalo.
        
        Implementa la ecuación:
        z_k_daily_values[d] = error_interval_k_days[d] * 
                              (sum(a_k_naive_daily_values) / sum(error_interval_k_days))
        
        Parameters
        ----------
        error_interval : pd.Series
            Serie de errores para el intervalo.
        naive_values : pd.Series
            Valores ingenuos originales para el intervalo.
        interval : Tuple[pd.Timestamp, pd.Timestamp]
            Tupla (fecha_inicio, fecha_fin) del intervalo.
            
        Returns
        -------
        pd.Series
            Serie de valores ajustados (z) para el intervalo.
        """
        start_date, end_date = interval
        days_in_interval = pd.date_range(start=start_date, end=end_date)
        
        # Inicializar serie de valores ajustados
        z_values = pd.Series(0.0, index=days_in_interval)
        
        # Calcular suma de valores ingenuos y errores para el intervalo
        naive_sum = naive_values.loc[days_in_interval].sum()
        error_sum = error_interval.sum()
        
        # Calcular valores ajustados
        if error_sum > 0:
            scaling_factor = naive_sum / error_sum
            for day in days_in_interval:
                z_values[day] = error_interval[day] * scaling_factor
        
        return z_values
    
    def update_interval_naive_values(
        self, 
        naive_values: pd.Series, 
        z_values: pd.Series,
        interval: Tuple[pd.Timestamp, pd.Timestamp]
    ) -> pd.Series:
        """
        Actualiza los valores ingenuos del intervalo usando el ajuste.
        
        Implementa la ecuación:
        a_k_updated_daily_values[d] = (1 - α) * a_k_naive_daily_values[d] + 
                                      α * z_k_daily_values[d]
        
        Parameters
        ----------
        naive_values : pd.Series
            Valores ingenuos originales para el intervalo.
        z_values : pd.Series
            Valores ajustados (z) para el intervalo.
        interval : Tuple[pd.Timestamp, pd.Timestamp]
            Tupla (fecha_inicio, fecha_fin) del intervalo.
            
        Returns
        -------
        pd.Series
            Serie de valores ingenuos actualizados.
        """
        start_date, end_date = interval
        days_in_interval = pd.date_range(start=start_date, end=end_date)
        
        # Inicializar serie de valores actualizados
        updated_values = naive_values.copy()
        
        # Actualizar valores para los días del intervalo
        for day in days_in_interval:
            updated_values[day] = (
                (1 - self.alpha) * naive_values[day] + 
                self.alpha * z_values[day]
            )
        
        return updated_values
    
    def restore_interval_contribution(
        self, 
        y_hat_minus_interval: pd.Series, 
        updated_values: pd.Series,
        interval: Tuple[pd.Timestamp, pd.Timestamp]
    ) -> pd.Series:
        """
        Reincorpora la contribución actualizada del intervalo a y_hat.
        
        Implementa la ecuación:
        y_hat[d] = y_hat_minus_interval_k[d] + a_k_updated_daily_values[d]
        
        Parameters
        ----------
        y_hat_minus_interval : pd.Series
            Serie y_hat sin la contribución del intervalo.
        updated_values : pd.Series
            Valores ingenuos actualizados para el intervalo.
        interval : Tuple[pd.Timestamp, pd.Timestamp]
            Tupla (fecha_inicio, fecha_fin) del intervalo.
            
        Returns
        -------
        pd.Series
            Serie y_hat actualizada.
        """
        start_date, end_date = interval
        days_in_interval = pd.date_range(start=start_date, end=end_date)
        
        # Crear copia de y_hat_minus_interval
        y_hat_updated = y_hat_minus_interval.copy()
        
        # Añadir la contribución actualizada del intervalo
        for day in days_in_interval:
            y_hat_updated[day] += updated_values[day]
        
        return y_hat_updated
    
    def disaggregate(self, verbose: bool = False) -> pd.Series:
        """
        Ejecuta el algoritmo de desagregación completo.
        
        Este método implementa el proceso iterativo de dos fases descrito en el paper:
        1. Fase de predicción: Entrena modelos de regresión para estimar patrones diarios
        2. Fase de actualización: Redistribuye observaciones manteniendo coherencia
        
        Parameters
        ----------
        verbose : bool, optional (default=False)
            Si es True, muestra información sobre el progreso del algoritmo.
            
        Returns
        -------
        pd.Series
            Serie de tiempo desagregada de alta frecuencia.
        """
        # Paso 1: Desagregación ingenua inicial y agregación
        if self.y_hat is None:
            self.naive_disaggregate()
            self.aggregate_series()
        
        # Paso 2: Construir matriz de diseño
        X_matriz = self.construct_design_matrix()
        
        # Paso 3: Ciclo de iteraciones de desagregación
        for i in range(self.n_lr_models):
            if verbose:
                print(f"Entrenando modelo {i+1}/{self.n_lr_models}")
            
            # Fase de predicción
            beta, metrics = self.train_regression_model(X_matriz, self.y_hat.values)
            y_tilde = self.compute_predicted_profile(X_matriz, beta)
            
            if verbose:
                print(f"  R² ajustado: {metrics['adj_r_squared']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
            for j in range(self.n_disagg_cycles):
                if verbose and j % 5 == 0:
                    print(f"  Ciclo de desagregación {j+1}/{self.n_disagg_cycles}")
                
                # Fase de actualización: para cada intervalo en todas las series
                for series in self.lf_series:
                    series_name = series.name
                    naive_values = self.naive_daily_series[series_name]
                    
                    for _, start_date, end_date in series.observations:
                        interval = (start_date, end_date)
                        
                        # 1. Eliminar contribución del intervalo
                        y_hat_minus_interval = self.remove_interval_contribution(
                            self.y_hat, interval, series_name
                        )
                        
                        # 2. Calcular error del intervalo
                        error_interval = self.calculate_interval_error(
                            y_tilde, y_hat_minus_interval, interval
                        )
                        
                        # 3. Calcular ajuste del intervalo
                        z_values = self.compute_interval_adjustment(
                            error_interval, naive_values, interval
                        )
                        
                        # 4. Actualizar valores ingenuos del intervalo
                        self.naive_daily_series[series_name] = self.update_interval_naive_values(
                            naive_values, z_values, interval
                        )
                        
                        # 5. Reincorporar contribución actualizada
                        self.y_hat = self.restore_interval_contribution(
                            y_hat_minus_interval, 
                            self.naive_daily_series[series_name].loc[
                                pd.date_range(start=start_date, end=end_date)
                            ], 
                            interval
                        )
        
        return self.y_hat
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Obtiene las métricas de rendimiento del algoritmo.
        
        Returns
        -------
        Dict[str, Any]
            Diccionario con métricas de rendimiento del algoritmo.
        """
        if not self.regression_metrics:
            return {}
        
        # Extraer métricas de la última iteración
        final_metrics = self.regression_metrics[-1]
        
        # Añadir métricas de evolución
        r_squared_trend = [m['r_squared'] for m in self.regression_metrics]
        rmse_trend = [m['rmse'] for m in self.regression_metrics]
        
        return {
            'final_metrics': final_metrics,
            'r_squared_trend': r_squared_trend,
            'rmse_trend': rmse_trend,
            'n_iterations': len(self.regression_metrics)
        }
    
    def transform_weather_variables(
        self, temperature: pd.Series, wind_speed: Optional[pd.Series] = None,
        t_ref: float = 65.0
    ) -> Dict[str, pd.Series]:
        """
        Transforma variables climáticas en HDD, CDD y HDDW.
        
        Implementa las transformaciones descritas en las ecuaciones (15)-(17) del paper.
        
        Parameters
        ----------
        temperature : pd.Series
            Serie temporal de temperaturas diarias.
        wind_speed : Optional[pd.Series], optional (default=None)
            Serie temporal de velocidades del viento diarias.
        t_ref : float, optional (default=65.0)
            Temperatura de referencia para HDD y CDD, en grados Fahrenheit.
            
        Returns
        -------
        Dict[str, pd.Series]
            Diccionario con las series transformadas HDD, CDD y HDDW.
        """
        # Validar que temperature tenga el índice temporal adecuado
        if not isinstance(temperature.index, pd.DatetimeIndex):
            raise ValueError("El índice de temperature debe ser DatetimeIndex")
        
        # Calcular HDD según ecuación (15)
        hdd = pd.Series(
            data=np.maximum(0, t_ref - temperature),
            index=temperature.index,
            name=f"HDD{int(t_ref)}"
        )
        
        # Calcular CDD según ecuación (17)
        cdd = pd.Series(
            data=np.maximum(0, temperature - t_ref),
            index=temperature.index,
            name=f"CDD{int(t_ref)}"
        )
        
        result = {
            f"HDD{int(t_ref)}": hdd,
            f"CDD{int(t_ref)}": cdd
        }
        
        # Calcular HDDW si se proporciona wind_speed, según ecuación (16)
        if wind_speed is not None:
            if not wind_speed.index.equals(temperature.index):
                raise ValueError(
                    "Los índices de temperature y wind_speed deben ser iguales"
                )
                
            hdd_ref = hdd.copy()
            hddw = pd.Series(
                index=temperature.index,
                name=f"HDDW{int(t_ref)}"
            )
            
            # Aplicar fórmula según velocidad del viento
            mask_low = wind_speed <= 8
            hddw[mask_low] = hdd_ref[mask_low] * (152 + wind_speed[mask_low]) / 160
            hddw[~mask_low] = hdd_ref[~mask_low] * (72 + wind_speed[~mask_low]) / 80
            
            result[f"HDDW{int(t_ref)}"] = hddw
        
        return result
