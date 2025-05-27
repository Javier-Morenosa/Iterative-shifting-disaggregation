import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats

# Logging configuration
logger = logging.getLogger(__name__)


@dataclass
class LowFrequencySeries:
    """
    Class to represent a low frequency time series.
    
    Parameters
    ----------
    name : str
        Identifying name of the series.
    observations : List[Tuple[float, pd.Timestamp, pd.Timestamp]]
        List of observations in the form of tuples (value, start_date, end_date).
        Each tuple represents an aggregated measurement over a time interval.
    """
    name: str
    observations: List[Tuple[float, pd.Timestamp, pd.Timestamp]]
    
    def get_days_covered(self) -> List[pd.Timestamp]:
        """Gets all days covered by this low frequency series."""
        all_days = []
        for _, start_date, end_date in self.observations:
            days = pd.date_range(start=start_date, end=end_date)
            all_days.extend(days)
        return sorted(set(all_days))
    
    def get_observation_for_day(self, day: pd.Timestamp) -> Optional[Tuple[float, int]]:
        """Gets the observation that covers a specific day."""
        for value, start_date, end_date in self.observations:
            if start_date <= day <= end_date:
                duration = (end_date - start_date).days + 1
                return (value, duration)
        return None


class ISDAlgorithm:
    """
    Implementation of the Iterative Shifting Disaggregation (ISD) Algorithm.
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
        """Initialize the ISD algorithm."""
        self.lf_series = lf_series
        self.exogenous_vars = exogenous_vars
        self.n_lr_models = n_lr_models
        self.n_disagg_cycles = n_disagg_cycles
        self.alpha = alpha
        self.handle_missing = handle_missing
        
        # Parameter validation
        self._validate_parameters()
        
        # Determine the complete period of days for disaggregation
        all_days = []
        for series in self.lf_series:
            all_days.extend(series.get_days_covered())
        self.days = sorted(set(all_days))
        self.n_days = len(self.days)
        
        # Create mapping from days to indices for vectorization
        self.day_to_idx = {day: idx for idx, day in enumerate(self.days)}
        
        # Ensure that the exogenous variables cover the entire period
        self._validate_exogenous_coverage()
        
        # Create interval information matrices for vectorization
        self._create_interval_matrices()
        
        # Dictionary to store naive daily series
        self.naive_daily_series = {}
        
        # Final result: disaggregated daily series
        self.y_hat = None
        
        # Fit metrics
        self.regression_metrics = []
    
    def _validate_parameters(self) -> None:
        """Validates the input parameters of the algorithm."""
        if not self.lf_series:
            raise ValueError("At least one low frequency series is required.")
        
        if self.n_lr_models <= 0:
            raise ValueError("n_lr_models must be a positive integer.")
            
        if self.n_disagg_cycles <= 0:
            raise ValueError("n_disagg_cycles must be a positive integer.")
            
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError("alpha must be in the range (0, 1).")
        
        if self.handle_missing not in ['zero', 'estimate']:
            raise ValueError(f"handle_missing must be 'zero' or 'estimate', not '{self.handle_missing}'")
    
    def _validate_exogenous_coverage(self) -> None:
        """Verifies that exogenous variables cover the entire required period."""
        missing_days = set(self.days) - set(self.exogenous_vars.index)
        if missing_days:
            raise ValueError(f"Exogenous variables do not cover all necessary days. Missing {len(missing_days)} days.")
    
    def _create_interval_matrices(self) -> None:
        """
        Creates matrices that represent interval membership for efficient vectorized operations.
        
        For each series, creates:
        - interval_masks: Binary matrix (n_intervals x n_days) indicating which days belong to each interval
        - interval_values: Array of interval values
        - interval_durations: Array of interval durations
        """
        self.interval_info = {}
        
        for series in self.lf_series:
            n_intervals = len(series.observations)
            
            # Initialize matrices
            interval_masks = np.zeros((n_intervals, self.n_days), dtype=bool)
            interval_values = np.zeros(n_intervals)
            interval_durations = np.zeros(n_intervals)
            
            # Fill matrices
            for i, (value, start_date, end_date) in enumerate(series.observations):
                interval_values[i] = value
                interval_durations[i] = (end_date - start_date).days + 1
                
                # Mark days that belong to this interval
                for day in pd.date_range(start=start_date, end=end_date):
                    if day in self.day_to_idx:
                        interval_masks[i, self.day_to_idx[day]] = True
            
            self.interval_info[series.name] = {
                'masks': interval_masks,
                'values': interval_values,
                'durations': interval_durations,
                'n_intervals': n_intervals
            }
    
    def naive_disaggregate_vectorized(self) -> Dict[str, np.ndarray]:
        """
        Performs vectorized naive disaggregation of all series.
        
        Uses matrix operations instead of loops for better performance.
        """
        naive_daily_series = {}
        
        for series in self.lf_series:
            info = self.interval_info[series.name]
            
            # Initialize daily values array
            daily_values = np.zeros(self.n_days)
            
            # Vectorized calculation: daily_value = value / duration for each interval
            daily_contributions = info['values'][:, np.newaxis] / info['durations'][:, np.newaxis]
            
            # Apply contributions to corresponding days using matrix multiplication
            daily_values = np.sum(info['masks'] * daily_contributions, axis=0)
            
            # Handle missing values if necessary
            if self.handle_missing == 'estimate' and np.any(daily_values == 0):
                self._estimate_missing_values_vectorized(daily_values, series.name)
            
            naive_daily_series[series.name] = daily_values
        
        self.naive_daily_series = naive_daily_series
        return naive_daily_series
    
    def _estimate_missing_values_vectorized(self, daily_values: np.ndarray, series_name: str) -> None:
        """Estimates missing values."""
        missing_mask = daily_values == 0
        n_missing = np.sum(missing_mask)
        
        if n_missing == 0:
            return
        
        logger.info(f"Estimating {n_missing} missing values for {series_name}")
        
        # Use mean of non-zero values
        mean_value = np.mean(daily_values[~missing_mask]) if np.any(~missing_mask) else 0
        daily_values[missing_mask] = mean_value
    
    def aggregate_series_vectorized(self) -> np.ndarray:
        """
        Aggregates all naive daily series.
        """
        if not self.naive_daily_series:
            self.naive_disaggregate_vectorized()
        
        # Stack all series and sum along axis 0
        all_series = np.stack(list(self.naive_daily_series.values()))
        y_hat = np.sum(all_series, axis=0)
        
        self.y_hat = y_hat
        return y_hat
    
    def construct_design_matrix(self) -> np.ndarray:
        """Constructs the design matrix with exogenous variables."""
        X_exog = self.exogenous_vars.loc[self.days].values
        X_matrix = np.hstack((np.ones((self.n_days, 1)), X_exog))
        return X_matrix
    
    def train_regression_model_vectorized(
        self, X_matrix: np.ndarray, y_hat: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Trains a linear regression model.
        """
        # Vectorized least squares: β̂ = (X^T X)^(-1) X^T y
        XTX = X_matrix.T @ X_matrix
        XTy = X_matrix.T @ y_hat
        beta = np.linalg.solve(XTX, XTy)
        
        # Calculate fit metrics
        y_pred = X_matrix @ beta
        residuals = y_hat - y_pred
        
        # Vectorized metric calculations
        n, p = X_matrix.shape
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
    
    def update_phase_vectorized(
        self, y_hat: np.ndarray, y_tilde: np.ndarray, series_name: str
    ) -> np.ndarray:
        """
        Vectorized update phase for all intervals of a series.
        """
        info = self.interval_info[series_name]
        naive_values = self.naive_daily_series[series_name].copy()
        
        # Process all intervals at once using broadcasting
        for _ in range(self.n_disagg_cycles):
            # Remove all interval contributions at once
            y_hat_minus_intervals = y_hat[np.newaxis, :] - (info['masks'] * naive_values[np.newaxis, :])
            
            # Calculate errors for all intervals (n_intervals x n_days)
            errors = y_tilde[np.newaxis, :] - y_hat_minus_intervals
            
            # Apply non-negative constraint
            errors = np.maximum(errors, 0)
            
            # Calculate sum of errors for each interval
            error_sums = np.sum(errors * info['masks'], axis=1)
            
            # Calculate sum of naive values for each interval
            naive_sums = np.sum(info['masks'] * naive_values[np.newaxis, :], axis=1)
            
            # Avoid division by zero
            scaling_factors = np.zeros_like(error_sums)
            non_zero_mask = error_sums > 0
            scaling_factors[non_zero_mask] = naive_sums[non_zero_mask] / error_sums[non_zero_mask]
            
            # Calculate z values for all intervals
            z_values = errors * scaling_factors[:, np.newaxis]
            
            # Update naive values using weighted combination
            # For each interval, update only the days that belong to it
            for i in range(info['n_intervals']):
                mask = info['masks'][i]
                naive_values[mask] = (
                    (1 - self.alpha) * naive_values[mask] + 
                    self.alpha * z_values[i, mask]
                )
            
            # Update y_hat with new naive values
            y_hat = self.aggregate_series_vectorized()
        
        # Update the stored naive values
        self.naive_daily_series[series_name] = naive_values
        
        return y_hat
    
    def disaggregate(self, verbose: bool = False) -> Dict[str, np.ndarray]:
        """
        Executes the complete disaggregation algorithm.
        """
        # Step 1: Initial naive disaggregation and aggregation
        if self.y_hat is None:
            self.naive_disaggregate_vectorized()
            self.aggregate_series_vectorized()
        
        # Step 2: Construct design matrix
        X_matrix = self.construct_design_matrix()
        
        # Step 3: Cycle of disaggregation iterations
        for i in range(self.n_lr_models):
            if verbose:
                print(f"Training model {i+1}/{self.n_lr_models}")
            
            # Prediction phase
            beta, metrics = self.train_regression_model_vectorized(X_matrix, self.y_hat)
            y_tilde = X_matrix @ beta
            
            if verbose:
                print(f"  Adjusted R²: {metrics['adj_r_squared']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
            # Update phase for each series
            for series in self.lf_series:
                self.y_hat = self.update_phase_vectorized(self.y_hat, y_tilde, series.name)
        
        # Convert to pandas Series for output
        result = {
            'aggregated': pd.Series(self.y_hat, index=self.days)
        }
        
        # Add individual series to the result
        for series_name, series_data in self.naive_daily_series.items():
            result[series_name] = pd.Series(series_data, index=self.days)
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Gets the performance metrics of the algorithm."""
        if not self.regression_metrics:
            return {}
        
        # Extract metrics from the last iteration
        final_metrics = self.regression_metrics[-1]
        
        # Add evolution metrics
        r_squared_trend = [m['r_squared'] for m in self.regression_metrics]
        rmse_trend = [m['rmse'] for m in self.regression_metrics]
        
        return {
            'final_metrics': final_metrics,
            'r_squared_trend': r_squared_trend,
            'rmse_trend': rmse_trend,
            'n_iterations': len(self.regression_metrics)
        }
