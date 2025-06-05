import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings

# Logging configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
            days = pd.date_range(start=start_date, end=end_date, freq='D')
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
    
    This version follows the paper's algorithm exactly, handling multiple
    nonuniformly spaced time series with overlapping intervals.
    """
    
    def __init__(
        self, 
        lf_series: List[LowFrequencySeries], 
        exogenous_vars: pd.DataFrame,
        n_lr_models: int = 10, 
        n_disagg_cycles: int = 10, 
        alpha: float = 0.05,
        handle_missing: str = 'zero',
        verbose: bool = False,
        should_validate_coherence: bool = True
    ):
        """Initialize the ISD algorithm."""
        self.lf_series = lf_series
        self.exogenous_vars = exogenous_vars
        self.n_lr_models = n_lr_models
        self.n_disagg_cycles = n_disagg_cycles
        self.alpha = alpha
        self.handle_missing = handle_missing
        self.verbose = verbose
        self.should_validate_coherence = should_validate_coherence
        
        # Parameter validation
        self._validate_parameters()
        
        # Determine the complete period of days for disaggregation
        all_days = []
        for series in self.lf_series:
            all_days.extend(series.get_days_covered())
        self.days = sorted(set(all_days))
        self.n_days = len(self.days)
        
        # Create mapping from days to indices
        self.day_to_idx = {day: idx for idx, day in enumerate(self.days)}
        
        # Ensure that the exogenous variables cover the entire period
        self._validate_exogenous_coverage()
        
        # Create interval information for each series
        self._create_interval_info()
        
        # Pre-compute vectorized indices for optimization
        self._precompute_vectorized_indices()
        
        # Dictionary to store naive daily series
        self.naive_daily_series = {}
        
        # Final result: disaggregated daily series
        self.y_hat = None
        
        # Fit metrics
        self.regression_metrics = []
        
        # For tracking convergence
        self.iteration_history = []
        
        # Coherence validation tracking
        self.coherence_violations = []
    
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
            logger.warning(f"Exogenous variables missing for {len(missing_days)} days")
            if self.handle_missing == 'zero':
                # Fill missing days with zeros
                for day in missing_days:
                    self.exogenous_vars.loc[day] = 0
            else:
                # Estimate missing values using interpolation
                full_index = pd.date_range(start=min(self.days), end=max(self.days), freq='D')
                self.exogenous_vars = self.exogenous_vars.reindex(full_index).interpolate(method='linear')
    
    def _create_interval_info(self) -> None:
        """
        Creates interval information for each series.
        This is critical for the algorithm's correctness.
        """
        self.interval_info = {}
        
        for series in self.lf_series:
            intervals = []
            
            for i, (value, start_date, end_date) in enumerate(series.observations):
                # Find indices for this interval
                interval_days = []
                current_date = start_date
                while current_date <= end_date:
                    if current_date in self.day_to_idx:
                        interval_days.append(self.day_to_idx[current_date])
                    current_date += pd.Timedelta(days=1)
                
                if interval_days:  # Only add if interval has days in our range
                    intervals.append({
                        'index': i,
                        'value': value,
                        'start_date': start_date,
                        'end_date': end_date,
                        'duration': (end_date - start_date).days + 1,
                        'day_indices': interval_days
                    })
            
            self.interval_info[series.name] = {
                'intervals': intervals,
                'n_intervals': len(intervals)
            }
            
            if self.verbose:
                logger.info(f"Series {series.name}: {len(intervals)} intervals")
    
    def _precompute_vectorized_indices(self) -> None:
        """Pre-computes numpy arrays for vectorized operations."""
        self.vectorized_indices = {}
        for series in self.lf_series:
            info = self.interval_info[series.name]
            self.vectorized_indices[series.name] = [
                np.array(interval['day_indices']) for interval in info['intervals']
            ]
    
    def _quick_coherence_check(self, series_name: str = None) -> bool:
        """Quick coherence validation using vectorized operations."""
        if not self.naive_daily_series:
            return True
            
        series_list = [s for s in self.lf_series if s.name == series_name] if series_name else self.lf_series
        
        for series in series_list:
            original_total = sum(obs[0] for obs in series.observations)
            current_total = np.sum(self.naive_daily_series[series.name])
            
            if not np.isclose(original_total, current_total, rtol=1e-9):
                if self.verbose:
                    logger.warning(f"Coherence violation in {series.name}: expected {original_total}, got {current_total}")
                return False
        return True
    
    def _validate_coherence_if_enabled(self, context: str = "") -> None:
        """Validates coherence if enabled, logs violations."""
        if self.should_validate_coherence:
            coherence_results = self.validate_coherence()
            if not all(coherence_results.values()):
                violation = f"Coherence violation detected {context}"
                self.coherence_violations.append(violation)
                if self.verbose:
                    logger.warning(violation)
    
    def perform_naive_disaggregation(self) -> Dict[str, np.ndarray]:
        """
        Performs naive disaggregation of all series.
        Distributes each interval's value equally across its days.
        """
        self.naive_daily_series = {}
        
        for series in self.lf_series:
            daily_values = np.zeros(self.n_days)
            info = self.interval_info[series.name]
            vectorized_indices = self.vectorized_indices[series.name]
            
            # Vectorized disaggregation
            for interval, day_indices in zip(info['intervals'], vectorized_indices):
                daily_value = interval['value'] / interval['duration']
                daily_values[day_indices] = daily_value  # Vectorized assignment
            
            # Handle missing values if necessary
            if self.handle_missing == 'estimate' and np.any(daily_values == 0):
                self._estimate_missing_values(daily_values, series.name)
            
            self.naive_daily_series[series.name] = daily_values
            
            if self.verbose:
                logger.info(f"Naive disaggregation for {series.name}: "
                          f"sum={np.sum(daily_values):.2f}, "
                          f"non-zero days={np.sum(daily_values > 0)}")
        
        # Validate coherence after naive disaggregation
        self._validate_coherence_if_enabled("after naive disaggregation")
        
        return self.naive_daily_series
    
    def _estimate_missing_values(self, daily_values: np.ndarray, series_name: str) -> None:
        """Estimates missing values using interpolation."""
        missing_mask = daily_values == 0
        n_missing = np.sum(missing_mask)
        
        if n_missing == 0 or n_missing == len(daily_values):
            return
        
        if self.verbose:
            logger.info(f"Estimating {n_missing} missing values for {series_name}")
        
        # Use pandas for interpolation
        series = pd.Series(daily_values, index=self.days)
        series.replace(0, np.nan, inplace=True)
        series.interpolate(method='linear', inplace=True)
        series.fillna(method='bfill', inplace=True)
        series.fillna(method='ffill', inplace=True)
        
        daily_values[:] = series.values
    
    def aggregate_series(self) -> np.ndarray:
        """Aggregates all naive daily series."""
        if not self.naive_daily_series:
            self.perform_naive_disaggregation()
        
        # Vectorized aggregation
        y_hat = np.sum(list(self.naive_daily_series.values()), axis=0)
        self.y_hat = y_hat
        return y_hat
    
    def construct_design_matrix(self) -> np.ndarray:
        """Constructs the design matrix with exogenous variables."""
        X_exog = self.exogenous_vars.loc[self.days].values
        X_matrix = np.hstack((np.ones((self.n_days, 1)), X_exog))
        return X_matrix
    
    def train_regression_model(
        self, 
        X_matrix: np.ndarray, 
        y_hat: np.ndarray,
        model_type: str = 'linear',
        reg_alpha: float = 1.0
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Trains regression model as specified in the paper."""
        
        # Remove intercept column for sklearn models
        X = X_matrix[:, 1:]
        
        # Select and train model
        if model_type == 'linear':
            model = LinearRegression(fit_intercept=True)
        elif model_type == 'ridge':
            model = Ridge(alpha=reg_alpha, fit_intercept=True)
        else:
            model = LinearRegression(fit_intercept=True)
        
        # Train model
        model.fit(X, y_hat)
        
        # Get coefficients (including intercept)
        coefficients = np.hstack([model.intercept_, model.coef_])
        
        # Calculate predictions and metrics
        y_pred = model.predict(X)
        
        metrics = {
            'r_squared': r2_score(y_hat, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_hat, y_pred)),
            'mae': mean_absolute_error(y_hat, y_pred)
        }
        
        self.regression_metrics.append(metrics)
        
        return coefficients, metrics
    
    def update_phase(
        self, 
        y_hat: np.ndarray, 
        y_tilde: np.ndarray, 
        series_name: str
    ) -> np.ndarray:
        """
        Update phase for a single series - this is the core of the ISD algorithm.
        Follows Algorithm 1 from the paper exactly.
        """
        info = self.interval_info[series_name]
        series_values = self.naive_daily_series[series_name].copy()
        vectorized_indices = self.vectorized_indices[series_name]
        
        # For each disaggregation cycle
        for cycle in range(self.n_disagg_cycles):
            # For each interval in the series
            for interval, day_indices in zip(info['intervals'], vectorized_indices):
                # Step 1: Remove interval contribution from aggregate (vectorized)
                y_hat_minus_k = y_hat.copy()
                y_hat_minus_k[day_indices] -= series_values[day_indices]
                
                # Step 2: Calculate error (already vectorized)
                error = np.maximum(y_tilde - y_hat_minus_k, 0)
                
                # Step 3: Calculate interval-specific values (vectorized)
                interval_error = np.zeros(self.n_days)
                interval_naive = np.zeros(self.n_days)
                
                interval_error[day_indices] = error[day_indices]
                interval_naive[day_indices] = series_values[day_indices]
                
                # Sum of errors and naive values in the interval (vectorized)
                error_sum = np.sum(interval_error)
                naive_sum = np.sum(interval_naive)
                
                # Step 4: Calculate z values (redistributed values)
                if error_sum > 0:
                    # Scale factor ensures sum is preserved
                    scale = naive_sum / error_sum
                    z_values = interval_error * scale
                else:
                    # If no error, keep original values
                    z_values = interval_naive.copy()
                
                # Step 5: Update values using weighted combination (vectorized)
                series_values[day_indices] = (
                    (1 - self.alpha) * series_values[day_indices] + 
                    self.alpha * z_values[day_indices]
                )
                
                # Update y_hat immediately with the new values (vectorized)
                y_hat[day_indices] = y_hat_minus_k[day_indices] + series_values[day_indices]
        
        # Update the stored series values
        self.naive_daily_series[series_name] = series_values
        
        # Validate coherence after update phase
        self._validate_coherence_if_enabled(f"after updating {series_name}")
        
        return y_hat
    
    def disaggregate(self, verbose: Optional[bool] = None) -> Dict[str, pd.Series]:
        """
        Executes the complete ISD algorithm as described in the paper.
        """
        if verbose is not None:
            self.verbose = verbose
        
        # Step 1: Initial naive disaggregation
        if self.verbose:
            logger.info("Step 1: Performing naive disaggregation")
        self.perform_naive_disaggregation()
        self.y_hat = self.aggregate_series()
        
        # Step 2: Construct design matrix
        if self.verbose:
            logger.info("Step 2: Constructing design matrix")
        X_matrix = self.construct_design_matrix()
        
        # Check for potential issues
        diagnostics = self.get_model_diagnostics(X_matrix, self.y_hat)
        if diagnostics['recommendations']:
            logger.warning(f"Model diagnostics: {diagnostics['recommendations']}")
        
        # Step 3: Main iteration loop
        if self.verbose:
            logger.info(f"Step 3: Starting {self.n_lr_models} regression iterations")
        
        for lr_iter in range(self.n_lr_models):
            if self.verbose:
                logger.info(f"\nRegression iteration {lr_iter + 1}/{self.n_lr_models}")
            
            # Prediction phase
            beta, metrics = self.train_regression_model(X_matrix, self.y_hat)
            y_tilde = X_matrix @ beta
            
            if self.verbose:
                logger.info(f"  R²: {metrics['r_squared']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
            # Store current state for convergence tracking
            old_y_hat = self.y_hat.copy()
            
            # Update phase for each series
            for series in self.lf_series:
                if self.verbose:
                    logger.info(f"  Updating series: {series.name}")
                self.y_hat = self.update_phase(self.y_hat, y_tilde, series.name)
            
            # Validate coherence after each iteration
            self._validate_coherence_if_enabled(f"after iteration {lr_iter + 1}")
            
            # Track convergence
            change = np.mean(np.abs(self.y_hat - old_y_hat))
            self.iteration_history.append({
                'iteration': lr_iter,
                'r_squared': metrics['r_squared'],
                'rmse': metrics['rmse'],
                'mean_change': change
            })
            
            if self.verbose:
                logger.info(f"  Mean change: {change:.6f}")
        
        # Final coherence validation
        final_coherence = {}
        if self.should_validate_coherence:
            final_coherence = self.validate_coherence()
            if not all(final_coherence.values()):
                logger.warning("Final result contains coherence violations!")
        
        # Prepare results
        result = {
            'aggregated': pd.Series(self.y_hat, index=self.days),
            'metrics': self.get_performance_metrics(),
            'coherence': final_coherence
        }
        
        # Add individual series
        for series_name, series_data in self.naive_daily_series.items():
            result[series_name] = pd.Series(series_data, index=self.days)
        
        return result
    
    def get_model_diagnostics(self, X_matrix: np.ndarray, y_hat: np.ndarray) -> Dict:
        """Model diagnostics as specified in the original implementation."""
        n, p = X_matrix.shape
        
        diagnostics = {
            'sample_to_feature_ratio': n / p,
            'recommendations': []
        }
        
        # Check for issues
        if diagnostics['sample_to_feature_ratio'] < 5:
            diagnostics['recommendations'].append("Consider regularization (Ridge/Lasso)")
        
        # Check for missing values
        if np.any(np.isnan(X_matrix)) or np.any(np.isnan(y_hat)):
            diagnostics['recommendations'].append("Handle missing values before training")
        
        # Check condition number
        try:
            cond_number = np.linalg.cond(X_matrix)
            if cond_number > 1e12:
                diagnostics['recommendations'].append("High multicollinearity - use Ridge regression")
        except:
            pass
        
        return diagnostics
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Gets performance metrics of the algorithm."""
        if not self.regression_metrics:
            return {}
        
        # Final metrics
        final_metrics = self.regression_metrics[-1]
        
        # Trends
        r_squared_trend = [m['r_squared'] for m in self.regression_metrics]
        rmse_trend = [m['rmse'] for m in self.regression_metrics]
        
        # Convergence info
        convergence_info = {}
        if self.iteration_history:
            changes = [h['mean_change'] for h in self.iteration_history]
            convergence_info = {
                'final_change': changes[-1],
                'converged': changes[-1] < 1e-6,
                'iterations_to_converge': len(changes)
            }
        
        return {
            'final_metrics': final_metrics,
            'r_squared_trend': r_squared_trend,
            'rmse_trend': rmse_trend,
            'n_iterations': len(self.regression_metrics),
            'convergence': convergence_info,
            'coherence_violations': len(self.coherence_violations)
        }
    
    def validate_coherence(self) -> Dict[str, bool]:
        """
        Validates that the disaggregated series maintain coherence
        (sum to original values) as required by the paper.
        """
        if not self.naive_daily_series:
            return {"error": "No disaggregation performed yet"}
        
        coherence_results = {}
        
        for series in self.lf_series:
            series_name = series.name
            daily_values = self.naive_daily_series[series_name]
            
            # Check each interval using vectorized operations
            for obs_idx, (value, start_date, end_date) in enumerate(series.observations):
                # Create date range and get indices
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                day_indices = np.array([self.day_to_idx[day] for day in date_range 
                                      if day in self.day_to_idx])
                
                # Vectorized sum
                interval_sum = np.sum(daily_values[day_indices]) if len(day_indices) > 0 else 0
                
                # Check if sum matches original value
                is_coherent = np.isclose(interval_sum, value, rtol=1e-9)
                coherence_results[f"{series_name}_interval_{obs_idx}"] = is_coherent
                
                if not is_coherent and self.verbose:
                    logger.warning(f"Coherence violation in {series_name} interval {obs_idx}: "
                                 f"expected {value}, got {interval_sum}")
        
        return coherence_results
