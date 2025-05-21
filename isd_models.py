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
    
    Notes
    -----
    This class provides methods to access the days covered by the series
    and to obtain the observation corresponding to a specific day.
    """
    name: str
    observations: List[Tuple[float, pd.Timestamp, pd.Timestamp]]
    
    def get_days_covered(self) -> List[pd.Timestamp]:
        """
        Gets all days covered by this low frequency series.
        
        Returns
        -------
        List[pd.Timestamp]
            Sorted list of daily timestamps covered by the series.
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
        Gets the observation that covers a specific day.
        
        Parameters
        ----------
        day : pd.Timestamp
            The day for which the observation is sought.
            
        Returns
        -------
        Optional[Tuple[float, int]]
            Tuple with (observation_value, duration_in_days) or None if the day 
            is not covered by any observation.
        """
        for value, start_date, end_date in self.observations:
            if start_date <= day <= end_date:
                duration = (end_date - start_date).days + 1
                return (value, duration)
        return None


class ISDAlgorithm:
    """
    Implementation of the Iterative Shifting Disaggregation (ISD) Algorithm.
    
    The ISD algorithm disaggregates multiple low-frequency time series with 
    overlapping intervals to a high-frequency (daily) series, using 
    independent correlated variables to improve accuracy.
    
    Parameters
    ----------
    lf_series : List[LowFrequencySeries]
        List of low frequency series to disaggregate.
    exogenous_vars : pd.DataFrame
        DataFrame with exogenous variables at daily frequency.
        The index must be a DatetimeIndex.
    n_lr_models : int, optional (default=10)
        Number of linear regression models to train.
        Recommended values: 10 if there are 3+ series, more for fewer series.
    n_disagg_cycles : int, optional (default=10)
        Number of disaggregation cycles per model.
        Recommended values: 10 if there are 3+ series, more for fewer series.
    alpha : float, optional (default=0.05)
        Weighting that controls error redistribution.
        Recommended values: 0.05 if there are 3+ series, adjust for fewer series.
    handle_missing : str, optional (default='zero')
        Method for handling missing data: 'zero' (initialize with zeros) or
        'estimate' (estimate using regression).
    
    Notes
    -----
    The algorithm operates in an iterative two-phase process:
    1. Prediction Phase: Trains linear regression models using exogenous
       variables to estimate daily patterns.
    2. Update Phase: Redistributes low-frequency observations among
       high-frequency periods, maintaining temporal coherence.
    
    References
    ----------
    Quinn, C.O., Brown, R.H., Corliss, G.F., & Povinelli, R.J. (2025). 
    An Iterative Shifting Disaggregation Algorithm for Multi-Source,
    Irregularly Sampled, and Overlapped Time Series. Sensors.
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
        
        # Ensure that the exogenous variables cover the entire period
        self._validate_exogenous_coverage()
            
        # Dictionary to store naive daily series
        self.naive_daily_series = {}
        
        # Final result: disaggregated daily series
        self.y_hat = None
        
        # Fit metrics
        self.regression_metrics = []
    
    def _validate_parameters(self) -> None:
        """
        Validates the input parameters of the algorithm.
        
        Raises
        ------
        ValueError
            If any parameter does not meet the required constraints.
        """
        if not self.lf_series:
            raise ValueError("At least one low frequency series is required.")
        
        if self.n_lr_models <= 0:
            raise ValueError("n_lr_models must be a positive integer.")
            
        if self.n_disagg_cycles <= 0:
            raise ValueError("n_disagg_cycles must be a positive integer.")
            
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError("alpha must be in the range (0, 1).")
        
        if self.handle_missing not in ['zero', 'estimate']:
            raise ValueError(
                "handle_missing must be 'zero' or 'estimate', "
                f"not '{self.handle_missing}'"
            )
    
    def _validate_exogenous_coverage(self) -> None:
        """
        Verifies that exogenous variables cover the entire required period.
        
        Raises
        ------
        ValueError
            If there are missing days in the exogenous variables.
        """
        missing_days = set(self.days) - set(self.exogenous_vars.index)
        if missing_days:
            raise ValueError(
                f"Exogenous variables do not cover all necessary days. "
                f"Missing {len(missing_days)} days."
            )
    
    def naive_disaggregate(self) -> Dict[str, pd.Series]:
        """
        Performs the initial naive disaggregation of each low frequency series.
        
        This method divides each low frequency observation uniformly among
        the days it spans, following the equation: 
        a_{i,j} = A_{i,d} / T^A_i, where j = d - T^A_i + 1, ..., d
        
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary with naive daily series for each input series.
        
        Notes
        -----
        For missing data, it can be initialized with zeros or estimated according to
        the value of the handle_missing parameter.
        """
        naive_daily_series = {}
        
        for series in self.lf_series:
            # Initialize daily series with zeros
            daily_values = pd.Series(0.0, index=self.days)
            
            # Disaggregate each low frequency observation
            for value, start_date, end_date in series.observations:
                days_in_interval = pd.date_range(start=start_date, end=end_date)
                duration = len(days_in_interval)
                
                # Daily value = observation value / number of days
                daily_value = value / duration
                
                # Assign the same daily value to all days in the interval
                for day in days_in_interval:
                    daily_values[day] = daily_value
            
            # Handle missing values if necessary
            if self.handle_missing == 'estimate' and daily_values.isna().any():
                self._estimate_missing_values(daily_values, series.name)
            
            naive_daily_series[series.name] = daily_values
        
        self.naive_daily_series = naive_daily_series
        return naive_daily_series
    
    def _estimate_missing_values(
        self, daily_values: pd.Series, series_name: str
    ) -> None:
        """
        Estimates missing values in the naive disaggregated series.
        
        Parameters
        ----------
        daily_values : pd.Series
            Series of daily values with possible missing values.
        series_name : str
            Name of the series for log registration.
        """
        # Identify days with missing values
        missing_days = daily_values[daily_values.isna()].index
        if len(missing_days) == 0:
            return
            
        logger.info(
            f"Estimating {len(missing_days)} missing values for {series_name}"
        )
        
        # Simple method: use the mean of available days
        mean_value = daily_values.mean()
        daily_values.fillna(mean_value, inplace=True)
    
    def aggregate_series(self) -> pd.Series:
        """
        Aggregates all naive daily series to obtain the initial estimate.
        
        This operation follows the equation:
        y_hat[d] = a[d] + b[d] + c[d] + ...
        
        Returns
        -------
        pd.Series
            Initial aggregated series (y_hat) representing the daily estimate.
        """
        if not self.naive_daily_series:
            self.naive_disaggregate()
        
        # Initialize aggregated series with zeros
        y_hat = pd.Series(0.0, index=self.days)
        
        # Sum all naive daily series
        for series_name, daily_values in self.naive_daily_series.items():
            y_hat = y_hat.add(daily_values)
        
        self.y_hat = y_hat
        return y_hat
    
    def construct_design_matrix(self) -> np.ndarray:
        """
        Constructs the design matrix with exogenous variables.
        
        Forms a matrix of dimensions (ND x P+1), where ND is the number of days
        and P+1 includes a column of ones for the intercept.
        
        Returns
        -------
        np.ndarray
            Design matrix X_matrix for the regression model.
        """
        X_exog = self.exogenous_vars.loc[self.days].values
        
        n_days = len(self.days)
        X_matrix = np.hstack((np.ones((n_days, 1)), X_exog))
        
        return X_matrix
    
    def train_regression_model(
        self, X_matrix: np.ndarray, y_hat: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Trains a linear regression model.
        
        Implements the least squares equation:
        β̂ = (X^T X)^(-1) X^T y
        
        Parameters
        ----------
        X_matrix : np.ndarray
            Design matrix with predictor variables.
        y_hat : np.ndarray
            Vector of target values (aggregated series).
            
        Returns
        -------
        Tuple[np.ndarray, Dict[str, float]]
            Beta coefficients of the model and fit metrics.
        """
        # Least squares implementation
        X_transpose = X_matrix.T
        XTX = X_transpose.dot(X_matrix)
        XTX_inv = np.linalg.inv(XTX)
        beta = XTX_inv.dot(X_transpose).dot(y_hat)
        
        # Calculate fit metrics
        y_pred = X_matrix.dot(beta)
        residuals = y_hat - y_pred
        
        # Regression metrics
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
    
    def compute_predicted_profile(
        self, X_matrix: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        """
        Calculates the predicted profile using the model coefficients.
        
        Implements the equation:
        y_tilde = X_matrix @ beta
        
        Parameters
        ----------
        X_matrix : np.ndarray
            Design matrix.
        beta : np.ndarray
            Regression coefficients.
            
        Returns
        -------
        np.ndarray
            Predicted profile y_tilde.
        """
        # y_tilde = X_matrix @ beta (matrix multiplication)
        y_tilde = X_matrix.dot(beta)
        
        return y_tilde
    
    def remove_interval_contribution(
        self, 
        y_hat: pd.Series, 
        interval: Tuple[pd.Timestamp, pd.Timestamp], 
        series_name: str
    ) -> pd.Series:
        """
        Removes the contribution of a specific interval from the current y_hat.
        
        Parameters
        ----------
        y_hat : pd.Series
            Current aggregated series.
        interval : Tuple[pd.Timestamp, pd.Timestamp]
            Tuple (start_date, end_date) of the interval.
        series_name : str
            Name of the series to which the interval belongs.
            
        Returns
        -------
        pd.Series
            y_hat series without the interval contribution.
        """
        start_date, end_date = interval
        days_in_interval = pd.date_range(start=start_date, end=end_date)
        
        # Create copy of y_hat
        y_hat_minus_interval = y_hat.copy()
        
        # Subtract the naive contribution of the interval
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
        Calculates the error for a specific interval and applies non-negative constraint.
        
        Implements the equation:
        error_interval_k_days[d] = max(0, y_tilde[d] - y_hat_minus_interval_k[d])
        
        Parameters
        ----------
        y_tilde : np.ndarray
            Predicted profile.
        y_hat_minus_interval : pd.Series
            y_hat series without the interval contribution.
        interval : Tuple[pd.Timestamp, pd.Timestamp]
            Tuple (start_date, end_date) of the interval.
            
        Returns
        -------
        pd.Series
            Series of errors for the days in the interval.
        """
        start_date, end_date = interval
        days_in_interval = pd.date_range(start=start_date, end=end_date)
        
        # Initialize error series
        error_interval = pd.Series(0.0, index=days_in_interval)
        
        # Calculate error for each day in the interval
        for i, day in enumerate(days_in_interval):
            day_idx = self.days.index(day)
            error = y_tilde[day_idx] - y_hat_minus_interval[day]
            
            # Apply non-negative constraint
            error_interval[day] = max(0, error)
        
        return error_interval
    
    def compute_interval_adjustment(
        self, 
        error_interval: pd.Series, 
        naive_values: pd.Series,
        interval: Tuple[pd.Timestamp, pd.Timestamp]
    ) -> pd.Series:
        """
        Calculates the adjusted values (z) for an interval.
        
        Implements the equation:
        z_k_daily_values[d] = error_interval_k_days[d] * 
                              (sum(a_k_naive_daily_values) / sum(error_interval_k_days))
        
        Parameters
        ----------
        error_interval : pd.Series
            Series of errors for the interval.
        naive_values : pd.Series
            Original naive values for the interval.
        interval : Tuple[pd.Timestamp, pd.Timestamp]
            Tuple (start_date, end_date) of the interval.
            
        Returns
        -------
        pd.Series
            Series of adjusted values (z) for the interval.
        """
        start_date, end_date = interval
        days_in_interval = pd.date_range(start=start_date, end=end_date)
        
        # Initialize adjusted values series
        z_values = pd.Series(0.0, index=days_in_interval)
        
        # Calculate sum of naive values and errors for the interval
        naive_sum = naive_values.loc[days_in_interval].sum()
        error_sum = error_interval.sum()
        
        # Calculate adjusted values
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
        Updates the naive values of the interval using the adjustment.
        
        Implements the equation:
        a_k_updated_daily_values[d] = (1 - α) * a_k_naive_daily_values[d] + 
                                      α * z_k_daily_values[d]
        
        Parameters
        ----------
        naive_values : pd.Series
            Original naive values for the interval.
        z_values : pd.Series
            Adjusted values (z) for the interval.
        interval : Tuple[pd.Timestamp, pd.Timestamp]
            Tuple (start_date, end_date) of the interval.
            
        Returns
        -------
        pd.Series
            Series of updated naive values.
        """
        start_date, end_date = interval
        days_in_interval = pd.date_range(start=start_date, end=end_date)
        
        # Initialize updated values series
        updated_values = naive_values.copy()
        
        # Update values for the days in the interval
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
        Reincorporates the updated contribution of the interval to y_hat.
        
        Implements the equation:
        y_hat[d] = y_hat_minus_interval_k[d] + a_k_updated_daily_values[d]
        
        Parameters
        ----------
        y_hat_minus_interval : pd.Series
            y_hat series without the interval contribution.
        updated_values : pd.Series
            Updated naive values for the interval.
        interval : Tuple[pd.Timestamp, pd.Timestamp]
            Tuple (start_date, end_date) of the interval.
            
        Returns
        -------
        pd.Series
            Updated y_hat series.
        """
        start_date, end_date = interval
        days_in_interval = pd.date_range(start=start_date, end=end_date)
        
        # Create copy of y_hat_minus_interval
        y_hat_updated = y_hat_minus_interval.copy()
        
        # Add the updated contribution of the interval
        for day in days_in_interval:
            y_hat_updated[day] += updated_values[day]
        
        return y_hat_updated
    
    def disaggregate(self, verbose: bool = False) -> Dict[str, pd.Series]:
        """
        Executes the complete disaggregation algorithm.
        
        This method implements the iterative two-phase process described in the paper:
        1. Prediction Phase: Trains regression models to estimate daily patterns
        2. Update Phase: Redistributes observations maintaining coherence
        
        Parameters
        ----------
        verbose : bool, optional (default=False)
            If True, displays information about the algorithm's progress.
            
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary with the aggregated series ('aggregated') and the individual series.
        """
        # Step 1: Initial naive disaggregation and aggregation
        if self.y_hat is None:
            self.naive_disaggregate()
            self.aggregate_series()
        
        # Step 2: Construct design matrix
        X_matrix = self.construct_design_matrix()
        
        # Step 3: Cycle of disaggregation iterations
        for i in range(self.n_lr_models):
            if verbose:
                print(f"Training model {i+1}/{self.n_lr_models}")
            
            # Prediction phase
            beta, metrics = self.train_regression_model(X_matrix, self.y_hat.values)
            y_tilde = self.compute_predicted_profile(X_matrix, beta)
            
            if verbose:
                print(f"  Adjusted R²: {metrics['adj_r_squared']:.4f}, RMSE: {metrics['rmse']:.4f}")
            
            for j in range(self.n_disagg_cycles):
                if verbose and j % 5 == 0:
                    print(f"  Disaggregation cycle {j+1}/{self.n_disagg_cycles}")
                
                # Update phase: for each interval in all series
                for series in self.lf_series:
                    series_name = series.name
                    naive_values = self.naive_daily_series[series_name]
                    
                    for _, start_date, end_date in series.observations:
                        interval = (start_date, end_date)
                        
                        # 1. Remove interval contribution
                        y_hat_minus_interval = self.remove_interval_contribution(
                            self.y_hat, interval, series_name
                        )
                        
                        # 2. Calculate interval error
                        error_interval = self.calculate_interval_error(
                            y_tilde, y_hat_minus_interval, interval
                        )
                        
                        # 3. Calculate interval adjustment
                        z_values = self.compute_interval_adjustment(
                            error_interval, naive_values, interval
                        )
                        
                        # 4. Update interval naive values
                        self.naive_daily_series[series_name] = self.update_interval_naive_values(
                            naive_values, z_values, interval
                        )
                        
                        # 5. Reincorporate updated contribution
                        self.y_hat = self.restore_interval_contribution(
                            y_hat_minus_interval, 
                            self.naive_daily_series[series_name].loc[
                                pd.date_range(start=start_date, end=end_date)
                            ], 
                            interval
                        )
        
        # Create dictionary with all series
        result = {'aggregated': self.y_hat}
        # Add individual series to the result
        for series_name, series_data in self.naive_daily_series.items():
            result[series_name] = series_data
        
        return result
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Gets the performance metrics of the algorithm.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with algorithm performance metrics.
        """
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
