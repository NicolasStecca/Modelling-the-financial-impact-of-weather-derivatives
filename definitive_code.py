
###**IMPORTAZIONE MODULI**
"""

pip install arch

pip install ipynb

pip install nbimporter

pip install hawkeslib

# Analisi dati e manipolazione
import pandas as pd
import numpy as np
import calendar

# Visualizzazione
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Statistica e distribuzioni
from scipy import stats
from scipy.stats import expon, norm, t, gamma, gaussian_kde
from scipy.optimize import minimize

# Test statistici
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

# Serie temporali
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from arch import arch_model
from ou import OrnsteinUhlenbeckEstimator_2
from hawkeslib import UnivariateExpHawkesProcess

"""###**CODICI PER ANALISI PRELIMINARE DELLA SERIE**"""

def fill_tavg(df, tavg_col='tavg', tmax_col='tmax', tmin_col='tmin', window=7):
    df = df.copy()  # Create a copy of the dataframe to avoid modifying the original
    half_window = window // 2  # Half of the window size for centered averaging

    for idx in df.index:
        tmax = df.at[idx, tmax_col]
        tmin = df.at[idx, tmin_col]
        tavg = df.at[idx, tavg_col]

        # Case 1: if both tmax and tmin are available → (tmax + tmin)/2
        if pd.notna(tmax) and pd.notna(tmin):
            # if either is NaN, the sum automatically ignores NaN
            df.at[idx, tavg_col] = (pd.Series([tmax, tmin]).mean())
        else:
            # Case 2: both tmax and tmin are NaN
            if pd.isna(tavg):
                # Try to fill using the mean of a centered window
                start = max(0, idx - half_window)
                end = min(len(df), idx + half_window + 1)
                window_values = df[tavg_col].iloc[start:end].dropna()  # Drop NaNs in the window
                if len(window_values) > 0:
                    df.at[idx, tavg_col] = window_values.mean()  # Fill with window mean
    return df  # Return the filled dataframe

def plots_stats(data, var):
    # Prepare data with datetime and separate columns for year, month, day
    x = data[["time", var]]
    x['time'] = pd.to_datetime(x['time'])
    x['year'] = x['time'].dt.year
    x['month'] = x['time'].dt.month
    x['day'] = x['time'].dt.day

    # Function to compute monthly statistics: mean and variance
    def compute_monthly_stats(df):
        groups = df.groupby(['year', 'month'])[var]
        stats = groups.agg(['mean', 'var']).reset_index()
        # Rename columns for clarity
        stats.rename(columns={'mean': f'mean {var}', 'var': f'variance {var}'}, inplace=True)
        stats['year'] = stats['year'].astype(int)
        return stats

    stats = compute_monthly_stats(x)  # Compute monthly statistics
    years = stats['year'].unique()
    months = stats["month"].unique()

    # Pivot tables for plotting: mean and variance per year and month
    stats_pivot_mean = stats.pivot(index='year', columns='month', values=f'mean {var}')
    stats_pivot_var = stats.pivot(index='year', columns='month', values=f'variance {var}')

    # Split data into two periods for plotting
    pivot_media_1 = stats_pivot_mean.loc[2000:2012]  # Mean 2000–2012
    pivot_varianza_1 = stats_pivot_var.loc[2000:2012]  # Variance 2000–2012
    pivot_media_2 = stats_pivot_mean.loc[2013:2024]  # Mean 2013–2024
    pivot_varianza_2 = stats_pivot_var.loc[2013:2024]  # Variance 2013–2024

    # Pivot tables to plot monthly trends across years
    pivot_mean_monthly = stats.pivot_table(index='month', columns='year', values=f'mean {var}')
    pivot_var_monthly = stats.pivot_table(index='month', columns='year', values=f'variance {var}')

    # Function to plot grouped bar chart by year
    def plot_grouped_bars_anno(data, title, ax):
        n_years = len(data.index)
        n_months = len(data.columns)
        x = np.arange(n_years)
        width = 0.7 / n_months
        for i, month in enumerate(data.columns):
            month_name = calendar.month_abbr[month]  # Month abbreviation
            ax.bar(x + i*width, data[month], width=width, label=month_name, edgecolor="black")
        ax.set_xticks(x + width*(n_months-1)/2)
        ax.set_xticklabels(data.index, rotation=45)
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend(ncol=6, fontsize=8)

    # Function to plot grouped bar chart by month
    def plot_grouped_bars_mese(data, title, ax):
        n_months = len(data.index)
        n_years = len(data.columns)
        x = np.arange(n_months)
        width = 0.9 / n_years
        for i, year in enumerate(data.columns):
            ax.bar(x + i*width, data[year], width=width, label=str(year), edgecolor="black")
        ax.set_xticks(x + width*(n_years-1)/2)
        ax.set_xticklabels([calendar.month_name[m] for m in data.index])
        ax.set_xlabel('Month')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend(ncol=6, fontsize=8)

    #  Monthly mean values 
    fig, axs = plt.subplots(2, 1, figsize=(16,10))
    plot_grouped_bars_anno(pivot_media_1, f'Monthly mean {var} (2000–2012)', axs[0])
    plot_grouped_bars_anno(pivot_media_2, f'Monthly mean {var} (2013–2024)', axs[1])
    plt.tight_layout()
    plt.show()

    # Figure 2: Monthly variance values 
    fig, axs = plt.subplots(2, 1, figsize=(16,10))
    plot_grouped_bars_anno(pivot_varianza_1, f'Monthly variance {var} (2000–2012)', axs[0])
    plot_grouped_bars_anno(pivot_varianza_2, f'Monthly variance {var} (2013–2024)', axs[1])
    plt.tight_layout()
    plt.show()

    # Monthly trends across years 
    fig, axs = plt.subplots(2, 1, figsize=(16,10))
    plot_grouped_bars_mese(pivot_mean_monthly, f'Yearly mean {var}', axs[0])
    plot_grouped_bars_mese(pivot_var_monthly, f'Yearly variance {var}', axs[1])
    plt.tight_layout()
    plt.show()

    return stats  # Return the computed monthly statistics

def rolling_window_estimates(df, var, years=[5,10,15,20]):
    # Sort dataframe by time to ensure proper rolling calculations
    series = df.sort_values("time")

    # Create a 2x2 subplot for different rolling window sizes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()  # Flatten to 1D array for easy iteration

    # Loop over each window size in years
    for i, width in enumerate(years):
        ax = axes[i]
        # Plot rolling mean over the window (365 days * width in years)
        series[var].rolling(window=365*width).mean().plot(
            ax=ax, color="tab:red", label="Mean",
            title=f"Rolling mean and std over {width} years"
        )
        # Plot rolling standard deviation (sqrt of rolling variance)
        np.sqrt(series[var].rolling(window=365*width).var()).plot(
            ax=ax, color="tab:blue", label="Standard deviation"
        )
        plt.plot()
        ax.legend()

    plt.tight_layout()
    plt.show()

def seasonal_decomposition(df, var):
    # Perform additive seasonal decomposition with a yearly period (365 days)
    decomposing = seasonal_decompose(
        df[var],
        model="additive",
        period=int(365),
        extrapolate_trend="freq"  # Fill trend values at the edges
    )

    # Extract components
    trend = decomposing.trend       # Long-term trend
    seasonal = decomposing.seasonal # Repeating seasonal pattern
    residual = decomposing.resid    # Remaining noise/residuals

    # Plot the decomposition (observed, trend, seasonal, residual)
    decomposing.plot()
    plt.show()

def deterministic_approximation(data, var, trunc=0, plot=True):
    # Create a new dataframe to store results
    df = pd.DataFrame()

    timeline = data["time"]
    df["time"] = timeline
    t = np.arange(len(timeline)) + 1  # Time index starting from 1

    series = data[f"{var}"]
    df[f"{var}"] = series

    w = 2 * np.pi / 365  # Annual frequency for sine/cosine terms

    # Build design matrix for OLS: constant, linear trend, seasonal sine and cosine
    X = pd.DataFrame({
        'const': 1,
        't': t,
        'sin_wt': np.sin(w * t),
        'cos_wt': np.cos(w * t)
    })
    X.index = data.index

    # Fit ordinary least squares (OLS) model
    model = sm.OLS(series, X).fit()
    params_dict = model.params.to_dict()  # Extract fitted parameters

    # Predict series using the deterministic approximation
    series_fit = model.predict(X)
    df["approx"] = series_fit

    # Compute residuals
    resid = series - series_fit
    df["residuals"] = resid

    # Optional plotting
    if plot:
        plt.figure(figsize=(15,5))
        plt.plot(timeline[-trunc:], series[-trunc:], label='Real data')
        plt.plot(timeline[-trunc:], series_fit[-trunc:], label='OLS fit', color='red')
        plt.xlabel('Time')
        plt.ylabel('Temperature')  # Translated from 'Temperatura'
        plt.legend()
        plt.show()

        plt.figure(figsize=(15,5))
        plt.plot(t, resid, label='Residuals')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.legend()
        plt.show()

    # Return fitted parameters and dataframe with original, approximated, and residual values
    return {'params': params_dict, 'df': df}

def approx_equation(t, p):
    # Ensure the parameter dictionary contains all required keys
    required_keys = {'const', 't', 'sin_wt', 'cos_wt'}
    if not required_keys.issubset(p):
        raise KeyError(f"Dictionary must contain: {required_keys}")

    w = 2 * np.pi / 365  # Annual frequency for sine/cosine terms

    # Compute the deterministic approximation equation:
    # constant + linear trend + seasonal sine + seasonal cosine
    approx_eq = p['const'] + p['t']*t + p['sin_wt']*np.sin(w*t) + p['cos_wt']*np.cos(w*t)

    return approx_eq

def preliminary_analysis(series):
    # Plot histogram of the data
    def histogram():
        plt.hist(series, bins=50, edgecolor='black')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title('Data')
        plt.show()

    # Perform Jarque–Bera test for normality
    def jb_test():
        jb_stat, jb_p, skew, kurtosis = jarque_bera(series)
        print(f"\nJarque–Bera: statistic={jb_stat:.4f}, p-value={jb_p:.4f}")
        print(f"Skewness = {skew:.4f}, Kurtosis = {kurtosis:.4f}")
        if jb_p > 0.05:
            print("Data compatible with normal distribution.")
        else:
            print(" Data not compatible with normal distribution.")  # Translated

    # Generate QQ plot to visually assess normality
    def qq_plot():
        qqplot(series, line='s')
        plt.title("QQ Plot - Normality test")
        plt.show()

    # Perform Augmented Dickey-Fuller test for stationarity
    def adf_test():
        adf_result = adfuller(series)
        print("ADF Statistic:", adf_result[0])
        print("p-value:", adf_result[1])
        if adf_result[1] < 0.05:
            print(" Stationary series.")
        else:
            print(" Non-stationary series.")  # Translated

    # Run all analysis steps
    histogram()
    jb_test()
    qq_plot()
    adf_test()

    # Compute and return mean and variance
    mean = np.mean(series)
    var = np.var(series)

    return mean, var

def extremes(df, var, quant=0.05):
    # Check if quantile is within valid range
    if quant > 0.5:
        print("Error: out of range quantile")
    else:
        copy = df.copy()  # Work on a copy to avoid modifying original dataframe

        # Extract year from datetime column
        copy["time"] = pd.to_datetime(copy["time"])
        copy["year"] = copy["time"].dt.year

        # Determine lower and upper quantiles
        lower_q = copy[var].quantile(quant)
        upper_q = copy[var].quantile(1 - quant)

        # Classify values as 'low', 'high', or 'normal' based on quantiles
        copy["class"] = np.where(copy[var] <= lower_q, "low",
                            np.where(copy[var] >= upper_q, "high", "normal"))

        # Count occurrences per year and class
        count = copy.groupby(["year", "class"]).size().unstack(fill_value=0)

        # Add total events per year
        count["total"] = count.sum(axis=1)

        # Print summary of extreme events
        print("Total extreme events per year:")
        print(count)

        # Plot the number of low and high extreme events per year
        count[["low", "high"]].plot(kind="bar", figsize=(12,6))
        plt.title(f"Tails events per year, {quant} quantile")
        plt.xlabel("Year")
        plt.ylabel("Frequency")
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

def correlation_analysis(series):
    #ACF
    plot_acf(series, lags=30)
    plt.title("Autocorrelation Function (ACF)")
    plt.show()
    #PACF
    plot_pacf(series, lags=30, method='ywm')
    plt.title("Partial Autocorrelation Function (PACF)")
    plt.show()

"""##**MODELLI**

###ARMA-GARCH
"""

def simulations_plotting(model, paths, test):
    # Create a daily date index for the year 2024
    index = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")

    plt.figure(figsize=(10,5))

    # Plot all simulated paths with light transparency
    for i in range(paths.shape[1]):
        plt.plot(index, paths[i,:], color='skyblue', alpha=0.3)

    # Plot the actual test series in solid blue
    plt.plot(index, test.values, color="blue")
    plt.xlabel('Future steps')
    plt.ylabel('Simulation value')
    plt.title(f"{città}, {model} 1000 simulations")
    plt.legend()
    plt.show()


def intervals_plotting(model, mean, lower, upper, test, approx):
    # Create a daily date index for the year 2024
    index = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")

    plt.figure(figsize=(12,5))

    # Plot the mean of simulations
    plt.plot(index, mean, color='yellow', label='Simulations mean')

    # Shade the area between the lower and upper percentile (confidence interval)
    plt.fill_between(index, lower, upper, color='yellow', alpha=0.3, label='5°–95° percentile')

    # Plot the difference between real test series and approximation
    plt.plot(index, (test-approx).values, color='blue', alpha=0.6, label='Real series', linewidth=2)

    plt.xlabel("Time")
    plt.ylabel("Forecast value")
    plt.title(f"{città}, statistics {model} vs real series")
    plt.legend()
    plt.show()

def best_BIC_ARMA(series, a, b):
    # Initialize the best BIC as infinity
    best_bic = np.inf
    best_order = None

    # Loop over all combinations of p (AR order) and q (MA order)
    for p in range(a + 1):
        for q in range(b + 1):
            try:
                # Fit ARMA model (implemented via ARIMA with d=0)
                model = ARIMA(series, order=(p, 0, q)).fit()

                # Update best model if BIC improves
                if model.bic < best_bic:
                    best_bic = model.bic
                    best_order = (p, q)
                    best_model = model
                    resid = model.resid  # Store residuals
            except:
                continue  # Skip invalid model combinations

    # Print summary of the best model
    print(f"Best model is ARMA{best_order} with BIC = {best_bic:.2f}")
    print(best_model.summary())

    # Plot the original series and fitted values
    plt.figure(figsize=(8, 4))
    plt.plot(series, label='Residuals', color="blue")
    plt.plot(best_model.fittedvalues, label='ARMA Estimated values', color='red')
    plt.legend()
    plt.title(f'Fit ARMA({best_order})')
    plt.show()

    # Return dictionary with model info, residuals, and fitted values
    ARMA_dict = {"Model": best_order, "Residuals": resid, "Estimate": best_model.fittedvalues}
    return ARMA_dict

def garch_tests(series):
    # Ljung–Box test on squared residuals to detect autocorrelation in volatility
    lb_test = acorr_ljungbox(series**2, lags=[10], return_df=True)
    lb_p = lb_test['lb_pvalue'].values[0]

    # Engle's ARCH test to detect autoregressive conditional heteroskedasticity
    arch_stat, arch_pvalue, _, _ = het_arch(series, nlags=12)

    # Print test results
    print("Ljung–Box (resid², lag=10) p-value:", round(lb_p, 4))
    print("Engle ARCH LM (lags=12) p-value:", round(arch_pvalue, 4))

    # Check if both tests indicate conditional heteroskedasticity (GARCH effects)
    if (arch_pvalue < 0.05) and (lb_p < 0.05):
        print("GARCH effects detected (conditional heteroskedasticity present).")
    else:
        print(" No GARCH effects detected (variance seems constant).")

    # Return True if GARCH effects are present, False otherwise
    return (arch_pvalue < 0.05) and (lb_p < 0.05)

def select_best_garch(returns, a=3, b=3, dist='t', criterion='bic'):
    # Initialize variables to store the best model
    best_val = np.inf
    best_order = None
    best_spec = None
    best_res = None

    # Loop over all combinations of p (ARCH order) and q (GARCH order)
    for p in range(a + 1):
        for q in range(b + 1):
            if p == 0 or q == 0:  # Skip combinations where either order is 0
                continue
            try:
                # Fit EGARCH model with zero mean
                am = arch_model(
                    returns, mean='Zero', vol='GARCH', p=p, q=q, dist=dist, rescale=True
                )
                res = am.fit(disp='off', show_warning=False)

                # Choose model based on BIC or AIC
                val = res.bic if criterion == 'bic' else res.aic
                if val < best_val:
                    best_val = val
                    best_order = (p, q)
                    best_spec = am
                    best_res = res
            except Exception as e:
                print(f"GARCH({p},{q}) failed: {e}")
                continue

    # Print the best model found
    print(f'Best GARCH{best_order}')

    # Store results in a dictionary
    GARCH_dict = {
        "Model": best_order,
        "Standard Residuals": best_res.std_resid,
        "Volatility": best_res.conditional_volatility
    }

    # Run GARCH diagnostic tests on standardized residuals
    garch_tests(best_res.std_resid)

    return GARCH_dict

def garch_estimate(series):
    # Perform correlation analysis on the series and its squared values
    correlation_analysis(series)
    correlation_analysis(series**2)

    # Check for presence of GARCH effects (conditional heteroskedasticity)
    if garch_tests(series):
        # If GARCH effects are present, select the best EGARCH model
        g_res = select_best_garch(series)

        # Check if standardized residuals from the GARCH model no longer show GARCH effects
        if not garch_tests(g_res["Standard Residuals"]):
            # Plot original residuals vs standardized residuals from GARCH
            plt.figure(figsize=(8,4))
            plt.plot(series, label='ARMA residuals', color="blue")
            plt.plot(g_res["Standard Residuals"], label='Standardized residuals', color='red')
            plt.legend()
            plt.title(f'Fit GARCH({g_res["Model"]})')
            plt.show()
        else:
            # If GARCH effects persist, model fit may not be sufficient
            print("Model isn't well fit")
    else:
        # No GARCH components needed if conditional heteroskedasticity is absent
        print("No GARCH components needed")

    return g_res  # Return dictionary with GARCH model results

def reconstruct_arma_garch(data, approx, arma_fit, garch_fit, plot=True):
  if plot:
    plt.plot(data)
    plt.plot(approx + arma_fit["Estimate"] + garch_fit["Standard Residuals"]*garch_fit["Volatility"])
    plt.show()
  ARMA_GARCH_MODEL = {"arma_order":arma_fit["Model"], "garch_order":garch_fit["Model"]}
  return ARMA_GARCH_MODEL

def dynamic_forecasting(series, test, ARMA_GARCH_model, approx, steps, pl1=1, pl2=1):
    # --- ARMA Fit ---
    arma_order = ARMA_GARCH_model["arma_order"]
    # Fit ARIMA model using ARMA order; ARMA(p,q) can be represented in ARIMA(p,0,q)
    arma_model = ARIMA(series, order=arma_order[1:] + (0,) + arma_order[1:])
    arma_fit = arma_model.fit()
    arma_resid = arma_fit.resid

    # Forecast ARMA component for the specified number of steps
    forecast_res = arma_fit.get_forecast(steps=steps)
    fc = forecast_res.predicted_mean

    # --- GARCH Fit ---
    garch_order = ARMA_GARCH_model["garch_order"]
    garch_or = list(garch_order)
    garch_model = arch_model(
        arma_fit.resid, mean="Zero", vol='Garch', p=garch_or[0], q=garch_or[1], dist="t"
    )
    garch_fit = garch_model.fit(disp="off")

    # Define time index for forecast
    fc.index = testing.index
    time = np.arange(steps)

    # --- Plot comparative forecast vs approximation ---
    if pl1 == 1:
        plt.figure(figsize=(10,6))
        plt.plot(test, label='Actual', color='blue')
        plt.plot(approx + fc, label='Mean Forecast', color='green')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Mean value forecasting')
        plt.legend()
        plt.show()

    # --- Nested function: generate ARMA+GARCH forecasts dynamically ---
    def arma_garch_forecasts(a, g, steps):
        forecasts = []
        vol_forecasts = []

        last_value = series.iloc[-1]
        last_resid = arma_fit.resid.iloc[-1]

        # Extract GARCH parameters
        params = garch_fit.params
        omega = params['omega']
        alpha_params = [params[k] for k in params.keys() if 'alpha' in k]
        beta_params  = [params[k] for k in params.keys() if 'beta' in k]
        nu = params.get('nu', None)
        p = len(alpha_params)
        q = len(beta_params)

        # Initialize history of residuals and variances for GARCH memory
        resid_history = list((arma_fit.resid.iloc[-p:] ** 2).values[::-1]) if p > 0 else []
        var_history   = list((garch_fit.conditional_volatility.iloc[-q:] ** 2).values[::-1]) if q > 0 else []

        for step in range(steps):
            # One-step ARMA forecast
            ar_forecast = arma_fit.forecast(steps=1).iloc[0]

            # GARCH(p,q) variance forecast: ω + Σ α_i ε²_(t−i) + Σ β_j σ²_(t−j)
            var_forecast = omega
            for i in range(p):
                var_forecast += alpha_params[i] * (resid_history[i] if i < len(resid_history) else 0)
            for j in range(q):
                var_forecast += beta_params[j] * (var_history[j] if j < len(var_history) else 0)

            vol_forecast = np.sqrt(var_forecast)

            # Simulate stochastic shock (t-distributed or normal)
            if nu is not None:
                shock = np.random.standard_t(df=nu) * vol_forecast
            else:
                shock = np.random.normal(0, vol_forecast)

            # Forecast value including ARMA + stochastic GARCH shock
            fc = ar_forecast + shock
            forecasts.append(fc)
            vol_forecasts.append(vol_forecast)

            # Update history for next step
            last_value = fc
            last_resid = fc - ar_forecast
            if p > 0:
                resid_history = [last_resid ** 2] + resid_history[:p - 1]
            if q > 0:
                var_history = [var_forecast] + var_history[:q - 1]

        return np.array(forecasts), np.array(vol_forecasts)

    # Generate forecasts and volatility estimates
    forecast_values, volatility_values = arma_garch_forecasts(arma_fit, garch_fit, steps)
    forecast_values = forecast_values + approx  # Add deterministic approximation
    forecast_df = pd.DataFrame(forecast_values, index=testing.index, columns=['Forecast'])
    forecast_df['Volatility Forecast'] = volatility_values

    # --- Plot forecast with confidence interval ---
    if pl2 == 1:
        plt.figure(figsize=(12, 6))
        plt.plot(testing.index, testing.values, label='Original data', color='blue')
        plt.plot(testing.index, forecast_df['Forecast'], label='Simulation ARMA + GARCH', color='red')

        plt.legend()
        plt.title(f'{città}, ARMA-GARCH Simulation for 2024')
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(testing.index, (testing-approx).values, label='Original data', color='blue')
        plt.plot(testing.index, forecast_df['Forecast']-approx, label='Simulation ARMA + GARCH', color='red')

        plt.legend()
        plt.title(f'{città}, ARMA-GARCH Simulation for 2024')
        plt.show()

    return forecast_df  # Return forecast dataframe with volatility

def simulate_arma_garch_vectorized(arma_fit, garch_fit, approx, steps, N, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # GARCH params
    params = garch_fit.params
    omega = params['omega']
    alpha_params = np.array([params[k] for k in params.keys() if 'alpha' in k])
    beta_params  = np.array([params[k] for k in params.keys() if 'beta' in k])
    nu = params.get('nu', None)

    p = len(alpha_params)
    q = len(beta_params)

    # init last eps (not squared) per-path for GARCH: shape (N, p)
    if p > 0:
        eps_lags = garch_fit.resid.iloc[-p:][::-1].values  # length p
        last_eps_garch = np.tile(eps_lags, (N, 1))         # (N, p)
    else:
        last_eps_garch = np.zeros((N, 0))

    # init last vars per-path for GARCH: shape (N, q)
    if q > 0:
        var_lags = (garch_fit.conditional_volatility.iloc[-q:] ** 2)[::-1].values
        last_vars = np.tile(var_lags, (N, 1))              # (N, q)
    else:
        last_vars = np.zeros((N, 0))

    # shocks
    if nu is not None:
        z = np.random.standard_t(df=nu, size=(steps, N))
        # standardize to variance 1
        if nu > 2:
            z = z / np.sqrt(nu / (nu - 2.0))
    else:
        z = np.random.randn(steps, N)

    forecasts = np.zeros((steps, N))
    vols = np.zeros((steps, N))

    # ARMA params
    mu = arma_fit.params.get('const', 0)
    ar_params = np.array([arma_fit.params[k] for k in arma_fit.params.index if 'ar.' in k])
    ma_params = np.array([arma_fit.params[k] for k in arma_fit.params.index if 'ma.' in k])
    p_ar = len(ar_params)
    q_ma = len(ma_params)

    # initialize ARMA lags per-path
    if p_ar > 0:
        # take last p_ar fitted values, reversed so last is first column
        last_x = np.tile(arma_fit.fittedvalues.iloc[-p_ar:][::-1].values, (N, 1))  # (N, p_ar)
    else:
        last_x = np.zeros((N, 0))

    if q_ma > 0:
        last_eps_arma = np.tile(arma_fit.resid.iloc[-q_ma:][::-1].values, (N, 1))   # (N, q_ma)
    else:
        last_eps_arma = np.zeros((N, 0))

    # simulation loop
    for t in range(steps):
        # GARCH variance: use eps^2
        var_t = np.full(N, omega)
        if p > 0:
            var_t += np.sum(alpha_params * (last_eps_garch ** 2), axis=1)
        if q > 0:
            var_t += np.sum(beta_params * last_vars, axis=1)

        var_t = np.maximum(var_t, 1e-12)
        sigma_t = np.sqrt(var_t)
        vols[t, :] = sigma_t

        # AR term and MA term per-path
        if p_ar > 0:
            ar_term = np.sum(ar_params * last_x, axis=1)      # (N,)
        else:
            ar_term = 0.0

        if q_ma > 0:
            ma_term = np.sum(ma_params * last_eps_arma, axis=1)  # (N,)
        else:
            ma_term = 0.0

        eps_t = z[t, :] * sigma_t   # (N,)
        x_t = mu + ar_term + ma_term + eps_t
        forecasts[t, :] = x_t

        # update ARMA lags per-path by shifting columns
        if p_ar > 0:
            last_x = np.column_stack([x_t, last_x[:, :-1]])
        if q_ma > 0:
            last_eps_arma = np.column_stack([eps_t, last_eps_arma[:, :-1]])

        # update GARCH lags
        if p > 0:
            last_eps_garch = np.column_stack([eps_t, last_eps_garch[:, :-1]])
        if q > 0:
            last_vars = np.column_stack([var_t, last_vars[:, :-1]])

    forecasts_no_approx = forecasts.copy()
    forecasts = forecasts + approx[:, None]

    return forecasts_no_approx, forecasts, vols





def multiple_dynamic_forecasting(series, test, ARMA_GARCH_model, approx, steps, N=1000, plots=1, seed=None):
    # Fit ARMA
    arma_order = ARMA_GARCH_model["arma_order"]
    arma_model = ARIMA(series, order=arma_order[1:] + (0,) + arma_order[1:])
    arma_fit = arma_model.fit()

    #Fit GARCH
    garch_order = ARMA_GARCH_model["garch_order"]
    garch_model = arch_model(arma_fit.resid, mean="Zero", vol='Garch', p=garch_order[0], q=garch_order[1], dist="t")
    garch_fit = garch_model.fit(disp='off')

    # Simulate multiple paths
    f_no_app, f_with_app, vols = simulate_arma_garch_vectorized(arma_fit, garch_fit, approx, steps, N)

    all_paths_no_approx = f_no_app.T
    all_paths_with_approx = f_with_app.T

    # compute ensemble stats
    mean_no_app = all_paths_no_approx.mean(axis=0)
    lower_no_app = np.percentile(all_paths_no_approx, 5, axis=0)
    upper_no_app = np.percentile(all_paths_no_approx, 95, axis=0)

    if plots:
        simulations_plotting(f"ARMA-GARCH ", all_paths_no_approx, test-approx)
        simulations_plotting(f"ARMA-GARCH with trend", all_paths_with_approx, test)
        intervals_plotting("ARMA-GARCH", mean_no_app, lower_no_app, upper_no_app, test, approx)

    return all_paths_no_approx, all_paths_with_approx, vols

"""def forecast_accuracy(forecast, actual):
    mask = actual != 0
    mape = np.mean(np.abs(forecast[mask] - actual[mask])/np.abs(actual[mask]))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast[mask] - actual[mask])/actual[mask])   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins[mask]/maxs[mask])         # minmax
    acf1 = acf(forecast-actual)[1]                      # ACF1
    results={'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1,
            'corr':corr, 'minmax':minmax}
    df_results = pd.DataFrame([results], index = ["city"])
    return df_results

###ORNSTEIN-UHLENBECK
"""

def best_Ornstein_Uhlenbeck(series, trend, pl1=1, pl2=1):

    x = series.to_numpy()
    t = np.arange(len(x))

    # Preparing data for OU estimation (still estimates sigma and eta)
    data = [(t, x)]
    est = OrnsteinUhlenbeckEstimator_2(data, n_it=3)

    # Extract parameters
    eta = float(est.eta)
    sigma = np.sqrt(est.sigma_sq())
    p_value = (est.eta_p)

    # Generate Gaussian noise
    std_eps = sigma * np.sqrt((1 - np.exp(-2 * eta)) / (2 * eta))
    noise = np.random.normal(0, std_eps, size=len(x))

    # Reconstruct OU process with time-dependent trend
    x_rec = np.zeros_like(x)
    x_rec[0] = x[0] + trend[0]
    for i in range(1, len(x)):
        dt = t[i] - t[i-1]
        x_rec[i] = trend[i] + (x_rec[i-1] - trend[i-1]) * np.exp(-eta * dt) + noise[i-1]

    #Plots 
    if pl1 == 1:
        plt.plot(t, series, label="Observed residuals")
        plt.plot(t, x_rec-trend, label="Reconstructed OU ")
        plt.xlabel("Days")
        plt.ylabel("X_t")
        plt.legend()
        plt.show()
    if pl2 == 1:
        plt.plot(t, trend + series, label="Observed series")
        plt.plot(t,  x_rec, label="Reconstructed OU + trend")
        plt.xlabel("Days")
        plt.ylabel("X_t")
        plt.legend()
        plt.show()

    # Return estimated parameters (eta and sigma)
    df_params = pd.DataFrame([{"eta": eta, "sigma": sigma, "Variance": est.variance, "pval":p_value}], index=["city"])
    return df_params

def OU_forecasts(series, model, testing, steps, approx, dt=1.0, plots=1):
    t = np.arange(len(series))  # Time index
    x_last = series.iloc[-1]    # Last observed value

    # Extract OU parameters from the model
    eta = float(model["eta"].iloc[0])   # Speed of mean reversion
    #mu = float(model["mu"].iloc[0])     # Long-term mean
    var = float(model["Variance"].iloc[0])  # Stationary variance

    forecast_df = pd.DataFrame()
    x_forecast_stoch = np.zeros(steps + 1)
    x_forecast_stoch[0] = x_last + det["approx"].iloc[-1]

    # Standard deviation of stochastic increment for each step
    sigma = np.sqrt(var * (1 - np.exp(-2 * eta * dt)))

    # Generate stochastic OU forecasts
    for i in range(1, steps + 1):
        epsilon = np.random.normal(0, sigma)  # Random shock
        x_forecast_stoch[i] = approx[i-1] + (x_forecast_stoch[i - 1] - approx[i-1]) * np.exp(-eta * dt) + epsilon

    # Store stochastic forecast (without deterministic approximation)
    forecast_df["Forecast"] = x_forecast_stoch[1:]

    # Compute residuals (observed minus deterministic approximation)
    residuals = testing - approx

    # Forecast including deterministic approximation
    forecast_df["Stochastic Forecast"] = -approx + x_forecast_stoch[1:]

    # Nested plotting function 
    def plot(series, forec, deter=""):
        plt.figure(figsize=(10,5))
        plt.plot(series, color="blue")  # Observed or residual series
        plt.plot(series.index, forec, label=f"{deter} Simulation for 2024", color="green")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"{città}, OU simulation")
        plt.legend()
        plt.show()

    # Optional plots
    if plots:
        plot(residuals, forecast_df["Stochastic Forecast"], "Stochastic")
        plot(testing, forecast_df["Forecast"], "Total")

    return forecast_df  # Return dataframe with stochastic OU forecasts

def multiple_OU_forecasts(series, test, model, approx, steps=366, N=1000, plots=1):
    # --- Initialize arrays to store all simulation paths ---
    all_paths = np.zeros((steps, N))         # With deterministic approximation
    all_paths_no_app = np.zeros((steps, N))  # Stochastic paths only

    #  Generate N independent OU forecast simulations 
    for i in range(N):
        forecast_df = OU_forecasts(series, model, test, steps, approx, plots=0)
        all_paths[:, i] = forecast_df["Forecast"].values           # With approx
        all_paths_no_app[:, i] = forecast_df["Stochastic Forecast"].values  # Stochastic only

    #Compute simulation statistics 
    mean_path = all_paths_no_app.mean(axis=1)    # Mean across simulations
    std_path = all_paths.std(axis=1)            # Standard deviation
    p05 = np.percentile(all_paths_no_app, 5, axis=1)   # 5th percentile
    p95 = np.percentile(all_paths_no_app, 95, axis=1)  # 95th percentile

    #Optional plotting 
    if plots:
        # Plot all stochastic paths without approximation
        simulations_plotting("OU", all_paths_no_app.T, testing - approx)
        # Plot all paths including deterministic approximation
        simulations_plotting("OU", all_paths.T, testing)
        # Plot mean path with 5–95% percentile intervals
        intervals_plotting("OU", mean_path, p05, p95, test, approx)

    # Return arrays of all paths (stochastic only and with approximation) 
    return all_paths_no_app.T, all_paths.T

"""###OU + JUMPS

"""

def jumps_identification(series, k=3, max_iter=2, use_mad=True, dispersion_index=1.5, plots=1):
    s = series.copy()  # Copy series to avoid modifying original
    jump_idx = np.array([], dtype=int)  # Stores indices of all identified jumps
    prev_idx = np.array([], dtype=int)  # Placeholder for previous iteration

    #Iterative jump detection 
    for it in range(max_iter):
        diffs = s.diff().dropna()  # Compute differences between consecutive points

        # Compute threshold for jump detection
        if use_mad:
            mad = np.median(np.abs(diffs - np.median(diffs)))  # Median Absolute Deviation
            thr = k * 1.4826 * mad    # MAD scaled to approximate std
        else:
            thr = np.abs(diffs).quantile(0.995)
# Standard deviation-based threshold

        # Identify indices where absolute differences exceed threshold
        idx = np.where(np.abs(diffs) > thr)[0] + 1  # shift by 1 to match s index

        # Stop if no new jumps found
        if not len(idx):
            print(f"No jumps found at iteration {it+1}. Stopping.")
            break

        new_jumps = np.setdiff1d(idx, jump_idx)
        if len(new_jumps) == 0:
            print(f"Converged after {it+1} iterations.")
            break

        # Add new jumps and interpolate the series to remove jumps
        jump_idx = np.union1d(jump_idx, idx)
        s.iloc[idx] = np.nan
        s = s.interpolate(method='linear', limit_direction='both')  # Linear interpolation

    # Separate positive and negative jumps 
    diffs_full = series.diff().fillna(0)
    jump_idx_pos = [i for i in jump_idx if diffs_full.iloc[i] > 0]
    jump_idx_neg = [i for i in jump_idx if diffs_full.iloc[i] < 0]

    # Inter-arrival times in days 
    jump_times = s.index[jump_idx]
    inter_arrivals_days = np.diff(jump_times.values).astype('timedelta64[D]').astype(float)
    mean_ia = np.mean(inter_arrivals_days) if len(inter_arrivals_days) > 0 else np.nan
    var_ia = np.var(inter_arrivals_days) if len(inter_arrivals_days) > 0 else np.nan

    # Optional plots 
    if plots:
        # Plot original series and identified jumps
        plt.figure(figsize=(10,4))
        plt.plot(series, linewidth=0.5, color='blue', alpha=0.5)
        plt.scatter(s.index[jump_idx], s.iloc[jump_idx], color='red', label='Jumps')
        plt.legend()
        plt.title('Data and identified jumps')
        plt.show()

        # Plot positive vs negative jumps
        plt.figure(figsize=(10,5))
        plt.plot(series, linewidth=0.8, color='gray', alpha=0.7)
        plt.scatter(series.index[jump_idx_pos], series.iloc[jump_idx_pos], color='green', s=40, label='Jump +', alpha=0.8)
        plt.scatter(series.index[jump_idx_neg], series.iloc[jump_idx_neg], color='blue', s=40, label='Jump -', alpha=0.8)
        plt.title('Positive and negative jumps')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Histogram of inter-arrival times with exponential fit
        if len(inter_arrivals_days) > 0:
            plt.figure(figsize=(6,4))
            plt.hist(inter_arrivals_days, bins='auto', color='blue', density=True, alpha=0.5, label='Empirical')
            x = np.linspace(0, max(inter_arrivals_days)+1, 100)
            plt.plot(x, expon(scale=mean_ia).pdf(x), 'r--', label='Expected exponential distribution')
            plt.title('Interval distribution')
            plt.legend()
            plt.show()

    # Prepare output for Hawkes modeling
    t0 = s.index[0]
    jump_times_days = (s.index[jump_idx] - t0).astype(float)  # Convert to days
    marks = np.abs(series.diff().iloc[jump_idx].values)       # Jump magnitudes
    print(f"Salti : {len(jump_idx)}, positivi {len(jump_idx_pos)}, negativi {len(jump_idx_neg)}")

    return jump_idx, s, jump_idx_pos, jump_idx_neg, jump_times_days, marks

def OU_with_jumps(series, trend, dt=1.0, pl1=0, pl2=0, plot_simulation=1):

    s = series.copy()
    approx = trend.copy()
    N = len(s)

    #  Identify jumps 
    jumps, clean, _, _, _, _ = jumps_identification(s, use_mad = False, plots=0)

    # Prepare trend without jumps 
    trend_clean = approx.drop(index=approx.index[jumps])

    # Interpolate trend to restore continuity
    trend_interpolated = approx.copy()
    trend_interpolated.iloc[jumps] = np.nan
    trend_interpolated = trend_interpolated.interpolate(method="linear")

    # OU ESTIMATION ON RESIDUALS (clean only) 
    df_params = best_Ornstein_Uhlenbeck(
        clean,
        trend_interpolated,
        pl1=pl1,
        pl2=pl2
    )

    eta = df_params["eta"].iloc[0]
    sigma = df_params["sigma"].iloc[0]
    ou_std = sigma

    #  Jump increments estimation
    ou_increments = np.diff(clean)
    jump_increments = np.diff(s.to_numpy())[jumps - 1] - ou_increments[jumps - 1]
    jump_increments -= np.mean(jump_increments)

    p_jump = len(jump_increments) / N
    λ = -np.log(1 - p_jump)
    kde_jump = gaussian_kde(jump_increments, bw_method=0.3)

    # Simulation OU + jumps 
    x_rec = np.zeros(N)
    x_rec[0] = s.iloc[0] + trend[0]

    trend_full = trend_interpolated.to_numpy()

    for i in range(1, N):

        ou_increment = (trend_full[i] - x_rec[i-1]) * (1 - np.exp(-eta * dt)) \
                        + ou_std * np.random.randn()

        jump_val = 0
        if np.random.rand() < 1 - np.exp(-λ * dt):
            jump_val = kde_jump.resample(1)[0][0]

        x_rec[i] = x_rec[i - 1] + ou_increment + jump_val

    #  Plot
    if plot_simulation:
        plt.figure(figsize=(10,4))
        plt.plot(s.index, s.values+trend, label="Original series")
        #plt.plot(s.index, trend_full, label="Trend (interpolated)", color='green')
        plt.plot(s.index, x_rec, '--', label="OU + jumps simulated")
        #plt.scatter(s.index[jumps], s.values[jumps], color='red', label="Jumps")
        plt.legend()
        plt.show()

    # --- Return ---
    return {
        'OU': df_params,
        'eta': eta,
        'sigma': sigma,
        'trend': trend_full,
        'lambda': λ,
        'kde_jump': kde_jump,
        'clean_series': clean,
        'jump_idx_init': jumps,
        'simulated_series': x_rec
    }

def simulate_OU_jumps(series, testing, OU_jump_model, approx, N=366, dt=1.0, plot=1):
    # Extract OU parameters 
    ou_params = OU_jump_model['OU']
    eta, sigma_sq =  ou_params['eta'], ou_params['sigma']

    # Jump intensity and magnitude distribution
    λ = OU_jump_model['lambda']              # Jump intensity per step
    kde_jump = OU_jump_model['kde_jump']     # KDE of jump magnitudes

    # Initialize simulated series
    x_rec = np.zeros(N + 1)
    x_rec[0] = series.iloc[-1] + det["approx"].iloc[-1] # Start from last observed value
    ou_std = np.sqrt(sigma_sq * (1 - np.exp(-2 * eta * dt)) / (2 * eta))  # OU increment std

    # Simulate OU + jumps 
    for i in range(1, N + 1):
        # OU increment (mean-reverting + Gaussian noise)
        ou_increment = (approx[i-1] - x_rec[i - 1]) * (1 - np.exp(-eta * dt)) + ou_std * np.random.randn()

        # Jump component: occurs with Poisson probability
        jump_val = 0
        if np.random.rand() < 1 - np.exp(-λ * dt):
            jump_val = kde_jump.resample(1)[0][0]  # Sample jump magnitude from KDE

        # Update process
        x_rec[i] = x_rec[i - 1] + ou_increment + jump_val

    #  Plotting function 
    def plots(simulated, reference):
        plt.figure(figsize=(10, 4))
        plt.plot(reference.index, simulated, label='Simulated path OU + Jumps', color="orange")
        plt.plot(reference.index, reference, label='Original series', color="blue")
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title('Future simulation OU + Jumps')
        plt.legend()
        plt.show()

    #  Optional plots 
    if plot:
        # Plot residual simulation (without deterministic approximation)
        plots(x_rec[1:] - approx, testing - approx)
        # Plot simulation with deterministic approximation added
        plots( x_rec[1:], testing)

    # Return simulated series excluding the initial value
    return x_rec[1:]

def multiple_OU_jumps(series, test, OU_jump_model, approx,
    N=366, n_sim=1000, dt=1.0, jump_threshold=1.0, plots=True):

    # Extract OU parameters 
    ou_params = OU_jump_model['OU']
    eta, sigma_sq = ou_params['eta'].iloc[0], ou_params['sigma'].iloc[0]
    ou_std = np.sqrt(sigma_sq * (1 - np.exp(-2 * eta * dt)) / (2 * eta))

    # Extract jump parameters 
    λ = OU_jump_model['lambda']            # Jump intensity per step
    kde_jump = OU_jump_model['kde_jump']   # KDE of jump magnitudes

    #Initialize simulated paths 
    x_rec = np.zeros((n_sim, N + 1))
    x_rec[:, 0] = series.iloc[-1] + det["approx"].iloc[-1]          # Start from last observed value
    approx = np.asarray(approx)

    # Simulate OU + jumps
    for t in range(1, N + 1):
        # OU increment: mean-reverting Gaussian noise
        ou_increment = (approx[t-1] - x_rec[:, t-1]) * (1 - np.exp(-eta*dt)) + ou_std * np.random.randn(n_sim)

        # Jump occurrence: Bernoulli trial with Poisson probability
        jump_mask = np.random.rand(n_sim) < min(1 - np.exp(-λ*dt), jump_threshold)
        jump_vals = np.zeros(n_sim)
        if jump_mask.any():
            jump_vals[jump_mask] = kde_jump.resample(jump_mask.sum()).flatten()

        # Update process
        x_rec[:, t] = x_rec[:, t-1] + ou_increment + jump_vals

    # Compute residuals vs trend approximation 
    sims = x_rec[:, 1:]                      # Simulated paths without initial value
    sims_no_approx = sims - approx       # Residuals vs trend
    res_mean = sims_no_approx.mean(axis=0)
    res_p05 = np.percentile(sims_no_approx, 5, axis=0)
    res_p95 = np.percentile(sims_no_approx, 95, axis=0)

    #  Optional plotting 
    if plots:
        simulations_plotting("OU + jumps", sims_no_approx, test - approx)       # Residuals
        simulations_plotting("OU + jumps", sims, test)                              # Full paths
        intervals_plotting("OU + jumps", res_mean, res_p05, res_p95, test, approx)  # CI

    # --- Return simulated paths ---
    return sims_no_approx, sims

"""###OU + HAWKES
Q
"""

def marked_hawkes_loglik(params, event_times, marks, T):
    """
    Compute negative log-likelihood for a marked Hawkes process.

    """
    mu, a, gamma_p, beta = params
    if mu <= 0 or a < 0 or beta <= 0:
        return 1e20  # Penalize invalid parameters

    n = len(event_times)
    lam = np.zeros(n)

    # Compute conditional intensity at each event
    for i in range(n):
        if i == 0:
            lam[i] = mu
        else:
            dt = event_times[i] - event_times[:i]
            lam[i] = mu + np.sum(a * (marks[:i] ** gamma_p) * np.exp(-beta * dt))
        if lam[i] <= 0:
            return 1e20  # Invalid intensity

    # Integral of intensity over [0, T]
    integral = mu * T + np.sum((a * (marks ** gamma_p)) / beta * (1 - np.exp(-beta * (T - event_times))))

    ll = -integral + np.sum(np.log(lam))  # Log-likelihood
    return -ll  # Return negative log-likelihood for minimization


def fit_marked_hawkes(event_times, marks, T, initial=None):
    """
    Fit a marked Hawkes process via MLE.
    """
    if initial is None:
        mu0 = max(1e-3, len(event_times) / T * 0.8)
        a0 = 0.1
        gamma0 = 1.0
        beta0 = 1.0
        initial = np.array([mu0, a0, gamma0, beta0])

    bounds = [(1e-9, None), (1e-9, None), (0.0, 5.0), (1e-6, None)]
    res = minimize(marked_hawkes_loglik, initial, args=(event_times, marks, T),
                   method='L-BFGS-B', bounds=bounds)
    return {'params': res.x, 'success': res.success, 'message': res.message, 'fun': res.fun}


def Hawkes_by_ogata_marked(T, mu_H, a_H, gamma_H, beta_H, mark_sampler, plot=True, seed=None):
    """
    Simulate a marked Hawkes process using Ogata's thinning algorithm.
    """
    rng = np.random.default_rng(seed)
    events, marks = [], []
    t = 0.0
    times = [0.0]
    intensities = [mu_H]

    while t < T:
        # Compute current intensity λ(t)
        if len(events) == 0:
            lam_t = mu_H
        else:
            dt = t - np.array(events)
            lam_t = mu_H + np.sum((a_H * (np.array(marks) ** gamma_H)) * np.exp(-beta_H * dt))

        if lam_t <= 0:
            t += 1e-3
            continue

        # Candidate event time using thinning
        u = rng.uniform()
        w = -np.log(u) / lam_t
        t += w

        # Candidate intensity
        if len(events) == 0:
            lam_candidate = mu_H
        else:
            dt = t - np.array(events)
            lam_candidate = mu_H + np.sum((a_H * (np.array(marks) ** gamma_H)) *  np.exp(-beta_H * dt))

        # Acceptance step
        D = rng.uniform()
        if D <= lam_candidate / lam_t and t < T:
            events.append(t)
            mark_old = mark_sampler()
            mark_new = mark_old + rng.normal(loc=0, scale=0.1)  # small noise
            marks.append(mark_new)

        times.append(t)
        intensities.append(lam_candidate)

    events, marks = np.array(events), np.array(marks)

    # Optional plotting
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(times, intensities, label=r'$\lambda(t)$', color='blue', lw=2)
        plt.vlines(events, ymin=0, ymax=max(intensities), color='red', lw=1.2, alpha=0.1, label='Hawkes events')
        plt.xlabel('Time t')
        plt.ylabel('Intensity λ(t)')
        plt.title('Marked Hawkes Simulation (Ogata thinning)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    return events, marks

def OU_Hawkes(series, trend=None, dt=1.0, hawkes_graph=True, plots=True, scatter=False, seed=None):
    rng = np.random.default_rng(seed)
    s = series.copy()
    N = len(s)

    # Identify jumps 
    jump_idx, clean, _, _, jump_times_days, marks = jumps_identification(s, use_mad =False, plots=0)

    # Trend handling 
    if trend is None:
        trend_full = s.copy()
    else:
        trend_interpolated = trend.copy()
        trend_interpolated.iloc[jump_idx] = np.nan
        trend_full = trend_interpolated.interpolate(method="linear").to_numpy()

    #OU estimation on clean series
    df_params = best_Ornstein_Uhlenbeck(clean, trend_full if trend is not None else None, pl1=0, pl2=0)
    eta = df_params["eta"].iloc[0]
    sigma = df_params["sigma"].iloc[0]
    ou_std = sigma

    #Jump increments 
    ou_increments = np.diff(clean)
    jump_increments = np.diff(s.to_numpy())[jump_idx - 1] - ou_increments[jump_idx - 1]
    jump_increments -= np.mean(jump_increments)
    kde_jump = gaussian_kde(jump_increments, bw_method=0.3)

    # Fit Hawkes 
    T = N
    fit = fit_marked_hawkes(jump_times_days, marks, T)
    mu_H, a_H, gamma_H, beta_H = fit["params"]

    # Simulate Hawkes events 
    mark_sampler = lambda: rng.choice(marks)
    events, sim_marks = Hawkes_by_ogata_marked(T, mu_H, a_H, gamma_H, beta_H, mark_sampler, plot=hawkes_graph, seed=seed)
    event_indices = np.floor(events).astype(int)
    event_indices = event_indices[event_indices < N]

    # Reconstruct OU + Hawkes series
    x_rec = np.zeros(N)
    x_rec[0] = s.iloc[0]+ trend[0]

    for i in range(1, N):
        # OU increment
        ou_increment = (trend[i-1] - x_rec[i-1]) * (1 - np.exp(-eta*dt)) + ou_std * rng.standard_normal()
        # Hawkes jumps
        jumps_idx = np.where((events >= i) & (events < i+1))[0]
        jump_val = np.sum(kde_jump.resample(len(jumps_idx))) if len(jumps_idx) > 0 else 0.0
        # Total update
        x_rec[i] = x_rec[i-1] + ou_increment + jump_val

    #Plots 
    if plots:
        plt.figure(figsize=(10,5))
        plt.plot(s.index, s.values + trend, alpha=0.6, label="Original series")
        plt.plot(s.index, x_rec , '--', alpha=0.7, label="OU + Hawkes simulated")
        if scatter:
            plt.scatter(s.index[event_indices], x_rec[event_indices] + eq, color="red", label="Hawkes events")
        plt.xlabel("Time")
        plt.title("OU + Hawkes reconstruction")
        plt.legend()
        plt.show()

    # --- 9️⃣ Return ---
    ou_params = {'eta': eta, 'sigma': sigma, 'ou_std': ou_std}
    hawkes_params = {'mu': mu_H, 'a': a_H, 'gamma': gamma_H, 'beta': beta_H}
    return {
        "simulated_series": x_rec,
        "hawkes_events": events,
        "marks": sim_marks,
        "kde_jump": kde_jump,
        "ou_params": ou_params,
        "hawkes_params": hawkes_params,
        "jump_idx_init": jump_idx,
        "clean_series": clean
    }

def simulate_OU_Hawkes_new_steps(series, testing, OU_Hawkes_model, approx=None, steps=366, scatter=0, hawkes_graph=1, plots=1, seed=None):
    rng = np.random.default_rng(seed)
    # --- OU parameters ---
    ou_params = OU_Hawkes_model["ou_params"]
    eta, ou_std =  ou_params['eta'], ou_params['ou_std']

    # --- Hawkes parameters ---
    hawkes_params = OU_Hawkes_model["hawkes_params"]
    mu_H, a_H, gamma_H, beta_H = hawkes_params['mu'], hawkes_params['a'], hawkes_params['gamma'], hawkes_params['beta']

    # --- Jump KDE ---
    kde_jump = OU_Hawkes_model['kde_jump']
    marks = OU_Hawkes_model['marks']

    # --- Simulate new Hawkes events ---
    mark_sampler = lambda: rng.choice(marks)
    hawkes_events_new, sim_marks_new = Hawkes_by_ogata_marked(
        T=steps, mu_H=mu_H, a_H=a_H, gamma_H=gamma_H, beta_H=beta_H,
        mark_sampler=mark_sampler, plot=hawkes_graph, seed=seed
    )

    event_indices_new = np.floor(hawkes_events_new).astype(int)
    event_indices_new = event_indices_new[event_indices_new < steps]

    # --- Initialize series ---
    x_rec = np.zeros(steps + 1)
    x_rec[0] = series.iloc[-1] + det["approx"].iloc[-1]  # Start from last observed value

    # --- OU + Hawkes simulation ---
    for i in range(1, steps + 1):
        # OU increment
        ou_increment = (approx[i-1] - x_rec[i-1]) * (1 - np.exp(-eta)) + ou_std * rng.standard_normal()

        # Hawkes jump increment
        jumps_idx = np.where((event_indices_new >= i) & (event_indices_new < i + 1))[0]
        jump_val = np.sum(kde_jump.resample(len(jumps_idx))) if len(jumps_idx) > 0 else 0.0

        # Update series
        x_rec[i] = x_rec[i-1] + ou_increment + jump_val

    x_rec = x_rec[1:]  # Remove initial placeholder

    # --- Plot ---
    if plots:
        plt.figure(figsize=(10,5))
        plt.plot(range(steps), x_rec, color="violet", label='Simulated OU + Hawkes')
        plt.plot(range(steps), testing, color="blue", label='Original residuals')
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"{città} OU + Hawkes simulation with trend")
        plt.legend()
        plt.show()


        plt.figure(figsize=(10,5))
        plt.plot(range(steps), x_rec - approx, color="violet", label='Simulated OU + Hawkes + approx')
        plt.plot(range(steps), testing-approx, color="blue", label='Original series')
        if scatter:
            plt.scatter(event_indices_new, x_rec[event_indices_new] + approx[event_indices_new], color='red', label='Hawkes events')
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"{città} OU + Hawkes simulation")
        plt.legend()
        plt.show()

    return x_rec, hawkes_events_new, sim_marks_new

def multiple_OU_Hawkes(series, test, OU_Hawkes_model, approx,
steps=366, n_sim=1000, scatter=0, hawkes_graph=0, plots=True, seed=None):

    rng = np.random.default_rng(seed)  # Initialize random number generator

    # OU parameters
    ou_params = OU_Hawkes_model["ou_params"]
    eta, ou_std = ou_params["eta"], ou_params["ou_std"]

    # Hawkes parameters 
    hawkes_params = OU_Hawkes_model["hawkes_params"]
    mu_H, a_H, gamma_H, beta_H = hawkes_params["mu"], hawkes_params["a"], hawkes_params["gamma"], hawkes_params["beta"]

    # KDE for jumps 
    kde_jump = OU_Hawkes_model["kde_jump"]
    marks = OU_Hawkes_model["marks"]

    # Initialize simulated paths 
    x_rec_all = np.zeros((n_sim, steps+1))
    x_rec_all[:, 0] = series.iloc[-1] + det["approx"].iloc[-1]  # Start from last observed value
    approx = np.asarray(approx)

    # Pre-generate Hawkes events for all simulations 
    all_event_indices = []
    for sim in range(n_sim):
        mark_sampler = lambda: rng.choice(marks)
        hawkes_events, _ = Hawkes_by_ogata_marked(
            T=steps, mu_H=mu_H, a_H=a_H, gamma_H=gamma_H, beta_H=beta_H,
            mark_sampler=mark_sampler, plot=0, seed=(None if seed is None else seed+sim)
        )
        event_idx = np.floor(hawkes_events).astype(int)
        event_idx = event_idx[event_idx < steps]
        all_event_indices.append(event_idx)

    # Simulate OU + Hawkes 
    for t in range(1, steps+1):
        # OU increment (vectorized)
        ou_increment = (approx[t-1] - x_rec_all[:, t-1]) * (1 - np.exp(-eta)) + ou_std * rng.standard_normal(n_sim)
        jump_increment = np.zeros(n_sim)

        # Add Hawkes jumps
        for j, events in enumerate(all_event_indices):
            n_jumps = np.sum(events == t)
            if n_jumps > 0:
                jump_increment[j] = kde_jump.resample(n_jumps).sum()

        # Update series
        x_rec_all[:, t] = x_rec_all[:, t-1] + ou_increment + jump_increment

    # --- Prepare outputs ---
    sims= x_rec_all[:,1:]
    sims_no_approx = x_rec_all[:,1:] - approx
    res = sims_no_approx  # residuals w.r.t approx
    mean_path = res.mean(axis=0)
    p05 = np.percentile(res, 5, axis=0)
    p95 = np.percentile(res, 95, axis=0)

    # --- Plotting ---
    if plots:
        simulations_plotting("OU + Hawkes", res, test[:steps] - approx[:steps])
        simulations_plotting("OU + Hawkes", sims, test[:steps])
        intervals_plotting("OU + Hawkes", mean_path, p05, p95, test[:steps], approx[:steps])

    return sims_no_approx, sims

"""##**CREAZIONE DEGLI INDICI E DELLE OPZIONI**"""

def hdd_index(T, T_base=18.0):
    """Heating Degree Days per periodo"""
    return np.sum(np.maximum(0, T_base - T))

def cdd_index(T, T_base=18.0):
    """Cooling Degree Days per periodo"""
    return np.sum(np.maximum(0, T - T_base))

def cat_index(T):
    """Cumulative Average Temperature"""
    return np.sum(T)

def hdd_option_payoff(T, K, option_type='call', T_base=18.0):
    """Payoff for HDD option"""
    index = hdd_index(T, T_base)
    if option_type == 'call':
        return np.maximum(index - K, 0.0)
    elif option_type == 'put':
        return np.maximum(K - index, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def cdd_option_payoff(T, K, option_type='call', T_base=18.0):
    """Payoff for CDD option"""
    index = cdd_index(T, T_base)
    if option_type == 'call':
        return np.maximum(index - K, 0.0)
    elif option_type == 'put':
        return np.maximum(K - index, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def cat_option_payoff(T, K, option_type='call'):
    """Payoff for CAT option"""
    index = cat_index(T)
    if option_type == 'call':
        return np.maximum(index - K, 0.0)
    elif option_type == 'put':
        return np.maximum(K - index, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

def rolling_temp_index(T, index='HDD', T_base=18.0, his=True):
    # Select the index function
    if index == 'HDD':
        func = lambda x: np.sum(np.maximum(0, T_base - x))  # Heating Degree Days
    elif index == 'CDD':
        func = lambda x: np.sum(np.maximum(0, x - T_base))  # Cooling Degree Days
    elif index == 'CAT':
        func = lambda x: np.sum(x)  # Catastrophic/average index
    else:
        raise ValueError("index must be 'HDD', 'CDD', or 'CAT'")

    T = np.asarray(T)
    n = len(T)
    window = 366  # Rolling window of 1 year

    if n < window:
        raise ValueError("Length must be at least 366")

    # Calculate rolling index
    results = np.array([func(T[i:i+window]) for i in range(n - window + 1)])
    mean_val = np.mean(results)
    std_dev = np.std(results)

    # Calculate percentiles
    q05, q25 = np.percentile(results, (5, 25))
    q50, q75 = np.percentile(results, (50, 75))
    q95 = np.percentile(results, 95)

    if his:
        # Plot histogram
        plt.figure(figsize=(8, 4))
        plt.hist(results, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean = {mean_val:.2f}")
        plt.axvline(q05, color='orange', linestyle='--', label=f"5th percentile = {q05:.2f}")
        plt.axvline(q95, color='orange', linestyle='--', label=f"95th percentile = {q95:.2f}")
        plt.title(f"Yearly '{index.upper()}' distribution")
        plt.xlabel(f"{index.upper()}")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return mean_val, q05, q25, q50, q75, q95, std_dev

def rolling_temp_option(T, K, index='HDD_call', T_base=18.0, his=True):
    # Select index function and option type
    if index.startswith('HDD'):
        func = hdd_option_payoff
    elif index.startswith('CDD'):
        func = cdd_option_payoff
    elif index.startswith('CAT'):
        func = cat_option_payoff
    else:
        raise ValueError("Invalid index")

    if index.endswith('call'):
        opt_type = "call"
    elif index.endswith('put'):
        opt_type = "put"
    else:
        raise ValueError("Index must end with 'call' or 'put'")

    # Rolling window calculation 
    T = np.asarray(T)
    n = len(T)
    window = 366
    if n < window:
        raise ValueError("Length must be at least 366")

    results = np.array([
        func(T[i:i+window], K, opt_type)
        for i in range(n - window + 1)
    ])

    mean_val = np.mean(results)
    std_dev = np.std(results)

    # Calculate percentiles
    q05, q25 = np.percentile(results, (5, 25))
    q50, q75 = np.percentile(results, (50, 75))
    q95 = np.percentile(results, 95)

    if his:
        # Plot histogram
        plt.figure(figsize=(8, 4))
        plt.hist(results, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(mean_val, color='red', linestyle='--', label=f"Mean = {mean_val:.2f}")
        plt.axvline(q05, color='orange', linestyle='--', label=f"5th percentile = {q05:.2f}")
        plt.axvline(q95, color='orange', linestyle='--', label=f"95th percentile = {q95:.2f}")
        plt.title(f"Yearly '{index.upper()}' distribution")
        plt.xlabel(f"{index.upper()}")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return mean_val, q05, q25, q50, q75, q95, std_dev

def burn_analysis(time, series, idx):
    # Select the appropriate index function
    def index_func(name):
        return {"HDD": hdd_index, "CDD": cdd_index, "CAT": cat_index}[name]

    func = index_func(idx)

    # Create a pandas Series with datetime index
    s = pd.Series(series, index=pd.to_datetime(time))

    # Apply the index function to each year
    results_per_year = series.groupby(s.index.year).apply(func)

    # Calculate the mean of yearly results
    overall_mean = results_per_year.mean()

    return overall_mean

"""##**SIMULAZIONI MONTECARLO**"""

def simmulate_models(series, test, approx,
                    AR_G=None, OU=None, OU_J=None, OU_H=None,
                    ag=True, ou=True, ou_j=True, ou_h=True, steps=366, n_sim=10000):
    # Call all model functions and store results in a dictionary
    raw_results = {}

    if ag:
        raw_results["ARMA-GARCH"] = multiple_dynamic_forecasting(series, test, AR_G, approx, steps, n_sim, plots=0)
    if ou:
        raw_results["OU"] = multiple_OU_forecasts(series, test, OU, approx, steps, n_sim, plots=0)
    if ou_j:
        raw_results["OU + Jump"] = multiple_OU_jumps(series, test, OU_J, approx, steps, n_sim, plots=0)
    if ou_h:
        raw_results["OU + Hawkes"] = multiple_OU_Hawkes(series, test, OU_H, approx, steps, n_sim, plots=0)

    # Separate tuples into two distinct dictionaries
    residuals = {key: val[0] for key, val in raw_results.items()}
    with_trend = {key: val[1] for key, val in raw_results.items()}

    return residuals, with_trend

def simulate_models(series, test, approx,
                    AR_G=None, OU=None, OU_J=None, OU_H=None,
                    ag=True, ou=True, ou_j=True, ou_h=True, steps=366, n_sim=10000):
    # Call all model functions and store results in a dictionary
    raw_results = {}

    if ag:
        raw_results["ARMA-GARCH"] = multiple_dynamic_forecasting(series, test, AR_G, approx, steps, n_sim, plots=0)
    if ou:
        raw_results["OU"] = multiple_OU_forecasts(series, test, OU, approx, steps, n_sim, plots=0)
    if ou_j:
        raw_results["OU + Jump"] = multiple_OU_jumps(series, test, OU_J, approx, steps, n_sim, plots=0)
    if ou_h:
        raw_results["OU + Hawkes"] = multiple_OU_Hawkes(series, test, OU_H, approx, steps, n_sim, plots=0)

    # Create date index from 1-11-2023 to 31-10-2024
    date_index = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")

    residuals = {key: pd.DataFrame(val[0].T, index=date_index) for key, val in raw_results.items()}
    with_trend = {key: pd.DataFrame(val[1].T, index=date_index) for key, val in raw_results.items()}


    return residuals, with_trend

def MonteCarlo_indexes(models, time, main, test, steps=366, n_sim=10000, plot=True):
    # ===  Generic function to select HDD/CDD/CAT index === #
    def index_func(name):
        return {"HDD": hdd_index, "CDD": cdd_index, "CAT": cat_index}[name]

    # Compute progressive mean of simulated paths
    def compute_progressive_mean(simulations, idx_name):
        func = index_func(idx_name)
        values = np.array([func(path) for path in simulations])
        return np.cumsum(values) / np.arange(1, len(simulations) + 1)

    # ===  Compute simulated and actual indices === #
    indices = ["HDD", "CDD", "CAT"]
    simulated = {
        model_name: {idx: compute_progressive_mean(sim, idx) for idx in indices}
        for model_name, sim in models.items()
    }

    actual = {idx: index_func(idx)(test) for idx in indices}

    # Dictionary where we will store percent errors
    percent_errors = {idx: {} for idx in indices}

    #Unified plotting function
    def convergence_plot(data_dict, actual_val, index):
        mean_val, q05, q25, q50, q75, q95, std_dev = rolling_temp_index(main, index=index, T_base=18.0, his=0)
        x = np.arange(n_sim)

        plt.figure(figsize=(12, 6))
        plt.plot([actual_val]*n_sim, label='Actual', color='black', linestyle='--', linewidth=2)

        for model_name, series in data_dict.items():
            plt.plot(series, label=model_name)

        plt.plot([mean_val]*n_sim, '--', color="yellow", label=f'{index} Mean')
        plt.plot([q50]*n_sim, '--', color='violet', label=f'{index} Median')

        plt.fill_between(x, [mean_val+2*std_dev]*n_sim, [mean_val-2*std_dev]*n_sim,
                         color='orange', alpha=0.1, label='5–95% Percentiles')
        plt.fill_between(x, [mean_val+std_dev]*n_sim, [mean_val-std_dev]*n_sim,
                         color='skyblue', alpha=0.2, label='25–75% Percentiles')

        plt.title(f"{città}, Models Convergence Comparison – {index}")
        plt.xlabel("Simulation")
        plt.ylabel("Value")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{città}, Models Convergence Comparison – {index}")
        plt.show()

    #Calculate percent error and optionally plot 
    for idx in indices:
        for model_name in models:
            converged_value = simulated[model_name][idx][-1]   # last progressive mean value
            true_value = actual[idx]

            percent_error = 100 * (converged_value - true_value) / true_value
            percent_errors[idx][model_name] = percent_error

        if plot:
            convergence_plot({m: simulated[m][idx] for m in simulated}, actual[idx], idx)

    return percent_errors

def MonteCarlo_option_payoffs(models, time, main, test, steps=366, n_sim=10000, plot=True, option_type='call'):
    import numpy as np
    import matplotlib.pyplot as plt

    # Payoff functions selection 
    def payoff_func(name):
        funcs = {
            "HDD_call": lambda path, K, option_type: hdd_option_payoff(path, K, option_type="call"),
            "CDD_call": lambda path, K, option_type: cdd_option_payoff(path, K, option_type="call"),
            "CAT_call": lambda path, K, option_type: cat_option_payoff(path, K, option_type="call"),
            "HDD_put":  lambda path, K, option_type: hdd_option_payoff(path, K, option_type="put"),
            "CDD_put":  lambda path, K, option_type: cdd_option_payoff(path, K, option_type="put"),
            "CAT_put":  lambda path, K, option_type: cat_option_payoff(path, K, option_type="put")
        }
        return funcs[name]

    # Compute progressive mean of simulated payoffs 
    def compute_progressive_payoff(simulations, idx_name, K, option_type):
        func = payoff_func(idx_name)
        values = np.array([func(path, K, option_type=option_type) for path in simulations])
        return np.cumsum(values) / np.arange(1, len(simulations) + 1)

    # Define indices and compute strikes 
    indices = ["HDD", "CDD", "CAT"]
    options = ["HDD_call", "CDD_call","CAT_call", "HDD_put","CDD_put", "CAT_put"]

    # Compute mean (strike) for each index
    base_strikes = {}
    for idx in indices:
        mean_val = rolling_temp_index(main, index=idx, T_base=18.0, his=0)[0]
        base_strikes[idx] = mean_val

    # Create a strike dictionary for each option
    strikes = {opt: base_strikes[opt.split('_')[0]] for opt in options}

    #  Compute simulated and actual payoffs 
    simulated = {
        model_name: {
            opt: compute_progressive_payoff(sim, opt, K=strikes[opt], option_type=option_type)
            for opt in options
        }
        for model_name, sim in models.items()
    }

    actual = {
        opt: payoff_func(opt)(test, K=strikes[opt], option_type=option_type)
        for opt in options
    }

    # Compute errors for convergence  #
    convergence_errors = {}
    for opt in options:
        convergence_errors[opt] = {}
        for model_name, series in simulated.items():
            final_sim = series.iloc[ -1 ]  # Ultimo valore della media progressiva
            if actual[opt] != 0:
                error = 100 * (final_sim - actual[opt]) / actual[opt]
            else:
                error = f"{final_sim - actual[opt]:.4f}*"  # Asterisco per indicare errore assoluto
            convergence_errors[opt][model_name] = error

    if plot:
        def convergence_plot(data_dict, K, actual, index):
            mean_val, q05, q25, q50, q75, q95, std_dev = rolling_temp_option(main, K, index=index, T_base=18.0, his=0)
            x = np.arange(n_sim)

            plt.figure(figsize=(12, 6))
            plt.plot([actual]*n_sim, label='Actual Payoff', color='black', linestyle='--', linewidth=2)

            for model_name, series in data_dict.items():
                plt.plot(series, label=model_name)

            plt.plot(mean_val, color='yellow', linestyle='--', label='Mean value')
            plt.plot(q50, color='violet', linestyle='--', label='Median')
            plt.fill_between(x, [mean_val+2*std_dev]*n_sim, [mean_val-2*std_dev]*n_sim, color='orange', alpha=0.1, label='5–95% Percentiles')
            plt.fill_between(x, [mean_val+std_dev]*n_sim, [mean_val-std_dev]*n_sim, color='skyblue', alpha=0.2, label='25–75% Percentiles')

            plt.title(f"Monte Carlo Option Payoff Convergence – {index}")
            plt.xlabel("Simulation")
            plt.ylabel("Payoff Value")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.ylim(bottom=0)
            plt.tight_layout()
            plt.show()

        for opt in options:
            convergence_plot({m: simulated[m][opt] for m in simulated}, strikes[opt], actual[opt], opt)

    # === 7. Payoff vs Strike plotting function (facoltativo) === #
    if plot:
        def payoff_vs_strike_plot(models, opt_name, option_type):
            func = payoff_func(opt_name)
            idx = opt_name.split('_')[0]

            mean_val, _, _, _, _, _, std_val = rolling_temp_index(main, index=idx, T_base=18.0, his=0)
            K_values = np.linspace(mean_val - std_val, mean_val + std_val, 50)

            plt.figure(figsize=(10, 6))
            for model_name, sim in models.items():
                payoffs = []
                for K in K_values:
                    prog = compute_progressive_payoff(sim, opt_name, K, option_type)
                    payoffs.append(prog[-1])
                plt.plot(K_values, payoffs, label=model_name)

            actual_payoffs = [func(test, K, option_type=option_type) for K in K_values]
            plt.plot(K_values, actual_payoffs, color='violet', linestyle='--', linewidth=2.2, label='Actual realization')
            plt.axvline(mean_val, color='black', linestyle=':', linewidth=1.5, label="Mean strike")

            plt.title(f"Expected Payoff vs Strike – {opt_name}")
            plt.xlabel("Strike (K)")
            plt.ylabel("Expected Payoff (final Monte Carlo mean)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.ylim(bottom=0)
            plt.tight_layout()
            plt.show()

        for opt in options:
            payoff_vs_strike_plot(models, opt, option_type=option_type)

    # === 8. Return convergence errors === #
    return convergence_errors

"""##BACKTESTING

"""

def backtest_model(real_series, simulations):

    y = np.asarray(real_series)
    sims = np.asarray(simulations)
    n_sim, T = sims.shape

    # Ensemble mean and std
    sim_mean = sims.mean(axis=0)
    sim_std = sims.std(axis=0, ddof=0)

    # Errors vs ensemble mean
    mae_mean = np.mean(np.abs(y - sim_mean))
    rmse_mean = np.sqrt(np.mean((y - sim_mean)**2))
    bias_mean = np.mean(sim_mean - y)

    # Per-path errors
    abs_errors_per_path = np.mean(np.abs(sims - y[np.newaxis, :]), axis=1)
    sq_errors_per_path = np.sqrt(np.mean((sims - y[np.newaxis, :])**2, axis=1))
    mean_path_mae = abs_errors_per_path.mean()
    mean_path_rmse = sq_errors_per_path.mean()

    # Coverage within ±1σ and ±2σ
    within_1std = np.mean(((y >= sim_mean - sim_std) & (y <= sim_mean + sim_std)).astype(float))
    within_2std = np.mean(((y >= sim_mean - 2*sim_std) & (y <= sim_mean + 2*sim_std)).astype(float))

    # Empirical CRPS (time-averaged)
    term1 = np.mean(np.abs(sims - y[np.newaxis, :]), axis=0)
    sorted_sims = np.sort(sims, axis=0)
    idx = np.arange(n_sim)
    coeffs = (2 * (idx+1) - n_sim - 1)[:, None]
    sum_pairwise = np.sum(coeffs * sorted_sims, axis=0)
    term2 = 2.0 * sum_pairwise / (2.0 * n_sim**2)
    crps = np.mean(term1 - term2)

    return {
        'MAE_vs_mean': mae_mean,
        'RMSE_vs_mean': rmse_mean,
        'Bias_vs_mean': bias_mean,
        'Mean_path_MAE': mean_path_mae,
        'Mean_path_RMSE': mean_path_rmse,
        'Coverage ±1σ': within_1std,
        'Coverage ±2σ': within_2std,
        'CRPS (time avg)': crps
    }

def evaluate_simulation_models(sim_mode, test):
    # Extract simulated series for each model
    simulated_ag = sim_mode[1]["ARMA-GARCH"]
    simulated_ou = sim_mode[1]["OU"]
    simulated_ou_jump = sim_mode[1]["OU + Jump"]
    simulated_ou_hawkes = sim_mode[1]["OU + Hawkes"]

    # Compute backtesting metrics vs test series
    metrics = {"ARMA-GARCH": backtest_model(test, simulated_ag),
        "OU": backtest_model(test, simulated_ou),
        "OU + Jump": backtest_model(test , simulated_ou_jump),
        "OU + Hawkes": backtest_model(test , simulated_ou_hawkes),
    }

    # Print results
    print("\n=== Backtest Results ===")
    for model_name, m in metrics.items():
        print(f"{model_name}: {m}")

    return metrics


"""## SIMULATIONS"""

def future_temperature(data, var, res, mean, variance, mode="exp", k=1.7, par=None, blend_days=0, pl=True, jump=False):
    """
    Extend historical temperature series into the future with a gradual increase.

    """

    df = data.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()

    # Extend the date index to 2050, filling missing dates
    new_index = pd.date_range(start=df.index.min(), end='2050-12-31', freq='D')
    df_extended = df.reindex(new_index).reset_index().rename(columns={'index': 'time'})

    # Split historical and future periods
    split_date = pd.Timestamp('2024-12-31')
    df_hist = df_extended[df_extended['time'] <= split_date].copy()
    df_future = df_extended[df_extended['time'] > split_date].copy()

    # Compute days relative to the start date
    t0 = df_extended['time'].min()
    t_hist = (df_hist['time'] - t0).dt.days.values
    t_future = (df_future['time'] - t0).dt.days.values

    # Compute approximation equations for historical and future periods
    eq_1 = np.asarray(approx_equation(t_hist, par))       # Historical trend
    eq_2_raw = np.asarray(approx_equation(t_future, par)) # Raw future trend

    # Adjust future series to ensure continuity with historical data
    offset = eq_1[-1] - eq_2_raw[0]
    eq_2_cont = eq_2_raw + offset

    # Compute scaling factor for future increase
    k_sc = k - 1.5
    t_rel = t_future - t_future[0]

    # Compute gradual increase: linear or exponential
    if mode == 'linear':
        increase = k_sc * (t_rel / t_rel[-1])
    elif mode == 'exp':
        r = 5 / len(t_rel)  # exponential growth rate
        norm = np.exp(r * t_rel[-1]) - 1
        increase = k_sc * (np.exp(r * t_rel) - 1) / norm
    else:
        raise ValueError("mode must be 'linear' or 'exp'")

    # Apply the increase to the future trend
    eq_2 = eq_2_cont + increase

    # Optional blending of the first few days of the future with the last historical value
    if blend_days and blend_days > 0:
        N = min(blend_days, len(eq_2))
        w = np.linspace(0, 1, N)
        eq_2[:N] = (1 - w) * eq_1[-1] + w * eq_2[:N]

    # Combine historical and future series
    future_series = np.concatenate((eq_1, eq_2))

    # Add stochastic noise to future values using KDE of residuals
    kde = gaussian_kde(res, bw_method=0.3)
    eps = kde.resample(len(t_future))[0]
    eq_2_noisy = eq_2 + eps
    serie = np.concatenate((df_hist[var].values, eq_2_noisy))
    t_extended = np.concatenate((t_hist, t_future))
    increase_full = np.concatenate((np.zeros(len(t_hist)), increase))

    trend_t = par["const"] + par["t"] * t_extended + increase_full

    # --- Plot results ---
    if pl:
        full_times = np.concatenate((df_hist['time'].values, df_future['time'].values))

        # Plot smoothed predicted series
        plt.figure(figsize=(10, 4))
        plt.plot(full_times, future_series, label='Predicted')
        plt.plot(full_times, trend_t, label='Trend line + growth')
        plt.axvline(split_date, color='k', linestyle='--', linewidth=0.7, label='Split 2024-12-31')
        plt.title(f"Historical + Future Prediction (gradual increase k={k})")
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot predicted series with noise
        plt.figure(figsize=(10, 4))
        plt.plot(full_times, serie, label='Predicted with noise')
        plt.plot(full_times, trend_t, label='Trend line + growth')
        plt.axvline(split_date, color='k', linestyle='--', linewidth=0.7, label='Split 2024-12-31')
        plt.title(f"Historical + Future Prediction (gradual increase k={k})")
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Plot only the noisy future predictions
        plt.figure(figsize=(10, 4))
        plt.plot(t_future, eq_2_noisy, label='Predicted future with noise')
        plt.plot(t_future, trend_t[-len(t_future):], label='Trend line + growth')
        plt.title(f"Future Prediction Only (gradual increase k={k})")
        plt.xlabel("Time (days from start)")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Update DataFrame with simulated series
    df_extended[var] = serie

    # Placeholder for jump identification
    # if jump:
    #     jumps_identification(serie)

    return df_extended

def simulate_future_temperature(data, var, res, mean,
                                variance, n_sim=1000, year_end='2050-12-31',
                                mode="exp", k=1.3, params=None, blend_days=0, plot=True):

    df = data.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    # Split historical and future
    split_date = pd.Timestamp("2024-12-31")
    future_index = pd.date_range(start=split_date + pd.Timedelta(days=1),
                                 end=year_end, freq="D")
    n_future = len(future_index)
    all_paths = np.zeros((n_sim, n_future))

    df_reset = df.reset_index()

    # Simulate paths
    for i in range(n_sim):
        df_sim = future_temperature(data=df_reset, var=var, res=res,
                                    mean=mean, variance=variance, mode=mode, k=k,
                                    par=params, blend_days=blend_days, pl=False)
        all_paths[i, :] = df_sim.loc[df_sim["time"] > split_date, var].values

    # Compute median and percentiles
    median = np.median(all_paths, axis=0)
    p5, p95 = np.percentile(all_paths, [5, 95], axis=0)

    # --- Plotting function ---
    def plot_simulations(df, var, future_index, median, p5, p95, split_date):
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df[var], label="Historical", color="C0")
        plt.plot(future_index, median, color="C1", label="Median Simulation")
        plt.fill_between(future_index, p5, p95, color="C1", alpha=0.25, label="5–95% Range")
        plt.axvline(split_date, color='k', linestyle='--', linewidth=0.8)
        plt.title(f"Monte Carlo Simulations until {future_index[-1].date()}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if plot:
        plot_simulations(df, var, future_index, median, p5, p95, split_date)

    return {
        "paths": all_paths,
        "future_dates": future_index,
        "median": median,
        "p5": p5,
        "p95": p95
    }

"""#**SVOLGIMENTO ANALISI**

##Importazione dati
"""

città = "Bahrain"
#city = pd.read_excel("Oslo.xlsx")
city = pd.read_excel("Bahrain.xlsx")
#city = pd.read_excel("Buenos Aires.xlsx")

"""Adjusting the dataframe to obtain a series without Nan"""
new = fill_tavg(city)
city["tavg"] = new["tavg"]
city.rename(columns={"Date": "time"}, inplace=True)

"""##Preliminary analysis"""

plots_stats(city,"tavg")

preliminary_analysis(city['tavg'])

extremes(city,"tavg",0.05)

seasonal_decomposition(city,"tavg")

rolling_window_estimates(city, "tavg")

"""Removing the sample to be used for forecasting"""
city_trunc=city.iloc[:-366,:]
testing = city["tavg"].iloc[-366:]

"""Approximation of the truncated dataset"""
det_app = deterministic_approximation(city_trunc, "tavg", trunc=0)
params=det_app["params"]
det = det_app["df"]
eq = approx_equation(np.arange(366),params)

"""Approximation of the whole dataset"""
dd_app=deterministic_approximation(city,"tavg")
whole_params = dd_app["params"]
dd_app = dd_app['df']

preliminary_analysis(det["residuals"])

extremes(det, "residuals", 0.05)

correlation_analysis(det["residuals"])

"""##Models

###Arma-Garch
"""

"""Arma fit"""
arma = best_BIC_ARMA(det["residuals"],3,3)

preliminary_analysis(arma["Residuals"])

"""Garch fit"""
garch =garch_estimate(arma["Residuals"])

ARMA_GARCH_MODEL=reconstruct_arma_garch(city_trunc["tavg"], det['approx'], arma, garch)

AG_single_for = dynamic_forecasting(det["residuals"], testing, ARMA_GARCH_MODEL, eq, 366)

AG_for = multiple_dynamic_forecasting(det["residuals"], testing, ARMA_GARCH_MODEL, eq, 366, N=1000)

"""###OU"""

OU_model = best_Ornstein_Uhlenbeck(det["residuals"], det["approx"])

OU_single_for = OU_forecasts(det["residuals"], OU_model, testing, 366, eq)

OU_for= multiple_OU_forecasts(det["residuals"], testing, OU_model, eq, 366, N=1000)

"""###jumps

"""

jumps= jumps_identification(det["residuals"], 3, max_iter = 10, use_mad=False)

OU_jumps_model = OU_with_jumps(det["residuals"], det["approx"])

OU_jump_single_for = simulate_OU_jumps(det["residuals"], testing, OU_jumps_model,  eq,
    N=366,
    dt=1.0)

OU_jump_for = multiple_OU_jumps(det["residuals"], testing, OU_jumps_model, N=366, approx=eq,
    dt=1.0, n_sim=1000)

"""###Hawkes"""

OU_Hawkes_model = OU_Hawkes(det["residuals"], det["approx"], 1, 1)

OU_hawkes_single_for = simulate_OU_Hawkes_new_steps(det["residuals"], testing, OU_Hawkes_model, eq, 366, 0,0)

OU_hawkes_for = multiple_OU_Hawkes(det["residuals"], testing,  OU_Hawkes_model, eq,
                          steps=366, n_sim=1000, scatter=0, hawkes_graph=0, plots=True)

"""##MonteCarlo"""

sim_mode = simmulate_models(det["residuals"], testing, eq,
                            ARMA_GARCH_MODEL, OU_model, OU_jumps_model, OU_Hawkes_model,
              steps = 366, n_sim = 10000)

MonteCarlo_indexes(sim_mode[1], city_trunc["time"], city_trunc["tavg"],testing)

MonteCarlo_option_payoffs(sim_mode[1], city_trunc["time"], city["tavg"], testing, steps=366, n_sim=10000, plot=True)




"""##Simulations

"""

mean, var = preliminary_analysis(dd_app["residuals"])

a = future_temperature(city[["time","tavg"]], "tavg", dd_app["residuals"], mean, var, "exp", 1.7, par = whole_params, jump=True)

b = future_temperature(city[["time","tavg"]], "tavg",  dd_app["residuals"], mean, var, "exp", 2.2, par = whole_params)

c = future_temperature(city[["time","tavg"]], "tavg",  dd_app["residuals"], mean, var, "exp", 2.6, par=whole_params)

d = future_temperature(city[["time","tavg"]], "tavg",  dd_app["residuals"], mean, var, "exp", 3.0, par=whole_params)

a_1 = simulate_future_temperature(data = city[["time","tavg"]], var= "tavg", res = dd_app["residuals"], mean = mean, variance = var,
                                n_sim=1000, year_end='2050-12-31', mode="exp", k=1.2, params=whole_params, blend_days=0, plot=True)

b_1 = simulate_future_temperature(data = city[["time","tavg"]], var= "tavg", res = dd_app["residuals"], mean = mean, variance = var,
                                n_sim=1000, year_end='2050-12-31', mode="exp", k=1.7, params=whole_params, blend_days=0, plot=True)

c_1 = simulate_future_temperature(data = city[["time","tavg"]], var= "tavg", res = dd_app["residuals"], mean = mean, variance = var,
                                n_sim=1000, year_end='2050-12-31', mode="exp", k=2, params=whole_params, blend_days=0, plot=True)

d_1 = simulate_future_temperature(data = city[["time","tavg"]], var= "tavg", res = dd_app["residuals"], mean = mean, variance = var,
                                n_sim=1000, year_end='2050-12-31', mode="exp", k=2, params=whole_params, blend_days=0, plot=True)

"""########### Studio sui futuri ancora in via di sviluppo ###########"""

# --- feature calculation ---
def features_from_series(x):
    feats = {}
    feats['mean'] = np.mean(x)
    feats['std'] = np.std(x, ddof=1)
    feats['skew'] = skew(x)
    feats['kurtosis'] = kurtosis(x)
    acf_vals = acf(x, nlags=30, fft=True)
    feats['acf_1'] = acf_vals[1]
    feats['acf_7'] = acf_vals[7]
    feats['acf_30'] = acf_vals[30]
    for q in [0.9, 0.95, 0.99]:
        feats[f'quantile_{int(q*100)}'] = np.quantile(x, q)
    return feats

# --- envelope calculation ---
def envelope_test(observed_series, sim_series_list):
    obs_feats = features_from_series(observed_series)
    feat_names = list(obs_feats.keys())
    env = {f: [] for f in feat_names}

    # compute features for each simulation
    sim_feats_list = [features_from_series(s) for s in sim_series_list]
    for f in feat_names:
        sim_values = [sf[f] for sf in sim_feats_list]
        env[f] = (np.percentile(sim_values, 5), np.percentile(sim_values, 95), np.median(sim_values))
    return obs_feats, env

# --- plot ---
def plot_envelope(obs_feats, env):
    feat_names = list(obs_feats.keys())
    fig, ax = plt.subplots(figsize=(12,5))

    obs_values = [obs_feats[f] for f in feat_names]
    low_values = [env[f][0] for f in feat_names]
    high_values = [env[f][1] for f in feat_names]
    median_values = [env[f][2] for f in feat_names]

    x = np.arange(len(feat_names))

    # plot envelope
    ax.fill_between(x, low_values, high_values, color='lightblue', alpha=0.5, label='5%-95% envelope')
    ax.plot(x, median_values, color='blue', linestyle='--', label='simulated median')
    ax.plot(x, obs_values, color='red', marker='o', label='observed')

    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, rotation=45, ha='right')
    ax.set_ylabel('Feature value')
    ax.set_title('Envelope test: observed vs simulated features')
    ax.legend()
    plt.tight_layout()
    plt.show()

# --- DEMO ---
if __name__ == '__main__':
    np.random.seed(42)
    # esempio: osservato
    obs = np.random.normal(10, 2, 365)

    # esempio: 100 simulazioni di un modello OU (o altro)
    sims = [np.random.normal(10, 2, 365) for _ in range(100)]

    obs_feats, env = envelope_test(obs, sims)
    plot_envelope(obs_feats, env)
