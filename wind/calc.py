import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from windrose import WindroseAxes


def create_windrose_plot(
    speed: pd.Series,
    direction: pd.Series,
    filename: str,
    bin_int: int | float,
) -> None:
    """
    Create a windrose plot from the wind speed and direction data,
    and save the plot to an EPS file.
    Parameters:
        speed (pd.Series): Wind speed data.
        direction (pd.Series): Wind direction data.
        filename (str): Output filename for the EPS image.
        bin_int (int | float): Interval size for bins (e.g., every 5 m/s).
    """
    bins = np.arange(0, speed.max() + 5, bin_int)
    ax = WindroseAxes.from_ax()
    ax.set_theta_zero_location("N")  # Set North (0°) at the top
    ax.set_theta_direction(-1)  # Set it clockwise
    ax.bar(
        direction,
        speed,
        normed=True,
        opening=0.8,
        edgecolor="white",
        nsector=16,
        bins=bins,
    )
    ax.set_legend(title="Wind Speed Distribution")
    plt.savefig(filename, format="eps")
    # plt.show()


def create_occurrence_table(
    speed: pd.Series,
    direction: pd.Series,
    bin_size: int,
) -> pd.DataFrame:
    """
    Create a 2D table (DataFrame) of wind occurrences,
    with wind speed binned in 1 m/s increments (0-1, 1-2, ..., 24-25)
    along the rows, and 16 wind direction bins (from North=0°) along the columns.
    Parameters:
        speed (pd.Series): Wind speed data (m/s).
        direction (pd.Series): Wind direction data (degrees from North).
        bin_size (int): Size of the bins for speed data
    Returns:
        pd.DataFrame: 2D table of occurrence counts.
    Raises:
        ValueError: If any speed is below 0, or direction is outside [0, 360].
    """
    # Validate data
    if (speed < 0).any():
        raise ValueError("Speed values must be non-negative.")
    if (direction < 0).any() or (direction > 360).any():
        raise ValueError("Direction values must be within [0, 360].")
    # Define bins (speed: 0..25, direction: 16 bins from 0..360)
    speed_bins = np.arange(0, 26, bin_size).tolist()
    direction_bins = np.arange(0, 361, 22.5).tolist()  # 0, 22.5, 45, ..., 360
    hist, speed_edges, dir_edges = np.histogram2d(
        speed, direction, bins=(speed_bins, direction_bins)
    )
    # Create row labels (speed) and column labels (direction)
    speed_labels = [
        f"{int(speed_edges[i])}-{int(speed_edges[i + 1])}"
        for i in range(len(speed_edges) - 1)
    ]
    dir_labels = [
        f"{int(dir_edges[j])}-{int(dir_edges[j + 1])}"
        for j in range(len(dir_edges) - 1)
    ]
    # Create DataFrame
    occurrence_df = pd.DataFrame(hist, index=speed_labels, columns=dir_labels)
    # Check: total count in the table should equal the original number of records
    total_occurrences: float = occurrence_df.to_numpy().sum()
    original_count: int = len(speed)
    if total_occurrences != original_count:
        raise ValueError(
            f"Total count mismatch: table sum is {total_occurrences} but original data count is {original_count}"
        )
    return occurrence_df


def comp_bowden_coeffs(
    uw_mean: float,
    uw_std: float,
) -> tuple[float, float]:
    """
    Compute the Bowden coefficients (Weibull shape parameter k and scale parameter c)
    using the method of moments approximation.
    Parameters:
        uw_mean (float): The mean wind speed.
        uv_std (float): The standard deviation of the wind speed.
    Returns:
        tuple[float, float]: A tuple containing:
            bowden_k (float): Weibull shape parameter.
            bowden_c (float): Weibull scale parameter.
    Raises:
        ValueError: If the mean wind speed is less than or equal to zero.
    """
    if uw_mean <= 0:
        raise ValueError("Mean wind speed must be greater than zero")
    # Empirical approximation for the Weibull shape parameter c and k (Bowden's estimate)
    bowden_c: float = (2 * uw_mean) / (math.sqrt(math.pi))
    bowden_k: float = (uw_std / uw_mean) ** (-1.086)
    return bowden_k, bowden_c


def create_cumulative_occurrence_frequency(speed: pd.Series) -> pd.DataFrame:
    """
    Create a table of bin values (e.g. 0, 0.2, 0.4, ... 36) and
    the number of occurrences up to (and including) that bin.
    Parameters:
        speed (pd.Series): Wind speed data.
    Returns:
        pd.DataFrame: Columns for bin_value and occurrences.
    Raises:
        ValueError: If any speed is below 0.
    """
    if (speed < 0).any():
        raise ValueError("Speed values must be non-negative.")
    bin_values = np.concatenate([np.linspace(0, 1, 6), np.arange(2, 27)])
    occurrences: list[int] = []
    for b in bin_values:
        count: int = speed[speed >= b].size
        occurrences.append(count)
        max_occurrence = max(occurrences)
    frequencies = [occ / max_occurrence for occ in occurrences]
    df = pd.DataFrame(
        {"bin_value": bin_values, "occurrences": occurrences, "frequency": frequencies}
    )
    return df


def fit_weibull_line(df: pd.DataFrame) -> tuple[float, float]:
    """
    Perform a linear least squares fit for ln(U) vs. ln(-ln(Q)) on the given DataFrame.
    The DataFrame must have:
        - "bin_value" column for speed bins (U)
        - "frequency" column for cumulative frequency (Q)
    Returns:
        tuple[float, float]: (ls_k, ls_c) of the Weibull distribution.
    """
    df_copy = df.copy()
    df_copy.loc[df_copy["bin_value"] <= 0, "bin_value"] = 1e-6
    df_copy.loc[df_copy["frequency"] <= 0, "frequency"] = 1e-6
    # Filter out invalid rows (log domain issues)
    mask = (df["bin_value"] > 0) & (df["frequency"] > 0) & (df["frequency"] < 1)
    # Prepare x = ln(U), y = ln(-ln(Q))
    x = np.log(df.loc[mask, "bin_value"])
    y = np.log(-np.log(df.loc[mask, "frequency"]))
    # Linear least squares
    slope, intercept = np.polyfit(x, y, 1)
    ls_k = slope
    ls_c = math.exp(-intercept / slope)
    return ls_k, ls_c


def compare_weibull(
    speed: pd.Series, k1: float, c1: float, k2: float, c2: float
) -> None:
    """
    Plot a histogram of wind speed data and two Weibull distributions
    (given by k1,c1 and k2,c2) on the same figure.
    Curve 1 corresponds to the Bowden coefficients,
    and curve 2 corresponds to the Least Squares fit.
    Parameters:
        speed (pd.Series): Wind speed data.
        k1 (float): Weibull shape parameter for the Bowden fit.
        c1 (float): Weibull scale parameter for the Bowden fit.
        k2 (float): Weibull shape parameter for the Least Squares fit.
        c2 (float): Weibull scale parameter for the Least Squares fit.
    Returns:
        None
    """
    plt.figure()
    speeds = speed.to_numpy()
    # Plot histogram (normalized)
    plt.hist(speeds, bins=30, density=True, alpha=0.5, label="Speed Data")
    # Generate a range of speed values
    v_range = np.linspace(0, speeds.max(), 200)
    # Bowden curve
    pdf1 = (k1 / c1) * (v_range / c1) ** (k1 - 1) * np.exp(-((v_range / c1) ** k1))
    plt.plot(v_range, pdf1, label=f"Bowden (k={k1:.2f}, c={c1:.2f})")
    # Least Squares curve
    pdf2 = (k2 / c2) * (v_range / c2) ** (k2 - 1) * np.exp(-((v_range / c2) ** k2))
    plt.plot(v_range, pdf2, label=f"Least Squares (k={k2:.2f}, c={c2:.2f})")
    plt.xlabel("Wind Speed (m/s)")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig("./out/weibull_histogram.eps", format="eps")
    # plt.show()


def mean_weibull(k: float, c: float, upper_limit: float) -> float:
    """
    Compute the mean value of a Weibull distribution by integrating
    the PDF from 0 to the specified upper limit.
    Parameters:
        k (float): Weibull shape parameter.
        c (float): Weibull scale parameter.
        upper_limit (float): The upper limit of integration.
    Returns:
        float: Mean value of the Weibull distribution.
    """
    x_range = np.linspace(0, upper_limit, 1000)
    pdf = (k / c) * (x_range / c) ** (k - 1) * np.exp(-((x_range / c) ** k))
    mean_val = np.trapz(x_range * pdf, x_range)
    return mean_val


def energy_density_weibull(k: float, c: float, upper_limit: float, rho: float) -> float:
    """
    Compute the energy density of a Weibull distribution by
    the PDF from 0 to the specified upper limit.
    Based on the formula: int_0^upper_limmit 0.5 * rho * v^3 * f(v) dv
    Parameters:
        k (float): Weibull shape parameter.
        c (float): Weibull scale parameter.
        upper_limit (float): The upper limit of integration.
        rho (float): Air density.
    Returns:
        float: Energy density of the Weibull distribution.
    """
    x_range = np.linspace(0, upper_limit, 1000)
    pdf = (k / c) * (x_range / c) ** (k - 1) * np.exp(-((x_range / c) ** k))
    energy_density = 0.5 * rho * x_range**3 * pdf
    energ_den = np.trapz(energy_density, x_range)
    return energ_den
