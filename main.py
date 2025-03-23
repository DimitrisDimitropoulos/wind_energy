import pandas as pd
from wind.calc import (
    create_windrose_plot,
    comp_bowden_coeffs,
    create_occurrence_table,
    create_cumulative_occurrence_frequency,
    fit_weibull_line,
    compare_weibull,
    mean_weibull,
    energy_density_weibull,
)

# Air Density
rho: float = 1.2  # [kg/m^3]

# Load Data
df_speed = pd.read_csv("./data/speed.txt", header=None, names=["Speed"])
df_direction = pd.read_csv("./data/dir.txt", header=None, names=["Direction"])
df = pd.concat([df_speed, df_direction], axis=1)

uw_mean: float = df["Speed"].mean()
uw_std: float = df["Speed"].std()
print(f"Mean Wind Speed: {uw_mean}\nStandard Deviation of Wind Speed: {uw_std}")

create_windrose_plot(df["Speed"], df["Direction"], "./out/windrose.eps", 5)

# occ_pd = create_occurrence_table(df["Speed"], df["Direction"], 1)
# print(occ_pd)

bow_k, bow_c = comp_bowden_coeffs(uw_mean, uw_std)
print(f"Bowden Coefficients: k={bow_k:.2f}, c={bow_c:.2f}")

cum_occ = create_cumulative_occurrence_frequency(df["Speed"])

ls_k, ls_c = fit_weibull_line(cum_occ)
print(f"Least Squares Coefficients: k={ls_k:.2f}, c={ls_c:.2f}")

compare_weibull(df["Speed"], bow_k, bow_c, ls_k, ls_c)

bow_uw_mean = mean_weibull(bow_k, bow_c, 30)
ls_uw_mean = mean_weibull(ls_k, ls_c, 30)
print(
    f"Bowden Mean Wind Speed: {bow_uw_mean:.2f}\nLeast Squares Mean Wind Speed: {ls_uw_mean:.2f}"
)


bow_energy_density = energy_density_weibull(bow_k, bow_c, 30, rho)
ls_energy_density = energy_density_weibull(ls_k, ls_c, 30, rho)
print(
    f"Bowden Energy Density: {bow_energy_density:.2f} [J/m^2]\nLeast Squares Energy Density: {ls_energy_density:.2f} [J/m^2]"
)
