import numpy as np
import scipy.stats as stats

# QUESTION 20
data = np.array([94,
90,
95,
93,
94,
95,
95,
90,
94,
94,
])

# Statistik deskriptif
mean = np.mean(data)
std_dev = np.std(data, ddof=1)  # Simpangan baku dengan ddof=1 (sampel)
n = len(data)

# Derajat kebebasan
df = n - 1

# Nilai t untuk tingkat kepercayaan 95% dan df = 9
t_value = stats.t.ppf(1 - 0.025, df)  # 0.025 untuk 95% CI

# Margin of error
margin_of_error = t_value * (std_dev / np.sqrt(n))

# Confidence Interval
ci_lower = mean - margin_of_error
ci_upper = mean + margin_of_error

print(f"Confidence Interval: [{ci_lower}, {ci_upper}]")
