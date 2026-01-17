import pandas as pd
import matplotlib.pyplot as plt
import os

FILENAME = "empirical_results.csv"

possible_paths = [
    FILENAME,
    os.path.join("cmake-build-debug", FILENAME),
    os.path.join("build", FILENAME)
]

file_path = None
for path in possible_paths:
    if os.path.exists(path):
        file_path = path
        print(f"Data found in: {file_path}")
        break

if file_path is None:
    print(f"ERROR: Couldn't find file '{FILENAME}'!")
    print(f"Checked location: {possible_paths}")
    exit()

df = pd.read_csv(file_path)

N = df['N'].values
Time = df['Time_Seconds'].values

base_n = N[0]
base_time = Time[0]
theoretical_time = [base_time * (x / base_n) for x in N]

plt.figure(figsize=(10, 6))


plt.plot(N, theoretical_time, 'r--', label='Theory O(N) (Ideal scaling)', linewidth=2, alpha=0.7)
plt.plot(N, Time, 'bo-', label='Practice', linewidth=2, markersize=8)

plt.xlabel('Number of points (N)', fontsize=12)
plt.ylabel('Execution time (s)', fontsize=12)
plt.title('Empirical Analysis: Computational Complexity of K-Means', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

def human_format(num, pos):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '%.1f%s' % (num, ['', 'K', 'M', 'G'][magnitude])

from matplotlib.ticker import FuncFormatter
plt.gca().xaxis.set_major_formatter(FuncFormatter(human_format))

plt.tight_layout()
plt.savefig("complexity_chart.png")
print("The chart has been saved as ‘complexity_chart.png’.'.")
plt.show()