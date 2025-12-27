import matplotlib.pyplot as plt
import pandas as pd
import io

data = """Threads,AvgTime_s,Speedup,Efficiency
1,14.51577,1.00,1.00
2,6.65914,2.18,1.09
3,4.60109,3.15,1.05
4,3.21920,4.51,1.13
6,2.35773,6.16,1.03
8,2.09494,6.93,0.87
12,2.25309,6.44,0.54
16,1.61908,8.97,0.56
"""

df = pd.read_csv(io.StringIO(data))

plt.style.use('ggplot')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1st plot (Speedup)
ax1.plot(df['Threads'], df['Speedup'], marker='o', markersize=8, linewidth=3, color='tab:blue', label='K-Means OpenMP')
ax1.plot(df['Threads'], df['Threads'], '--', color='gray', alpha=0.6, label='Perfect speedup')

ax1.set_title('Scalability analysis: Speedup', fontsize=14, pad=15)
ax1.set_xlabel('Number of threads', fontsize=12)
ax1.set_ylabel('Speedup S(p)', fontsize=12)
ax1.set_xticks(df['Threads'])
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

for x, y in zip(df['Threads'], df['Speedup']):
    ax1.annotate(f'{y:.2f}x', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

# 2nd plot (Efficiency)
ax2.plot(df['Threads'], df['Efficiency'], marker='s', markersize=8, linewidth=3, color='tab:green', label='Efficiency')
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Theoretical boundary (1.0)')

ax2.set_title('Resource Efficiency Analysis', fontsize=14, pad=15)
ax2.set_xlabel('Number of threads', fontsize=12)
ax2.set_ylabel('Efficiency E(p)', fontsize=12)
ax2.set_xticks(df['Threads'])
ax2.set_ylim(0, 1.25)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

for x, y in zip(df['Threads'], df['Efficiency']):
    ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.tight_layout()
plt.savefig('kmeans_scalability_final.png', dpi=300)
print("Plot saved as 'kmeans_scalability_final.png'")
plt.show()