import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def plot_mpi_gantt():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_path = os.path.join(script_dir, "mpi_log_rank_*.csv")
    files = glob.glob(search_path)

    if not files:
        print(f"mpi_log_rank_*.csv files not found in: {script_dir}")
        return

    print(f"Found {len(files)} log files in {script_dir}. Generating chart...")

    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading file {f}: {e}")

    if not df_list:
        return

    all_data = pd.concat(df_list)

    print("Normalizing time per rank (Fixing clock desync)...")

    for rank in all_data['Rank'].unique():
        rank_mask = (all_data['Rank'] == rank)
        min_rank_start = all_data.loc[rank_mask, 'Start'].min()

        all_data.loc[rank_mask, 'Start'] -= min_rank_start
        all_data.loc[rank_mask, 'End'] -= min_rank_start

    fig, ax = plt.subplots(figsize=(15, 8))

    colors = {'COMP': '#2ca02c', 'COMM': '#d62728'}
    labels_added = set()

    for index, row in all_data.iterrows():
        duration = row['End'] - row['Start']
        color = colors.get(row['Type'], 'gray')

        ax.barh(y=row['Rank'], width=duration, left=row['Start'],
                color=color, edgecolor='black', height=0.6, alpha=0.9)

        if row['Type'] not in labels_added:
            ax.barh(y=row['Rank'], width=0, left=0, color=color, label=row['Type'])
            labels_added.add(row['Type'])

    ax.set_xlabel('Time (seconds from local process start)')
    ax.set_ylabel('MPI Rank (Process Number)')
    ax.set_title('MPI Execution Gantt Chart (Normalized)')

    ranks = sorted(all_data['Rank'].unique())
    ax.set_yticks(ranks)
    y_labels = [f"Rank {r} ({'Master' if r==0 else 'Worker'})" for r in ranks]
    ax.set_yticklabels(y_labels)

    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_mpi_gantt()