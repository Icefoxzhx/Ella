import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_avg_series(csv_path, num_samples=200):
    df = pd.read_csv(csv_path)
    sim_hours = df['sim_hours'].values
    agent_cols = [c for c in df.columns if c != 'sim_hours']
    avg_total = df[agent_cols].mean(axis=1).values
    std_total = df[agent_cols].std(axis=1).values

    if len(sim_hours) > num_samples:
        idx = np.unique(np.round(np.linspace(0, len(sim_hours) - 1, num_samples)).astype(int))
        sim_hours = sim_hours[idx]
        avg_total = avg_total[idx]
        std_total = std_total[idx]

    return sim_hours, avg_total, std_total, len(agent_cols)


def hours_to_label(h):
    m = int(h * 60)
    return f"+{m}m" if m > 0 else "0m"


def plot(csv_paths, labels, output_path):
    plt.rcParams.update({
        'axes.facecolor': 'white',
        'axes.edgecolor': '#cccccc',
        'axes.grid': True,
        'grid.color': '#dddddd',
        'grid.linestyle': '-',
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
    })

    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728']
    markers = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        sim_hours, avg_total, std_total, n_agents = load_avg_series(csv_path)
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.plot(sim_hours, avg_total,
                label=f"{label} (n={n_agents})",
                color=color, marker=marker,
                markersize=6, linewidth=2,
                markevery=max(1, len(sim_hours) // 20))
        ax.fill_between(sim_hours,
                        avg_total - std_total,
                        avg_total + std_total,
                        alpha=0.15, color=color)
        ax.annotate(f"{avg_total[-1]:.0f}",
                    xy=(sim_hours[-1], avg_total[-1]),
                    xytext=(6, 0), textcoords='offset points',
                    color=color, fontsize=12, va='center')

    ax.set_title('Average Total Memory Nodes Over Simulation Time')
    ax.set_xlabel('Simulation Time')
    ax.set_ylabel('Avg Total Node Count (per agent)')
    ax.legend()

    xticks = [x for x in ax.get_xticks() if x >= 0]
    ax.set_xticks(xticks)
    ax.set_xticklabels([hours_to_label(x) for x in xticks])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare avg total memory nodes across runs.')
    parser.add_argument('csvs', nargs='+', help='CSV file(s) produced by node_data export scripts')
    parser.add_argument('--labels', nargs='+', help='Legend label per CSV (default: filename)')
    parser.add_argument('--output', default='node_comparison.png')
    args = parser.parse_args()

    labels = args.labels if args.labels else [p.replace('.csv', '').split('/')[-1] for p in args.csvs]
    if len(labels) != len(args.csvs):
        parser.error('--labels count must match number of CSVs')

    plot(args.csvs, labels, args.output)
