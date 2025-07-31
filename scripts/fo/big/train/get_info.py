from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-muted")  # Optional: Use a nicer default style


def parse_tensorboard(path, scalars):
    """Returns a dictionary of pandas dataframes for each requested scalar."""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "Some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


cwd = os.path.dirname(os.path.abspath(__file__))
logs = '/home/g15farris/bin/bayesaenet/bnn_aenet/logs'
method = cwd.split('/')[-2]
runs = glob.glob(f'{logs}/{method}_train/runs/*')

events = ['epoch', 'elbo/train', 'elbo/val', 'mse/train', 'mse/val']

for run in runs:
    if run[-5:] == 'small':
        continue
    tfevent_files = glob.glob(f'{run}/**/events.out.tfevents*', recursive=True)
    if not tfevent_files:
        print(f'No tfevent file for {run}')
        continue

    tfevent = tfevent_files[0]
    data = parse_tensorboard(tfevent, events)

    # Prepare figure
    os.makedirs(f'{cwd}/figs_{run.split("/")[-1]}', exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=200, sharex=True)
    axes = axes.flatten()

    for i, (ax, event) in enumerate(zip(axes, events[1:])):
        loss = data[event]['value'].to_numpy()

        # Adaptive ylim based on last 20%
        focus_fraction = 0.2
        focus_start = int(len(loss) * (1 - focus_fraction))
        focus_range = loss[focus_start:]
        margin = 0.1
        ymin = focus_range.min() - margin * abs(focus_range.min())
        ymax = focus_range.max() + margin * abs(focus_range.max())

        ax.plot(loss, lw=1.5)
        ax.set_ylim(ymin, ymax)
        ax.set_title(event.replace('/', ' ').capitalize())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.5)

        # Only add legend for train/val distinction
        if i % 2 == 1:
            ax.legend(['Validation'], loc='upper right', fontsize=8)
        else:
            ax.legend(['Training'], loc='upper right', fontsize=8)

    fig.suptitle(f"Training Curves: {run.split('/')[-1]}", fontsize=12, y=1.05)
    fig.tight_layout()
    fig.savefig(f'{cwd}/figs_{run.split("/")[-1]}/{run.split("/")[-1]}_events.png', bbox_inches='tight')
    plt.close(fig)
