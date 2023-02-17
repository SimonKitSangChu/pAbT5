import argparse
import matplotlib.pyplot as plt
from pathlib import Path

from pabt5.dataset import get_oas_statistics

plt.style.use('seaborn-colorblind')

# cmd options
cli = argparse.ArgumentParser()
cli.add_argument('--output_dir', type=str, default='visualization/oas_statistics')
cli.add_argument('--format', type=str, default='png', help='output figure format')
args = cli.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

stats = get_oas_statistics()

# dataset by species
data = stats['Species']
data_species = {
    'human': data['human'],
    'rat': data['rat_SD'],
    'mouse': data['mouse_BALB/c'] + data['mouse_C57BL/6'],
}

ax = plt.pie(
    x=data_species.values(),
    labels=data_species.keys(),
    autopct='%.0f%%',
    startangle=90,
    textprops={'fontsize': 'x-large'},
)
plt.tight_layout()
plt.savefig(output_dir / f'species.{args.format}', dpi=300)
