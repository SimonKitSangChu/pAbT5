# This script is specific to DeepAb structures.
# pymol -r show_structures.py -- pt src_dir pdb_name chain_names scheme key_h key_l
#
# command list
# pymol -r show_structures.py -- evaluations_sars.pt 6WPS.fasta 6WPS ED (scheme) 6WPS_CFIBEH 6WPS_BEHCFI

# pymol -r show_structures.py -- evaluations_sars.pt 6wps 6WPS DE (scheme) 6WPT_D 6WPT_E
# pymol -r show_structures.py -- evaluations_sars.pt 6wpt 6WPT DE (scheme) 6WPT_D 6WPT_E
# pymol -r show_structures.py -- evaluations_sars.pt 7tb8_1 7TB8 DE (scheme) 7TB8_D 7TB8_E
# pymol -r show_structures.py -- evaluations_sars.pt 7tb8_2 7TB8 HI (scheme) 7TB8_H 7TB8_I


from pathlib import Path
import sys

from pymol import cmd
from pymol import stored
from pymol.util import cbaw, cbag, cbac
from scipy.special import softmax
from scipy.stats import entropy
import torch


def load_pdb(src_dir, obj_name0=None, chain_names=None, label_name=None):
    src_dir = Path(src_dir)
    if label_name is None:
        label_name = src_dir.stem

    obj_names = []
    for state_i, pdb in enumerate(src_dir.glob('decoys/pred.deepab.*.pdb'), 1):
        obj_name = f'{label_name}_{pdb.stem}'
        obj_names.append(obj_name)

        cmd.load(pdb, object=obj_name)
        if obj_name0 is None:
            obj_name0 = obj_name
        elif chain_names is None:
            cmd.cealign(obj_name0, obj_name)
        else:
            chain_selection = f'chain {"+".join(list(chain_names))}'
            cmd.cealign(f'{obj_name0} and {chain_selection}', obj_name)

    assert obj_names, f'no pred.deepab.*.pdb found under {src_dir}'
    cmd.join_states(name=label_name, selection=' or '.join(obj_names))
    for obj_name in obj_names:
        if obj_name != obj_name0:
            cmd.delete(obj_name)

    return obj_name0


# parse input
inputs = sys.argv[1:] + (7 - len(sys.argv[1:])) * [None]
pt, src_dir, pdb_name, chain_names, scheme, key_h, key_l = inputs

if scheme not in ('entropy', 'cross_attentions', 'none', None):
    raise ValueError('only support schemes - entropy, cross_attentions')

src_dir = Path(src_dir)
if not src_dir.exists():
    raise FileNotFoundError(f'{src_dir} not found')

if pt not in ('none', None):
    pt = Path(pt)
    if not pt.exists():
        raise FileNotFoundError(f'{pt} not found')

if pdb_name is not None and pdb_name != 'none':
    obj_name0 = pdb_name
    pdb_name = pdb_name.lower()
    cmd.fetch(pdb_name)
    cbaw(pdb_name)
    cmd.set('cartoon_transparency', 0.5, pdb_name)
else:
    obj_name0 = None
    chain_names = None

label_name = src_dir.stem + 'deepab'
load_pdb(src_dir, obj_name0=obj_name0, chain_names=chain_names, label_name=label_name)
cbag(f'{label_name} and chain H')
cbac(f'{label_name} and chain L')
cmd.alter(label_name, 'b=0.0')

use_scheme = not (scheme is None or key_l is None or key_h is None)
if use_scheme:
    data = torch.load(pt)
    if scheme == 'entropy':
        logits_h = data[key_h]['evaluation']['logits'].squeeze(0)
        logits_l = data[key_l]['evaluation']['logits'].squeeze(0)
        values_h = entropy(softmax(logits_h, axis=-1), axis=-1)[1:-1]  # drop BOS, EOS token
        values_l = entropy(softmax(logits_l, axis=-1), axis=-1)[1:-1]
        cap_value = 1.75
    elif scheme == 'cross_attentions':  # average attention TO these residues, average FROM always equals one
        attention_h = data[key_l]['evaluation']['cross_attentions'][..., 1:-1, :-1]  # drop EOS token
        attention_l = data[key_h]['evaluation']['cross_attentions'][..., 1:-1, :-1]
        values_h = attention_h.mean(axis=(0, 1, 2))
        values_l = attention_l.mean(axis=(0, 1, 2))
        cap_value = 0.01
    else:
        raise NotImplementedError

    values_h[values_h > cap_value] = cap_value
    values_l[values_l > cap_value] = cap_value

    count = cmd.count_atoms(f'{label_name} and chain H and name CA')
    assert len(values_h) == count, f'mismatch between sizes: {count} v.s. {len(values_h)} in heavy chain'
    count = cmd.count_atoms(f'{label_name} and chain L and name CA')
    assert len(values_l) == count, f'mismatch between sizes: {count} v.s. {len(values_l)} in heavy chain'

    stored.values_h = values_h.tolist()
    stored.values_l = values_l.tolist()
    cmd.alter(f'{label_name} and chain H and name CA', 'b=stored.values_h.pop(0)')
    cmd.alter(f'{label_name} and chain L and name CA', 'b=stored.values_l.pop(0)')

    # normalize color together
    cmd.spectrum(expression='b', palette='blue_white_red', selection=f'{label_name} and name CA')
    # normalize color separately
    # cmd.spectrum(expression='b', palette='blue_white_red', selection=f'{label_name} and chain H and name CA')
    # cmd.spectrum(expression='b', palette='blue_white_red', selection=f'{label_name} and chain L and name CA')

# pymol visualization config
cmd.hide('everything', 'not polymer.protein')
cmd.set('ray_shadow', 0)
cmd.orient(label_name)
