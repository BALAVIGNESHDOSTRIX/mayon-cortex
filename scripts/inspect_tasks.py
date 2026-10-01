import json
import os
import numpy as np

failed_tids = [
    '03560426', '05a7bcf2', '0607ce86', '08573cc6', '0934a4d8', 
    '09c534e7', '0a1d4ef5', '0bb8deee', '0becf7df', '0c9aba6e', 
    '0d87d2a6', '0e671a1a', '0f63c0b9', '103eff5b', '11e1fe23', '12997ef3'
]

for tid in failed_tids:
    path = os.path.join('data', 'arc_evaluation', f'{tid}.json')
    if not os.path.exists(path):
        continue
    with open(path) as f:
        d = json.load(f)
    p0 = d['train'][0]
    inp0 = np.array(p0['input'])
    out0 = np.array(p0['output'])
    same_shape = all(np.array(p['input']).shape == np.array(p['output']).shape for p in d['train'])
    in_shapes = [f"{len(p['input'])}x{len(p['input'][0])}" for p in d['train']]
    out_shapes = [f"{len(p['output'])}x{len(p['output'][0])}" for p in d['train']]
    print(f"{tid}: same_shape={same_shape} | in: {in_shapes[0]} -> out: {out_shapes[0]} | colors in: {set(inp0.flatten())} -> out: {set(out0.flatten())}")
