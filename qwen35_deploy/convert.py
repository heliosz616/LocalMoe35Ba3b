"""Streaming, resumable Qwen3.6 W8A16 baseline; no VQ payload is consumed.

Per-128 absmax RTN is an explicit baseline, not a claim of optimal quality.
Each output is atomically published and bound to its source shard checksum.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def publish(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(tmp, path)


def quantize(w):
    shape = w.shape
    v = w.float().reshape(*shape[:-1], shape[-1] // 128, 128)
    scale = v.abs().amax(-1).div(127).clamp_min(1e-30)
    q = (v / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
    err = (q.float() * scale.unsqueeze(-1) - v).double()
    return q.reshape(shape).cpu(), scale.cpu(), float(err.square().sum()), float(v.double().square().sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-tensors', type=int, default=0)
    a = p.parse_args()
    torch.set_num_threads(4)
    a.output.mkdir(parents=True, exist_ok=True)
    index = json.loads((a.source / 'model.safetensors.index.json').read_text())
    identity = {'format': 'q36-w8a16-g128-rtn-fp32scale-v1',
                'source_index_sha256': sha(a.source / 'model.safetensors.index.json'),
                'source_config_sha256': sha(a.source / 'config.json'),
                'source': str(a.source.resolve())}
    config = a.output / 'conversion.json'
    if config.exists() and json.loads(config.read_text()) != identity:
        raise RuntimeError('Conversion identity mismatch')
    publish(config, identity)
    inventory, records, hashes = [], [], {}
    for shard in sorted(set(index['weight_map'].values())):
        path = a.source / shard
        hashes[shard] = sha(path)
        with safe_open(path, framework='pt', device='cpu') as f:
            for name in sorted(f.keys()):
                shape = f.get_slice(name).get_shape()
                inventory.append({'name': name, 'shape': shape,
                                  'dtype': f.get_slice(name).get_dtype(), 'shard': shard})
    publish(a.output / 'source_inventory.json', {'shard_sha256': hashes, 'tensors': inventory})
    selected = [r for r in inventory if r['name'].startswith('model.language_model.') or r['name'] == 'lm_head.weight']
    for i, item in enumerate(selected):
        name, shape, shard = item['name'], item['shape'], item['shard']
        key = hashlib.sha256(name.encode()).hexdigest()[:24]
        output = a.output / (key + '.safetensors')
        record = a.output / (key + '.json')
        if record.exists():
            r = json.loads(record.read_text())
            if r['source_sha256'] != hashes[shard] or r['name'] != name or sha(output) != r['sha256']:
                raise RuntimeError('Resume validation failed: ' + name)
            records.append(r)
            continue
        expert = '.mlp.experts.' in name
        quant = ((expert or (len(shape) == 2 and min(shape) >= 512))
                 and 'embed_tokens' not in name and shape[-1] % 128 == 0)
        values, error, energy = {}, 0., 0.
        with safe_open(a.source / shard, framework='pt', device='cpu') as f:
            if quant:
                # Large vocabulary heads must not allocate full FP32/FP64
                # reconstruction temporaries on a 12 GiB device.
                tile_rows = 4096
                count = shape[0] if expert else (shape[0] + tile_rows - 1) // tile_rows
                qs, scales = [], []
                for e in range(count):
                    w = (f.get_slice(name)[e] if expert else
                         f.get_slice(name)[e*tile_rows:(e+1)*tile_rows]).to('cuda')
                    q, s, er, en = quantize(w)
                    qs.append(q); scales.append(s); error += er; energy += en
                    del w
                values = {'q': torch.stack(qs) if expert else torch.cat(qs),
                          'scale': torch.stack(scales) if expert else torch.cat(scales)}
            else:
                values = {'weight': f.get_tensor(name)}
        tmp = output.with_suffix('.tmp')
        save_file(values, tmp)
        os.replace(tmp, output)
        r = {**item, 'source_sha256': hashes[shard], 'file': output.name,
             'sha256': sha(output), 'bytes': output.stat().st_size,
             'format': 'int8_g128_fp32scale' if quant else 'original',
             'weight_relative_l2': (error / energy) ** .5 if energy else 0.,
             'quality_validation': 'pending_model_level'}
        publish(record, r)
        records.append(r)
        publish(a.output / 'progress.json', {'completed': len(records), 'total': len(selected), 'last': name})
        print(json.dumps({'completed': len(records), 'total': len(selected), 'name': name,
                          'weight_relative_l2': r['weight_relative_l2']}), flush=True)
        if a.max_tensors and len(records) >= a.max_tensors:
            return
    publish(a.output / 'manifest.json', {'identity': identity, 'records': records,
            'complete': True, 'quality_validation': 'pending_model_level',
            'scope': 'text inference; vision and MTP excluded',
            'bytes': sum(r['bytes'] for r in records)})


if __name__ == '__main__':
    main()
