"""Offline single-stream deployment of the validated 4K profile."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
import torch
from transformers import AutoTokenizer
from qwen35_deploy import model as model_loader
from qwen35_deploy.operators import int8
from qwen35_deploy.operators.gdn_graph import install as install_graph
from qwen35_deploy.runtime.seeded_cache import install as install_cache
from qwen35_deploy.runtime.gpu_routes import install as install_routes
from qwen35_deploy.runtime.frequency_seed import install as install_frequency
from qwen35_deploy.runtime.prepare_decode import prepare
from qwen35_deploy.generation import generate
from qwen35_deploy.release import verify_release,validate_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True,help='Local converted INT8 artifact directory')
    parser.add_argument('--source',type=Path,required=True,help='Local original config/tokenizer directory; no downloads')
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prompt');mode.add_argument('--jsonl',action='store_true')
    parser.add_argument('--max-new-tokens',type=int,default=32)
    parser.add_argument('--activation',choices=('w8a8','w8a16'),default='w8a8')
    parser.add_argument('--allow-unvalidated-model',action='store_true')
    args=parser.parse_args()
    if not 1<=args.max_new_tokens<=33:parser.error('Supported output budget is 1..33 tokens')
    try:
        lock=verify_release();payload,matched=validate_manifest(args.model,lock,args.allow_unvalidated_model)
    except (ValueError,OSError,KeyError) as error:parser.error(str(error))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    if args.activation=='w8a8':
        from qwen35_deploy.operators.w8a8 import grouped_linear
        int8.grouped_linear=grouped_linear
    started=time.perf_counter()
    model,cache,source=model_loader.load(args.model,args.source,7)
    cache.queued_transfers=True;cache=install_cache(model,cache)
    if install_graph(model)!=30:raise ValueError('Expected 30 GDN layers')
    install_routes(model);preparation=prepare(model);frequency=install_frequency(model,cache)
    if frequency.quota!=45:raise ValueError('Unexpected expert retention geometry')
    cache.frequency_seed_deployment=True
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    gc.collect();gc.freeze()
    print(json.dumps({'ready':{'load_seconds':time.perf_counter()-started,'decode_preparation_seconds':preparation['seconds'],
        'profile':'rtx3060-4k','activation':args.activation,'payload_fingerprint':payload,'validated_payload':matched,
        'prefill_chunk':2048,'cache_gib':7,'prompt_limit':4096,'output_limit':33,
        'single_stream':True,'universal_1000_20_guarantee':False}}),file=sys.stderr,flush=True)
    def request(prompt,budget):return generate(model,tokenizer,cache,prompt,budget,4129,2048,True)
    try:
        if args.jsonl:
            for line in sys.stdin:
                try:
                    data=json.loads(line)
                    if not isinstance(data,dict):raise ValueError('Expected a JSON object')
                    result=request(data['prompt'],data.get('max_new_tokens',args.max_new_tokens))
                except (ValueError,KeyError,TypeError) as error:result={'error':str(error)}
                print(json.dumps(result,ensure_ascii=False),flush=True)
        else:print(json.dumps(request(args.prompt,args.max_new_tokens),ensure_ascii=False),flush=True)
    finally:gc.unfreeze()


if __name__=='__main__':main()
