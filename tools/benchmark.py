"""Public synthetic-prompt CLI benchmark; retain first request separately."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    for name in ('model','source','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    sentence='The sky appears blue because molecules scatter shorter wavelengths. Explain the physical mechanism clearly. '
    request={'prompt':sentence*254+'Summarize this in one sentence.','max_new_tokens':32}
    command=[sys.executable,'-m','qwen35_deploy','--model',str(a.model),'--source',str(a.source),'--jsonl']
    run=subprocess.run(command,cwd=root,input=(json.dumps(request)+'\n')*4,capture_output=True,text=True)
    if run.returncode:raise RuntimeError(run.stderr)
    responses=[json.loads(line) for line in run.stdout.splitlines() if line.strip()]
    if len(responses)!=4 or any('error' in r for r in responses):raise RuntimeError('Benchmark request failed')
    keys=('prompt_tokens','generated_tokens','prefill_seconds','prefill_tokens_per_second',
          'decode_tokens_per_second','model_seconds','peak_cuda_bytes')
    rows=[{'repeat':i,'warmup':i==0,**{k:r[k] for k in keys}} for i,r in enumerate(responses)]
    measured=responses[1:]
    summary={'prefill_tokens_per_second':sum(r['prompt_tokens'] for r in measured)/sum(r['prefill_seconds'] for r in measured),
        'decode_tokens_per_second':sum(r['generated_tokens']-1 for r in measured)/sum(sum(r['step_seconds'][1:]) for r in measured),
        'all_measured_minimum_met':all(r['prefill_tokens_per_second']>=1000 and r['decode_tokens_per_second']>=20 for r in measured)}
    report={'rows':rows,'summary':summary,'scope':'One repeated public synthetic prompt, first request retained separately; not private eight-domain evaluation or universal acceptance.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
