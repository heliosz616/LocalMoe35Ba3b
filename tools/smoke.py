"""Run both public CLI paths; save counts/timing only, never prompt/output text."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    for name in ('model','source','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    sentence='The sky appears blue because molecules scatter shorter wavelengths. Explain the physical mechanism clearly. '
    requests=[{'prompt':'Check output limit.','max_new_tokens':34},
        {'prompt':sentence*500,'max_new_tokens':1},
        {'prompt':sentence*220+'Summarize this in one sentence.','max_new_tokens':32},
        {'prompt':'What is 7 plus 5? Reply with only the number.','max_new_tokens':8}]
    rows=[]
    for activation in ('w8a16','w8a8'):
        command=[sys.executable,'-m','qwen35_deploy','--model',str(args.model),
            '--source',str(args.source),'--jsonl','--activation',activation]
        completed=subprocess.run(command,cwd=root,input=''.join(json.dumps(r)+'\n' for r in requests),
                                 capture_output=True,text=True)
        if completed.returncode:raise RuntimeError(completed.stderr)
        responses=[json.loads(line) for line in completed.stdout.splitlines() if line.strip()]
        ready=[json.loads(line)['ready'] for line in completed.stderr.splitlines() if line.startswith('{"ready":')]
        if len(responses)!=4 or len(ready)!=1:raise RuntimeError('Unexpected CLI response count')
        if '33 output' not in responses[0].get('error','') or '4096 prompt' not in responses[1].get('error',''):
            raise RuntimeError('Request bounds not enforced')
        if responses[3].get('text','').strip()!='12':raise RuntimeError('Arithmetic smoke failed')
        if ready[0]['activation']!=activation or not ready[0]['validated_payload']:raise RuntimeError('Wrong profile')
        for response in responses[2:]:
            if not response.get('token_ids'):raise RuntimeError('No generated output')
        fields=('prompt_tokens','generated_tokens','prefill_seconds','prefill_tokens_per_second',
                'decode_tokens_per_second','peak_cuda_bytes','retained_experts')
        rows.append({'activation':activation,'ready_seconds':ready[0]['load_seconds'],
            'requests':[{k:r[k] for k in fields} for r in responses[2:]],'bounds_and_recovery_passed':True})
        print(json.dumps(rows[-1]),flush=True)
    report={'passed':True,'rows':rows,'scope':'Public CLI smoke, bounds/recovery and arithmetic; not a rerun of the private quality suite or universal throughput acceptance.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
