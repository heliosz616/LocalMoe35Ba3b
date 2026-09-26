"""Conservative pre-publication hygiene scan of Git-eligible files."""
import json
from pathlib import Path
import re
import subprocess


RULES={
    'private_home_path':r'/(?:home|Users)/[A-Za-z0-9_.-]+/',
    'windows_home_path':r'[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+',
    'email_address':r'[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
    'private_key':r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
    'github_token':r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})',
    'hf_token':r'\bhf_[A-Za-z0-9]{20,}',
    'credential_assignment':r'(?i)(?:api_key|access_token|password)\s*[:=]\s*[\x22\x27][A-Za-z0-9_+/-]{12,}',
    'credential_url':r'https?://[^\s/]+:[^\s/]+@',
}
ALLOWED={'.py','.md','.txt','.json','.toml','.yml','.yaml'}


def audit(root):
    result=subprocess.run(['git','ls-files','--cached','--others','--exclude-standard','-z'],
                          cwd=root,capture_output=True,check=True)
    paths=sorted(set(p for p in result.stdout.decode().split('\0') if p));errors=[]
    for name in paths:
        path=root/name
        if path.is_symlink():errors.append({'file':name,'reason':'symlink'});continue
        if path.suffix not in ALLOWED and name!='.gitignore':errors.append({'file':name,'reason':'unexpected file type'});continue
        if path.stat().st_size>512000:errors.append({'file':name,'reason':'oversized file'});continue
        try:content=path.read_text()
        except UnicodeError:errors.append({'file':name,'reason':'binary content'});continue
        for reason,pattern in RULES.items():
            if re.search(pattern,content):errors.append({'file':name,'reason':reason})
    return {'passed':not errors,'checked_files':len(paths),'errors':errors,
        'scope':'Heuristic content/type/size scan plus curated allowlist review; not proof that arbitrary text can never contain sensitive information.'}


if __name__=='__main__':
    result=audit(Path(__file__).resolve().parents[1]);print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['passed'] else 1)
