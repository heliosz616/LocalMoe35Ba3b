"""Fail-closed source/environment checks; payload fingerprint excludes paths."""
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(manifest):
    records=[{k:r[k] for k in ('name','shape','dtype','sha256','bytes','format')}
             for r in sorted(manifest['records'],key=lambda r:r['name'])]
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def verify_release():
    root=Path(__file__).resolve().parents[1]
    lock=json.loads((root/'release.json').read_text())
    actual={str(p.relative_to(root)) for p in (root/'qwen35_deploy').rglob('*.py')}
    if actual!=set(lock['source_sha256']):raise ValueError('Release module inventory changed')
    for name,value in lock['source_sha256'].items():
        if digest(root/name)!=value:raise ValueError('Release source changed: '+name)
    if sys.version_info[:2]!=(3,12):raise ValueError('Validated release requires Python 3.12')
    for name,value in lock['versions'].items():
        if importlib.metadata.version(name)!=value:raise ValueError('Untested dependency version: '+name)
    return lock


def validate_manifest(root,lock,allow_unvalidated=False):
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest.get('complete') is not True:raise ValueError('Incomplete converted model')
    names=set()
    for record in manifest['records']:
        target=(root/record['file']).resolve()
        if not target.is_relative_to(root.resolve()) or target.suffix!='.safetensors':raise ValueError('Unsafe model record path')
        if record['name'] in names:raise ValueError('Duplicate tensor record')
        names.add(record['name'])
    value=fingerprint(manifest)
    matched=value==lock['artifact_fingerprint']
    if not matched and not allow_unvalidated:raise ValueError('Unvalidated model payload; use --allow-unvalidated-model to acknowledge that published quality/performance do not apply')
    return value,matched
