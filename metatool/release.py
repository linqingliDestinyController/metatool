"""Release integrity and explicit portable paths. CPU only."""
from pathlib import Path
import json
from metatool.io import ROOT,sha,read,atomic,digest,model_identity

def verify_release():
    expected=read(ROOT/'MANIFEST.json')
    bad=[n for n,h in expected.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    if bad:raise ValueError(f'Release files differ: {bad}')
    return expected

def disjoint_work(work):
    work=Path(work).resolve()
    if work==ROOT or ROOT in work.parents or work in ROOT.parents:
        raise ValueError('Work directory must be outside the release tree')
    return work
