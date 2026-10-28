"""Internal subprocess entry; import paths work independently of current directory."""
from pathlib import Path
import sys,runpy
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/verl'));sys.path.insert(0,str(ROOT))
mode=sys.argv.pop(1)
module={'preflight':'runtime.preflight','finalize':'runtime.finalize','summarize':'runtime.summarize_training'}[mode]
runpy.run_module(module,run_name='__main__')
