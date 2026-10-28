"""CPU-only API-Bank scoring of caller-supplied predictions."""
import argparse,json
from pathlib import Path
from metatool.release import ROOT,atomic
from metatool.evaluation.api_score import load_cases,load_predictions,score

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    metrics,items=score(load_cases(ROOT/'data/api_bank'),load_predictions(a.predictions))
    atomic(a.output,dict(suite='api',metrics=metrics,items=items,scope='development_only'))
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
