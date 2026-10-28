"""CPU-only integration and numerical checks. No language-model weights or GPU jobs."""
from pathlib import Path
import sys,os,json,tempfile,shutil,random,argparse
from unittest.mock import patch
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/verl')]
import torch,numpy as np
from safetensors.torch import save_file
from omegaconf import OmegaConf
from metatool.io import read,model_identity
from metatool.rl.branch_actor import route_loss_from_logits
from metatool.rl.recovery_state import cpu_rng_state,restore_cpu_rng
from verl.trainer.ppo.core_algos import compute_policy_loss_vanilla
from runtime.model import bind_input,verify_input
from runtime.cleanup import cleanup
import run as entry


def tiny_model(folder,metadata):
    folder.mkdir()
    for name in ['config.json','generation_config.json','merges.txt','vocab.json','tokenizer.json','tokenizer_config.json']:
        src=metadata/name
        if src.is_file():shutil.copy2(src,folder/name)
    # A real tiny safetensors container for file-binding tests, not a usable Qwen model.
    save_file({'cpu_fixture':torch.zeros(1)},folder/'model.safetensors')


def run(args):
    checks={}
    if args.model_metadata is not None:
        # Verify new-run orchestration with a tiny file fixture. Mock only the GPU subprocess/cleanup.
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);model=base/'input';tiny_model(model,args.model_metadata);work=base/'work'
            initial=bind_input(model);assert verify_input(model,initial).startswith('bound_sha256')
            commands=[]
            with patch.object(sys,'argv',['run.py','train','--model',str(model),'--work',str(work),'--train-data',str(args.train_data),'--validation-data',str(args.validation_data)]),\
                 patch.object(entry,'child',side_effect=lambda *a,**k:commands.append((a,k))),\
                 patch('runtime.cleanup.cleanup') as finalize_cleanup:
                entry.main()
            assert len(commands)==1 and commands[0][0][0]=='runtime.experiment'
            assert commands[0][0][1][0:2]==['--work',work]
            assert read(work/'INSTALL.json')['initial_model']['path']==str(model.resolve())
            assert (work/'model').resolve()==model.resolve()
            assert finalize_cleanup.call_count==1
            checks['direct_model_install_and_train_dispatch_without_external_receipt']='PASS'
            # Same directory cannot switch to a different model on resume.
            other=base/'other';tiny_model(other,args.model_metadata)
            with patch.object(sys,'argv',['run.py','train','--model',str(other),'--work',str(work),'--resume']),patch.object(entry,'child') as child:
                try:entry.main()
                except ValueError:pass
                else:raise AssertionError('Different input model accepted')
                assert not child.called
            checks['resume_model_binding']='PASS'
            f=model/'model.safetensors';stat=f.stat();data=bytearray(f.read_bytes());data[-1]^=1;f.write_bytes(data)
            os.utime(f,ns=(stat.st_atime_ns,stat.st_mtime_ns))
            try:verify_input(model,initial)
            except ValueError:pass
            else:raise AssertionError('Changed weights accepted after preserved mtime')
            checks['input_content_change_detection']='PASS'
            # Final cleanup touches only this run's checkpoint paths and preserves both model directories.
            run_dir=work/'runs/metatool_seed42';export=run_dir/'merged_hf';export.mkdir(parents=True)
            save_file({'cpu_fixture':torch.ones(1)},export/'model.safetensors')
            (run_dir/'MERGE_COMPLETE.json').write_text(json.dumps(model_identity(export)))
            (run_dir/'TRAINING_SUMMARY.json').write_text(json.dumps({'complete':True}))
            (run_dir/'EXPERIMENT.json').write_text(json.dumps({'run_uid':'cpu-file-fixture'}))
            for step in (50,100):
                d=run_dir/f'checkpoints/global_step_{step}';d.mkdir(parents=True);(d/'state').write_text('fixture')
            record=cleanup(work);assert cleanup(work)==record
            assert f.exists() and (export/'model.safetensors').exists()
            assert not (run_dir/'checkpoints/global_step_100').exists()
            (run_dir/'checkpoints').rmdir();outside=base/'outside';outside.mkdir();(outside/'global_step_100').mkdir()
            (run_dir/'checkpoints').symlink_to(outside,target_is_directory=True)
            try:cleanup(work)
            except ValueError:pass
            else:raise AssertionError('Symlink parent accepted')
            assert (outside/'global_step_100').is_dir()
            checks['checkpoint_retirement_scope_idempotence_and_symlink_guard']='PASS'
    else:
        checks['external_input_integration']='NOT_RUN: provide --model-metadata, --train-data and --validation-data'
    # Direct log-prob gradients on small CPU tensors; not model parameter gradients.
    mask=torch.tensor([[1.,1.,1.,0.],[1.,1.,1.,1.]])
    weights=torch.tensor([[1.,0.,2.,0.],[.1,1.,2.,3.]])
    for adv in (-1.,0.,1.):
        logp=torch.zeros_like(mask,requires_grad=True)
        out=compute_policy_loss_vanilla(torch.zeros_like(mask),logp,torch.full_like(mask,adv),mask,
            loss_agg_mode='token-mean-weighted',config=OmegaConf.create(dict(clip_ratio=.2,clip_ratio_low=.2,clip_ratio_high=.2,clip_ratio_c=3.)),token_weight=weights)
        out[0].backward();assert torch.allclose(logp.grad,-adv*weights*mask/(mask*weights).sum(),atol=1e-7,rtol=1e-6)
    checks['native_PPO_gradient_positive_negative_zero']='PASS'
    for margin in (torch.tensor([.1,.5]),torch.zeros(2)):
        z=torch.tensor([[.1,.2,.3,.4],[.4,.3,.2,.1]],requires_grad=True)
        loss,_=route_loss_from_logits(z,[0,1,2],3,torch.tensor([0,1]),margin,.01)
        loss.backward();assert torch.isfinite(z.grad).all()
        assert bool(z.grad.count_nonzero())==bool(margin.count_nonzero())
    checks['route_gradient_and_inactive_slots']='PASS'
    state=cpu_rng_state();expected=(random.random(),np.random.rand(),torch.rand(3))
    restore_cpu_rng(state);actual=(random.random(),np.random.rand(),torch.rand(3))
    assert expected[:2]==actual[:2] and torch.equal(expected[2],actual[2]);checks['CPU_RNG_roundtrip']='PASS'
    return dict(status='PASS',checks=checks,new_language_model_training_updates=0,new_model_outputs=0,
                GPU_subprocess_mocked=True,full_GPU_training='NOT_VERIFIED',distributed_step50_resume='NOT_VERIFIED')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-metadata',type=Path);parser.add_argument('--train-data',type=Path);parser.add_argument('--validation-data',type=Path)
    args=parser.parse_args()
    supplied=[args.model_metadata is not None,args.train_data is not None,args.validation_data is not None]
    if any(supplied) and not all(supplied):parser.error('Provide all three external inputs for integration checks, or none for numerical checks')
    print(json.dumps(run(args),indent=2))
