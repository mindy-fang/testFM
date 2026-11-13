from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import random
import time
import torch
from .model import FoundationV13,ModelConfig
from .denoising import ObservationDenoising
from .native_data import NativeCorpus
from .registry import build_model


def evaluate(model,objective,corpus,seed=1301,per_dataset=8):
    rng=random.Random(seed);result={};model.eval();objective.eval()
    with torch.no_grad():
        for dataset in sorted(corpus.by_dataset):
            scores=[];identity=[];means=[]
            for i in range(per_dataset):
                sample,_=corpus.sample(rng,dataset=dataset,window_s=2.,device=next(model.parameters()).device,max_streams=1)
                generator=torch.Generator(device=next(model.parameters()).device).manual_seed(seed+i)
                loss,base=objective.compute(model,sample,generator)
                scores.append(float(loss.loss));identity.append(sum(base['identity'].values())/len(base['identity']));means.append(sum(base['channel_mean'].values())/len(base['channel_mean']))
            result[dataset]=dict(loss=sum(scores)/len(scores),identity=sum(identity)/len(identity),channel_mean=sum(means)/len(means),windows=len(scores))
    model.train();objective.train()
    return result


def checkpoint(path,model,objective,optimizer,corpus,step,contributions,args):
    if path.exists():raise FileExistsError(path)
    payload=dict(format='motion-v13-denoising-1',config=model.config_dict(),model=model.state_dict(),objective=objective.state_dict(),optimizer=optimizer.state_dict(),fingerprint=corpus.fingerprint,step=step,contributions=dict(contributions),python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],args=vars(args),limitations=['observation-space denoising, not missing modality reconstruction','independent device streams','no downstream transfer validation'])
    torch.save(payload,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--seed',type=int,default=13);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--device',default='cuda:0');p.add_argument('--eval-every',type=int,default=500);p.add_argument('--resume');p.add_argument('--model-config');p.add_argument('--execute-training',action='store_true');args=p.parse_args()
    if not args.execute_training:p.error('explicit training flag required')
    out=Path(args.out)
    if out.exists() and not args.resume:raise FileExistsError(out)
    out.mkdir(exist_ok=bool(args.resume));random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    if args.steps < 1 or args.eval_every < 1 or args.lr <= 0:
        p.error('positive steps, eval-every and lr required')
    model=build_model(args.model_config).to(args.device)
    objective=ObservationDenoising(model.config.dim,model.config.heads).to(args.device)
    train=NativeCorpus(args.manifest,'train',patch_s=model.config.patch_s,token_budget=model.config.max_tokens)
    valid=NativeCorpus(args.manifest,'validation',patch_s=model.config.patch_s,token_budget=model.config.max_tokens)
    optimizer=torch.optim.AdamW(list(model.parameters())+list(objective.parameters()),lr=args.lr)
    start=0;contributions=collections.Counter()
    if args.resume:
        d=torch.load(args.resume,map_location=args.device,weights_only=True)
        if d.get('format')!='motion-v13-denoising-1':raise ValueError('incompatible denoising checkpoint')
        if d['config']!=model.config_dict():raise ValueError('resume model config differs')
        if d['fingerprint']!=train.fingerprint:raise ValueError('resume corpus differs')
        model.load_state_dict(d['model']);objective.load_state_dict(d['objective']);optimizer.load_state_dict(d['optimizer']);start=d['step'];contributions.update(d['contributions']);random.setstate(d['python_rng']);torch.set_rng_state(d['torch_rng'].cpu());torch.cuda.set_rng_state_all([state.cpu() for state in d['cuda_rng']]) if d['cuda_rng'] else None
    else:
        checkpoint(out/'initial.pt',model,objective,optimizer,train,0,contributions,args)
        before=evaluate(model,objective,valid);(out/'validation_initial.json').write_text(json.dumps(before,indent=2));print('INITIAL',before,flush=True)
    tic=time.monotonic();datasets=sorted(train.by_dataset)
    with (out/'metrics.jsonl').open('a') as log:
        for step in range(start+1,args.steps+1):
            dataset=datasets[(step-1)%len(datasets)]
            sample,_=train.sample(random,dataset=dataset,window_s=random.choice([1.8,2.,2.4,3.2,4.]),device=args.device,max_streams=1)
            optimizer.zero_grad(set_to_none=True);result,base=objective.compute(model,sample)
            if not torch.isfinite(result.loss):raise ValueError('nonfinite loss')
            result.loss.backward();norm=torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(objective.parameters()),1.)
            if not torch.isfinite(norm):raise ValueError('nonfinite gradients')
            optimizer.step();contributions[dataset]+=1
            row=dict(step=step,dataset=dataset,loss=float(result.loss.detach()),targets=result.valid_targets,grad_norm=float(norm),elapsed_s=time.monotonic()-tic)
            log.write(json.dumps(row)+'\n')
            if step%25==0:log.flush();print('STEP',row,flush=True)
            if step%args.eval_every==0 or step==args.steps:
                scores=evaluate(model,objective,valid);(out/f'validation_{step:06d}.json').write_text(json.dumps(scores,indent=2));checkpoint(out/f'step_{step:06d}.pt',model,objective,optimizer,train,step,contributions,args);print('VALIDATION',step,scores,'CONTRIBUTIONS',dict(contributions),flush=True)
    print('DONE',dict(step=args.steps,contributions=dict(contributions),elapsed_s=time.monotonic()-tic),flush=True)


if __name__=='__main__':main()
