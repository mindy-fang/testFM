import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import os
import tarfile
import zipfile
import numpy as np
import pandas as pd
from .audit_inputs import DATA, ROOT, rar_read, rar_list, object_npz
from .build_corpus import Builder, imu, pressure
from .extend_corpus import worker_init

TWENTE_ARCHIVE = Path(os.environ.get('MOTION_TWENTE_ARCHIVE', str(DATA / 'twente/Raw_data.rar')))

SPEC = 'https://moticon.com/wp-content/uploads/2021/09/OpenGo-Sensor-Insole-Specification-A4-RGB-EN-03.03-WEB.pdf'


def foot_groups(side):
    return [dict(cols=list(range(16)), channels=pressure(16, side), modality='pressure'),
            dict(cols=list(range(16,22)), channels=imu('foot', side), modality='imu')]


def convert(job):
    kind, member, out = job
    b = Builder(out)
    if kind == 'twente':
        blob = rar_read(TWENTE_ARCHIVE, member)
        df = pd.read_csv(io.BytesIO(blob), sep=r'\s+')
        required = ['time','side'] + ['pressure'+str(i) for i in range(1,17)] + ['acc'+str(i) for i in range(1,4)] + ['ang'+str(i) for i in range(1,4)]
        if not set(required).issubset(df.columns):
            raise ValueError('Twente source column mismatch')
        if not set(df.side.unique()).issubset({0,1}):
            raise ValueError('unknown insole side')
        for code, side in [(0,'left'),(1,'right')]:
            rows = df[df.side == code]
            if len(rows)<101:
                continue
            x = rows[required[2:]].to_numpy(dtype=float)
            x[:,:16] *= 10
            x[:,16:19] *= 9.80665
            x[:,19:22] *= np.pi/180
            stamp = rows['time'].to_numpy(dtype=float)
            positive = np.diff(stamp)
            positive = positive[positive>0]
            if not len(positive) or not 8 <= np.median(positive) <= 12:
                raise ValueError('Twente timestamp scale not compatible with source100Hz')
            times = (stamp-stamp[0])/1000
            b.put('twente_insole','twente:'+member.split('/')[1],member+'!'+side,x,times,foot_groups(side),time_evidence='measured',evidence='Native TCP insole export; BioMechPro S3_importINSOLE100Hz and procInsole side0 left/1 right; timestamp ms scale cross-checked against100Hz per-source increments, gaps retained; manufacturer raw PR N/cm2 ACC g ANG dps '+SPEC+'; no cross-foot synchronization',source_sha=hashlib.sha256(blob).hexdigest())
    elif kind == 'whip':
        with zipfile.ZipFile(DATA/'whip/dataset.zip') as z:
            blob = z.read(member)
        sequence = member.split('/')[1]
        actor = sequence.split('_seq')[0]
        subject = 'whip:'+actor
        action = Path(member).stem
        if action.startswith('test_') and not actor.startswith('test_'):
            return [], dict(source=member,reason='official unseen-action test; excluded from pretraining across non-test actors')
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            count = int(t.extractfile('n_frames.txt').read())
            data = object_npz(t.extractfile('insoles/readings.npz').read())
        for side in ['left','right']:
            fields = data[side]
            x = np.concatenate([fields['pressure'],fields['acceleration'],fields['angular_vel']],axis=1).astype(float)
            if x.shape!=(count,22):
                raise ValueError('WHIP frame count or channel contract')
            x[:,:16] *= 10
            x[:,16:19] *= 9.80665
            x[:,19:22] *= np.pi/180
            b.put('whip_insole',subject,member+'!'+side,x,np.arange(count)/30,foot_groups(side),split='test' if actor.startswith('test_') else None,time_evidence='verified_relative_grid',evidence='WHIP paper2607.09780: raw Moticon insole, released synchronized signals30Hz; official loader b7d058306c6c6ba15d0cc2d227f536af3c1b9697 load_insoles pass-through; manufacturer native units '+SPEC+'; unknown resampling dependency and sync error, independent streams only',source_sha=hashlib.sha256(blob).hexdigest())
    return b.rows, None


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--sources',nargs='+',default=['twente','whip']);args=p.parse_args()
    if not 1<=args.workers<=8:
        p.error('use 1..8 workers')
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    builder=Builder(args.out);jobs=[];excluded=[]
    if 'twente' in args.sources:
        jobs.extend(('twente',n,args.out) for n in rar_list(TWENTE_ARCHIVE) if any(folder in n for folder in ('/Insoles/', '/Insole/')) and n.endswith('.txt'))
    if 'whip' in args.sources:
        with zipfile.ZipFile(DATA/'whip/dataset.zip') as z:
            jobs.extend(('whip',n,args.out) for n in sorted(z.namelist()) if n.endswith('.tar') and '/actions/' in n)
    print('JOBS',len(jobs),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init) as pool:
        futures={pool.submit(convert,j):j for j in jobs}
        for i,f in enumerate(as_completed(futures),1):
            job=futures[f]
            try:
                rows, exclusion=f.result();builder.rows.extend(rows)
                if exclusion:excluded.append(exclusion)
            except Exception as e:
                builder.errors.append(dict(dataset=job[0],source=job[1],error=repr(e)))
            if i%20==0 or i==len(jobs):
                builder.save();print('CONVERTED',i,'/',len(jobs),'records',len(builder.rows),'errors',len(builder.errors),'excluded',len(excluded),flush=True)
    builder.audit=[dict(dataset=s,records=sum(r['dataset']==s+'_insole' for r in builder.rows)) for s in args.sources]
    builder.save()
    (Path(args.out)/'excluded_sources.json').write_text(json.dumps(excluded,indent=2))
    if builder.errors:
        raise RuntimeError('partial conversion; inspect errors')
    print('DONE',len(builder.rows),flush=True)


if __name__=='__main__':main()
