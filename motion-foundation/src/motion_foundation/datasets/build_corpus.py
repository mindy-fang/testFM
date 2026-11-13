from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import wfdb
from ..adapters import sha256
from .audit_inputs import DATA, ROOT, rar_list, rar_read


def imu(site='unknown', side='unknown'):
    return [dict(quantity=q, unit=u, axis=a, site=site, side=side) for q,u in [('acc','m/s^2'),('gyro','rad/s')] for a in 'xyz']


def pressure(n, side='unknown'):
    return [dict(quantity='pressure',unit='kPa',site='foot',side=side) for _ in range(n)]


def shoe_groups():
    return [dict(cols=list(range(16)),channels=pressure(16,'left'),modality='pressure'),dict(cols=list(range(16,32)),channels=pressure(16,'right'),modality='pressure'),dict(cols=list(range(32,38)),channels=imu('foot','left'),modality='imu'),dict(cols=list(range(38,44)),channels=imu('foot','right'),modality='imu')]


def assign(subject):
    v=int(hashlib.sha256(('v13-corpus-20261113:'+subject).encode()).hexdigest()[:8],16)%10
    return 'validation' if v==0 else 'test' if v==1 else 'train'


from ..corpus import Builder as CorpusBuilder


class Builder(CorpusBuilder):

    def legacy(self):
        idx=json.loads((ROOT/'corpus_v11_research_20261109/source_index_frozen.json').read_text())
        for r in idx['rows']:
            if r['dataset'] not in ('grouvel','imbalansense','underpressure') or not r['train_admitted'] or r['proposed_split']=='quarantine':continue
            if sha256(r['grid_path'])!=r['grid_sha256']:raise ValueError('legacy grid fingerprint')
            with np.load(r['grid_path'],allow_pickle=False) as z:
                meta=json.loads(Path(r['grid_path']).with_suffix('.json').read_text());radius=meta['filter']['half_width']/meta['filter']['source_hz']
                self.put(r['dataset'],r['subject_key'],r['source_id'],z['values'],z['time_s'],shoe_groups(),observed=z['available'],split=r['proposed_split'],dependency=radius,evidence='frozen FIR physical units and timestamp audit',source_sha=r['member_sha256'],spans=r['quality']['eligible_spans'])
        m=json.loads((ROOT/'v12/body_corpus/manifest.json').read_text())
        sites=[('foot','left'),('foot','right'),('shank','left'),('shank','right'),('thigh','left'),('thigh','right'),('sacrum','unknown'),('trunk','unknown')]
        groups=[dict(cols=list(range(i*6,(i+1)*6)),channels=imu(*site),modality='imu') for i,site in enumerate(sites)]
        for r in m['rows']:
            if r['split']=='quarantine':continue
            if sha256(r['path'])!=r['sha256']:raise ValueError('body fingerprint')
            with np.load(r['path'],allow_pickle=False) as z:
                self.put('grouvel_body',r['subject_key'],r['source_id'],z['values'],np.arange(r['rows'])/r['fs'],groups,observed=z['observed'],split=r['split'],time_evidence='verified_relative_grid',evidence='CSV frame count and paired C3D rate verified; resampling dependency unknown',source_sha=r['member_sha256'])

    def pd(self):
        from .weargait import BODY_SITES
        m=json.loads((ROOT/'weargait_pd_corpus_v2/manifest.json').read_text())
        groups=[]
        for i,(_,s) in enumerate(BODY_SITES):
            side='right' if s.startswith('right_') else 'left' if s.startswith('left_') else 'unknown'
            site=s.split('_',1)[1] if side!='unknown' else s
            groups.append(dict(cols=list(range(i*6,(i+1)*6)),channels=imu(site,side),modality='imu'))
        for r in m['rows']:
            if sha256(r['path'])!=r['sha256']:raise ValueError('PD fingerprint')
            with np.load(r['path'],allow_pickle=False) as z:
                self.put('weargait_pd_'+r['stream'],r['subject_key'],r['source_id'],z['x'],z['time_s'],shoe_groups() if r['stream']=='insole' else groups,observed=z['observed'],split=r['split'],transform='asinh_reference_v1',time_evidence='verified_relative_grid',evidence='100Hz export grid verified; Table S5 unit conversion; raw preprocessing dependency unknown',source_sha=r['source_sha256'])

    def physionet(self):
        for p in sorted((DATA/'physionet_pedar').glob('*.hea')):
            r=wfdb.rdrecord(str(p.with_suffix('')))
            if not all(u=='N/cm2' for u in r.units):raise ValueError('unexpected WFDB units')
            n=r.p_signal.shape[1]
            if n!=198:raise ValueError('unknown foot grouping')
            groups=[dict(cols=list(range(99)),channels=pressure(99,'left'),modality='pressure'),dict(cols=list(range(99,198)),channels=pressure(99,'right'),modality='pressure')]
            # Foot labels stay unknown unless header explicitly identifies anatomical laterality.
            for g in groups:
                for c in g['channels']:c['side']='unknown'
            self.put('physionet_pedar','physionet_pedar:'+p.stem,p.stem,r.p_signal*10,np.arange(r.sig_len)/r.fs,groups,time_evidence='verified_relative_grid',evidence='WFDB header frequency, signal calibration and N/cm2 unit; m1/m2 laterality unknown',source_sha=sha256(p.with_suffix('.dat')))

    def pedar(self):
        with zipfile.ZipFile(DATA/'pedar_cycling/Pedar Data.zip') as z:
            for n in sorted(z.namelist()):
                if not n.endswith('.mat') or n.startswith('__MACOSX'):continue
                b=z.read(n);d=scipy.io.loadmat(io.BytesIO(b),simplify_cells=True)
                if 'insolepressures' not in d:continue
                x=np.asarray(d['insolepressures'],dtype=float);area=np.asarray(d['insoleareas']);force=np.asarray(d['insoleforce_cell'])
                valid=x>0
                if not valid.any():continue
                ratio=force[valid]/(x*area[None,:])[valid]
                if not np.allclose(ratio,1,rtol=.001,atol=.001):raise ValueError('pressure Pa/force N/area m2 contract differs')
                fs=float(d['pedar_sf']);subject='pedar_cycling:'+str(d['subjstr'])
                self.put('pedar_cycling',subject,n,x/1000,np.arange(len(x))/fs,[dict(cols=list(range(x.shape[1])),channels=pressure(x.shape[1]),modality='pressure')],time_evidence='verified_relative_grid',evidence='native insolepressures and pedar_sf; Pa verified by per-cell force/area relation; side unknown',source_sha=hashlib.sha256(b).hexdigest())

    def second_skin(self):
        with zipfile.ZipFile(DATA/'second_skin/OpenSource_Dataset.zip') as z:
            for n in sorted(z.namelist()):
                if '/MachineLearning/' not in n or not n.endswith('.csv'):continue
                b=z.read(n);df=pd.read_csv(io.BytesIO(b));groups=[];columns=[]
                prefixes=sorted({c.rsplit('_acc_',1)[0] for c in df if re.fullmatch(r'.+_acc_[xyz]',c) and not c.startswith('sim_')})
                for prefix in prefixes:
                    selected=[f'{prefix}_{q}_{a}' for q in ['acc','gyro'] for a in 'xyz' if f'{prefix}_{q}_{a}' in df]
                    desc=[];side='left' if prefix.startswith('left_') else 'right' if prefix.startswith('right_') else 'unknown'
                    site=prefix.split('_',1)[1] if side!='unknown' else {'back':'trunk','pelvis':'pelvis'}.get(prefix,prefix)
                    for c in selected:
                        q,a=c.rsplit('_',2)[-2:];desc.append(dict(quantity=q,unit='m/s^2' if q=='acc' else 'rad/s',axis=a,site=site,side=side))
                    groups.append(dict(cols=list(range(len(columns),len(columns)+len(selected))),channels=desc,modality='imu'));columns+=selected
                if columns:self.put('second_skin','second_skin:'+n.split('/')[2],n,df[columns].to_numpy(),df['time'].to_numpy(),groups,evidence='archive README: real sensor acc m/s2 and gyro rad/s, time seconds; no sim, orientation, CoP or model outputs',source_sha=hashlib.sha256(b).hexdigest())

    def xsensor(self):
        arc=DATA/"xsensor_locomotion/PARTICIPANT'S DATA CODE.rar"
        for member in rar_list(arc):
            if 'PLANTAR' not in member or not member.endswith('.zip'):continue
            with zipfile.ZipFile(io.BytesIO(rar_read(arc,member))) as z:
                for n in z.namelist():
                    if not n.lower().endswith(('.csv','.txt')):continue
                    b=z.read(n);lines=list(csv.reader(io.StringIO(b.decode('utf-8-sig'))))
                    units=next((r[1] for r in lines if r and r[0]=='Units:'),None)
                    if units!='kPa':raise ValueError('XSENSOR units unknown')
                    frames=collections.defaultdict(list);stamp=None;sensor=None;date=None;shape=None
                    i=0
                    while i<len(lines):
                        row=lines[i];i+=1
                        if not row:continue
                        if row[0]=='Date':date=row[1].strip()
                        elif row[0]=='Time':stamp=pd.Timestamp(date+' '+row[1].strip()).value/1e9
                        elif row[0]=='SENSOR:':sensor=row[1]
                        elif row[0]=='Rows':shape=[int(row[1]),0]
                        elif row[0]=='Columns':shape[1]=int(row[1])
                        elif row[0]=='SENSELS':
                            block=lines[i:i+shape[0]];i+=shape[0]
                            values=np.array([[float(v.replace(',','.')) for v in r] for r in block])
                            if values.shape!=tuple(shape):raise ValueError('ragged XSENSOR frame')
                            frames[sensor].append((stamp,values.flatten()))
                    for sensor,fr in frames.items():
                        if len(fr)<2:continue
                        t=np.array([v[0] for v in fr]);x=np.stack([v[1] for v in fr]);side='left' if '-LF ' in sensor else 'right' if '-RF ' in sensor else 'unknown'
                        subject='xsensor_locomotion:'+n.split('/')[0]
                        self.put('xsensor_locomotion',subject,member+'!'+n+'!'+sensor,x,t-t[0],[dict(cols=list(range(x.shape[1])),channels=pressure(x.shape[1],side),modality='pressure')],evidence='CSV kPa header and per-frame date/time; rectangular sensor grid retained without invented anatomical mapping',source_sha=hashlib.sha256(b).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--sources',nargs='+',default=['legacy','pd','physionet','pedar','second_skin','xsensor']);args=p.parse_args()
    builder=Builder(args.out)
    for name in args.sources:builder.attempt(name,getattr(builder,name))
    print(json.dumps(dict(records=len(builder.rows),audit=builder.audit,errors=builder.errors)),flush=True)


if __name__=='__main__':main()
