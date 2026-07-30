"""Why do empty patches score high? Correlate per-patch FTLE with actual image content."""
import json, pickle
from pathlib import Path
import cv2, hydra, lmdb, numpy as np, torch
from torchvision import transforms
from server_single_max import load_model

AM=[0.45678952,0.00051019,0.50954217,0.21926114]; ASD=[0.03182372,0.01151787,0.03419121,0.41397065]
PM=[0.4564166,0.00056233,0.50817657,0.21921302];  PSD=[0.03217997,0.01056713,0.0327194,0.4139551]
NH,NP,N,GRID=3,8,50,14; dev='cuda'
with hydra.initialize(config_path="conf",version_base=None): cfg=hydra.compose(config_name="train")
model=load_model(Path('outputs/model_latest_single.pth'),cfg,dev); model.eval()
tf=transforms.Compose([transforms.Resize(cfg.img_size),transforms.CenterCrop(cfg.img_size),
                       transforms.Normalize([0.5]*3,[0.5]*3)])
am=torch.tensor(AM,device=dev);asd=torch.tensor(ASD,device=dev)
pm=torch.tensor(PM,device=dev);psd=torch.tensor(PSD,device=dev)
lab=json.load(open('/home/sanger/wksp/panda_express/labels_noise100.json'))
env=lmdb.open('/home/sanger/wksp/panda_express/tasks/jenga_noise_50/jenga_single_100.lmdb',readonly=True,lock=False)
def dec(b): return cv2.cvtColor(cv2.imdecode(np.frombuffer(b,np.uint8),1),cv2.COLOR_BGR2RGB)
def crop(im,s=224):
    h,w=im.shape[:2]; sc=s/min(h,w); im=cv2.resize(im,(int(w*sc),int(h*sc)),interpolation=cv2.INTER_AREA)
    sh,sw=im.shape[:2]; return im[(sh-s)//2:(sh+s)//2,(sw-s)//2:(sw+s)//2]
span=NH+NP
LAM=[];DS=[];DE=[];OCC=[]
with env.begin() as t:
    meta=pickle.loads(t.get(b'__metadata__'))
    eps=[e for e,v in lab.items() if v['outcome']=='failure'][:10]
    for ep in eps:
        keys=meta['episodes'][ep]['keys']['cam2']; fs=lab[ep]['failure_step']
        s=max(0,((fs-NH-NP+1)//NP)*NP)
        if s+span>=len(keys): continue
        acts=pickle.loads(t.get(f'{ep}_actions'.encode())); props=pickle.loads(t.get(f'{ep}_proprio'.encode()))
        raw=[dec(t.get(keys[s+i].encode())) for i in range(span)]
        g=crop(raw[0]).astype(np.float32).mean(2)
        occ=g.reshape(GRID,16,GRID,16).std(axis=(1,3)).reshape(-1)
        vis=torch.from_numpy(np.stack([np.transpose(r,(2,0,1)) for r in raw])).float().to(dev)/255.
        vis=tf(vis)
        obs={'visual':vis[:NH].unsqueeze(0).repeat(N,1,1,1,1),
             'proprio':((torch.from_numpy(props[s:s+NH]).float().to(dev)-pm)/psd).unsqueeze(0).repeat(N,1,1)}
        a=torch.from_numpy(acts[s:s+span]).float().to(dev).unsqueeze(0).repeat(N,1,1)
        a[1:,:,:3]+=torch.randn(N-1,span,3,device=dev)*0.05; a=(a-am)/asd
        with torch.no_grad(): z,_=model.rollout(obs,a)
        zv=z['visual']; zo,zn=zv[0:1],zv[1:]
        ds=(1-torch.nn.functional.cosine_similarity(zn[:,NH],zo[:,NH],dim=-1))+1e-4
        de=(1-torch.nn.functional.cosine_similarity(zn[:,-1],zo[:,-1],dim=-1))+1e-4
        lam=(1.0/NP)*torch.log(de/ds)
        LAM.append(lam.mean(0).cpu().numpy()); DS.append(ds.mean(0).cpu().numpy())
        DE.append(de.mean(0).cpu().numpy()); OCC.append(occ)
        print('.',end='',flush=True)
print()
LAM=np.stack(LAM);DS=np.stack(DS);DE=np.stack(DE);OCC=np.stack(OCC)
print(f'\n{len(LAM)} chunks x 196 patches\n')
q=np.quantile(OCC.flatten(),[0,.2,.4,.6,.8,1.0])
print(f"{'patch content (occupancy)':<28}{'n':>6}{'mean d_start':>14}{'mean d_end':>12}{'mean FTLE':>11}")
print('-'*72)
for i in range(5):
    m=(OCC>=q[i])&(OCC<q[i+1]+(1e-9 if i==4 else 0))
    lbl=f'{q[i]:.1f} - {q[i+1]:.1f}' + ('   <- blank' if i==0 else ('   <- textured' if i==4 else ''))
    print(f'{lbl:<28}{m.sum():>6}{DS[m].mean():>14.6f}{DE[m].mean():>12.6f}{LAM[m].mean():>11.4f}')
print()
top=np.argsort(-LAM.flatten())[:200]; bot=np.argsort(-DE.flatten())[:200]
print(f'top-200 patches by FTLE      : mean occupancy {OCC.flatten()[top].mean():6.2f}, mean d_end {DE.flatten()[top].mean():.5f}')
print(f'top-200 patches by d_end     : mean occupancy {OCC.flatten()[bot].mean():6.2f}, mean d_end {DE.flatten()[bot].mean():.5f}')
print(f'overall mean occupancy       : {OCC.mean():6.2f}')
c=np.corrcoef(OCC.flatten(),LAM.flatten())[0,1]; c2=np.corrcoef(OCC.flatten(),DE.flatten())[0,1]
print(f'\ncorrelation(content, FTLE)  = {c:+.3f}   <- negative means emptier patches score HIGHER')
print(f'correlation(content, d_end) = {c2:+.3f}')
