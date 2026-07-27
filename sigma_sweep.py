import json,pickle,sys
from pathlib import Path
import cv2,hydra,lmdb,numpy as np,torch
from torchvision import transforms
sys.path.insert(0,'/home/sanger/wksp/dino_wm')
from server_single_max import load_model
AM=[0.45678952,0.00051019,0.50954217,0.21926114]; ASD=[0.03182372,0.01151787,0.03419121,0.41397065]
PM=[0.4564166,0.00056233,0.50817657,0.21921302];  PSD=[0.03217997,0.01056713,0.0327194,0.4139551]
NH,NP,MASK=3,8,28; N_PERT=24
dev='cuda'
with hydra.initialize(config_path="conf",version_base=None): cfg=hydra.compose(config_name="train")
model=load_model(Path('/home/sanger/wksp/dino_wm/outputs/model_latest_single.pth'),cfg,dev); model.eval()
tf=transforms.Compose([transforms.Resize(cfg.img_size),transforms.CenterCrop(cfg.img_size),
                       transforms.Normalize([0.5]*3,[0.5]*3)])
am=torch.tensor(AM,device=dev);asd=torch.tensor(ASD,device=dev)
pm=torch.tensor(PM,device=dev);psd=torch.tensor(PSD,device=dev)
lab=json.load(open('/home/sanger/wksp/panda_express/labels_noise50.json'))
env=lmdb.open('/home/sanger/wksp/panda_express/tasks/jenga_noise_50/jenga_single.lmdb',readonly=True,lock=False)
def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b,np.uint8),1),cv2.COLOR_BGR2RGB)
SIGMAS=[0.001,0.002,0.005,0.01,0.02,0.05]
out={s:{'failure':[],'success':[]} for s in SIGMAS}
span=NH+NP
with env.begin() as txn:
    meta=pickle.loads(txn.get(b'__metadata__')); targets=[];ns=0
    for ep,v in lab.items():
        if ep not in meta['episodes']: continue
        keys=meta['episodes'][ep]['keys']['cam2']
        if v['outcome']=='failure':
            s=v['failure_step']-8
            if 0<=s and s+span<len(keys): targets.append((ep,s,'failure'))
        elif ns<12:
            s=len(keys)//2
            if s+span<len(keys): targets.append((ep,s,'success')); ns+=1
    for ep,start,kind in targets:
        keys=meta['episodes'][ep]['keys']['cam2']
        acts=pickle.loads(txn.get(f'{ep}_actions'.encode())); props=pickle.loads(txn.get(f'{ep}_proprio'.encode()))
        raw=[dec(txn.get(keys[start+t].encode())) for t in range(span)]
        vis=torch.from_numpy(np.stack([np.transpose(r,(2,0,1)) for r in raw])).float().to(dev)/255.
        vis=tf(vis).unsqueeze(0)
        pro=((torch.from_numpy(props[start:start+span]).float().to(dev)-pm)/psd).unsqueeze(0)
        a0=torch.from_numpy(acts[start:start+span]).float().to(dev)
        obs0={'visual':vis[:,:NH].repeat(N_PERT,1,1,1,1),'proprio':pro[:,:NH].repeat(N_PERT,1,1)}
        with torch.no_grad():
            for sg in SIGMAS:
                A=a0.unsqueeze(0).repeat(N_PERT,1,1).clone()
                A[1:,:,:3]+=torch.randn(N_PERT-1,span,3,device=dev)*sg
                A=(A-am)/asd
                z,_=model.rollout(obs0,A); zv=z['visual']
                zo,zn=zv[0:1],zv[1:]
                ds=(1-torch.nn.functional.cosine_similarity(zn[:,NH],zo[:,NH],dim=-1))+1e-4
                de=(1-torch.nn.functional.cosine_similarity(zn[:,-1],zo[:,-1],dim=-1))+1e-4
                lam=(1.0/NP)*torch.log(de/ds)
                lam[:,:MASK]=-float('inf'); lam[de<1e-3]=-float('inf')
                out[sg][kind].append(float(lam.max()))
        print('.',end='',flush=True)
print()
print(f"{'sigma(m)':>9} | {'FAIL mean':>10} {'SUCC mean':>10} | {'gap':>7} | {'AUC':>6}")
print('-'*54)
for sg in SIGMAS:
    f=np.array(out[sg]['failure']); s=np.array(out[sg]['success'])
    auc=np.mean([(x>y)+0.5*(x==y) for x in f for y in s])
    print(f'{sg:>9.3f} | {f.mean():>10.4f} {s.mean():>10.4f} | {f.mean()-s.mean():>+7.4f} | {auc:>6.3f}')
json.dump({str(k):v for k,v in out.items()},open('results/sigma_sweep.json','w'),indent=2)
