"""Source-only FusionVLAD training; held-out query traversals are never opened."""
import argparse,csv,json,platform,shutil
from pathlib import Path
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader,Subset
from fusionvlad.config import load_config
from fusionvlad.dataset import ViewDataset,TupleDataset,compute_normalization_stats
from fusionvlad.losses import fusionvlad_loss
from fusionvlad.model import FusionVLAD
from fusionvlad.nclt_manifest import load_manifest
from fusionvlad.pose_utils import load_aligned_poses,planar_positions

def forward_sets(model,at,asp,pt,ps,nt,ns):
 b,p,n=at.shape[0],pt.shape[1],nt.shape[1]; anchor=model(at,asp); positives=model(pt.flatten(0,1),ps.flatten(0,1)); negatives=model(nt.flatten(0,1),ns.flatten(0,1))
 positives={k:v.reshape(b,p,-1) for k,v in positives.items()}; negatives={k:v.reshape(b,n,-1) for k,v in negatives.items()}; return anchor,positives,negatives
@torch.no_grad()
def extract(model,loader,device):
 model.eval(); return np.concatenate([model(t.to(device),s.to(device))['final'].cpu().numpy() for t,s,_ in loader])
def source_validation_recall(model,views,split,positions,batch,workers,device):
 train=DataLoader(Subset(views,range(split)),batch_size=batch,num_workers=workers); val=DataLoader(Subset(views,range(split,len(views))),batch_size=batch,num_workers=workers)
 db,q=extract(model,train,device),extract(model,val,device); pred=np.argmax(q@db.T,axis=1); distances=np.linalg.norm(positions[split:,None]-positions[None,:split],axis=-1); eligible=np.any(distances<5,axis=1)
 return float(np.mean(distances[np.arange(len(q)),pred][eligible]<5)) if eligible.any() else 0.0

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='config/nclt_fusionvlad.yaml'); p.add_argument('--views-checked',action='store_true',help='Confirm generated preview PNGs passed human inspection'); p.add_argument('--batch-queries',type=int); a=p.parse_args()
 if not a.views_checked: raise RuntimeError('Refusing to train before view sanity check; inspect previews, then pass --views-checked')
 cfg=load_config(a.config); tc=cfg['training']; seed=tc['seed']; np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 sequence=cfg['protocol']['database']; count=cfg['protocol']['expected_counts'][sequence]; manifest=load_manifest(cfg['paths']['manifest_root'],sequence,count); poses=load_aligned_poses(cfg,manifest); positions=planar_positions(poses); split=int(count*tc['train_fraction'])
 stats=Path(cfg['paths']['normalization_stats']); compute_normalization_stats(cfg['paths']['views_root'],sequence,split,stats); views=ViewDataset(cfg['paths']['views_root'],sequence,count,stats,manifest)
 tuples=TupleDataset(views,positions,np.arange(split),tc['positive_distance'],tc['negative_distance'],tc['positives_per_query'],tc['negatives_per_query'],seed,tc['yaw_step_degrees']); actual_batch=a.batch_queries or tc['batch_queries']; loader=DataLoader(tuples,batch_size=actual_batch,shuffle=True,num_workers=tc['workers'])
 mc=cfg['model']; model=FusionVLAD(mc['num_clusters'],mc['branch_dim'],mc['joint_dim'],mc['imagenet_init']).to(device); optimizer=torch.optim.Adam(model.parameters(),lr=tc['learning_rate']); run=Path(cfg['paths']['run_dir']); run.mkdir(parents=True,exist_ok=True)
 used=dict(cfg); used['training']=dict(tc,actual_batch_queries=actual_batch); (run/'config_used.yaml').write_text(yaml.safe_dump(used,sort_keys=False),encoding='utf-8'); shutil.copy2(stats,run/'normalization_stats.npz'); (run/'environment.json').write_text(json.dumps({'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.cuda.is_available(),'device':str(device)},indent=2),encoding='utf-8')
 best=-1.; log=run/'train_log.csv'
 with log.open('w',newline='',encoding='utf-8') as stream:
  writer=csv.writer(stream); writer.writerow(['epoch','loss_total','loss_top','loss_spherical','loss_fusion','source_val_recall_at_1'])
  for epoch in range(1,tc['epochs']+1):
   model.train(); totals=np.zeros(4); seen=0
   for batch_data in loader:
    data=[x.to(device) for x in batch_data]; outputs=forward_sets(model,*data); losses=fusionvlad_loss(*outputs,tc['margin_translation'],tc['margin_rotation']); optimizer.zero_grad(); losses['total'].backward(); optimizer.step(); size=data[0].shape[0]; seen+=size; totals+=size*np.array([losses[k].item() for k in ('total','top','spherical','final')])
   val=source_validation_recall(model,views,split,positions,actual_batch,tc['workers'],device); row=[epoch,*list(totals/seen),val]; writer.writerow(row); stream.flush(); state={'epoch':epoch,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'source_val_recall_at_1':val,'config':used}; torch.save(state,run/'model_latest.pth.tar')
   if val>best: best=val; torch.save(state,run/'model_best.pth.tar')
   print(f'Epoch {epoch:02d}: loss={totals[0]/seen:.5f}, source validation Recall@1={100*val:.2f}')
if __name__=='__main__': main()
