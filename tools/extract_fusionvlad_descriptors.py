"""Extract normalized 2048-D descriptors in immutable manifest order."""
import argparse
from pathlib import Path
import numpy as np,torch
from torch.utils.data import DataLoader
from fusionvlad.config import load_config
from fusionvlad.dataset import ViewDataset
from fusionvlad.model import FusionVLAD
from fusionvlad.nclt_manifest import load_manifest

@torch.no_grad()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='config/nclt_fusionvlad.yaml'); p.add_argument('--checkpoint'); p.add_argument('--sequence',action='append'); p.add_argument('--batch-size',type=int,default=8); a=p.parse_args(); cfg=load_config(a.config); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); checkpoint=Path(a.checkpoint or Path(cfg['paths']['run_dir'])/'model_best.pth.tar'); state=torch.load(checkpoint,map_location=device,weights_only=False); mc=cfg['model']; model=FusionVLAD(mc['num_clusters'],mc['branch_dim'],mc['joint_dim'],False).to(device); model.load_state_dict(state['model']); model.eval(); sequences=a.sequence or [cfg['protocol']['database'],*cfg['protocol']['queries']]; out=Path(cfg['paths']['descriptors_root']); out.mkdir(parents=True,exist_ok=True)
 for seq in sequences:
  count=cfg['protocol']['expected_counts'][seq]; manifest=load_manifest(cfg['paths']['manifest_root'],seq,count); dataset=ViewDataset(cfg['paths']['views_root'],seq,count,Path(cfg['paths']['run_dir'])/'normalization_stats.npz',manifest); values=[]
  for top,spherical,_ in DataLoader(dataset,batch_size=a.batch_size): values.append(model(top.to(device),spherical.to(device))['final'].cpu().numpy())
  descriptors=np.concatenate(values).astype(np.float32); expected=(count,2048)
  if descriptors.shape!=expected: raise RuntimeError(f'{seq}: expected descriptor shape {expected}, got {descriptors.shape}')
  if not np.isfinite(descriptors).all(): raise RuntimeError(f'{seq}: descriptors contain non-finite values')
  norm=float(np.linalg.norm(descriptors,axis=1).mean())
  if not np.isclose(norm,1.,atol=1e-3): raise RuntimeError(f'{seq}: mean L2 norm {norm} is not near one')
  np.savez_compressed(out/f'{seq}.npz',descriptors=descriptors,frame_ids=manifest.frame_ids,source_filenames=manifest.source_filenames,scan_timestamps=manifest.scan_timestamps,sequence=seq); print(f'{seq}: {descriptors.shape}, mean norm={norm:.6f}')
if __name__=='__main__': main()
