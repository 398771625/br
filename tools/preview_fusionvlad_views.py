"""Render cached views for mandatory human sanity checking before training."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from fusionvlad.config import load_config
from fusionvlad.nclt_manifest import load_manifest

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='config/nclt_fusionvlad.yaml'); p.add_argument('--sequence',default='2012-01-15'); p.add_argument('--num-samples',type=int,default=10); a=p.parse_args(); cfg=load_config(a.config)
 m=load_manifest(cfg['paths']['manifest_root'],a.sequence,cfg['protocol']['expected_counts'][a.sequence]); rng=np.random.default_rng(cfg['training']['seed']); ids=np.sort(rng.choice(len(m),min(a.num_samples,len(m)),replace=False)); out=Path(cfg['paths']['preview_root']); out.mkdir(parents=True,exist_ok=True)
 for i in ids:
  path=Path(cfg['paths']['views_root'])/a.sequence/f'{i:06d}.npz'
  with np.load(path,allow_pickle=False) as d:
   top=d['topdown']; spherical=d['spherical']; counts=(int(d['local_point_count']),int(d['valid_topdown_cells']),int(d['valid_spherical_cells']))
  fig,axes=plt.subplots(2,6,figsize=(18,6)); images=[top[0],*[spherical[2*j] for j in range(5)],top[1],*[spherical[2*j+1] for j in range(5)]]; titles=['top max height',*[f'range slice {j}' for j in range(5)],'top mean intensity',*[f'normal angle {j}' for j in range(5)]]
  for ax,image,title in zip(axes.flat,images,titles): ax.imshow(image,cmap='viridis'); ax.set_title(title); ax.axis('off')
  fig.suptitle(f'{a.sequence} index={i} frame_id={m.frame_ids[i]}'); fig.tight_layout(); target=out/f'{a.sequence}_{i:06d}.png'; fig.savefig(target,dpi=150); plt.close(fig)
  print(f'{target}: local point count={counts[0]}, valid top-down cells={counts[1]}, valid spherical cells={counts[2]}')
 print('Human action required: inspect every preview before starting training.')
if __name__=='__main__': main()
