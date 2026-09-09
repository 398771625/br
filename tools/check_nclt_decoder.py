"""Decode selected manifest frames with the configured verified decoder."""
import argparse
import numpy as np
from fusionvlad.config import load_config
from fusionvlad.nclt_manifest import load_manifest,scan_path
from fusionvlad.nclt_raw import make_raw_loader

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='config/nclt_fusionvlad.yaml'); p.add_argument('--sequence',default='2012-01-15'); p.add_argument('--num-samples',type=int,default=10); a=p.parse_args(); cfg=load_config(a.config)
 m=load_manifest(cfg['paths']['manifest_root'],a.sequence,cfg['protocol']['expected_counts'][a.sequence]); loader=make_raw_loader(cfg)
 for i in np.linspace(0,len(m)-1,min(a.num_samples,len(m)),dtype=int):
  path=scan_path(cfg['paths']['raw_root'],a.sequence,m.source_filenames[i]); points=loader(path)
  if not np.isfinite(points).all(): raise RuntimeError(f'Non-finite decoded points: {path}')
  print(f'{i}: {path.name}, shape={points.shape}')
if __name__=='__main__': main()
