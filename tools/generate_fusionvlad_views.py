"""Generate resumable cached FusionVLAD views in manifest order."""
import argparse
from pathlib import Path
import numpy as np
from fusionvlad.config import load_config
from fusionvlad.local_map import build_local_map
from fusionvlad.nclt_manifest import load_manifest
from fusionvlad.nclt_raw import make_raw_loader
from fusionvlad.pose_utils import load_aligned_poses
from fusionvlad.projections import make_views

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/nclt_fusionvlad.yaml"); p.add_argument("--sequence",action="append"); p.add_argument("--overwrite",action="store_true"); args=p.parse_args()
    cfg=load_config(args.config); sequences=args.sequence or [cfg["protocol"]["database"],*cfg["protocol"]["queries"]]
    raw_loader=make_raw_loader(cfg); settings=cfg["views"]
    for sequence in sequences:
        manifest=load_manifest(cfg["paths"]["manifest_root"],sequence,cfg["protocol"]["expected_counts"][sequence]); poses=load_aligned_poses(cfg,manifest)
        directory=Path(cfg["paths"]["views_root"])/sequence; directory.mkdir(parents=True,exist_ok=True)
        for i in range(len(manifest)):
            output=directory/f"{i:06d}.npz"
            if output.exists() and not args.overwrite: continue
            points=build_local_map(i,manifest,poses,cfg["paths"]["raw_root"],raw_loader,settings["local_window"],settings["voxel_size"],settings["xy_min"],settings["xy_max"],settings["max_range"])
            top,spherical,metrics=make_views(points,settings)
            np.savez_compressed(output,topdown=top,spherical=spherical,frame_id=manifest.frame_ids[i],scan_timestamp=manifest.scan_timestamps[i],source_filename=manifest.source_filenames[i],**metrics)
            if i%100==0: print(f"{sequence}: {i}/{len(manifest)}")
if __name__=="__main__": main()
