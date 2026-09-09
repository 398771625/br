"""Strict NCLT Standard Recall@1/@5 evaluation with denominator guards."""
import argparse,csv,json
from pathlib import Path
import faiss,numpy as np
from scipy.spatial import cKDTree
from fusionvlad.config import load_config
from fusionvlad.nclt_manifest import load_manifest
from fusionvlad.pose_utils import load_aligned_poses,planar_positions

def load_descriptors(root,manifest):
 path=Path(root)/f'{manifest.sequence}.npz'
 with np.load(path,allow_pickle=False) as data:
  required=('descriptors','frame_ids','source_filenames','scan_timestamps'); missing=[k for k in required if k not in data]
  if missing: raise RuntimeError(f'{path}: missing {missing}')
  desc=np.asarray(data['descriptors'],np.float32)
  for key,expected in (('frame_ids',manifest.frame_ids),('source_filenames',manifest.source_filenames),('scan_timestamps',manifest.scan_timestamps)):
   if not np.array_equal(data[key].astype(str) if key=='source_filenames' else data[key],expected.astype(str) if key=='source_filenames' else expected): raise RuntimeError(f'{path}: {key} differs from immutable manifest ordering')
 if desc.shape!=(len(manifest),2048) or not np.isfinite(desc).all(): raise RuntimeError(f'{path}: invalid descriptors {desc.shape}')
 if not np.allclose(np.linalg.norm(desc,axis=1),1,atol=1e-3): raise RuntimeError(f'{path}: descriptors are not L2 normalized')
 return desc

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='config/nclt_fusionvlad.yaml'); a=p.parse_args(); cfg=load_config(a.config); protocol=cfg['protocol']; root=cfg['paths']['descriptors_root']; db_seq=protocol['database']; db_manifest=load_manifest(cfg['paths']['manifest_root'],db_seq,protocol['expected_counts'][db_seq]); db_desc=load_descriptors(root,db_manifest); db_pos=planar_positions(load_aligned_poses(cfg,db_manifest)); index=faiss.IndexFlatL2(2048); index.add(db_desc); tree=cKDTree(db_pos); results=[]
 for seq in protocol['queries']:
  manifest=load_manifest(cfg['paths']['manifest_root'],seq,protocol['expected_counts'][seq]); desc=load_descriptors(root,manifest); positions=planar_positions(load_aligned_poses(cfg,manifest)); _,pred=index.search(desc,5); correct1=correct5=valid=0
  for position,neighbors in zip(positions,pred):
   candidates=tree.query_ball_point(position,protocol['gt_threshold']); gt={i for i in candidates if np.linalg.norm(db_pos[i]-position)<protocol['gt_threshold']}
   if not gt: continue
   valid+=1; correct1+=int(int(neighbors[0]) in gt); correct5+=int(any(int(i) in gt for i in neighbors))
  no_gt=len(manifest)-valid; expected_valid,expected_no_gt=protocol['expected_denominators'][seq]
  if (valid,no_gt)!=(expected_valid,expected_no_gt): raise RuntimeError(f'{seq}: denominator mismatch: got valid_queries={valid}, no_gt_queries={no_gt}; expected {expected_valid}, {expected_no_gt}. Final Recall is forbidden.')
  results.append({'query':seq,'correct_at_1':correct1,'correct_at_5':correct5,'valid_queries':valid,'no_gt_queries':no_gt,'recall_at_1':100*correct1/valid,'recall_at_5':100*correct5/valid})
 out=Path(cfg['paths']['results_root']); out.mkdir(parents=True,exist_ok=True); mean1=float(np.mean([r['recall_at_1'] for r in results])); mean5=float(np.mean([r['recall_at_5'] for r in results])); payload={'database':db_seq,'gt_threshold_m':protocol['gt_threshold'],'queries':results,'mean_recall_at_1':mean1,'mean_recall_at_5':mean5}
 (out/'fusionvlad_nclt_standard.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
 with (out/'fusionvlad_nclt_standard.csv').open('w',newline='',encoding='utf-8') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
 for r in results: print(f"DB {db_seq} -> Query {r['query']}:\nRecall@1 {r['recall_at_1']:.2f}\nRecall@5 {r['recall_at_5']:.2f}\n")
 print(f'Mean Recall@1:\n{mean1:.2f}\n\nMean Recall@5:\n{mean5:.2f}')
if __name__=='__main__': main()
