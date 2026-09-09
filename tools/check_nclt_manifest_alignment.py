"""Verify exact manifest schema/count/order and every manifest-addressed raw scan."""
import argparse
from fusionvlad.config import load_config
from fusionvlad.nclt_manifest import load_manifest,scan_path

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',default='config/nclt_fusionvlad.yaml'); a=p.parse_args(); cfg=load_config(a.config)
 for seq,count in cfg['protocol']['expected_counts'].items():
  manifest=load_manifest(cfg['paths']['manifest_root'],seq,count); missing=[str(scan_path(cfg['paths']['raw_root'],seq,n)) for n in manifest.source_filenames if not scan_path(cfg['paths']['raw_root'],seq,n).is_file()]
  if missing: raise RuntimeError(f'{seq}: {len(missing)} missing scans; first={missing[0]}')
  print(f'{seq}: {len(manifest)}/{count} exists; frame ordering retained')
if __name__=='__main__': main()
