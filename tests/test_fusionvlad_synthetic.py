"""Small synthetic checks that require no NCLT data."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from fusionvlad.nclt_manifest import validate_manifest_arrays
from fusionvlad.netvlad import NetVLAD
from fusionvlad.model import FusionVLAD
from fusionvlad.pose_utils import align_poses,parse_pose_rows
from fusionvlad.projections import spherical_projection,topdown_projection

class SyntheticTests(unittest.TestCase):
 def test_netvlad_shape_and_norm(self):
  output=NetVLAD(dim=8,num_clusters=4)(torch.randn(2,8,4,4)); self.assertEqual(tuple(output.shape),(2,32)); self.assertTrue(torch.allclose(torch.linalg.norm(output,dim=1),torch.ones(2),atol=1e-5))
 def test_model_shape_and_l2_norm(self):
  model=FusionVLAD(num_clusters=2).eval()
  with torch.no_grad(): output=model(torch.randn(1,2,64,64),torch.randn(1,10,64,64))['final']
  self.assertEqual(tuple(output.shape),(1,2048)); self.assertTrue(torch.allclose(torch.linalg.norm(output,dim=1),torch.ones(1),atol=1e-5))
 def test_projection_shapes(self):
  points=np.array([[1,1,0.5,2],[2,0,1,4],[-1,-1,-.5,1]],np.float32); top,_=topdown_projection(points); spherical,_=spherical_projection(points,normal_max_nn=3)
  self.assertEqual(top.shape,(2,64,64)); self.assertEqual(spherical.shape,(10,64,64)); self.assertEqual(top.dtype,np.float32)
 def test_manifest_schema(self):
  m=validate_manifest_arrays([9,3],['a.bin','b.bin'],[100,200],np.array('2012-01-15'),2); self.assertEqual(m.frame_ids.tolist(),[9,3])
  with self.assertRaises(ValueError): validate_manifest_arrays([0],['missing_suffix'],[1],'x')
 def test_pose_parsing_and_alignment(self):
  matrix=np.arange(12,dtype=float); ts,poses=parse_pose_rows(np.r_[100,matrix]); self.assertEqual(ts.tolist(),[100]); np.testing.assert_array_equal(poses[0,:3,:4],matrix.reshape(3,4)); self.assertEqual(poses[0,3,3],1)
  aligned=align_poses([102],ts,poses,max_delta=5); np.testing.assert_array_equal(aligned,poses)

if __name__=='__main__': unittest.main()
