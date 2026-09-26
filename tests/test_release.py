import copy
import unittest
import torch
from qwen35_deploy.convert import quantize
from qwen35_deploy.release import fingerprint,verify_release


class ReleaseTests(unittest.TestCase):
    def test_integrity_and_versions(self):
        self.assertIn('artifact_fingerprint',verify_release())

    def test_fingerprint_does_not_include_local_source_path(self):
        first={'identity':{'source':'private-location-A'},'records':[
            dict(name='w',shape=[1,128],dtype='BF16',sha256='example',bytes=12,format='int8')]}
        second=copy.deepcopy(first);second['identity']['source']='private-location-B'
        self.assertEqual(fingerprint(first),fingerprint(second))
        second['records'][0]['sha256']='changed'
        self.assertNotEqual(fingerprint(first),fingerprint(second))

    def test_converter_quantization(self):
        weight=torch.linspace(-1,1,256).reshape(2,128).bfloat16()
        quant,scale,error,energy=quantize(weight)
        reference_scale=weight.float().abs().amax(-1,keepdim=True)/127
        reference=(weight.float()/reference_scale).round().clamp(-127,127).to(torch.int8)
        self.assertTrue(torch.equal(quant,reference));self.assertTrue(torch.equal(scale,reference_scale))
        self.assertGreaterEqual(error,0);self.assertGreater(energy,0)


if __name__=='__main__':unittest.main()
