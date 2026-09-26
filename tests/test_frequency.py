import sys
from pathlib import Path
import unittest
import types
import torch
from qwen35_deploy.runtime.frequency_seed import select_seeds,install
from qwen35_deploy.runtime.gpu_routes import GPUPlan
from qwen35_deploy.operators.int8 import Int8Experts


class FrequencySeedTests(unittest.TestCase):
    def test_preserve_last_and_frequent(self):
        self.assertEqual(select_seeds([[3,4],[3,5],[3,4],[7,8]],4),[4,3,7,8])

    def test_tie_break_is_deterministic(self):
        self.assertEqual(select_seeds([[3,2],[1,4],[7,8]],3),[1,7,8])

    def test_no_extra_inactive_experts(self):
        self.assertEqual(select_seeds([[2,1],[2,1]],45),[2,1])

    def test_invalid_quota(self):
        with self.assertRaises(ValueError):select_seeds([[1,2]],1)

    def fixture(self):
        cache=types.SimpleNamespace(seed=lambda *a:None,entries={},bytes=0,reserved_bytes=2316)
        pipeline=types.SimpleNamespace(host=[[torch.empty(256,1,dtype=torch.int8)]],
            run=lambda *a:None,packed_host=None,cache=cache)
        plan=GPUPlan(pipeline);pipeline.run=types.MethodType(plan.run,pipeline)
        modules=[Int8Experts(i,None,None,cache) for i in range(40)]
        for m in modules:m.prefill_pipeline=pipeline
        model=types.SimpleNamespace(modules=lambda:iter(modules),
            config=types.SimpleNamespace(text_config=types.SimpleNamespace(num_experts_per_tok=8)))
        return model,cache,pipeline

    def test_capacity_and_double_install(self):
        model,cache,pipeline=self.fixture()
        self.assertEqual(install(model,cache).quota,45)
        with self.assertRaises(ValueError):install(model,cache)

    def test_wrong_pipeline_or_cache_rejected(self):
        for kind in ('override','packed','occupied','wrong_cache'):
            model,cache,pipeline=self.fixture()
            if kind=='override':pipeline.run=lambda *a:None
            elif kind=='packed':pipeline.packed_host={0:object()}
            elif kind=='occupied':cache.bytes=1
            else:pipeline.cache=object()
            with self.assertRaises(ValueError):install(model,cache)


if __name__=='__main__':unittest.main()
