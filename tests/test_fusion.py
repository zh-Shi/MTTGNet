"""Fusion behavior, builder forwarding, and exact archived checkpoint loading."""
import copy
import unittest
from types import SimpleNamespace

import numpy as np  # Initialize the server's MKL runtime before PyTorch/libgomp.
import torch

from src.fusion_config import config_for_saved_fusion, record_fusion_config
from src.mttgnet import MultiSourceGateFusion
from src.mttgnet_test import MTTGNetTestWrapper


class FusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.cfg = {'model_params': {'arch': 'clean', 'hidden_dim': 16,
                    'use_memory': False, 'dropout': 0., 'var_encoder_type': 'gat_compact',
                    'mem_clean': True, 'decoder_bypass': False, 'query_refine': False,
                    'time_bias_mode': 'additive', 'co2_mode': 'counterfactual',
                    'num_gru_layers': 2},
                    'dataset': {'horizon': 3, 'recent_length': 4},
                    'data': {'target_index': 0, 'co2_index': 1}}
        self.args = {'x_recent': torch.rand(2, 4, 2),
                     'yearly_anchor': torch.rand(2, 3, 2, 2),
                     'yearly_mask': torch.ones(2, 3, 2, dtype=torch.bool),
                     'yearly_gaps': torch.full((2, 3, 2), 365.),
                     'future_calendar': torch.rand(2, 3, 5)}

    def test_two_active_weights_and_branch_gradients(self):
        fusion = MultiSourceGateFusion(16, 3, num_sources=2)
        temporal = torch.randn(2, 3, 16, requires_grad=True)
        periodic = torch.randn(2, 3, 16, requires_grad=True)
        out, gate = fusion(temporal, periodic, None, self.args['future_calendar'])
        torch.testing.assert_close(gate.sum(-1), torch.ones(2, 3))
        torch.testing.assert_close(out, fusion.norm(gate[..., :1]*temporal + gate[..., 1:]*periodic))
        out[..., 0].sum().backward()
        for tensor in (temporal, periodic, fusion.gate_predictor[-1].weight):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_default_clean_model_preserves_zero_third_source(self):
        model = MTTGNetTestWrapper(2, hidden_dim=16, horizon=3, use_memory=False)
        seen = []
        hook = model.model.fusion.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs['memory_ctx']), with_kwargs=True)
        result = model(**self.args)
        hook.remove()
        self.assertEqual(result['fusion_gate'].shape, (2, 3, 3))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].shape, (2, 3, 16))
        self.assertTrue((seen[0] == 0).all())

    def test_explicit_two_source_ablation_has_no_zero_fusion_input(self):
        model = MTTGNetTestWrapper(2, hidden_dim=16, horizon=3,
                                  use_memory=False, fusion_sources=2)
        seen = []
        hook = model.model.fusion.register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs['memory_ctx']), with_kwargs=True)
        result = model(**self.args)
        hook.remove()
        self.assertEqual(result['fusion_gate'].shape, (2, 3, 2))
        self.assertEqual(seen, [None])

    def test_memory_enabled_requires_three_sources(self):
        model = MTTGNetTestWrapper(2, hidden_dim=16, horizon=3, use_memory=True)
        self.assertEqual(model.model.fusion.num_sources, 3)
        with self.assertRaises(ValueError):
            MTTGNetTestWrapper(2, use_memory=True, fusion_sources=2)
        with self.assertRaises(ValueError):
            MultiSourceGateFusion(16, 3)(torch.rand(2,3,16), torch.rand(2,3,16),
                                         None, self.args['future_calendar'])

    def test_zero_source_rescales_but_layernorm_is_not_exactly_invariant(self):
        logits = torch.tensor([.3, -.4, .2], dtype=torch.float64)
        gate3, gate2 = logits.softmax(0), logits[:2].softmax(0)
        temporal, periodic = torch.randn(2, 16, dtype=torch.float64), torch.randn(2, 16, dtype=torch.float64)
        v2 = gate2[0]*temporal + gate2[1]*periodic
        v3 = gate3[0]*temporal + gate3[1]*periodic
        torch.testing.assert_close(v3, (1-gate3[2])*v2)
        exact = torch.nn.LayerNorm(16, eps=0).double()
        torch.testing.assert_close(exact(v3), exact(v2))
        practical = torch.nn.LayerNorm(16).double()
        self.assertFalse(torch.equal(practical(v3), practical(v2)))
        torch.testing.assert_close(practical(v3), practical(v2), rtol=1e-4, atol=1e-4)
        # A nearly closed active pair makes epsilon matter; never claim exact equivalence.
        self.assertGreater((practical(v2*1e-6)-practical(v2)).abs().max().item(), .1)

    def test_archived_three_source_weights_load_without_conversion(self):
        from experiments.cmip6.run_cmip6_pure_ml import build_model
        old_cfg = copy.deepcopy(self.cfg)
        old_cfg['model_params']['fusion_sources'] = 3
        original = build_model(old_cfg, 2).eval()
        state = original.state_dict()
        metadata = copy.deepcopy(old_cfg)
        del metadata['model_params']['fusion_sources']
        before = copy.deepcopy(metadata)
        restored_cfg = config_for_saved_fusion(metadata, state)
        self.assertEqual(metadata, before)
        self.assertEqual(restored_cfg['model_params']['fusion_sources'], 3)
        restored = build_model(restored_cfg, 2).eval()
        restored.load_state_dict(state, strict=True)
        with torch.no_grad():
            self.assertTrue(torch.equal(original(**self.args)['prediction'], restored(**self.args)['prediction']))

    def test_new_saved_weights_remain_two_source_and_conflicts_fail(self):
        from experiments.cmip6.run_cmip6_pure_ml import build_model
        self.cfg['model_params']['fusion_sources'] = 2
        model = build_model(self.cfg, 2)
        self.assertEqual(self.cfg['model_params']['fusion_sources'], 2)
        state = model.state_dict()
        restored = config_for_saved_fusion(self.cfg, state)
        self.assertEqual(restored['model_params']['fusion_sources'], 2)
        restored['model_params']['fusion_sources'] = 3
        with self.assertRaises(ValueError):
            config_for_saved_fusion(restored, state)
        baseline_cfg = {'model_params': {'arch': 'baseline'}}
        self.assertEqual(config_for_saved_fusion(baseline_cfg, {'linear.weight': torch.ones(2,2)}), baseline_cfg)

    def test_multiseed_forwards_fusion_and_co2_controls(self):
        from scripts.run_multiseed import build_model
        self.cfg['model_params'].update(fusion_sources=2, co2_attention_modulation=False, co2_additive_bias=False)
        trainer = SimpleNamespace(num_features=2, device=torch.device('cpu'))
        model = build_model(self.cfg, trainer)
        self.assertEqual(model.model.fusion.num_sources, 2)
        self.assertFalse(model.model.co2_attention_modulation)
        self.assertFalse(model.model.co2_additive_bias)
        self.cfg['model_params']['fusion_sources'] = 3
        self.assertEqual(build_model(self.cfg, trainer).model.fusion.num_sources, 3)

    def test_training_records_source_count_and_rejects_unsupported_legacy_request(self):
        record_fusion_config(self.cfg)
        self.assertEqual(self.cfg['model_params']['fusion_sources'], 3)
        with self.assertRaises(ValueError):
            record_fusion_config({'model_params': {'arch': 'mttgnet', 'fusion_sources': 2}})


if __name__ == '__main__':
    unittest.main()
