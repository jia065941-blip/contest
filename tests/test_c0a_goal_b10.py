"""Analytic objective and complete-coverage integrity tests for b10."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from c0a_goal_b10_objective import regret_loss

class RegretTests(unittest.TestCase):
    def call(self, logits, returns, old=None, valid=None, indices=None, mask=None, kl=0.):
        if old is None: old=logits.detach().clone()
        if valid is None: valid=torch.ones_like(logits,dtype=torch.bool)
        if indices is None: indices=torch.arange(logits.shape[1]).expand(logits.shape[0],-1)
        if mask is None: mask=torch.ones_like(indices,dtype=torch.bool)
        return regret_loss(logits,old,valid,indices,returns,mask,kl)

    def test_exact_formula_gradient_and_detached_labels_reference(self):
        logits=torch.tensor([[.1,.5,-.2]],dtype=torch.float64,requires_grad=True)
        old=torch.tensor([[.3,.1,.2]],dtype=torch.float64,requires_grad=True)
        ret=torch.tensor([[.2,.7,.9]],dtype=torch.float64,requires_grad=True)
        loss,detail=self.call(logits,ret,old=old,kl=.01)
        p=logits.detach().softmax(-1);q=old.detach().softmax(-1);gaps=ret.detach().max()-ret.detach()
        expected=(p*gaps).sum()+.01*(q*(q.log()-p.log())).sum()
        torch.testing.assert_close(loss,expected)
        loss.backward()
        torch.testing.assert_close(logits.grad,p*(gaps-(p*gaps).sum())+.01*(p-q))
        self.assertIsNone(ret.grad);self.assertIsNone(old.grad)

    def test_return_gap_scale_is_preserved(self):
        z=torch.zeros(1,2,requires_grad=True)
        small,_=self.call(z,torch.tensor([[0.,.001]]))
        large,_=self.call(z,torch.tensor([[0.,.1]]))
        torch.testing.assert_close(large,100*small)
        grad_small=torch.autograd.grad(small,z)[0];grad_large=torch.autograd.grad(large,z)[0]
        torch.testing.assert_close(grad_large,100*grad_small)

    def test_ties_do_not_force_uniform_distribution(self):
        z=torch.tensor([[3.,0.,-2.]],requires_grad=True)
        loss,_=self.call(z,torch.ones(1,3));loss.backward()
        self.assertEqual(float(loss.detach()),0.);torch.testing.assert_close(z.grad,torch.zeros_like(z))

    def test_higher_return_remains_preferred_inside_old_elite_band(self):
        z=torch.zeros(1,3,requires_grad=True)
        loss,_=self.call(z,torch.tensor([[1.,.95,0.]]));loss.backward()
        self.assertLess(float(z.grad[0,0]),float(z.grad[0,1]))

    def test_missing_duplicate_and_illegal_candidates_rejected(self):
        z=torch.zeros(1,3);r=torch.zeros(1,2)
        for idx in [torch.tensor([[0,1]]),torch.tensor([[0,0]])]:
            with self.assertRaises(ValueError):self.call(z,r,indices=idx)
        with self.assertRaises(ValueError):self.call(z,torch.zeros(1,3),valid=torch.tensor([[True,True,False]]))

    def test_padding_and_invalid_logits_have_zero_gradient(self):
        z=torch.tensor([[0.,float('nan'),1.],[2.,0.,float('nan')]],requires_grad=True)
        valid=torch.tensor([[True,False,True],[True,False,False]])
        idx=torch.tensor([[2,0],[0,-1]]);mask=torch.tensor([[True,True],[True,False]])
        loss,d=self.call(z,torch.tensor([[.9,.2],[.5,float('nan')]]),valid=valid,indices=idx,mask=mask,kl=.01)
        loss.backward();self.assertTrue(torch.isfinite(z.grad).all());self.assertTrue((z.grad[~valid]==0).all())
        self.assertAlmostEqual(float(loss.detach()),float(d['state_regret'][0].detach())/2,places=6)

class DatasetGateTests(unittest.TestCase):
    def base(self):
        item={'start_state_id':'s0','seed':0,'fresh_all_legal':True,'old_return_labels_reused':False,
              'all_legal_evaluated':True,'candidate_target_indices':torch.tensor([0,1]),
              'candidate_target_ids':torch.tensor([51,52]),'official_joint_returns':torch.tensor([.1,.2]),
              'target_valid_mask':torch.tensor([[True,True]]),'branch_results':[], 'b10_provenance':[]}
        return {'stage':'C0a_goal_b10_full_legal','items':[item]*64,
                'protocol':{'fresh_all_legal':True,'old_return_labels_reused':False}}

    def test_missing_branch_records_cannot_claim_reconstructed_labels(self):
        from train_c0a_goal_b10_regret import verify_dataset
        with self.assertRaisesRegex(ValueError,'cardinality'):verify_dataset(self.base())

    def test_cached_labels_rejected_at_protocol_and_item(self):
        from train_c0a_goal_b10_regret import verify_dataset
        payload=self.base();payload['protocol']['old_return_labels_reused']=True
        with self.assertRaisesRegex(ValueError,'fresh complete'):verify_dataset(payload)
        payload=self.base();payload['items'][0]['old_return_labels_reused']=True
        with self.assertRaisesRegex(ValueError,'historical labels'):verify_dataset(payload)

    def test_provenance_must_cover_exact_legal_slots(self):
        from train_c0a_goal_b10_regret import verify_dataset
        payload=self.base();payload['items'][0]['branch_results']=[{},{}]
        payload['items'][0]['b10_provenance']=[{'slot':0},{'slot':0}]
        with self.assertRaisesRegex(ValueError,'provenance must cover'):verify_dataset(payload)

if __name__=='__main__':unittest.main()
