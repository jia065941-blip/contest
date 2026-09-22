import sys,math,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from c0a_goal_b10_objective import regret_loss
from train_c0a_goal_b11_diagnostic import probability_kl,norm,dot

class DiagnosticGradientTests(unittest.TestCase):
    def terms(self,gap):
        logits=torch.tensor([[.9,.1]],dtype=torch.float64).log().requires_grad_()
        old=torch.tensor([[.1,.9]],dtype=torch.float64).log()
        valid=torch.ones((1,2),dtype=torch.bool)
        loss,d=regret_loss(logits,old,valid,torch.tensor([[0,1]]),torch.tensor([[gap,0.]],dtype=torch.float64),valid,.01)
        gr=torch.autograd.grad(d['regret'],logits,retain_graph=True)[0]
        gk=torch.autograd.grad(.01*d['kl'],logits,retain_graph=True)[0]
        gt=torch.autograd.grad(loss,logits)[0]
        return d,gr,gk,gt

    def test_exact_kl_and_opposing_gradients(self):
        d,gr,gk,gt=self.terms(1.)
        self.assertAlmostEqual(float(d['kl'].detach()),.8*math.log(9),places=12)
        torch.testing.assert_close(gr,torch.tensor([[-.09,.09]],dtype=torch.float64))
        torch.testing.assert_close(gk,torch.tensor([[.008,-.008]],dtype=torch.float64))
        torch.testing.assert_close(gt,gr+gk)
        self.assertAlmostEqual(dot([gr],[gk])/(norm([gr])*norm([gk])),-1.,places=12)

    def test_regularizer_can_overpower_small_return_gap(self):
        _,gr,gk,gt=self.terms(.001)
        self.assertGreater(norm([gk])/norm([gr]),80.)
        self.assertGreater(float(gt[0,0]),0.)
        self.assertLess(float(gr[0,0]),0.)

    def test_mask_and_state_average(self):
        valid=torch.tensor([[True,True,False],[True,True,False]])
        old=torch.tensor([[.1,.9,10.],[.2,.8,50.]],dtype=torch.float64).log()
        new=torch.tensor([[.9,.1,99.],[.2,.8,.01]],dtype=torch.float64).log()
        kl=probability_kl(old,new,valid)
        self.assertAlmostEqual(float(kl[0]),.8*math.log(9),places=12)
        self.assertAlmostEqual(float(kl[1]),0.,places=12)
        self.assertAlmostEqual(float(kl.mean()),.4*math.log(9),places=12)

if __name__=='__main__':unittest.main()
