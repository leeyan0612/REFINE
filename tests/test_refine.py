import unittest
import torch
from refine.model import predicted_geometry
from refine.metrics import image_ap
from refine.losses import dense_queries

class NumericalContractTests(unittest.TestCase):
    def test_geometry_has_ten_differentiable_channels(self):
        anchors=torch.tensor([[[.7,.4],[.4,.5]]],requires_grad=True)
        D=torch.tensor([.2],requires_grad=True);R=torch.tensor([.3],requires_grad=True)
        phi,_=predicted_geometry(anchors,D,R,28)
        self.assertEqual(tuple(phi.shape),(1,784,10))
        phi.square().mean().backward()
        for v in (anchors,D,R):
            self.assertTrue(torch.isfinite(v.grad).all())
            self.assertGreater(float(v.grad.abs().sum()),0)

    def test_geometry_coincident_points_are_finite(self):
        phi,axes=predicted_geometry(torch.full((1,2,2),.5),torch.tensor([0.]),torch.tensor([0.]),28)
        self.assertTrue(torch.isfinite(phi).all())
        self.assertFalse(bool(axes['predicted_axis_valid'][0]))

    def test_unknown_support_contributes_no_gradient(self):
        logits=torch.tensor([[[[.2,-.5]]]],requires_grad=True)
        gt={'P':torch.tensor([[[[1.,0.]]]]),'K':torch.tensor([[[[1.,0.]]]]),
            'cells':torch.ones((1,1,2)),'weight':torch.ones((1,1))}
        value,_=dense_queries(logits,gt,torch.ones(1));value.sum().backward()
        self.assertEqual(float(logits.grad[0,0,0,1]),0.)
        self.assertNotEqual(float(logits.grad[0,0,0,0]),0.)

    def test_ap_unknown_and_ties(self):
        self.assertEqual(image_ap([.9,.8,.7],[1,-1,0]),1.)
        self.assertEqual(image_ap([.5,.5],[1,0]),.5)
        self.assertIsNone(image_ap([.5],[0]))

if __name__=='__main__':unittest.main()
