"""Verify the published b14 artifact and run CPU shape/finite inference checks."""
from pathlib import Path
import argparse,hashlib,json,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from train_c0a_goal_b8_advantage import load_model
EXPECTED_SHA256='4609add14e70a5e25df87ddb3377756c2e45b0d3fa136d4de66406f26ca2be6e'

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'models/b14_c0a_goal.pt')
    args=parser.parse_args();torch.set_num_threads(1)
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if digest!=EXPECTED_SHA256:raise ValueError('Checkpoint does not match published b14 SHA256')
    payload,config,model=load_model(args.checkpoint,'cpu');model.eval()
    observation=torch.zeros(2,config.observation_dim)
    features=torch.zeros(2,config.target_slots,config.target_feature_dim)
    valid=torch.ones(2,config.target_slots,dtype=torch.bool)
    with torch.no_grad():out=model.distribution_parameters(observation,target_features=features,target_valid_mask=valid)
    if not all(torch.isfinite(v).all() for v in out.values()):raise ValueError('Nonfinite inference')
    print(json.dumps({'status':'PASS','checkpoint':str(args.checkpoint),'sha256':digest,
                     'algorithm':payload['algorithm'],'parameters':sum(p.numel() for p in model.parameters()),
                     'target_logits_shape':list(out['target_logits'].shape),
                     'scope':'artifact load and synthetic-input finite inference only; not an E01 evaluation'},indent=2))
if __name__=='__main__':main()
