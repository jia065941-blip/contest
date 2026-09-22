#!/usr/bin/env python3
"""Recompute frozen b13 inference and validate all 32 fresh E01 branch artifacts."""
from pathlib import Path
import json
import torch
from evaluate_c0a_goal_b13_e01 import BASE, TARGET_SLOT_IDS, file_sha256, validate_branch, summarize, write_json, weighted_type_score
from train_c0a_goal_b8_advantage import load_model

def main():
    torch.set_num_threads(1)
    directory=BASE/'validations/b13_e01_new32_20260921'
    protocol=json.loads((directory/'protocol.json').read_text())
    rows=json.loads((directory/'episodes.json').read_text())
    _,_,model=load_model(Path(protocol['checkpoint']),'cpu');model.eval()
    assert file_sha256(Path(protocol['checkpoint']))==protocol['checkpoint_sha256']
    assert len(rows)==len({r['seed'] for r in rows})==32
    assert [r['seed'] for r in rows]==protocol['seeds']
    assert not set(protocol['seeds'])&set(protocol['training_seeds'])
    assert not set(protocol['seeds'])&set(protocol['excluded_previous_evaluation_seeds'])
    code_checks=0
    for path,sha in protocol['code_sha256'].items():
        assert file_sha256(Path(path))==sha
        code_checks+=1
    source_checks=0
    for key in ['teacher_checkpoint','scenario','manifest','seed_boundary_source','training_protocol','previous_b12_protocol']:
        sha_key={'teacher_checkpoint':'teacher_sha256','scenario':'scenario_sha256','manifest':'manifest_sha256','seed_boundary_source':'seed_boundary_source_sha256','training_protocol':'training_protocol_sha256','previous_b12_protocol':'previous_b12_protocol_sha256'}[key]
        assert file_sha256(Path(protocol[key]))==protocol[sha_key]
        source_checks+=1
    for path,sha in protocol['trace_sha256'].items():
        assert file_sha256(Path(path))==sha
        source_checks+=1
    records=[];artifact_checks=0;branch_checks=0
    for row,job in zip(rows,protocol['jobs']):
        episode=directory/'episodes'/f"i{row['order']:02d}_s{row['seed']}"
        required={str(episode/name) for name in ['capture.pt','branch_input.pt','branch_output.pt','decision.json','branches/branch_00.json','branches/branch_01.json']}
        assert set(row['artifact_sha256'])==required
        for path,sha in row['artifact_sha256'].items():
            assert file_sha256(Path(path))==sha
            artifact_checks+=1
        capture=torch.load(episode/'capture.pt',map_location='cpu',weights_only=True)
        payload=torch.load(episode/'branch_output.pt',map_location='cpu',weights_only=True)
        decision=json.loads((episode/'decision.json').read_text())
        for key in ['seed','timestep','executor_id','start_state_id']:
            assert capture[key]==payload[key]==job[key]
        for row_key,job_key in [('order','order'),('seed','seed'),('start_state_id','start_state_id'),('decision_step','timestep'),('executor_id','executor_id')]:
            assert row[row_key]==job[job_key]
        assert capture['snapshot_model_sha256']==row['checkpoint_sha256']==protocol['checkpoint_sha256']
        assert decision['checkpoint_sha256']==protocol['checkpoint_sha256']
        assert decision['teacher_target_id']==row['teacher_target_id']==job['teacher_target_id']
        assert row['public_rng_sha256']==payload['public_rng_sha256']
        assert capture['physical_state_sha256']==payload['physical_state_sha256']==row['physical_state_sha256']
        valid=capture['target_valid_mask']
        with torch.no_grad():
            logits=model.distribution_parameters(capture['observation'],target_features=capture['target_features'],target_valid_mask=valid)['target_logits'][0]
        assert torch.equal(logits,capture['old_logits']),f"logits differ at {row['seed']}"
        slot=int(logits.masked_fill(~valid[0],-torch.inf).argmax())
        assert bool(valid[0,slot]) and slot==decision['student_slot']==int(payload['candidate_target_indices'][0])
        assert TARGET_SLOT_IDS[slot]==row['student_target_id']==decision['student_target_id']
        assert row['legal_target_count']==int(valid.sum())
        assert decision['legal_slots']==valid[0].nonzero().flatten().tolist()
        assert row['teacher_target_legal_in_student_mask']==bool(valid[0,TARGET_SLOT_IDS.index(job['teacher_target_id'])])
        assert row['illegal_samples']==0
        assert len(payload['branch_results'])==2
        for i,branch in enumerate(payload['branch_results']):
            validate_branch(branch,payload,i)
            assert json.loads((episode/f'branches/branch_{i:02d}.json').read_text())==branch
            prefix='student' if i==0 else 'teacher'
            expected={'target_id':branch['target_id'],'score':branch['score'],
                      'fixed_target_score':weighted_type_score(branch['summary'],{9400,9600}),
                      'suffix_return':branch['official_joint_return'],'end_step':branch['end_step']}
            for key,value in expected.items():assert row[prefix+'_'+key]==value,(row['seed'],prefix,key)
            assert branch['physical_state_sha256']==row['physical_state_sha256']
            assert branch['public_rng_sha256']==row['public_rng_sha256']
            branch_checks+=1
        records.append({'seed':row['seed'],'logits_exact':True,'legal_argmax_exact':True,'student_target_id':row['student_target_id']})
    assert summarize(rows,protocol)==json.loads((directory/'evaluation_summary.json').read_text())
    result={'status':'PASS','episodes':32,'fresh_student_episodes':32,'teacher_reference_branches':32,'exact_logits_checks':32,'legal_argmax_checks':32,'artifact_hash_checks':artifact_checks,'branch_contract_checks':branch_checks,'immutable_code_hash_checks':code_checks,'source_and_trace_hash_checks':source_checks,'all_row_scalars_rebound_to_branch_artifacts':True,'summary_reproduces':True,'checkpoint_sha256':protocol['checkpoint_sha256'],'verifier_code_sha256':file_sha256(Path(__file__)),'records':records}
    write_json(directory/'deterministic_checks.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':main()
