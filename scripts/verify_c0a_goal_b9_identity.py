#!/usr/bin/env python3
"""Independent complete-trace verifier for the b9 same-target null experiment."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify(root):
    protocol=json.loads((root/'protocol.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    assert summary['seeds']==protocol['seeds'] and summary['completed']==4
    evidence=[]
    keys=['step','actions','actions_sha256','full_observation_sha256','teacher_state_sha256','public_rng_sha256','objective_health','official_return']
    for d in sorted((root/'episodes').iterdir()):
        result=json.loads((d/'result.json').read_text())
        branches=[json.loads((d/f'branches/branch_{i:02d}.json').read_text()) for i in range(3)]
        traces=[]
        for branch in branches:
            p=Path(branch['identity_trace']['path'])
            assert sha(p)==branch['identity_trace']['sha256']
            trace=[json.loads(s) for s in p.read_text().splitlines()]
            assert len(trace)==branch['identity_trace']['steps']
            assert [row['step'] for row in trace]==list(range(result['boundary'],branch['end_step']+1))
            assert trace[-1]['official_return']==branch['official_joint_return']
            assert trace[-1]['objective_health']==branch['final_objective_health']
            assert trace[-1]['suppressed_total']==branch['suppressed_target_change_commands']
            assert branch['score']==branch['summary']['score']['score']
            assert branch['exact_boundary_actions_equal']
            traces.append(trace)
        # Null pair checks include metadata that the online compare omitted.
        assert traces[0]==traces[1], f"seed {result['seed']} full paired traces differ"
        excluded={'candidate_index','target_index','identity_trace'}
        assert {k:v for k,v in branches[0].items() if k not in excluded}=={k:v for k,v in branches[1].items() if k not in excluded}
        assert result['same_control']['all_compared_fields_equal']
        assert result['same_control']['score_delta']==0 and result['same_control']['return_delta']==0
        assert len({b['common_snapshot_python_sha256'] for b in branches})==1
        assert len({b['public_rng_sha256'] for b in branches})==1
        assert len({b['physical_state_sha256'] for b in branches})==1
        first_native={k: next((a['step'] for a,b in zip(traces[0],traces[2]) if a[k]!=b[k]),None) for k in keys}
        equal_native=len(traces[0])==len(traces[2]) and all(x is None for x in first_native.values())
        assert equal_native==result['native_unlocked_control']['all_compared_fields_equal']
        assert branches[2]['native_unlocked'] is True
        assert branches[2]['suppressed_target_change_commands']==0
        trace=json.loads(Path(json.loads((d/'request.json').read_text())['trace']).read_text())
        boundary=next(row['actions'] for row in trace['steps'] if row['step']==result['boundary'])
        assert all(t[0]['actions']==boundary for t in traces)
        evidence.append({
            'seed':result['seed'],'boundary':result['boundary'],'end_step':branches[0]['end_step'],
            'paired_steps_exactly_equal':len(traces[0]),'all_paired_result_fields_equal':True,
            'trajectory_sha256':branches[0]['identity_trace']['sha256'],
            'teacher_score':branches[1]['score'],'forced_student_score':branches[0]['score'],
            'official_suffix_return':branches[0]['official_joint_return'],
            'native_unlocked_score':branches[2]['score'],'native_measured_fields_equal':equal_native,
            'first_native_difference':first_native,
            'lock_released_at':branches[0]['option_termination_step'],
            'suppressed_commands':branches[0]['suppressed_target_change_commands'],
            'termination_reason':branches[0]['summary'].get('termination_reason'),
        })
    modified=[p for p,h in protocol['hashes'].items() if sha(p)!=h]
    extra=json.loads((root/'supplemental_provenance.json').read_text())
    modified.extend(p for p,info in extra['files'].items() if sha(p)!=info['sha256'])
    assert not modified, modified
    for name,key in [('b9_checkpoint','b9_sha256'),('teacher_checkpoint','teacher_sha256')]:
        assert sha(protocol[name])==protocol[key]
    return {'status':'PASS','evaluation_type':'simulation_only','seed_count':len(evidence),
            'total_paired_steps':sum(e['paired_steps_exactly_equal'] for e in evidence),
            'source_and_checkpoint_hashes_unchanged':True,'rows':evidence,
            'scope':'exact duplicate b9 continuation identity under teacher-target intervention; legacy student handoff not exercised',
            'metadata_caveat':'native-unlocked branch retains runtime locked_target_id label; actual native_unlocked flag, per-step locked field and zero suppression establish unlocked behavior'}

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('root',type=Path)
    args=p.parse_args()
    result=verify(args.root)
    (args.root/'independent_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
