"""Reproduce the original seven checks and the revised statistical attachment.

All computation happens in a temporary directory. Archived outputs are read,
never overwritten. NumPy is required by the statistical attachment.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile
import platform
import numpy as np
import openpyxl

versions={'python':platform.python_version(),'numpy':np.__version__,'openpyxl':openpyxl.__version__}
expected={'python':'3.12.14','numpy':'2.3.5','openpyxl':'3.1.5'}
if versions!=expected:
    raise RuntimeError('Byte-exact replay requires '+str(expected)+'; current environment '+str(versions)+'. Version drift is not a mathematical counterexample.')

base=Path(__file__).resolve().parent
names=['switch_common_filter_positive_worlds','switch_three_cost_three_lag_positive_pool',
       'switch_three_cost_two_lag_positive_base','switch_three_cost_finite_noise_certificate',
       'switch_three_cost_finite_filter_signal_floor','switch_three_cost_heterogeneous_risk_rejection']
with tempfile.TemporaryDirectory(prefix='paper_upgrade_reproduce_') as temp:
    dst=Path(temp)
    for name in names:
        for ext in ['.py','_results.json']:
            shutil.copy2(base/(name+ext),dst/(name+ext))
    for name in ['switch_three_cost_common_filter_identification.py','paper_review_reproduce_20261006.py',
                 'paper_statistical_audit_20261006.py']:
        shutil.copy2(base/name,dst/name)
    obs=dst/'market_observation';obs.mkdir()
    for name in ['mtum_two_snapshot_fields_20261006.json','mtum_two_snapshot_observation_20261006.py',
                 'mtum_two_snapshot_results_20261006.json']:
        shutil.copy2(base/'market_observation'/name,obs/name)
    for name in ['paper_review_reproduce_20261006.py','paper_statistical_audit_20261006.py']:
        proc=subprocess.run([sys.executable,str(dst/name)],capture_output=True,text=True,timeout=120)
        if proc.returncode:
            raise RuntimeError(name+': '+proc.stderr)
    actual=json.loads((dst/'paper_upgrade_results_20261006'/'statistical_results.json').read_text(encoding='utf-8'))
    archived=json.loads((base/'paper_upgrade_results_20261006'/'statistical_results.json').read_text(encoding='utf-8'))
    assert actual==archived,'Statistical result differs from archived record'
    assert (dst/'paper_upgrade_results_20261006'/'statistical_results.csv').read_bytes()==(base/'paper_upgrade_results_20261006'/'statistical_results.csv').read_bytes()
    public_src=base/'public_audit_20261006'
    public_dst=dst/'public_audit_20261006';public_dst.mkdir()
    for name in ['MTUM_latest.csv','ishares_product.html','audit_public_holdings.py']:
        shutil.copy2(public_src/name,public_dst/name)
    proc=subprocess.run([sys.executable,str(public_dst/'audit_public_holdings.py')],capture_output=True,text=True,timeout=30)
    if proc.returncode: raise RuntimeError('Public holdings audit: '+proc.stderr)
    for name in ['holdings_audit_results.json','holdings_row_audit.csv']:
        assert (public_src/name).read_bytes()==(public_dst/name).read_bytes(),name+' differs from archived result'
    for name in ['paper_calibration_audit_20261006.py','independent_tracking_controller_20261006.py']:
        shutil.copy2(base/name,dst/name)
    proc=subprocess.run([sys.executable,str(dst/'paper_calibration_audit_20261006.py')],capture_output=True,text=True,timeout=120)
    if proc.returncode:raise RuntimeError('Joint calibration audit: '+proc.stderr)
    for name in ['calibration_and_decision_results.json','paired_replay_observation_logs.json']:
        assert (base/'paper_substantive_revision_20261006'/name).read_bytes()==(dst/'paper_substantive_revision_20261006'/name).read_bytes(),name+' differs from archived result'
    for name in ['paper_self_calibration_20261006.py','paper_external_rule_replay_20261006.py',
                 'audit_external_logs_20261006.py','paper_real_path_holdout_20261006.py']:
        shutil.copy2(base/name,dst/name)
    breakthrough=dst/'paper_breakthrough_20261006';breakthrough.mkdir()
    shutil.copytree(base/'paper_breakthrough_20261006'/'external',breakthrough/'external')
    shutil.copy2(base/'paper_breakthrough_20261006'/'后续留出检验方案.md',breakthrough/'后续留出检验方案.md')
    for name in ['paper_self_calibration_20261006.py','paper_external_rule_replay_20261006.py',
                 'audit_external_logs_20261006.py','paper_real_path_holdout_20261006.py']:
        proc=subprocess.run([sys.executable,str(dst/name)],capture_output=True,text=True,timeout=120)
        if proc.returncode:raise RuntimeError('Cost-free/real-path reproduction '+name+': '+proc.stderr)
    for name in ['self_calibration_results.json','external_rule_and_real_path_results.json','external_paired_observations.json',
                 'blind_observation_audit_results.json','later_holdout_results.json']:
        assert (base/'paper_breakthrough_20261006'/name).read_bytes()==(breakthrough/name).read_bytes(),name+' differs from archived result'
    for name in ['paper_stability_revision_20261006.py','paper_overlap_revision_20261006.py','audit_noisy_logs_20261006.py']:
        shutil.copy2(base/name,dst/name)
        proc=subprocess.run([sys.executable,str(dst/name)],capture_output=True,text=True,timeout=240)
        if proc.returncode:raise RuntimeError('Stability/overlap reproduction '+name+': '+proc.stderr)
    for name in ['stability_results.json','noisy_paired_observations.json','noisy_observer_audit.json','overlap_excluded_results.json']:
        assert (base/'paper_stability_revision_20261006'/name).read_bytes()==(dst/'paper_stability_revision_20261006'/name).read_bytes(),name+' differs from archived result'
    import numpy as np
    print(json.dumps({'original_checks':7,'statistical_json_and_csv_exact_match':True,
                      'public_holdings_audit_json_and_csv_exact_match':True,'public_holdings_records':129,
                      'statistical_configurations':actual['configurations'],
                      'correct_model_replications':actual['total_correct_model_replications'],
                      'bounded_noise_corner_checks':actual['bounded_noise_corner_checks'],
                      'jacobian':actual['jacobian_check'],'nonzero_initial_state':actual['nonzero_initial_state_check'],
                      'joint_calibration_and_raw_replay_exact_match':True,'joint_box_checks':1280,
                      'decision_replications':3500,'numpy_version':np.__version__,
                      'cost_free_exact_rational_witness':True,'cost_free_perturbation_checks':256,
                      'external_core_policy_checks':27,'eia_later_holdout_series':3,
                      'cost_free_external_and_later_holdout_exact_match':True,
                      'stability_and_overlap_exact_match':True,'exact_shared_polynomial_checks':120,
                      'bounded_stability_checks':128,'reproduction_versions':versions,
                      'archived_outputs_overwritten':False},ensure_ascii=False))
