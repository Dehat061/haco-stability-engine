import tempfile, json
from pathlib import Path
from haco_stability.core import run_audit, canonical_report_hash, file_sha256

def base(model,policy):
    return {
      "data":{"kind":"sklearn","name":"iris"},
      "split":{"folds":3,"seed":20260827},
      "preprocessing":{"kind":"center-global-rms"},
      "estimator":{"builtin":model,"params":{"n_neighbors":5}},
      "transformations":{"family":"proper-orthogonal","count":20,"seed":20260827},
      "policy":policy
    }

def test_knn_l2_invariant():
    with tempfile.TemporaryDirectory() as td:
        r,p=run_audit(base("knn-l2",{"expected_relation":"invariant","max_pfr":0.0}),td)
        assert r["decision"]=="PASS"
        assert r["measurement"]["max_pfr"]==0.0
        assert canonical_report_hash(r)==r["integrity"]["report_sha256"]

def test_knn_l1_sensitive_recurrent():
    policy={"expected_relation":"sensitive","min_pfr":1e-12,"min_detection_rate":0.10,"min_positive_folds":2}
    with tempfile.TemporaryDirectory() as td:
        r,p=run_audit(base("knn-l1",policy),td)
        assert r["decision"]=="PASS"
        assert r["measurement"]["detection_rate"]>=0.10
        assert r["measurement"]["positive_folds"]>=2
