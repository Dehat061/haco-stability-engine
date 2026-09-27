from __future__ import annotations
import copy, csv, hashlib, importlib, importlib.metadata as md, json, platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics.pairwise import laplacian_kernel
from . import __version__

SKLEARN_DATASETS={"iris":load_iris,"wine":load_wine,"breast-cancer":load_breast_cancer,"digits":load_digits}

def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical_report_hash(report):
    tmp=copy.deepcopy(report)
    tmp["integrity"]["report_sha256"]="PENDING"
    return hashlib.sha256(json.dumps(tmp,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def array_fingerprint(X,y):
    h=hashlib.sha256()
    Xc=np.ascontiguousarray(np.asarray(X))
    h.update(str(Xc.shape).encode()); h.update(str(Xc.dtype).encode()); h.update(Xc.tobytes())
    ys=np.asarray(y).astype(str).ravel()
    h.update(str(ys.shape).encode()); h.update("\n".join(ys.tolist()).encode())
    return h.hexdigest()

def runtime_versions():
    out={"python":platform.python_version()}
    for name in ["numpy","scikit-learn","scipy","joblib","threadpoolctl","PyYAML","jsonschema"]:
        try: out[name]=md.version(name)
        except Exception: out[name]="not-recorded"
    return out

def load_dataset(cfg):
    kind=cfg.get("kind","sklearn")
    if kind=="sklearn":
        name=cfg["name"]
        if name not in SKLEARN_DATASETS: raise ValueError(f"Unsupported sklearn dataset: {name}")
        X,y=SKLEARN_DATASETS[name](return_X_y=True)
        X=np.asarray(X,float); y=np.asarray(y)
        return X,y,f"sklearn:{name}",{"effective_X_y_sha256":array_fingerprint(X,y)}
    if kind=="csv":
        path=Path(cfg["path"]).expanduser().resolve(); target=cfg["target"]
        if not path.exists(): raise FileNotFoundError(path)
        with path.open("r",encoding="utf-8-sig",newline="") as f:
            reader=csv.DictReader(f)
            if not reader.fieldnames or target not in reader.fieldnames:
                raise ValueError(f"Target column {target!r} not found in CSV")
            feats=[c for c in reader.fieldnames if c!=target]
            Xrows=[]; yrows=[]
            for line,row in enumerate(reader,start=2):
                try: Xrows.append([float(row[c]) for c in feats])
                except Exception as e: raise ValueError(f"Non-numeric feature at CSV line {line}") from e
                yrows.append(row[target])
        if not Xrows: raise ValueError("CSV contains no data")
        X=np.asarray(Xrows,float)
        try: y=np.asarray([float(v) for v in yrows])
        except Exception: y=np.asarray(yrows)
        return X,y,f"csv:{path.name}",{path.name:file_sha256(path),"effective_X_y_sha256":array_fingerprint(X,y)}
    raise ValueError(f"Unsupported data.kind: {kind}")

def _builtin_factory(name,params):
    p=dict(params or {})
    if name=="knn-l1": return lambda: KNeighborsClassifier(n_neighbors=p.get("n_neighbors",5),metric="minkowski",p=1,algorithm="brute",weights=p.get("weights","uniform"))
    if name=="knn-l2": return lambda: KNeighborsClassifier(n_neighbors=p.get("n_neighbors",5),metric="minkowski",p=2,algorithm="brute",weights=p.get("weights","uniform"))
    if name=="logistic-l1": return lambda: LogisticRegression(penalty="l1",solver="liblinear",random_state=p.get("random_state",0),max_iter=p.get("max_iter",2000))
    if name=="logistic-l2": return lambda: LogisticRegression(penalty="l2",solver="liblinear",random_state=p.get("random_state",0),max_iter=p.get("max_iter",2000))
    if name=="decision-tree": return lambda: DecisionTreeClassifier(random_state=p.get("random_state",0),**{k:v for k,v in p.items() if k!="random_state"})
    if name=="random-forest": return lambda: RandomForestClassifier(n_estimators=p.get("n_estimators",100),random_state=p.get("random_state",0),n_jobs=p.get("n_jobs",1))
    if name=="svc-rbf": return lambda: SVC(kernel="rbf",**p)
    if name=="svc-laplacian":
        gamma=float(p.pop("gamma",1.0))
        def kernel(X,Y): return laplacian_kernel(X,Y,gamma=gamma)
        return lambda: SVC(kernel=kernel,**p)
    raise ValueError(f"Unknown built-in estimator: {name}")

def load_estimator_factory(cfg):
    if "builtin" in cfg:
        name=cfg["builtin"]; params=dict(cfg.get("params",{}))
        return lambda:_builtin_factory(name,dict(params))(), f"builtin:{name}", None
    if "factory" in cfg:
        spec=cfg["factory"]
        if ":" not in spec: raise ValueError("factory must be MODULE:CALLABLE")
        module_name,attr=spec.split(":",1); module=importlib.import_module(module_name); fn=getattr(module,attr)
        if not callable(fn): raise TypeError(f"{spec} is not callable")
        modfile=getattr(module,"__file__",None)
        fh=file_sha256(modfile) if modfile and Path(modfile).exists() else None
        def fac():
            m=fn()
            if not hasattr(m,"fit") or not hasattr(m,"predict"): raise TypeError("factory must return fit/predict object")
            return m
        return fac,f"factory:{spec}",fh
    raise ValueError("Estimator requires builtin or factory")

def preprocess_fit_apply(Xtr,Xte,cfg):
    kind=cfg.get("kind","center-global-rms")
    if kind=="none": return np.asarray(Xtr,float),np.asarray(Xte,float)
    if kind!="center-global-rms": raise ValueError(f"Unsupported preprocessing: {kind}")
    mu=Xtr.mean(axis=0); A=Xtr-mu; B=Xte-mu
    s=float(np.sqrt(np.mean(np.sum(A*A,axis=1))))
    if not np.isfinite(s) or s==0: s=1.0
    return A/s,B/s

def proper_rotations(d,n,seed):
    rng=np.random.default_rng(seed); mats=[]; audit=[]
    for _ in range(n):
        Q,R=np.linalg.qr(rng.normal(size=(d,d)))
        signs=np.sign(np.diag(R)); signs[signs==0]=1; Q=Q*signs
        if np.linalg.det(Q)<0: Q[:,-1]*=-1
        audit.append((float(np.max(np.abs(Q.T@Q-np.eye(d)))),float(np.linalg.det(Q))))
        mats.append(Q)
    return mats,audit

def pfr(a,b): return float(np.mean(np.asarray(a)!=np.asarray(b)))

def policy_decision(policy,m,audit):
    if audit!="PASS": return "WARN"
    exp=policy.get("expected_relation","informational")
    if exp=="informational": return "UNSCOPED"
    if exp=="invariant":
        if "max_pfr" not in policy: raise ValueError("policy.max_pfr required")
        return "PASS" if m["max_pfr"]<=float(policy["max_pfr"]) else "FAIL"
    if exp=="sensitive":
        for k in ["min_pfr","min_detection_rate","min_positive_folds"]:
            if k not in policy: raise ValueError(f"policy.{k} required for sensitive")
        ok=(m["max_pfr"]>=float(policy["min_pfr"])
            and m["detection_rate"]>=float(policy["min_detection_rate"])
            and m["positive_folds"]>=int(policy["min_positive_folds"]))
        return "PASS" if ok else "FAIL"
    raise ValueError(f"Unsupported expected_relation: {exp}")

def run_audit(config,output_dir):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    X,y,dataset_id,dhash=load_dataset(config["data"])
    factory,estimator_id,factory_hash=load_estimator_factory(config["estimator"])
    split=config.get("split",{}); folds=int(split.get("folds",3)); seed=int(split.get("seed",20260827))
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    tc=config.get("transformations",{}); count=int(tc.get("count",20)); tseed=int(tc.get("seed",seed))
    if tc.get("family","proper-orthogonal")!="proper-orthogonal": raise ValueError("RC3 supports proper-orthogonal only")
    pc=config.get("preprocessing",{"kind":"center-global-rms"}); q=config.get("numeric_precision",{}).get("quantize_decimals")
    raw=[]; errors=[]; dets=[]; base_accs=[]; rot_accs=[]
    for fold,(tri,tei) in enumerate(cv.split(X,y),start=1):
        Xtr,Xte=preprocess_fit_apply(X[tri],X[tei],pc); ytr,yte=y[tri],y[tei]
        if q is not None: Xtr=np.round(Xtr,int(q)); Xte=np.round(Xte,int(q))
        b=factory(); b.fit(Xtr,ytr); bp=np.asarray(b.predict(Xte)); ba=float(np.mean(bp==yte)); base_accs.append(ba)
        rots,aud=proper_rotations(Xtr.shape[1],count,tseed+1009*fold); errors += [x[0] for x in aud]; dets += [x[1] for x in aud]
        for i,Q in enumerate(rots,start=1):
            A=Xtr@Q; B=Xte@Q
            if q is not None: A=np.round(A,int(q)); B=np.round(B,int(q))
            m=factory(); m.fit(A,ytr); pred=np.asarray(m.predict(B)); acc=float(np.mean(pred==yte)); rot_accs.append(acc)
            raw.append({"fold":fold,"transform":i,"pfr":pfr(bp,pred),"accuracy":acc,"identity_accuracy":ba})
    vals=np.asarray([r["pfr"] for r in raw],float); tol=float(tc.get("orthogonality_tolerance",1e-12))
    audit="PASS" if max(errors)<=tol and min(dets)>0 else "FAIL"
    ms={"max_pfr":float(vals.max()),"mean_pfr":float(vals.mean()),"median_pfr":float(np.median(vals)),
        "nonzero_transformations":int(np.sum(vals>0)),"detection_rate":float(np.mean(vals>0)),
        "positive_folds":int(len({r["fold"] for r in raw if r["pfr"]>0})),"total_folds":folds}
    decision=policy_decision(config.get("policy",{}),ms,audit)
    raw_path=output_dir/"transform_results.json"; raw_path.write_text(json.dumps(raw,indent=2),encoding="utf-8")
    pol=config.get("policy",{})
    cfg_hash=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
    report={
      "schema_version":"1.0-RC3",
      "protocol":{"name":"HACO Coordinate-Frame Stability Audit","version":"1.0-RC3"},
      "run":{"run_id":config.get("run_id",f"{dataset_id}-{estimator_id}-{seed}"),"timestamp_utc":datetime.now(timezone.utc).isoformat(),
             "engine_version":__version__,"seeds":[seed,tseed],"runtime_versions":runtime_versions()},
      "object_of_measurement":{"kind":"learning-procedure","estimator":estimator_id,"notes":"Fresh estimator fitted in identity and every transformed frame."},
      "data":{"datasets":[dataset_id],"dataset_fingerprints":dhash,"split_strategy":f"{folds}-fold StratifiedKFold, shuffle=True, seed={seed}","preprocessing":pc.get("kind","center-global-rms")},
      "transformations":{"family":"proper orthogonal rotations","count":count,"strata_count":folds,"total_transform_evaluations":len(raw),"audit_status":audit,
                         "max_orthogonality_error":float(max(errors)),"min_determinant":float(min(dets))},
      "measurement":{**ms,"endpoint":"prediction_flip_rate","transformation_count":len(raw),"accuracy_identity_mean":float(np.mean(base_accs)),
                     "accuracy_rotated_min":float(np.min(rot_accs)),"accuracy_rotated_max":float(np.max(rot_accs)),"raw_artifact":raw_path.name},
      "policy":{"expected_relation":pol.get("expected_relation","informational"),
                "max_pfr_tolerance":float(pol["max_pfr"]) if "max_pfr" in pol else None,
                "min_pfr_expected":float(pol["min_pfr"]) if "min_pfr" in pol else None,
                "min_detection_rate":float(pol["min_detection_rate"]) if "min_detection_rate" in pol else None,
                "min_positive_folds":int(pol["min_positive_folds"]) if "min_positive_folds" in pol else None,
                "notes":pol.get("notes","User-declared policy.")},
      "decision":decision,
      "limitations":["RC3 audits refit behavior of supervised fit/predict procedures.",
                     "A mathematically valid coordinate transform may be semantically invalid in some domains.",
                     "Twenty rotations are empirical screening guidance, not a universal guarantee.",
                     "Factory mode executes trusted local Python code and is not a security sandbox."],
      "integrity":{"report_sha256":"PENDING","artifact_sha256":{raw_path.name:file_sha256(raw_path)},
                   "config_sha256":cfg_hash,"factory_sha256":factory_hash}
    }
    report["integrity"]["report_sha256"]=canonical_report_hash(report)
    rp=output_dir/"haco_report.json"; rp.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report,rp
