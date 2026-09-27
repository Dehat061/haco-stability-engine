from __future__ import annotations
import argparse, json
from importlib.resources import files
from pathlib import Path
import yaml, jsonschema
from . import __version__
from .core import run_audit, canonical_report_hash, file_sha256

def _schema():
    return json.loads(files("haco_stability").joinpath("schema/report.schema.json").read_text(encoding="utf-8"))

def validate_report(path):
    r=json.loads(Path(path).read_text(encoding="utf-8"))
    v=jsonschema.Draft202012Validator(_schema(),format_checker=jsonschema.FormatChecker())
    return [e.message for e in sorted(v.iter_errors(r),key=lambda e:list(e.path))]

def cmd_test(a):
    cfg=yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    out=a.output_dir or cfg.get("output",{}).get("directory","haco-output")
    r,p=run_audit(cfg,out)
    errs=validate_report(p)
    if errs:
        print("Schema validation: FAIL"); [print(" -",e) for e in errs]; return 3
    print("HACO Stability Engine",__version__)
    print("Report:",p); print("Max PFR:",f"{r['measurement']['max_pfr']:.8f}")
    print("Detection rate:",f"{r['measurement']['detection_rate']:.4f}")
    print("Positive folds:",f"{r['measurement']['positive_folds']}/{r['measurement']['total_folds']}")
    print("Transform audit:",r["transformations"]["audit_status"]); print("Decision:",r["decision"])
    return 0 if r["decision"] in ("PASS","UNSCOPED") else (1 if r["decision"]=="FAIL" else 2)

def cmd_validate(a):
    errs=validate_report(a.report)
    if errs:
        print("FAIL"); [print(" -",e) for e in errs]; return 3
    print("PASS"); return 0

def cmd_verify(a):
    rp=Path(a.report).resolve(); r=json.loads(rp.read_text(encoding="utf-8"))
    failures=[f"schema: {e}" for e in validate_report(rp)]
    actual=canonical_report_hash(r); expected=r.get("integrity",{}).get("report_sha256")
    if actual!=expected: failures.append(f"report_sha256 mismatch expected={expected} actual={actual}")
    for rel,exp in r.get("integrity",{}).get("artifact_sha256",{}).items():
        p=rp.parent/rel
        if not p.exists(): failures.append(f"missing artifact: {rel}")
        elif file_sha256(p)!=exp: failures.append(f"artifact hash mismatch: {rel}")
    if failures:
        print("INTEGRITY FAIL"); [print(" -",f) for f in failures]; return 1
    print("INTEGRITY PASS"); print("schema: PASS"); print("report_sha256: PASS"); print("artifact_sha256: PASS"); return 0

def cmd_explain(a):
    r=json.loads(Path(a.report).read_text(encoding="utf-8")); m=r["measurement"]; p=r["policy"]
    print("Decision:",r["decision"]); print("Estimator:",r["object_of_measurement"]["estimator"])
    print("Max PFR:",f"{m['max_pfr']:.6f}"); print("Detection rate:",f"{m['detection_rate']:.3f}")
    print("Positive folds:",f"{m['positive_folds']}/{m['total_folds']}"); print("Expected relation:",p["expected_relation"])
    print("Claim ceiling: decision applies only to this declared run configuration."); return 0

def cmd_doctor(_):
    import importlib.metadata as md, platform
    print("HACO doctor: PASS"); print("haco-stability:",__version__); print("python:",platform.python_version())
    for pkg in ["numpy","scikit-learn","scipy","joblib","threadpoolctl","PyYAML","jsonschema"]:
        try: print(f"{pkg}:",md.version(pkg))
        except Exception: print(f"{pkg}: not-installed")
    return 0

def build_parser():
    p=argparse.ArgumentParser(prog="haco"); s=p.add_subparsers(dest="command",required=True)
    t=s.add_parser("test"); t.add_argument("--config",required=True); t.add_argument("--output-dir"); t.set_defaults(func=cmd_test)
    v=s.add_parser("validate-report"); v.add_argument("report"); v.set_defaults(func=cmd_validate)
    vr=s.add_parser("verify-report"); vr.add_argument("report"); vr.set_defaults(func=cmd_verify)
    e=s.add_parser("explain"); e.add_argument("report"); e.set_defaults(func=cmd_explain)
    d=s.add_parser("doctor"); d.set_defaults(func=cmd_doctor)
    ver=s.add_parser("version"); ver.set_defaults(func=lambda _:(print(__version__) or 0))
    return p

def main():
    a=build_parser().parse_args(); return a.func(a)
if __name__=="__main__": raise SystemExit(main())
