# Offline clean installation

RC3 distinguishes cloud-sandbox execution from offline clean installation.

To prepare a wheelhouse, use a CONNECTED machine matching the evaluator's OS/Python ABI:

```bash
python prepare_offline_wheelhouse.py
```

Then freeze the resulting `wheelhouse/` and its hashes before evaluation.

On the offline evaluator machine:

```bash
python -m venv .venv
# activate
python -m pip install --no-index --find-links=wheelhouse haco-stability==1.0.0rc3
haco doctor
```

Do not claim a wheelhouse is portable across incompatible operating systems or Python ABIs.
