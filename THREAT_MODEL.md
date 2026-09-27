# Threat Model — estimator factory

`factory: module:function` executes user-provided Python code.

RC3 treats factory mode as a **trusted-code interface**, not a sandbox. A malicious factory may read/write files, inspect protocol details, exhaust resources, or engineer outputs.

Rules:
- never execute an untrusted factory in a privileged environment;
- record the factory source hash when available;
- treat the factory as part of the audited configuration;
- future SaaS use must isolate uploaded code or prohibit arbitrary factories.
