# Security

HACO is research software.

## Trusted estimator factories

The `factory: module:function` interface executes Python code supplied by the user.
It is a **trusted-code interface**, not a sandbox.

Do not execute untrusted factory modules in privileged or sensitive environments.

## Reporting security issues

Before a public support/security address is designated, please use GitHub's private security reporting feature if enabled.
