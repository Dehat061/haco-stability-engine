# CSV mode

A user can audit a numeric tabular CSV:

```yaml
data:
  kind: csv
  path: /absolute/or/relative/data.csv
  target: label
```

RC2 requires all non-target feature columns to be numeric.

For a user-defined model:

```yaml
estimator:
  factory: my_module:build_model
```

`build_model()` must return a fresh unfitted object exposing `fit(X, y)` and `predict(X)`.
Because importing a factory executes local Python code, only use trusted modules.
