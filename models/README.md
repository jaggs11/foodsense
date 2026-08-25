# Models

Trained Stage-1 suitability surrogates and their metrics JSON land here.

```bash
make train       # ./make.ps1 train on Windows
```

Model binaries are produced by `foodsense.stage1_prediction.train` and are small enough to
commit; the metrics JSON alongside each one records the exact training run that produced it.
