# Can a voice model work for a new patient?

Estimate motor UPDRS (Parkinson's severity) from voice measurements. Compare three models, keeping each patient's recordings together. The mean baseline wins validation; its test MAE is 6.90 points.

Use Python 3.12:

```bash
python -m pip install -r requirements.txt
python SLT.py
```

Try `--development-only` to compare adding age and sex.

Data: [UCI Parkinsons Telemonitoring](https://doi.org/10.24432/C5ZS3N), Tsanas & Little (2009), CC BY 4.0. This is a small-cohort exercise, not a diagnostic tool.
