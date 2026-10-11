# Dataset attribution

`parkinsons_updrs.data` and `parkinsons_updrs.names` are the unchanged files from UCI Parkinsons Telemonitoring (dataset 189), retrieved 11 October 2026.

Tsanas, A. & Little, M. (2009). Parkinsons Telemonitoring [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5ZS3N.

Source: https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring

Download: https://archive.ics.uci.edu/static/public/189/parkinsons+telemonitoring.zip

License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.

The dataset contains 5,875 recordings, 42 patient identities, 16 voice measures, demographics, recording time, and two interpolated clinical scores. The original `.names` file contains an inconsistent attribute-count heading; the actual table has 22 columns. No raw audio is provided.

The original data are unmodified. `SLT.py` selects 16 columns as predictors and uses `motor_UPDRS` as the target.

SHA-256 of `parkinsons_updrs.data`:
`f2c7d5025dec4e92e7feae367a5f7ccf58789a10ac6b54bdf15976c599f9dd39`
