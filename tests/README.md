# Test suite

Activate the project Conda environment before running tests:

```console
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ./venv
```

Run the ordinary suite without collecting coverage:

```console
make test
```

Skip the real-device ingestion regression during a fast development loop:

```console
make test-fast
```

Run the complete suite and enforce the 80% project coverage threshold:

```console
make test-coverage
```

Generated Versioneer code is excluded from the coverage denominator.
