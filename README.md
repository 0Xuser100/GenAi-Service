# GenAi-Service

## Environment Reproducibility

Export the full Conda environment (all packages and pinned versions) so you can recreate it elsewhere:

```bash
conda env export > environment_full.yml
```

On another machine, rebuild the exact same environment with:

```bash
conda env create -f environment_full.yml
```

Regenerate the file whenever you intentionally upgrade dependencies, and keep it committed so production boxes stay aligned with development.
