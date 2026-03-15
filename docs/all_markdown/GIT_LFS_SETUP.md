# Git LFS setup for large QSAR / REINVENT artifacts

This repository contains large binary artifacts such as:

- REINVENT checkpoint files (`*.chkpt`)
- trained model files (`*.model`, `*.joblib`)
- generated workbook outputs (`*.xlsx`)

These files are now marked in `.gitattributes` for Git LFS tracking.

## Why this is needed

GitHub accepted the push, but warned that several files are larger than the recommended 50 MB size.
Without Git LFS, future pushes will remain large and repository history will continue to grow quickly.

## Install Git LFS

On Fedora/RHEL-compatible systems:

```bash
sudo dnf install git-lfs
git lfs install
```

## Verify tracking rules

```bash
git lfs track
cat .gitattributes
```

## Migrate existing history to LFS

Warning: this rewrites git history.
Only do this when you are ready to update the remote branch.

```bash
git lfs migrate import --include="*.chkpt,*.model,*.joblib,*.xlsx"
git push --force-with-lease origin main
```

## Recommended practice going forward

- Keep large generated outputs and checkpoints in LFS.
- Keep lightweight source code, configs, and CSV metadata in normal Git.
- Avoid committing duplicate binary exports when a CSV equivalent already exists.
