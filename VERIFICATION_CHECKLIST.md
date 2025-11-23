# Repository Verification Checklist

Quick checklist to verify the repository is ready for thesis submission.

## Pre-Submission Checks

### 1. Configuration Files
- [ ] Update all hardcoded paths in `config.json`:
  - Device ID (e.g., `cuda:0`, `cuda:2`)
  - Checkpoint paths for all models
  - Distinctiveness file paths
  - Threshold file paths (if using pre-computed thresholds)
- [ ] Verify `requirements.yml` is up to date
- [ ] Test that conda environment can be created: `conda env create -f requirements.yml`

### 2. Script Verification
- [ ] Syntax check passed (already verified)
- [ ] Test imports in conda environment:
  ```bash
  conda activate xai
  python -c "from third_party import run_models, utils; from ensemble import ensemble, evaluator; print('Imports OK')"
  ```
- [ ] Verify main scripts can at least parse arguments:
  ```bash
  python run_experiments.py --help
  python optimizer.py --help
  ```

### 3. Data Paths
- [ ] Update CheXpert dataset paths in `third_party/run_models.py`:
  - Line ~193: validation CSV path
  - Line ~211: test CSV path
- [ ] Ensure dataset is accessible at specified paths
- [ ] Verify checkpoint files exist at paths specified in `config.json`

### 4. Output Directories
- [ ] Check that `.gitignore` properly excludes:
  - Large output directories (`ensemble_*/`, `distinctiveness_*/`, etc.)
  - Cache files (`__pycache__/`, `*.pyc`)
  - Database files (`*.db`)
  - Log files (`*.txt`, `*.log`)
- [ ] Verify no large files are tracked: `git status` should not show output directories

### 5. Documentation
- [ ] README.md is complete and accurate
- [ ] All hardcoded paths are documented with `NOTE:` comments
- [ ] Usage examples in README match actual script behavior

### 6. Code Quality
- [ ] All main scripts have module docstrings
- [ ] Key functions have docstrings
- [ ] Hardcoded paths are clearly marked
- [ ] No obvious syntax errors (verified)

## Quick Test Run (Optional)

If you have the environment set up, you can do a minimal test:

```bash
conda activate xai
cd /home/fkirchhofer/repo/xai_thesis

# Test 1: Check imports
python -c "from third_party import run_models; print('✓ Imports work')"

# Test 2: Check argument parsing
python run_experiments.py --help

# Test 3: Verify config can be loaded (if config.json exists)
python -c "import json; f=open('config.json'); json.load(f); print('✓ Config valid')"
```

## Known Issues / Notes

1. **Config argument**: The `--config` argument in `run_experiments.py` should work via command line. If it doesn't, ensure you're running from the repo root.

2. **Hardcoded paths**: Several scripts contain hardcoded paths starting with `/home/fkirchhofer/`. These are marked with `NOTE:` comments and should be updated for different environments.

3. **Dependencies**: The code requires PyTorch, Captum, and other packages. Ensure the conda environment is properly set up before running scripts.

4. **GPU requirement**: Scripts are configured for GPU use. If running on CPU, update `device` in `config.json` to `"cpu"`.

## Files Modified for Cleanup

- ✅ `README.md` - Fixed merge conflict, added documentation
- ✅ `.gitignore` - Expanded to exclude output files
- ✅ `run_experiments.py` - Added comments for hardcoded paths
- ✅ `third_party/run_models.py` - Added module docstring, path comments
- ✅ `grid_search.py` - Added module docstring, parameter explanations
- ✅ `third_party/utils.py` - Added module docstring, key function docstrings
- ✅ `optimizer.py` - Already had good documentation

## Submission Readiness

Once all items above are checked, the repository should be ready for thesis submission. The code is well-documented, properly organized, and all critical paths are marked for easy adaptation to different environments.

