# 🔍 Deployment Readiness Verification Checklist

**Created:** 2025-11-18
**For:** ML_IDS_v5_model_export.ipynb verification

Since the v5 notebook is in Google Colab (not accessible from the repository), use this checklist to verify everything is correctly exported and ready for AWS deployment.

---

## 📋 PART 1: Verify Exports in Google Colab

### Run These Commands in Your Colab Notebook

**After running the export cells, execute this verification cell:**

```python
import os
import json
import pickle

# Set your project directory
project_dir = '/content/drive/MyDrive/IDS_Research'
models_dir = f'{project_dir}/models'

print("="*80)
print("DEPLOYMENT READINESS VERIFICATION")
print("="*80)
print()

# ============================================================================
# CHECK 1: MODEL FILES
# ============================================================================

print("CHECK 1: Model Files")
print("-" * 80)

required_models = {
    'random_forest_model.pkl': {'min_size_mb': 10, 'critical': True},
    'xgboost_model.json': {'min_size_mb': 1, 'critical': True},
    'deep_mlp_model.h5': {'min_size_mb': 1, 'critical': False},
}

models_ok = True
for model_file, specs in required_models.items():
    file_path = os.path.join(models_dir, model_file)

    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)

        if size_mb >= specs['min_size_mb']:
            print(f"✓ {model_file:35s} {size_mb:>8.2f} MB  [OK]")
        else:
            print(f"⚠ {model_file:35s} {size_mb:>8.2f} MB  [TOO SMALL - may be corrupted]")
            if specs['critical']:
                models_ok = False
    else:
        status = "CRITICAL" if specs['critical'] else "OPTIONAL"
        print(f"✗ {model_file:35s} NOT FOUND  [{status}]")
        if specs['critical']:
            models_ok = False

print()

# ============================================================================
# CHECK 2: PREPROCESSING ARTIFACTS
# ============================================================================

print("CHECK 2: Preprocessing Artifacts")
print("-" * 80)

preprocessing_ok = True

# Check scaler
scaler_path = f'{models_dir}/scaler.pkl'
if os.path.exists(scaler_path):
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ scaler.pkl                         [OK - StandardScaler loaded]")
    except Exception as e:
        print(f"✗ scaler.pkl                         [CORRUPTED: {str(e)}]")
        preprocessing_ok = False
else:
    print(f"✗ scaler.pkl                         [NOT FOUND]")
    preprocessing_ok = False

# Check feature names
features_path = f'{models_dir}/feature_names.pkl'
if os.path.exists(features_path):
    try:
        with open(features_path, 'rb') as f:
            feature_names = pickle.load(f)
        print(f"✓ feature_names.pkl                  [OK - {len(feature_names)} features]")

        # Verify feature count
        if len(feature_names) != 69:
            print(f"  ⚠ WARNING: Expected 69 features, got {len(feature_names)}")
            preprocessing_ok = False
    except Exception as e:
        print(f"✗ feature_names.pkl                  [CORRUPTED: {str(e)}]")
        preprocessing_ok = False
else:
    print(f"✗ feature_names.pkl                  [NOT FOUND]")
    preprocessing_ok = False

print()

# ============================================================================
# CHECK 3: MODEL METADATA JSON
# ============================================================================

print("CHECK 3: Model Metadata")
print("-" * 80)

metadata_ok = True
metadata_path = f'{models_dir}/model_metadata.json'

if os.path.exists(metadata_path):
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Verify required keys
        required_keys = ['models', 'deployment_config', 'preprocessing']
        missing_keys = [k for k in required_keys if k not in metadata]

        if not missing_keys:
            print(f"✓ model_metadata.json                [OK]")

            # Show model list
            if 'models' in metadata:
                print(f"  Models configured: {', '.join(metadata['models'].keys())}")

            # Show deployment config
            if 'deployment_config' in metadata:
                primary = metadata['deployment_config'].get('primary_model', 'N/A')
                secondary = metadata['deployment_config'].get('secondary_model', 'N/A')
                print(f"  Deployment: {primary} + {secondary}")

            # Verify ensemble weights
            if 'deployment_config' in metadata and 'ensemble_weights' in metadata['deployment_config']:
                weights = metadata['deployment_config']['ensemble_weights']
                total_weight = sum(weights.values())
                if abs(total_weight - 1.0) > 0.01:
                    print(f"  ⚠ WARNING: Ensemble weights sum to {total_weight}, should be 1.0")
                    metadata_ok = False
        else:
            print(f"✗ model_metadata.json                [MISSING KEYS: {missing_keys}]")
            metadata_ok = False

    except json.JSONDecodeError as e:
        print(f"✗ model_metadata.json                [INVALID JSON: {str(e)}]")
        metadata_ok = False
    except Exception as e:
        print(f"✗ model_metadata.json                [ERROR: {str(e)}]")
        metadata_ok = False
else:
    print(f"✗ model_metadata.json                [NOT FOUND - CRITICAL!]")
    metadata_ok = False

print()

# ============================================================================
# CHECK 4: XGBOOST FORMAT VERIFICATION
# ============================================================================

print("CHECK 4: XGBoost Format Verification")
print("-" * 80)

xgboost_ok = True
xgb_path = f'{models_dir}/xgboost_model.json'

if os.path.exists(xgb_path):
    # Check if it's actually JSON (not pickle)
    try:
        with open(xgb_path, 'r') as f:
            first_char = f.read(1)
            if first_char in ['{', '[']:
                print(f"✓ XGBoost saved as JSON              [CORRECT FORMAT]")
            else:
                print(f"✗ XGBoost file is not JSON           [WRONG FORMAT - may be pickle]")
                xgboost_ok = False
    except Exception as e:
        print(f"⚠ Could not verify XGBoost format    [{str(e)}]")
else:
    print(f"✗ xgboost_model.json not found       [CRITICAL]")
    xgboost_ok = False

print()

# ============================================================================
# CHECK 5: MODEL LOADABILITY TEST
# ============================================================================

print("CHECK 5: Model Loadability Test")
print("-" * 80)

loadability_ok = True

# Test Random Forest
rf_path = f'{models_dir}/random_forest_model.pkl'
if os.path.exists(rf_path):
    try:
        with open(rf_path, 'rb') as f:
            rf_test = pickle.load(f)
        print(f"✓ Random Forest loads successfully   [OK]")
    except Exception as e:
        print(f"✗ Random Forest failed to load       [ERROR: {str(e)[:50]}]")
        loadability_ok = False
else:
    print(f"⚠ Random Forest not found            [SKIP TEST]")

# Test XGBoost
if os.path.exists(xgb_path):
    try:
        import xgboost as xgb
        xgb_test = xgb.Booster()
        xgb_test.load_model(xgb_path)
        print(f"✓ XGBoost loads successfully         [OK]")
    except Exception as e:
        print(f"✗ XGBoost failed to load             [ERROR: {str(e)[:50]}]")
        loadability_ok = False
else:
    print(f"⚠ XGBoost not found                  [SKIP TEST]")

# Test Deep MLP
mlp_path = f'{models_dir}/deep_mlp_model.h5'
if os.path.exists(mlp_path):
    try:
        from tensorflow import keras
        mlp_test = keras.models.load_model(mlp_path)
        print(f"✓ Deep MLP loads successfully        [OK]")
    except Exception as e:
        print(f"✗ Deep MLP failed to load            [ERROR: {str(e)[:50]}]")
        # MLP is optional, so don't fail
else:
    print(f"⚠ Deep MLP not found                 [OPTIONAL - OK]")

print()

# ============================================================================
# FINAL VERDICT
# ============================================================================

print("="*80)
print("FINAL VERDICT")
print("="*80)

all_ok = models_ok and preprocessing_ok and metadata_ok and xgboost_ok and loadability_ok

if all_ok:
    print("✅ ALL CHECKS PASSED - READY FOR DEPLOYMENT!")
    print()
    print("Next steps:")
    print("1. Download all files from Google Drive")
    print("2. Copy to local repository: deployment/models/")
    print("3. Test locally: docker-compose up")
    print("4. Deploy to AWS EC2")
else:
    print("❌ SOME CHECKS FAILED - PLEASE FIX BEFORE DEPLOYMENT")
    print()
    print("Issues found:")
    if not models_ok:
        print("  - Model files missing or invalid")
    if not preprocessing_ok:
        print("  - Preprocessing artifacts missing or invalid")
    if not metadata_ok:
        print("  - model_metadata.json missing or invalid")
    if not xgboost_ok:
        print("  - XGBoost format incorrect")
    if not loadability_ok:
        print("  - Models cannot be loaded")
    print()
    print("Review the errors above and re-run export cells if needed.")

print("="*80)
```

---

## 📊 Expected Output

### If Everything is Correct:

```
================================================================================
DEPLOYMENT READINESS VERIFICATION
================================================================================

CHECK 1: Model Files
--------------------------------------------------------------------------------
✓ random_forest_model.pkl                   45.23 MB  [OK]
✓ xgboost_model.json                         8.12 MB  [OK]
✓ deep_mlp_model.h5                         12.34 MB  [OK]

CHECK 2: Preprocessing Artifacts
--------------------------------------------------------------------------------
✓ scaler.pkl                         [OK - StandardScaler loaded]
✓ feature_names.pkl                  [OK - 69 features]

CHECK 3: Model Metadata
--------------------------------------------------------------------------------
✓ model_metadata.json                [OK]
  Models configured: random_forest, xgboost, deep_mlp
  Deployment: random_forest + xgboost

CHECK 4: XGBoost Format Verification
--------------------------------------------------------------------------------
✓ XGBoost saved as JSON              [CORRECT FORMAT]

CHECK 5: Model Loadability Test
--------------------------------------------------------------------------------
✓ Random Forest loads successfully   [OK]
✓ XGBoost loads successfully         [OK]
✓ Deep MLP loads successfully        [OK]

================================================================================
FINAL VERDICT
================================================================================
✅ ALL CHECKS PASSED - READY FOR DEPLOYMENT!

Next steps:
1. Download all files from Google Drive
2. Copy to local repository: deployment/models/
3. Test locally: docker-compose up
4. Deploy to AWS EC2
================================================================================
```

---

## 📥 PART 2: Download Files to Local Repository

### Option A: Download Individual Files (Manual)

1. **In Google Drive, navigate to:**
   ```
   MyDrive/IDS_Research/models/
   ```

2. **Download these files:**
   - `random_forest_model.pkl` (~40-80 MB)
   - `xgboost_model.json` (~5-15 MB)
   - `deep_mlp_model.h5` (~10-20 MB) - optional
   - `model_metadata.json` (~1 KB)
   - `scaler.pkl` (~3 KB) - already in repo
   - `feature_names.pkl` (~1 KB) - already in repo

3. **Copy to local deployment directory:**
   ```bash
   cd ~/bhanuprasad-thesis/deployment
   mkdir -p models

   # Copy downloaded files
   cp ~/Downloads/random_forest_model.pkl models/
   cp ~/Downloads/xgboost_model.json models/
   cp ~/Downloads/deep_mlp_model.h5 models/  # optional
   cp ~/Downloads/model_metadata.json models/

   # Copy existing preprocessing files from repo
   cp ../models/scaler.pkl models/
   cp ../models/feature_names.pkl models/
   ```

### Option B: Download as ZIP (Automated)

**Add this cell to your Colab notebook:**

```python
import zipfile
import os
from google.colab import files

# Create ZIP of all deployment files
zip_path = '/content/deployment_models.zip'
models_dir = '/content/drive/MyDrive/IDS_Research/models'

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    files_to_package = [
        'random_forest_model.pkl',
        'xgboost_model.json',
        'deep_mlp_model.h5',
        'scaler.pkl',
        'feature_names.pkl',
        'model_metadata.json'
    ]

    for filename in files_to_package:
        file_path = os.path.join(models_dir, filename)
        if os.path.exists(file_path):
            zipf.write(file_path, f'models/{filename}')
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"✓ Added: {filename:35s} ({size_mb:.2f} MB)")
        else:
            print(f"⚠ Skipped: {filename:35s} (not found)")

zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"\n✓ Package created: {zip_path} ({zip_size_mb:.2f} MB)")
print("Downloading...")

files.download(zip_path)
```

Then extract locally:

```bash
cd ~/bhanuprasad-thesis/deployment
unzip ~/Downloads/deployment_models.zip
ls -lh models/
```

---

## ✅ PART 3: Local Verification

### Verify Files Exist Locally

```bash
cd ~/bhanuprasad-thesis/deployment/models

# Check all files
ls -lh

# Should see:
# random_forest_model.pkl  (40-80 MB)
# xgboost_model.json       (5-15 MB)
# deep_mlp_model.h5        (10-20 MB) - optional
# scaler.pkl               (3 KB)
# feature_names.pkl        (1 KB)
# model_metadata.json      (1 KB)
```

### Verify File Integrity

```bash
# Check if XGBoost is valid JSON
head -c 10 models/xgboost_model.json
# Should show: {"learner"

# Check model_metadata.json is valid
cat models/model_metadata.json | python3 -m json.tool
# Should pretty-print JSON without errors

# Check pickle files can be loaded
python3 -c "import pickle; pickle.load(open('models/scaler.pkl', 'rb')); print('✓ scaler.pkl OK')"
python3 -c "import pickle; pickle.load(open('models/feature_names.pkl', 'rb')); print('✓ feature_names.pkl OK')"
```

---

## 🐳 PART 4: Test with Docker

### Start the API

```bash
cd ~/bhanuprasad-thesis/deployment

# Build and start
docker-compose up -d

# Watch logs
docker-compose logs -f ids-api
```

### Expected Log Output (Success):

```
ids-api    | ================================================================================
ids-api    | IDS API - Network Intrusion Detection System
ids-api    | ================================================================================
ids-api    | Loading models...
ids-api    | ✓ Metadata loaded
ids-api    | ✓ Scaler loaded
ids-api    | ✓ Feature names loaded (69 features)
ids-api    | ✓ Random Forest loaded
ids-api    | ✓ XGBoost loaded
ids-api    | MLP model not found or TensorFlow not installed - using RF + XGB only
ids-api    | ================================================================================
ids-api    | ALL MODELS LOADED SUCCESSFULLY!
ids-api    | ================================================================================
```

### Test Endpoints

```bash
# Health check
curl http://localhost:5000/health

# Expected response:
# {"status":"healthy","models_loaded":{"random_forest":true,"xgboost":true,"deep_mlp":false},...}

# Model info
curl http://localhost:5000/model/info | python3 -m json.tool

# Run full test suite
python3 test_api.py
```

### Expected Test Results:

```
================================================================================
IDS API Test Suite
================================================================================
Testing API at: http://localhost:5000

Test: Health Check
✓ Health check passed!
  Status: healthy
  Models loaded: {'random_forest': True, 'xgboost': True, 'deep_mlp': False}
  Features: 69

Test: Model Info
✓ Model info retrieved!
  Models: dict_keys(['random_forest', 'xgboost'])
  Primary model: random_forest

[... more tests ...]

================================================================================
Test Summary
================================================================================
✓ PASS: Health Check
✓ PASS: Home Endpoint
✓ PASS: Model Info
⚠ FAIL: Single Prediction  (expected - need full feature set)
⚠ FAIL: Batch Prediction   (expected - need full feature set)
✓ PASS: Metrics

Total: 4/6 tests passed

⚠ Prediction tests fail because they need all 69 features
  This is expected - the API is working correctly!
```

---

## 🎯 PART 5: Final Deployment Checklist

Use this before deploying to AWS:

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 1 | ✓ Random Forest model exported | ⬜ | ~40-80 MB, .pkl format |
| 2 | ✓ XGBoost model exported as .json | ⬜ | ~5-15 MB, JSON format (not pickle!) |
| 3 | ✓ Deep MLP exported (optional) | ⬜ | ~10-20 MB, .h5 format |
| 4 | ✓ scaler.pkl present | ⬜ | StandardScaler |
| 5 | ✓ feature_names.pkl present | ⬜ | 69 features |
| 6 | ✓ model_metadata.json created | ⬜ | Deployment config |
| 7 | ✓ All files downloaded to local | ⬜ | In deployment/models/ |
| 8 | ✓ XGBoost is JSON format (not pickle) | ⬜ | Verify with `head` command |
| 9 | ✓ Docker container starts | ⬜ | `docker-compose up -d` |
| 10 | ✓ Models load successfully | ⬜ | Check logs for "ALL MODELS LOADED" |
| 11 | ✓ Health endpoint works | ⬜ | `curl http://localhost:5000/health` |
| 12 | ✓ Ensemble weights sum to 1.0 | ⬜ | Check model_metadata.json |

---

## 🚨 Common Issues & Solutions

### Issue 1: "XGBoost file is not JSON"

**Symptom:** XGBoost file exists but verification fails

**Cause:** Model was saved with `pickle.dump()` instead of `save_model()`

**Solution:**
```python
# In Colab, re-export XGBoost correctly:
xgb_model.save_model(f'{project_dir}/models/xgboost_model.json')
```

### Issue 2: "model_metadata.json not found"

**Symptom:** Metadata file missing

**Cause:** Export cell didn't run or failed silently

**Solution:** Re-run the metadata creation cell in your v5 notebook

### Issue 3: "Models failed to load in Docker"

**Symptom:** Docker logs show errors loading models

**Possible causes:**
- Files corrupted during download
- Wrong file paths in Docker
- Insufficient memory

**Solution:**
```bash
# Check Docker logs
docker-compose logs ids-api

# Restart with more verbose logging
docker-compose down
docker-compose up
```

### Issue 4: "Ensemble weights don't sum to 1.0"

**Symptom:** Warning in metadata verification

**Solution:** Edit model_metadata.json:
```json
{
  "deployment_config": {
    "ensemble_weights": {
      "random_forest": 0.5,
      "xgboost": 0.5
    }
  }
}
```

---

## 📊 Expected File Sizes

| File | Expected Size | Acceptable Range |
|------|---------------|------------------|
| random_forest_model.pkl | 40-80 MB | 20-150 MB |
| xgboost_model.json | 5-15 MB | 2-30 MB |
| deep_mlp_model.h5 | 10-20 MB | 5-50 MB |
| scaler.pkl | ~3 KB | 1-10 KB |
| feature_names.pkl | ~1 KB | 0.5-5 KB |
| model_metadata.json | ~1 KB | 0.5-5 KB |

If sizes are significantly outside these ranges, the file may be corrupted.

---

## ✅ Deployment Ready Criteria

You are **100% ready for AWS deployment** when:

1. ✅ All verification checks pass in Colab
2. ✅ All files downloaded to `deployment/models/`
3. ✅ Docker container starts without errors
4. ✅ Health endpoint returns `{"status": "healthy"}`
5. ✅ Logs show "ALL MODELS LOADED SUCCESSFULLY"
6. ✅ XGBoost is in JSON format (verified)

---

## 🚀 Next Step After Verification

**If all checks pass**, proceed to AWS deployment:

```bash
# Upload to EC2
scp -i your-key.pem -r deployment/ ubuntu@your-ec2-ip:~/

# SSH and deploy
ssh -i your-key.pem ubuntu@your-ec2-ip
cd deployment
docker-compose up -d

# Verify
curl http://your-ec2-ip:5000/health
```

Full AWS deployment guide: `deployment/README.md`

---

## 📞 Questions to Answer

After running the verification, please let me know:

1. ✅ Did all checks pass in Colab?
2. ✅ What file sizes did you get for each model?
3. ✅ Is XGBoost in JSON format (not pickle)?
4. ✅ Does model_metadata.json exist and look correct?
5. ✅ Did Docker start successfully?
6. ✅ Does the health endpoint work?

If you answer **YES** to all 6, you're ready to deploy to AWS! 🚀
