# RunPod Deployment Guide

## Quick Start

### 1. Launch RunPod Instance

1. Go to [RunPod.io](https://www.runpod.io/)
2. Select **GPU Instances** → **Deploy**
3. **Recommended GPU**: NVIDIA A100 (40GB or 80GB)
   - Alternative: RTX 4090, A6000, or any CUDA-capable GPU
4. **Template**: PyTorch (or any template with PyTorch pre-installed)
5. **Container Disk**: 20GB minimum
6. **Volume**: Optional (for persistent storage)
7. Click **Deploy**

### 2. Connect to Your Instance

Once the instance is running:
- Click **Connect** → **Start Jupyter Lab** or **SSH**
- For SSH: Use the provided SSH command
- For Jupyter: Open a terminal within Jupyter Lab

### 3. Run the Setup Script

**Option A: Automatic Setup (Recommended)**

```bash
# Clone the repository
cd /workspace
git clone https://github.com/PakwhanNK/ese-3060-project.git
cd ese-3060-project

# Checkout the baseline experiment branch
git checkout experiment/exp000-baseline-100runs

# Make the setup script executable and run it
chmod +x runpod_setup.sh
./runpod_setup.sh
```

**Option B: Manual Setup**

```bash
# Clone and setup
cd /workspace
git clone https://github.com/PakwhanNK/ese-3060-project.git
cd ese-3060-project
git checkout experiment/exp000-baseline-100runs

# Install dependencies
pip install -r requirements.txt

# Run the experiment
python airbench94.py \
    --exp_name exp000-baseline-100runs \
    --desc "Baseline CIFAR-10 with default hyperparameters - 100 runs for statistical significance" \
    --runs 100
```

### 4. Monitor Progress

The experiment will display:
- Real-time progress for each run
- Per-run metrics (accuracy, time, epochs)
- Live updates to CSV file

Expected runtime on A100: **6-7 minutes** (100 runs × ~3.83s/run)

### 5. Download Results

After completion, results are saved to `experiments/exp000-baseline-100runs/`:

**Method 1: Zip and Download via Jupyter**
```bash
zip -r exp000-baseline-100runs.zip experiments/exp000-baseline-100runs/
# Download the zip file through Jupyter Lab file browser
```

**Method 2: SCP (if using SSH)**
```bash
# From your local machine:
scp -r -P YOUR_SSH_PORT root@YOUR_POD_IP:/workspace/ese-3060-project/experiments/exp000-baseline-100runs/ ~/Downloads/
```

**Method 3: Git Commit and Push**
```bash
git add experiments/exp000-baseline-100runs/
git commit -m "Add baseline experiment results (100 runs)"
git push origin exp000-baseline-100runs
```

---

## File Structure After Experiment

```
experiments/exp000-baseline-100runs/
├── metadata.json          # Git commit, GPU info, hyperparameters, RunPod info
├── runs_detailed.csv      # Per-run results with seeds (100 rows)
├── summary.json           # Statistical summary with confidence intervals
└── results.pt             # PyTorch checkpoint format
```

---

## Statistical Output

The `summary.json` will include:

**Overall Statistics:**
- `num_runs`: 100
- `accuracy_mean`, `accuracy_std`, `accuracy_sem`
- `accuracy_ci_95_lower`, `accuracy_ci_95_upper` ← **95% confidence interval**
- `accuracy_min`, `accuracy_max`, `accuracy_median`
- `time_mean`, `time_std`, `time_sem`
- `time_ci_95_lower`, `time_ci_95_upper`
- `success_rate`: Fraction achieving ≥94% accuracy
- `num_successful`: Count achieving ≥94% accuracy

**Successful Runs (≥94% accuracy):**
- Same metrics as above, but only for successful runs
- `best_run_time`: Fastest successful run
- `best_run_id`: ID of fastest run

---

## Cost Estimation

- **A100 (40GB/80GB)**: ~$0.40-$0.80/hour on RunPod
- **Experiment time**: ~7 minutes = 0.117 hours
- **Cost per experiment**: ~$0.05-$0.10
- **Total budget**: 40 A100-hours = ~$16-$32
- **Experiments possible**: 342+ full experiments (100 runs each)

---

## Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller batch size (edit airbench94.py line 44)
batch_size = 512  # instead of 1024
```

### Missing CIFAR-10 Dataset
```bash
python -c "from torchvision import datasets; datasets.CIFAR10(root='.', train=True, download=True)"
```

### Git Authentication Issues
```bash
# Repository is public, no authentication needed for cloning
git clone https://github.com/PakwhanNK/ese-3060-project.git

# For pushing results (if you have write access), use SSH or token:
# SSH:
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub  # Add to GitHub
# OR HTTPS with token:
git clone https://YOUR_TOKEN@github.com/PakwhanNK/ese-3060-project.git
```

### Logger Import Errors
Ensure scipy is installed:
```bash
pip install scipy
```

---

## Next Steps After Baseline

Once baseline is complete:

1. **Analyze Results**: Check `summary.json` for accuracy mean and CI
2. **Compare with Target**: Baseline should achieve ~94.01% accuracy
3. **Run Other Experiments**: Switch to other experiment branches
   ```bash
   git checkout experiment/exp100-muon-test
   python airbench94.py --exp_name exp100-muon-test --desc "Muon optimizer test" --runs 100
   ```
4. **Compare Experiments**: Use the comparison feature
   ```bash
   python airbench94.py --compare exp000-baseline-100runs exp100-muon-test
   ```

---

## Support

- **RunPod Docs**: https://docs.runpod.io/
- **Project Guidelines**: See `project_guidelines.md`
- **Issues**: Check GitHub repo issues

---

## Important Notes

⚠️ **Remember to STOP your RunPod instance** when not in use to avoid unnecessary charges!

✅ The experiment logger automatically detects RunPod instances and includes:
- RunPod Pod ID (from `RUNPOD_POD_ID` env variable)
- Instance type (GPU name)
- All required metadata for reproducibility
