# Setup & Environment Configuration

## Python Environment

### Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
```
pandas>=1.3.0           # Data manipulation
numpy>=1.21.0           # Numerical computing
scikit-learn>=1.0.0     # ML utilities (preprocessing, metrics)
xgboost>=1.5.0          # Gradient boosting
tensorflow>=2.8.0       # LSTM & CNN models
optuna>=2.10.0          # Hyperparameter tuning
MetaTrader5>=5.0.33     # MT5 API
```

---

## MT5 Configuration

### Step 1: Create/Access Demo Account
1. Open MetaTrader 5
2. Go to File → Open Account
3. Select "Open Demo Account"
4. Choose "MetaQuotes-Demo" server
5. Fill in your details (name, email, etc.)
6. Set leverage (default 100:1)
7. Initial deposit (default $100,000)

### Step 2: Get Your Login Credentials
```
Login: 5050913403         # Demo account number
Password: Ahmed@477447    # Account password
Server: MetaQuotes-Demo   # Server name
```

### Step 3: Update Configuration
File: `src/mt5_trader.py`
```python
MT5_LOGIN = 5050913403
MT5_PASS = "Ahmed@477447"
MT5_SERVER = "MetaQuotes-Demo"
```

### Step 4: Test Connection
```python
from src.mt5_trader import MT5Trader

trader = MT5Trader(login=5050913403, password="Ahmed@477447", server="MetaQuotes-Demo")
if trader.connect():
    print("Connected!")
    summary = trader.get_account_summary()
    print(f"Equity: ${summary['equity']}")
    trader.disconnect()
else:
    print("Connection failed!")
```

---

## GPU Acceleration (Optional)

### For NVIDIA GPU
```bash
# Uninstall default TensorFlow
pip uninstall tensorflow

# Install GPU version
pip install tensorflow[and-cuda]
```

### Check GPU Availability
```python
import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))
```

### For Apple Silicon (M1/M2)
```bash
pip install tensorflow-macos tensorflow-metal
```

---

## File Paths Configuration

### Data Directory
Default: `training/data/raw/mt5/`

If you want to change:
```python
DATA_DIR = Path('path/to/data')
```

### Models Directory
Default: `training/models/`

All trained models save here automatically.

### Logs Directory (Optional)
Create for experiment logging:
```bash
mkdir logs
```

---

## Testing Installation

### Quick Test
```bash
python -c "import pandas, numpy, sklearn, xgboost, tensorflow; print('All packages OK')"
```

### Test MT5 Connection
```bash
python -c "
from src.mt5_trader import MT5Trader
trader = MT5Trader(login=5050913403, password='Ahmed@477447', server='MetaQuotes-Demo')
print('Connected' if trader.connect() else 'Failed')
trader.disconnect()
"
```

### Test Models
```bash
python check_training_status.py
```

---

## Troubleshooting

### ModuleNotFoundError: No module named 'MetaTrader5'
```bash
pip install MetaTrader5
```

### TensorFlow GPU Not Detected
```python
import tensorflow as tf
devices = tf.config.list_physical_devices()
print("Devices:", devices)
```

### MT5 Connection Timeout
- Check internet connection
- Verify MetaTrader 5 is open
- Check login credentials
- Try "MetaQuotes-Demo" server

### Out of Memory During Training
```bash
# Reduce batch size in trainer scripts
BATCH_SIZE = 16  # was 32
EPOCHS = 50      # was 100
```

### Models Not Found Error
```bash
# Make sure models directory exists
mkdir training/models

# Check for trained models
ls training/models/
```

---

## System Requirements

### Minimum
- **CPU:** Intel i5 / AMD Ryzen 5 (8 cores)
- **RAM:** 8 GB
- **Storage:** 10 GB (for data + models)
- **GPU:** Optional (2GB VRAM)

### Recommended
- **CPU:** Intel i7 / AMD Ryzen 7+ (8+ cores)
- **RAM:** 16+ GB
- **Storage:** 20+ GB SSD
- **GPU:** NVIDIA RTX 2060+ or equivalent

### Training Time (Estimated)
| Model | TF | CPU | GPU |
|-------|----|----|-----|
| XGBoost | 1D | 5 min | 3 min |
| XGBoost | 4H-15min | 5 min each | 3 min each |
| LSTM | 1D | 20 min | 8 min |
| LSTM | 4H-15min | 30 min each | 15 min each |
| 1D-CNN | 15min-1H | 15 min each | 7 min each |
| Ensemble | All | 5 min | 5 min |
| **Total** | **All TFs** | **2-3 hours** | **1-1.5 hours** |

---

## Conda Alternative

### Create Conda Environment
```bash
conda create -n ml-trading python=3.9
conda activate ml-trading
conda install pandas numpy scikit-learn xgboost tensorflow optuna
pip install MetaTrader5
```

---

## Docker Setup (Advanced)

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0"]
```

### Build & Run
```bash
docker build -t ml-trading:latest .
docker run -it -p 8888:8888 ml-trading:latest
```

---

## IDE Setup

### VS Code
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.python"
  }
}
```

### PyCharm
1. Preferences → Python Interpreter → Add Interpreter → Add Local Interpreter
2. Select `.venv/bin/python`
3. Apply

---

## Dependencies Explained

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | ≥1.3 | Data manipulation (OHLC, features) |
| numpy | ≥1.21 | Numerical computing (arrays, math) |
| scikit-learn | ≥1.0 | ML preprocessing & metrics |
| xgboost | ≥1.5 | Gradient boosting model |
| tensorflow | ≥2.8 | LSTM & CNN models |
| optuna | ≥2.10 | Hyperparameter optimization |
| MetaTrader5 | ≥5.0 | MT5 API for live data |

---

## Performance Optimization

### NumPy Multi-threading
```python
import os
os.environ['OMP_NUM_THREADS'] = '8'  # Set to CPU core count
```

### XGBoost GPU
```python
xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0)
```

### TensorFlow Mixed Precision
```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

---

## Verify Everything Works

### Step-by-Step Verification
```bash
# 1. Activate environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 2. Test imports
python -c "import pandas; print(f'Pandas {pandas.__version__} OK')"
python -c "import xgboost; print(f'XGBoost {xgboost.__version__} OK')"
python -c "import tensorflow; print(f'TensorFlow {tensorflow.__version__} OK')"

# 3. Test MT5
python -c "from src.mt5_trader import MT5Trader; print('MT5Trader imported OK')"

# 4. Test feature engineering
python -c "from training.feature_engineering_v2 import build_features_v2; print('Feature engineering OK')"

# 5. Test models
python check_training_status.py

# 6. Run demo (if models available)
python examples/demo_signal_generation.py
```

---

## Next Steps

1. ✅ Install Python & packages
2. ✅ Configure MT5 credentials
3. ✅ Test connections
4. ✅ Run training scripts
5. ✅ Validate models
6. 🚀 Deploy!

---

**Setup Complete? Run:** `python examples/demo_signal_generation.py` 🎉
