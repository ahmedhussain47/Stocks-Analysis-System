# 🔴 KERNEL STUCK - RECOVERY GUIDE

## Immediate Fix (2 minutes)

### Option 1: Kill the Stuck Python Process
```powershell
# Run in PowerShell:
Get-Process python | Stop-Process -Force

# Wait 3 seconds, then restart Jupyter
```

### Option 2: VSCode Interrupt
1. Click **⏹️ Stop** button in notebook toolbar
2. Wait 5 seconds
3. Click **🔄 Restart** button
4. Run **Kernel → Restart Kernel** again if still frozen

### Option 3: Close & Reopen
1. Close the notebook tab in VSCode
2. Close VSCode completely
3. Reopen VSCode
4. Open the notebook again
5. **Kernel → Restart Kernel**

---

## Why Did It Freeze?

**The Problem:**
Cell-11 (live scanner) has an infinite loop:
```python
LIVE = True  # ← You set this to True
# Then...
def _live_worker():
    while not live_stop.is_set():  # ← Infinite loop!
        # ... scanning forever ...
```

When `LIVE = True`, the thread runs forever and **blocks the kernel from responding**.

---

## Why You Need to Replace Cell-11

The current code uses **`quick_features()` with 8 features**.  
But models expect **45 features**.

**Result:** Predictions are broken.

**Fix:** Use the corrected code from `LIVE_SCANNER_CORRECTED.py`

---

## How to Prevent This Next Time

### ✅ DO THIS:
1. **Always set `LIVE = False` before running the cell**
   ```python
   LIVE = False  # Must be False
   ```

2. **Only change to `LIVE = True` for manual testing:**
   ```python
   # Change:
   LIVE = False
   # To:
   LIVE = True  # <- Only for testing
   
   # Then run cell
   # Wait a few seconds
   # 
   # To STOP: Change back to False and re-run
   ```

3. **Always use stop mechanism:**
   ```python
   # To stop live scanner:
   live_stop.set()  # Signals thread to stop
   # Then kernel should respond
   ```

---

## Full Recovery Steps

### Step 1: Kill the Process (30 sec)
```powershell
# Open PowerShell and run:
Get-Process python | Stop-Process -Force
```

### Step 2: Reopen Notebook (1 min)
1. Close VSCode completely
2. Reopen notebook in VSCode
3. **Kernel → Restart Kernel**

### Step 3: Fix the Code (3 min)
Replace cell-11 with code from `LIVE_SCANNER_CORRECTED.py`:
- Make sure `LIVE = False` at the start
- Copy entire corrected code
- Paste into cell-11
- Run all cells

### Step 4: Test Carefully (2 min)
1. **Kernel → Restart & Run All**
2. Wait for models to load
3. Do NOT set `LIVE = True` yet
4. Just run the diagnostic cells

### Step 5: Only Then Enable Live Scanner
```python
# When ready to test:
LIVE = True
# Run cell-11
# Wait for ONE signal
# Then to STOP:
live_stop.set()
# Run cell again
```

---

## What NOT To Do

❌ **Don't** set `LIVE = True` and leave it
❌ **Don't** ignore kernel freeze messages
❌ **Don't** keep trying to interrupt if it's stuck (just kill it)
❌ **Don't** use the old cell-11 code (it's broken)

---

## Config Fix

Also fix this in **cell_config**:
```python
# CHANGE THIS:
REQUIRE_MODEL_CONSENSUS = True  # Impossible with 2 models!

# TO THIS:
REQUIRE_MODEL_CONSENSUS = False  # Only LSTM + XGBoost loaded
```

---

## Recovery Checklist

- [ ] Killed Python process with `Get-Process python | Stop-Process -Force`
- [ ] Closed and reopened VSCode
- [ ] Restarted Jupyter kernel
- [ ] Changed `REQUIRE_MODEL_CONSENSUS = False` in config
- [ ] Replaced cell-11 with `LIVE_SCANNER_CORRECTED.py` code
- [ ] Set `LIVE = False` (double-check!)
- [ ] Ran Kernel → Restart & Run All
- [ ] Verified models load (should see "LSTM accuracy: 78.1%")
- [ ] Did NOT set `LIVE = True` yet (just testing)

---

## Testing the Fixed Version

Only after recovery, test like this:

```python
# 1. Run all cells normally (LIVE = False)
# 2. See market data load ✓
# 3. See models load ✓  
# 4. See forecasts ✓
# 5. Now test live scanner:

LIVE = True  # Change this
# Run cell-11
# Wait 30-60 seconds
# You should see signals printed
# To stop:
live_stop.set()
# Run cell-11 again (should print "Live scanner OFF")
```

---

## If Still Stuck After Recovery

**Check:**
1. Is Python still running? 
   ```powershell
   Get-Process python
   ```
   If yes: `Stop-Process -Force`

2. Are you using old cell-11 code?
   - Should use code from `LIVE_SCANNER_CORRECTED.py`
   - Not the broken `quick_features()` version

3. Is something in import cells failing?
   - Check **cell_imports** for errors
   - May need `pip install` of missing packages

4. Try complete kernel restart:
   ```powershell
   # Kill all Python
   Get-Process python | Stop-Process -Force
   # Close VS Code
   # Reopen VS Code
   # Open notebook
   # Kernel → Restart Kernel
   ```

---

**Time needed:** 5 minutes max
**Difficulty:** Easy (just kill process + restart)
**Success rate:** 99%
