# Streamlit Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud (Recommended)

Streamlit Cloud makes it easy to share the app with your brother via a public URL.

### Step 1: Sign Up for Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click "Sign up" and log in with GitHub account
3. Authorize Streamlit to access your GitHub repos

### Step 2: Deploy the App
1. Click **"New app"** button
2. Fill in the form:
   - **Repository:** ahmedhussain47/Stocks-Analysis-System (or your fork)
   - **Branch:** master
   - **Main file path:** app.py
3. Click **Deploy**

The app will deploy in ~2-3 minutes. You'll get a URL like:
```
https://[your-repo-name]-[random].streamlit.app
```

---

## ⚠️ Important: Environment Variables

Streamlit Cloud runs in a sandboxed environment. The app uses local MT5 connections which won't work on cloud servers.

### Solution: Two Deployment Options

#### Option A: Share with Brother (Recommended)
- Brother runs the app locally on his PC with MT5 installed
- Use the `run_streamlit.ps1` script or `streamlit run app.py`
- Brother logs in with credentials and generates signals

**Pros:**
- ✓ Full MT5 access
- ✓ Real-time signals
- ✓ No cloud restrictions

**Cons:**
- ✗ Requires MT5 installed on brother's PC

#### Option B: Cloud Deployment (Limited)
- Deploy to Streamlit Cloud for public access
- **Limitation:** Can't connect to local MT5
- **Use case:** Share UI/access controls only, or use paper trading API

### To Deploy with Option B:
You would need to:
1. Set up MT5 API access from cloud
2. Use WebSocket connection to local MT5 terminal
3. Or deploy a separate MT5 bridge server

This is more complex and not recommended for your use case.

---

## Best Setup for Your Brother

### Step 1: Give Brother Access
Share the login credentials:
```
Role: brother
Password: Brother_Access_2024
```

### Step 2: Brother's Local Setup
```bash
# Clone the repo
git clone https://github.com/ahmedhussain47/Stocks-Analysis-System.git

# Install Python dependencies
pip install -r requirements.txt

# Install MT5
pip install MetaTrader5

# Run the app
streamlit run app.py
```

### Step 3: Admin Control (You)
- You keep Streamlit app on your PC
- Log in as admin: `Ahmed_Admin_2024`
- Control when brother can use it
- Set daily signal limits
- Monitor usage

---

## If You Want Cloud Hosting Anyway

For a fully cloud-hosted solution without MT5, you could:

1. **Use Paper Trading API** (no real MT5 connection)
2. **Deploy Flask backend** that connects to MT5 locally
3. **Use Streamlit Community Cloud** with hardcoded test data

But this defeats the purpose of live signals.

---

## Recommended: Keep It Local

The best approach is:
- ✓ **Admin (you):** Run `streamlit run app.py` locally
- ✓ **Brother:** Use same app locally, log in with credentials
- ✓ **GitHub:** Code is on GitHub for backup + version control
- ✓ **MT5:** Both have direct access to real-time data

---

## Troubleshooting Cloud Deployment

**"ModuleNotFoundError: No module named 'MetaTrader5'"**
- MetaTrader5 doesn't work on Streamlit Cloud (platform restriction)
- Use local deployment instead

**"Cannot import src.signal_engine_v2"**
- Make sure the repo structure matches:
  ```
  .
  ├── app.py
  ├── src/
  │   ├── signal_engine_v2.py
  │   ├── model_loader.py
  │   └── feature_engineering.py
  ├── training/models/
  │   └── [model files]
  └── requirements.txt
  ```

---

## Sharing the App with Brother

### Option 1: Local Network
If on same WiFi:
```bash
streamlit run app.py --server.address 0.0.0.0
```
Brother can access via: `http://your-ip:8501`

### Option 2: Ngrok Tunnel (Temporary)
```bash
pip install ngrok
ngrok http 8501
```
Share the ngrok URL with brother

### Option 3: Tailscale VPN (Persistent)
- Set up Tailscale for private network access
- Brother can access even outside your WiFi

---

## Updating the App

When you update `app.py`:
```bash
git add app.py
git commit -m "Update: ..."
git push origin master
```

If deployed to Streamlit Cloud, it auto-redeploys.
If brother runs locally, he pulls the latest:
```bash
git pull origin master
```

---

## Support

For Streamlit Cloud issues: [docs.streamlit.io](https://docs.streamlit.io)
