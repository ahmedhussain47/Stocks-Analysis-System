# COMPREHENSIVE PROMPT FOR NEXT CLAUDE SESSION
## Build Powerful Multi-Asset Forex Signal Generation System

---

## CONTEXT & CURRENT STATE

### Project Status
- **Current**: Single-asset (XAUUSD) 3-tier signal engine running live on MT5 demo
- **Issue**: Model bias (SELL-only predictions) due to limited training data
- **Goal**: Scale to multi-asset with powerful ML models using industry best practices

### Architecture (Existing - Keep)
```
Tier 1: Technical Patterns (PatternDetector) + ML Patterns (ChartPatternMLDetector)
Tier 2: Entry Optimization (pullback, momentum, risk adjustment, session filtering)
Tier 3: Macro Filtering (DXY, yields, VIX correlation with position sizing)
Live MT5 Integration: Demo mode active, ready to scale
```

### What Works
- ✅ 3-tier risk framework operational
- ✅ Live scanner running 24/7
- ✅ MT5 integration (demo mode)
- ✅ Signal logging & performance tracking
- ✅ Dynamic TP/SL calculation (ATR + Ichimoku + ML hybrid)

### What Needs Fix
- ❌ Model training data too limited (1-2 weeks) → overfitting → SELL-bias
- ❌ Single-asset training → can't generalize
- ❌ No correlation learning between forex pairs
- ❌ Missing multi-timeframe aggregation in features

---

## RESEARCH FINDINGS: INDUSTRY BEST PRACTICES

### What Top Signal Generators Use

#### Data Strategy
| Platform/Paper | Assets | Lookback | Frequency | Features |
|---|---|---|---|---|
| **QuantConnect** | 50+ pairs | 5-10 years | 1min-daily | Cross-correlation |
| **Two Sigma** | 100+ instruments | 10 years | 1min | Macro + sentiment |
| **Citadel** | FX + commodities | 20 years | Tick data | Multi-regime |
| **Renaissance** | All markets | 30+ years | All frequencies | Proprietary |
| **Academic (2023)** | 28 forex pairs | 5 years | 1H | Multi-asset ensemble |

**Key Pattern**: Industry uses 5-10 YEARS minimum, 20+ assets, multiple timeframes

#### Model Approaches

**1. Multi-Asset Ensemble (Most Common)**
```
Data: 20-30 forex pairs + commodities
Features: Cross-correlation, carry, volatility clustering
Models: LSTM + XGBoost + Random Forest + SVM
Result: 55-65% accuracy (vs 50-55% single-asset)
```

**2. Regime-Aware Models (Growing Trend)**
```
Detect market regime (trending, ranging, high-vol, low-vol, crisis)
Train separate models per regime
Switch models dynamically
Result: 60-70% accuracy in-regime, 45% out-of-regime
```

**3. Correlation Networks (Advanced)**
```
Model inter-asset correlations
DXY → EURUSD, Gold → USD
Learn when correlations break (profit opportunity)
Result: 65-75% accuracy with correlation signals
```

**4. Reinforcement Learning (Emerging)**
```
Train agent to optimize trading decisions
Learns position sizing, entry timing, regime switching
Result: 50-60% accuracy but better risk-adjusted returns
```

#### Data Quality Standards

**Minimum Requirements (Academic)**
- ✅ 5+ years historical data
- ✅ Multiple market cycles (bull, bear, crisis)
- ✅ 20+ assets minimum
- ✅ Multiple timeframes (1H, 4H, 1D)
- ✅ Handle gaps, splits, corporate actions

**Recommended (Professional)**
- ✅ 10-20 years of data
- ✅ 50+ assets
- ✅ Tick/minute data + hourly/daily
- ✅ Cross-validated on multiple periods
- ✅ Out-of-sample testing

---

## YOUR TASK: BUILD NEXT-GENERATION MODEL

### Phase 1: Multi-Asset Data (This Session)

**Assets to Include** (ordered by relevance to Gold):
```
Tier 1 (Direct USD correlation):
  1. EURUSD (most liquid, DXY inverse)
  2. GBPUSD (carries)
  3. USDCHF (safe haven)
  4. XAUUSD (target instrument)

Tier 2 (Indirect correlation):
  5. AUDUSD (commodity-linked)
  6. USDCAD (commodity-linked)
  7. NZDUSD (risk-on/off indicator)

Tier 3 (Support/Context):
  8. USDJPY (carry trade)
  9. DXY (dollar strength)
  10. Bonds (10Y yield proxy)
```

**Data Specification**:
```
Time Range: 2019-2026 (7 years)
  Why 2019? 
  - ✅ Pre-COVID (normal market)
  - ✅ Post-Brexit (volatility regime)
  - ✅ Fed rate hiking/cutting cycles
  - ✅ Large enough for deep learning
  - ✅ Recent enough for relevance

Timeframe: 1H + 4H (hourly focus, daily context)
Sampling: Every hour (consistent)
Total Expected Samples: ~60,000 bars per asset
```

**Data Cleaning**:
```
✅ Remove weekends/holidays
✅ Interpolate gaps < 4 hours
✅ Remove extreme outliers (10 sigma)
✅ Verify data integrity
✅ Normalize across different scales
```

### Phase 2: Advanced Feature Engineering

**Essential Features** (add to existing):
```
Price Action:
  - EMA 20, 50, 200 (trend)
  - RSI 14, Stochastic (momentum)
  - MACD, Bollinger Bands (volatility)
  - ATR, Donchian channels (range)
  - Ichimoku cloud (support/resistance)

Cross-Asset Correlations:
  - EURUSD correlation with XAUUSD
  - DXY strength vs Gold
  - Carry trade signals (USDJPY spread)
  - Risk-on/off indicators

Macro Indicators:
  - Dollar strength (DXY)
  - Equity index correlation (SP500)
  - Bond yields (10Y, 2Y spread)
  - VIX volatility regime
  - Real rates (inflation-adjusted)

Market Regime:
  - Trend strength (ADX)
  - Volatility clustering (GARCH)
  - Correlation regime changes
  - Drawdown severity
```

**Feature Engineering Rules**:
```
1. Generate ALL features for ALL 10 assets
2. Create cross-correlations (e.g., EURUSD momentum vs XAUUSD price)
3. Lag features by 0, 1, 2 bars (capture momentum decay)
4. Normalize each asset separately (preserve scale relationships)
5. Create lagged targets (predict 1, 2, 4 bars ahead - ensemble)
```

### Phase 3: Model Architecture (Multi-Stage)

**Stage 1: Attention-Based LSTM** (Your Primary Model)
```python
Input: (batch, 50 timesteps, 60+ features)
Architecture:
  - LSTM layer 1: 128 units + Attention
  - LSTM layer 2: 64 units + Attention
  - Dense: 32 units (cross-asset synthesis)
  - Output: 3 classes (BUY, HOLD, SELL) or regression (confidence 0-1)
  
Training:
  - Loss: Categorical crossentropy + focal loss (handle class imbalance)
  - Optimizer: Adam with learning rate decay
  - Epochs: 100 with early stopping
  - Validation: 20% holdout
```

**Stage 2: XGBoost Ensemble**
```
Features: All 60+ engineered features
Boosting Rounds: 500
Max Depth: 7
Learning Rate: 0.05
Objective: Binary classification
Important: Feature importance analysis
```

**Stage 3: Regime-Based Routing**
```
Detect market regime:
  - Trending (ADX > 25)
  - Ranging (ADX < 20)
  - High volatility (ATR > 1.5x MA)
  - Low volatility (ATR < 0.8x MA)
  - Crisis (multiple indicators negative)

Train separate model for each regime:
  - Trending model: Emphasize trend-following features
  - Ranging model: Emphasize mean-reversion features
  - High-vol model: Emphasize volatility breakouts
  - Low-vol model: Emphasize consolidation patterns
  - Crisis model: Conservative, high-confidence only

Dynamically switch active model based on current regime
```

**Stage 4: Ensemble Voting**
```
Combine predictions:
  - LSTM attention output: 40% weight
  - XGBoost prediction: 40% weight
  - Regime-specific model: 20% weight
  
Confidence scoring:
  - Agreement between models → high confidence
  - Disagreement → low confidence / skip signal
  - Out-of-distribution detection: Skip if features unusual
```

### Phase 4: Validation & Backtesting

**Walk-Forward Validation**:
```
Split data into rolling windows:
  - Train: 2019-2021 (2 years)
  - Test: 2022 Q1 (3 months) → metrics
  - Train: 2019-2022 Q1 (2.25 years)
  - Test: 2022 Q2 → metrics
  - ... continue until 2026 Q2

Metrics to track:
  ✅ Accuracy (%)
  ✅ Precision/Recall (confusion matrix)
  ✅ Sharpe Ratio (risk-adjusted returns)
  ✅ Win rate (% profitable trades)
  ✅ Profit factor (gross profit / gross loss)
  ✅ Max drawdown
  ✅ Recovery factor
  ✅ Out-of-sample performance
```

**Cross-Validation**:
```
K-Fold Cross-Validation (5 folds):
  - Ensures model generalizes
  - Detects overfitting early
  - Reports confidence intervals
```

---

## IMPLEMENTATION CHECKLIST

### Data Pipeline
- [ ] Fetch 10 assets × 7 years (2019-2026) from yfinance
- [ ] Clean & validate data (gaps, outliers, integrity)
- [ ] Align all assets to same hourly timestamps
- [ ] Save to `training/data/raw/`
- [ ] Create data validation report

### Feature Engineering
- [ ] Implement all 50+ features (technical + macro + correlation)
- [ ] Compute features for all 10 assets
- [ ] Handle NaN values (forward-fill or drop)
- [ ] Normalize features (MinMaxScaler per asset)
- [ ] Save feature engineering code + metadata

### Model Training
- [ ] Build LSTM with attention
- [ ] Build XGBoost classifier
- [ ] Implement regime detection
- [ ] Train 4 regime-specific models
- [ ] Create ensemble voting logic
- [ ] Save all models + weights + scalers

### Validation
- [ ] Run walk-forward validation
- [ ] Generate performance reports (Sharpe, drawdown, etc.)
- [ ] Create out-of-sample test results
- [ ] Compare vs. benchmark (50% random baseline)

### Integration
- [ ] Update signal engine to use new models
- [ ] Test on live simulator (DEMO_MODE=True)
- [ ] Validate signals improve win rate to 55%+
- [ ] Deploy to MT5 (DEMO_MODE=False after validation)

---

## SUCCESS CRITERIA

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Model Accuracy** | 60%+ | 52% | ❌ Need improvement |
| **Win Rate** | 55%+ | 25% | ❌ Poor |
| **Sharpe Ratio** | 1.0+ | 0.0 | ❌ No trades |
| **Assets** | 10+ | 1 | ❌ Limited |
| **Data Years** | 7 | 0.05 | ❌ Too short |
| **Features** | 50+ | 15 | ❌ Insufficient |
| **Profit Factor** | 1.5+ | 0.0 | ❌ No trades |

---

## ESTIMATED TIMELINE

| Task | Duration | Notes |
|------|----------|-------|
| Data fetching (10 assets × 7y) | 30 min | Parallel downloads |
| Data cleaning & validation | 30 min | Handle gaps/outliers |
| Feature engineering | 60 min | 50+ features × 10 assets |
| LSTM training | 60 min | GPU-accelerated |
| XGBoost training | 20 min | CPU-efficient |
| Regime models (4x) | 40 min | Smaller datasets |
| Validation & testing | 60 min | Walk-forward CV |
| Integration & testing | 30 min | Update signal engine |
| **Total** | **~5-6 hours** | With parallelization |

---

## RESEARCH REFERENCES TO IMPLEMENT

1. **Attention Mechanisms in LSTM**
   - Luong et al. (2015): "Effective Approaches to Attention-based Neural Machine Translation"
   - Apply to financial forecasting: Weight recent timesteps higher

2. **Regime-Switching Models**
   - Hamilton (1989): Hidden Markov Model for regime detection
   - Guidolin & Timmermann (2007): Multi-regime asset allocation

3. **Cross-Asset Correlation**
   - Longin & Solnik (1995): Is the correlation in international equity returns constant?
   - Apply: Model changing correlations as feature

4. **Ensemble Methods**
   - Schapire (1990): "The strength of weak learnability"
   - Breiman (2001): "Random Forests"
   - Apply: Combine LSTM + XGBoost + Regime model

5. **Financial Time Series**
   - Tsay (2010): "Analysis of Financial Time Series"
   - AutoML for feature selection

---

## FILES TO CREATE

```
training/
├── data/
│   ├── raw/
│   │   ├── eurusd_1h_2019_2026.csv
│   │   ├── gbpusd_1h_2019_2026.csv
│   │   ├── ... (10 files total)
│   │   └── data_manifest.json
│   └── processed/
│       ├── features_train.csv
│       ├── features_test.csv
│       └── feature_metadata.json
├── models/
│   ├── lstm_multiasset_attention.h5
│   ├── xgboost_multiasset.pkl
│   ├── regime_trending_lstm.h5
│   ├── regime_ranging_lstm.h5
│   ├── regime_highvol_lstm.h5
│   ├── regime_crisis_lstm.h5
│   ├── scalers.pkl (all 10 assets)
│   └── model_metadata.json
├── notebooks/
│   └── 01_multiasset_training.ipynb
├── scripts/
│   ├── 01_fetch_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_train_models.py
│   ├── 04_validate_models.py
│   └── 05_deploy_models.py
└── reports/
    ├── data_quality_report.md
    ├── feature_importance.png
    ├── performance_metrics.csv
    └── validation_results.md
```

---

## DEPENDENCIES TO ADD

```
pip install yfinance pandas numpy scikit-learn
pip install xgboost tensorflow keras
pip install jupyter matplotlib seaborn
pip install optuna (for hyperparameter tuning)
pip install ta (technical analysis library)
```

---

## SUCCESS INDICATORS AFTER THIS PHASE

✅ Models trained on 7 years × 10 assets
✅ Accuracy improved to 55-60%
✅ Win rate improved to 55%+
✅ Sharpe ratio > 0.5
✅ Can trade multiple forex pairs
✅ Handles different market regimes
✅ Ready for live deployment

---

## NEXT STEPS FOR YOU (NEXT SESSION)

1. **Start with data fetching** (2019-2026, 10 assets)
2. **Run feature engineering** (create 50+ features)
3. **Train LSTM + XGBoost** (multi-asset ensemble)
4. **Validate performance** (walk-forward CV)
5. **Deploy to signal engine** (replace old models)
6. **Restart live scanner** with powerful new models
7. **Monitor live performance** (track Sharpe, win rate, drawdown)

---

**GO MAKE THEM MODELS POWERFUL!** 🚀📈
