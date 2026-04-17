# 📊 TradingAI Analyst — Setup Guide

## What This App Does
- Upload any candlestick chart screenshot
- AI identifies trend, chart patterns, SMC levels, Fibonacci, FVGs, Order Blocks
- Gives you a clear BUY / SELL / WAIT signal
- Provides Entry, Stop Loss, TP1, TP2 levels
- Auto-annotates your chart with key levels drawn on it

---

## Step 1 — Install Python
If you don't have Python, download it from:
👉 https://www.python.org/downloads/

During install, **check "Add Python to PATH"**.

---

## Step 2 — Install Dependencies
Open Terminal (Mac/Linux) or Command Prompt (Windows), navigate to this folder, then run:

```bash
pip install -r requirements.txt
```

---

## Step 3 — Get a Free API Key
1. Go to 👉 https://console.anthropic.com
2. Sign up (free account)
3. Go to **API Keys** → **Create Key**
4. Copy the key (starts with `sk-ant-...`)

> Free tier gives you enough credits to analyse hundreds of charts.

---

## Step 4 — Run the App
In your terminal/command prompt, in this folder, run:

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

---

## Step 5 — Use the App
1. Paste your API key in the sidebar
2. Select the market (Gold, BTC, EUR/USD, etc.)
3. Select the timeframe of your chart
4. Upload your chart screenshot (PNG, JPG)
5. Click **Analyse This Chart**
6. Read the full analysis — Entry, SL, TP1, TP2
7. Download the annotated chart

---

## Deploy Online (Free) — Share With Friends
To share the app with your friends without them needing to install anything:

1. Create a free account at 👉 https://streamlit.io/cloud
2. Push this folder to a GitHub repo
3. Connect GitHub to Streamlit Cloud
4. Deploy — you get a free public URL like `https://yourapp.streamlit.app`
5. Everyone can use it from their phone/browser!

---

## Tips for Best Results
- Use **H1 or H4 charts** for most reliable signals
- Make sure chart is **not too zoomed in** — show at least 50–100 candles
- Include indicator panels (RSI/MACD) in your screenshot if possible
- Add context notes like "DXY bearish today" for better analysis
- Always check the **Confidence Score** — trade only 7+/10 setups

---

## Strategies Baked Into the AI
- **SMC**: Order Blocks, FVG, BOS, CHoCH, Liquidity
- **Fibonacci**: 23.6%, 38.2%, 50%, 61.8%, 78.6% retracement + extensions
- **SNR**: Support & Resistance flip zones
- **Chart Patterns**: 23 patterns (flags, triangles, H&S, double tops, diamonds, etc.)
- **RSI**: Divergence detection, overbought/oversold
- **MACD**: Crossovers, divergence, zero-line
- **ICT**: OTE zones, killzones, power of 3
- **Risk Management**: 1:2 minimum R:R enforcement

---

## Disclaimer
This tool is for **educational purposes only**. It does not constitute financial advice.
Trading involves risk of loss. Always do your own research.
