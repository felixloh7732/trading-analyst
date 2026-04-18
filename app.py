"""
TradingAI Analyst - Chart Analysis App
Built with Streamlit + Claude Vision API
"""

import streamlit as st
import anthropic
from google import genai as google_genai
from google.genai import types as google_types
from PIL import Image, ImageDraw, ImageFont
import base64
import io
import json
import re
import os

# ── localStorage persistence (browser-side) ──────────────────
try:
    from streamlit_local_storage import LocalStorage as _LocalStorageClass
    _ls = _LocalStorageClass()
    _LS_AVAILABLE = True
except Exception:
    _ls = None
    _LS_AVAILABLE = False

# ============================================================
# FONT SETUP — download if system fonts not available
# ============================================================
_FONT_BOLD_PATH    = "/tmp/trading_font_bold.ttf"
_FONT_REGULAR_PATH = "/tmp/trading_font_regular.ttf"

def _ensure_fonts():
    """Download fonts to /tmp if not already available. Called once at startup."""
    import requests as _req

    SYSTEM_BOLD = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    SYSTEM_REG = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]

    def _first_existing(paths):
        for p in paths:
            if os.path.exists(p):
                return p
        return None

    # Bold
    if not os.path.exists(_FONT_BOLD_PATH):
        src = _first_existing(SYSTEM_BOLD)
        if src:
            import shutil; shutil.copy(src, _FONT_BOLD_PATH)
        else:
            try:
                url = "https://github.com/liberationfonts/liberation-fonts/raw/main/src/LiberationSans-Bold.ttf"
                r = _req.get(url, timeout=15)
                r.raise_for_status()
                with open(_FONT_BOLD_PATH, "wb") as f: f.write(r.content)
            except Exception:
                pass

    # Regular
    if not os.path.exists(_FONT_REGULAR_PATH):
        src = _first_existing(SYSTEM_REG)
        if src:
            import shutil; shutil.copy(src, _FONT_REGULAR_PATH)
        else:
            try:
                url = "https://github.com/liberationfonts/liberation-fonts/raw/main/src/LiberationSans-Regular.ttf"
                r = _req.get(url, timeout=15)
                r.raise_for_status()
                with open(_FONT_REGULAR_PATH, "wb") as f: f.write(r.content)
            except Exception:
                pass

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a font at given size. Falls back gracefully."""
    path = _FONT_BOLD_PATH if bold else _FONT_REGULAR_PATH
    if os.path.exists(path):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    # Pillow 10+ built-in fallback with size
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

# Run font setup at import time (cached to /tmp)
try:
    _ensure_fonts()
except Exception:
    pass

# ============================================================
# COMPREHENSIVE TRADING KNOWLEDGE BASE (System Prompt)
# ============================================================

TRADING_SYSTEM_PROMPT = """
You are an elite professional trading analyst with 20+ years of experience across Forex, Crypto, Commodities, and Indices.
You combine multiple advanced trading methodologies to deliver high-probability analysis.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 CHART PATTERN LIBRARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## BULLISH CONTINUATION PATTERNS (看涨形态)
1. **Bull Flag (旗形)**: Strong impulse up → shallow parallel channel down → breakout above flag resistance
   - Entry: Break + close above flag resistance
   - Target: Flag pole height projected from breakout
   - SL: Below flag support

2. **Bull Pennant / Triangle Flag (三角旗)**: Strong impulse up → converging trendlines → breakout
   - Entry: Break + close above upper trendline
   - Target: Pole height from breakout point
   - SL: Below pennant low

3. **Cup & Handle (杯柄形态)**: U-shaped recovery (cup) + small downward drift (handle) → breakout
   - Entry: Break above cup rim / handle resistance
   - Target: Cup depth projected upward
   - SL: Below handle low

4. **Ascending Triangle (上升三角形)**: Flat resistance + rising lows → bullish breakout
   - Entry: Break + close above flat resistance
   - Target: Triangle height from breakout
   - SL: Below last higher low

5. **Symmetrical Triangle (对称三角形)**: Converging highs and lows → breakout direction = trend
   - Entry: Break above/below trendline with momentum
   - Target: Widest part of triangle projected from breakout
   - SL: Opposite side of triangle

6. **Measured Move Up (衡量看涨)**: Two equal upward legs with consolidation in between
   - Entry: Break above consolidation high
   - Target: Equal to first leg length
   - SL: Below consolidation low

7. **Ascending Scallop (上升贝壳)**: Series of U-shaped patterns trending upward
   - Entry: Each breakout above previous scallop high
   - Target: Pattern height from breakout
   - SL: Below scallop low

8. **Triple Bottom / Three Rising Valleys (上升三连谷)**: Three lows with each valley higher than last
   - Entry: Break above neckline
   - Target: Depth of valleys projected up
   - SL: Below third valley low

## BEARISH CONTINUATION PATTERNS (看跌形态)
9. **Bear Flag (旗形)**: Strong impulse down → shallow parallel channel up → breakdown
   - Entry: Break + close below flag support
   - Target: Flag pole depth projected down
   - SL: Above flag resistance

10. **Bear Pennant (三角旗)**: Strong impulse down → converging trendlines → breakdown
    - Entry: Break + close below lower trendline
    - Target: Pole depth from breakdown
    - SL: Above pennant high

11. **Inverted Cup & Handle (倒置杯柄形态)**: Inverted U-shape + small upward drift → breakdown
    - Entry: Break below inverted cup rim
    - Target: Cup depth projected downward
    - SL: Above handle high

12. **Descending Triangle (下降三角形)**: Flat support + falling highs → bearish breakdown
    - Entry: Break + close below flat support
    - Target: Triangle height from breakdown
    - SL: Above last lower high

13. **Measured Move Down (衡量下降)**: Two equal downward legs with consolidation
    - Entry: Break below consolidation low
    - Target: Equal to first leg length down
    - SL: Above consolidation high

14. **Descending Scallop (下降贝壳)**: Series of inverted U-patterns trending downward
    - Entry: Each breakdown below previous scallop low
    - SL: Above scallop high

15. **Triple Top / Three Falling Peaks (下降三连峰)**: Three highs with each peak lower than last
    - Entry: Break below neckline
    - Target: Depth of peaks projected down
    - SL: Above third peak high

## REVERSAL PATTERNS (反转形态)
16. **Double Bottom (双重底)**: W-shape — strong bullish reversal
    - Entry: Break + close above middle peak (neckline)
    - Target: Pattern depth projected from neckline
    - SL: Below either bottom (whichever is lower)

17. **Double Top (双重顶)**: M-shape — strong bearish reversal
    - Entry: Break + close below middle valley (neckline)
    - Target: Pattern height projected from neckline down
    - SL: Above either top (whichever is higher)

18. **Diamond Bottom (钻石底)**: Price broadens then narrows in diamond shape at lows — bullish reversal
    - Entry: Break above upper-right trendline
    - Target: Widest diamond height projected up
    - SL: Below diamond low

19. **Diamond Top (钻石顶)**: Price broadens then narrows in diamond shape at highs — bearish reversal
    - Entry: Break below lower-right trendline
    - Target: Widest diamond height projected down
    - SL: Above diamond high

20. **Rectangle Top (矩形顶)**: Price consolidates in flat range at highs then breaks down
    - Entry: Break + close below range support
    - Target: Rectangle height projected down
    - SL: Above rectangle resistance

21. **Rectangle Bottom (矩形底)**: Price consolidates in flat range at lows then breaks up
    - Entry: Break + close above range resistance
    - Target: Rectangle height projected up
    - SL: Below rectangle support

22. **Head & Shoulders Top (头肩顶)**: Left shoulder + higher head + lower right shoulder → bearish reversal
    - Entry: Break below neckline connecting the two troughs
    - Target: Head height from neckline projected down
    - SL: Above right shoulder high

23. **Head & Shoulders Bottom / Inverse H&S (头肩底)**: Left shoulder + lower head + higher right shoulder → bullish reversal
    - Entry: Break above neckline connecting the two peaks
    - Target: Head depth from neckline projected up
    - SL: Below right shoulder low

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏦 SMART MONEY CONCEPTS (SMC)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Break of Structure (BOS)**
- Bullish BOS: Price closes ABOVE a previous swing high (confirms uptrend continuation).
- Bearish BOS: Price closes BELOW a previous swing low (confirms downtrend continuation).
- Rule: Body close required. Wicks alone do not count.

**Change of Character (CHoCH)**
- First BOS against the current trend = potential reversal signal.
- Bullish CHoCH: In a downtrend, price closes above previous swing high for the FIRST TIME.
- Bearish CHoCH: In an uptrend, price closes below previous swing low for the FIRST TIME.

**Liquidity**
- Equal Highs/Lows: Magnets for price — institutions trigger stops there before reversing.
- Buy-side Liquidity (BSL): Above equal highs — targeted in bullish moves.
- Sell-side Liquidity (SSL): Below equal lows — targeted in bearish moves.
- Inducement: False breakout to sweep liquidity BEFORE the real move.

**Premium / Discount Zones**
- Draw Fibonacci from the most recent major swing low to high.
- 50% level = Equilibrium.
- Above 50% = Premium zone (look for sells).
- Below 50% = Discount zone (look for buys).
- Best buys: 61.8%–78.6% discount zone.
- Best sells: 61.8%–78.6% premium zone.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 FIBONACCI ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Retracement Levels (draw from swing low to swing high for bullish, reverse for bearish)**
- 0.0% - Start of move
- 23.6% - Shallow retracement (strong trend)
- 38.2% - Moderate retracement
- 50.0% - Psychological midpoint (key level)
- 61.8% - Golden ratio (MOST IMPORTANT - high probability reversal zone)
- 70.5% - Optimal Trade Entry (OTE) zone start (SMC)
- 78.6% - Deep retracement (still valid if trend is strong)
- 88.6% - Very deep (borderline)
- 100% - Full retracement (trend may be reversing)

**OTE (Optimal Trade Entry) Zone**: 61.8% to 78.6% — highest probability entry zone in SMC.

**Extension Levels (profit targets)**
- 127.2% - TP1 (conservative)
- 141.4% - TP2 (moderate)
- 161.8% - TP3 (golden extension — main target)
- 200.0% - TP4 (extended run)
- 261.8% - TP5 (major move)

**Multi-Timeframe Fibonacci**
- Levels that align across 2+ timeframes = high-confluence zones.
- These "cluster zones" have 40% higher accuracy than single-TF levels.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 SUPPORT & RESISTANCE (SNR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- **Previous Highs/Lows**: Strong levels where price reversed before.
- **Round Numbers**: 1.1000, 2000, 50000 — psychological magnets.
- **Flip Zones**: Old resistance that becomes support after break (and vice versa).
- **Volume Nodes**: High volume at a price = strong acceptance zone.
- **Confluence Rule**: The more times a level has been tested (2–3 times = stronger, 4+ = weaker/ready to break).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 TECHNICAL INDICATORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**RSI (14-period)**
- >70 = Overbought → potential sell signal.
- <30 = Oversold → potential buy signal.
- Regular Bearish Divergence: Price makes higher high, RSI makes lower high → sell signal.
- Regular Bullish Divergence: Price makes lower low, RSI makes higher low → buy signal.
- Hidden Bearish Divergence: Price makes lower high, RSI makes higher high → continuation down.
- Hidden Bullish Divergence: Price makes higher low, RSI makes lower low → continuation up.
- RSI 50 line: Above = bullish momentum. Below = bearish momentum.

**MACD (12, 26, 9)**
- MACD line crosses ABOVE signal line = bullish momentum building.
- MACD line crosses BELOW signal line = bearish momentum building.
- Histogram bars growing = momentum accelerating.
- Histogram bars shrinking = momentum slowing (potential reversal).
- Divergence from price (same rules as RSI divergence).
- Zero line: MACD above zero = overall bullish. Below zero = overall bearish.

**Moving Averages**
- EMA 20: Short-term trend (price above = bullish short-term).
- EMA 50: Medium-term trend.
- EMA 200: Long-term trend bias (price above = bull market).
- Golden Cross: EMA50 crosses above EMA200 = strong bullish signal.
- Death Cross: EMA50 crosses below EMA200 = strong bearish signal.
- Price bouncing off EMA = dynamic support/resistance opportunity.

**Bollinger Bands (20, 2)**
- Price touching upper band = overbought in ranging market.
- Price touching lower band = oversold in ranging market.
- Band squeeze (narrowing) = big move coming soon.
- Band expansion = trend acceleration.
- "Walking the band" = strong trend.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ ICT CONCEPTS (Inner Circle Trader)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Killzones (High-probability trading times)**
- Asian Session: 00:00–03:00 UTC (range building)
- London Open: 07:00–10:00 UTC (major moves begin)
- New York Open: 13:00–16:00 UTC (highest volatility)
- London Close: 16:00–18:00 UTC

**Market Structure Shift (MSS)**: Similar to CHoCH — first opposite-direction BOS.
**Propulsion Block**: Strong engulfing candle that causes a structural break.
**Power of 3 (PO3)**: Accumulation → Manipulation (fake move) → Distribution (real move).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 CONFLUENCE SCORING SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Score each trade setup from 0–10 based on number of confluences:
- Trend alignment (HTF + LTF) = +2
- Pattern confirmation = +2
- Fibonacci level (61.8%/78.6% OTE) = +1.5
- S&R level (key swing high/low, flip zone) = +1.5
- RSI divergence/extreme = +1
- MACD confirmation = +0.5
- Liquidity sweep before entry = +1.5

Score 7+/10 = High confidence trade
Score 5-6/10 = Moderate confidence (trade with caution)
Score <5/10 = Skip or wait

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📏 RISK MANAGEMENT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Minimum R:R = 1:2 (risk $1 to make $2)
- Ideal R:R = 1:3 or better
- Never risk more than 1-2% of account per trade
- SL placement: Beyond last swing point or key S/R level
- TP1 at 1:1.5 R:R (move SL to breakeven after hit)
- TP2 at 1:3 R:R
- TP3 at 1:5+ R:R (if strong trend)
- Partial TP strategy: Close 50% at TP1, move SL to breakeven, let 50% run to TP2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 WYCKOFF METHOD (威科夫分析法)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Wyckoff identifies 4 market phases using price + volume:

**Phase 1 — ACCUMULATION (积累/吸筹)**
- Smart money quietly buying at lows
- Signs: Selling climax (SC), Automatic Rally (AR), Secondary Test (ST), Spring (false breakdown below support)
- Price action: Range-bound after a downtrend, decreasing volume on drops
- Bias: Prepare for BUY after Spring + Sign of Strength (SOS)

**Phase 2 — MARKUP (上涨阶段)**
- Price trending up, institutions already loaded
- Signs: Higher highs and higher lows, strong BOS upward
- Bias: BUY on pullbacks to Last Point of Support (LPS)

**Phase 3 — DISTRIBUTION (派发/出货)**
- Smart money quietly selling at highs
- Signs: Buying climax (BC), Automatic Reaction (AR), Upthrust (UT) — false breakout above resistance
- Price action: Range-bound after an uptrend, decreasing volume on rallies
- Bias: Prepare for SELL after Upthrust After Distribution (UTAD) + Sign of Weakness (SOW)

**Phase 4 — MARKDOWN (下跌阶段)**
- Price trending down
- Signs: Lower highs and lower lows, strong BOS downward
- Bias: SELL on rallies to Last Point of Supply (LPSY)

**How to use in analysis:**
- Always identify which Wyckoff phase the chart is in
- Look for Spring/Upthrust as the highest-probability entry triggers
- Volume confirmation is key: rising volume on impulse, falling volume on correction

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💵 DXY CORRELATION GUIDE (美元指数关联)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The US Dollar Index (DXY) measures USD strength. It inversely correlates with most instruments:

**STRONG NEGATIVE correlation (DXY up = these go DOWN):**
- Gold (XAUUSD) — correlation: -0.85 to -0.95
- EUR/USD — correlation: -0.90 to -0.98
- GBP/USD — correlation: -0.75 to -0.88
- AUD/USD — correlation: -0.70 to -0.82
- Most commodities (Oil, Silver, Copper)

**STRONG POSITIVE correlation (DXY up = these go UP):**
- USD/JPY — correlation: +0.75 to +0.88
- USD/CAD — correlation: +0.65 to +0.78
- USD/CHF — correlation: +0.70 to +0.85

**Rules for DXY analysis:**
- If DXY is Bullish → Expect Gold/EUR/GBP pairs to face headwinds (bearish pressure)
- If DXY is Bearish → Expect Gold/EUR/GBP pairs to have tailwinds (bullish pressure)
- DXY at key HTF resistance + Gold at key support = HIGH probability Gold bounce
- Always mention DXY context when analysing Gold or major USD pairs
"""

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = io.BytesIO()
    # Ensure RGB mode
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def analyze_chart_with_ai(
    image: Image.Image,
    api_key: str,
    model: str,
    market_type: str,
    timeframe: str,
    context: str = ""
) -> str:
    """Send chart image to Gemini or Claude for analysis."""
    img_b64 = encode_image_to_base64(image)

    user_prompt = f"""
Analyze this {market_type} chart on the {timeframe} timeframe.
{f"Trader note: {context}" if context else ""}

Give a SHORT, combined analysis. Write EVERY section in BOTH English AND Chinese (Mandarin). Keep each section to 1-2 lines max — no long paragraphs.

**TREND 趋势:**
[EN] State BOTH:
  - Short-term trend (last 10-20 candles on THIS chart): Bullish/Bearish/Sideways
  - Long-term trend (overall structure across the full chart): Bullish/Bearish/Sideways
  Example: "Short-term: Bearish pullback. Long-term: Bullish uptrend."
[中文] 分别说明：
  - 短期趋势（最近10-20根K线）：看涨/看跌/横盘
  - 长期趋势（整体图表结构）：看涨/看跌/横盘
  例如：「短期：看跌回调。长期：看涨上升趋势。」

**PATTERN 形态:**
[EN] CRITICAL RULE — Check if the pattern is still ACTIVE or already COMPLETED/BROKEN:
  - If pattern is still forming (price inside it): name it and say "FORMING — still valid"
  - If pattern has already broken out (price clearly above/below it): say "COMPLETED — [direction] breakout at [position]. Now trading at [position relative to breakout]." Do NOT draw the pattern as if it's still active.
  - If no pattern: "No clear pattern — structure only."
[中文] 关键规则 — 判断形态是否仍然有效或已经突破完成：
  - 若形态仍在形成中（价格在其内部）：说明形态名称并注明「形成中 — 仍然有效」
  - 若已突破（价格明显超出形态）：说「已完成 — [方向]突破。现价位于[相对位置]。」不要画成仍在形成的形态。
  - 若无形态：「无明显形态，仅结构分析」

**KEY LEVELS 关键位:**
[EN] 2-3 most important S/R levels — position on chart (e.g. "Strong resistance at swing high (top)", "Support at recent low (lower quarter)").
[中文] 2-3个最重要的支撑/阻力位，标注在图表上的位置。

**SMC 智能资金:**
[EN] BOS/CHoCH, liquidity sweeps, supply/demand zones, equal highs/lows visible? 1-2 lines.
[中文] 结构突破/变化、流动性扫描、供需区域、平顶/平底？1-2行。

**WYCKOFF 威科夫:**
[EN] Which Wyckoff phase is this chart in? (Accumulation / Markup / Distribution / Markdown). Any Spring, Upthrust, SC, BC visible? 1-2 lines.
[中文] 此图处于哪个威科夫阶段？（积累/上涨/派发/下跌）。是否有弹簧位、上冲、卖出高潮、买入高潮？1-2行。

**CANDLESTICK PATTERNS 单K线形态:**
[EN] Identify any significant single or multi-candle patterns on the LAST 3-5 candles: Doji, Hammer, Shooting Star, Engulfing (Bullish/Bearish), Morning Star, Evening Star, Pin Bar, Marubozu, Harami, Tweezer Top/Bottom. If none significant: "No key candle pattern."
[中文] 识别最近3-5根K线的重要形态：十字星、锤子线、流星线、吞没（看涨/看跌）、晨星、暮星、钉线、大阳/大阴线、孕线、镊子顶/底。若无：「无明显K线形态」。

**DXY CORRELATION 美元指数关联:** (skip this section if instrument has no USD correlation)
[EN] Is DXY bullish or bearish based on chart context? How does this affect the current instrument? 1 line.
[中文] 根据图表背景，美元指数是强势还是弱势？这对当前交易品种有何影响？1行。

**TRADE SETUP 交易方案:**
- Signal 信号: BUY 🟢 / SELL 🔴 / WAIT ⏳
- Entry 入场: [EN price zone] / [中文价格区域]
- SL 止损: [EN level + reason] / [中文止损位说明]
- TP1 目标1: [EN level + R:R] / [中文目标位]
- TP2 目标2: [EN level + R:R] / [中文目标位]
- Confluences 汇合因素: [comma-separated list in English]
- Confidence 信心: X/10
- Warning 风险提示: [EN 1 line] / [中文一句话]

---
Now output the drawing instructions as a JSON block to annotate the chart.

════ MARKET STRUCTURE ANNOTATION RULES ════

PURPOSE: Show the most important S&R levels and any clear chart pattern forming.
Keep it CLEAN and MINIMAL — maximum 5 annotations total. No clutter.
Do NOT draw SL / TP / Entry points, trendlines, or Fibonacci levels.

MAXIMUM 5 ANNOTATIONS TOTAL — STRICT LIMIT.

PRIORITY ORDER (draw in this order, stop when you reach 5):
  1. Key S/R horizontal lines (most important — always include the 2 most critical levels)
  2. ONE supply OR demand zone (the most obvious one only)
  3. ONE BOS or CHoCH (the most recent structure break only — not historical ones)
  4. ONE liquidity zone (only if equal highs/lows are clearly visible)

ANNOTATION TYPES ALLOWED:

1. "horizontal_line" — KEY S/R LEVELS (most important, always draw these first)
   - Only mark price levels the market has clearly respected 2+ times
   - color: green=strong support, red=strong resistance, yellow=equal highs/lows

2. "zone_box" — SUPPLY or DEMAND zone (max 1 total)
   - Only if there is a clear strong impulse move from the zone
   - color: green=Demand/Support zone, red=Supply/Resistance zone, yellow=Liquidity pool

3. "structure_break" — BOS or CHoCH (max 1 total — most recent only)
   - color: "teal" for BOS, "orange" for CHoCH
   - direction: "bullish" or "bearish"

DO NOT USE: fibonacci, diagonal_line, pattern_triangle, pattern_flag — these clutter the chart.

CHART PATTERN DETECTION — identify in "pattern_name" field:
  Bullish: Bull Flag 旗形, Pennant 三角旗, Cup & Handle 杯柄形态, Ascending Triangle 上升三角形,
           Symmetrical Triangle 对称三角形, Double Bottom 双重底, Inv Head & Shoulders 头肩底,
           Diamond Bottom 钻石底, Rectangle Bottom 矩形底
  Bearish: Bear Flag 旗形, Pennant 三角旗, Inv Cup & Handle 倒置杯柄, Descending Triangle 下降三角形,
           Symmetrical Triangle 对称三角形, Double Top 双重顶, Head & Shoulders 头肩顶,
           Diamond Top 钻石顶, Rectangle Top 矩形顶
  If NO clear pattern: use "No Clear Pattern"

COLOUR CONVENTION:
  green=Support/Demand  |  red=Resistance/Supply  |  yellow=Liquidity/Equal levels
  teal=BOS  |  orange=CHoCH

LABEL RULES — SHORT only (under 18 characters):
  "Support 支撑" / "Resistance 阻力" / "Demand Zone 需求区" / "Supply Zone 供给区"
  "BOS ↑ 结构突破" / "CHoCH ↓ 结构变化" / "Liquidity 流动性" / "Equal Lows 平底" / "Equal Highs 平顶"

For y positions use: "top"(0.06), "upper_quarter"(0.20), "upper_third"(0.30), "middle"(0.50), "lower_third"(0.65), "lower_quarter"(0.78), "bottom"(0.93)

Example — bearish with double top pattern:
```json
{{
  "signal": "SELL",
  "confidence": 8,
  "pattern_name": "Double Top 双重顶",
  "annotations": [
    {{"type": "horizontal_line", "y_position": "upper_third", "color": "red", "label": "Resistance 阻力"}},
    {{"type": "horizontal_line", "y_position": "lower_third", "color": "green", "label": "Support 支撑"}},
    {{"type": "zone_box", "y_start": "upper_quarter", "y_end": "upper_third", "color": "red", "label": "Supply Zone 供给区"}},
    {{"type": "structure_break", "y_position": "middle", "color": "orange", "label": "CHoCH ↓ 结构变化", "direction": "bearish"}}
  ]
}}
```

Example — bullish with ascending triangle:
```json
{{
  "signal": "BUY",
  "confidence": 7,
  "pattern_name": "Ascending Triangle 上升三角形",
  "annotations": [
    {{"type": "horizontal_line", "y_position": "upper_third", "color": "red", "label": "Resistance 阻力"}},
    {{"type": "horizontal_line", "y_position": "lower_third", "color": "green", "label": "Support 支撑"}},
    {{"type": "zone_box", "y_start": "lower_quarter", "y_end": "lower_third", "color": "green", "label": "Demand Zone 需求区"}},
    {{"type": "structure_break", "y_position": "upper_third", "color": "teal", "label": "BOS ↑ 结构突破", "direction": "bullish"}}
  ]
}}
```
"""

    # ── Google Gemini (FREE) ──────────────────────────────
    if model.startswith("gemini"):
        client = google_genai.Client(api_key=api_key)
        full_prompt = TRADING_SYSTEM_PROMPT + "\n\n" + user_prompt

        # Convert image to bytes for the API
        img_buf = io.BytesIO()
        img_copy = image.copy()
        if img_copy.mode in ("RGBA", "P"):
            img_copy = img_copy.convert("RGB")
        img_copy.save(img_buf, format="JPEG", quality=90)
        img_bytes = img_buf.getvalue()

        response = client.models.generate_content(
            model=model,
            contents=[
                full_prompt,
                google_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            ],
        )
        return response.text

    # ── Anthropic Claude (Paid) ───────────────────────────
    else:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=TRADING_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                }
            ],
        )
        return response.content[0].text


def parse_json_from_analysis(analysis_text: str) -> dict:
    """Extract the JSON block from the AI analysis."""
    try:
        match = re.search(r"```json\s*(\{.*?\})\s*```", analysis_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except Exception:
        pass
    return {}


def _draw_dashed_line(draw, x1, y1, x2, y2, fill, width=2, dash=12, gap=6):
    """Draw a dashed line between two points."""
    import math
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    pos = 0
    drawing = True
    while pos < length:
        seg = min(pos + (dash if drawing else gap), length)
        if drawing:
            draw.line([(x1 + ux * pos, y1 + uy * pos),
                       (x1 + ux * seg, y1 + uy * seg)], fill=fill, width=width)
        pos = seg
        drawing = not drawing


def _label_box(draw, x, y, text, font, text_color, bg=(0, 0, 0, 190), padding=5):
    """Draw a label with dark background."""
    tw = len(text) * 8 + padding * 2
    th = 16
    draw.rectangle([x, y - th, x + tw, y + padding], fill=bg)
    draw.text((x + padding, y - th + 2), text, fill=text_color, font=font)


def annotate_chart(image: Image.Image, annotations: list, signal: str, meta: dict = {}) -> Image.Image:
    """Draw market structure annotations — BOS / CHoCH / Supply & Demand / S&R / Liquidity."""
    img = image.copy().convert("RGBA")

    # ── Scale up for sharp annotations ────────────────────
    MIN_W = 2000
    w_orig, h_orig = img.size
    if w_orig < MIN_W:
        scale = MIN_W / w_orig
        img   = img.resize((int(w_orig * scale), int(h_orig * scale)), Image.LANCZOS)

    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    # ── Font sizes ────────────────────────────────────────
    fs_b  = max(22, int(w / 75))    # bold labels
    fs_sm = max(19, int(w / 85))    # normal text
    fs_lg = max(27, int(w / 58))    # signal badge
    fs_xs = max(17, int(w / 100))   # small tags

    font_b  = _load_font(fs_b,  bold=True)
    font_sm = _load_font(fs_sm, bold=False)
    font_lg = _load_font(fs_lg, bold=True)
    font_xs = _load_font(fs_xs, bold=False)

    # ── Line thickness ─────────────────────────────────────
    LW_SR    = max(4, int(w / 380))   # S/R and key structural lines
    LW_ZONE  = max(2, int(w / 650))   # zone box border
    LW_TREND = max(3, int(w / 520))   # trendlines / diagonals
    LW_BOS   = max(4, int(w / 380))   # BOS / CHoCH marker lines
    DASH_LEN = max(18, int(w / 90))
    GAP_LEN  = max(9,  int(w / 180))

    # ── Colour palette ─────────────────────────────────────
    # Colour convention (shown in legend strip):
    #   green  = Demand zone / Support / Bullish structure
    #   red    = Supply zone / Resistance / Bearish structure
    #   blue   = (reserved / not used)
    #   yellow = Liquidity zone / Equal H&L / Liquidity pool
    #   teal   = BOS (Break of Structure — trend continuation)
    #   orange = CHoCH (Change of Character — potential reversal)
    #   white  = Trendlines / general structure
    C = {
        "red":    (255,  55,  55),
        "green":  (  0, 215,  80),
        "blue":   ( 50, 145, 255),
        "yellow": (255, 210,   0),
        "orange": (255, 135,   0),
        "purple": (190,  60, 255),
        "white":  (235, 235, 235),
        "teal":   (  0, 215, 195),
        "pink":   (255,  60, 165),
        "lime":   (115, 255,  40),
        "cyan":   (  0, 205, 255),
    }

    def col(name, alpha=255):
        r, g, b = C.get(name, C["white"])
        return (r, g, b, alpha)

    def solid(name):
        r, g, b = C.get(name, C["white"])
        return (r, g, b, 255)

    # ── Position map ───────────────────────────────────────
    POS = {
        "top":           int(h * 0.06),
        "upper_quarter": int(h * 0.20),
        "upper_third":   int(h * 0.30),
        "middle":        int(h * 0.50),
        "lower_third":   int(h * 0.65),
        "lower_quarter": int(h * 0.78),
        "bottom":        int(h * 0.93),
    }

    def yp(key):
        if isinstance(key, (int, float)):
            return int(h * float(key))
        return POS.get(str(key), int(h * 0.5))

    def xp(val):
        return int(w * float(val))

    # ── Chart content area: leave right 18% for labels ────
    CHART_R = int(w * 0.82)   # right edge of drawn elements
    LABEL_X = CHART_R + 8     # where label boxes begin

    # ── Right-edge label helper ────────────────────────────
    right_label_y_used = []

    def right_label(y, text, txt_col, border_col, bold=False):
        font    = font_b if bold else font_xs
        fsize   = fs_b  if bold else fs_xs
        label_h = fsize + 14

        # Collision avoidance
        adj_y = y
        for used_y in right_label_y_used:
            if abs(adj_y - used_y) < label_h + 4:
                adj_y = used_y + label_h + 5
        right_label_y_used.append(adj_y)

        try:
            bbox = font.getbbox(text)
            tw   = bbox[2] - bbox[0] + 20
        except Exception:
            tw = len(text) * 10 + 20

        half = label_h // 2
        rx   = LABEL_X
        lx2  = min(rx + tw, w - 4)
        # Pill background + border
        draw.rectangle([rx, adj_y - half, lx2, adj_y + half],
                       fill=(6, 8, 18, 240), outline=border_col, width=2)
        draw.text((rx + 8, adj_y - half + 5), text, fill=txt_col, font=font)
        # Tick from chart body to label
        draw.line([(CHART_R - 4, adj_y), (rx, adj_y)],
                  fill=border_col, width=max(1, LW_ZONE))

    # ── Enforce max 5 annotations — S/R first, then zone, then BOS/CHoCH ──
    PRIORITY = {
        "horizontal_line": 0,   # S/R levels — most important, always first
        "zone_box":        1,   # Demand/Supply / liquidity zones
        "structure_break": 2,   # BOS / CHoCH — one only
        "dashed_line":     3,
        # Everything below is filtered out — kept for backward compat only
        "diagonal_line":   9,
        "fibonacci":       9,
        "pattern_triangle":9,
        "pattern_flag":    9,
        "pattern_hs":      9,
        "pattern_double":  9,
        "entry_arrow":     9,
        "pattern_label":   9,
    }
    # Filter out noisy types entirely, sort, cap at 5
    _allowed = {"horizontal_line", "zone_box", "structure_break", "dashed_line"}
    sorted_anns = sorted(
        [a for a in annotations if a.get("type","") in _allowed],
        key=lambda a: PRIORITY.get(a.get("type", ""), 9)
    )
    # Enforce sub-caps: max 3 horizontal_lines, max 1 zone_box, max 1 structure_break
    _counts = {}
    _sub_caps = {"horizontal_line": 3, "zone_box": 1, "structure_break": 1}
    filtered_anns = []
    for a in sorted_anns:
        t = a.get("type","")
        _counts[t] = _counts.get(t, 0) + 1
        if _counts[t] <= _sub_caps.get(t, 99):
            filtered_anns.append(a)
    annotations = filtered_anns[:5]

    # ═══════════════════════════════════════════════════════
    # DRAW MARKET STRUCTURE ANNOTATIONS
    # ═══════════════════════════════════════════════════════
    for ann in annotations:
        atype = ann.get("type", "")
        cname = ann.get("color", "white")
        label = ann.get("label", "")

        # ── BOS / CHoCH structural break marker ───────────
        # Draw a solid line with a highlighted badge at the right
        if atype == "structure_break":
            y    = yp(ann.get("y_position", "middle"))
            # Extend line full chart width
            draw.line([(int(w*0.01), y), (CHART_R, y)],
                      fill=col(cname), width=LW_BOS)
            # Small diagonal arrow chevrons along the line to indicate direction
            direction = ann.get("direction", "bullish")
            chev_col  = col(cname, 210)
            chev_sz   = max(10, int(w / 140))
            for cx in range(int(w*0.15), CHART_R - 20, int(w * 0.12)):
                if direction == "bullish":
                    draw.line([(cx, y + chev_sz), (cx + chev_sz, y), (cx + chev_sz*2, y + chev_sz)],
                              fill=chev_col, width=LW_TREND)
                else:
                    draw.line([(cx, y - chev_sz), (cx + chev_sz, y), (cx + chev_sz*2, y - chev_sz)],
                              fill=chev_col, width=LW_TREND)
            if label:
                right_label(y, label, solid(cname), col(cname), bold=True)

        # ── Solid S/R horizontal line ─────────────────────
        elif atype == "horizontal_line":
            y = yp(ann.get("y_position", "middle"))
            draw.line([(int(w*0.01), y), (CHART_R, y)], fill=col(cname, 210), width=LW_SR)
            if label:
                right_label(y, label, solid(cname), col(cname))

        # ── Dashed level line ─────────────────────────────
        elif atype == "dashed_line":
            y = yp(ann.get("y_position", "middle"))
            _draw_dashed_line(draw, int(w*0.01), y, CHART_R, y,
                              fill=col(cname, 200), width=LW_SR,
                              dash=DASH_LEN, gap=GAP_LEN)
            if label:
                right_label(y, label, solid(cname), col(cname))

        # ── Zone box: Demand / Supply / Liquidity / S-R zone ──
        elif atype == "zone_box":
            y1 = yp(ann.get("y_start", "upper_third"))
            y2 = yp(ann.get("y_end",   "upper_quarter"))
            if y1 > y2:
                y1, y2 = y2, y1
            zone_h = y2 - y1
            # Faint fill — candles still fully visible (alpha 22)
            draw.rectangle([int(w*0.01), y1, CHART_R, y2],
                           fill=col(cname, 22), outline=col(cname, 170), width=LW_ZONE)
            # Subtle left-edge marker bar (3px thick, full zone height, more opaque)
            draw.rectangle([int(w*0.01), y1, int(w*0.01) + max(4, LW_ZONE*2), y2],
                           fill=col(cname, 200))
            if label:
                mid_y = (y1 + y2) // 2
                right_label(mid_y, label, solid(cname), col(cname, 200))

        # ── Diagonal trendline ────────────────────────────
        elif atype == "diagonal_line":
            x1 = xp(ann.get("x1", 0.05))
            x2 = min(xp(ann.get("x2", 0.80)), CHART_R)
            y1 = yp(ann.get("y1", "upper_third"))
            y2 = yp(ann.get("y2", "lower_third"))
            draw.line([(x1, y1), (x2, y2)], fill=col(cname, 195), width=LW_TREND)
            if label:
                # Label at ~60% along the line
                mx = int(x1 + (x2 - x1) * 0.6)
                my = int(y1 + (y2 - y1) * 0.6)
                right_label(my, label, solid(cname), col(cname, 200))

        # ── Fibonacci retracement ─────────────────────────
        elif atype == "fibonacci":
            y_high = yp(ann.get("swing_high_y", "upper_quarter"))
            y_low  = yp(ann.get("swing_low_y",  "lower_quarter"))
            rng    = y_low - y_high
            fibs = [
                (0.382, "38.2%", "cyan",   LW_TREND),
                (0.500, "50%",   "yellow", LW_TREND),
                (0.618, "61.8%", "orange", LW_SR),
                (0.786, "78.6%", "pink",   LW_TREND),
            ]
            for ratio, flabel, fcol, lw in fibs:
                fy = int(y_high + rng * ratio)
                _draw_dashed_line(draw, int(w*0.01), fy, CHART_R, fy,
                                  fill=col(fcol, 170), width=lw,
                                  dash=DASH_LEN//2, gap=GAP_LEN)
                right_label(fy, f"Fib {flabel}", solid(fcol), col(fcol, 200))

        # ── Triangle / wedge / pennant ────────────────────
        elif atype == "pattern_triangle":
            top_y    = yp(ann.get("top_y",    "upper_third"))
            bottom_y = yp(ann.get("bottom_y", "lower_third"))
            apex_y   = (top_y + bottom_y) // 2
            xs, xa   = int(w*0.05), CHART_R - int(w*0.02)
            draw.line([(xs, top_y),    (xa, apex_y)], fill=col(cname, 180), width=LW_TREND)
            draw.line([(xs, bottom_y), (xa, apex_y)], fill=col(cname, 180), width=LW_TREND)
            if label:
                right_label(apex_y, label, solid(cname), col(cname, 200))

        # ── Flag / channel ────────────────────────────────
        elif atype == "pattern_flag":
            top_y    = yp(ann.get("top_y",    "upper_third"))
            bottom_y = yp(ann.get("bottom_y", "middle"))
            is_bear  = "BEAR" in label.upper() or cname in ("red", "orange", "pink")
            tilt_dir = 1 if is_bear else -1
            tilt     = int(abs(bottom_y - top_y) * 0.18) * tilt_dir
            x1, x2   = int(w * 0.12), CHART_R - int(w*0.02)
            draw.line([(x1, top_y),    (x2, top_y    + tilt)], fill=col(cname, 180), width=LW_TREND)
            draw.line([(x1, bottom_y), (x2, bottom_y + tilt)], fill=col(cname, 180), width=LW_TREND)
            if label:
                mid_y = (top_y + bottom_y) // 2 + tilt // 2
                right_label(mid_y, label, solid(cname), col(cname, 200))

        # ── Head & Shoulders ──────────────────────────────
        elif atype == "pattern_hs":
            neck_y = yp(ann.get("neck_y", "lower_third"))
            head_y = yp(ann.get("head_y", "upper_quarter"))
            lsh_y  = yp(ann.get("lsh_y",  "upper_third"))
            rsh_y  = yp(ann.get("rsh_y",  "upper_third"))
            r = max(10, int(w / 120))
            for cx, cy in [(int(w*0.22), lsh_y), (int(w*0.50), head_y), (int(w*0.76), rsh_y)]:
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=col(cname, 200), width=LW_TREND)
                draw.line([(cx, cy+r), (cx, neck_y)], fill=col(cname, 100), width=LW_TREND)
            draw.line([(int(w*0.15), neck_y), (CHART_R, neck_y)],
                      fill=col("yellow", 200), width=LW_TREND)
            right_label(neck_y, "Neckline 颈线", solid("yellow"), col("yellow", 200))

        # ── Double top/bottom ─────────────────────────────
        elif atype == "pattern_double":
            peak_y = yp(ann.get("peak_y", "upper_quarter"))
            neck_y = yp(ann.get("neck_y", "lower_third"))
            r = max(12, int(w / 105))
            for cx in [int(w*0.28), int(w*0.62)]:
                draw.ellipse([cx-r, peak_y-r, cx+r, peak_y+r],
                             outline=col(cname, 200), width=LW_TREND)
            draw.line([(int(w*0.15), neck_y), (CHART_R, neck_y)],
                      fill=col("yellow", 200), width=LW_TREND)
            right_label(neck_y, "Neckline 颈线", solid("yellow"), col("yellow", 200))

    # ═══════════════════════════════════════════════════════
    # COLOUR LEGEND STRIP — compact, top-left
    # ═══════════════════════════════════════════════════════
    legend_items = [
        ("green",  "Support/Demand"),
        ("red",    "Resistance/Supply"),
        ("yellow", "Liquidity"),
        ("teal",   "BOS"),
        ("orange", "CHoCH"),
    ]
    leg_x, leg_y = 10, 10
    leg_sw = max(10, int(w / 140))
    leg_h  = fs_xs + 6
    for lc, lt in legend_items:
        r2, g2, b2 = C[lc]
        draw.rectangle([leg_x, leg_y, leg_x + leg_sw, leg_y + leg_h],
                       fill=(r2, g2, b2, 220))
        draw.text((leg_x + leg_sw + 4, leg_y + 2), lt,
                  fill=(r2, g2, b2, 230), font=font_xs)
        try:
            tw = font_xs.getbbox(lt)[2] - font_xs.getbbox(lt)[0]
        except Exception:
            tw = len(lt) * 9
        leg_x += leg_sw + tw + 16

    # ── Pattern name banner (below legend, prominent) ──────
    pattern_name = meta.get("pattern_name", "") if meta else ""
    if pattern_name and pattern_name.upper() not in ("", "NO CLEAR PATTERN", "NONE"):
        _pn_txt  = f"📐 {pattern_name}"
        _pn_bg   = (30, 30, 60, 200)
        _pn_col  = (251, 191, 36, 240)   # amber
        try:
            _pn_bbox = font_sm.getbbox(_pn_txt)
            _pn_w    = _pn_bbox[2] - _pn_bbox[0] + 20
        except Exception:
            _pn_w = len(_pn_txt) * 10 + 20
        _pn_h  = fs_sm + 10
        _pn_y  = leg_y + leg_h + 8
        draw.rectangle([10, _pn_y, 10 + _pn_w, _pn_y + _pn_h],
                       fill=_pn_bg, outline=_pn_col[:3] + (160,), width=1)
        draw.text((18, _pn_y + 4), _pn_txt, fill=_pn_col, font=font_sm)

    # ═══════════════════════════════════════════════════════
    # SIGNAL BADGE — top-right corner (compact)
    # ═══════════════════════════════════════════════════════
    if signal in ("BUY", "SELL", "WAIT"):
        sc   = (10, 185, 60, 245) if signal == "BUY" else ((195, 28, 28, 245) if signal == "SELL" else (185, 125, 0, 245))
        stxt = f"▲ {signal}" if signal == "BUY" else (f"▼ {signal}" if signal == "SELL" else f"⏳ {signal}")
        bw2  = max(110, int(w * 0.085))
        bh2  = fs_lg + 8
        draw.rectangle([w - bw2 - 10, 10, w - 10, 10 + bh2],
                       fill=sc, outline=(255, 255, 255, 190), width=2)
        draw.text((w - bw2 + 5, 14), stxt, fill=(255, 255, 255, 255), font=font_lg)

        # Confidence indicator below signal badge
        conf = meta.get("confidence", 0)
        if conf:
            conf_col = (20, 200, 60) if conf >= 7 else ((210, 160, 0) if conf >= 5 else (200, 40, 40))
            ctxt     = f"Conf {conf}/10"
            draw.rectangle([w - bw2 - 10, 10 + bh2 + 4, w - 10, 10 + bh2 + fs_xs + 14],
                           fill=(*conf_col, 210), outline=(255, 255, 255, 150), width=1)
            draw.text((w - bw2 + 5, 10 + bh2 + 6), ctxt,
                      fill=(255, 255, 255, 255), font=font_xs)

    # ── Watermark ─────────────────────────────────────────
    draw.text((10, h - fs_xs - 6), "TradingAI Analyst",
              fill=(200, 200, 200, 60), font=font_xs)

    # ── Composite and return ───────────────────────────────
    return Image.alpha_composite(img, overlay).convert("RGB")


def get_news_warning(symbol_label: str) -> list:
    """Return list of high-impact events in the next 2 hours relevant to the symbol.
    Uses cached calendar_events from session_state if available, else fetches silently."""
    from datetime import datetime, timezone, timedelta
    events = st.session_state.get("calendar_events", [])
    if not events:
        try:
            import requests as _req
            r = _req.get("https://nfs.faireconomy.media/ff_calendar_thisweek.json", timeout=5)
            if r.status_code == 200:
                events = [e for e in r.json() if e.get("impact") == "High"]
                st.session_state["calendar_events"] = events
        except Exception:
            return []

    # Extract currencies from symbol (e.g. "EUR/USD" → ["EUR","USD"], "Gold" → ["XAU","USD"])
    sym_up = symbol_label.upper()
    curs = []
    for pair in [sym_up, sym_up.replace(" ", "")]:
        if "/" in pair:
            curs += pair.split("/")[:2]
    if not curs:
        # Map common names
        _map = {"GOLD": ["XAU", "USD"], "SILVER": ["XAG", "USD"],
                "OIL": ["USD"], "BTC": ["USD"], "ETH": ["USD"]}
        for k, v in _map.items():
            if k in sym_up:
                curs = v
                break
    curs = [c[:3] for c in curs if c]

    now_utc = datetime.now(timezone.utc)
    window  = now_utc + timedelta(hours=2)
    warnings = []
    for ev in events:
        try:
            ev_dt = datetime.fromisoformat(ev.get("date","").replace("Z","+00:00"))
            if now_utc <= ev_dt <= window:
                ev_cur = ev.get("currency","").upper()
                if not curs or ev_cur in curs:
                    warnings.append({
                        "currency": ev_cur,
                        "title":    ev.get("title",""),
                        "date":     ev_dt.strftime("%H:%M UTC"),
                        "impact":   ev.get("impact",""),
                    })
        except Exception:
            continue
    return warnings


def render_news_warning_banner(warnings: list):
    """Show a big red warning banner if high-impact news is imminent."""
    if not warnings:
        return
    items_html = "".join([
        f"<li style='margin:4px 0'><b style='color:#fbbf24'>{w['currency']}</b> — "
        f"{w['title']} <span style='color:#94a3b8'>@ {w['date']}</span></li>"
        for w in warnings
    ])
    st.markdown(f"""
<div style='background:linear-gradient(135deg,#7f1d1d,#991b1b);border:3px solid #ef4444;
border-radius:12px;padding:16px 20px;margin:12px 0;animation:pulse 2s infinite'>
<h3 style='color:#fef2f2;margin:0 0 8px 0'>⚠️ HIGH IMPACT NEWS PENDING · 高影响力新闻即将发布</h3>
<ul style='color:#fca5a5;margin:0;padding-left:20px;font-size:14px'>{items_html}</ul>
<p style='color:#fca5a5;margin:10px 0 0 0;font-size:13px;font-weight:700'>
🚫 DO NOT OPEN NEW TRADES until news passes! · 新闻发布前后30分钟内不要开仓！</p>
</div>
""", unsafe_allow_html=True)


def generate_chart_image_from_df(df, symbol_label: str, tf_label: str) -> Image.Image:
    """Generate a matplotlib candlestick chart from a DataFrame. Returns PIL Image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    ema50 = df["Close"].ewm(span=50, adjust=False).mean()
    _df   = df.reset_index()
    n     = len(_df)
    W     = 0.4

    fig_c, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [4, 1]}, facecolor="#0f172a")
    ax1.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")

    for i, row in _df.iterrows():
        _o, _h, _l, _c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        color = "#10b981" if _c >= _o else "#ef4444"
        ax1.plot([i, i], [_l, _h], color=color, linewidth=0.8, zorder=1)
        ax1.add_patch(mpatches.FancyBboxPatch(
            (i - W, min(_o, _c)), 2 * W, max(abs(_c - _o), 1e-9),
            boxstyle="square,pad=0", linewidth=0, facecolor=color, zorder=2,
        ))

    ax1.plot(list(range(n)), ema20.values, color="#fbbf24", linewidth=1.2, label="EMA20", alpha=0.85)
    ax1.plot(list(range(n)), ema50.values, color="#818cf8", linewidth=1.2, label="EMA50", alpha=0.85)
    ax1.legend(loc="upper left", facecolor="#1e293b", labelcolor="#f1f5f9", fontsize=9)

    tick_step = max(1, n // 10)
    ticks     = list(range(0, n, tick_step))
    date_col  = _df.columns[0]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([str(_df.iloc[i][date_col])[:16] for i in ticks],
                        rotation=30, ha="right", color="#94a3b8", fontsize=7)
    ax1.set_xlim(-1, n)
    ax1.tick_params(colors="#94a3b8")
    ax1.yaxis.tick_right()
    ax1.yaxis.set_tick_params(labelcolor="#94a3b8")
    ax1.grid(color="#1e293b", linewidth=0.5)
    ax1.set_title(f"{symbol_label}  {tf_label}  ({n} candles)", color="#f1f5f9", fontsize=13, pad=8)

    for i, row in _df.iterrows():
        _o, _c = float(row["Open"]), float(row["Close"])
        _v = float(row.get("Volume", 0) or 0)
        ax2.bar(i, _v, color="#10b981" if _c >= _o else "#ef4444", alpha=0.5, width=0.8)
    ax2.tick_params(colors="#94a3b8", labelsize=7)
    ax2.yaxis.tick_right()
    ax2.set_xlim(-1, n)
    ax2.set_ylabel("Vol", color="#94a3b8", fontsize=8)
    ax2.grid(color="#1e293b", linewidth=0.3)

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig_c.savefig(buf, format="PNG", dpi=130, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig_c)
    buf.seek(0)
    return Image.open(buf).copy()


def pil_to_download_bytes(image: Image.Image) -> bytes:
    """Return lossless PNG bytes — pass these to st.image() to avoid Streamlit re-encoding."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=1)   # lossless, fast
    buf.seek(0)
    return buf.getvalue()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="TradingAI Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Restore persisted state from localStorage (runs once per session) ──
if _LS_AVAILABLE and not st.session_state.get("_ls_loaded"):
    try:
        saved = _ls.getItem("trading_analyst_prefs")
        if saved and isinstance(saved, dict):
            # API key
            if saved.get("api_key") and "saved_api_key" not in st.session_state:
                st.session_state["saved_api_key"] = saved["api_key"]
            # Settings
            for _k in ("model_choice", "market_type", "timeframe"):
                if saved.get(_k) and _k not in st.session_state:
                    st.session_state[f"saved_{_k}"] = saved[_k]
    except Exception:
        pass

    try:
        saved_chat = _ls.getItem("trading_analyst_chat")
        if saved_chat and isinstance(saved_chat, list) and "coach_messages" not in st.session_state:
            # Only restore text messages (no image data)
            clean = [m for m in saved_chat if isinstance(m.get("content"), str)]
            if clean:
                st.session_state["coach_messages"] = clean
    except Exception:
        pass

    st.session_state["_ls_loaded"] = True

# Dark trading terminal CSS
st.markdown("""
<style>
  /* ── Main background ── */
  [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #f0f4ff 0%, #faf0ff 50%, #f0fff8 100%);
  }
  [data-testid="stMain"] { background: transparent; }

  /* ── Main text (exclude sidebar) ── */
  .stApp p, .stApp li { color: #1a1a2e !important; }

  /* ── Title ── */
  h1 {
    background: linear-gradient(90deg, #7c3aed, #2563eb, #059669);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 42px !important; font-weight: 900 !important;
  }
  h2 { color: #7c3aed !important; font-weight: 800 !important; }
  h3 { color: #2563eb !important; font-weight: 700 !important; }

  /* ── Analyse button ── */
  .stButton>button {
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 50%, #059669 100%);
    color: white !important; border: none; border-radius: 10px;
    font-weight: 900 !important; font-size: 18px !important; padding: 14px;
    width: 100%; transition: all 0.3s; box-shadow: 0 4px 15px rgba(124,58,237,0.4);
    letter-spacing: 0.5px;
  }
  .stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(124,58,237,0.6);
  }

  /* ── BUY / SELL / WAIT badges ── */
  .buy-badge {
    background: linear-gradient(135deg, #059669, #10b981);
    border: 3px solid #065f46; color: #ffffff;
    border-radius: 12px; padding: 16px 32px;
    font-size: 26px; font-weight: 900; display: inline-block;
    box-shadow: 0 4px 20px rgba(5,150,105,0.5); letter-spacing: 1px;
  }
  .sell-badge {
    background: linear-gradient(135deg, #dc2626, #ef4444);
    border: 3px solid #7f1d1d; color: #ffffff;
    border-radius: 12px; padding: 16px 32px;
    font-size: 26px; font-weight: 900; display: inline-block;
    box-shadow: 0 4px 20px rgba(220,38,38,0.5); letter-spacing: 1px;
  }
  .wait-badge {
    background: linear-gradient(135deg, #d97706, #f59e0b);
    border: 3px solid #78350f; color: #ffffff;
    border-radius: 12px; padding: 16px 32px;
    font-size: 26px; font-weight: 900; display: inline-block;
    box-shadow: 0 4px 20px rgba(217,119,6,0.5); letter-spacing: 1px;
  }

  /* ── Info / result box ── */
  .info-box {
    background: white; border: 2px solid #c4b5fd;
    border-radius: 12px; padding: 20px; margin: 8px 0;
    box-shadow: 0 2px 12px rgba(124,58,237,0.1);
  }

  /* ── Metric card ── */
  .metric-card {
    background: white; border: 2px solid #bfdbfe; border-radius: 10px;
    padding: 16px; margin: 6px 0; text-align: center;
    box-shadow: 0 2px 8px rgba(37,99,235,0.1);
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab"] {
    color: #6b7280 !important; font-weight: 600; font-size: 15px;
  }
  .stTabs [aria-selected="true"] {
    color: #7c3aed !important;
    border-bottom: 3px solid #7c3aed !important;
  }

  /* ── Expanders ── */
  .stExpander {
    border: 2px solid #c4b5fd !important;
    border-radius: 10px !important;
    background: white !important;
  }
  .stExpander summary { color: #7c3aed !important; font-weight: 700 !important; }

  /* ── Upload area ── */
  [data-testid="stFileUploader"] {
    border: 2px dashed #7c3aed !important;
    border-radius: 12px !important; background: white !important;
  }

  /* ── Divider ── */
  hr { border: none; border-top: 2px solid #e9d5ff !important; }

  /* ── Success / warning / error ── */
  .stSuccess { background: #d1fae5 !important; border-left: 4px solid #059669 !important; }
  .stWarning { background: #fef3c7 !important; border-left: 4px solid #d97706 !important; }
  .stError   { background: #fee2e2 !important; border-left: 4px solid #dc2626 !important; }
  .stInfo    { background: #ede9fe !important; border-left: 4px solid #7c3aed !important; }

  /* ── Caption ── */
  .stApp .stCaption, .stApp caption { color: #6b21a8 !important; font-weight: 500 !important; }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #7c3aed !important; }

  /* ══════════════════════════════════════════
     SIDEBAR — placed LAST so it always wins
  ══════════════════════════════════════════ */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0533 0%, #0d1b4b 50%, #002b1a 100%) !important;
    border-right: 3px solid #a855f7 !important;
  }
  /* Every single text node inside sidebar → white */
  section[data-testid="stSidebar"] *:not(button) {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
  }
  /* Headings → bright yellow */
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] h4 {
    color: #ffd700 !important;
    -webkit-text-fill-color: #ffd700 !important;
    font-weight: 800 !important;
  }
  /* Widget labels → bright yellow */
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
  section[data-testid="stSidebar"] [data-baseweb="form-control-label"],
  section[data-testid="stSidebar"] [data-baseweb="form-control-label"] * {
    color: #ffd700 !important;
    -webkit-text-fill-color: #ffd700 !important;
    font-weight: 700 !important;
    font-size: 14px !important;
  }
  /* Dropdown fields */
  section[data-testid="stSidebar"] [data-baseweb="select"],
  section[data-testid="stSidebar"] [data-baseweb="select"] *,
  section[data-testid="stSidebar"] [data-baseweb="base-input"],
  section[data-testid="stSidebar"] [data-baseweb="base-input"] * {
    background-color: #2a1a4a !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border-color: #a855f7 !important;
  }
  /* Text input & textarea */
  section[data-testid="stSidebar"] input,
  section[data-testid="stSidebar"] textarea {
    background-color: #2a1a4a !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 2px solid #a855f7 !important;
  }
  /* Placeholder text */
  section[data-testid="stSidebar"] input::placeholder,
  section[data-testid="stSidebar"] textarea::placeholder {
    color: #c4b5fd !important;
    -webkit-text-fill-color: #c4b5fd !important;
  }
  /* Divider */
  section[data-testid="stSidebar"] hr {
    border-color: #a855f7 !important;
    border-top: 1px solid #a855f7 !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 📊")
with col_title:
    st.title("TradingAI Analyst")
    st.caption("Professional chart analysis powered by AI · Forex · Crypto · Gold · Indices")

st.divider()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("### 🆓 FREE Option — Google Gemini")
    st.caption("Get a free key at aistudio.google.com → 1,500 analyses/day free!")

    st.markdown("### 💳 Paid Option — Claude (Anthropic)")
    st.caption("Better accuracy · get key at console.anthropic.com")

    # ── Auto-load key from Streamlit Secrets (cloud deployment) ──
    _secret_key = ""
    try:
        _secret_key = st.secrets.get("ANTHROPIC_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    # ── Resolve saved API key from localStorage ──
    _saved_api_key = st.session_state.get("saved_api_key", "")

    if _secret_key:
        st.success("✅ API Key loaded automatically (cloud mode)")
        api_key = _secret_key
        st.text_input(
            "🔑 API Key",
            value="••••••••••••••••••••",
            disabled=True,
            help="Key is pre-configured by the app owner",
        )
    else:
        api_key = st.text_input(
            "🔑 Paste Your API Key Here",
            type="password",
            value=_saved_api_key,
            placeholder="Gemini: AIza...   or   Claude: sk-ant-...",
            help="Gemini key from aistudio.google.com (FREE) or Claude key from console.anthropic.com (paid)",
            key="api_key_input",
        )

    # ── Remember settings checkbox (defined here, save logic runs after all widgets) ──
    if _LS_AVAILABLE and not _secret_key:
        _remember = st.checkbox(
            "🔒 Remember my API key & settings",
            value=bool(_saved_api_key),
            help="Saves your API key and preferences in this browser only (localStorage). Never sent anywhere.",
            key="remember_settings",
        )
    else:
        _remember = False

    # ── AI Model selector ─────────────────────────────────────
    _model_options = [
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "claude-opus-4-5",
        "claude-sonnet-4-5",
    ]
    _saved_model = st.session_state.get("saved_model_choice", "gemini-2.0-flash")
    _model_idx = _model_options.index(_saved_model) if _saved_model in _model_options else 0

    model_choice = st.selectbox(
        "🤖 AI Model",
        _model_options,
        index=_model_idx,
        help="✅ Gemini models = FREE  |  Claude models = paid",
        key="model_select",
    )

    st.divider()

    _market_options = [
        "Forex (EUR/USD, GBP/USD, etc.)",
        "Gold (XAUUSD)",
        "Silver (XAGUSD)",
        "BTC/USD (Bitcoin)",
        "ETH/USD (Ethereum)",
        "Other Crypto",
        "US Stocks",
        "Index (S&P500, Nasdaq, etc.)",
        "Oil (WTI/Brent)",
    ]
    _saved_market = st.session_state.get("saved_market_type", "Forex (EUR/USD, GBP/USD, etc.)")
    _market_idx = _market_options.index(_saved_market) if _saved_market in _market_options else 0

    market_type = st.selectbox(
        "📈 Market / Instrument",
        _market_options,
        index=_market_idx,
        key="market_select",
    )

    _tf_options = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
    _saved_tf = st.session_state.get("saved_timeframe", "H1")
    _tf_idx = _tf_options.index(_saved_tf) if _saved_tf in _tf_options else 4

    timeframe = st.selectbox(
        "⏱️ Timeframe",
        _tf_options,
        index=_tf_idx,
        key="timeframe_select",
    )

    additional_context = st.text_area(
        "💬 Notes / Context (optional)",
        placeholder="e.g. Waiting for H4 close, news tomorrow, DXY bearish...",
        height=90,
    )

    annotate_chart_flag = st.checkbox("🎨 Annotate chart automatically", value=True)

    st.divider()

    st.markdown("### 📊 Analysis Mode")
    mtf_mode = st.checkbox(
        "🔭 Multi-Timeframe Mode (MTF)",
        value=False,
        help="Top-Down Analysis: D1 → H1 → Entry chart. Only trade when all timeframes align!"
    )
    if mtf_mode:
        st.markdown("""
<div style='background:linear-gradient(135deg,#1e1b4b,#312e81);border-radius:8px;padding:10px;margin-top:6px;border:1px solid #6366f1'>
<p style='color:#a5b4fc;font-size:12px;margin:0;font-weight:600'>
📋 MTF Flow:<br>
① Upload D1 → Get HTF bias<br>
② Upload H1 → Confirm direction<br>
③ Upload Entry TF → Get precise signal<br><br>
<span style='color:#fbbf24'>⚡ Only BUY/SELL when all 3 align!</span>
</p>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📚 Strategies Active")
    strategies = [
        ("⚡", "BOS / CHoCH Market Structure"),
        ("💧", "Liquidity Sweeps & Pools"),
        ("🎯", "Supply & Demand Zones"),
        ("📐", "Fibonacci Retracement/Extension"),
        ("🎯", "Support & Resistance (SNR)"),
        ("📊", "RSI & MACD Signals"),
        ("🕯️", "23 Chart Patterns"),
        ("🔭", "Wyckoff Phase Analysis"),
    ]
    for icon, name in strategies:
        st.markdown(f"{icon} {name}")

    st.divider()

    # ── Twelve Data API Key (for Live Data tab) ───────────────
    st.markdown("### 📡 Live Data Key")
    st.caption("For real-time forex/gold/crypto in the Live Data tab.")
    _td_secret = ""
    try:
        _td_secret = st.secrets.get("TWELVE_DATA_API_KEY", "")
    except Exception:
        pass
    if _td_secret:
        st.success("✅ Twelve Data key loaded from secrets")
        twelve_data_key = _td_secret
        st.text_input("Twelve Data Key", value="••••••••••••••••••••",
                      disabled=True, key="td_key_display")
    else:
        twelve_data_key = st.text_input(
            "🔑 Twelve Data API Key",
            type="password",
            placeholder="Get free key at twelvedata.com",
            help="Free at twelvedata.com — 800 requests/day. Gives real-time forex, gold, crypto.",
            key="td_key_input",
        )
        if not twelve_data_key:
            st.caption("Without this, Live Data uses yfinance (15-min delayed).")

    st.divider()

    # ── Persist settings to localStorage (after all widgets are resolved) ──
    if _LS_AVAILABLE and not _secret_key:
        if _remember and api_key:
            try:
                _ls.setItem("trading_analyst_prefs", {
                    "api_key":      api_key,
                    "model_choice": model_choice,
                    "market_type":  market_type,
                    "timeframe":    timeframe,
                })
                st.session_state["saved_api_key"]    = api_key
                st.session_state["saved_model_choice"] = model_choice
                st.session_state["saved_market_type"] = market_type
                st.session_state["saved_timeframe"]  = timeframe
            except Exception:
                pass
        elif not _remember and _saved_api_key:
            # User un-ticked — wipe saved prefs
            try:
                _ls.deleteItem("trading_analyst_prefs")
                for _k in ("saved_api_key", "saved_model_choice", "saved_market_type", "saved_timeframe"):
                    st.session_state.pop(_k, None)
            except Exception:
                pass

        if st.button("🗑️ Clear all saved data", help="Removes your saved API key and chat history from this browser", key="clear_ls_btn"):
            try:
                _ls.deleteItem("trading_analyst_prefs")
                _ls.deleteItem("trading_analyst_chat")
                for _k in ("saved_api_key", "saved_model_choice", "saved_market_type", "saved_timeframe", "coach_messages"):
                    st.session_state.pop(_k, None)
                st.success("✅ Saved data cleared!")
                st.rerun()
            except Exception as _e:
                st.error(f"Could not clear: {_e}")

    st.caption("⚠️ For educational purposes only.\nAlways manage your own risk.")


# ── Main Layout ────────────────────────────────────────────

# ════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME MODE
# ════════════════════════════════════════════════════════════
if mtf_mode:
    st.markdown("## 🔭 Multi-Timeframe Top-Down Analysis")
    st.markdown("""
<div style='background:linear-gradient(135deg,#1e3a5f,#1a1a2e);border-radius:10px;padding:12px 18px;margin-bottom:18px;border:2px solid #3b82f6'>
<p style='color:#93c5fd;font-size:14px;margin:0;font-weight:700'>
📋 Strategy: <span style='color:#fbbf24'>Only enter a trade when D1 + H1 + Entry chart ALL point the same direction.</span><br>
<span style='color:#86efac;font-size:12px'>⬇️ 按顺序上传图表：先日线 → 再1小时 → 最后进场时间框架</span>
</p>
</div>
""", unsafe_allow_html=True)

    step1_tab, step2_tab, step3_tab = st.tabs([
        "① 📅 D1 — HTF Bias",
        "② ⏰ H1 — Confirmation",
        "③ ⚡ Entry — Signal",
    ])

    # ── Helper: render one MTF step ─────────────────────────
    def render_mtf_step(tab, step_key, step_label, step_tf, step_hint, htf_context=""):
        img_key  = f"mtf_{step_key}_image"
        text_key = f"mtf_{step_key}_analysis"
        ann_key  = f"mtf_{step_key}_annotated"

        with tab:
            st.markdown(f"### {step_label}")
            st.caption(step_hint)

            col_up, col_res = st.columns([1, 1], gap="large")

            with col_up:
                uploaded = st.file_uploader(
                    f"Upload {step_tf} chart",
                    type=["png", "jpg", "jpeg", "webp"],
                    key=f"uploader_{step_key}",
                    label_visibility="collapsed",
                )
                if uploaded:
                    img = Image.open(uploaded)
                    st.session_state[img_key] = img
                    st.image(img, caption=f"{market_type} · {step_tf}", use_container_width=True)

                if st.session_state.get(img_key) and api_key:
                    if st.button(f"🔍 Analyse {step_tf} Chart", key=f"btn_{step_key}", use_container_width=True):
                        ctx = htf_context + ("\n" + additional_context if additional_context else "")
                        with st.spinner(f"🤖 Analysing {step_tf} chart..."):
                            try:
                                result = analyze_chart_with_ai(
                                    st.session_state[img_key],
                                    api_key, model_choice,
                                    market_type, step_tf, ctx,
                                )
                                st.session_state[text_key]  = result
                                st.session_state[ann_key]   = None
                                st.success(f"✅ {step_tf} analysis done!")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                elif st.session_state.get(img_key) and not api_key:
                    st.warning("👈 Enter API key in sidebar first.")
                elif not st.session_state.get(img_key):
                    st.info(f"👆 Upload a {step_tf} chart screenshot above.")

            with col_res:
                if text_key in st.session_state:
                    result_text = st.session_state[text_key]
                    meta = parse_json_from_analysis(result_text)
                    sig  = meta.get("signal", "WAIT").upper()
                    conf = meta.get("confidence", 5)

                    badge_html = {
                        "BUY":  '<div class="buy-badge" style="font-size:18px;padding:10px 20px">🟢 BUY BIAS</div>',
                        "SELL": '<div class="sell-badge" style="font-size:18px;padding:10px 20px">🔴 SELL BIAS</div>',
                    }.get(sig, '<div class="wait-badge" style="font-size:18px;padding:10px 20px">⏳ NEUTRAL</div>')
                    st.markdown(badge_html, unsafe_allow_html=True)

                    bar_color = "#059669" if conf >= 7 else ("#d97706" if conf >= 5 else "#dc2626")
                    st.markdown(f"""
<div style="margin:8px 0 10px 0">
  <span style="color:#1a1a2e;font-size:13px;font-weight:600">Confidence: <b style="color:{bar_color}">{conf}/10</b></span>
  <div style="background:#e9d5ff;border-radius:6px;height:10px;margin-top:4px">
    <div style="background:{bar_color};width:{conf*10}%;height:10px;border-radius:6px"></div>
  </div>
</div>""", unsafe_allow_html=True)

                    # Annotate
                    if annotate_chart_flag and meta.get("annotations") and st.session_state.get(ann_key) is None:
                        with st.spinner("🎨 Drawing annotations..."):
                            ann_img = annotate_chart(
                                st.session_state[img_key],
                                meta["annotations"], sig, meta,
                            )
                            st.session_state[ann_key] = ann_img

                    if st.session_state.get(ann_key):
                        st.image(pil_to_download_bytes(st.session_state[ann_key]), use_container_width=True)
                        st.download_button(
                            f"⬇️ Download {step_tf} Chart",
                            data=pil_to_download_bytes(st.session_state[ann_key]),
                            file_name=f"annotated_{step_tf}.png",
                            mime="image/png",
                            use_container_width=True,
                            key=f"dl_{step_key}",
                        )

                    st.divider()
                    clean = re.sub(r"```json.*?```", "", result_text, flags=re.DOTALL).strip()
                    st.markdown(clean)

        # Return signal for confluence check
        if text_key in st.session_state:
            return parse_json_from_analysis(st.session_state[text_key]).get("signal", "WAIT").upper()
        return None

    # ── Build HTF context strings for lower TFs ─────────────
    d1_context_for_h1 = ""
    if "mtf_d1_analysis" in st.session_state:
        d1_meta = parse_json_from_analysis(st.session_state["mtf_d1_analysis"])
        d1_sig  = d1_meta.get("signal", "WAIT").upper()
        d1_context_for_h1 = (
            f"[HTF CONTEXT — D1 Chart already analysed]\n"
            f"D1 Signal: {d1_sig}. "
            f"D1 Pattern: {d1_meta.get('pattern_name','N/A')}.\n"
            f"Your job: Analyse the H1 chart and check if it CONFIRMS this D1 direction. "
            f"If H1 is pulling back but D1 is bullish, that's normal — note it as 'pullback within uptrend'. "
            f"Signal WAIT if H1 strongly contradicts D1."
        )

    h1_context_for_entry = ""
    if "mtf_d1_analysis" in st.session_state and "mtf_h1_analysis" in st.session_state:
        d1_meta = parse_json_from_analysis(st.session_state["mtf_d1_analysis"])
        h1_meta = parse_json_from_analysis(st.session_state["mtf_h1_analysis"])
        d1_sig  = d1_meta.get("signal", "WAIT").upper()
        h1_sig  = h1_meta.get("signal", "WAIT").upper()
        h1_context_for_entry = (
            f"[MULTI-TIMEFRAME CONTEXT — Higher TFs already analysed]\n"
            f"D1 Bias: {d1_sig} | H1 Bias: {h1_sig}\n"
            f"Your job: Find a PRECISE entry on this lower timeframe chart.\n"
            f"CRITICAL RULE: Only signal BUY if D1={d1_sig} AND H1={h1_sig} AND this chart also shows bullish entry.\n"
            f"Only signal SELL if D1={d1_sig} AND H1={h1_sig} AND this chart shows bearish entry.\n"
            f"If this chart contradicts the higher TF direction, signal WAIT — do not fight the trend.\n"
            f"Focus on: exact entry trigger, tight SL below/above nearest structure, TP at HTF levels."
        )

    # ── Render all 3 steps ───────────────────────────────────
    d1_sig    = render_mtf_step(step1_tab, "d1",    "📅 Step 1: Daily Chart (D1)", "D1",
                                "Get the big picture — which direction is the market going long-term?  先看日线，判断大方向。",
                                "")
    h1_sig    = render_mtf_step(step2_tab, "h1",    "⏰ Step 2: 1-Hour Chart (H1)", "H1",
                                "Confirm the D1 direction — is H1 aligned?  确认1小时方向与日线一致。",
                                d1_context_for_h1)
    entry_sig = render_mtf_step(step3_tab, "entry", "⚡ Step 3: Entry Chart", timeframe,
                                f"Find the precise entry on {timeframe} — only enter if D1 + H1 confirm!  找进场点，只在高时间框架一致时入场。",
                                h1_context_for_entry)

    # ── MTF Confluence Summary Banner ───────────────────────
    if d1_sig and h1_sig and entry_sig:
        st.divider()
        st.markdown("## 🎯 MTF Confluence Summary 多时间框架综合结论")

        all_sigs = [d1_sig, h1_sig, entry_sig]
        buy_count  = all_sigs.count("BUY")
        sell_count = all_sigs.count("SELL")

        if buy_count == 3:
            final_html = """
<div style='background:linear-gradient(135deg,#064e3b,#065f46);border:3px solid #10b981;border-radius:14px;padding:20px;text-align:center'>
<p style='color:#6ee7b7;font-size:28px;font-weight:900;margin:0'>✅ STRONG BUY 强烈看涨</p>
<p style='color:#a7f3d0;font-size:16px;margin:6px 0 0 0'>D1 ✅ BUY &nbsp;·&nbsp; H1 ✅ BUY &nbsp;·&nbsp; Entry ✅ BUY</p>
<p style='color:#6ee7b7;font-size:14px;margin:8px 0 0 0'>全部时间框架一致看涨 — 高概率做多机会！</p>
</div>"""
        elif sell_count == 3:
            final_html = """
<div style='background:linear-gradient(135deg,#7f1d1d,#991b1b);border:3px solid #ef4444;border-radius:14px;padding:20px;text-align:center'>
<p style='color:#fca5a5;font-size:28px;font-weight:900;margin:0'>✅ STRONG SELL 强烈看跌</p>
<p style='color:#fecaca;font-size:16px;margin:6px 0 0 0'>D1 ✅ SELL &nbsp;·&nbsp; H1 ✅ SELL &nbsp;·&nbsp; Entry ✅ SELL</p>
<p style='color:#fca5a5;font-size:14px;margin:8px 0 0 0'>全部时间框架一致看跌 — 高概率做空机会！</p>
</div>"""
        elif buy_count == 2:
            final_html = f"""
<div style='background:linear-gradient(135deg,#1e3a5f,#1a1a2e);border:3px solid #f59e0b;border-radius:14px;padding:20px;text-align:center'>
<p style='color:#fcd34d;font-size:24px;font-weight:900;margin:0'>⚠️ PARTIAL BUY 部分看涨</p>
<p style='color:#fde68a;font-size:15px;margin:6px 0 0 0'>D1: {d1_sig} &nbsp;·&nbsp; H1: {h1_sig} &nbsp;·&nbsp; Entry: {entry_sig}</p>
<p style='color:#93c5fd;font-size:13px;margin:8px 0 0 0'>2/3 时间框架看涨 — 谨慎考虑，建议等待全部一致再入场。</p>
</div>"""
        elif sell_count == 2:
            final_html = f"""
<div style='background:linear-gradient(135deg,#1e3a5f,#1a1a2e);border:3px solid #f59e0b;border-radius:14px;padding:20px;text-align:center'>
<p style='color:#fcd34d;font-size:24px;font-weight:900;margin:0'>⚠️ PARTIAL SELL 部分看跌</p>
<p style='color:#fde68a;font-size:15px;margin:6px 0 0 0'>D1: {d1_sig} &nbsp;·&nbsp; H1: {h1_sig} &nbsp;·&nbsp; Entry: {entry_sig}</p>
<p style='color:#93c5fd;font-size:13px;margin:8px 0 0 0'>2/3 时间框架看跌 — 谨慎考虑，建议等待全部一致再入场。</p>
</div>"""
        else:
            final_html = f"""
<div style='background:linear-gradient(135deg,#1a1a2e,#0f0f23);border:3px solid #6b7280;border-radius:14px;padding:20px;text-align:center'>
<p style='color:#d1d5db;font-size:24px;font-weight:900;margin:0'>⏳ NO TRADE 暂时观望</p>
<p style='color:#9ca3af;font-size:15px;margin:6px 0 0 0'>D1: {d1_sig} &nbsp;·&nbsp; H1: {h1_sig} &nbsp;·&nbsp; Entry: {entry_sig}</p>
<p style='color:#6b7280;font-size:13px;margin:8px 0 0 0'>时间框架方向不一致 — 等待明确方向，不要强行入场。</p>
</div>"""

        st.markdown(final_html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SINGLE CHART MODE (original)
# ════════════════════════════════════════════════════════════
else:
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("📸 Upload Chart Screenshot")

        uploaded_file = st.file_uploader(
            "Drop your chart here",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )

        if uploaded_file:
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption=f"{market_type} · {timeframe}", use_container_width=True)

            if not api_key:
                st.warning("👈 Enter your API key in the sidebar to analyse this chart.")
            else:
                run_btn = st.button("🚀 Analyse This Chart", use_container_width=True)

                if run_btn:
                    with st.spinner("🤖 AI is analysing your chart — this may take 15-30 seconds..."):
                        try:
                            result_text = analyze_chart_with_ai(
                                original_image,
                                api_key,
                                model_choice,
                                market_type,
                                timeframe,
                                additional_context,
                            )
                            st.session_state["analysis"]   = result_text
                            st.session_state["image"]      = original_image
                            st.session_state["annotated"]  = None
                            st.success("✅ Analysis complete!")
                        except anthropic.AuthenticationError:
                            st.error("❌ Invalid API key. Please check your key and try again.")
                        except anthropic.RateLimitError:
                            st.error("❌ Rate limit reached. Wait a moment and try again.")
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
        else:
            st.info("👆 Upload a chart screenshot to begin analysis.")

            with st.expander("📖 Quick Guide — How to Use"):
                st.markdown("""
1. **Enter your API key** in the sidebar (free at console.anthropic.com)
2. **Set the market and timeframe** to match your chart
3. **Upload a screenshot** of any candlestick chart
4. **Click Analyse** and wait ~20 seconds
5. **Read the full analysis** — trend, patterns, entry, SL, TP
6. **Download the annotated chart** if enabled
                """)

            with st.expander("🗺️ Pattern Cheat Sheet"):
                tab1, tab2, tab3 = st.tabs(["📈 Bullish", "📉 Bearish", "🔄 Reversal"])
                with tab1:
                    st.markdown("""
| Pattern | Signal | Key Rule |
|---------|--------|----------|
| Bull Flag | Continuation ↑ | Break above parallel channel |
| Bull Pennant | Continuation ↑ | Break above converging lines |
| Cup & Handle | Continuation ↑ | Break above cup rim |
| Ascending Triangle | Continuation ↑ | Break above flat resistance |
| Triple Bottom | Reversal ↑ | Break above neckline |
| Measured Move Up | Continuation ↑ | Equal leg projection |
                    """)
                with tab2:
                    st.markdown("""
| Pattern | Signal | Key Rule |
|---------|--------|----------|
| Bear Flag | Continuation ↓ | Break below parallel channel |
| Bear Pennant | Continuation ↓ | Break below converging lines |
| Inverted Cup | Continuation ↓ | Break below inverted rim |
| Descending Triangle | Continuation ↓ | Break below flat support |
| Triple Top | Reversal ↓ | Break below neckline |
| Measured Move Down | Continuation ↓ | Equal leg projection |
                    """)
                with tab3:
                    st.markdown("""
| Pattern | Signal | Key Rule |
|---------|--------|----------|
| Double Bottom (W) | Reversal ↑ | Break above middle peak |
| Double Top (M) | Reversal ↓ | Break below middle valley |
| H&S Top | Reversal ↓ | Break below neckline |
| Inverse H&S | Reversal ↑ | Break above neckline |
| Diamond Bottom | Reversal ↑ | Break above upper right |
| Diamond Top | Reversal ↓ | Break below lower right |
| Rectangle Bottom | Reversal ↑ | Break above range |
| Rectangle Top | Reversal ↓ | Break below range |
                    """)

    with right_col:
        st.subheader("🔍 Analysis Results")

        if "analysis" in st.session_state:
            text = st.session_state["analysis"]
            meta = parse_json_from_analysis(text)
            signal     = meta.get("signal", "WAIT").upper()
            confidence = meta.get("confidence", 5)
            pattern    = meta.get("pattern_name", "")

            # ── Signal badge ──────────────────────────────
            badge_html = {
                "BUY":  '<div class="buy-badge">🟢 BUY SIGNAL</div>',
                "SELL": '<div class="sell-badge">🔴 SELL SIGNAL</div>',
            }.get(signal, '<div class="wait-badge">⏳ WAIT — NO CLEAR SETUP</div>')
            st.markdown(badge_html, unsafe_allow_html=True)

            # ── News warning overlay ───────────────────────
            _news_warns = get_news_warning(market_type)
            render_news_warning_banner(_news_warns)

            if pattern:
                st.markdown(f"<p style='color:#7c3aed;font-weight:700;font-size:15px;margin:4px 0'>📐 Pattern: {pattern}</p>",
                            unsafe_allow_html=True)

            # ── Confidence bar ────────────────────────────
            bar_color = "#059669" if confidence >= 7 else ("#d97706" if confidence >= 5 else "#dc2626")
            st.markdown(f"""
<div style="margin:8px 0 12px 0">
  <span style="color:#1a1a2e;font-size:14px;font-weight:600">
    Confidence: <b style="color:{bar_color};font-size:16px">{confidence}/10</b>
  </span>
  <div style="background:#e9d5ff;border-radius:6px;height:12px;margin-top:5px;border:1px solid #c4b5fd">
    <div style="background:{bar_color};width:{confidence*10}%;height:12px;border-radius:6px;box-shadow:0 2px 6px rgba(0,0,0,0.2)"></div>
  </div>
</div>""", unsafe_allow_html=True)

            # ── Build + show annotated chart first ────────
            if annotate_chart_flag and meta.get("annotations"):
                if st.session_state.get("annotated") is None:
                    with st.spinner("🎨 Drawing annotations on chart..."):
                        ann_img = annotate_chart(
                            st.session_state["image"],
                            meta["annotations"],
                            signal,
                            meta,
                        )
                        st.session_state["annotated"] = ann_img

            if st.session_state.get("annotated") is not None:
                st.image(pil_to_download_bytes(st.session_state["annotated"]),
                         caption="Market Structure Analysis — BOS / CHoCH / Demand & Supply / S&R / Liquidity",
                         use_container_width=True)
                st.download_button(
                    "⬇️ Download Annotated Chart",
                    data=pil_to_download_bytes(st.session_state["annotated"]),
                    file_name="annotated_chart.png",
                    mime="image/png",
                    use_container_width=True,
                )

            st.divider()

            # ── Analysis text (json block stripped) ───────
            clean_text = re.sub(r"```json.*?```", "", text, flags=re.DOTALL).strip()
            st.markdown(clean_text)

            st.divider()

            # ── Follow-up Q&A on this analysis ────────────
            st.markdown("#### 💬 Ask a follow-up question · 追问")
            st.caption("Ask anything about this analysis — e.g. 'Where exactly is the liquidity swept?' or 'What if DXY pumps?'")

            # Init per-analysis chat (reset when new analysis runs)
            _analysis_id = hash(text[:100])
            if st.session_state.get("analysis_chat_id") != _analysis_id:
                st.session_state["analysis_chat_id"]  = _analysis_id
                st.session_state["analysis_chat"]     = []

            # Render conversation
            for _msg in st.session_state["analysis_chat"]:
                with st.chat_message(_msg["role"]):
                    st.markdown(_msg["content"])

            # Chat input
            if _followup_q := st.chat_input("Ask about this analysis...", key="analysis_followup_input"):
                if not api_key:
                    st.warning("👈 Enter your API key first.")
                else:
                    st.session_state["analysis_chat"].append({"role": "user", "content": _followup_q})
                    with st.chat_message("user"):
                        st.markdown(_followup_q)

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            try:
                                _sys = f"""You are an elite trading analyst. You have just completed a full analysis of a {market_type} chart on the {timeframe} timeframe.

Your analysis summary:
{clean_text[:2000]}

Now the trader is asking follow-up questions about your analysis. Answer specifically and precisely — refer back to exact price levels, patterns, and structures you identified. Be direct. Use the same language the trader uses (English or Chinese)."""

                                _chat_hist = st.session_state["analysis_chat"]
                                if model_choice.startswith("gemini"):
                                    _gc = google_genai.Client(api_key=api_key)
                                    _hist_text = "\n".join([
                                        f"{'Trader' if m['role']=='user' else 'Analyst'}: {m['content']}"
                                        for m in _chat_hist
                                    ])
                                    _fa = _gc.models.generate_content(
                                        model=model_choice,
                                        contents=[_sys + "\n\nConversation:\n" + _hist_text],
                                    )
                                    _ans = _fa.text
                                else:
                                    _ac = anthropic.Anthropic(api_key=api_key)
                                    _api_msgs = [{"role": m["role"], "content": m["content"]}
                                                 for m in _chat_hist if m["role"] in ("user","assistant")]
                                    # Include the chart image in the first message for Claude
                                    if len(_api_msgs) == 1 and st.session_state.get("image"):
                                        _img_b64 = encode_image_to_base64(st.session_state["image"])
                                        _api_msgs[0] = {
                                            "role": "user",
                                            "content": [
                                                {"type": "image", "source": {"type": "base64",
                                                 "media_type": "image/jpeg", "data": _img_b64}},
                                                {"type": "text", "text": _followup_q},
                                            ]
                                        }
                                    _fa = _ac.messages.create(
                                        model=model_choice, max_tokens=1200,
                                        system=_sys, messages=_api_msgs[-12:],
                                    )
                                    _ans = _fa.content[0].text

                                st.markdown(_ans)
                                st.session_state["analysis_chat"].append({"role": "assistant", "content": _ans})
                            except Exception as _e:
                                st.error(f"Error: {_e}")
                    st.rerun()

        else:
            st.markdown("""
<div class="info-box">
<p style="color:#5b21b6;text-align:center;margin-top:40px;font-size:17px;font-weight:700">
📊 Analysis results will appear here after you upload a chart and click Analyse.
</p>
<br>
<p style="color:#1e40af;text-align:center;font-size:14px;font-weight:600">
The AI will identify:<br><br>
⚡ BOS / CHoCH &nbsp;·&nbsp; 💧 Liquidity &nbsp;·&nbsp; 🎯 Supply & Demand<br>
📐 Fibonacci &nbsp;·&nbsp; 🕯️ Chart Patterns &nbsp;·&nbsp; 🔭 Wyckoff<br><br>
🎯 Entry &nbsp;·&nbsp; 🛑 Stop Loss &nbsp;·&nbsp; ✅ Take Profit 1 &nbsp;·&nbsp; 🚀 Take Profit 2
</p>
</div>
            """, unsafe_allow_html=True)


# ============================================================
# EXTRA TOOLS SECTION
# ============================================================
st.divider()
st.markdown("## 🛠️ Trading Tools 交易工具")

tool_tab1, tool_tab2, tool_tab3, tool_tab4, tool_tab5, tool_tab6, tool_tab7, tool_tab8 = st.tabs([
    "🧮 Position Size",
    "📰 News Calendar",
    "📡 Chart Scanner",
    "🤖 AI Coach",
    "📄 PDF Report",
    "💹 Currency Strength",
    "📈 Live Data",
    "🔭 MTF Panel",
])

# ════════════════════════════════════════════════════════════
# TOOL 1 — POSITION SIZE CALCULATOR
# ════════════════════════════════════════════════════════════
with tool_tab1:
    st.markdown("### 🧮 Position Size Calculator 仓位计算器")
    st.caption("Calculate exact lot size based on your account risk. 根据账户风险计算精确手数。")

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        ps_balance   = st.number_input("💰 Account Balance ($)", min_value=10.0, value=1000.0, step=100.0)
        ps_risk_pct  = st.number_input("⚠️ Risk per Trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    with pc2:
        ps_instrument = st.selectbox("📈 Instrument Type", [
            "Forex (Major Pairs)",
            "Gold (XAUUSD)",
            "Silver (XAGUSD)",
            "BTC/USD",
            "ETH/USD",
            "US30 / NAS100 (Index)",
            "Oil (WTI/Brent)",
        ])
        ps_account_currency = st.selectbox("💵 Account Currency", ["USD", "EUR", "GBP", "MYR", "SGD", "AUD"])
    with pc3:
        ps_entry = st.number_input("🎯 Entry Price", min_value=0.0001, value=1.1000, format="%.5f")
        ps_sl    = st.number_input("❌ Stop Loss Price", min_value=0.0001, value=1.0950, format="%.5f")

    if st.button("⚡ Calculate Position Size", use_container_width=True):
        sl_distance = abs(ps_entry - ps_sl)
        risk_amount = ps_balance * (ps_risk_pct / 100)

        if sl_distance == 0:
            st.error("Stop loss cannot equal entry price!")
        else:
            # Pip/point values per lot
            if ps_instrument == "Forex (Major Pairs)":
                pip_value_per_lot = 10.0   # $10 per pip per standard lot (USD account)
                sl_pips = sl_distance * 10000
                lot_size = risk_amount / (sl_pips * pip_value_per_lot)
                unit = "lots"
                pip_label = f"{sl_pips:.1f} pips"
            elif ps_instrument == "Gold (XAUUSD)":
                pip_value_per_lot = 100.0  # $1 per 0.01 move, lot=100oz
                sl_pips = sl_distance * 100
                lot_size = risk_amount / (sl_distance * 100)
                unit = "lots"
                pip_label = f"${sl_distance:.2f}/oz × 100oz"
            elif ps_instrument in ["BTC/USD", "ETH/USD"]:
                lot_size = risk_amount / sl_distance
                unit = "units"
                pip_label = f"${sl_distance:.2f} price move"
            elif ps_instrument in ["US30 / NAS100 (Index)"]:
                pip_value_per_lot = 1.0
                lot_size = risk_amount / sl_distance
                unit = "units"
                pip_label = f"{sl_distance:.1f} points"
            else:
                lot_size = risk_amount / sl_distance
                unit = "units"
                pip_label = f"${sl_distance:.4f} move"

            # MYR/SGD conversion approximation
            fx_note = ""
            if ps_account_currency == "MYR":
                risk_amount_local = risk_amount * 4.7
                fx_note = f" (≈ RM {risk_amount_local:.2f})"
            elif ps_account_currency == "SGD":
                risk_amount_local = risk_amount * 1.35
                fx_note = f" (≈ SGD {risk_amount_local:.2f})"

            rr2_tp = ps_entry + (ps_entry - ps_sl) * 2 if ps_entry > ps_sl else ps_entry - (ps_sl - ps_entry) * 2
            rr3_tp = ps_entry + (ps_entry - ps_sl) * 3 if ps_entry > ps_sl else ps_entry - (ps_sl - ps_entry) * 3

            st.markdown(f"""
<div style='background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);border-radius:12px;padding:20px;margin-top:10px;border:2px solid #3b82f6'>
<h3 style='color:#60a5fa;margin:0 0 12px 0'>📊 Calculation Results 计算结果</h3>
<table style='width:100%;color:white;font-size:15px'>
<tr><td style='padding:5px 0;color:#94a3b8'>Account Risk 风险金额</td>
    <td style='color:#fbbf24;font-weight:700;font-size:18px'>${risk_amount:.2f}{fx_note}</td></tr>
<tr><td style='padding:5px 0;color:#94a3b8'>SL Distance 止损距离</td>
    <td style='color:#f87171;font-weight:600'>{pip_label}</td></tr>
<tr><td style='padding:5px 0;color:#94a3b8'>Position Size 仓位大小</td>
    <td style='color:#34d399;font-weight:700;font-size:22px'>{lot_size:.3f} {unit}</td></tr>
<tr><td style='padding:5px 0;color:#94a3b8'>TP1 (1:2 R:R)</td>
    <td style='color:#86efac;font-weight:600'>{rr2_tp:.5f} → Profit: ${risk_amount*2:.2f}</td></tr>
<tr><td style='padding:5px 0;color:#94a3b8'>TP2 (1:3 R:R)</td>
    <td style='color:#6ee7b7;font-weight:600'>{rr3_tp:.5f} → Profit: ${risk_amount*3:.2f}</td></tr>
</table>
<p style='color:#64748b;font-size:11px;margin:10px 0 0 0'>⚠️ Approximate values. Always verify with your broker. 以上为参考值，请以券商为准。</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TOOL 2 — ECONOMIC CALENDAR
# ════════════════════════════════════════════════════════════
with tool_tab2:
    st.markdown("### 📰 Economic Calendar 经济日历")
    st.caption("Check upcoming high-impact news before trading. 交易前查看高影响力新闻。")

    cal_col1, cal_col2 = st.columns([1, 2])
    with cal_col1:
        if st.button("🔄 Load This Week's News", use_container_width=True):
            try:
                import requests
                response = requests.get(
                    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                    timeout=8
                )
                if response.status_code == 200:
                    events = response.json()
                    # Filter high impact only
                    high_impact = [e for e in events if e.get("impact") == "High"]
                    st.session_state["calendar_events"] = high_impact
                    st.success(f"✅ Loaded {len(high_impact)} high-impact events!")
                else:
                    st.error("Failed to load calendar. Try again.")
            except Exception as e:
                st.error(f"Network error: {str(e)}")
                st.info("💡 Try visiting https://www.forexfactory.com/calendar for news manually.")

    with cal_col2:
        filter_currency = st.multiselect(
            "Filter by Currency",
            ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "CNY", "XAU"],
            default=["USD", "EUR", "GBP"],
        )

    if "calendar_events" in st.session_state:
        events = st.session_state["calendar_events"]
        shown = 0
        for ev in events:
            if not filter_currency or ev.get("currency") in filter_currency:
                title    = ev.get("title", "")
                currency = ev.get("currency", "")
                date_str = ev.get("date", "")
                impact   = ev.get("impact", "")
                forecast = ev.get("forecast", "—")
                previous = ev.get("previous", "—")

                impact_color = "#ef4444" if impact == "High" else ("#f59e0b" if impact == "Medium" else "#6b7280")
                st.markdown(f"""
<div style='background:#1e293b;border-left:4px solid {impact_color};border-radius:6px;padding:10px 14px;margin:6px 0'>
<span style='color:{impact_color};font-weight:700;font-size:13px'>🔴 HIGH IMPACT</span>
<span style='color:#94a3b8;font-size:12px;margin-left:10px'>{date_str}</span><br>
<span style='color:white;font-weight:600;font-size:15px'>{currency} — {title}</span><br>
<span style='color:#64748b;font-size:12px'>Forecast: {forecast} &nbsp;|&nbsp; Previous: {previous}</span>
</div>
""", unsafe_allow_html=True)
                shown += 1
        if shown == 0:
            st.info("No high-impact events found for selected currencies this week.")
    else:
        st.markdown("""
<div style='background:#1e293b;border-radius:10px;padding:20px;text-align:center'>
<p style='color:#94a3b8;font-size:15px'>📅 Click "Load This Week\'s News" to see upcoming high-impact events.<br><br>
<span style='color:#fbbf24'>⚡ Rule: Avoid opening trades 30 mins before and after red news events!</span><br>
<span style='color:#86efac;font-size:13px'>规则：高影响力新闻发布前后30分钟内不要开仓！</span>
</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TOOL 3 — MULTI-CHART SCANNER
# ════════════════════════════════════════════════════════════
with tool_tab3:
    st.markdown("### 📡 Multi-Chart Scanner 多图扫描")

    scan_mode = st.radio(
        "Scan mode · 扫描模式",
        ["📤 Manual Upload", "🤖 Auto Scan (Live Data)"],
        horizontal=True,
        key="scan_mode_radio",
    )
    st.divider()

    if not api_key:
        st.warning("👈 Enter your API key in the sidebar first.")

    # ── AUTO SCAN MODE ────────────────────────────────────────
    elif scan_mode == "🤖 Auto Scan (Live Data)":
        st.caption("App auto-fetches live charts for your watchlist and ranks the best setups. 自动拉取实时图表，找出最佳机会。")

        # Watchlist picker
        AUTO_WATCHLIST = {
            "EUR/USD": ("EUR/USD", "EURUSD=X"),
            "GBP/USD": ("GBP/USD", "GBPUSD=X"),
            "USD/JPY": ("USD/JPY", "USDJPY=X"),
            "AUD/USD": ("AUD/USD", "AUDUSD=X"),
            "NZD/USD": ("NZD/USD", "NZDUSD=X"),
            "USD/CAD": ("USD/CAD", "USDCAD=X"),
            "USD/CHF": ("USD/CHF", "USDCHF=X"),
            "GBP/JPY": ("GBP/JPY", "GBPJPY=X"),
            "Gold (XAU/USD)": ("XAU/USD", "GC=F"),
            "Silver (XAG/USD)": ("XAG/USD", "SI=F"),
            "BTC/USD": ("BTC/USD", "BTC-USD"),
            "ETH/USD": ("ETH/USD", "ETH-USD"),
        }
        AUTO_TF_MAP = {
            "M15": ("15min", "15m", "5d"),
            "M30": ("30min", "30m", "10d"),
            "H1":  ("1h",    "1h",  "30d"),
            "H4":  ("4h",    "1h",  "60d"),
            "D1":  ("1day",  "1d",  "180d"),
        }

        as_col1, as_col2, as_col3 = st.columns([3, 1, 1])
        with as_col1:
            selected_pairs = st.multiselect(
                "📋 Select watchlist · 选择监控列表",
                list(AUTO_WATCHLIST.keys()),
                default=["EUR/USD", "GBP/USD", "Gold (XAU/USD)", "BTC/USD"],
                key="auto_scan_pairs",
            )
        with as_col2:
            auto_tf = st.selectbox("⏱️ Timeframe", list(AUTO_TF_MAP.keys()),
                                   index=2, key="auto_scan_tf")
        with as_col3:
            auto_candles = st.selectbox("🕯️ Candles", [50, 100], index=1, key="auto_scan_candles")

        if st.button("🚀 Run Auto Scan", use_container_width=True, type="primary", key="auto_scan_btn"):
            if not selected_pairs:
                st.warning("Select at least one pair from the watchlist.")
            else:
                td_int, yf_int, yf_period = AUTO_TF_MAP[auto_tf]
                auto_results = []
                prog = st.progress(0)
                stat = st.empty()
                total = len(selected_pairs)

                for idx, pair_name in enumerate(selected_pairs):
                    stat.text(f"📡 Fetching & analysing {pair_name}... ({idx+1}/{total})")
                    prog.progress(idx / total)
                    try:
                        td_sym_a, yf_sym_a = AUTO_WATCHLIST[pair_name]

                        # Fetch data
                        if twelve_data_key:
                            import requests as _req2, pandas as _pd2
                            _url = "https://api.twelvedata.com/time_series"
                            _p   = {"symbol": td_sym_a, "interval": td_int,
                                    "outputsize": auto_candles, "apikey": twelve_data_key, "format": "JSON"}
                            _r   = _req2.get(_url, params=_p, timeout=12)
                            _d   = _r.json()
                            if _d.get("status") == "error":
                                raise ValueError(_d.get("message", "API error"))
                            _rows = [{"Datetime": v["datetime"],
                                      "Open": float(v["open"]), "High": float(v["high"]),
                                      "Low": float(v["low"]), "Close": float(v["close"]),
                                      "Volume": float(v.get("volume", 0))}
                                     for v in _d.get("values", [])]
                            df_a = _pd2.DataFrame(_rows)
                            df_a["Datetime"] = _pd2.to_datetime(df_a["Datetime"])
                            df_a = df_a.sort_values("Datetime").set_index("Datetime")
                        else:
                            import yfinance as _yf2
                            _raw = _yf2.download(yf_sym_a, period=yf_period,
                                                 interval=yf_int, auto_adjust=True, progress=False)
                            if _raw.empty:
                                raise ValueError(f"No data for {yf_sym_a}")
                            if hasattr(_raw.columns, "levels"):
                                _raw.columns = _raw.columns.get_level_values(0)
                            if auto_tf == "H4":
                                _raw = _raw.resample("4h").agg({"Open":"first","High":"max",
                                                                 "Low":"min","Close":"last","Volume":"sum"}).dropna()
                            _raw.index = _raw.index.tz_localize(None) if _raw.index.tzinfo else _raw.index
                            df_a = _raw.tail(auto_candles).copy()

                        # Generate chart image
                        chart_img_a = generate_chart_image_from_df(df_a, pair_name, auto_tf)

                        # AI quick scan
                        _last = float(df_a["Close"].iloc[-1])
                        _qp   = f"""Analyse this {pair_name} chart on {auto_tf} timeframe. Current price: {_last:.5g}.
Output ONLY this JSON, nothing else:
{{"signal": "BUY" or "SELL" or "WAIT", "confidence": 1-10, "pattern": "pattern name or none",
"trend": "Bullish/Bearish/Sideways", "wyckoff_phase": "Accumulation/Markup/Distribution/Markdown/Unknown",
"key_level": "one key price level as price", "reason": "one sentence max"}}"""

                        _qa = analyze_chart_with_ai(chart_img_a, api_key, model_choice,
                                                     pair_name, auto_tf, _qp)
                        _jm = re.search(r'\{.*?\}', _qa, re.DOTALL)
                        _data = json.loads(_jm.group()) if _jm else {
                            "signal": "WAIT", "confidence": 5, "pattern": "N/A",
                            "trend": "Unknown", "reason": "Parse error"}
                        _data["label"]      = pair_name
                        _data["last_price"] = f"{_last:.5g}"
                        # News warning flag
                        _warns = get_news_warning(pair_name)
                        _data["news_warn"]  = len(_warns) > 0
                        _data["news_items"] = _warns
                        auto_results.append(_data)

                    except Exception as _ae:
                        auto_results.append({"label": pair_name, "signal": "ERROR", "confidence": 0,
                                             "pattern": "—", "trend": "—", "reason": str(_ae)[:80],
                                             "last_price": "—", "news_warn": False, "news_items": []})

                prog.progress(1.0)
                stat.text("✅ Auto scan complete!")
                st.session_state["auto_scan_results"] = auto_results

        # Show auto scan results
        if "auto_scan_results" in st.session_state:
            _ares = sorted(st.session_state["auto_scan_results"],
                           key=lambda x: x.get("confidence", 0), reverse=True)
            st.markdown("#### 🏆 Ranked Setups · 最佳机会排名")
            st.caption(f"Sorted by confidence · Timeframe: {auto_tf if 'auto_tf' in dir() else ''} · "
                       f"{'🟢 Twelve Data real-time' if twelve_data_key else '🟡 yfinance delayed'}")

            for rank, r in enumerate(_ares):
                sig   = r.get("signal", "WAIT")
                conf  = r.get("confidence", 5)
                sc    = "#10b981" if sig == "BUY" else ("#ef4444" if sig == "SELL" else "#6b7280")
                medal = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"][rank] if rank < 10 else ""
                news_badge = " &nbsp;<span style='background:#dc2626;color:white;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700'>⚠️ NEWS</span>" if r.get("news_warn") else ""
                high_conf  = conf >= 8
                border_w   = "3px" if high_conf else "1px"
                border_c   = sc if high_conf else "#334155"
                st.markdown(f"""
<div style='background:#1e293b;border:{border_w} solid {border_c};border-radius:10px;padding:14px 18px;margin:6px 0;display:flex;align-items:center;gap:16px'>
<span style='font-size:26px'>{medal}</span>
<div style='flex:1'>
  <span style='color:white;font-weight:700;font-size:16px'>{r.get("label","")}</span>
  <span style='color:#94a3b8;font-size:12px;margin-left:8px'>@ {r.get("last_price","")}</span>
  <span style='margin-left:10px;background:{sc};color:white;padding:2px 10px;border-radius:20px;font-weight:700;font-size:13px'>{sig}</span>
  <span style='margin-left:6px;color:#fbbf24;font-weight:600'>{conf}/10</span>{news_badge}<br>
  <span style='color:#94a3b8;font-size:12px'>📐 {r.get("pattern","—")} &nbsp;·&nbsp; 📈 {r.get("trend","—")} &nbsp;·&nbsp; 🔄 {r.get("wyckoff_phase","—")}</span><br>
  <span style='color:#cbd5e1;font-size:12px'>💬 {r.get("reason","")}</span>
  {''.join([f"<br><span style='color:#fca5a5;font-size:11px'>⚠️ {w['currency']} {w['title']} @ {w['date']}</span>" for w in r.get("news_items",[])])}
</div>
</div>
""", unsafe_allow_html=True)

    # ── MANUAL UPLOAD MODE ────────────────────────────────────
    else:
        st.caption("Upload up to 5 charts — AI ranks them by signal strength. 上传最多5张图，AI自动排名最佳机会。")
        scan_cols = st.columns(5)
        scan_images = {}
        scan_labels = {}

        for i, col in enumerate(scan_cols):
            with col:
                st.markdown(f"**Chart {i+1}**")
                lbl = st.text_input(f"Label", value=f"Chart {i+1}", key=f"scan_lbl_{i}", label_visibility="collapsed")
                img_file = st.file_uploader("Upload", type=["png","jpg","jpeg","webp"],
                                            key=f"scan_img_{i}", label_visibility="collapsed")
                if img_file:
                    scan_images[i] = Image.open(img_file)
                    scan_labels[i] = lbl
                    st.image(scan_images[i], use_container_width=True)

        if len(scan_images) > 0:
            if st.button(f"🔍 Scan All {len(scan_images)} Charts", use_container_width=True):
                scan_results = []
                progress = st.progress(0)
                status_text = st.empty()

                for idx, (i, img) in enumerate(scan_images.items()):
                    lbl = scan_labels.get(i, f"Chart {i+1}")
                    status_text.text(f"🤖 Analysing {lbl}...")
                    progress.progress((idx) / len(scan_images))

                    quick_prompt = f"""
Analyse this {market_type} chart QUICKLY. Output ONLY this JSON, nothing else:
{{"signal": "BUY" or "SELL" or "WAIT", "confidence": 1-10, "pattern": "pattern name or none",
"trend": "Bullish/Bearish/Sideways", "wyckoff_phase": "Accumulation/Markup/Distribution/Markdown/Unknown",
"key_level": "one key price level description", "reason": "one sentence max"}}
"""
                    try:
                        quick_analysis = analyze_chart_with_ai(
                            img, api_key, model_choice, market_type, timeframe, quick_prompt
                        )
                        json_match = re.search(r'\{.*?\}', quick_analysis, re.DOTALL)
                        if json_match:
                            data = json.loads(json_match.group())
                        else:
                            data = {"signal": "WAIT", "confidence": 5, "pattern": "N/A",
                                    "trend": "Unknown", "reason": "Parse error"}
                        data["label"] = lbl
                        data["image_idx"] = i
                        scan_results.append(data)
                    except Exception as e:
                        scan_results.append({"label": lbl, "signal": "ERROR", "confidence": 0,
                                             "pattern": "Error", "trend": "N/A", "reason": str(e)[:60]})

                progress.progress(1.0)
                status_text.text("✅ Scan complete! Ranking results...")
                st.session_state["scan_results"] = scan_results

        if "scan_results" in st.session_state:
            results = sorted(st.session_state["scan_results"],
                             key=lambda x: x.get("confidence", 0), reverse=True)
            st.markdown("#### 🏆 Ranked Results — Best Setups First")

            for rank, r in enumerate(results):
                sig   = r.get("signal", "WAIT")
                conf  = r.get("confidence", 5)
                sig_color = "#10b981" if sig == "BUY" else ("#ef4444" if sig == "SELL" else "#6b7280")
                medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][rank] if rank < 5 else ""

                st.markdown(f"""
<div style='background:#1e293b;border:2px solid {sig_color};border-radius:10px;padding:14px 18px;margin:8px 0;display:flex;align-items:center;gap:16px'>
<span style='font-size:28px'>{medal}</span>
<div style='flex:1'>
  <span style='color:white;font-weight:700;font-size:17px'>{r.get("label","")}</span>
  <span style='margin-left:12px;background:{sig_color};color:white;padding:3px 10px;border-radius:20px;font-weight:700;font-size:13px'>{sig}</span>
  <span style='margin-left:8px;color:#fbbf24;font-weight:600'>{conf}/10</span><br>
  <span style='color:#94a3b8;font-size:13px'>📐 {r.get("pattern","N/A")} &nbsp;·&nbsp; 📈 {r.get("trend","N/A")} &nbsp;·&nbsp; 🔄 {r.get("wyckoff_phase","N/A")}</span><br>
  <span style='color:#cbd5e1;font-size:13px'>💬 {r.get("reason","")}</span>
</div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TOOL 4 — AI TRADING COACH
# ════════════════════════════════════════════════════════════
with tool_tab4:
    st.markdown("### 🤖 AI Trading Coach 交易导师")

    # ── Coach mode selector ────────────────────────────────
    coach_mode = st.radio(
        "选择模式 / Select mode",
        ["💬 Ask anything", "🔍 Review my analysis"],
        horizontal=True,
        key="coach_mode",
        help="Ask anything = general Q&A | Review = upload YOUR chart and get expert feedback on your analysis",
    )
    st.divider()

    # ══════════════════════════════════════════════════════
    # MODE 1 — Ask Anything (general Q&A chat)
    # ══════════════════════════════════════════════════════
    if coach_mode == "💬 Ask anything":
        st.caption("用中文或英文问任何交易问题 · Ask any trading question in English or Chinese")

        # Initialise chat history
        if "coach_messages" not in st.session_state:
            st.session_state["coach_messages"] = [
                {"role": "assistant", "content": (
                    "你好！我是你的AI交易导师 👋\n\n"
                    "我专注于 **SMC / Wyckoff / Price Action / Scalping / Risk Management**。\n\n"
                    "你可以问我：\n"
                    "- 「Supply & Demand Zone 跟 S/R 有什么区别？」\n"
                    "- 「CHoCH 和 BOS 有什么区别？」\n"
                    "- 「Scalper 应该在什么时候进场？」\n"
                    "- 「How do I identify a liquidity sweep?」\n\n"
                    "随时问，我都在！😊"
                )}
            ]

        for msg in st.session_state["coach_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if user_question := st.chat_input("问我任何交易问题... / Ask any trading question...", key="coach_qa_input"):
            if not api_key:
                st.warning("👈 请先在侧边栏输入 API Key")
            else:
                st.session_state["coach_messages"].append({"role": "user", "content": user_question})
                with st.chat_message("user"):
                    st.markdown(user_question)

                with st.chat_message("assistant"):
                    with st.spinner("思考中..."):
                        try:
                            coach_system = """You are an elite trading coach and mentor with 20+ years of experience in Forex, Crypto, Commodities, and Indices.
You specialise in Wyckoff Method, Price Action, Market Structure (BOS/CHoCH), Liquidity, Supply & Demand, Fibonacci, Scalping, Day Trading, and Risk Management.
You respond in the same language the student uses (English or Chinese/Mandarin). If they mix languages, respond in Chinese primarily.
Keep answers practical, clear, and educational. Use bullet points and examples where helpful.
Always emphasise risk management and discipline. Be direct — tell students clearly when something is right or wrong.
Never give specific financial advice or tell someone to buy/sell a specific real asset."""

                            if model_choice.startswith("gemini"):
                                client_coach = google_genai.Client(api_key=api_key)
                                history_text = "\n".join([
                                    f"{'Student' if m['role']=='user' else 'Coach'}: {m['content']}"
                                    for m in st.session_state["coach_messages"][-10:]
                                ])
                                full_q = coach_system + "\n\nConversation:\n" + history_text
                                coach_resp = client_coach.models.generate_content(
                                    model=model_choice, contents=[full_q])
                                answer = coach_resp.text
                            else:
                                client_coach = anthropic.Anthropic(api_key=api_key)
                                history_msgs = [
                                    {"role": m["role"], "content": m["content"]}
                                    for m in st.session_state["coach_messages"]
                                    if m["role"] in ("user", "assistant")
                                ][-12:]
                                coach_resp = client_coach.messages.create(
                                    model=model_choice, max_tokens=1500,
                                    system=coach_system, messages=history_msgs)
                                answer = coach_resp.content[0].text

                            st.markdown(answer)
                            st.session_state["coach_messages"].append({"role": "assistant", "content": answer})
                            # ── Persist chat to localStorage ──
                            if _LS_AVAILABLE:
                                try:
                                    _ls.setItem("trading_analyst_chat",
                                                st.session_state["coach_messages"][-40:])
                                except Exception:
                                    pass
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        if len(st.session_state.get("coach_messages", [])) > 2:
            if st.button("🗑️ Clear Chat 清除对话", key="clear_coach"):
                st.session_state["coach_messages"] = [st.session_state["coach_messages"][0]]
                if _LS_AVAILABLE:
                    try:
                        _ls.deleteItem("trading_analyst_chat")
                    except Exception:
                        pass
                st.rerun()

    # ══════════════════════════════════════════════════════
    # MODE 2 — Review My Analysis (continuous conversation)
    # ══════════════════════════════════════════════════════
    else:
        # ── Session state keys for review conversation ────
        # coach_review_active  : bool  — is a session in progress?
        # coach_review_conv    : list  — full message history [{role, content}]
        # coach_review_img_b64 : str   — base64 image kept for API calls
        # coach_review_img_bytes: bytes — PNG bytes for display
        # coach_review_system  : str   — system prompt for this session

        active = st.session_state.get("coach_review_active", False)

        # ══════════════════════════════════════════════════
        # SUB-PHASE A — Setup form (no active conversation)
        # ══════════════════════════════════════════════════
        if not active:
            st.caption("上传你自己画好分析线的图表，AI导师帮你检查，之后可以继续追问直到完全明白为止。")
            st.caption("Upload YOUR chart → get expert review → keep chatting until every doubt is cleared.")

            # Upload
            review_img_file = st.file_uploader(
                "📷 Upload your chart (with your own drawings) · 上传你的分析图",
                type=["png", "jpg", "jpeg", "webp"],
                key="coach_review_img",
            )
            if review_img_file:
                preview_img = Image.open(review_img_file)
                st.image(preview_img, caption="Your chart · 你的图表", use_container_width=True)

            # Description
            st.markdown("**描述你的分析 / Describe your analysis:**")
            review_description = st.text_area(
                label="analysis_desc",
                label_visibility="collapsed",
                placeholder=(
                    "例如 / Example:\n"
                    "- 我画了一条下降趋势线，从左上到右下\n"
                    "- 我认为这里是一个 Supply Zone（供给区）\n"
                    "- 我认为价格在这里形成了 CHoCH，因为打破了低点\n"
                    "- I drew support at the recent swing low and resistance at the last swing high\n"
                    "- I think this is a bull flag consolidation"
                ),
                height=140,
                key="coach_review_desc",
            )

            # Options
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                review_focus = st.selectbox(
                    "Review focus · 重点检查",
                    [
                        "Overall analysis · 整体分析",
                        "Trend & structure · 趋势与结构",
                        "Supply & Demand Zones",
                        "BOS & CHoCH identification",
                        "Support & Resistance levels",
                        "Trendlines · 趋势线画法",
                        "Liquidity zones · 流动性",
                        "Pattern identification · 形态识别",
                    ],
                    key="coach_review_focus",
                )
            with col_r2:
                review_lang = st.selectbox(
                    "Response language · 回复语言",
                    ["中文为主 (Chinese)", "English", "两种都用 (Both)"],
                    key="coach_review_lang",
                )

            # Start button
            if st.button("🔍 Start Review Session · 开始分析点评", use_container_width=True, key="coach_review_start"):
                if not api_key:
                    st.warning("👈 请先在侧边栏输入 API Key")
                elif not review_img_file:
                    st.warning("👆 请先上传你的图表 / Please upload your chart first")
                elif not review_description.strip():
                    st.warning("📝 请先描述你的分析 / Please describe your analysis first")
                else:
                    with st.spinner("导师正在审查你的分析... Reviewing your analysis..."):
                        try:
                            lang_instruction = {
                                "中文为主 (Chinese)": "Respond primarily in Chinese (Mandarin). Use English only for technical terms.",
                                "English": "Respond in English.",
                                "两种都用 (Both)": "Respond in both English and Chinese — write each point in English first, then the Chinese translation below it.",
                            }.get(review_lang, "Respond in Chinese.")

                            review_system = f"""You are a strict but fair elite trading coach and mentor with 20+ years of experience in Forex, Crypto, Commodities, and Indices.
You specialise in Wyckoff, Price Action, BOS/CHoCH, Liquidity, Supply & Demand, Fibonacci, and Scalping.

{lang_instruction}

You are in an ONGOING COACHING SESSION with a student. A chart image was shared at the start of the session.
You have already seen the chart and given an initial review. The student may now ask follow-up questions, disagree with your points, ask for more detail on specific areas, or ask you to explain concepts they don't understand.

BEHAVIOUR RULES:
- Always refer back to what you can see in the chart when answering follow-ups
- If the student disagrees with your point, re-examine your reasoning and either stand firm with a clear explanation, or acknowledge if they have a valid point
- If they ask "why is this wrong?", explain the exact RULE or PRINCIPLE that was violated
- If they ask "what should I have done instead?", show them the correct approach
- Be direct and specific — never give vague answers
- Use numbered steps or bullet points when explaining rules
- Focus on this area throughout the session: {review_focus}

For the INITIAL review, always follow this structure:
## ✅ What You Got Right · 做对的地方
## ❌ What Is Wrong · 需要改正的地方
## 💡 What You Missed · 你没有注意到的
## 📝 How to Improve · 改进建议
## 🎯 Overall Score · 整体评分 (X/10)

For FOLLOW-UP messages, respond naturally to the question — no need to repeat the full structure."""

                            # Encode the image
                            review_img_obj = Image.open(review_img_file)
                            img_b64 = encode_image_to_base64(review_img_obj)

                            # Store PNG bytes for display (lossless)
                            img_disp_buf = io.BytesIO()
                            disp_img = review_img_obj.copy()
                            if disp_img.mode in ("RGBA", "P"):
                                disp_img = disp_img.convert("RGB")
                            disp_img.save(img_disp_buf, format="PNG", compress_level=1)
                            img_disp_buf.seek(0)

                            first_user_msg = f"""Please review my chart analysis.

My analysis · 我的分析：
{review_description}

Focus area: {review_focus}

Please give me your full review — what I got right, what is wrong, what I missed, and how to improve."""

                            # Call AI with the image
                            if model_choice.startswith("gemini"):
                                client_coach = google_genai.Client(api_key=api_key)
                                img_buf_r = io.BytesIO()
                                review_img_rgb = review_img_obj.copy()
                                if review_img_rgb.mode in ("RGBA", "P"):
                                    review_img_rgb = review_img_rgb.convert("RGB")
                                review_img_rgb.save(img_buf_r, format="JPEG", quality=92)
                                img_bytes_r = img_buf_r.getvalue()

                                full_prompt_r = review_system + "\n\nStudent: " + first_user_msg
                                coach_resp_r = client_coach.models.generate_content(
                                    model=model_choice,
                                    contents=[
                                        full_prompt_r,
                                        google_types.Part.from_bytes(data=img_bytes_r, mime_type="image/jpeg"),
                                    ],
                                )
                                initial_answer = coach_resp_r.text
                            else:
                                client_coach = anthropic.Anthropic(api_key=api_key)
                                coach_resp_r = client_coach.messages.create(
                                    model=model_choice,
                                    max_tokens=2500,
                                    system=review_system,
                                    messages=[{
                                        "role": "user",
                                        "content": [
                                            {"type": "image", "source": {
                                                "type": "base64",
                                                "media_type": "image/jpeg",
                                                "data": img_b64,
                                            }},
                                            {"type": "text", "text": first_user_msg},
                                        ],
                                    }],
                                )
                                initial_answer = coach_resp_r.content[0].text

                            # ── Activate the conversation session ──
                            st.session_state["coach_review_active"]    = True
                            st.session_state["coach_review_system"]    = review_system
                            st.session_state["coach_review_img_b64"]   = img_b64
                            st.session_state["coach_review_img_bytes"] = img_disp_buf.getvalue()
                            st.session_state["coach_active_focus"]     = review_focus
                            # Conversation history: user's first message + coach's initial review
                            st.session_state["coach_review_conv"] = [
                                {"role": "user",      "content": first_user_msg},
                                {"role": "assistant", "content": initial_answer},
                            ]
                            st.rerun()

                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        # ══════════════════════════════════════════════════
        # SUB-PHASE B — Active conversation
        # ══════════════════════════════════════════════════
        else:
            focus_label = st.session_state.get("coach_active_focus", "")

            # ── Header bar with focus + stop button ───────
            hdr_col1, hdr_col2 = st.columns([3, 1])
            with hdr_col1:
                st.markdown(f"**🎓 Coaching Session · 导师辅导中** — Focus: {focus_label}")
                st.caption("继续追问导师，直到你完全明白为止。当你满意了，按右边的按钮结束对话。")
            with hdr_col2:
                if st.button("🛑 Stop Conversation\n结束对话", use_container_width=True, key="coach_stop_btn"):
                    # Save to history before clearing
                    if "coach_review_history" not in st.session_state:
                        st.session_state["coach_review_history"] = []
                    st.session_state["coach_review_history"].append({
                        "focus":        focus_label,
                        "conversation": list(st.session_state.get("coach_review_conv", [])),
                    })
                    # Clear active session
                    for k in ["coach_review_active", "coach_review_conv",
                              "coach_review_img_b64", "coach_review_img_bytes",
                              "coach_review_system", "coach_active_focus",
                              "coach_extra_img_counter"]:
                        st.session_state.pop(k, None)
                    st.success("✅ 对话已结束，已保存到历史记录。/ Session ended and saved to history.")
                    st.rerun()

            st.divider()

            # ── Show chart thumbnail ───────────────────────
            img_bytes_disp = st.session_state.get("coach_review_img_bytes")
            if img_bytes_disp:
                with st.expander("📷 View your chart · 查看图表", expanded=False):
                    st.image(img_bytes_disp, use_container_width=True)

            # ── Render conversation history ────────────────
            conv = st.session_state.get("coach_review_conv", [])
            for msg in conv:
                role  = msg["role"]
                # For the first user message, show a simplified version
                if role == "user" and msg == conv[0]:
                    with st.chat_message("user"):
                        desc_lines = [l for l in msg["content"].split("\n") if l.strip() and not l.startswith("Please review") and not l.startswith("Focus area") and not l.startswith("Please tell")]
                        st.markdown("\n".join(desc_lines[:8]))
                else:
                    with st.chat_message(role):
                        # Show attached image thumbnail if present
                        if role == "user" and msg.get("extra_img_b64"):
                            try:
                                extra_thumb = base64.b64decode(msg["extra_img_b64"])
                                st.image(extra_thumb, width=220, caption="📎 Chart attached")
                            except Exception:
                                pass
                        st.markdown(msg["content"])

            # ── Optional image attachment for follow-up ────
            st.markdown("<br>", unsafe_allow_html=True)
            _extra_counter = st.session_state.get("coach_extra_img_counter", 0)
            extra_img_file = st.file_uploader(
                "📎 Attach a new chart (optional) · 可选：上传新图表",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"coach_extra_img_{_extra_counter}",
                help="Upload another chart — e.g. your updated drawing or a result screenshot",
            )
            if extra_img_file:
                st.caption("✅ Image attached — it will be sent with your next message.")

            # ── Follow-up chat input ───────────────────────
            followup = st.chat_input(
                "继续追问导师... / Ask a follow-up question...",
                key="coach_review_followup",
            )

            if followup:
                if not api_key:
                    st.warning("👈 请先在侧边栏输入 API Key")
                else:
                    # Encode extra image if attached
                    extra_img_b64 = None
                    extra_img_bytes_send = None
                    if extra_img_file:
                        try:
                            _extra_pil = Image.open(extra_img_file)
                            _extra_buf = io.BytesIO()
                            _extra_rgb = _extra_pil.copy()
                            if _extra_rgb.mode in ("RGBA", "P"):
                                _extra_rgb = _extra_rgb.convert("RGB")
                            _extra_rgb.save(_extra_buf, format="JPEG", quality=92)
                            extra_img_bytes_send = _extra_buf.getvalue()
                            extra_img_b64 = base64.b64encode(extra_img_bytes_send).decode("utf-8")
                            # Bump counter so uploader resets after send
                            st.session_state["coach_extra_img_counter"] = _extra_counter + 1
                        except Exception:
                            pass

                    # Build display text for the user message
                    user_display = followup
                    if extra_img_b64:
                        user_display = followup  # image shown via extra_img_b64 field

                    # Add user message to history
                    user_conv_entry = {"role": "user", "content": followup}
                    if extra_img_b64:
                        user_conv_entry["extra_img_b64"] = extra_img_b64
                    conv.append(user_conv_entry)
                    st.session_state["coach_review_conv"] = conv

                    with st.chat_message("user"):
                        if extra_img_b64:
                            try:
                                st.image(base64.b64decode(extra_img_b64), width=220, caption="📎 Chart attached")
                            except Exception:
                                pass
                        st.markdown(followup)

                    with st.chat_message("assistant"):
                        with st.spinner("导师思考中... Thinking..."):
                            try:
                                review_sys  = st.session_state.get("coach_review_system", "")
                                img_b64_key = st.session_state.get("coach_review_img_b64", "")

                                if model_choice.startswith("gemini"):
                                    client_coach = google_genai.Client(api_key=api_key)
                                    # Build full history text for Gemini
                                    history_text = "\n\n".join([
                                        f"{'Student' if m['role']=='user' else 'Coach'}:\n{m['content']}"
                                        for m in conv
                                    ])
                                    full_q = review_sys + "\n\n---\nConversation so far:\n" + history_text
                                    # Build contents list
                                    gemini_contents = [full_q]
                                    # Include original chart for context
                                    orig_img_bytes = st.session_state.get("coach_review_img_bytes", b"")
                                    if orig_img_bytes:
                                        gemini_contents.append(
                                            google_types.Part.from_bytes(data=orig_img_bytes, mime_type="image/png")
                                        )
                                    # Include new chart if attached
                                    if extra_img_bytes_send:
                                        gemini_contents.append(
                                            google_types.Part.from_bytes(data=extra_img_bytes_send, mime_type="image/jpeg")
                                        )
                                        gemini_contents[0] += "\n\n[The student has attached a NEW chart image above. Please analyse it in the context of your conversation.]"
                                    coach_resp_fu = client_coach.models.generate_content(
                                        model=model_choice,
                                        contents=gemini_contents,
                                    )
                                    followup_answer = coach_resp_fu.text

                                else:
                                    # Claude: first message carries original image; extra image goes in current user message
                                    client_coach = anthropic.Anthropic(api_key=api_key)
                                    api_msgs = []
                                    for i, m in enumerate(conv):
                                        if i == 0 and m["role"] == "user" and img_b64_key:
                                            # First message: include original image
                                            api_msgs.append({
                                                "role": "user",
                                                "content": [
                                                    {"type": "image", "source": {
                                                        "type": "base64",
                                                        "media_type": "image/jpeg",
                                                        "data": img_b64_key,
                                                    }},
                                                    {"type": "text", "text": m["content"]},
                                                ],
                                            })
                                        elif m["role"] == "user" and m.get("extra_img_b64") and i == len(conv) - 1:
                                            # Current user message with new image attached
                                            api_msgs.append({
                                                "role": "user",
                                                "content": [
                                                    {"type": "image", "source": {
                                                        "type": "base64",
                                                        "media_type": "image/jpeg",
                                                        "data": m["extra_img_b64"],
                                                    }},
                                                    {"type": "text", "text": m["content"]},
                                                ],
                                            })
                                        else:
                                            api_msgs.append({"role": m["role"], "content": m["content"]})

                                    coach_resp_fu = client_coach.messages.create(
                                        model=model_choice,
                                        max_tokens=1800,
                                        system=review_sys,
                                        messages=api_msgs[-20:],
                                    )
                                    followup_answer = coach_resp_fu.content[0].text

                                conv.append({"role": "assistant", "content": followup_answer})
                                st.session_state["coach_review_conv"] = conv

                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                    st.rerun()   # re-render page so chat_input reappears

        # ── Past session history ───────────────────────────
        if not active and st.session_state.get("coach_review_history"):
            history = st.session_state["coach_review_history"]
            with st.expander(f"📚 Past Sessions · 历史对话 ({len(history)} sessions)", expanded=False):
                for i, sess in enumerate(reversed(history[-5:]), 1):
                    st.markdown(f"**Session {i}** — Focus: {sess.get('focus', '—')}")
                    for msg in sess.get("conversation", []):
                        role_label = "🧑 You" if msg["role"] == "user" else "🎓 Coach"
                        content_preview = msg["content"][:300] + ("..." if len(msg["content"]) > 300 else "")
                        st.markdown(f"**{role_label}:** {content_preview}")
                    st.divider()


# ════════════════════════════════════════════════════════════
# TOOL 5 — PDF REPORT GENERATOR
# ════════════════════════════════════════════════════════════
with tool_tab5:
    st.markdown("### 📄 PDF Report Generator 分析报告")
    st.caption("Generate a professional PDF report from your latest analysis. 一键生成专业PDF交易分析报告。")

    if "analysis" not in st.session_state:
        st.info("👆 Run a chart analysis first, then come back here to generate the PDF report.")
    else:
        st.success("✅ Analysis found! Ready to generate PDF.")

        report_trader_name = st.text_input("👤 Trader Name (optional)", placeholder="e.g. Chee")
        report_notes       = st.text_area("📝 Personal Notes (optional)", placeholder="e.g. Waiting for NY session confirmation before entry...", height=80)

        if st.button("📄 Generate PDF Report", use_container_width=True):
            try:
                from fpdf import FPDF
                import datetime

                analysis_text = st.session_state["analysis"]
                clean_text    = re.sub(r"```json.*?```", "", analysis_text, flags=re.DOTALL).strip()
                parsed        = parse_json_from_analysis(analysis_text)
                sig           = parsed.get("signal", "WAIT")
                conf          = parsed.get("confidence", 5)
                pattern       = parsed.get("pattern_name", "N/A")
                now_str       = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)

                # Header
                pdf.set_fill_color(30, 30, 60)
                pdf.rect(0, 0, 210, 32, 'F')
                pdf.set_font("Helvetica", "B", 22)
                pdf.set_text_color(255, 255, 255)
                pdf.set_xy(10, 8)
                pdf.cell(0, 10, "TradingAI Analyst - Chart Analysis Report", ln=True)
                pdf.set_font("Helvetica", "", 11)
                pdf.set_xy(10, 20)
                pdf.cell(0, 8, f"Generated: {now_str}  |  Trader: {report_trader_name or 'Anonymous'}  |  {market_type}  {timeframe}", ln=True)

                pdf.set_text_color(0, 0, 0)
                pdf.ln(8)

                # Signal box
                sig_colors = {"BUY": (5,150,105), "SELL": (220,38,38), "WAIT": (180,130,0)}
                sc = sig_colors.get(sig, (100,100,100))
                pdf.set_fill_color(*sc)
                pdf.set_text_color(255,255,255)
                pdf.set_font("Helvetica","B",18)
                pdf.cell(0, 14, f"  SIGNAL: {sig}   |   Confidence: {conf}/10   |   Pattern: {pattern}", ln=True, fill=True)
                pdf.set_text_color(0,0,0)
                pdf.ln(5)

                # Annotated chart image
                if st.session_state.get("annotated"):
                    img_buf = io.BytesIO()
                    st.session_state["annotated"].save(img_buf, format="PNG")
                    img_buf.seek(0)
                    tmp_img_path = "/tmp/report_chart.png"
                    with open(tmp_img_path, "wb") as f:
                        f.write(img_buf.read())
                    try:
                        pdf.image(tmp_img_path, x=10, w=190)
                        pdf.ln(4)
                    except Exception:
                        pass

                # Analysis text
                pdf.set_font("Helvetica","B",13)
                pdf.set_fill_color(240,240,255)
                pdf.cell(0, 9, " AI Analysis", ln=True, fill=True)
                pdf.set_font("Helvetica","",10)
                pdf.ln(2)

                for line in clean_text.split("\n"):
                    line = line.strip()
                    if not line:
                        pdf.ln(2)
                        continue
                    # Strip markdown bold
                    line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
                    # Skip emoji-heavy lines that can't render
                    safe_line = line.encode("latin-1", errors="replace").decode("latin-1")
                    if line.startswith("**") or line.startswith("#"):
                        pdf.set_font("Helvetica","B",11)
                    else:
                        pdf.set_font("Helvetica","",10)
                    pdf.multi_cell(0, 6, safe_line)

                # Personal notes
                if report_notes:
                    pdf.ln(4)
                    pdf.set_font("Helvetica","B",12)
                    pdf.set_fill_color(255,250,220)
                    pdf.cell(0, 9, " Trader Notes", ln=True, fill=True)
                    pdf.set_font("Helvetica","",10)
                    safe_notes = report_notes.encode("latin-1", errors="replace").decode("latin-1")
                    pdf.multi_cell(0, 6, safe_notes)

                # Footer
                pdf.ln(6)
                pdf.set_font("Helvetica","I",8)
                pdf.set_text_color(150,150,150)
                pdf.cell(0, 5, "DISCLAIMER: For educational purposes only. Not financial advice. Always manage your own risk.", ln=True)

                # Output
                pdf_bytes = pdf.output(dest="S").encode("latin-1")
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"TradingAI_Report_{now_str[:10]}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success("✅ PDF ready! Click above to download.")

            except ImportError:
                st.error("PDF library not installed. Add 'fpdf2' to requirements.txt and redeploy.")
            except Exception as e:
                st.error(f"PDF generation error: {str(e)}")


# ════════════════════════════════════════════════════════════
# TOOL 6 — CURRENCY STRENGTH METER
# ════════════════════════════════════════════════════════════
with tool_tab6:
    st.markdown("### 💹 Currency Strength Meter 货币强弱表")
    st.caption("Upload H1 charts for each currency pair to compute relative strength. 上传各货币对H1图表，自动计算货币强弱。")

    if not api_key:
        st.warning("👈 Enter your API key in the sidebar first.")
    else:
        st.markdown("""
<div style='background:#1e293b;border-radius:8px;padding:12px;margin-bottom:12px'>
<p style='color:#94a3b8;font-size:13px;margin:0'>
📋 <b style='color:#fbbf24'>How it works:</b> Upload H1 charts for any pairs below.
AI analyses each one and scores each currency's strength from 1-10.
The meter shows who's strongest today.<br>
<span style='color:#86efac'>怎么用：上传下面任何货币对的H1图，AI分析后自动给每个货币打分，显示今天谁最强。</span>
</p>
</div>
""", unsafe_allow_html=True)

        csm_pairs = {
            "EUR/USD": ("EUR", "USD"), "GBP/USD": ("GBP", "USD"),
            "USD/JPY": ("USD", "JPY"), "AUD/USD": ("AUD", "USD"),
            "USD/CAD": ("USD", "CAD"), "USD/CHF": ("USD", "CHF"),
            "NZD/USD": ("NZD", "USD"), "XAU/USD": ("XAU", "USD"),
        }

        csm_cols = st.columns(4)
        csm_uploads = {}
        for idx, (pair, currencies) in enumerate(csm_pairs.items()):
            with csm_cols[idx % 4]:
                st.markdown(f"**{pair}**")
                f = st.file_uploader("", type=["png","jpg","jpeg","webp"],
                                     key=f"csm_{pair}", label_visibility="collapsed")
                if f:
                    csm_uploads[pair] = (Image.open(f), currencies)
                    st.image(csm_uploads[pair][0], use_container_width=True)

        if len(csm_uploads) >= 2:
            if st.button(f"⚡ Calculate Strength from {len(csm_uploads)} pairs", use_container_width=True):
                currency_scores = {}
                prog = st.progress(0)
                stat = st.empty()

                for i, (pair, (img, (base_ccy, quote_ccy))) in enumerate(csm_uploads.items()):
                    stat.text(f"Analysing {pair}...")
                    prog.progress(i / len(csm_uploads))
                    try:
                        quick = f"""Analyse this {pair} H1 chart. Output ONLY this JSON:
{{"signal": "BUY" or "SELL" or "WAIT", "confidence": 1-10, "trend": "Bullish/Bearish/Sideways"}}"""
                        r = analyze_chart_with_ai(img, api_key, model_choice, pair, "H1", quick)
                        m = re.search(r'\{.*?\}', r, re.DOTALL)
                        if m:
                            d = json.loads(m.group())
                            sig_val = d.get("signal","WAIT")
                            conf_val = d.get("confidence", 5)
                            # Score: BUY = base strong, SELL = quote strong
                            strength = conf_val if sig_val == "BUY" else (10 - conf_val if sig_val == "SELL" else 5)
                            # Base currency gets strength, quote gets inverse
                            currency_scores[base_ccy]  = currency_scores.get(base_ccy, []) + [strength]
                            currency_scores[quote_ccy] = currency_scores.get(quote_ccy, []) + [10 - strength]
                    except Exception:
                        pass

                prog.progress(1.0)
                stat.empty()

                if currency_scores:
                    # Average scores
                    avg_scores = {c: sum(v)/len(v) for c, v in currency_scores.items()}
                    sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
                    st.session_state["csm_scores"] = sorted_scores

        if "csm_scores" in st.session_state:
            scores = st.session_state["csm_scores"]
            st.markdown("#### 📊 Currency Strength Ranking 货币强弱排名")

            max_score = max(s for _, s in scores) if scores else 10
            for rank, (ccy, score) in enumerate(scores):
                pct   = score / 10
                bar_w = int(pct * 100)
                if pct >= 0.70:
                    bar_color = "#10b981"; label_color = "#6ee7b7"; tag = "STRONG 强势 💪"
                elif pct >= 0.45:
                    bar_color = "#f59e0b"; label_color = "#fcd34d"; tag = "NEUTRAL 中性 ➡️"
                else:
                    bar_color = "#ef4444"; label_color = "#fca5a5"; tag = "WEAK 弱势 📉"
                medal = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣"][rank] if rank < 8 else ""
                st.markdown(f"""
<div style='background:#1e293b;border-radius:8px;padding:12px 16px;margin:6px 0'>
<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>
  <span style='color:white;font-weight:700;font-size:17px'>{medal} {ccy}</span>
  <span style='color:{label_color};font-weight:600;font-size:13px'>{tag}</span>
  <span style='color:#94a3b8;font-size:14px;font-weight:700'>{score:.1f}/10</span>
</div>
<div style='background:#334155;border-radius:4px;height:14px'>
  <div style='background:{bar_color};width:{bar_w}%;height:14px;border-radius:4px;transition:width 0.5s'></div>
</div>
</div>
""", unsafe_allow_html=True)

            # Trading suggestions based on strength
            if len(scores) >= 2:
                strongest = scores[0][0]
                weakest   = scores[-1][0]
                st.markdown(f"""
<div style='background:linear-gradient(135deg,#0f2027,#203a43);border:2px solid #3b82f6;border-radius:10px;padding:16px;margin-top:12px'>
<p style='color:#60a5fa;font-weight:700;font-size:15px;margin:0 0 8px 0'>💡 Trading Suggestion 交易建议</p>
<p style='color:#e2e8f0;margin:0'>
Strongest: <b style='color:#10b981'>{strongest}</b> &nbsp;·&nbsp; Weakest: <b style='color:#ef4444'>{weakest}</b><br>
<span style='color:#fbbf24'>➡️ Look for {strongest}/{weakest} pair — buy {strongest}, sell {weakest}</span><br>
<span style='color:#86efac;font-size:13px'>寻找 {strongest}/{weakest} 货币对 — 买入{strongest}，卖出{weakest}</span>
</p>
</div>
""", unsafe_allow_html=True)
        elif len(csm_uploads) < 2:
            st.info("👆 Upload at least 2 currency pair charts to calculate strength.")


# ════════════════════════════════════════════════════════════
# TOOL 7 — LIVE DATA ANALYSIS
# ════════════════════════════════════════════════════════════
with tool_tab7:
    st.markdown("### 📈 Live Data Analysis 实时数据分析")
    st.caption("Fetch live candles directly — no chart upload needed. 直接拉取实时K线，无需上传图表。")

    # ── Lazy imports ──────────────────────────────────────────
    try:
        import yfinance as yf
        import plotly.graph_objects as go
        _LIVE_AVAILABLE = True
    except ImportError:
        _LIVE_AVAILABLE = False
        st.error("📦 Live Data requires `yfinance` and `plotly`. Please redeploy after updating requirements.txt.")

    if _LIVE_AVAILABLE:
        # ── Data source indicator ─────────────────────────────
        if twelve_data_key:
            st.success("🟢 **Real-time data** via Twelve Data · No delay")
        else:
            st.warning("🟡 Using yfinance (15-min delayed). Add Twelve Data key in sidebar for real-time.")

        # ── Ticker presets ────────────────────────────────────
        # Each entry: display_name → (twelve_data_symbol, yfinance_fallback)
        TICKER_PRESETS = {
            "EUR/USD":        ("EUR/USD",  "EURUSD=X"),
            "GBP/USD":        ("GBP/USD",  "GBPUSD=X"),
            "USD/JPY":        ("USD/JPY",  "USDJPY=X"),
            "AUD/USD":        ("AUD/USD",  "AUDUSD=X"),
            "NZD/USD":        ("NZD/USD",  "NZDUSD=X"),
            "USD/CAD":        ("USD/CAD",  "USDCAD=X"),
            "USD/CHF":        ("USD/CHF",  "USDCHF=X"),
            "GBP/JPY":        ("GBP/JPY",  "GBPJPY=X"),
            "EUR/JPY":        ("EUR/JPY",  "EURJPY=X"),
            "Gold (XAU/USD)": ("XAU/USD",  "GC=F"),
            "Silver (XAG/USD)":("XAG/USD", "SI=F"),
            "WTI Oil":        ("WTI/USD",  "CL=F"),
            "BTC/USD":        ("BTC/USD",  "BTC-USD"),
            "ETH/USD":        ("ETH/USD",  "ETH-USD"),
            "S&P 500":        ("SPX",      "^GSPC"),
            "Nasdaq 100":     ("NDX",      "^NDX"),
            "Custom ✏️":      ("__custom__", "__custom__"),
        }

        # Timeframe → (twelve_data_interval, yf_interval, yf_period)
        TF_MAP = {
            "M1":  ("1min",  "1m",  "1d"),
            "M5":  ("5min",  "5m",  "5d"),
            "M15": ("15min", "15m", "5d"),
            "M30": ("30min", "30m", "10d"),
            "H1":  ("1h",    "1h",  "30d"),
            "H4":  ("4h",    "1h",  "60d"),
            "D1":  ("1day",  "1d",  "180d"),
            "W1":  ("1week", "1wk", "3y"),
        }

        # ── Controls row ──────────────────────────────────────
        ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([2, 1, 1, 1])

        with ctrl_c1:
            preset_choice = st.selectbox(
                "📌 Symbol",
                list(TICKER_PRESETS.keys()),
                index=0,
                key="ld_preset",
            )
            td_sym, yf_sym = TICKER_PRESETS[preset_choice]
            if preset_choice == "Custom ✏️":
                custom_input = st.text_input(
                    "Twelve Data symbol (if key set) or yfinance ticker",
                    placeholder="e.g. EUR/USD  or  AAPL",
                    key="ld_custom_ticker",
                ).strip().upper()
                td_sym  = custom_input
                yf_sym  = custom_input
                ticker_sym = custom_input
            else:
                ticker_sym = td_sym if twelve_data_key else yf_sym
                src_label  = "Twelve Data" if twelve_data_key else "yfinance"
                st.caption(f"{src_label} symbol: `{ticker_sym}`")

        with ctrl_c2:
            ld_tf = st.selectbox(
                "⏱️ Timeframe",
                ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"],
                index=4,
                key="ld_tf",
            )

        with ctrl_c3:
            ld_candles = st.selectbox(
                "🕯️ Candles",
                [50, 100, 150, 200],
                index=1,
                key="ld_candles",
            )

        with ctrl_c4:
            st.markdown("<br>", unsafe_allow_html=True)
            fetch_btn = st.button("🔄 Fetch & Analyse", use_container_width=True,
                                  key="ld_fetch_btn", type="primary")

        # ── Helper: fetch via Twelve Data ─────────────────────
        def _fetch_twelve_data(symbol, interval, outputsize, api_key_td):
            """Fetch OHLCV from Twelve Data REST API. Returns a pandas DataFrame."""
            import requests, pandas as pd
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol":     symbol,
                "interval":   interval,
                "outputsize": outputsize,
                "apikey":     api_key_td,
                "format":     "JSON",
            }
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") == "error":
                raise ValueError(data.get("message", "Twelve Data API error"))
            values = data.get("values", [])
            if not values:
                raise ValueError("No data returned — check symbol or interval.")
            rows = []
            for v in values:
                rows.append({
                    "Datetime": v["datetime"],
                    "Open":   float(v["open"]),
                    "High":   float(v["high"]),
                    "Low":    float(v["low"]),
                    "Close":  float(v["close"]),
                    "Volume": float(v.get("volume", 0)),
                })
            df_td = pd.DataFrame(rows)
            df_td["Datetime"] = pd.to_datetime(df_td["Datetime"])
            df_td = df_td.sort_values("Datetime").reset_index(drop=True)
            df_td = df_td.set_index("Datetime")
            return df_td

        # ── Fetch data ────────────────────────────────────────
        if fetch_btn:
            if not ticker_sym or ticker_sym == "__custom__":
                st.warning("Please enter a symbol.")
            elif not api_key:
                st.warning("👈 Enter your AI API key in the sidebar first.")
            else:
                td_interval, yf_interval, yf_period = TF_MAP.get(ld_tf, ("1h", "1h", "30d"))
                with st.spinner(f"Fetching {preset_choice} {ld_tf} data..."):
                    try:
                        if twelve_data_key:
                            # ── Twelve Data (real-time) ───────
                            df_raw = _fetch_twelve_data(
                                td_sym if preset_choice != "Custom ✏️" else td_sym,
                                td_interval,
                                ld_candles,
                                twelve_data_key,
                            )
                            st.session_state["ld_source"] = "Twelve Data 🟢 Real-time"
                        else:
                            # ── yfinance fallback (delayed) ───
                            import pandas as pd
                            raw = yf.download(
                                yf_sym if preset_choice != "Custom ✏️" else yf_sym,
                                period=yf_period,
                                interval=yf_interval,
                                auto_adjust=True,
                                progress=False,
                            )
                            if raw.empty:
                                raise ValueError(f"No data for `{yf_sym}`. Check the symbol.")
                            # Flatten MultiIndex if present
                            if hasattr(raw.columns, "levels"):
                                raw.columns = raw.columns.get_level_values(0)
                            # Resample H4
                            if ld_tf == "H4":
                                raw = raw.resample("4h").agg({
                                    "Open": "first", "High": "max",
                                    "Low": "min", "Close": "last",
                                    "Volume": "sum",
                                }).dropna()
                            raw.index = raw.index.tz_localize(None) if raw.index.tzinfo else raw.index
                            df_raw = raw.tail(ld_candles).copy()
                            st.session_state["ld_source"] = "yfinance 🟡 15-min delayed"

                        st.session_state["ld_df"]     = df_raw
                        st.session_state["ld_symbol"] = preset_choice
                        st.session_state["ld_tf_sel"] = ld_tf

                    except Exception as _fe:
                        st.error(f"Fetch error: {_fe}")

        # ── Display chart + analysis if data is loaded ────────
        if "ld_df" in st.session_state:
            df         = st.session_state["ld_df"]
            sym_label  = st.session_state.get("ld_symbol", ticker_sym)
            tf_label   = st.session_state.get("ld_tf_sel", ld_tf)

            # ── Stats bar ─────────────────────────────────────
            last_close  = float(df["Close"].iloc[-1])
            prev_close  = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
            chg         = last_close - prev_close
            chg_pct     = (chg / prev_close * 100) if prev_close else 0
            high_val    = float(df["High"].max())
            low_val     = float(df["Low"].min())
            chg_color   = "#10b981" if chg >= 0 else "#ef4444"
            arrow       = "▲" if chg >= 0 else "▼"

            st.markdown(f"""
<div style='display:flex;gap:20px;background:#1e293b;border-radius:10px;padding:14px 20px;margin:10px 0;flex-wrap:wrap'>
  <div><span style='color:#94a3b8;font-size:12px'>Symbol</span><br>
       <span style='color:#f1f5f9;font-weight:700;font-size:18px'>{sym_label}</span>
       <span style='color:#94a3b8;font-size:11px;margin-left:6px'>{tf_label} · {len(df)} candles · {st.session_state.get("ld_source","")}</span></div>
  <div><span style='color:#94a3b8;font-size:12px'>Last Price</span><br>
       <span style='color:#f1f5f9;font-weight:700;font-size:18px'>{last_close:.5g}</span></div>
  <div><span style='color:#94a3b8;font-size:12px'>Change</span><br>
       <span style='color:{chg_color};font-weight:700;font-size:18px'>{arrow} {abs(chg):.5g} ({chg_pct:+.2f}%)</span></div>
  <div><span style='color:#94a3b8;font-size:12px'>Period High</span><br>
       <span style='color:#10b981;font-weight:600;font-size:16px'>{high_val:.5g}</span></div>
  <div><span style='color:#94a3b8;font-size:12px'>Period Low</span><br>
       <span style='color:#ef4444;font-weight:600;font-size:16px'>{low_val:.5g}</span></div>
</div>
""", unsafe_allow_html=True)

            # ── Interactive Plotly candlestick chart ──────────
            fig = go.Figure(data=[go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color="#10b981",
                decreasing_line_color="#ef4444",
                increasing_fillcolor="#10b981",
                decreasing_fillcolor="#ef4444",
                name="Price",
            )])

            # Add 20 & 50 EMA overlays
            ema20 = df["Close"].ewm(span=20, adjust=False).mean()
            ema50 = df["Close"].ewm(span=50, adjust=False).mean()
            fig.add_trace(go.Scatter(
                x=df.index, y=ema20, name="EMA 20",
                line=dict(color="#fbbf24", width=1.2), opacity=0.8,
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=ema50, name="EMA 50",
                line=dict(color="#818cf8", width=1.2), opacity=0.8,
            ))

            # Volume bars at bottom
            vol_colors = ["#10b981" if c >= o else "#ef4444"
                          for c, o in zip(df["Close"], df["Open"])]
            fig.add_trace(go.Bar(
                x=df.index, y=df["Volume"],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.35,
                yaxis="y2",
            ))

            fig.update_layout(
                title=dict(text=f"{sym_label} — {tf_label}", font=dict(color="#f1f5f9", size=15)),
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                font=dict(color="#94a3b8"),
                xaxis=dict(
                    gridcolor="#1e293b", showgrid=True,
                    rangeslider=dict(visible=False),
                    color="#94a3b8",
                ),
                yaxis=dict(gridcolor="#1e293b", showgrid=True, color="#94a3b8", side="right"),
                yaxis2=dict(overlaying="y", side="left", showgrid=False,
                            color="#475569", showticklabels=False),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
                height=480,
                margin=dict(l=10, r=60, t=40, b=10),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── AI Analysis section ───────────────────────────
            st.markdown("#### 🤖 AI Analysis")
            ai_col1, ai_col2 = st.columns([3, 1])
            with ai_col1:
                ld_extra_context = st.text_input(
                    "💬 Add context (optional)",
                    placeholder="e.g. Near daily resistance, news tonight...",
                    key="ld_extra_ctx",
                )
            with ai_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                analyse_btn = st.button("🧠 Run AI Analysis", use_container_width=True, key="ld_analyse_btn")

            if analyse_btn:
                if not api_key:
                    st.warning("👈 Enter your API key in the sidebar first.")
                else:
                    with st.spinner("Generating chart image and running AI analysis..."):
                        try:
                            # ── Generate static chart image for AI using matplotlib ──
                            import matplotlib
                            matplotlib.use("Agg")
                            import matplotlib.pyplot as plt
                            import matplotlib.patches as mpatches

                            _df = df.reset_index()
                            n   = len(_df)
                            xs  = list(range(n))
                            W   = 0.4  # candle body half-width

                            fig_ai, (ax1, ax2) = plt.subplots(
                                2, 1, figsize=(16, 10),
                                gridspec_kw={"height_ratios": [4, 1]},
                                facecolor="#0f172a",
                            )
                            ax1.set_facecolor("#0f172a")
                            ax2.set_facecolor("#0f172a")

                            for i, row in _df.iterrows():
                                _o, _h, _l, _c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
                                color = "#10b981" if _c >= _o else "#ef4444"
                                # Wick
                                ax1.plot([i, i], [_l, _h], color=color, linewidth=0.8, zorder=1)
                                # Body
                                ax1.add_patch(mpatches.FancyBboxPatch(
                                    (i - W, min(_o, _c)), 2 * W, max(abs(_c - _o), 1e-9),
                                    boxstyle="square,pad=0", linewidth=0,
                                    facecolor=color, zorder=2,
                                ))

                            # EMA lines on AI chart
                            ax1.plot(xs, ema20.values, color="#fbbf24", linewidth=1.2, label="EMA20", alpha=0.85)
                            ax1.plot(xs, ema50.values, color="#818cf8", linewidth=1.2, label="EMA50", alpha=0.85)
                            ax1.legend(loc="upper left", facecolor="#1e293b",
                                       labelcolor="#f1f5f9", fontsize=9)

                            # X-axis labels — show every ~10th candle datetime
                            tick_step = max(1, n // 10)
                            tick_positions = list(range(0, n, tick_step))
                            tick_labels = [
                                str(_df.iloc[i]["Date"] if "Date" in _df.columns
                                    else _df.index[i])[:16]
                                for i in tick_positions
                            ]
                            ax1.set_xticks(tick_positions)
                            ax1.set_xticklabels(tick_labels, rotation=30, ha="right",
                                                color="#94a3b8", fontsize=7)
                            ax1.set_xlim(-1, n)
                            ax1.tick_params(colors="#94a3b8")
                            ax1.yaxis.tick_right()
                            ax1.yaxis.set_tick_params(labelcolor="#94a3b8")
                            ax1.grid(color="#1e293b", linewidth=0.5)
                            ax1.set_title(f"{sym_label}  {tf_label}  ({n} candles)",
                                          color="#f1f5f9", fontsize=13, pad=8)

                            # Volume bars
                            for i, row in _df.iterrows():
                                _o, _c = float(row["Open"]), float(row["Close"])
                                _v = float(row["Volume"]) if "Volume" in row and row["Volume"] == row["Volume"] else 0
                                color = "#10b981" if _c >= _o else "#ef4444"
                                ax2.bar(i, _v, color=color, alpha=0.5, width=0.8)
                            ax2.set_facecolor("#0f172a")
                            ax2.tick_params(colors="#94a3b8", labelsize=7)
                            ax2.yaxis.tick_right()
                            ax2.set_xlim(-1, n)
                            ax2.set_ylabel("Vol", color="#94a3b8", fontsize=8)
                            ax2.grid(color="#1e293b", linewidth=0.3)

                            plt.tight_layout(pad=0.5)

                            # Convert to PIL Image
                            _buf = io.BytesIO()
                            fig_ai.savefig(_buf, format="PNG", dpi=130,
                                           bbox_inches="tight", facecolor="#0f172a")
                            plt.close(fig_ai)
                            _buf.seek(0)
                            chart_pil = Image.open(_buf).copy()

                            # ── Run through existing AI analysis pipeline ──
                            ld_market_type = sym_label if "Custom" not in sym_label else "Financial instrument"
                            ld_context = ld_extra_context or f"Live {tf_label} data — {n} candles fetched automatically via yfinance."

                            analysis_result = analyze_chart_with_ai(
                                chart_pil, api_key, model_choice,
                                ld_market_type, tf_label, ld_context,
                            )

                            st.session_state["ld_analysis"]   = analysis_result
                            st.session_state["ld_chart_pil"]  = chart_pil

                        except Exception as _ae:
                            st.error(f"Analysis error: {_ae}")
                            import traceback; st.text(traceback.format_exc())

            # ── Show analysis result ──────────────────────────
            if "ld_analysis" in st.session_state:
                # News warning for live data
                _ld_news = get_news_warning(st.session_state.get("ld_symbol", ""))
                render_news_warning_banner(_ld_news)
                st.markdown(st.session_state["ld_analysis"])

                # Offer to annotate the chart
                annotate_col1, annotate_col2 = st.columns([1, 3])
                with annotate_col1:
                    annotate_live_btn = st.button(
                        "🎨 Annotate Chart", key="ld_annotate_btn", use_container_width=True,
                    )
                if annotate_live_btn:
                    if "ld_chart_pil" in st.session_state:
                        with st.spinner("Annotating market structure..."):
                            try:
                                _ld_meta   = parse_json_from_analysis(st.session_state["ld_analysis"])
                                _ld_signal = _ld_meta.get("signal", "WAIT").upper()
                                _ld_anns   = _ld_meta.get("annotations", [])
                                ann_img = annotate_chart(
                                    st.session_state["ld_chart_pil"],
                                    _ld_anns,
                                    _ld_signal,
                                    _ld_meta,
                                )
                                st.image(
                                    pil_to_download_bytes(ann_img),
                                    caption=f"{sym_label} {tf_label} — Market Structure Annotation",
                                    use_container_width=True,
                                )
                            except Exception as _ann_e:
                                st.error(f"Annotation error: {_ann_e}")

# ════════════════════════════════════════════════════════════
# TOOL 8 — MULTI-TIMEFRAME STRUCTURE PANEL
# ════════════════════════════════════════════════════════════
with tool_tab8:
    st.markdown("### 🔭 Multi-Timeframe Structure Panel")
    st.caption("One click → AI analyses D1 + H4 + H1 + M15 simultaneously. See if all timeframes agree before you trade.")

    if not api_key:
        st.warning("👈 Enter your API key in the sidebar first.")
    else:
        # ── Symbol selector ───────────────────────────────────
        MTF_SYMBOLS = {
            "EUR/USD":          ("EUR/USD",  "EURUSD=X"),
            "GBP/USD":          ("GBP/USD",  "GBPUSD=X"),
            "USD/JPY":          ("USD/JPY",  "USDJPY=X"),
            "AUD/USD":          ("AUD/USD",  "AUDUSD=X"),
            "NZD/USD":          ("NZD/USD",  "NZDUSD=X"),
            "USD/CAD":          ("USD/CAD",  "USDCAD=X"),
            "USD/CHF":          ("USD/CHF",  "USDCHF=X"),
            "GBP/JPY":          ("GBP/JPY",  "GBPJPY=X"),
            "EUR/JPY":          ("EUR/JPY",  "EURJPY=X"),
            "Gold (XAU/USD)":   ("XAU/USD",  "GC=F"),
            "Silver (XAG/USD)": ("XAG/USD",  "SI=F"),
            "BTC/USD":          ("BTC/USD",  "BTC-USD"),
            "ETH/USD":          ("ETH/USD",  "ETH-USD"),
            "S&P 500":          ("SPX",      "^GSPC"),
            "Nasdaq 100":       ("NDX",      "^NDX"),
        }

        # Timeframes to scan: label → (td_interval, yf_interval, yf_period, candles)
        MTF_TFS = [
            ("D1",  "1day",  "1d",  "180d", 80),
            ("H4",  "4h",    "1h",  "60d",  80),
            ("H1",  "1h",    "1h",  "30d",  80),
            ("M15", "15min", "15m", "5d",   80),
        ]

        mtf_c1, mtf_c2 = st.columns([3, 1])
        with mtf_c1:
            mtf_symbol = st.selectbox("📌 Select Symbol", list(MTF_SYMBOLS.keys()),
                                       index=0, key="mtf_symbol")
        with mtf_c2:
            st.markdown("<br>", unsafe_allow_html=True)
            mtf_run_btn = st.button("🚀 Run MTF Analysis", use_container_width=True,
                                     type="primary", key="mtf_run_btn")

        td_sym_mtf, yf_sym_mtf = MTF_SYMBOLS[mtf_symbol]

        # ── News warning for this symbol ──────────────────────
        _mtf_news = get_news_warning(mtf_symbol)
        render_news_warning_banner(_mtf_news)

        # ── Run analysis across all 4 TFs ─────────────────────
        if mtf_run_btn:
            mtf_results = {}
            prog_mtf = st.progress(0)
            stat_mtf = st.empty()

            for idx, (tf_label, td_int, yf_int, yf_period, n_candles) in enumerate(MTF_TFS):
                stat_mtf.text(f"📡 Fetching {mtf_symbol} {tf_label}... ({idx+1}/4)")
                prog_mtf.progress(idx / 4)

                try:
                    # ── Fetch data ────────────────────────────
                    import pandas as _pd_mtf
                    if twelve_data_key:
                        import requests as _rq_mtf
                        _url = "https://api.twelvedata.com/time_series"
                        _p   = {"symbol": td_sym_mtf, "interval": td_int,
                                "outputsize": n_candles, "apikey": twelve_data_key, "format": "JSON"}
                        _r   = _rq_mtf.get(_url, params=_p, timeout=15)
                        _d   = _r.json()
                        if _d.get("status") == "error":
                            raise ValueError(_d.get("message", "Twelve Data error"))
                        _rows = [{"Datetime": v["datetime"],
                                  "Open": float(v["open"]), "High": float(v["high"]),
                                  "Low": float(v["low"]), "Close": float(v["close"]),
                                  "Volume": float(v.get("volume", 0))}
                                 for v in _d.get("values", [])]
                        df_mtf = _pd_mtf.DataFrame(_rows)
                        df_mtf["Datetime"] = _pd_mtf.to_datetime(df_mtf["Datetime"])
                        df_mtf = df_mtf.sort_values("Datetime").set_index("Datetime")
                    else:
                        import yfinance as _yf_mtf
                        _raw = _yf_mtf.download(yf_sym_mtf, period=yf_period,
                                                interval=yf_int, auto_adjust=True, progress=False)
                        if _raw.empty:
                            raise ValueError(f"No data for {yf_sym_mtf}")
                        if hasattr(_raw.columns, "levels"):
                            _raw.columns = _raw.columns.get_level_values(0)
                        if tf_label == "H4":
                            _raw = _raw.resample("4h").agg({"Open":"first","High":"max",
                                                             "Low":"min","Close":"last","Volume":"sum"}).dropna()
                        _raw.index = _raw.index.tz_localize(None) if _raw.index.tzinfo else _raw.index
                        df_mtf = _raw.tail(n_candles).copy()

                    # ── Generate chart image ──────────────────
                    chart_pil_mtf = generate_chart_image_from_df(df_mtf, mtf_symbol, tf_label)

                    # ── AI quick scan ─────────────────────────
                    _last_mtf = float(df_mtf["Close"].iloc[-1])
                    _qp_mtf = f"""Analyse this {mtf_symbol} chart on the {tf_label} timeframe. Price: {_last_mtf:.5g}.
Output ONLY this JSON — nothing else:
{{"signal": "BUY" or "SELL" or "WAIT",
  "confidence": 1-10,
  "trend": "Strongly Bullish" or "Bullish" or "Neutral" or "Bearish" or "Strongly Bearish",
  "structure": "one sentence — what is the dominant market structure right now?",
  "key_level": "the single most important price level right now",
  "pattern": "chart pattern name or None",
  "action": "what should a trader watch for on this timeframe? one sentence"}}"""

                    _qa_mtf = analyze_chart_with_ai(chart_pil_mtf, api_key, model_choice,
                                                     mtf_symbol, tf_label, _qp_mtf)
                    _jm = re.search(r'\{.*\}', _qa_mtf, re.DOTALL)
                    try:
                        _data_mtf = json.loads(_jm.group()) if _jm else {}
                    except Exception:
                        _data_mtf = {}
                    if not _data_mtf.get("signal"):
                        _data_mtf = {"signal": "WAIT", "confidence": 5, "trend": "Neutral",
                                     "structure": "Could not parse AI response", "key_level": "—",
                                     "pattern": "None", "action": "—"}
                    _data_mtf["last_price"] = f"{_last_mtf:.5g}"
                    _data_mtf["tf"]         = tf_label
                    mtf_results[tf_label]   = _data_mtf

                except Exception as _mtf_e:
                    mtf_results[tf_label] = {
                        "signal": "ERROR", "confidence": 0, "trend": "—",
                        "structure": str(_mtf_e)[:80], "key_level": "—",
                        "pattern": "—", "action": "—",
                        "last_price": "—", "tf": tf_label,
                    }

            prog_mtf.progress(1.0)
            stat_mtf.text("✅ MTF analysis complete!")
            st.session_state["mtf_panel_results"] = mtf_results
            st.session_state["mtf_panel_symbol"]  = mtf_symbol

        # ── Display MTF results ───────────────────────────────
        if "mtf_panel_results" in st.session_state:
            res       = st.session_state["mtf_panel_results"]
            sym_shown = st.session_state.get("mtf_panel_symbol", mtf_symbol)

            # ── Confluence summary banner ─────────────────────
            signals   = [r.get("signal","WAIT") for r in res.values() if r.get("signal") not in ("ERROR","—")]
            buys      = signals.count("BUY")
            sells     = signals.count("SELL")
            waits     = signals.count("WAIT")
            avg_conf  = sum(r.get("confidence",0) for r in res.values() if isinstance(r.get("confidence"),int)) / max(len([r for r in res.values() if isinstance(r.get("confidence"),int)]),1)

            if buys >= 3:
                conf_bg   = "#0a2e1a"
                conf_bdr  = "#22c55e"
                conf_txt_color = "#4ade80"
                conf_icon = "🟢"
                conf_label = "STRONG BUY"
                conf_detail = f"{buys}/4 Timeframes Bullish — High-probability long setup"
            elif sells >= 3:
                conf_bg   = "#2e0a0a"
                conf_bdr  = "#ef4444"
                conf_txt_color = "#f87171"
                conf_icon = "🔴"
                conf_label = "STRONG SELL"
                conf_detail = f"{sells}/4 Timeframes Bearish — High-probability short setup"
            elif buys >= 2 and sells == 0:
                conf_bg   = "#0a1f12"
                conf_bdr  = "#86efac"
                conf_txt_color = "#86efac"
                conf_icon = "🟡"
                conf_label = "CAUTIOUS BUY"
                conf_detail = f"{buys}/4 TFs Bullish — Wait for M15 confirmation before entering"
            elif sells >= 2 and buys == 0:
                conf_bg   = "#1f0a0a"
                conf_bdr  = "#fca5a5"
                conf_txt_color = "#fca5a5"
                conf_icon = "🟡"
                conf_label = "CAUTIOUS SELL"
                conf_detail = f"{sells}/4 TFs Bearish — Wait for M15 confirmation before entering"
            else:
                conf_bg   = "#0f0f1a"
                conf_bdr  = "#6366f1"
                conf_txt_color = "#a5b4fc"
                conf_icon = "⏳"
                conf_label = "NO CONFLUENCE — WAIT"
                conf_detail = "Timeframes are mixed. No high-probability setup right now. Stay patient."

            st.markdown(f"""
<div style='background:{conf_bg};border:2px solid {conf_bdr};border-radius:16px;
padding:22px 28px;margin:16px 0 24px 0;text-align:center;box-shadow:0 4px 24px rgba(0,0,0,0.4)'>
  <div style='font-size:13px;color:#94a3b8;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px'>
    {sym_shown} · MTF Confluence
  </div>
  <div style='font-size:26px;font-weight:900;color:{conf_txt_color};margin-bottom:8px'>
    {conf_icon} {conf_label}
  </div>
  <div style='font-size:14px;color:#e2e8f0;margin-bottom:12px'>{conf_detail}</div>
  <div style='display:flex;justify-content:center;gap:28px;flex-wrap:wrap'>
    <span style='background:#14532d;color:#86efac;padding:5px 16px;border-radius:20px;font-size:13px;font-weight:700'>▲ BUY: {buys}</span>
    <span style='background:#7f1d1d;color:#fca5a5;padding:5px 16px;border-radius:20px;font-size:13px;font-weight:700'>▼ SELL: {sells}</span>
    <span style='background:#1e293b;color:#94a3b8;padding:5px 16px;border-radius:20px;font-size:13px;font-weight:700'>⏳ WAIT: {waits}</span>
    <span style='background:#1e1b4b;color:#a5b4fc;padding:5px 16px;border-radius:20px;font-size:13px;font-weight:700'>⚡ Avg Conf: {avg_conf:.1f}/10</span>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── 4 TF cards in 2×2 grid ────────────────────────
            row1 = st.columns(2)
            row2 = st.columns(2)
            grid = [("D1", row1[0]), ("H4", row1[1]), ("H1", row2[0]), ("M15", row2[1])]
            tf_icons = {"D1": "📅", "H4": "🕓", "H1": "🕐", "M15": "⚡"}

            for tf_lbl, col_cell in grid:
                r = res.get(tf_lbl, {})
                sig   = r.get("signal", "WAIT")
                conf  = r.get("confidence", 0)
                trend = r.get("trend", "—")
                struc = r.get("structure", "—")
                klvl  = r.get("key_level", "—")
                pat   = r.get("pattern", "None")
                act   = r.get("action", "—")
                price = r.get("last_price", "—")
                tf_ico = tf_icons.get(tf_lbl, "")

                if sig == "BUY":
                    sig_bg = "#14532d"; sig_txt = "#4ade80"; sig_bdr = "#22c55e"; sig_ico = "▲"
                elif sig == "SELL":
                    sig_bg = "#7f1d1d"; sig_txt = "#fca5a5"; sig_bdr = "#ef4444"; sig_ico = "▼"
                elif sig == "ERROR":
                    sig_bg = "#1c1917"; sig_txt = "#f59e0b"; sig_bdr = "#78716c"; sig_ico = "⚠"
                else:
                    sig_bg = "#1e293b"; sig_txt = "#94a3b8"; sig_bdr = "#475569"; sig_ico = "⏳"

                conf_int = int(conf) if isinstance(conf, (int, float)) else 0
                bar_c = "#22c55e" if conf_int >= 7 else ("#f59e0b" if conf_int >= 5 else "#ef4444")

                trend_map = {
                    "Strongly Bullish": ("🔼", "#4ade80"),
                    "Bullish":          ("▲",  "#86efac"),
                    "Neutral":          ("➡",  "#fbbf24"),
                    "Bearish":          ("▼",  "#f87171"),
                    "Strongly Bearish": ("🔽", "#ef4444"),
                }
                trend_ico, trend_col = trend_map.get(trend, ("—", "#94a3b8"))

                has_pat = pat and pat.lower() not in ("none", "—", "")
                pat_row = f"<div style='background:#1e1b4b;border-radius:6px;padding:5px 10px;margin-top:6px;font-size:12px;color:#c7d2fe'>📐 <b>{pat}</b></div>" if has_pat else ""

                with col_cell:
                    st.markdown(f"""
<div style='background:#0f172a;border:2px solid {sig_bdr};border-radius:14px;
padding:18px;margin:6px 0;box-shadow:0 2px 12px rgba(0,0,0,0.5)'>

  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px'>
    <div>
      <span style='font-size:22px;font-weight:900;color:#f1f5f9'>{tf_ico} {tf_lbl}</span>
      <div style='font-size:11px;color:#64748b;margin-top:1px'>Price: <b style='color:#fbbf24'>{price}</b></div>
    </div>
    <span style='background:{sig_bg};color:{sig_txt};border:1px solid {sig_bdr};
    padding:6px 16px;border-radius:20px;font-weight:800;font-size:15px'>
      {sig_ico} {sig}
    </span>
  </div>

  <div style='margin-bottom:10px'>
    <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
      <span style='color:#94a3b8;font-size:12px'>Confidence</span>
      <b style='color:{bar_c};font-size:12px'>{conf_int}/10</b>
    </div>
    <div style='background:#1e293b;border-radius:6px;height:10px'>
      <div style='background:{bar_c};width:{conf_int*10}%;height:10px;border-radius:6px;
      transition:width 0.3s'></div>
    </div>
  </div>

  <div style='background:#1e293b;border-radius:8px;padding:10px 12px;margin-bottom:8px'>
    <div style='font-size:12px;color:#64748b;margin-bottom:4px'>TREND</div>
    <div style='font-size:13px;font-weight:700;color:{trend_col}'>{trend_ico} {trend}</div>
  </div>

  <div style='font-size:12px;color:#cbd5e1;margin:6px 0;line-height:1.5'>
    🏗️ {struc}
  </div>

  <div style='background:#1e293b;border-radius:6px;padding:6px 10px;margin-top:6px;
  font-size:12px;color:#e2e8f0'>
    🎯 Key Level: <b style='color:#a78bfa'>{klvl}</b>
  </div>

  {pat_row}

  <div style='border-top:1px solid #1e293b;margin-top:10px;padding-top:8px;
  font-size:11px;color:#64748b;line-height:1.5'>
    👁️ {act}
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Trading decision guide ────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
<div style='background:#0a0f1e;border:1px solid #1e3a5f;border-radius:14px;padding:20px 24px;margin-top:4px'>
  <div style='font-size:15px;font-weight:700;color:#fbbf24;margin-bottom:14px'>
    📋 How to Read This Panel &nbsp;·&nbsp; 如何使用
  </div>
  <table style='width:100%;border-collapse:collapse'>
    <tr>
      <td style='padding:8px 12px;border-bottom:1px solid #1e293b;color:#f1f5f9;font-weight:700;width:60px'>D1</td>
      <td style='padding:8px 12px;border-bottom:1px solid #1e293b;color:#94a3b8;font-size:13px'>Overall <b style='color:#4ade80'>Bias</b> — only trade in this direction. 大方向判断.</td>
    </tr>
    <tr>
      <td style='padding:8px 12px;border-bottom:1px solid #1e293b;color:#f1f5f9;font-weight:700'>H4</td>
      <td style='padding:8px 12px;border-bottom:1px solid #1e293b;color:#94a3b8;font-size:13px'>Trend <b style='color:#4ade80'>Structure</b> — look for pullbacks to key levels. 趋势结构.</td>
    </tr>
    <tr>
      <td style='padding:8px 12px;border-bottom:1px solid #1e293b;color:#f1f5f9;font-weight:700'>H1</td>
      <td style='padding:8px 12px;border-bottom:1px solid #1e293b;color:#94a3b8;font-size:13px'>Entry <b style='color:#4ade80'>Zone</b> — BOS/CHoCH forming here = setup confirmed. 入场区域.</td>
    </tr>
    <tr>
      <td style='padding:8px 12px;color:#f1f5f9;font-weight:700'>M15</td>
      <td style='padding:8px 12px;color:#94a3b8;font-size:13px'>Entry <b style='color:#4ade80'>Trigger</b> — precise timing only. Wait for confirmation candle. 精确入场.</td>
    </tr>
  </table>
  <div style='margin-top:14px;background:#1a0a0a;border:1px solid #ef4444;border-radius:8px;
  padding:10px 14px;font-size:13px;color:#fca5a5'>
    ⚡ <b>Golden Rule:</b> Only enter when D1 + H4 + H1 ALL agree on direction. Use M15 for timing only.<br>
    <span style='color:#86efac;font-size:12px'>黄金法则：只有D1+H4+H1三个时间框架方向一致时才进场，M15仅用于确定入场时机。</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer**: TradingAI Analyst is for **educational and informational purposes only**. "
    "It does NOT constitute financial advice. Trading involves substantial risk of loss. "
    "Always conduct your own research and manage your risk responsibly."
)
