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

**Order Blocks (OB)**
- Bullish OB: Last bearish (red/down) candle before a strong bullish move. Price often returns here for retesting.
- Bearish OB: Last bullish (green/up) candle before a strong bearish move.
- Best OBs: Have high volume, sharp impulse away, and haven't been revisited multiple times.

**Fair Value Gap (FVG) / Imbalance**
- Bullish FVG: Gap between candle[1] HIGH and candle[3] LOW in a 3-candle bullish sequence.
- Bearish FVG: Gap between candle[1] LOW and candle[3] HIGH in a 3-candle bearish sequence.
- Price tends to retrace into FVG to fill it before continuing.
- Key: Wait for price to reach 50% of FVG, look for rejection/confirmation there.

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
**Breaker Blocks**: Failed order blocks that flip to opposite polarity.
**Mitigation Block**: OB that has been partially tested.
**Propulsion Block**: Strong engulfing candle that causes a structural break.
**Power of 3 (PO3)**: Accumulation → Manipulation (fake move) → Distribution (real move).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 CONFLUENCE SCORING SYSTEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Score each trade setup from 0–10 based on number of confluences:
- Trend alignment (HTF + LTF) = +2
- Pattern confirmation = +1.5
- Fibonacci level (61.8%/78.6% OTE) = +1.5
- Order Block present = +1
- FVG present = +1
- S&R level = +1
- RSI divergence/extreme = +0.5
- MACD confirmation = +0.5
- Liquidity sweep before entry = +1

Score 7+/10 = High confidence trade
Score 5-6/10 = Moderate confidence (trade with caution)
Score <5/10 = Skip or wait

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📏 RISK MANAGEMENT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Minimum R:R = 1:2 (risk $1 to make $2)
- Ideal R:R = 1:3 or better
- Never risk more than 1-2% of account per trade
- SL placement: Beyond last swing point, OB, or key level
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
[EN] Order Blocks, FVGs, BOS/CHoCH, liquidity sweeps visible? 1-2 lines.
[中文] 可见的订单块、公允价值缺口、结构突破/变化、流动性扫描？1-2行。

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

PURPOSE: Show what YOU SEE on the chart — trend structure, zones, and key levels.
Do NOT draw SL / TP / Entry points. This is a market structure analysis overlay, not a trade setup.

MAXIMUM 7 ANNOTATIONS TOTAL.

ANNOTATION TYPES TO USE:

1. "structure_break" — for BOS (Break of Structure) or CHoCH (Change of Character)
   - Required fields: y_position, color, label, direction ("bullish" or "bearish")
   - color: "teal" for BOS (trend continuation), "orange" for CHoCH (reversal signal)
   - label: "BOS ↑ 结构突破" or "CHoCH ↓ 结构变化"

2. "zone_box" — for Order Blocks, FVGs, Liquidity zones, S/R zones
   - Required: y_start, y_end, color, label
   - color rules: green=Bullish OB/Demand, red=Bearish OB/Supply, blue=FVG, yellow=Liquidity
   - label examples: "Bullish OB 看涨OB", "FVG 缺口", "Supply Zone", "Liquidity 流动性"

3. "horizontal_line" — for key S/R levels, equal highs/lows, swing points
   - color: green=support, red=resistance, yellow=equal highs/lows

4. "diagonal_line" — for active trendlines (still being respected by price)
   - x1/x2 range: 0.05 to 0.82 (never 0.0 or 1.0)
   - color: white=main trend, purple=secondary channel

5. "fibonacci" — ONLY if there is a clean, obvious swing high-to-low visible
   - swing_high_y, swing_low_y positions

6. "pattern_triangle" / "pattern_flag" — ONLY if pattern is still FORMING (price is inside it)

COLOUR CONVENTION (match the legend on the chart):
  green  = Bullish OB / Demand zone / Support
  red    = Bearish OB / Supply zone / Resistance
  blue   = FVG / Imbalance
  yellow = Liquidity zone / Equal H&L
  teal   = BOS (Break of Structure)
  orange = CHoCH (Change of Character)
  white  = Trendlines

LABEL RULES — SHORT labels only (under 22 characters):
  - "Bullish OB 看涨OB" ✅    "This is a very important bullish order block" ❌
  - "FVG 缺口" ✅             "Fair Value Gap Imbalance Zone" ❌
  - "BOS ↑ 结构突破" ✅       "Break of Structure Upward" ❌
  - "CHoCH ↓ 变化" ✅
  - "Support 支撑" ✅
  - "Resistance 阻力" ✅
  - "Liquidity 流动性" ✅
  - "Equal Lows 平底" ✅

WHAT TO LOOK FOR (in priority order):
  1. BOS / CHoCH — where did market structure change? Mark these first.
  2. Order Blocks — last bearish candle before bullish impulse (Bullish OB), last bullish candle before bearish impulse (Bearish OB)
  3. FVG — gaps between candle 1 high and candle 3 low (or vice versa) that haven't been filled
  4. Key S/R — obvious swing highs/lows that price has respected multiple times
  5. Liquidity zones — equal highs or equal lows where stops are likely resting
  6. Trendline — only if price has touched it 2+ times and it's still active

For y positions use: "top"(0.06), "upper_quarter"(0.20), "upper_third"(0.30), "middle"(0.50), "lower_third"(0.65), "lower_quarter"(0.78), "bottom"(0.93)

Example — bullish structure with key zones:
```json
{{
  "signal": "BUY",
  "confidence": 7,
  "pattern_name": "ACCUMULATION — MARKUP PHASE",
  "annotations": [
    {{"type": "structure_break", "y_position": "upper_third", "color": "teal", "label": "BOS ↑ 结构突破", "direction": "bullish"}},
    {{"type": "zone_box", "y_start": "lower_third", "y_end": "middle", "color": "green", "label": "Bullish OB 看涨OB"}},
    {{"type": "zone_box", "y_start": "upper_third", "y_end": "upper_quarter", "color": "blue", "label": "FVG 缺口"}},
    {{"type": "horizontal_line", "y_position": "upper_quarter", "color": "red", "label": "Resistance 阻力"}},
    {{"type": "horizontal_line", "y_position": "lower_third", "color": "green", "label": "Support 支撑"}},
    {{"type": "zone_box", "y_start": "lower_quarter", "y_end": "bottom", "color": "yellow", "label": "Liquidity 流动性"}}
  ]
}}
```

Example — bearish structure with CHoCH:
```json
{{
  "signal": "SELL",
  "confidence": 6,
  "pattern_name": "DISTRIBUTION — MARKDOWN PHASE",
  "annotations": [
    {{"type": "structure_break", "y_position": "middle", "color": "orange", "label": "CHoCH ↓ 结构变化", "direction": "bearish"}},
    {{"type": "zone_box", "y_start": "upper_third", "y_end": "middle", "color": "red", "label": "Bearish OB 看跌OB"}},
    {{"type": "zone_box", "y_start": "middle", "y_end": "lower_third", "color": "blue", "label": "FVG 缺口"}},
    {{"type": "horizontal_line", "y_position": "upper_third", "color": "red", "label": "Supply Zone 供给"}},
    {{"type": "zone_box", "y_start": "top", "y_end": "upper_quarter", "color": "yellow", "label": "Equal Highs 平顶"}},
    {{"type": "diagonal_line", "x1": 0.05, "y1": "upper_quarter", "x2": 0.80, "y2": "middle", "color": "white", "label": "Downtrend 下降趋势"}}
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
    """Draw market structure annotations — OB / FVG / BOS / CHoCH / S&R / Liquidity."""
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
    #   green  = Bullish OB / Demand zone / Support
    #   red    = Bearish OB / Supply zone / Resistance
    #   blue   = FVG / Imbalance / Fair Value Gap
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

    # ── Enforce max 8 annotations (structure needs more room) ──
    PRIORITY = {
        "structure_break": 0,   # BOS / CHoCH — most important context
        "horizontal_line": 1,   # S/R levels
        "zone_box":        2,   # OB / FVG / liquidity zones
        "diagonal_line":   3,   # trendlines
        "fibonacci":       4,
        "pattern_triangle":5,
        "pattern_flag":    5,
        "pattern_hs":      5,
        "pattern_double":  5,
        "dashed_line":     6,
        "entry_arrow":     7,
        "pattern_label":   8,
    }
    sorted_anns = sorted(annotations, key=lambda a: PRIORITY.get(a.get("type", ""), 9))
    annotations = sorted_anns[:8]

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

        # ── Zone box: OB / FVG / Liquidity / S-R zone ─────
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
    # COLOUR LEGEND STRIP — top-left, shows what each colour means
    # ═══════════════════════════════════════════════════════
    legend_items = [
        ("green",  "Bullish OB/Demand"),
        ("red",    "Bearish OB/Supply"),
        ("blue",   "FVG/Imbalance"),
        ("yellow", "Liquidity Zone"),
        ("teal",   "BOS"),
        ("orange", "CHoCH"),
    ]
    leg_x, leg_y = 10, 10
    leg_sw = max(10, int(w / 140))   # colour swatch width
    leg_h  = fs_xs + 8
    for lc, lt in legend_items:
        r2, g2, b2 = C[lc]
        # Colour swatch
        draw.rectangle([leg_x, leg_y, leg_x + leg_sw, leg_y + leg_h],
                       fill=(r2, g2, b2, 220))
        # Text
        draw.text((leg_x + leg_sw + 5, leg_y + 2), lt,
                  fill=(r2, g2, b2, 230), font=font_xs)
        try:
            bbox = font_xs.getbbox(lt)
            tw   = bbox[2] - bbox[0]
        except Exception:
            tw = len(lt) * 9
        leg_x += leg_sw + tw + 18

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
            placeholder="Gemini: AIza...   or   Claude: sk-ant-...",
            help="Gemini key from aistudio.google.com (FREE) or Claude key from console.anthropic.com (paid)",
        )

    model_choice = st.selectbox(
        "🤖 AI Model",
        [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "claude-opus-4-5",
            "claude-sonnet-4-5",
        ],
        index=0,
        help="✅ Gemini models = FREE  |  Claude models = paid",
    )

    st.divider()

    market_type = st.selectbox(
        "📈 Market / Instrument",
        [
            "Forex (EUR/USD, GBP/USD, etc.)",
            "Gold (XAUUSD)",
            "Silver (XAGUSD)",
            "BTC/USD (Bitcoin)",
            "ETH/USD (Ethereum)",
            "Other Crypto",
            "US Stocks",
            "Index (S&P500, Nasdaq, etc.)",
            "Oil (WTI/Brent)",
        ],
    )

    timeframe = st.selectbox(
        "⏱️ Timeframe",
        ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"],
        index=4,
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
        ("🏦", "Smart Money Concepts (SMC)"),
        ("📐", "Fibonacci Retracement/Extension"),
        ("🎯", "Support & Resistance (SNR)"),
        ("📦", "Fair Value Gaps (FVG)"),
        ("⚡", "BOS / CHoCH Structure"),
        ("📊", "RSI & MACD Signals"),
        ("🕯️", "23 Chart Patterns"),
        ("💧", "Liquidity Analysis"),
    ]
    for icon, name in strategies:
        st.markdown(f"{icon} {name}")

    st.divider()
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
                         caption="Market Structure Analysis — OB / FVG / BOS / CHoCH / S&R / Liquidity",
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

        else:
            st.markdown("""
<div class="info-box">
<p style="color:#5b21b6;text-align:center;margin-top:40px;font-size:17px;font-weight:700">
📊 Analysis results will appear here after you upload a chart and click Analyse.
</p>
<br>
<p style="color:#1e40af;text-align:center;font-size:14px;font-weight:600">
The AI will identify:<br><br>
🔵 Trend &nbsp;·&nbsp; 🟣 Chart Patterns &nbsp;·&nbsp; 🟠 SMC Levels<br>
📐 Fibonacci &nbsp;·&nbsp; 📦 Order Blocks &nbsp;·&nbsp; ⚡ FVGs<br><br>
🎯 Entry &nbsp;·&nbsp; 🛑 Stop Loss &nbsp;·&nbsp; ✅ Take Profit 1 &nbsp;·&nbsp; 🚀 Take Profit 2
</p>
</div>
            """, unsafe_allow_html=True)


# ============================================================
# EXTRA TOOLS SECTION
# ============================================================
st.divider()
st.markdown("## 🛠️ Trading Tools 交易工具")

tool_tab1, tool_tab2, tool_tab3, tool_tab4, tool_tab5, tool_tab6 = st.tabs([
    "🧮 Position Size",
    "📰 News Calendar",
    "📡 Chart Scanner",
    "🤖 AI Coach",
    "📄 PDF Report",
    "💹 Currency Strength",
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
    st.caption("Upload up to 5 charts — AI ranks them by signal strength. 上传最多5张图，AI自动排名最佳机会。")

    if not api_key:
        st.warning("👈 Enter your API key in the sidebar first.")
    else:
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

                    # Quick scan prompt
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
                        # Try parse JSON
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
                    "- 「什么是 Order Block？怎么画？」\n"
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
You specialise in SMC (Smart Money Concepts), Wyckoff Method, Price Action, ICT concepts, Scalping, Day Trading, and Risk Management.
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
                        except Exception as e:
                            st.error(f"Error: {str(e)}")

        if len(st.session_state.get("coach_messages", [])) > 2:
            if st.button("🗑️ Clear Chat 清除对话", key="clear_coach"):
                st.session_state["coach_messages"] = [st.session_state["coach_messages"][0]]
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
                    "- 我认为中间那根大阴线是 Bearish OB\n"
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
                        "Order Blocks & FVG",
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
You specialise in SMC, Wyckoff, ICT, Price Action, Order Blocks, FVG, BOS/CHoCH, Liquidity, and Scalping.

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
                              "coach_review_system", "coach_active_focus"]:
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
                        st.markdown(msg["content"])

            # ── Follow-up chat input ───────────────────────
            followup = st.chat_input(
                "继续追问导师... / Ask a follow-up question...",
                key="coach_review_followup",
            )

            if followup:
                if not api_key:
                    st.warning("👈 请先在侧边栏输入 API Key")
                else:
                    # Add user message to history and display it
                    conv.append({"role": "user", "content": followup})
                    st.session_state["coach_review_conv"] = conv
                    with st.chat_message("user"):
                        st.markdown(followup)

                    with st.chat_message("assistant"):
                        with st.spinner("导师思考中... Thinking..."):
                            try:
                                review_sys  = st.session_state.get("coach_review_system", "")
                                img_b64_key = st.session_state.get("coach_review_img_b64", "")

                                if model_choice.startswith("gemini"):
                                    client_coach = google_genai.Client(api_key=api_key)
                                    # Build full history text for Gemini (no native multi-turn with image)
                                    history_text = "\n\n".join([
                                        f"{'Student' if m['role']=='user' else 'Coach'}:\n{m['content']}"
                                        for m in conv
                                    ])
                                    full_q = review_sys + "\n\n---\nConversation so far:\n" + history_text
                                    # Re-include image bytes for context
                                    img_buf_fu = io.BytesIO(st.session_state.get("coach_review_img_bytes", b""))
                                    img_fu_bytes = img_buf_fu.getvalue()
                                    if img_fu_bytes:
                                        coach_resp_fu = client_coach.models.generate_content(
                                            model=model_choice,
                                            contents=[
                                                full_q,
                                                google_types.Part.from_bytes(data=img_fu_bytes, mime_type="image/png"),
                                            ],
                                        )
                                    else:
                                        coach_resp_fu = client_coach.models.generate_content(
                                            model=model_choice, contents=[full_q])
                                    followup_answer = coach_resp_fu.text

                                else:
                                    # Claude: first message carries the image, rest are plain text
                                    client_coach = anthropic.Anthropic(api_key=api_key)
                                    api_msgs = []
                                    for i, m in enumerate(conv):
                                        if i == 0 and m["role"] == "user" and img_b64_key:
                                            # First user message: include image
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
                                        else:
                                            api_msgs.append({"role": m["role"], "content": m["content"]})

                                    coach_resp_fu = client_coach.messages.create(
                                        model=model_choice,
                                        max_tokens=1800,
                                        system=review_sys,
                                        messages=api_msgs[-20:],  # keep last 20 turns
                                    )
                                    followup_answer = coach_resp_fu.content[0].text

                                st.markdown(followup_answer)
                                conv.append({"role": "assistant", "content": followup_answer})
                                st.session_state["coach_review_conv"] = conv

                            except Exception as e:
                                st.error(f"Error: {str(e)}")

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


# ── Footer ─────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer**: TradingAI Analyst is for **educational and informational purposes only**. "
    "It does NOT constitute financial advice. Trading involves substantial risk of loss. "
    "Always conduct your own research and manage your risk responsibly."
)
