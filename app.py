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
Now output the drawing instructions as JSON to annotate the chart.

IMPORTANT RULES FOR ANNOTATIONS:
1. Choose ONLY 3-6 annotations that are clearly visible and relevant to THIS specific chart.
2. Do NOT always add Fibonacci — only include it if there is a clear swing high-to-low to measure.
3. Do NOT add annotations just to fill space — fewer, accurate annotations are better.
4. ALWAYS include: Entry arrow, SL line, TP1 line, TP2 line.
5. CRITICAL — COMPLETED PATTERNS: If a pattern (triangle, flag, H&S etc.) has ALREADY broken out (price has moved clearly beyond the pattern boundary), do NOT draw the pattern shape. Instead draw a "zone_box" at the breakout level and label it "✅ Breakout Zone 突破位" or a "horizontal_line" at the broken level. Only draw pattern shapes if the pattern is STILL FORMING and price is still INSIDE it.
6. Add zone_box ONLY for clearly visible Order Blocks, FVGs, or breakout zones.
7. Add diagonal_line ONLY for clearly visible trendlines that are still relevant.

Available annotation types:
- "horizontal_line": straight line across chart
- "dashed_line": dashed line (use for Entry / SL / TP levels)
- "zone_box": shaded rectangle (use for OB, FVG, S/R zones, breakout zones)
- "diagonal_line": trendline from point to point (x1,y1 → x2,y2 as 0.0-1.0)
- "fibonacci": fib levels between swing_high_y and swing_low_y — USE ONLY IF CLEAR SWING EXISTS
- "pattern_triangle": ONLY if triangle is still FORMING and price is still inside
- "pattern_flag": ONLY if flag/pennant is still FORMING and price is still inside
- "pattern_hs": ONLY if H&S is still FORMING and not yet broken
- "entry_arrow": bold arrow at entry zone
- "pattern_label": text label for the pattern name

For y positions use: "top"(0.08), "upper_quarter"(0.22), "upper_third"(0.30), "middle"(0.50), "lower_third"(0.65), "lower_quarter"(0.78), "bottom"(0.92)

Example A — Triangle still FORMING (price still inside, draw the pattern):
```json
{{
  "signal": "WAIT",
  "confidence": 6,
  "pattern_name": "ASCENDING TRIANGLE (FORMING)",
  "annotations": [
    {{"type": "pattern_triangle", "top_y": "upper_third", "bottom_y": "middle", "color": "yellow", "label": "上升三角形成中 ASCENDING TRIANGLE"}},
    {{"type": "zone_box", "y_start": "middle", "y_end": "lower_third", "color": "orange", "label": "🏦 Bullish OB 看涨订单块"}},
    {{"type": "entry_arrow", "y_position": "upper_third", "color": "green", "label": "⚡ WAIT FOR BREAKOUT 等待突破"}},
    {{"type": "dashed_line", "y_position": "lower_quarter", "color": "red", "label": "❌ SL 止损"}},
    {{"type": "dashed_line", "y_position": "upper_quarter", "color": "lime", "label": "✅ TP1 目标1"}},
    {{"type": "dashed_line", "y_position": "top", "color": "cyan", "label": "🚀 TP2 目标2"}}
  ]
}}
```

Example B — Triangle ALREADY broken out upward (do NOT draw the triangle shape):
```json
{{
  "signal": "BUY",
  "confidence": 7,
  "pattern_name": "ASCENDING TRIANGLE (COMPLETED — BULLISH BREAKOUT)",
  "annotations": [
    {{"type": "zone_box", "y_start": "upper_third", "y_end": "middle", "color": "yellow", "label": "✅ 已突破区域 Breakout Zone"}},
    {{"type": "horizontal_line", "y_position": "upper_third", "color": "yellow", "label": "— 三角突破位 Triangle Breakout"}},
    {{"type": "zone_box", "y_start": "middle", "y_end": "lower_third", "color": "orange", "label": "🏦 Support OB 支撑订单块"}},
    {{"type": "entry_arrow", "y_position": "upper_third", "color": "green", "label": "⚡ BUY RETEST 回测买入"}},
    {{"type": "dashed_line", "y_position": "lower_quarter", "color": "red", "label": "❌ SL 止损"}},
    {{"type": "dashed_line", "y_position": "upper_quarter", "color": "lime", "label": "✅ TP1 目标1"}},
    {{"type": "dashed_line", "y_position": "top", "color": "cyan", "label": "🚀 TP2 目标2"}}
  ]
}}
```

Example for a chart with fibonacci retracement (only when a clear swing is visible):
```json
{{
  "signal": "WAIT",
  "confidence": 5,
  "pattern_name": "NO CLEAR PATTERN",
  "annotations": [
    {{"type": "fibonacci", "swing_high_y": "upper_quarter", "swing_low_y": "lower_quarter", "color": "purple"}},
    {{"type": "zone_box", "y_start": "lower_third", "y_end": "lower_quarter", "color": "blue", "label": "📦 FVG 公允缺口"}},
    {{"type": "entry_arrow", "y_position": "middle", "color": "green", "label": "⚡ WAIT FOR ENTRY 等待入场"}},
    {{"type": "dashed_line", "y_position": "bottom", "color": "red", "label": "❌ SL 止损"}},
    {{"type": "dashed_line", "y_position": "upper_third", "color": "lime", "label": "✅ TP1 目标1"}},
    {{"type": "dashed_line", "y_position": "upper_quarter", "color": "cyan", "label": "🚀 TP2 目标2"}}
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
    """Draw clean, sharp trading annotations on the chart image."""
    img = image.copy().convert("RGBA")

    # ── Scale up small images so annotations are sharp ────
    MIN_W = 1800
    w_orig, h_orig = img.size
    if w_orig < MIN_W:
        scale  = MIN_W / w_orig
        img    = img.resize((int(w_orig * scale), int(h_orig * scale)), Image.LANCZOS)

    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    # ── Font sizes — much larger for readability ──────────
    fs_b  = max(22, int(w / 65))   # bold labels
    fs_sm = max(19, int(w / 75))   # normal labels
    fs_lg = max(28, int(w / 50))   # big pattern banners
    fs_xs = max(17, int(w / 90))   # small right-edge tags

    try:
        font_b  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs_b)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      fs_sm)
        font_lg = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs_lg)
        font_xs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      fs_xs)
    except Exception:
        font_b = font_sm = font_lg = font_xs = ImageFont.load_default()

    # ── Line thickness scales with image width ────────────
    LW_MAIN  = max(4, int(w / 360))   # main lines (SL, TP, entry)
    LW_ZONE  = max(3, int(w / 450))   # zone borders
    LW_PAT   = max(4, int(w / 360))   # pattern lines
    LW_DIAG  = max(4, int(w / 360))   # trendlines
    DASH_LEN = max(18, int(w / 80))
    GAP_LEN  = max(9,  int(w / 160))

    # ── Vivid, high-contrast colour palette ───────────────
    C = {
        "red":    (255,  50,  50),
        "green":  ( 0,  230,  80),
        "blue":   ( 30, 140, 255),
        "yellow": (255, 220,   0),
        "orange": (255, 140,   0),
        "purple": (200,  60, 255),
        "white":  (255, 255, 255),
        "teal":   (  0, 230, 210),
        "pink":   (255,  60, 170),
        "lime":   (130, 255,  40),
        "cyan":   (  0, 220, 255),
    }

    def col(name, alpha=255):
        r, g, b = C.get(name, C["white"])
        return (r, g, b, alpha)

    def solid(name):
        r, g, b = C.get(name, C["white"])
        return (r, g, b, 255)

    # ── Position map ──────────────────────────────────────
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

    # ── Pill label helper (larger, bolder) ───────────────
    def pill(x, y, text, txt_col, bg_col, border_col=None, font=None):
        font = font or font_sm
        # Use font to measure text
        try:
            bbox = font.getbbox(text)
            tw = bbox[2] - bbox[0] + 18
            th = bbox[3] - bbox[1] + 12
        except Exception:
            tw = len(text) * 10 + 18
            th = 24
        draw.rectangle([x, y - th//2, x + tw, y + th//2],
                       fill=bg_col, outline=border_col or txt_col, width=2)
        draw.text((x + 9, y - th//2 + 4), text, fill=txt_col, font=font)

    # ── Right-edge label ─────────────────────────────────
    right_label_y_used = []

    def right_label(y, text, txt_col, line_col):
        label_h = fs_xs + 10
        adjusted_y = y
        for used_y in right_label_y_used:
            if abs(adjusted_y - used_y) < label_h + 4:
                adjusted_y = used_y + label_h + 5
        right_label_y_used.append(adjusted_y)
        try:
            bbox = font_xs.getbbox(text)
            tw = bbox[2] - bbox[0] + 18
        except Exception:
            tw = len(text) * 9 + 18
        half = label_h // 2
        rx = w - tw - 8
        draw.rectangle([rx, adjusted_y - half, w - 4, adjusted_y + half],
                       fill=(10, 10, 10, 230), outline=line_col, width=2)
        draw.text((rx + 8, adjusted_y - half + 3), text, fill=txt_col, font=font_xs)

    # ═══════════════════════════════════════════════════════
    # DRAW ANNOTATIONS
    # ═══════════════════════════════════════════════════════
    for ann in annotations:
        atype = ann.get("type", "")
        cname = ann.get("color", "white")
        label = ann.get("label", "")

        # ── Solid line ────────────────────────────────────
        if atype == "horizontal_line":
            y = yp(ann.get("y_position", "middle"))
            draw.line([(int(w*0.02), y), (int(w*0.98), y)], fill=col(cname), width=LW_MAIN)
            if label:
                right_label(y, label, solid(cname), col(cname))

        # ── Dashed line (SL / TP / Entry) ─────────────────
        elif atype == "dashed_line":
            y = yp(ann.get("y_position", "middle"))
            _draw_dashed_line(draw, int(w*0.02), y, int(w*0.94), y,
                              fill=col(cname), width=LW_MAIN, dash=DASH_LEN, gap=GAP_LEN)
            if label:
                right_label(y, label, solid(cname), col(cname))

        # ── Zone box (OB / FVG / S&R) ─────────────────────
        elif atype == "zone_box":
            y1 = yp(ann.get("y_start", "upper_third"))
            y2 = yp(ann.get("y_end",   "upper_quarter"))
            if y1 > y2:
                y1, y2 = y2, y1
            # Semi-transparent fill + vivid thick border
            draw.rectangle([int(w*0.02), y1, int(w*0.98), y2],
                           fill=col(cname, 65), outline=col(cname, 255), width=LW_ZONE)
            if label:
                pill(int(w*0.03) + 6, (y1+y2)//2, label, solid(cname),
                     (10, 10, 10, 220), col(cname, 255), font=font_b)

        # ── Diagonal trendline ────────────────────────────
        elif atype == "diagonal_line":
            x1 = xp(ann.get("x1", 0.05));  x2 = xp(ann.get("x2", 0.95))
            y1 = yp(ann.get("y1", "upper_third")); y2 = yp(ann.get("y2", "lower_third"))
            draw.line([(x1, y1), (x2, y2)], fill=col(cname), width=LW_DIAG)
            if label:
                mx, my = (x1+x2)//2, (y1+y2)//2
                pill(mx, my, label, solid(cname), (10, 10, 10, 220), col(cname))

        # ── Fibonacci levels ──────────────────────────────
        elif atype == "fibonacci":
            y_high = yp(ann.get("swing_high_y", "upper_quarter"))
            y_low  = yp(ann.get("swing_low_y",  "lower_quarter"))
            rng    = y_low - y_high
            fibs = [
                (0.382, "38.2%",  "cyan",   LW_ZONE),
                (0.500, "50%",    "yellow", LW_ZONE),
                (0.618, "61.8%★", "orange", LW_MAIN),  # Golden ratio — thicker
                (0.786, "78.6%",  "pink",   LW_ZONE),
            ]
            for ratio, flabel, fcol, lw in fibs:
                fy = int(y_high + rng * ratio)
                draw.line([(int(w*0.02), fy), (int(w*0.88), fy)],
                          fill=col(fcol), width=lw)
                right_label(fy, f"Fib {flabel}", solid(fcol), col(fcol))

        # ── Entry arrow ───────────────────────────────────
        elif atype == "entry_arrow":
            y  = yp(ann.get("y_position", "middle"))
            ax = int(w * 0.04)
            asz = max(20, int(w / 55))   # arrow size scales with image
            pts = [
                (ax,          y - asz//2),
                (ax + asz*2,  y - asz//2),
                (ax + asz*2,  y - asz),
                (ax + asz*3,  y),
                (ax + asz*2,  y + asz),
                (ax + asz*2,  y + asz//2),
                (ax,          y + asz//2),
            ]
            draw.polygon(pts, fill=col(cname, 240), outline=solid("white"))
            if label:
                pill(ax + asz*3 + 10, y, label, solid(cname),
                     (10, 10, 10, 230), col(cname), font=font_b)

        # ── Triangle pattern ──────────────────────────────
        elif atype == "pattern_triangle":
            top_y    = yp(ann.get("top_y",    "upper_third"))
            bottom_y = yp(ann.get("bottom_y", "lower_third"))
            apex_y   = (top_y + bottom_y) // 2
            xs, xa   = int(w*0.06), int(w*0.82)
            draw.line([(xs, top_y),    (xa, apex_y)], fill=col(cname), width=LW_PAT)
            draw.line([(xs, bottom_y), (xa, apex_y)], fill=col(cname), width=LW_PAT)
            if label:
                pill(int(w*0.30), apex_y, label, solid(cname),
                     (10, 10, 10, 230), col(cname), font=font_b)

        # ── Flag / Bear Flag / channel pattern ───────────
        elif atype == "pattern_flag":
            top_y    = yp(ann.get("top_y",    "upper_third"))
            bottom_y = yp(ann.get("bottom_y", "middle"))
            # Bear flag tilts UP (price drifting up before breakdown)
            # Bull flag tilts DOWN (price drifting down before breakout)
            is_bear = "BEAR" in label.upper() or cname in ("red", "orange", "pink")
            tilt_dir = 1 if is_bear else -1
            channel_h = abs(bottom_y - top_y)
            tilt = int(channel_h * 0.20) * tilt_dir

            x1, x2 = int(w * 0.15), int(w * 0.80)

            # Draw the flag channel (parallelogram)
            t1_y = top_y;          t2_y = top_y    + tilt
            b1_y = bottom_y;       b2_y = bottom_y + tilt

            # Filled semi-transparent channel
            pts = [(x1, t1_y), (x2, t2_y), (x2, b2_y), (x1, b1_y)]
            draw.polygon(pts, fill=col(cname, 50))

            # Bold channel border lines
            draw.line([(x1, t1_y), (x2, t2_y)], fill=col(cname), width=LW_PAT)
            draw.line([(x1, b1_y), (x2, b2_y)], fill=col(cname), width=LW_PAT)

            # Flagpole (vertical bar at left edge, going to bottom)
            pole_bot = int(h * 0.93) if is_bear else int(h * 0.10)
            draw.line([(x1, t1_y), (x1, pole_bot)], fill=col(cname, 180), width=LW_PAT - 1)

            if label:
                mid_x = (x1 + x2) // 2
                mid_y = (t1_y + b1_y) // 2 + tilt // 2
                pill(mid_x - 60, mid_y, label, solid(cname),
                     (10, 10, 10, 230), col(cname), font=font_b)

        # ── Head & Shoulders ──────────────────────────────
        elif atype == "pattern_hs":
            neck_y = yp(ann.get("neck_y", "lower_third"))
            head_y = yp(ann.get("head_y", "upper_quarter"))
            lsh_y  = yp(ann.get("lsh_y",  "upper_third"))
            rsh_y  = yp(ann.get("rsh_y",  "upper_third"))
            r = max(16, int(w / 80))
            for cx, cy, lbl in [
                (int(w*0.22), lsh_y, "LS"),
                (int(w*0.50), head_y, "H"),
                (int(w*0.78), rsh_y, "RS"),
            ]:
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline=col(cname), width=LW_ZONE)
                draw.text((cx - r//2, cy - r//2 + 2), lbl, fill=solid(cname), font=font_xs)
                draw.line([(cx, cy+r), (cx, neck_y)], fill=col(cname, 140), width=LW_ZONE - 1)
            draw.line([(int(w*0.18), neck_y), (int(w*0.82), neck_y)],
                      fill=col("yellow"), width=LW_PAT)
            pill(int(w*0.42), neck_y, "NECKLINE 颈线", solid("yellow"),
                 (10, 10, 10, 220), col("yellow"), font=font_b)

        # ── Double top/bottom ─────────────────────────────
        elif atype == "pattern_double":
            peak_y = yp(ann.get("peak_y", "upper_quarter"))
            neck_y = yp(ann.get("neck_y", "lower_third"))
            r = max(18, int(w / 70))
            for cx in [int(w*0.30), int(w*0.65)]:
                draw.ellipse([cx-r, peak_y-r, cx+r, peak_y+r],
                             outline=col(cname), width=LW_ZONE)
            draw.line([(int(w*0.22), neck_y), (int(w*0.78), neck_y)],
                      fill=col("yellow"), width=LW_PAT)
            pill(int(w*0.42), neck_y, "NECKLINE 颈线", solid("yellow"),
                 (10, 10, 10, 220), col("yellow"), font=font_b)

        # ── Pattern name banner ───────────────────────────
        elif atype == "pattern_label":
            y  = yp(ann.get("y_position", "top"))
            try:
                bbox = font_lg.getbbox(label)
                tw = bbox[2] - bbox[0] + 30
            except Exception:
                tw = len(label) * 14 + 30
            px = int(w * 0.5)
            draw.rectangle([px - tw//2, y - 20, px + tw//2, y + 20],
                           fill=(10, 10, 10, 230), outline=col(cname), width=3)
            draw.text((px - tw//2 + 15, y - 14), label, fill=solid(cname), font=font_lg)

    # ═══════════════════════════════════════════════════════
    # TRADE SUMMARY CARD (bottom-left — always shown)
    # ═══════════════════════════════════════════════════════
    cy0 = h   # default (used in crop logic below)
    card_h = 0
    if signal in ("BUY", "SELL", "WAIT"):
        sig_col  = "green" if signal == "BUY" else ("red" if signal == "SELL" else "yellow")
        sig_icon = "▲ BUY" if signal == "BUY" else ("▼ SELL" if signal == "SELL" else "⏳ WAIT")

        # Card data rows
        rows = [
            (sig_icon,                  sig_col,  True),
        ]
        for key, label_text in [
            ("pattern_name", "Pattern"),
            ("entry",        "Entry  "),
            ("sl",           "SL     "),
            ("tp1",          "TP1    "),
            ("tp2",          "TP2    "),
        ]:
            val = meta.get(key, "")
            if val:
                rows.append((f"{label_text}: {val}", "white", False))

        conf = meta.get("confidence", 0)
        if conf:
            conf_col = "green" if conf >= 7 else ("yellow" if conf >= 5 else "red")
            rows.append((f"Confidence: {conf}/10", conf_col, False))

        row_h   = fs_sm + 10
        pad     = 14
        card_w  = max(320, int(w * 0.22))
        card_h  = len(rows) * row_h + pad * 2
        cx0     = 14
        cy0     = h - card_h - 20

        # Card background — thick vivid border
        sig_border = C.get("green" if signal == "BUY" else ("red" if signal == "SELL" else "yellow"), C["white"])
        draw.rectangle([cx0, cy0, cx0 + card_w, cy0 + card_h],
                       fill=(8, 8, 18, 235), outline=(*sig_border, 255), width=3)

        for i, (text, color_name, bold) in enumerate(rows):
            ty = cy0 + pad + i * row_h
            if bold:
                r2, g2, b2 = C.get(color_name, C["white"])
                draw.rectangle([cx0 + 2, ty - 2, cx0 + card_w - 2, ty + row_h - 2],
                               fill=(r2, g2, b2, 65))
                draw.text((cx0 + 10, ty + 2), text,
                          fill=solid(color_name), font=font_b)
            else:
                draw.text((cx0 + 10, ty + 2), text,
                          fill=solid(color_name), font=font_sm)

    # ── Signal badge top-right ────────────────────────────
    if signal in ("BUY", "SELL", "WAIT"):
        sc   = (20, 200, 70, 245) if signal == "BUY" else ((210, 30, 30, 245) if signal == "SELL" else (200, 140, 0, 245))
        stxt = f"▲  {signal}" if signal == "BUY" else (f"▼  {signal}" if signal == "SELL" else f"⏳ {signal}")
        bw2 = max(140, int(w * 0.10))
        bh2 = fs_lg + 10
        draw.rectangle([w - bw2 - 10, 10, w - 10, 10 + bh2],
                       fill=sc, outline=(255, 255, 255, 220), width=3)
        draw.text((w - bw2 + 6, 16), stxt, fill=(255, 255, 255, 255), font=font_lg)

    # ── Watermark ─────────────────────────────────────────
    draw.text((10, h - fs_xs - 8), "TradingAI Analyst • Educational Only",
              fill=(200, 200, 200, 90), font=font_xs)

    # ── Composite overlay onto chart ──────────────────────
    composite = Image.alpha_composite(img, overlay).convert("RGB")

    # ── Smart zoom: crop to the active annotation zone ────
    # Collect all y-positions used in annotations
    ann_ys = []
    for ann in annotations:
        for key in ["y_position", "y_start", "y_end", "top_y", "bottom_y",
                    "swing_high_y", "swing_low_y", "neck_y", "head_y", "peak_y", "lsh_y", "rsh_y"]:
            if key in ann:
                ann_ys.append(yp(ann[key]))

    if len(ann_ys) >= 2:
        pad_v = int(h * 0.10)            # 10% vertical padding
        crop_top = max(0, min(ann_ys) - pad_v)
        crop_bot = min(h, max(ann_ys) + pad_v)
        # Ensure the card at the bottom is always visible
        card_top = cy0 - 10 if signal in ("BUY","SELL","WAIT") else h
        crop_bot = max(crop_bot, min(h, card_top + card_h + 30))
        # Ensure signal badge at top is always visible
        crop_top = min(crop_top, 8)
        # Only apply crop if it removes at least 12% of height
        if (crop_top > h * 0.05) or (crop_bot < h * 0.92):
            composite = composite.crop((0, crop_top, w, crop_bot))

    return composite


def pil_to_download_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=1)   # lossless, fast
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
                        st.image(st.session_state[ann_key], use_container_width=True)
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
                st.image(st.session_state["annotated"],
                         caption="AI-annotated chart — Entry / SL / TP / Patterns drawn",
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

# ── Footer ─────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer**: TradingAI Analyst is for **educational and informational purposes only**. "
    "It does NOT constitute financial advice. Trading involves substantial risk of loss. "
    "Always conduct your own research and manage your risk responsibly."
)
