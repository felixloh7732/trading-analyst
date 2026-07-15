"""
Chee AI — AI Financial Analyst
Built with Streamlit + Claude / Gemini Vision API
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

APP_VERSION = "2026.07.15-ui-dark-aggressive"

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
Your foundation is SUPPORT & RESISTANCE — SNR is the root of all trading. Every setup starts from a key horizontal level
the market has proven it respects. Fibonacci is a SUPPORTING tool, not a requirement: when you use it, focus on the
38.2% / 50% / 61.8% retracement zone — a fib level lining up with S/R adds confluence and confidence, but a setup at a
strong tested S/R level is valid even without Fibonacci. Patterns, structure and momentum exist to confirm or reject
the level — never to replace it.

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
🧭 MARKET STRUCTURE (kept simple — trend reading only)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Break of Structure (BOS)**
- Bullish BOS: Price closes ABOVE a previous swing high (confirms uptrend continuation).
- Bearish BOS: Price closes BELOW a previous swing low (confirms downtrend continuation).
- Rule: Body close required. Wicks alone do not count.

**Change of Character (CHoCH)**
- First BOS against the current trend = potential reversal signal.
- Bullish CHoCH: In a downtrend, price closes above previous swing high for the FIRST TIME.
- Bearish CHoCH: In an uptrend, price closes below previous swing low for the FIRST TIME.

**Equal Highs / Equal Lows**
- Stop clusters sit above equal highs and below equal lows — price often spikes through them before reversing.
- Never place a SL exactly at equal highs/lows; place it beyond them.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 FIBONACCI ANALYSIS (SUPPORTING TOOL — optional confluence)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Retracement Levels (draw from swing low to swing high for bullish, reverse for bearish)**
- 23.6% - Shallow retracement (very strong trend)
- 38.2% - ★ KEY — healthy pullback in a strong trend
- 50.0% - ★ KEY — psychological midpoint
- 61.8% - ★ KEY — golden ratio, deepest high-quality pullback
- 78.6%+ - Deep retracement — trend weakening, be careful

**THE KEY ZONE: 38.2% → 61.8% (including 50%).** This is the pullback area worth watching.
A retracement holding inside this zone keeps the trend healthy; use it as BONUS confluence when it overlaps S/R.
Fibonacci is OPTIONAL — if no clean swing exists, skip fib entirely and rely on S/R.

Do not use Fibonacci extensions, premium/discount labels, or arbitrary Fib targets.
Targets come from the next opposing tested S/R zone. Fibonacci never creates an entry or target by itself.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 SUPPORT & RESISTANCE — THE FOUNDATION (SNR is the root of trading)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- **Previous Highs/Lows**: Strong levels where price reversed before.
- **Round Numbers**: 1.1000, 2000, 50000 — psychological magnets.
- **Flip Zones**: Old resistance that becomes support after break (and vice versa).
- **Volume Nodes**: High volume at a price = strong acceptance zone.
- **Confluence Rule**: The more times a level has been tested (2–3 times = stronger, 4+ = weaker/ready to break).

**THE A / A+ SETUP (what this system hunts for):**
A setup  = trend direction + price pulls back to a TESTED S/R level or flip zone + a rejection candle (pin bar / engulfing)
forms there → enter, SL beyond the level/swing, TP at the next S/R level.
A+ setup = the same, PLUS a Fibonacci 38.2% / 50% / 61.8% level lining up with that S/R level → raise confidence.
S/R is the foundation: no valid S/R level = no trade. Missing Fibonacci does NOT invalidate a setup — it only means less confluence.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 TECHNICAL INDICATORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**RSI (14-period)**
- RSI never creates a trade by itself. Overbought is not an automatic sell and oversold is not an automatic buy.
- Above 50 supports bullish momentum; below 50 supports bearish momentum.
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
🔍 CONFLUENCE SCORING SYSTEM (SNR first, Fib as bonus)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

First apply the NON-NEGOTIABLE GATES. A score can never override a failed gate:
1. Direction gate: D1 bias and H4 structure agree.
2. Location gate: price is inside or has just retested a valid tested S/R zone.
3. Trigger gate: a rejection candle has CLOSED at that zone.
4. Invalidation gate: SL is beyond the zone/swing with a volatility buffer.
5. Reward gate: the next opposing S/R provides at least 2.0R after costs.
If any gate fails, output WAIT. Never label it A or A+.

Only after every gate passes, score the setup from 0–10:
- Key S/R level / flip zone in play (tested 2+ times) = +3
- Trend alignment (short-term + long-term agree) = +2
- Rejection candle at the level (pin bar, engulfing) = +2
- BONUS: Fibonacci 38.2% / 50% / 61.8% lines up with the S/R level = +1.5
- Chart pattern confirmation = +1
- RSI/MACD momentum agreement or divergence = +0.5

All gates + score 7+/10 = A setup.
All gates + score 7+/10 + Fib 38.2/50/61.8 overlap within 0.20 ATR of S/R = A+ setup.
Score below 7, or any failed gate = WAIT. No exceptions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📏 RISK MANAGEMENT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Minimum executable R:R = 1:2 after spread/fees/slippage. If the next opposing S/R is closer than 2R, WAIT.
- Ideal R:R = 1:3 or better
- Never risk more than 1-2% of account per trade
- SL placement: Beyond last swing point or key S/R level
- First target must be at least 2R and must coincide with a logical opposing S/R zone.
- Optional runners may target the next tested S/R only after the first 2R target is valid.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💵 DXY CORRELATION GUIDE (美元指数关联)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DXY is optional context, never an entry trigger. Correlations change by regime and must not be quoted as fixed numbers.
Never infer DXY direction from a Gold/FX chart. Mention DXY only when an actual timestamped DXY data block is supplied;
otherwise state that DXY context is unavailable and continue with price/S&R evidence only.
"""

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def claude_text(resp) -> str:
    """Extract text from a Claude API response, skipping thinking blocks (new Claude models)."""
    try:
        if getattr(resp, "stop_reason", None) == "refusal":
            return "Claude declined this request. Rephrase it as educational market analysis or choose another model."
        chunks = []
        for _blk in resp.content:
            _type = _blk.get("type", "") if isinstance(_blk, dict) else getattr(_blk, "type", "")
            if _type == "text":
                _text = _blk.get("text", "") if isinstance(_blk, dict) else getattr(_blk, "text", "")
                if _text:
                    chunks.append(_text)
        return "\n".join(chunks).strip()
    except Exception:
        return ""


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
    context: str = "",
    mode: str = "auto",
    risk: str = "balanced",
) -> str:
    """Send chart image to Gemini or Claude for analysis.
    mode: auto | signal | analysis — how the read is delivered.
    risk: conservative | balanced | aggressive — shapes entries/SL/TP and the WAIT threshold."""
    img_b64 = encode_image_to_base64(image)

    _mode_directive = {
        "signal": ("OUTPUT MODE = SIGNAL ONLY ⚡: Do NOT write the TREND/PATTERN/KEY LEVELS/FIBONACCI/"
                   "STRUCTURE/CANDLESTICK/DXY sections. Output ONLY: the **TRADE SETUP 交易方案** section (bilingual), "
                   "then a short **WHY 理由** list (2-3 bullets citing the exact levels), then a short "
                   "**WHEN THIS IDEA IS WRONG 无效条件** list (1-2 bullets). Be decisive. "
                   "Still output the JSON block at the end as required."),
        "analysis": ("OUTPUT MODE = DEEP ANALYSIS 🔬: This is an educational deep-dive. Write ALL sections "
                     "thoroughly (2-3 lines each). Explain WHY behind every observation — the goal is that the "
                     "trader learns to see what you see. In TRADE SETUP, lean towards WAIT unless the setup is "
                     "truly A-grade; focus on reading the chart, not pushing a trade."),
    }.get(mode, "")

    _risk_directive = {
        "conservative": ("RISK PROFILE = CONSERVATIVE 🛡️: Only signal BUY/SELL for A/A+ setups (confidence 7+); "
                         "otherwise say WAIT. Entries only after full confirmation (rejection candle closed). "
                         "SL beyond structure with extra buffer. TP at the nearest opposing S/R only if it offers 2R+. Max 1% risk. "
                         "When in doubt — WAIT."),
        "aggressive": ("RISK PROFILE = AGGRESSIVE 🔥: You MUST output a directional signal: BUY or SELL. WAIT is forbidden. "
                       "If all A/A+ gates pass, label it A or A+. If gates are missing, still choose the stronger direction but "
                       "label it SPECULATIVE AGGRESSIVE, list every failed gate, use at most 0.5% account risk, and give a fast "
                       "invalidation level. Never pretend a speculative signal is A-grade."),
    }.get(risk, "RISK PROFILE = BALANCED ⚖️: standard rules — confirmed setups only, 1-2% risk, minimum 1:2 R:R.")

    user_prompt = f"""
{_mode_directive}
{_risk_directive}

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

**FIBONACCI 斐波那契:** (supporting tool — skip gracefully if no clean swing)
[EN] If a clean dominant swing exists: where is price relative to the 38.2% / 50% / 61.8% retracements?
Does one of those levels OVERLAP a key S/R level? (= bonus confluence). If fib adds nothing here, say "No clean fib swing — trading pure S/R." 2 lines max.
[中文] 若有清晰波段：价格相对 38.2%/50%/61.8% 回撤位在哪里？是否与关键支撑阻力重合（加分项）？若无清晰波段就直说「无斐波那契参考，纯看支撑阻力」。最多2行。

**STRUCTURE 市场结构:**
[EN] Higher highs/lows or lower highs/lows? Most recent BOS or CHoCH? Equal highs/lows nearby (stop clusters)? 1 line.
[中文] 高点低点结构如何？最近的结构突破/转变？附近是否有平顶/平底？1行。

**CANDLESTICK PATTERNS 单K线形态:**
[EN] Identify any significant single or multi-candle patterns on the LAST 3-5 candles: Doji, Hammer, Shooting Star, Engulfing (Bullish/Bearish), Morning Star, Evening Star, Pin Bar, Marubozu, Harami, Tweezer Top/Bottom. If none significant: "No key candle pattern."
[中文] 识别最近3-5根K线的重要形态：十字星、锤子线、流星线、吞没（看涨/看跌）、晨星、暮星、钉线、大阳/大阴线、孕线、镊子顶/底。若无：「无明显K线形态」。

**DXY CONTEXT 美元指数:**
[EN] Only discuss DXY if timestamped DXY data was explicitly supplied. Never guess it from this instrument's chart.
[中文] 只在已提供带时间戳的 DXY 数据时评论；不可从当前品种图表猜测 DXY。

**TRADE SETUP 交易方案:**
- Signal 信号: BUY 🟢 / SELL 🔴 / WAIT ⏳
- Entry 入场: [EN price zone — at a key S/R level; note if a fib 38.2/50/61.8 level adds confluence] / [中文价格区域]
- SL 止损: [EN beyond the level / swing, never at equal highs-lows] / [中文止损位说明]
- TP1 目标1: [EN next S/R level + R:R] / [中文目标位]
- TP2 目标2: [EN level + R:R] / [中文目标位]
- Confluences 汇合因素: [comma-separated — S/R quality first, fib overlap if any]
- Confidence 信心: X/10 (use the SNR-first scoring system)
- Warning 风险提示: [EN 1 line] / [中文一句话]

---
Now output the drawing instructions as a JSON block to annotate the chart.
The JSON MUST also contain `bias_signal` (always BUY or SELL) and `setup_grade`
(A, A+, SPECULATIVE, or WAIT). In aggressive mode `signal` MUST equal `bias_signal`; WAIT is not allowed.

════ MARKET STRUCTURE ANNOTATION RULES ════

PURPOSE: Show the most important S&R levels and any clear chart pattern forming.
Keep it CLEAN and MINIMAL — maximum 5 annotations total. No clutter.
Do NOT draw SL / TP / Entry points, trendlines, or Fibonacci levels.

MAXIMUM 5 ANNOTATIONS TOTAL — STRICT LIMIT.

PRIORITY ORDER (draw in this order, stop when you reach 5):
  1. Key S/R horizontal lines (most important — always include the 2 most critical levels)
  2. ONE zone box: the most obvious supply/demand zone, OR the fib 38.2–61.8% zone if it overlaps an S/R level
  3. ONE BOS or CHoCH (the most recent structure break only — not historical ones)
  4. ONE liquidity zone (only if equal highs/lows are clearly visible)

ANNOTATION TYPES ALLOWED:

1. "horizontal_line" — KEY S/R LEVELS (most important, always draw these first)
   - Only mark price levels the market has clearly respected 2+ times
   - color: green=strong support, red=strong resistance, yellow=equal highs/lows

2. "zone_box" — SUPPLY/DEMAND zone or FIB ZONE (max 1 total)
   - Supply/Demand: only if there is a clear strong impulse move from the zone
   - Fib zone: the 38.2–61.8% retracement area, only when it overlaps S/R — label "Fib Zone 斐波区"
   - color: green=Demand/Fib in uptrend, red=Supply/Fib in downtrend, yellow=Liquidity pool

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
  "Support 支撑" / "Resistance 阻力" / "Demand Zone 需求区" / "Supply Zone 供给区" / "Fib Zone 斐波区"
  "BOS ↑ 结构突破" / "CHoCH ↓ 结构变化" / "Liquidity 流动性" / "Equal Lows 平底" / "Equal Highs 平顶"

For y positions use: "top"(0.06), "upper_quarter"(0.20), "upper_third"(0.30), "middle"(0.50), "lower_third"(0.65), "lower_quarter"(0.78), "bottom"(0.93)

Example — bearish with double top pattern:
```json
{{
  "signal": "SELL",
  "bias_signal": "SELL",
  "setup_grade": "A",
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
  "bias_signal": "BUY",
  "setup_grade": "A",
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
        return claude_text(response)


def parse_json_from_analysis(analysis_text: str) -> dict:
    """Extract chart metadata through the resilient multi-stage JSON parser."""
    return parse_ai_json(analysis_text)


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
# LIVE DATA HELPERS — auto-fetch for AI Analyst & Market Scout
# ============================================================

_SYMBOL_KEYWORDS = [
    (("gold", "xauusd", "xau", "黄金", "金价"),            "XAU/USD", "Gold (XAUUSD)"),
    (("silver", "xagusd", "xag", "白银", "银价"),          "XAG/USD", "Silver (XAGUSD)"),
    (("eurusd", "eur/usd", "euro", "欧元"),               "EUR/USD", "EURUSD"),
    (("gbpusd", "gbp/usd", "pound", "cable", "英镑"),      "GBP/USD", "GBPUSD"),
    (("usdjpy", "usd/jpy", "yen", "日元", "日币"),         "USD/JPY", "USDJPY"),
    (("audusd", "aud/usd", "aussie", "澳元"),             "AUD/USD", "AUDUSD"),
    (("nzdusd", "nzd/usd", "kiwi", "纽元"),               "NZD/USD", "NZDUSD"),
    (("usdcad", "usd/cad", "loonie", "加元"),             "USD/CAD", "USDCAD"),
    (("usdchf", "usd/chf", "瑞郎"),                        "USD/CHF", "USDCHF"),
    (("btc", "bitcoin", "比特币"),                         "BTC/USD", "Bitcoin (BTCUSD)"),
    (("eth", "ethereum", "以太坊"),                        "ETH/USD", "Ethereum (ETHUSD)"),
]


def detect_symbols_in_text(text: str) -> list:
    """Find tradable symbols mentioned in a chat message. Returns [(td_symbol, label), ...]"""
    t = (text or "").lower()
    found = []
    for kws, td_sym, label in _SYMBOL_KEYWORDS:
        if any(k in t for k in kws):
            found.append((td_sym, label))
    return found


@st.cache_data(ttl=300, show_spinner=False)
def td_fetch_df(symbol: str, interval: str, outputsize: int, api_key_td: str):
    """Fetch candles from Twelve Data → DataFrame with Open/High/Low/Close columns."""
    import requests as _rq
    import pandas as _pd
    r = _rq.get("https://api.twelvedata.com/time_series", params={
        "symbol": symbol, "interval": interval,
        "outputsize": outputsize, "apikey": api_key_td,
    }, timeout=20)
    j = r.json()
    if j.get("status") == "error" or "values" not in j:
        raise RuntimeError(j.get("message", "no data returned"))
    df = _pd.DataFrame(j["values"]).iloc[::-1].reset_index(drop=True)
    df = df.rename(columns={"datetime": "Date", "open": "Open", "high": "High",
                            "low": "Low", "close": "Close", "volume": "Volume"})
    for c in ("Open", "High", "Low", "Close"):
        df[c] = df[c].astype(float)
    if "Volume" in df.columns:
        df["Volume"] = _pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    return df.set_index("Date")


# yfinance tickers — free live data, no API key needed
_YF_TICKERS = {
    "Gold (XAUUSD)":     "GC=F",
    "Silver (XAGUSD)":   "SI=F",
    "EURUSD":            "EURUSD=X",
    "GBPUSD":            "GBPUSD=X",
    "USDJPY":            "USDJPY=X",
    "AUDUSD":            "AUDUSD=X",
    "NZDUSD":            "NZDUSD=X",
    "USDCAD":            "USDCAD=X",
    "USDCHF":            "USDCHF=X",
    "Bitcoin (BTCUSD)":  "BTC-USD",
    "Ethereum (ETHUSD)": "ETH-USD",
}


@st.cache_data(ttl=300, show_spinner=False)
def yf_fetch_df(ticker: str, tf: str = "H1"):
    """Fetch candles from yfinance (free, no key). tf: H1 / H4 / D1. Returns Open/High/Low/Close df."""
    import yfinance as yf
    if tf == "D1":
        df = yf.Ticker(ticker).history(period="2y", interval="1d")
    else:
        df = yf.Ticker(ticker).history(period="60d", interval="1h")
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {ticker} — try another timeframe.")
    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    df = df[cols].copy()
    if tf == "H4":
        _agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in df.columns:
            _agg["Volume"] = "sum"
        df = df.resample("4h").agg(_agg).dropna(subset=["Open", "High", "Low", "Close"])
    return df


def fetch_candles_any(label: str, tf: str, td_key: str = ""):
    """Fetch live candles: yfinance first (free, no key); fall back to Twelve Data.
    Returns (df, source_name). Raises RuntimeError with a friendly fix if neither works."""
    _yf_problem = None
    try:
        return yf_fetch_df(_YF_TICKERS[label], tf), "Yahoo Finance"
    except (ImportError, ModuleNotFoundError):
        _yf_problem = "yfinance is not installed"
    except Exception as _e:
        _yf_problem = str(_e)
    # Fallback: Twelve Data
    _td_map = {lbl: td for _kws, td, lbl in _SYMBOL_KEYWORDS}
    if td_key and label in _td_map:
        _iv = {"H1": "1h", "H4": "4h", "D1": "1day"}.get(tf, "1h")
        return td_fetch_df(_td_map[label], _iv, 350, td_key), "Twelve Data"
    raise RuntimeError(
        f"Live data unavailable ({_yf_problem}). Fix: run `pip install yfinance` locally, "
        "or on Streamlit Cloud just reboot the app (requirements.txt already includes yfinance), "
        "or add a free Twelve Data key in the sidebar as backup. "
        "本地请运行 pip install yfinance；云端重启 app 即可；或在侧栏填入 Twelve Data key 作为备用。"
    )


def _atr_series(df, period: int = 14):
    """Wilder ATR, aligned to df.index."""
    import pandas as pd
    prev = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev).abs(),
        (df["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _wilder_rsi(closes, period: int = 14):
    delta = closes.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
    return 100 - 100 / (1 + rs)


def detect_rejection_candle(df) -> dict:
    """Deterministically classify the latest CLOSED candle; never anticipates an open candle."""
    if len(df) < 3:
        return {"confirmed": False, "direction": "none", "pattern": "none"}
    prev, cur = df.iloc[-2], df.iloc[-1]
    rng = max(float(cur.High - cur.Low), 1e-12)
    body = abs(float(cur.Close - cur.Open))
    upper = float(cur.High - max(cur.Open, cur.Close))
    lower = float(min(cur.Open, cur.Close) - cur.Low)
    bull_pin = lower >= max(2 * body, 0.45 * rng) and cur.Close >= cur.Low + 0.65 * rng
    bear_pin = upper >= max(2 * body, 0.45 * rng) and cur.Close <= cur.Low + 0.35 * rng
    bull_engulf = (cur.Close > cur.Open and prev.Close < prev.Open
                   and cur.Open <= prev.Close and cur.Close >= prev.Open)
    bear_engulf = (cur.Close < cur.Open and prev.Close > prev.Open
                   and cur.Open >= prev.Close and cur.Close <= prev.Open)
    if bull_engulf or bull_pin:
        return {"confirmed": True, "direction": "bullish",
                "pattern": "bullish engulfing" if bull_engulf else "bullish pin bar"}
    if bear_engulf or bear_pin:
        return {"confirmed": True, "direction": "bearish",
                "pattern": "bearish engulfing" if bear_engulf else "bearish pin bar"}
    return {"confirmed": False, "direction": "none", "pattern": "none"}


def detect_impulse_fib(df, lookback: int = 180, pivot_window: int = 5):
    """Return Fib levels only for a chronological, confirmed impulse of at least 2 ATR."""
    d = df.tail(lookback)
    if len(d) < 40:
        return None
    highs, lows = d["High"].to_numpy(), d["Low"].to_numpy()
    atr_now = float(_atr_series(d).dropna().iloc[-1])
    ema20 = d["Close"].ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = d["Close"].ewm(span=50, adjust=False).mean().iloc[-1]
    bias = "bullish" if ema20 > ema50 else "bearish"
    piv_h, piv_l = [], []
    w = pivot_window
    for i in range(w, len(d) - w):
        if highs[i] == max(highs[i - w:i + w + 1]):
            piv_h.append((i, float(highs[i])))
        if lows[i] == min(lows[i - w:i + w + 1]):
            piv_l.append((i, float(lows[i])))
    candidates = []
    if bias == "bullish":
        for hi_i, hi in piv_h:
            prior = [(i, p) for i, p in piv_l if i < hi_i and hi_i - i <= 100]
            if prior:
                lo_i, lo = min(prior, key=lambda x: x[1])
                if hi - lo >= 2 * atr_now and hi_i < len(d) - 1:
                    candidates.append((hi_i, lo_i, lo, hi))
    else:
        for lo_i, lo in piv_l:
            prior = [(i, p) for i, p in piv_h if i < lo_i and lo_i - i <= 100]
            if prior:
                hi_i, hi = max(prior, key=lambda x: x[1])
                if hi - lo >= 2 * atr_now and lo_i < len(d) - 1:
                    candidates.append((lo_i, hi_i, lo, hi))
    if not candidates:
        return None
    end_i, start_i, lo, hi = max(candidates, key=lambda x: x[0])
    span = hi - lo
    levels = ({0.382: hi - 0.382 * span, 0.5: hi - 0.5 * span, 0.618: hi - 0.618 * span}
              if bias == "bullish" else
              {0.382: lo + 0.382 * span, 0.5: lo + 0.5 * span, 0.618: lo + 0.618 * span})
    cur = float(d["Close"].iloc[-1])
    retr = (hi - cur) / span if bias == "bullish" else (cur - lo) / span
    return {"direction": bias, "low": lo, "high": hi, "levels": levels,
            "retracement": retr, "atr": atr_now,
            "start_time": str(d.index[start_i]), "end_time": str(d.index[end_i])}


def build_market_digest(df, label: str, tf: str) -> str:
    """Compress candles into deterministic context; AI explains but does not invent inputs."""
    closes = df["Close"]
    cur    = float(closes.iloc[-1])
    ema20  = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50  = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(closes.ewm(span=200, adjust=False).mean().iloc[-1]) if len(closes) >= 200 else None
    rsi = float(_wilder_rsi(closes).iloc[-1])
    atr = float(_atr_series(df).dropna().iloc[-1])
    up_bias = ema20 > ema50
    rejection = detect_rejection_candle(df)
    fib = detect_impulse_fib(df)

    last5   = df.tail(5)
    candles = " | ".join(f"O{r.Open:.6g} H{r.High:.6g} L{r.Low:.6g} C{r.Close:.6g}"
                         for r in last5.itertuples())
    parts = [
        f"{label} {tf} — CURRENT PRICE: {cur:.6g}",
        f"  ATR(14, Wilder): {atr:.6g}",
        f"  Trend: EMA20 {ema20:.6g} {'>' if up_bias else '<='} EMA50 {ema50:.6g}"
        + (f", EMA200 {ema200:.6g}" if ema200 else "") + f" → {'bullish' if up_bias else 'bearish'} bias",
        f"  RSI(14, Wilder): {rsi:.1f} (momentum context only; never an entry trigger)",
        f"  Latest closed-candle trigger: {rejection['pattern']} ({rejection['direction']}); confirmed={rejection['confirmed']}",
        f"  Last 5 candles: {candles}",
    ]
    if fib:
        lv = fib["levels"]
        overlap = "inside 38.2-61.8 pullback" if 0.382 <= fib["retracement"] <= 0.618 else f"{fib['retracement']*100:.1f}% retraced"
        parts.append(
            f"  Confirmed chronological Fib impulse {fib['direction']} ({fib['start_time']} → {fib['end_time']}): "
            f"low={fib['low']:.6g}, high={fib['high']:.6g}, price={overlap}; "
            f"38.2%={lv[0.382]:.6g}, 50%={lv[0.5]:.6g}, 61.8%={lv[0.618]:.6g}"
        )
    else:
        parts.append("  Fibonacci: SKIP — no valid chronological confirmed impulse.")
    _zones = find_sr_zones(df, max_zones=4)
    if _zones:
        _zsum = "; ".join(
            f"{z['kind'][:3].upper()} {z['low']:.6g}-{z['high']:.6g} ({z['touches']}x tested{', flip' if z['flip'] else ''})"
            for z in _zones)
        parts.append(f"  Auto-detected S/R zones: {_zsum}")
    return "\n".join(parts)


def find_sr_zones(df, lookback: int = 250, max_zones: int = 6, htf_df=None) -> list:
    """Detect S/R zones the way a professional trader draws them.

    Method (transparent, all stats are real):
    1. Significant swing pivots only — window 5 (plus MAJOR swings, window 12, weighted heavier).
       Minor 3-bar wiggles a trader would ignore are ignored here too.
    2. Clustering tolerance = 0.6 × ATR(14) — adapts to each market's volatility instead of a fixed %.
    3. Each zone is SCORED like a trader judges it:
       touches (capped — 5+ touches means worn out, not stronger) + REACTION strength (how many ATR
       price bounced away after each touch) + recency decay (fresh levels beat stale ones) +
       flip behaviour + contains a major swing + ROUND NUMBER proximity + HIGHER-TIMEFRAME confirmation.
    4. Zone width is kept realistic: 0.25–1.2 ATR.
    """
    import math
    d = df.tail(lookback)
    highs, lows, closes = d["High"].values, d["Low"].values, d["Close"].values
    n = len(d)
    if n < 30:
        return []

    # ── Wilder ATR(14); local values are also used when judging old reactions ──
    atr_values = _atr_series(d).bfill().to_numpy()
    atr = float(atr_values[-1])
    if atr <= 0:
        atr = max((float(highs.max()) - float(lows.min())) / 100.0, 1e-9)

    # ── Swing pivots: significant (w=5) + major (w=12, heavier weight) ──
    pivots = []
    for w, wt in ((5, 1.0), (12, 1.6)):
        for i in range(w, n - w):
            if highs[i] >= max(highs[i - w:i + w + 1]):
                pivots.append(["H", float(highs[i]), i, wt])
            if lows[i] <= min(lows[i - w:i + w + 1]):
                pivots.append(["L", float(lows[i]), i, wt])
    # A major pivot also appears in the window-5 pass. Deduplicate it instead of
    # pretending one market reaction was two separate touches.
    _dedup = {}
    for p in pivots:
        _key = (p[0], p[2])
        if _key not in _dedup or p[3] > _dedup[_key][3]:
            _dedup[_key] = p
    pivots = list(_dedup.values())
    if not pivots:
        return []

    # ── Reaction strength: how far price moved away within 6 bars after the touch (in ATR) ──
    for p in pivots:
        _k, _pr, _i, _wt = p[0], p[1], p[2], p[3]
        j2 = min(n, _i + 7)
        if _i + 1 < j2:
            _local_atr = max(float(atr_values[_i]), 1e-12)
            react = ((_pr - float(min(lows[_i + 1:j2]))) / _local_atr) if _k == "H" \
                else ((float(max(highs[_i + 1:j2])) - _pr) / _local_atr)
        else:
            react = 0.0
        p.append(max(react, 0.0))

    # ── Cluster around a stable centre. This prevents single-link chaining
    # (A near B, B near C) from merging distinct levels into one oversized zone. ──
    tol = 0.6 * atr
    pivots.sort(key=lambda p: p[1])
    clusters, cur = [], [pivots[0]]
    for p in pivots[1:]:
        _centre = sum(x[1] for x in cur) / len(cur)
        _new_range = max(p[1], max(x[1] for x in cur)) - min(p[1], min(x[1] for x in cur))
        if abs(p[1] - _centre) <= tol and _new_range <= 1.2 * atr:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)

    price_now = float(closes[-1])

    # ── Round-number ladder (step ≈ ≥0.4% of price, from a 'nice' ladder) ──
    _steps = [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05,
              0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50, 100, 250, 500, 1000]
    step = next((s for s in _steps if s >= price_now * 0.004), _steps[-1])

    # ── Higher-timeframe zones for confirmation ──
    htf_zones = []
    if htf_df is not None and len(htf_df) >= 30:
        htf_zones = find_sr_zones(htf_df, lookback=min(len(htf_df), 300), max_zones=8)

    half_life = lookback / 3.0
    out = []
    for z in clusters:
        prices     = [p[1] for p in z]
        _raw_touch_bars = sorted(set(p[2] for p in z))
        # Plateau highs/lows across adjacent candles are one test, not many.
        touch_bars = []
        for _bar in _raw_touch_bars:
            if not touch_bars or _bar - touch_bars[-1] >= 4:
                touch_bars.append(_bar)
        center     = sum(prices) / len(prices)
        touches    = len(touch_bars)
        n_high     = sum(1 for p in z if p[0] == "H")
        n_low      = sum(1 for p in z if p[0] == "L")
        recency    = sum(math.exp(-(n - 1 - i) / half_life) for i in touch_bars)
        _reactions = [p[4] for p in sorted(z, key=lambda p: p[2])]
        react      = sum(_reactions) / len(_reactions)
        reaction_decay = (len(_reactions) >= 3 and
                          sum(_reactions[-2:]) / 2 < 0.65 * max(sum(_reactions[:2]) / 2, 1e-9))
        major      = any(p[3] > 1.0 for p in z)
        flip       = n_high > 0 and n_low > 0
        rnd        = abs(center / step - round(center / step)) * step <= max(0.25 * atr, step * 0.15)
        rnd_level  = round(center / step) * step if rnd else None
        htf_ok     = any(hz["low"] - tol <= center <= hz["high"] + tol for hz in htf_zones)

        score = (1.6 * min(touches, 3)          # independent tests; repeated tests can consume orders
                 + 1.4 * recency                # fresh levels matter more
                 + 1.1 * min(react, 3.0)        # strong rejections = real level
                 + (1.5 if flip else 0.0)
                 + (1.2 if major else 0.0)
                 + (2.0 if htf_ok else 0.0)
                 + (0.9 if rnd else 0.0)
                 - (1.5 if touches >= 4 and reaction_decay else 0.0))

        lo, hi = min(prices), max(prices)
        if hi - lo < 0.25 * atr:
            lo, hi = center - 0.125 * atr, center + 0.125 * atr
        if hi - lo > 1.2 * atr:
            lo, hi = center - 0.6 * atr, center + 0.6 * atr

        _kind = "decision" if lo <= price_now <= hi else ("resistance" if center > price_now else "support")
        out.append({
            "low": lo, "high": hi, "center": center,
            "touches": touches,
            "kind": _kind,
            "flip": flip, "swing_highs": n_high, "swing_lows": n_low,
            "bars_since_last_touch": n - 1 - max(touch_bars),
            "avg_reaction_atr": round(react, 2),
            "reaction_decay": reaction_decay,
            "major_swing": major,
            "round_level": rnd_level,
            "htf_confirmed": htf_ok,
            "score": round(score, 2),
        })

    out = [z for z in out if z["touches"] >= 2 or z["major_swing"]]
    out.sort(key=lambda z: -z["score"])
    strongest = out[:max_zones]
    strongest.sort(key=lambda z: -z["center"])
    return strongest


def sr_zones_text(zones: list) -> str:
    """Compact one-line-per-zone text for AI prompts — includes the real scoring drivers."""
    lines = []
    for z in zones:
        bits = [
            f"{z['kind'].upper()} zone {z['low']:.6g}–{z['high']:.6g} (score {z.get('score', '—')})",
            f"touched {z['touches']}x ({z['swing_highs']} swing-highs, {z['swing_lows']} swing-lows)",
            f"avg reaction after touch ≈ {z.get('avg_reaction_atr', 0)} ATR",
            f"last tested {z['bars_since_last_touch']} bars ago",
        ]
        if z.get("flip"):
            bits.append("FLIP ZONE (acted as both support & resistance)")
        if z.get("major_swing"):
            bits.append("contains a MAJOR swing point")
        if z.get("htf_confirmed"):
            bits.append("CONFIRMED on the D1 higher timeframe")
        if z.get("round_level"):
            bits.append(f"sits at round number {z['round_level']:.6g}")
        if z.get("reaction_decay"):
            bits.append("WEAKENING: recent rejections are materially smaller than early reactions")
        if z.get("kind") == "decision":
            bits.append("PRICE IS INSIDE THIS ZONE: no directional trade until rejection/close confirms")
        lines.append(" | ".join(bits))
    return "\n".join(lines)


def build_setup_snapshot(d1_df, h4_df, h1_df) -> dict:
    """Deterministic A/A+ gate engine. The LLM may explain this result, never override it."""
    def _bias(frame):
        close = frame["Close"]
        return "bullish" if close.ewm(span=20, adjust=False).mean().iloc[-1] > close.ewm(span=50, adjust=False).mean().iloc[-1] else "bearish"

    d1_bias, h4_bias = _bias(d1_df), _bias(h4_df)
    direction = "BUY" if d1_bias == h4_bias == "bullish" else (
        "SELL" if d1_bias == h4_bias == "bearish" else "WAIT")
    h1_atr = float(_atr_series(h1_df).dropna().iloc[-1])
    h4_zones = find_sr_zones(h4_df, lookback=300, max_zones=8, htf_df=d1_df)
    price = float(h1_df["Close"].iloc[-1])
    rejection = detect_rejection_candle(h1_df)
    expected_rejection = "bullish" if direction == "BUY" else "bearish"

    relevant = []
    for zone in h4_zones:
        if direction == "BUY" and zone["kind"] not in ("support", "decision"):
            continue
        if direction == "SELL" and zone["kind"] not in ("resistance", "decision"):
            continue
        distance = 0.0 if zone["low"] <= price <= zone["high"] else min(abs(price - zone["low"]), abs(price - zone["high"]))
        relevant.append((distance, zone))
    zone = min(relevant, key=lambda x: x[0])[1] if relevant else None
    location_ok = bool(zone and zone["low"] - 0.25 * h1_atr <= price <= zone["high"] + 0.25 * h1_atr)
    last = h1_df.iloc[-1]
    candle_touches_zone = bool(zone and float(last.Low) <= zone["high"] + 0.10 * h1_atr
                               and float(last.High) >= zone["low"] - 0.10 * h1_atr)
    rejection_ok = bool(location_ok and candle_touches_zone and rejection["confirmed"]
                        and rejection["direction"] == expected_rejection)

    entry = price
    stop = target = rr = None
    opposing = []
    if zone and direction == "BUY":
        stop = zone["low"] - 0.20 * h1_atr
        opposing = [z for z in h4_zones if z["low"] > entry and z["kind"] == "resistance"]
        target = min(opposing, key=lambda z: z["low"])["low"] if opposing else None
    elif zone and direction == "SELL":
        stop = zone["high"] + 0.20 * h1_atr
        opposing = [z for z in h4_zones if z["high"] < entry and z["kind"] == "support"]
        target = max(opposing, key=lambda z: z["high"])["high"] if opposing else None
    if stop is not None and target is not None and abs(entry - stop) > 1e-12:
        rr = abs(target - entry) / abs(entry - stop)

    fib = detect_impulse_fib(h4_df)
    fib_overlap = False
    fib_level = None
    if fib and zone:
        for ratio, level in fib["levels"].items():
            if zone["low"] - 0.20 * h1_atr <= level <= zone["high"] + 0.20 * h1_atr:
                fib_overlap, fib_level = True, {"ratio": ratio, "price": level}
                break

    gates = {
        "direction": direction != "WAIT",
        "location": location_ok,
        "rejection": rejection_ok,
        "invalidation": stop is not None and ((direction == "BUY" and stop < entry) or (direction == "SELL" and stop > entry)),
        "reward": rr is not None and rr >= 2.0,
    }
    executable = all(gates.values())
    grade = "A+" if executable and fib_overlap else ("A" if executable else "WAIT")
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "direction": direction if executable else "WAIT", "candidate_direction": direction,
        "grade": grade, "gates": gates, "failed_gates": failed,
        "d1_bias": d1_bias, "h4_bias": h4_bias,
        "entry": entry, "stop": stop, "target": target, "rr": rr,
        "zone": zone, "rejection": rejection,
        "fib_overlap": fib_overlap, "fib_level": fib_level,
    }


def ai_text_call(prompt: str, api_key: str, model: str, json_mode: bool = False) -> str:
    """One-shot text call to Gemini or Claude. json_mode forces/nudges pure-JSON output."""
    if model.startswith("gemini"):
        _c = google_genai.Client(api_key=api_key)
        _cfg = None
        if json_mode:
            try:
                _cfg = google_types.GenerateContentConfig(response_mime_type="application/json")
            except Exception:
                _cfg = None
        r = _c.models.generate_content(model=model, contents=[prompt], config=_cfg)
        return r.text
    _c = anthropic.Anthropic(api_key=api_key)
    _p = prompt
    if json_mode:
        # Newer Claude models reject assistant prefill — use a hard instruction instead;
        # parse_ai_json() downstream handles extraction & repair.
        _p = (prompt + "\n\nCRITICAL: Respond with ONLY the raw JSON object. No markdown fences, "
              "no commentary. The very first character of your reply must be '{' and the last must be '}'.")
    r = _c.messages.create(model=model, max_tokens=3000,
                           messages=[{"role": "user", "content": _p}])
    return claude_text(r)


def parse_ai_json(raw: str, api_key: str = "", model: str = "") -> dict:
    """Extract a JSON object from an AI reply. Repairs common issues; falls back to AI self-repair."""
    if not raw:
        return {}
    decoder = json.JSONDecoder()
    for start in (i for i, char in enumerate(raw) if char == "{"):
        try:
            value, _ = decoder.raw_decode(raw[start:])
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    txt = m.group(0)
    # 1) as-is
    try:
        return json.loads(txt)
    except Exception:
        pass
    # 2) mechanical repairs: smart quotes, trailing commas, control chars
    t = txt
    for _a, _b in (("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), (" ", " ")):
        t = t.replace(_a, _b)
    t = re.sub(r",\s*([}\]])", r"\1", t)
    t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", t)
    try:
        return json.loads(t)
    except Exception:
        pass
    # 3) AI self-repair
    if api_key and model:
        try:
            fixed = ai_text_call(
                "Rewrite the following as STRICT valid JSON (RFC 8259). Escape all double quotes and "
                "newlines inside string values. Do not change the content. Output ONLY the JSON object:\n\n"
                + txt[:9000],
                api_key, model, json_mode=True)
            m2 = re.search(r"\{.*\}", fixed, re.DOTALL)
            if m2:
                return json.loads(m2.group(0))
        except Exception:
            pass
    return {}


def force_aggressive_direction(raw: str) -> tuple[str, str]:
    """Guarantee BUY/SELL for Aggressive mode while keeping non-A setups explicit."""
    meta = parse_ai_json(raw)
    original = str(meta.get("signal", "WAIT")).upper()
    if original in ("BUY", "SELL"):
        grade = str(meta.get("setup_grade", "SPECULATIVE")).upper()
        return original, grade if grade in ("A", "A+", "SPECULATIVE") else "SPECULATIVE"
    bias = str(meta.get("bias_signal", "")).upper()
    if bias not in ("BUY", "SELL"):
        long_term = re.search(r"long[- ]term[^\n:]*:\s*(bullish|bearish)", raw or "", re.IGNORECASE)
        if long_term:
            bias = "BUY" if long_term.group(1).lower() == "bullish" else "SELL"
        else:
            text = (raw or "").lower()
            bulls = text.count("bullish") + text.count("看涨") + text.count("买入")
            bears = text.count("bearish") + text.count("看跌") + text.count("卖出")
            bias = "BUY" if bulls >= bears else "SELL"
    return bias, "SPECULATIVE"


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="Chee AI",
    page_icon="⚡",
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

# ══════════════════════════════════════════════════════════
# CHEE AI — Black/Green terminal theme
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500;1,600&family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

  :root {
    --bg:        #050807;
    --surface:   #0b100d;
    --surface2:  #101712;
    --border:    #1c2a21;
    --border-hi: #2b4534;
    --green:     #22c55e;
    --green-hi:  #4ade80;
    --green-dim: #16803c;
    --text:      #e8f0ea;
    --muted:     #7d8f83;
    --red:       #ef4444;
    --amber:     #f59e0b;
  }

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Main background: near-black with green aurora glow ── */
  [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(ellipse 80% 50% at 70% -10%, rgba(34,197,94,0.13), transparent 60%),
      radial-gradient(ellipse 60% 40% at 10% 110%, rgba(34,197,94,0.07), transparent 60%),
      #050807 !important;
  }
  [data-testid="stMain"] { background: transparent; }
  [data-testid="stHeader"] { background: rgba(5,8,7,0.7) !important; backdrop-filter: blur(8px); }

  /* ── Text ── */
  .stApp p, .stApp li, .stApp span { color: var(--text); }
  .stApp p, .stApp li { color: #cfe0d4 !important; }

  h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; }
  h1 {
    background: linear-gradient(90deg, #e8f0ea 20%, #4ade80 80%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 40px !important; font-weight: 700 !important; letter-spacing: -1px;
  }
  h2 { color: #e8f0ea !important; font-weight: 700 !important; letter-spacing: -0.5px; }
  h3 { color: #d5e5da !important; font-weight: 600 !important; }

  /* ── Buttons ── */
  .stButton>button {
    background: linear-gradient(180deg, #101812, #0b100d);
    color: #cfe0d4 !important; border: 1px solid var(--border-hi); border-radius: 12px;
    font-weight: 600 !important; font-size: 15px !important; padding: 10px 16px;
    width: 100%; transition: all 0.2s; box-shadow: none; letter-spacing: 0.2px;
  }
  .stButton>button:hover {
    border-color: var(--green); color: #4ade80 !important;
    box-shadow: 0 0 18px rgba(34,197,94,0.25); transform: translateY(-1px);
  }
  .stButton>button[kind="primary"], .stButton>button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #16a34a, #22c55e);
    color: #04120a !important; border: 1px solid #4ade80;
    font-weight: 800 !important;
    box-shadow: 0 4px 20px rgba(34,197,94,0.35);
  }
  .stButton>button[kind="primary"]:hover {
    box-shadow: 0 6px 28px rgba(34,197,94,0.55); color: #04120a !important;
  }
  .stDownloadButton>button {
    background: linear-gradient(180deg, #101812, #0b100d) !important;
    color: #4ade80 !important; border: 1px solid var(--border-hi) !important; border-radius: 12px !important;
  }

  /* ── BUY / SELL / WAIT badges ── */
  .buy-badge {
    background: rgba(34,197,94,0.12);
    border: 1px solid #22c55e; color: #4ade80;
    border-radius: 14px; padding: 14px 30px;
    font-size: 24px; font-weight: 800; display: inline-block;
    box-shadow: 0 0 26px rgba(34,197,94,0.30); letter-spacing: 2px;
    font-family: 'Space Grotesk', sans-serif;
  }
  .sell-badge {
    background: rgba(239,68,68,0.12);
    border: 1px solid #ef4444; color: #f87171;
    border-radius: 14px; padding: 14px 30px;
    font-size: 24px; font-weight: 800; display: inline-block;
    box-shadow: 0 0 26px rgba(239,68,68,0.30); letter-spacing: 2px;
    font-family: 'Space Grotesk', sans-serif;
  }
  .wait-badge {
    background: rgba(245,158,11,0.12);
    border: 1px solid #f59e0b; color: #fbbf24;
    border-radius: 14px; padding: 14px 30px;
    font-size: 24px; font-weight: 800; display: inline-block;
    box-shadow: 0 0 26px rgba(245,158,11,0.25); letter-spacing: 2px;
    font-family: 'Space Grotesk', sans-serif;
  }

  /* ── Info / result box ── */
  .info-box {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 16px; padding: 20px; margin: 8px 0;
  }

  /* ── Metric card ── */
  .metric-card {
    background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
    padding: 16px; margin: 6px 0; text-align: center;
  }

  /* ── HOME: hero ── */
  .chee-hero {
    padding: 44px 8px 10px 8px;
  }
  .chee-hero .hi {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 54px; font-weight: 700; line-height: 1.04;
    color: #eef5f0; letter-spacing: -2px; margin: 0;
  }
  .chee-hero .hi .accent {
    background: linear-gradient(90deg, #22c55e, #86efac);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .chee-hero .sub {
    color: var(--muted); font-size: 15px; margin: 14px 0 0 2px; letter-spacing: 0.3px;
  }
  .chee-chip {
    display: inline-block; background: rgba(34,197,94,0.10); border: 1px solid rgba(74,222,128,0.35);
    color: #4ade80; font-size: 11.5px; font-weight: 700; letter-spacing: 1.5px;
    padding: 5px 14px; border-radius: 999px; margin: 14px 6px 0 2px; text-transform: uppercase;
  }
  .chee-section-label {
    color: var(--muted); font-size: 11.5px; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; margin: 26px 0 4px 2px;
  }

  /* ── HOME: agent cards ── */
  .agent-card {
    background: linear-gradient(180deg, #0d130f, #0a0e0b);
    border: 1px solid var(--border); border-radius: 18px;
    padding: 18px 16px 12px 16px; position: relative; min-height: 148px;
    transition: all .2s;
  }
  .agent-card:hover { border-color: var(--border-hi); box-shadow: 0 6px 30px rgba(34,197,94,0.10); }
  .agent-card .ic {
    width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
    background: rgba(34,197,94,0.10); border: 1px solid rgba(74,222,128,0.25); font-size: 19px; margin-bottom: 10px;
  }
  .agent-card .nm { color: #eef5f0; font-size: 15px; font-weight: 700; font-family: 'Space Grotesk', sans-serif; }
  .agent-card .ds { color: var(--muted); font-size: 12px; line-height: 1.45; margin-top: 4px; }
  .agent-card .live {
    position: absolute; top: 12px; right: 12px;
    background: rgba(34,197,94,0.12); border: 1px solid rgba(74,222,128,0.4);
    color: #4ade80; font-size: 9.5px; font-weight: 800; letter-spacing: 1.2px;
    padding: 2.5px 8px; border-radius: 999px;
  }
  .agent-card .soon {
    position: absolute; top: 12px; right: 12px;
    background: rgba(125,143,131,0.10); border: 1px solid rgba(125,143,131,0.35);
    color: #7d8f83; font-size: 9.5px; font-weight: 800; letter-spacing: 1.2px;
    padding: 2.5px 8px; border-radius: 999px;
  }

  /* ── Signal card (THISystem style) ── */
  .chee-signal-card {
    background: linear-gradient(180deg, rgba(34,197,94,0.05), rgba(11,16,13,0.9));
    border: 1px solid rgba(74,222,128,0.45); border-radius: 20px;
    padding: 20px 22px; margin: 10px 0;
    box-shadow: 0 0 40px rgba(34,197,94,0.10);
  }
  .chee-signal-card.sell {
    background: linear-gradient(180deg, rgba(239,68,68,0.05), rgba(16,11,11,0.9));
    border-color: rgba(248,113,113,0.45);
    box-shadow: 0 0 40px rgba(239,68,68,0.10);
  }
  .chee-signal-card .tag {
    display:inline-block; border-radius:999px; padding:5px 14px; font-size:11px;
    font-weight:800; letter-spacing:2px; text-transform:uppercase;
  }
  .chee-signal-card .rowline {
    display:flex; justify-content:space-between; align-items:center;
    border-bottom: 1px solid rgba(125,143,131,0.12); padding: 11px 0;
  }
  .chee-signal-card .k { color: var(--muted); font-size: 12px; letter-spacing: 1.8px; text-transform: uppercase; }
  .chee-signal-card .v { color: #eef5f0; font-size: 17px; font-weight: 700; font-family: 'JetBrains Mono', monospace; }

  /* ── Chat styling ── */
  [data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important; padding: 14px 16px !important; margin: 4px 0 !important;
  }
  [data-testid="stChatInput"] {
    background: #0c120e !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 24px rgba(34,197,94,0.10);
  }
  [data-testid="stChatInput"] textarea {
    background: transparent !important; color: #e8f0ea !important;
    caret-color: #4ade80;
  }
  [data-testid="stChatInput"] textarea::placeholder { color: #7d8f83 !important; }
  [data-testid="stChatInput"] button { background: transparent !important; }
  [data-testid="stChatInput"] button svg { fill: #4ade80 !important; }

  /* ── Inputs / selects (main area) ── */
  .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="base-input"] {
    background-color: #0c120e !important; border-color: var(--border-hi) !important;
    color: #e8f0ea !important; border-radius: 10px !important;
  }
  .stApp input, .stApp textarea { color: #e8f0ea !important; }
  .stApp [data-testid="stWidgetLabel"] p { color: #a8bcae !important; font-weight: 600; }

  /* ── Tabs (inside pages) ── */
  .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid var(--border) !important; }
  .stTabs [data-baseweb="tab"] {
    color: #7d8f83 !important; font-weight: 600; font-size: 14px;
  }
  .stTabs [aria-selected="true"] { color: #4ade80 !important; }
  .stTabs [data-baseweb="tab-highlight"] { background-color: #22c55e !important; }

  /* ── Expanders ── */
  .stExpander, [data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    background: var(--surface) !important;
  }
  .stExpander summary, [data-testid="stExpander"] summary { color: #a8bcae !important; font-weight: 600 !important; }
  .stExpander summary:hover, [data-testid="stExpander"] summary:hover { color: #4ade80 !important; }

  /* ── Upload area ── */
  [data-testid="stFileUploader"] {
    border: 1px dashed var(--border-hi) !important;
    border-radius: 14px !important; background: rgba(34,197,94,0.03) !important;
    padding: 6px !important;
  }
  [data-testid="stFileUploader"] section { background: transparent !important; }
  [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small { color: #a8bcae !important; }
  [data-testid="stFileUploader"] button {
    background: #101812 !important; color: #4ade80 !important;
    border: 1px solid var(--border-hi) !important; border-radius: 10px !important;
  }

  /* ── Divider ── */
  hr { border: none; border-top: 1px solid var(--border) !important; }

  /* ── Alerts ── */
  .stSuccess, [data-testid="stAlert"][data-baseweb="notification"] { border-radius: 12px !important; }
  .stSuccess { background: rgba(34,197,94,0.10) !important; border: 1px solid rgba(74,222,128,0.35) !important; }
  .stWarning { background: rgba(245,158,11,0.10) !important; border: 1px solid rgba(251,191,36,0.35) !important; }
  .stError   { background: rgba(239,68,68,0.10) !important; border: 1px solid rgba(248,113,113,0.35) !important; }
  .stInfo    { background: rgba(34,197,94,0.06) !important; border: 1px solid var(--border-hi) !important; }
  .stAlert p, [data-testid="stAlert"] p { color: #d5e5da !important; }

  /* ── Caption ── */
  .stApp .stCaption, .stApp caption, .stApp [data-testid="stCaptionContainer"] p {
    color: #7d8f83 !important; font-weight: 500 !important;
  }

  /* ── Metric widget ── */
  [data-testid="stMetric"] {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 14px; padding: 12px 16px;
  }
  [data-testid="stMetricValue"] { color: #eef5f0 !important; font-family: 'Space Grotesk', sans-serif; }
  [data-testid="stMetricLabel"] p { color: #7d8f83 !important; }

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #22c55e !important; }

  /* ══════════════════════════════════════════
     SIDEBAR — black glass, green accents
  ══════════════════════════════════════════ */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #070b08 0%, #050807 100%) !important;
    border-right: 1px solid #16211a !important;
  }
  section[data-testid="stSidebar"] *:not(button) {
    color: #cfe0d4 !important;
    -webkit-text-fill-color: #cfe0d4 !important;
  }
  section[data-testid="stSidebar"] h1,
  section[data-testid="stSidebar"] h2,
  section[data-testid="stSidebar"] h3,
  section[data-testid="stSidebar"] h4 {
    color: #eef5f0 !important;
    -webkit-text-fill-color: #eef5f0 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
  }
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
  section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
  section[data-testid="stSidebar"] [data-baseweb="form-control-label"],
  section[data-testid="stSidebar"] [data-baseweb="form-control-label"] * {
    color: #8fa896 !important;
    -webkit-text-fill-color: #8fa896 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
  }
  section[data-testid="stSidebar"] [data-baseweb="select"],
  section[data-testid="stSidebar"] [data-baseweb="select"] *,
  section[data-testid="stSidebar"] [data-baseweb="base-input"],
  section[data-testid="stSidebar"] [data-baseweb="base-input"] * {
    background-color: #0c120e !important;
    color: #e8f0ea !important;
    -webkit-text-fill-color: #e8f0ea !important;
    border-color: #1c2a21 !important;
  }
  section[data-testid="stSidebar"] input,
  section[data-testid="stSidebar"] textarea {
    background-color: #0c120e !important;
    color: #e8f0ea !important;
    -webkit-text-fill-color: #e8f0ea !important;
    border: 1px solid #1c2a21 !important;
    border-radius: 10px !important;
  }
  section[data-testid="stSidebar"] input::placeholder,
  section[data-testid="stSidebar"] textarea::placeholder {
    color: #5c6f63 !important;
    -webkit-text-fill-color: #5c6f63 !important;
  }
  section[data-testid="stSidebar"] hr {
    border-color: #16211a !important;
    border-top: 1px solid #16211a !important;
  }
  /* Sidebar nav buttons */
  section[data-testid="stSidebar"] .stButton>button {
    background: transparent; border: 1px solid transparent;
    color: #a8bcae !important; text-align: left; justify-content: flex-start;
    font-size: 14px !important; font-weight: 600 !important;
    padding: 8px 12px; border-radius: 10px; width: 100%;
  }
  section[data-testid="stSidebar"] .stButton>button:hover {
    background: rgba(34,197,94,0.07); color: #4ade80 !important;
    border-color: transparent; box-shadow: none; transform: none;
  }
  section[data-testid="stSidebar"] .stButton>button[kind="primary"],
  section[data-testid="stSidebar"] .stButton>button[data-testid="stBaseButton-primary"] {
    background: rgba(34,197,94,0.12); border: 1px solid rgba(74,222,128,0.35);
    color: #4ade80 !important; box-shadow: none; font-weight: 700 !important;
  }
  .chee-brand {
    display:flex; align-items:center; gap:10px; padding: 6px 4px 14px 4px;
  }
  .chee-brand .logo {
    width: 34px; height: 34px; border-radius: 10px;
    background: linear-gradient(135deg, #16a34a, #4ade80);
    display:flex; align-items:center; justify-content:center;
    font-size: 18px; box-shadow: 0 0 18px rgba(34,197,94,0.45);
  }
  .chee-brand .nm {
    font-family: 'Space Grotesk', sans-serif; font-size: 19px; font-weight: 700;
    color: #eef5f0 !important; -webkit-text-fill-color: #eef5f0 !important; letter-spacing: -0.5px;
  }
  .chee-brand .nm .g { color:#4ade80 !important; -webkit-text-fill-color:#4ade80 !important; }

  /* ══════════════════════════════════════════
     V2 — richer palette + white-widget fixes
  ══════════════════════════════════════════ */

  /* Aurora background: emerald + teal + cyan */
  [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(ellipse 70% 45% at 80% -8%, rgba(34,197,94,0.20), transparent 62%),
      radial-gradient(ellipse 55% 40% at 0% 30%, rgba(20,184,166,0.12), transparent 60%),
      radial-gradient(ellipse 65% 45% at 55% 115%, rgba(34,211,238,0.10), transparent 60%),
      linear-gradient(180deg, #060a08 0%, #04070a 100%) !important;
  }

  h1 {
    background: linear-gradient(90deg, #f0fdf4 5%, #4ade80 55%, #22d3ee 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .chee-hero .hi .accent {
    background: linear-gradient(90deg, #34d399 0%, #22d3ee 90%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .chee-chip {
    background: linear-gradient(90deg, rgba(34,197,94,0.14), rgba(34,211,238,0.10));
    border: 1px solid rgba(74,222,128,0.45);
  }

  /* Agent cards: gradient top edge + hover glow */
  .agent-card {
    background: linear-gradient(180deg, rgba(34,197,94,0.06), rgba(10,14,11,0.95) 45%);
    overflow: hidden;
  }
  .agent-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #22c55e 30%, #22d3ee 70%, transparent);
    opacity: .65;
  }
  .agent-card:hover {
    border-color: rgba(74,222,128,0.5);
    box-shadow: 0 10px 44px rgba(34,197,94,0.18), 0 0 0 1px rgba(34,211,238,0.10);
    transform: translateY(-2px);
  }
  .agent-card .ic {
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(34,211,238,0.12));
    border: 1px solid rgba(74,222,128,0.35);
    box-shadow: 0 0 16px rgba(34,197,94,0.15);
  }

  /* Signal cards: stronger presence */
  .chee-signal-card {
    box-shadow: 0 0 50px rgba(34,197,94,0.14), inset 0 1px 0 rgba(74,222,128,0.15);
  }
  .chee-signal-card.sell {
    box-shadow: 0 0 50px rgba(239,68,68,0.14), inset 0 1px 0 rgba(248,113,113,0.15);
  }

  /* MAIN AREA buttons — force dark (fix white buttons) */
  section[data-testid="stMain"] .stButton>button,
  section[data-testid="stMain"] button[data-testid="stBaseButton-secondary"] {
    background: linear-gradient(180deg, #121b14, #0c110e) !important;
    color: #cfe0d4 !important;
    border: 1px solid #2b4534 !important;
  }
  section[data-testid="stMain"] .stButton>button:hover {
    color: #4ade80 !important; border-color: #22c55e !important;
    box-shadow: 0 0 18px rgba(34,197,94,0.25) !important;
  }
  section[data-testid="stMain"] .stButton>button[kind="primary"],
  section[data-testid="stMain"] button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #16a34a, #10b981 55%, #0ea5a4) !important;
    color: #04120a !important; border: 1px solid #4ade80 !important;
    box-shadow: 0 4px 24px rgba(34,197,94,0.35) !important;
  }
  section[data-testid="stMain"] .stDownloadButton>button {
    background: linear-gradient(180deg, #121b14, #0c110e) !important;
    color: #4ade80 !important; border: 1px solid #2b4534 !important;
  }

  /* CHAT INPUT — force dark on every inner layer (fix white pill) */
  [data-testid="stChatInput"],
  [data-testid="stChatInput"] > div,
  [data-testid="stChatInput"] div[data-baseweb="textarea"],
  [data-testid="stChatInput"] div[data-baseweb="base-input"],
  [data-testid="stChatInputContainer"],
  [data-testid="stChatInputContainer"] > div,
  .stChatInput, .stChatInput > div {
    background: #0c130f !important;
    background-color: #0c130f !important;
    border-color: #2b4534 !important;
    color: #e8f0ea !important;
  }
  [data-testid="stChatInput"] { border-radius: 999px !important; box-shadow: 0 0 26px rgba(34,197,94,0.14) !important; }
  [data-testid="stChatInput"] textarea { background: transparent !important; color: #e8f0ea !important; }
  [data-testid="stChatInputSubmitButton"],
  [data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #16a34a, #0ea5a4) !important;
    border-radius: 999px !important; border: none !important;
  }
  [data-testid="stChatInputSubmitButton"] svg, [data-testid="stChatInput"] button svg { fill: #04120a !important; }
  [data-testid="stBottomBlockContainer"], [data-testid="stBottom"], [data-testid="stBottom"] > div {
    background: transparent !important;
  }

  /* Chat bubbles: user tinted green, assistant neutral */
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: linear-gradient(180deg, rgba(34,197,94,0.09), rgba(11,16,13,0.9)) !important;
    border-color: rgba(74,222,128,0.28) !important;
  }
  [data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #16a34a, #0ea5a4) !important; color: #04120a !important;
  }

  /* Sidebar active nav: gradient */
  section[data-testid="stSidebar"] .stButton>button[kind="primary"],
  section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, rgba(34,197,94,0.18), rgba(34,211,238,0.08)) !important;
    border: 1px solid rgba(74,222,128,0.4) !important;
    color: #4ade80 !important;
  }

  /* ══════════════════════════════════════════
     V3 — Luxury serif + champagne gold (THISystem DNA)
     Gold = brand & headings · Green = data & signals
  ══════════════════════════════════════════ */

  h1, h2 {
    font-family: 'Playfair Display', Georgia, serif !important;
    letter-spacing: 0 !important;
  }
  h1 {
    background: linear-gradient(90deg, #f5edda 15%, #e8c76e 85%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-weight: 600 !important; font-size: 44px !important;
  }
  h2 { color: #f3ead7 !important; font-weight: 600 !important; }
  h3 { color: #e9dfc8 !important; font-weight: 600 !important; }

  /* Hero — big serif greeting like the reference */
  .chee-hero { position: relative; padding: 52px 8px 6px 8px; }
  .chee-hero .hi {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 66px; font-weight: 600; line-height: 1.04;
    color: #f5edda; letter-spacing: 0; margin: 0;
    text-shadow: 0 2px 40px rgba(232,199,110,0.15);
  }
  .chee-hero .hi .accent {
    background: linear-gradient(90deg, #e8c76e 10%, #f7ecd0 90%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-style: italic;
  }
  .chee-hero .sub {
    color: #9aa89e; font-size: 15px; margin: 16px 0 0 3px; letter-spacing: 1.2px;
    text-transform: uppercase; font-size: 12.5px; font-weight: 600;
  }
  .chee-chip.gold {
    background: rgba(232,199,110,0.10);
    border: 1px solid rgba(232,199,110,0.5); color: #e8c76e;
  }
  .chee-chip.dim {
    background: rgba(125,143,131,0.08);
    border: 1px solid rgba(125,143,131,0.3); color: #9aa89e;
  }
  .chee-art {
    position: absolute; top: -6px; right: 6px; width: 330px;
    opacity: 0.95; pointer-events: none; filter: drop-shadow(0 0 30px rgba(232,199,110,0.18));
  }
  @media (max-width: 1100px) { .chee-art { width: 230px; opacity: .55; } }

  /* Sidebar brand → serif wordmark */
  .chee-brand .nm {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 21px; font-weight: 600; letter-spacing: .3px;
  }
  .chee-brand .nm .g {
    background: linear-gradient(90deg, #e8c76e, #f7ecd0);
    -webkit-background-clip: text;
    color: transparent !important; -webkit-text-fill-color: transparent !important;
    font-style: italic;
  }
  .chee-brand .logo {
    background: linear-gradient(135deg, #b9973f, #e8c76e) !important;
    box-shadow: 0 0 18px rgba(232,199,110,0.4) !important;
  }

  /* Agent cards: gold name, refined */
  .agent-card .nm { font-family: 'Playfair Display', Georgia, serif; font-size: 16.5px; color: #f3ead7; }
  .agent-card::before {
    background: linear-gradient(90deg, transparent, #e8c76e 30%, #22c55e 70%, transparent);
  }

  /* Chat: assistant answers clean & open, user tinted — like a real AI product */
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: transparent !important; border: none !important;
  }
  [data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #b9973f, #e8c76e) !important;
  }
  [data-testid="stChatInput"] { border-radius: 26px !important; }
  [data-testid="stChatInput"] textarea { font-size: 15px !important; }

  /* Section labels slightly warmer */
  .chee-section-label { color: #8a9a8e; }

  /* ── Segmented control (Read/Risk pills) — force dark + green/gold ── */
  [data-testid="stSegmentedControl"] button,
  div[data-baseweb="button-group"] button,
  button[data-testid="stBaseButton-segmented_control"] {
    background: #0c130f !important;
    color: #8fa896 !important;
    border: 1px solid #2b4534 !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
  }
  [data-testid="stSegmentedControl"] button p,
  div[data-baseweb="button-group"] button p { color: inherit !important; }
  [data-testid="stSegmentedControl"] button:hover,
  div[data-baseweb="button-group"] button:hover {
    color: #4ade80 !important; border-color: rgba(74,222,128,0.5) !important;
  }
  /* SELECTED pill — solid green, dark text, unmistakable */
  button[kind="segmented_controlActive"],
  button[data-testid="stBaseButton-segmented_controlActive"],
  [data-testid="stSegmentedControl"] button[aria-checked="true"],
  [data-testid="stSegmentedControl"] button[aria-selected="true"],
  div[data-baseweb="button-group"] button[aria-checked="true"],
  div[data-baseweb="button-group"] button[aria-selected="true"] {
    background: linear-gradient(135deg, #16a34a, #22c55e) !important;
    color: #04120a !important;
    border: 1px solid #4ade80 !important;
    box-shadow: 0 0 16px rgba(34,197,94,0.45) !important;
    font-weight: 800 !important;
  }
  button[kind="segmented_controlActive"] p,
  button[data-testid="stBaseButton-segmented_controlActive"] p,
  [data-testid="stSegmentedControl"] button[aria-checked="true"] p,
  [data-testid="stSegmentedControl"] button[aria-selected="true"] p,
  div[data-baseweb="button-group"] button[aria-checked="true"] p,
  div[data-baseweb="button-group"] button[aria-selected="true"] p {
    color: #04120a !important; font-weight: 800 !important;
  }

  /* =========================================================
     FINAL CONTROL OVERRIDES — Streamlit 1.40+ / 1.5x DOMs
     Keep this block last so BaseWeb cannot paint white layers.
     ========================================================= */
  [data-testid="stSelectbox"] [data-baseweb="select"],
  [data-testid="stSelectbox"] [data-baseweb="select"] > div,
  [data-testid="stMultiSelect"] [data-baseweb="select"],
  [data-testid="stMultiSelect"] [data-baseweb="select"] > div,
  div[data-baseweb="select"] > div {
    background: #0b120e !important;
    background-color: #0b120e !important;
    color: #e8f0ea !important;
    -webkit-text-fill-color: #e8f0ea !important;
    border-color: #2b4534 !important;
    box-shadow: none !important;
  }
  [data-testid="stSelectbox"] [data-baseweb="select"] *,
  [data-testid="stMultiSelect"] [data-baseweb="select"] * {
    color: #e8f0ea !important;
    -webkit-text-fill-color: #e8f0ea !important;
  }
  [data-testid="stSelectbox"] svg,
  [data-testid="stMultiSelect"] svg { fill: #8fb49b !important; color: #8fb49b !important; }

  /* Dropdown portal is rendered outside the widget tree. */
  [data-baseweb="popover"],
  [data-baseweb="popover"] > div,
  [data-baseweb="menu"],
  ul[role="listbox"] {
    background: #0b120e !important;
    background-color: #0b120e !important;
    color: #e8f0ea !important;
    border-color: #2b4534 !important;
  }
  [role="option"], li[role="option"] {
    background: #0b120e !important;
    color: #d9e8dd !important;
    -webkit-text-fill-color: #d9e8dd !important;
  }
  [role="option"]:hover, li[role="option"]:hover {
    background: #14251a !important; color: #6ee7a0 !important;
  }
  [role="option"][aria-selected="true"], li[role="option"][aria-selected="true"] {
    background: #16a34a !important; color: #04120a !important;
    -webkit-text-fill-color: #04120a !important;
  }

  /* Text, number, date, time and textarea controls — every nested BaseWeb layer. */
  [data-testid="stTextInput"] [data-baseweb="input"],
  [data-testid="stTextInput"] [data-baseweb="base-input"],
  [data-testid="stNumberInput"] [data-baseweb="input"],
  [data-testid="stNumberInput"] [data-baseweb="base-input"],
  [data-testid="stTextArea"] [data-baseweb="textarea"],
  [data-testid="stDateInput"] [data-baseweb="input"],
  [data-testid="stTimeInput"] [data-baseweb="input"] {
    background: #0b120e !important; background-color: #0b120e !important;
    color: #e8f0ea !important; border-color: #2b4534 !important;
  }
  [data-testid="stTextInput"] input,
  [data-testid="stNumberInput"] input,
  [data-testid="stDateInput"] input,
  [data-testid="stTimeInput"] input,
  [data-testid="stTextArea"] textarea {
    background: transparent !important; color: #e8f0ea !important;
    -webkit-text-fill-color: #e8f0ea !important; caret-color: #4ade80 !important;
  }
  input::placeholder, textarea::placeholder {
    color: #6f8576 !important; -webkit-text-fill-color: #6f8576 !important; opacity: 1 !important;
  }

  /* Segmented controls: covers st.segmented_control, st.pills and radio fallback. */
  [data-testid="stButtonGroup"],
  [data-testid="stSegmentedControl"],
  div[data-baseweb="button-group"],
  [role="radiogroup"] {
    background: transparent !important; background-color: transparent !important;
  }
  [data-testid="stButtonGroup"] button,
  [data-testid="stSegmentedControl"] button,
  div[data-baseweb="button-group"] button,
  [role="radiogroup"] button,
  button[kind*="segmented"],
  button[data-testid*="segmented"] {
    background: #0b120e !important; background-color: #0b120e !important;
    color: #cfe0d4 !important; -webkit-text-fill-color: #cfe0d4 !important;
    border-color: #2b4534 !important; box-shadow: none !important;
  }
  [data-testid="stButtonGroup"] button *,
  [data-testid="stSegmentedControl"] button *,
  div[data-baseweb="button-group"] button *,
  [role="radiogroup"] button * {
    color: inherit !important; -webkit-text-fill-color: inherit !important;
  }
  [data-testid="stButtonGroup"] button:hover,
  [data-testid="stSegmentedControl"] button:hover,
  div[data-baseweb="button-group"] button:hover {
    background: #13251a !important; color: #6ee7a0 !important; border-color: #22c55e !important;
  }
  [data-testid="stButtonGroup"] button[aria-pressed="true"],
  [data-testid="stButtonGroup"] button[aria-checked="true"],
  [data-testid="stButtonGroup"] button[aria-selected="true"],
  [data-testid="stSegmentedControl"] button[aria-pressed="true"],
  [data-testid="stSegmentedControl"] button[aria-checked="true"],
  [data-testid="stSegmentedControl"] button[aria-selected="true"],
  div[data-baseweb="button-group"] button[aria-pressed="true"],
  div[data-baseweb="button-group"] button[aria-checked="true"],
  div[data-baseweb="button-group"] button[aria-selected="true"],
  button[kind="segmented_controlActive"],
  button[data-testid="stBaseButton-segmented_controlActive"] {
    background: #22c55e !important; background-color: #22c55e !important;
    color: #031208 !important; -webkit-text-fill-color: #031208 !important;
    border-color: #6ee7a0 !important; box-shadow: 0 0 16px rgba(34,197,94,.42) !important;
    font-weight: 800 !important;
  }
  [data-testid="stButtonGroup"] button[aria-pressed="true"] *,
  [data-testid="stButtonGroup"] button[aria-checked="true"] *,
  [data-testid="stButtonGroup"] button[aria-selected="true"] *,
  [data-testid="stSegmentedControl"] button[aria-pressed="true"] *,
  [data-testid="stSegmentedControl"] button[aria-checked="true"] *,
  [data-testid="stSegmentedControl"] button[aria-selected="true"] *,
  div[data-baseweb="button-group"] button[aria-pressed="true"] *,
  div[data-baseweb="button-group"] button[aria-checked="true"] *,
  div[data-baseweb="button-group"] button[aria-selected="true"] * {
    color: #031208 !important; -webkit-text-fill-color: #031208 !important; font-weight: 800 !important;
  }
  [data-testid="stButtonGroup"] button:disabled,
  [data-testid="stSegmentedControl"] button:disabled,
  div[data-baseweb="button-group"] button:disabled {
    background: #0a100c !important; color: #718478 !important;
    -webkit-text-fill-color: #718478 !important; opacity: .72 !important;
  }

  /* Upload dropzone and secondary buttons. */
  [data-testid="stFileUploaderDropzone"],
  [data-testid="stFileUploaderDropzone"] > div,
  [data-testid="stFileUploader"] section {
    background: #0b120e !important; background-color: #0b120e !important;
    color: #cfe0d4 !important; border-color: #2b4534 !important;
  }
  [data-testid="stFileUploader"] button,
  [data-testid="stFileUploaderDropzone"] button {
    background: #121b14 !important; color: #4ade80 !important; border-color: #2b4534 !important;
  }

  /* Labels, help copy, captions and tooltips must remain readable. */
  [data-testid="stWidgetLabel"] p,
  [data-testid="stCaptionContainer"] p,
  .stCaption p,
  [data-testid="stMarkdownContainer"] small {
    color: #91a99a !important; -webkit-text-fill-color: #91a99a !important; opacity: 1 !important;
  }
  [data-baseweb="tooltip"], [role="tooltip"] {
    background: #101812 !important; color: #e8f0ea !important; border: 1px solid #2b4534 !important;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# SIDEBAR — Navigation (THISystem style)
# ══════════════════════════════════════════════════════════
_NAV_PAGES = [
    ("🏠", "Home"),
    ("✨", "AI Analyst"),
    ("📷", "Read My Chart"),
    ("🎯", "Market Scout"),
    ("🧱", "Key Levels"),
    ("🌐", "Markets"),
]

if "nav" not in st.session_state:
    st.session_state["nav"] = "Home"

with st.sidebar:
    st.markdown("""
<div class='chee-brand'>
  <div class='logo'>⚡</div>
  <div class='nm'>Chee <span class='g'>AI</span></div>
</div>
""", unsafe_allow_html=True)

    for _ic, _pg in _NAV_PAGES:
        _active = st.session_state["nav"] == _pg
        if st.button(f"{_ic}  {_pg}", key=f"nav_{_pg}",
                     use_container_width=True,
                     type="primary" if _active else "secondary"):
            if st.session_state["nav"] != _pg:
                st.session_state["nav"] = _pg
                st.rerun()

    st.divider()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.caption(f"Build {APP_VERSION}")

    st.caption("Gemini key = FREE (aistudio.google.com) · Claude key = best accuracy (console.anthropic.com)")

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
        "gemini-3.5-flash",
        "gemini-3.1-pro-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "claude-fable-5",
        "claude-opus-4-8",
        "claude-sonnet-5",
    ]
    _saved_model = st.session_state.get("saved_model_choice", "gemini-3.5-flash")
    _model_idx = _model_options.index(_saved_model) if _saved_model in _model_options else 0

    model_choice = st.selectbox(
        "🤖 AI Model",
        _model_options,
        index=_model_idx,
        help="✅ Gemini = free tier available (3.5 Flash = newest, fast & smart) | Claude = paid (Fable 5 = most intelligent, Opus 4.8 / Sonnet 5 = strong & cheaper)",
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

    st.markdown("### 🎯 The Method — SNR First")
    strategies = [
        ("🎯", "Support & Resistance — the foundation"),
        ("📈", "Trend + Market Structure"),
        ("🕯️", "Rejection Candles (Pin Bar / Engulfing)"),
        ("📐", "Fib 38.2 / 50 / 61.8 — bonus confluence"),
        ("🧮", "Risk 1-2% · Min 1:2 R:R"),
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
            st.caption("Optional fallback when Yahoo Finance is unavailable.")

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


# ── Main Layout / Page Router ──────────────────────────────
_nav = st.session_state.get("nav", "Home")

# ════════════════════════════════════════════════════════════
# HOME — Greeting hero + AI agent cards (THISystem style)
# ════════════════════════════════════════════════════════════
if _nav == "Home":
    import datetime as _dt
    _now  = _dt.datetime.now()
    _hour = _now.hour
    if 5 <= _hour < 12:
        _greet = "Good morning,"
    elif 12 <= _hour < 18:
        _greet = "Good afternoon,"
    else:
        _greet = "Good evening,"
    if 7 <= _hour < 15:
        _session = "Tokyo Session"
    elif 15 <= _hour < 21:
        _session = "London Session"
    else:
        _session = "New York Session"
    _date_str = _now.strftime("%A · %b %d")

    st.markdown(f"""
<div class='chee-hero'>
  <svg class='chee-art' viewBox='0 0 400 250' xmlns='http://www.w3.org/2000/svg'>
    <defs>
      <linearGradient id='gGold' x1='0' y1='0' x2='1' y2='1'>
        <stop offset='0' stop-color='#f7ecd0'/><stop offset='0.5' stop-color='#e8c76e'/><stop offset='1' stop-color='#8a6f2e'/>
      </linearGradient>
      <radialGradient id='gOrb' cx='0.5' cy='0.5' r='0.5'>
        <stop offset='0' stop-color='#e8c76e' stop-opacity='0.28'/>
        <stop offset='0.6' stop-color='#e8c76e' stop-opacity='0.07'/>
        <stop offset='1' stop-color='#e8c76e' stop-opacity='0'/>
      </radialGradient>
    </defs>
    <circle cx='265' cy='125' r='120' fill='url(#gOrb)'/>
    <ellipse cx='265' cy='125' rx='112' ry='36' fill='none' stroke='#e8c76e' stroke-opacity='.28' stroke-width='1' transform='rotate(-18 265 125)'/>
    <ellipse cx='265' cy='125' rx='134' ry='46' fill='none' stroke='#e8c76e' stroke-opacity='.13' stroke-width='1' transform='rotate(-18 265 125)'/>
    <circle cx='168' cy='158' r='2.6' fill='#f0d68a'/>
    <circle cx='368' cy='94' r='2' fill='#f0d68a' opacity='.8'/>
    <path d='M 332 78 A 82 82 0 1 0 332 172' fill='none' stroke='url(#gGold)' stroke-width='7' stroke-linecap='round'/>
    <g stroke-linecap='round'>
      <line x1='237' y1='98' x2='237' y2='168' stroke='#8a6f2e' stroke-width='2'/>
      <rect x='229' y='114' width='16' height='38' rx='3' fill='#b9973f'/>
      <line x1='267' y1='80' x2='267' y2='154' stroke='#c7a651' stroke-width='2'/>
      <rect x='259' y='94' width='16' height='42' rx='3' fill='#e8c76e'/>
      <line x1='297' y1='62' x2='297' y2='134' stroke='#e8c76e' stroke-width='2'/>
      <rect x='289' y='76' width='16' height='40' rx='3' fill='#f0d68a'/>
    </g>
    <path d='M 344 50 l 3.5 9 9 3.5 -9 3.5 -3.5 9 -3.5 -9 -9 -3.5 9 -3.5 z' fill='#f7ecd0' opacity='.95'/>
    <path d='M 196 58 l 2.3 6 6 2.3 -6 2.3 -2.3 6 -2.3 -6 -6 -2.3 6 -2.3 z' fill='#e8c76e' opacity='.7'/>
    <circle cx='352' cy='152' r='1.6' fill='#f7ecd0' opacity='.8'/>
    <circle cx='206' cy='190' r='1.3' fill='#f7ecd0' opacity='.5'/>
    <circle cx='182' cy='104' r='1.1' fill='#f7ecd0' opacity='.6'/>
    <circle cx='318' cy='34' r='1.2' fill='#f7ecd0' opacity='.6'/>
  </svg>
  <p class='hi'>{_greet}<br><span class='accent'>Chee</span></p>
  <p class='sub'>Your Personal AI Financial Analyst</p>
  <div>
    <span class='chee-chip gold'>👑 Pro Trader</span>
    <span class='chee-chip'>● {_session} · Live</span>
    <span class='chee-chip dim'>{_date_str}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Quick ask → routes to AI Analyst ──
    _home_q = st.chat_input("What moved the market this morning?  ·  问我任何交易问题…", key="home_quick_ask")
    if _home_q:
        st.session_state["pending_question"] = _home_q
        st.session_state["nav"] = "AI Analyst"
        st.rerun()

    # ── Suggested prompts (like ChatGPT) ──
    _SUGGESTIONS = [
        ("🌅  What moved the market this morning?", "AI Analyst", "What moved the market this morning?"),
        ("🥇  Can I buy gold right now?",            "AI Analyst", "Can I buy gold right now? 现在可以买黄金吗?"),
        ("🔍  Scan the market for opportunities",    "Market Scout", None),
    ]
    _sug_cols = st.columns(len(_SUGGESTIONS), gap="small")
    for _si, (_slabel, _spage, _sq) in enumerate(_SUGGESTIONS):
        with _sug_cols[_si]:
            if st.button(_slabel, key=f"home_sug_{_si}", use_container_width=True):
                if _sq:
                    st.session_state["pending_question"] = _sq
                st.session_state["nav"] = _spage
                st.rerun()

    st.markdown("<div class='chee-section-label'>AI Agents</div>", unsafe_allow_html=True)

    _AGENTS = [
        ("✨", "AI Analyst",    "Ask anything — it fetches live prices & charts automatically, like ChatGPT for trading.", "AI Analyst"),
        ("📷", "Read My Chart", "Drop a screenshot — choose Auto, Signal or Analysis, and your risk style.",               "Read My Chart"),
        ("🎯", "Market Scout",  "AI scans the whole market and picks today's best opportunities for you.",                  "Market Scout"),
        ("🧱", "Key Levels",    "Key S/R zones for any pair — with the reasoning behind every level.",                      "Key Levels"),
        ("🌐", "Markets",       "Economic calendar and live charts in one place.",                                           "Markets"),
    ]

    for _row_start in range(0, len(_AGENTS), 3):
        _row_agents = _AGENTS[_row_start:_row_start + 3]
        _cols = st.columns(3, gap="small")
        for _ci, (_a_ic, _a_nm, _a_ds, _a_pg) in enumerate(_row_agents):
            with _cols[_ci]:
                st.markdown(f"""
<div class='agent-card'>
  <span class='live'>LIVE</span>
  <div class='ic'>{_a_ic}</div>
  <div class='nm'>{_a_nm}</div>
  <div class='ds'>{_a_ds}</div>
</div>
""", unsafe_allow_html=True)
                if st.button("Open →", key=f"agent_open_{_a_pg}", use_container_width=True):
                    st.session_state["nav"] = _a_pg
                    st.rerun()
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.markdown("<div class='chee-section-label'>The Method 方法论</div>", unsafe_allow_html=True)
    st.markdown("""
<div class='info-box' style='display:flex;gap:26px;flex-wrap:wrap;align-items:center'>
  <span style='color:#4ade80;font-weight:800;font-family:Space Grotesk,sans-serif;font-size:15px'>SNR First 支撑阻力为根</span>
  <span style='color:#7d8f83;font-size:13px'>① Trend direction</span>
  <span style='color:#7d8f83;font-size:13px'>② Pullback to tested S/R level</span>
  <span style='color:#7d8f83;font-size:13px'>③ Rejection candle → enter</span>
  <span style='color:#7d8f83;font-size:13px'>④ Fib 38.2/50/61.8 = bonus confluence</span>
  <span style='color:#7d8f83;font-size:13px'>⑤ Risk 1-2% · min 1:2 R:R</span>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# MARKET SCOUT — AI scans the market, picks today's best setups
# ════════════════════════════════════════════════════════════
if _nav == "Market Scout":
    st.markdown("## Market Scout")
    st.caption("AI fetches live data across the market and picks today's best opportunities · AI 自动扫描市场，挑出今天最有机会的交易对")

    _SCOUT_UNIVERSE = ["XAU/USD", "XAG/USD", "EUR/USD", "GBP/USD", "USD/JPY",
                       "AUD/USD", "NZD/USD", "USD/CAD", "USD/CHF", "BTC/USD", "ETH/USD"]
    _scout_sel = st.multiselect(
        "Markets to scan 扫描范围",
        _SCOUT_UNIVERSE,
        default=["XAU/USD", "XAG/USD", "EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD"],
        help="Each market is checked on D1 + H4 + H1. Yahoo Finance is used first; Twelve Data is fallback.",
    )

    if st.button("🔍 Scan the Market Now 立即扫描", type="primary", use_container_width=True):
        if not api_key:
            st.warning("👈 Enter your AI API key in the sidebar first.")
        elif not _scout_sel:
            st.warning("Pick at least one market to scan.")
        else:
            _scout_labels = {
                "XAU/USD": "Gold (XAUUSD)", "XAG/USD": "Silver (XAGUSD)",
                "EUR/USD": "EURUSD", "GBP/USD": "GBPUSD", "USD/JPY": "USDJPY",
                "AUD/USD": "AUDUSD", "NZD/USD": "NZDUSD", "USD/CAD": "USDCAD",
                "USD/CHF": "USDCHF", "BTC/USD": "Bitcoin (BTCUSD)",
                "ETH/USD": "Ethereum (ETHUSD)",
            }
            _digests = []
            _snapshots = {}
            _prog = st.progress(0.0, text="Fetching live data…")
            for _i, _sym in enumerate(_scout_sel):
                try:
                    _label = _scout_labels[_sym]
                    _d1, _src1 = fetch_candles_any(_label, "D1", twelve_data_key)
                    _h4, _src4 = fetch_candles_any(_label, "H4", twelve_data_key)
                    _h1, _srch = fetch_candles_any(_label, "H1", twelve_data_key)
                    _snapshot = build_setup_snapshot(_d1, _h4, _h1)
                    _snapshots[_sym] = _snapshot
                    _digests.append(
                        build_market_digest(_d1, _sym, "D1") + "\n" +
                        build_market_digest(_h4, _sym, "H4") + "\n" +
                        build_market_digest(_h1, _sym, "H1") + "\n" +
                        "  DETERMINISTIC GATE RESULT: " + json.dumps({
                            "direction": _snapshot["direction"], "grade": _snapshot["grade"],
                            "gates": _snapshot["gates"], "failed": _snapshot["failed_gates"],
                            "entry": _snapshot["entry"], "stop": _snapshot["stop"],
                            "target": _snapshot["target"], "rr": _snapshot["rr"],
                            "fib_overlap": _snapshot["fib_overlap"],
                        }, default=str)
                    )
                except Exception as _se:
                    _digests.append(f"{_sym}: DATA UNAVAILABLE ({_se})")
                _prog.progress((_i + 1) / len(_scout_sel), text=f"Fetched {_sym} ({_i + 1}/{len(_scout_sel)})")
            _prog.empty()

            _nl = "\n\n"
            _scout_prompt = f"""You are Chee AI Market Scout — an elite trading analyst. Your foundation is SUPPORT & RESISTANCE (SNR is the root of trading); Fibonacci 38.2%/50%/61.8% is bonus confluence only.

Below is LIVE market data (fetched seconds ago) for {len(_scout_sel)} markets:

{_nl.join("── " + d for d in _digests)}

TASK: Explain and rank up to 3 opportunities, but ONLY where DETERMINISTIC GATE RESULT has grade A or A+.
You may never override a WAIT result or invent different entry/SL/TP values. A means all five gates passed.
A+ means the same gates passed plus Fib 38.2/50/61.8 overlaps the tested S/R zone.
If no deterministic result qualifies, return zero picks. Quality over quantity.

Output STRICT JSON only, no other text. Never put double-quote characters inside text values (use single quotes if needed):
{{"market_note_en": "1-2 sentence market overview",
 "market_note_cn": "中文一两句市场总览",
 "picks": [
   {{"pair": "XAU/USD", "direction": "BUY or SELL", "grade": "A or A+", "confidence": 7,
     "entry_zone": "copy deterministic entry", "stop_loss": "copy deterministic stop", "take_profit": "copy deterministic target",
     "reason_en": "2-3 sentences citing the exact fib levels and swing levels from the data",
     "reason_cn": "中文理由 2-3 句，引用具体价位"}}
 ]}}"""
            with st.spinner("🤖 AI is analysing the whole market…"):
                try:
                    _raw = ai_text_call(_scout_prompt, api_key, model_choice, json_mode=True)
                    _parsed = parse_ai_json(_raw, api_key, model_choice)
                    if not _parsed:
                        st.error("❌ AI returned an unreadable response — hit Scan again. AI 返回格式异常，请再扫描一次。")
                    else:
                        # The model writes explanations only. Enforce the program's gates and levels.
                        _safe_picks = []
                        for _pick in (_parsed.get("picks", []) or []):
                            _pair = str(_pick.get("pair", "")).upper().replace(" ", "")
                            _matched = next((s for s in _snapshots if s.replace(" ", "").upper() == _pair), None)
                            _snap = _snapshots.get(_matched) if _matched else None
                            if not _snap or _snap["grade"] not in ("A", "A+"):
                                continue
                            _pick.update({
                                "pair": _matched, "direction": _snap["direction"], "grade": _snap["grade"],
                                "entry_zone": f"{_snap['entry']:.6g}", "stop_loss": f"{_snap['stop']:.6g}",
                                "take_profit": f"{_snap['target']:.6g}", "rr": round(_snap["rr"], 2),
                                "confidence": 8 if _snap["grade"] == "A+" else 7,
                            })
                            _safe_picks.append(_pick)
                        _parsed["picks"] = _safe_picks[:3]
                        st.session_state["scout_result"]  = _parsed
                        st.session_state["scout_digests"] = _digests
                        import datetime as _dt_sc
                        st.session_state["scout_time"] = _dt_sc.datetime.now().strftime("%b %d, %H:%M")
                except Exception as _ae:
                    st.error(f"❌ Scout failed: {_ae}")

    _sr = st.session_state.get("scout_result")
    if _sr:
        st.caption(f"🕐 Last scan: {st.session_state.get('scout_time', '—')} · data cached 5 min")
        if _sr.get("market_note_en") or _sr.get("market_note_cn"):
            st.markdown(f"""<div class='info-box'><p style='color:#cfe0d4;font-size:14px;margin:0'>
🧭 {_sr.get('market_note_en', '')}<br>
<span style='color:#7d8f83'>{_sr.get('market_note_cn', '')}</span></p></div>""", unsafe_allow_html=True)

        _picks = _sr.get("picks", []) or []
        if not _picks:
            st.info("😴 No high-confluence setups right now — patience is a position too. 目前没有高质量机会，耐心等待。")
        for _p in _picks:
            _dirn = str(_p.get("direction", "WAIT")).upper()
            _pc = {
                "BUY":  ("#4ade80", "rgba(34,197,94,0.15)",  "#22c55e", ""),
                "SELL": ("#f87171", "rgba(239,68,68,0.15)",  "#ef4444", "sell"),
            }.get(_dirn, ("#fbbf24", "rgba(245,158,11,0.15)", "#f59e0b", ""))
            st.markdown(f"""
<div class='chee-signal-card {_pc[3]}'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px'>
    <span style='color:#eef5f0;font-size:19px;font-weight:800;font-family:Space Grotesk,sans-serif'>{_p.get('pair', '—')}</span>
    <span class='tag' style='background:{_pc[1]};border:1px solid {_pc[2]};color:{_pc[0]};font-size:14px;padding:6px 20px'>{_dirn} · {_p.get('grade', '—')} · {_p.get('confidence', '—')}/10</span>
  </div>
  <div class='rowline'><span class='k'>Entry 入场</span><span class='v'>{_p.get('entry_zone', '—')}</span></div>
  <div class='rowline'><span class='k'>Stop Loss 止损</span><span class='v'>{_p.get('stop_loss', '—')}</span></div>
  <div class='rowline'><span class='k'>Take Profit 目标</span><span class='v'>{_p.get('take_profit', '—')}</span></div>
  <div class='rowline'><span class='k'>Risk : Reward</span><span class='v'>1 : {_p.get('rr', '—')}</span></div>
  <p style='color:#cfe0d4;font-size:13.5px;margin:12px 0 4px 0'>{_p.get('reason_en', '')}</p>
  <p style='color:#7d8f83;font-size:13px;margin:0'>{_p.get('reason_cn', '')}</p>
</div>""", unsafe_allow_html=True)

        if _picks:
            st.caption("⚠️ AI analysis, not financial advice — always confirm on your own chart before entering. 仅供参考，入场前请自行确认。")
        with st.expander("📊 Raw scan data 原始扫描数据", expanded=False):
            for _d in st.session_state.get("scout_digests", []):
                st.code(_d)
    else:
        st.markdown("""
<div class='info-box' style='text-align:center;padding:44px 20px'>
<p style='color:#cfe0d4;font-size:15px;font-weight:600;margin:0'>Hit Scan — I'll fetch live data for every market and pick today's best setups.<br><br>
<span style='color:#7d8f83;font-size:13px;font-weight:400'>点击扫描 — AI 获取所有市场的实时数据后，自动挑出今天最有机会的交易对，并给出入场、止损、目标位和理由。可能是一个，也可能是多个，AI 自己判断。</span></p>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# READ MY CHART — upload + read-mode + risk-profile (THISystem style)
# ════════════════════════════════════════════════════════════
if _nav == "Read My Chart":
    st.markdown("<div style='text-align:center;margin-top:10px'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center'>Read my chart</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#7d8f83;font-size:14px;margin-top:-6px'>"
                "Drop a screenshot, choose how you want it read, and I'll detect the pair and timeframe.<br>"
                "上传截图，选择解读方式和风险偏好，AI 自动识别品种和时间框架。</p>", unsafe_allow_html=True)

    def _rc_seg(label, options, default_opt, key):
        """Segmented control with radio fallback for older Streamlit versions."""
        try:
            _v = st.segmented_control(label, options, default=default_opt, key=key)
            return _v or default_opt
        except Exception:
            return st.radio(label, options, index=options.index(default_opt), key=f"{key}_radio", horizontal=True)

    _rc_pad_l, _rc_mid, _rc_pad_r = st.columns([1, 2.6, 1])
    with _rc_mid:
        _rc_src_opts = ["📡 Auto-fetch live data 自动抓取实时数据"] + list(_YF_TICKERS.keys())
        _rc_src = st.selectbox(
            "Source 来源 — auto-fetch or upload 自动抓取或上传",
            _rc_src_opts, index=1, key="rc_src",
            help="Pick a market → live candles are fetched from Yahoo Finance automatically (free, no key). Or choose the first option to upload your own screenshot.",
        )
        _rc_auto = _rc_src != _rc_src_opts[0]

        _rc_file = None
        if _rc_auto:
            _rc_atf = _rc_seg("Timeframe 时间框架", ["H1", "H4", "D1"], "H1", "rc_atf")
            st.caption(f"📡 I'll fetch live {_rc_src} {_rc_atf} candles automatically and read them — no screenshot needed. 自动抓取实时K线，无需截图。")
        else:
            _rc_file = st.file_uploader(
                "Drop chart, or click to choose 拖入图表或点击选择",
                type=["png", "jpg", "jpeg", "webp"],
                key="rc_upload",
            )
            if _rc_file:
                st.image(_rc_file, use_container_width=True)

        _rc_mode_label = _rc_seg("Read 解读方式", ["⚡ Auto", "🎯 Signal", "🔬 Analysis"], "⚡ Auto", "rc_mode")
        _rc_mode = {"⚡ Auto": "auto", "🎯 Signal": "signal", "🔬 Analysis": "analysis"}[_rc_mode_label]
        st.caption({
            "auto":     "I'll pick what's most useful for this chart — full read + setup. 自动：完整解读+交易方案。",
            "signal":   "Straight to the trade: signal, entry, SL, TPs, why, and invalidation. 只要信号：直接给交易方案。",
            "analysis": "Educational deep-dive — learn to see what the AI sees. No trade pushing. 深度分析：教学式详解。",
        }[_rc_mode])

        _rc_risk_label = _rc_seg("Risk 风险偏好", ["🛡️ Conservative", "⚖️ Balanced", "🔥 Aggressive"], "⚖️ Balanced", "rc_risk")
        _rc_risk = {"🛡️ Conservative": "conservative", "⚖️ Balanced": "balanced", "🔥 Aggressive": "aggressive"}[_rc_risk_label]
        st.caption({
            "conservative": "A/A+ setups only, full confirmation, wider SL, max 1% risk. 保守：只做高确定性。",
            "balanced":     "Confirmed setups, 1-2% risk, minimum 1:2 R:R. 平衡：标准规则。",
            "aggressive":   "First confirmed retest, smaller buffer, max 1.5% risk; all A/A+ gates still required. 激进：仍需确认，只调整入场与风险。",
        }[_rc_risk])

        with st.expander("✏️ Pair & context (optional 可选)", expanded=False):
            _rc_pair = st.text_input("Pair / instrument (blank = auto-detect)", placeholder="e.g. XAUUSD H1", key="rc_pair")
            _rc_note = st.text_area("Note for the AI", placeholder="e.g. I'm already long from 4120…", height=70, key="rc_note")

        _rc_go = st.button("✨ Analyse Chart", type="primary", use_container_width=True,
                           disabled=not (_rc_auto or _rc_file), key="rc_go")

    if _rc_go and (_rc_auto or _rc_file):
        if not api_key:
            st.warning("👈 Enter your AI API key in the sidebar first.")
        else:
            try:
                _rc_ctx = _rc_note.strip() if _rc_note else ""
                if _rc_auto:
                    # ── Auto-fetch live candles (yfinance → Twelve Data fallback) ──
                    with st.spinner(f"📡 Fetching live {_rc_src} {_rc_atf} data…"):
                        _ydf, _rc_srcname = fetch_candles_any(_rc_src, _rc_atf, twelve_data_key)
                        _rc_img = generate_chart_image_from_df(_ydf.tail(160), _rc_src, _rc_atf)
                    _rc_ctx = (
                        f"[LIVE MARKET DATA — fetched seconds ago via {_rc_srcname}, treat as ground truth. "
                        "The chart image was generated from this exact data.]\n"
                        + build_market_digest(_ydf, _rc_src, _rc_atf)
                        + (f"\n\nTrader note: {_rc_ctx}" if _rc_ctx else "")
                    )
                    _rc_market = _rc_src
                    _rc_tf = _rc_atf
                else:
                    _rc_img = Image.open(_rc_file)
                    _rc_market = _rc_pair.strip() if _rc_pair and _rc_pair.strip() else \
                        "(auto-detect the instrument from the chart itself)"
                    _rc_tf = "(auto-detect the timeframe from the chart itself)" if not (_rc_pair and _rc_pair.strip()) \
                        else "(as stated or visible on the chart)"
                with st.spinner("🤖 Reading the chart — 15-30 seconds…"):
                    _rc_text = analyze_chart_with_ai(
                        _rc_img, api_key, model_choice,
                        _rc_market, _rc_tf,
                        context=_rc_ctx,
                        mode=_rc_mode, risk=_rc_risk,
                    )
                if _rc_risk == "aggressive":
                    _forced_signal, _forced_grade = force_aggressive_direction(_rc_text)
                    st.session_state["rc_forced_signal"] = _forced_signal
                    st.session_state["rc_forced_grade"] = _forced_grade
                else:
                    st.session_state.pop("rc_forced_signal", None)
                    st.session_state.pop("rc_forced_grade", None)
                st.session_state["rc_result"] = _rc_text
                st.session_state["rc_image"]  = _rc_img
                st.session_state["rc_ann"]    = None
                st.session_state["rc_modes"]  = (_rc_mode_label, _rc_risk_label)
                st.session_state["rc_live"]   = bool(_rc_auto)
            except anthropic.AuthenticationError:
                st.error("❌ Invalid API key.")
            except Exception as _rce:
                st.error(f"❌ Error: {_rce}")

    if st.session_state.get("rc_result"):
        _rt   = st.session_state["rc_result"]
        _meta = parse_json_from_analysis(_rt)
        _sig  = st.session_state.get("rc_forced_signal") or str(_meta.get("signal", "WAIT")).upper()
        _grade = st.session_state.get("rc_forced_grade") or str(_meta.get("setup_grade", "WAIT")).upper()
        _conf = int(_meta.get("confidence", 5) or 5)
        _pat  = _meta.get("pattern_name", "")
        _m_lb, _r_lb = st.session_state.get("rc_modes", ("⚡ Auto", "⚖️ Balanced"))

        st.divider()
        _sty = {
            "BUY":  ("", "rgba(34,197,94,0.15)",  "#4ade80", "#22c55e", "BUY"),
            "SELL": ("sell", "rgba(239,68,68,0.15)", "#f87171", "#ef4444", "SELL"),
        }.get(_sig, ("", "rgba(245,158,11,0.15)", "#fbbf24", "#f59e0b", "WAIT"))
        _bias = 50 + _conf * 5 if _sig == "BUY" else (50 - _conf * 5 if _sig == "SELL" else 50)
        _bias = max(4, min(96, _bias))
        _conf_lb = "High confidence" if _conf >= 7 else ("Medium confidence" if _conf >= 5 else "Low confidence")

        st.markdown(f"""
<div class='chee-signal-card {_sty[0]}'>
  <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px'>
    <span class='tag' style='background:rgba(232,199,110,0.10);border:1px solid rgba(232,199,110,0.4);color:#e8c76e'>⚡ CHEE AI READ · {_grade}</span>
    <span class='tag' style='background:{_sty[1]};border:1px solid {_sty[3]};color:{_sty[2]};font-size:14px;padding:7px 22px'>{_sty[4]}</span>
  </div>
  <span style='display:inline-block;background:#0c120e;border:1px solid #1c2a21;border-radius:999px;
  padding:5px 14px;color:#cfe0d4;font-size:12px;font-weight:700'>{_m_lb} · {_r_lb}{(" · 📐 " + _pat) if _pat and "No Clear" not in _pat else ""}</span>
  <div style='margin-top:18px'>
    <div style='display:flex;justify-content:space-between;align-items:center'>
      <span class='k'>Which way it leans</span>
      <span style='color:#7d8f83;font-size:12px;font-family:JetBrains Mono,monospace'>{_conf_lb} · {_conf}/10</span>
    </div>
    <div style='position:relative;height:8px;border-radius:999px;margin-top:10px;
    background:linear-gradient(90deg,#7f1d1d,#3f1d1d 35%,#123322 65%,#14532d)'>
      <div style='position:absolute;left:{_bias}%;top:-4px;transform:translateX(-50%);
      width:10px;height:16px;border-radius:4px;background:{_sty[2]};box-shadow:0 0 12px {_sty[2]}'></div>
    </div>
    <div style='display:flex;justify-content:space-between;margin-top:7px'>
      <span style='color:#7d8f83;font-size:11px;letter-spacing:2px'>BEARISH</span>
      <span style='color:#7d8f83;font-size:11px;letter-spacing:2px'>BULLISH</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        if _grade == "SPECULATIVE":
            st.warning("🔥 Aggressive forced signal: direction is provided as requested, but one or more A-grade gates are missing. Suggested risk: max 0.5%. 激进强制信号：已给出方向，但并非 A/A+ 设置，建议风险不超过 0.5%。")

        # ── Annotated chart ──
        if _meta.get("annotations") and st.session_state.get("rc_ann") is None:
            try:
                with st.spinner("🎨 Drawing levels on your chart…"):
                    st.session_state["rc_ann"] = annotate_chart(
                        st.session_state["rc_image"], _meta["annotations"], _sig, _meta)
            except Exception:
                st.session_state["rc_ann"] = None
        if st.session_state.get("rc_ann") is not None:
            st.image(pil_to_download_bytes(st.session_state["rc_ann"]),
                     caption="Key levels & zones drawn by Chee AI", use_container_width=True)
            st.download_button("⬇️ Download annotated chart",
                               data=pil_to_download_bytes(st.session_state["rc_ann"]),
                               file_name="chee_ai_chart.png", mime="image/png",
                               use_container_width=True, key="rc_dl")
        elif st.session_state.get("rc_live") and st.session_state.get("rc_image") is not None:
            st.image(pil_to_download_bytes(st.session_state["rc_image"]),
                     caption="Live chart · auto-fetched 自动抓取的实时图表", use_container_width=True)

        _rc_clean = re.sub(r"```json.*?```", "", _rt, flags=re.DOTALL).strip()
        st.markdown(_rc_clean)
        st.caption("AI analysis only — not financial advice · 仅供参考")


# ════════════════════════════════════════════════════════════
# KEY LEVELS — auto-detected S/R zones + AI explains WHY
# ════════════════════════════════════════════════════════════
if _nav == "Key Levels":
    st.markdown("## Key Levels")
    st.caption("Auto-detected support & resistance zones — and the reasoning behind every one · 自动检测关键支撑阻力区域，并解释为什么")

    _kl_c1, _kl_c2, _kl_c3 = st.columns([2.4, 1.3, 1.3])
    with _kl_c1:
        _kl_label = st.selectbox("Market 市场", list(_YF_TICKERS.keys()), key="kl_market")
    with _kl_c2:
        _kl_tf = st.selectbox("Timeframe 时间框架", ["H1", "H4", "D1"], index=2, key="kl_tf")
    with _kl_c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _kl_go = st.button("🧱 Find Key Levels", type="primary", use_container_width=True, key="kl_go")

    with st.expander("🧠 How zones are detected · 检测依据（完全透明）", expanded=False):
        st.markdown("""
Zones are drawn the way a professional draws them — not just "count the pivots". Every stat shown is real:

1. **Significant swings only 只取重要摆动点** — window-5 swing highs/lows, plus window-12 MAJOR swings (weighted heavier). Minor 3-bar wiggles are ignored, like your eye ignores them.
2. **ATR-adaptive clustering 按波动率聚类** — pivots within 0.6 × ATR(14) merge into one zone, so zone width matches the market's actual volatility (gold zones are wider than EURUSD zones, automatically).
3. **Scored like a trader judges 按交易员逻辑打分** — independent touches · **reaction strength** · reaction decay (repeated tests only weaken a zone when bounces are shrinking) · recency · flip behaviour · major swing · **round-number proximity** · **D1 confirmation**.
4. **Zone width kept realistic** — 0.25 to 1.2 ATR, never a hairline or a huge blob.

Top 6 zones by score are shown. If it still differs from your hand-drawn levels, tell the AI Analyst your level in chat — it will compare both against the data.
""")

    if _kl_go:
        if not api_key:
            st.warning("👈 Enter your AI API key in the sidebar first.")
        else:
            try:
                with st.spinner(f"📡 Fetching {_kl_label} candles & detecting zones…"):
                    _kl_df, _kl_srcname = fetch_candles_any(_kl_label, _kl_tf, twelve_data_key)
                    _kl_htf = None
                    if _kl_tf != "D1":
                        try:
                            _kl_htf, _ = fetch_candles_any(_kl_label, "D1", twelve_data_key)
                        except Exception:
                            _kl_htf = None
                    _kl_zones = find_sr_zones(_kl_df, lookback=350, max_zones=6, htf_df=_kl_htf)
                    _kl_price = float(_kl_df["Close"].iloc[-1])
                if not _kl_zones:
                    st.info("Not enough swing structure to detect zones — try a different timeframe.")
                else:
                    _kl_digest = build_market_digest(_kl_df, _kl_label, _kl_tf.upper())
                    _kl_zone_txt = sr_zones_text(_kl_zones)
                    _kl_prompt = f"""You are Chee AI — an elite analyst whose foundation is Support & Resistance. SNR is the root of trading.

Live market context for {_kl_label} ({_kl_tf}):
{_kl_digest}

Auto-detected S/R zones (from swing-pivot clustering, sorted high→low). These stats are REAL — use them:
{_kl_zone_txt}

Current price: {_kl_price:.6g}

TASK: For EACH zone, in the SAME order, explain WHY it is a valid support/resistance zone. Be specific and educational:
- how many independent times it was tested; call repeated tests weakening only when reaction_decay is flagged
- reaction strength (the avg ATR bounce shown — hard rejections prove real orders sit there)
- swing structure (rejected as swing-high? held as swing-low? contains a major swing?)
- flip behaviour if flagged (old support became resistance or vice versa — why that matters)
- D1 confirmation if flagged (a level visible on the higher timeframe is respected by bigger players)
- round-number psychology if flagged
- Fibonacci 38.2/50/61.8 overlap if any (bonus confluence only)
- recency (freshly tested vs stale)
Rate each zone strength 1-5. Then say which single zone matters MOST right now given current price, and what to watch for there.

Output STRICT JSON only. Never put double-quote characters inside text values (use single quotes if needed):
{{"summary_en": "2-3 sentences: which zone matters most now and why",
 "summary_cn": "中文总结 2-3 句",
 "zones": [
   {{"range": "3970–3985", "type": "resistance", "strength": 4,
     "why_en": "3-4 sentences explaining why this zone is S/R, citing the real stats",
     "why_cn": "中文解释 3-4 句，引用真实数据"}}
 ]}}"""
                    with st.spinner("🤖 AI is explaining every zone…"):
                        _kl_raw = ai_text_call(_kl_prompt, api_key, model_choice, json_mode=True)
                        _kl_parsed = parse_ai_json(_kl_raw, api_key, model_choice)
                        if not _kl_parsed:
                            st.error("❌ AI returned an unreadable response — please try again. AI 返回格式异常，请再试一次。")
                            with st.expander("🔍 Raw AI response (debug)"):
                                st.code((_kl_raw or "")[:4000])
                        else:
                            st.session_state["kl_result"] = _kl_parsed
                            st.session_state["kl_meta"] = {
                                "label": _kl_label, "tf": _kl_tf, "price": _kl_price,
                                "zones": _kl_zones,
                            }
            except Exception as _kle:
                st.error(f"❌ Key Levels failed: {_kle}")

    _klr = st.session_state.get("kl_result")
    _klm = st.session_state.get("kl_meta")
    if _klr and _klm:
        st.markdown(f"""<div class='info-box' style='display:flex;justify-content:space-between;align-items:center'>
<span style='font-family:Playfair Display,Georgia,serif;font-size:20px;color:#f3ead7'>{_klm['label']}
<span style='color:#7d8f83;font-size:13px;font-family:Inter,sans-serif'>&nbsp;· {_klm['tf'].upper()}</span></span>
<span style='font-family:JetBrains Mono,monospace;font-size:19px;font-weight:700;color:#e8c76e'>{_klm['price']:.6g}</span>
</div>""", unsafe_allow_html=True)

        if _klr.get("summary_en") or _klr.get("summary_cn"):
            st.markdown(f"""<div class='info-box'><p style='color:#cfe0d4;font-size:14px;margin:0'>
🧭 {_klr.get('summary_en', '')}<br><span style='color:#7d8f83'>{_klr.get('summary_cn', '')}</span></p></div>""",
                        unsafe_allow_html=True)

        _kl_ai_zones = _klr.get("zones", []) or []
        _price_marker_drawn = False
        for _zi, _z in enumerate(_klm["zones"]):
            _ai = _kl_ai_zones[_zi] if _zi < len(_kl_ai_zones) else {}
            # current-price marker between resistance block and support block
            if not _price_marker_drawn and _z["center"] < _klm["price"]:
                st.markdown(f"""<div style='display:flex;align-items:center;gap:12px;margin:14px 2px'>
<div style='flex:1;height:1px;background:linear-gradient(90deg,transparent,#e8c76e,transparent)'></div>
<span style='color:#e8c76e;font-family:JetBrains Mono,monospace;font-size:13px;font-weight:700'>▸ PRICE NOW {_klm['price']:.6g} ◂</span>
<div style='flex:1;height:1px;background:linear-gradient(90deg,transparent,#e8c76e,transparent)'></div>
</div>""", unsafe_allow_html=True)
                _price_marker_drawn = True

            _is_res = _z["kind"] == "resistance"
            _zc  = "#f87171" if _is_res else "#4ade80"
            _zbg = "rgba(239,68,68,0.07)" if _is_res else "rgba(34,197,94,0.07)"
            _zbd = "rgba(248,113,113,0.35)" if _is_res else "rgba(74,222,128,0.35)"
            _stars = "★" * int(_ai.get("strength", min(_z["touches"], 5))) or "★"
            _badge_css = ("font-size:10px;font-weight:800;letter-spacing:1px;padding:2px 8px;"
                          "border-radius:999px;margin-left:8px")
            _flip_badge = (f"<span style='background:rgba(232,199,110,0.12);border:1px solid rgba(232,199,110,0.45);"
                           f"color:#e8c76e;{_badge_css}'>FLIP ZONE 翻转区</span>") if _z["flip"] else ""
            _htf_badge = (f"<span style='background:rgba(34,211,238,0.10);border:1px solid rgba(34,211,238,0.4);"
                          f"color:#67e8f9;{_badge_css}'>D1 ✓ 大级别确认</span>") if _z.get("htf_confirmed") else ""
            _rnd_badge = (f"<span style='background:rgba(125,143,131,0.12);border:1px solid rgba(125,143,131,0.4);"
                          f"color:#a8bcae;{_badge_css}'>ROUND {_z['round_level']:.6g}</span>") if _z.get("round_level") else ""
            st.markdown(f"""
<div style='background:{_zbg};border:1px solid {_zbd};border-radius:16px;padding:16px 18px;margin:8px 0'>
  <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px'>
    <span>
      <span style='color:{_zc};font-size:11px;font-weight:800;letter-spacing:2px;text-transform:uppercase'>{_z['kind']}</span>
      {_flip_badge}{_htf_badge}{_rnd_badge}
    </span>
    <span style='color:#e8c76e;font-size:13px;letter-spacing:2px'>{_stars}</span>
  </div>
  <div style='font-family:JetBrains Mono,monospace;font-size:20px;font-weight:700;color:#eef5f0;margin:6px 0 2px 0'>
    {_z['low']:.6g} – {_z['high']:.6g}
  </div>
  <div style='color:#7d8f83;font-size:12px;margin-bottom:10px'>
    tested {_z['touches']}× · reaction ≈ {_z.get('avg_reaction_atr', '—')} ATR · {_z['swing_highs']} swing-highs / {_z['swing_lows']} swing-lows · last touch {_z['bars_since_last_touch']} bars ago
  </div>
  <p style='color:#cfe0d4;font-size:13.5px;margin:0 0 4px 0'>{_ai.get('why_en', '')}</p>
  <p style='color:#7d8f83;font-size:13px;margin:0'>{_ai.get('why_cn', '')}</p>
</div>""", unsafe_allow_html=True)

        if not _price_marker_drawn:
            st.markdown(f"<p style='color:#e8c76e;text-align:center;font-family:JetBrains Mono,monospace;font-size:13px'>▸ PRICE NOW {_klm['price']:.6g} ◂ (below all zones)</p>", unsafe_allow_html=True)
        st.caption("⚠️ AI analysis, not financial advice · 仅供参考")
    elif not _kl_go:
        st.markdown("""
<div class='info-box' style='text-align:center;padding:44px 20px'>
<p style='color:#cfe0d4;font-size:15px;font-weight:600;margin:0'>Pick a market, hit Find — I'll detect every key S/R zone from real swing data,<br>then explain exactly why each level is support or resistance.<br><br>
<span style='color:#7d8f83;font-size:13px;font-weight:400'>选择市场后点击查找 — 程序先从真实K线的摆动点检测出关键区域（测试次数、翻转行为都是真实统计），再由 AI 逐个解释为什么这里是支撑/阻力。聊天里直接问某个品种的关键位也可以。</span></p>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# CHART ANALYSIS — merged into AI Analyst (page removed)
# ════════════════════════════════════════════════════════════
mtf_mode = False

# ════════════════════════════════════════════════════════════
# MULTI-TIMEFRAME MODE
# ════════════════════════════════════════════════════════════


# ============================================================
# TOOL PAGES (routed via sidebar navigation)
# ============================================================

# ════════════════════════════════════════════════════════════
# TOOL 1 — POSITION SIZE CALCULATOR
# ════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════
# TOOL 2 — ECONOMIC CALENDAR
# ════════════════════════════════════════════════════════════
if _nav == "Markets":
    st.markdown("## Markets")
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

# ════════════════════════════════════════════════════════════
# TOOL 4 — AI TRADING COACH
# ════════════════════════════════════════════════════════════
if _nav == "AI Analyst":
    st.markdown("## AI Analyst")

    # ── Initialise conversation store ──────────────────────────
    if "coach_convs" not in st.session_state:
        st.session_state["coach_convs"] = []
    if "coach_active_id" not in st.session_state:
        st.session_state["coach_active_id"] = None

    COACH_SYSTEM = """You are Chee AI — an elite trading analyst and mentor with 20+ years of experience in Forex, Gold, Crypto, and Indices. Your foundation is SUPPORT & RESISTANCE — SNR is the root of all trading. Fibonacci (38.2% / 50% / 61.8%) is a supporting tool you add for bonus confluence when a clean swing exists, never a requirement.

Your role:
• Answer ALL trading questions clearly, whether beginner or advanced
• When a chart image is shared, analyse it thoroughly: trend, the key S/R levels the market respects, rejection candles at those levels, entry/SL/TP — and note if a fib 38.2/50/61.8 level lines up with S/R
• Give honest, direct feedback on the trader's setups — praise what is right, correct what is wrong
• Speak with authority but stay encouraging; trading is a journey
• Use examples and analogies to explain complex concepts
• Default language: answer in the same language the trader uses (English or Chinese)

You follow these trading principles:
- Top-Down analysis: D1 → H4 → H1 → M15
- A setup  = trend + pullback to a TESTED S/R level + rejection candle
- A+ setup = the same, plus a fib 38.2/50/61.8 level lining up with that S/R level (bonus confluence)
- No valid S/R level = no trade. Missing Fibonacci never invalidates a setup
- Risk management: never risk more than 1-2% per trade, minimum 1:2 R:R, always define SL before entry
- Patience: no level = no trade

IMPORTANT — LIVE DATA: when a [LIVE MARKET DATA] block appears in a message, it contains REAL prices fetched from the market seconds ago. Treat it as ground truth. Reference the exact numbers (current price, swing high/low, fib levels) in your answer and give concrete levels for entry/SL/TP. A live chart image may also be attached — analyse it.
The block may include "Auto-detected S/R zones" with real touch counts and flip flags — when the trader asks about support/resistance, use those zones and EXPLAIN WHY each one is valid (times tested, swing structure, flip behaviour, round numbers, fib overlap, recency)."""

    def _coach_title(messages):
        """Auto-generate a conversation title from the first user message."""
        for m in messages:
            if m.get("role") == "user":
                txt = m.get("content", "")
                if isinstance(txt, str) and txt.strip():
                    t = txt.strip().replace("\n", " ")
                    return (t[:36] + "…") if len(t) > 36 else t
        return "New Chat"

    def _new_coach_conv():
        """Create a new blank conversation, set it active."""
        import uuid, datetime
        cid = str(uuid.uuid4())[:8]
        st.session_state["coach_convs"].append({
            "id":         cid,
            "title":      "New Chat",
            "messages":   [],
            "created_at": datetime.datetime.now().strftime("%b %d, %H:%M"),
            "img_b64":    None,
            "img_bytes":  None,
        })
        st.session_state["coach_active_id"]  = cid
        st.session_state["coach_img_counter"] = st.session_state.get("coach_img_counter", 0) + 1

    def _active_conv():
        aid = st.session_state.get("coach_active_id")
        for c in st.session_state["coach_convs"]:
            if c["id"] == aid:
                return c
        return None

    # ── Question forwarded from the Home quick-ask bar ─────────
    if st.session_state.get("pending_question"):
        _new_coach_conv()

    # ── CSS for the coach panel ────────────────────────────────
    st.markdown("""
<style>
/* Left chat-list panel */
.coach-list-panel {
    background: #0d1117;
    border-right: 1px solid #21262d;
    border-radius: 12px;
    padding: 0;
    min-height: 520px;
}
</style>
""", unsafe_allow_html=True)

    # ── Two-column layout ──────────────────────────────────────
    c_left, c_right = st.columns([1, 3], gap="small")

    # ══════════════════════════════════════════════════════
    # LEFT PANEL — conversation list
    # ══════════════════════════════════════════════════════
    with c_left:
        st.markdown("""
<div style='font-size:11.5px;color:#7d8f83;letter-spacing:2.5px;text-transform:uppercase;
margin-bottom:10px;padding:0 4px;font-weight:700'>Conversations</div>
""", unsafe_allow_html=True)

        # New Chat button
        if st.button("✏️ New Chat", use_container_width=True, key="coach_new_btn", type="primary"):
            _new_coach_conv()
            st.rerun()

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        convs = st.session_state["coach_convs"]
        if not convs:
            st.markdown("<div style='color:#6e7681;font-size:12px;padding:8px 4px'>No chats yet.<br>Click New Chat to start.</div>",
                        unsafe_allow_html=True)
        else:
            active_id = st.session_state.get("coach_active_id")
            # Show newest first
            for conv in reversed(convs):
                is_active = conv["id"] == active_id
                label = conv.get("title", "New Chat")
                date  = conv.get("created_at", "")
                has_img = bool(conv.get("img_b64"))
                img_ico = "📷 " if has_img else "💬 "

                if is_active:
                    st.markdown(f"""
<div style='background:rgba(34,197,94,0.10);border:1px solid rgba(74,222,128,0.4);border-radius:10px;
padding:8px 10px;margin:3px 0;cursor:pointer'>
  <div style='font-size:13px;font-weight:600;color:#eef5f0;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis'>{img_ico}{label}</div>
  <div style='font-size:10px;color:#7d8f83;margin-top:2px'>{date}</div>
</div>""", unsafe_allow_html=True)
                else:
                    if st.button(f"{img_ico}{label}", key=f"conv_sel_{conv['id']}",
                                 use_container_width=True, help=date):
                        st.session_state["coach_active_id"] = conv["id"]
                        st.rerun()

    # ══════════════════════════════════════════════════════
    # RIGHT PANEL — active conversation
    # ══════════════════════════════════════════════════════
    with c_right:
        conv = _active_conv()

        if conv is None:
            # ── Welcome screen ─────────────────────────────
            st.markdown("""
<div style='text-align:center;padding:60px 20px'>
  <div style='width:64px;height:64px;margin:0 auto 18px auto;border-radius:18px;
  background:linear-gradient(135deg,#b9973f,#e8c76e);display:flex;align-items:center;
  justify-content:center;font-size:30px;box-shadow:0 0 34px rgba(232,199,110,0.4)'>⚡</div>
  <h2 style='color:#f3ead7;margin:0 0 8px 0;font-family:Playfair Display,Georgia,serif'>Chee <span style='font-style:italic;color:#e8c76e'>AI</span> Analyst</h2>
  <p style='color:#7d8f83;font-size:15px;margin:0 0 24px 0'>
    Ask about any market — I fetch live prices and charts automatically, then analyse them for you.<br>
    试试问「can I sell gold now?」— 我会自动抓取实时数据并分析。
  </p>
  <div style='display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-bottom:32px'>
    <div style='background:#0b100d;border:1px solid #1c2a21;border-radius:14px;padding:14px 16px;
    text-align:left;max-width:200px'>
      <div style='font-size:20px;margin-bottom:6px'>📡</div>
      <div style='color:#eef5f0;font-size:13px;font-weight:700'>Live Data Auto-Fetch</div>
      <div style='color:#7d8f83;font-size:12px'>Mention gold, EURUSD, BTC… I pull real prices & a live chart myself</div>
    </div>
    <div style='background:#0b100d;border:1px solid #1c2a21;border-radius:14px;padding:14px 16px;
    text-align:left;max-width:200px'>
      <div style='font-size:20px;margin-bottom:6px'>📷</div>
      <div style='color:#eef5f0;font-size:13px;font-weight:700'>Chart Review</div>
      <div style='color:#7d8f83;font-size:12px'>Upload any chart — S&R + price action read with entry, SL, TP</div>
    </div>
    <div style='background:#0b100d;border:1px solid #1c2a21;border-radius:14px;padding:14px 16px;
    text-align:left;max-width:200px'>
      <div style='font-size:20px;margin-bottom:6px'>💬</div>
      <div style='color:#eef5f0;font-size:13px;font-weight:700'>Ask Anything</div>
      <div style='color:#7d8f83;font-size:12px'>Strategy, psychology, risk, concepts — 24/7</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
            if st.button("✏️ Start a New Chat", use_container_width=False, type="primary", key="coach_start_btn"):
                _new_coach_conv()
                st.rerun()

        else:
            # ── Active conversation ────────────────────────

            # Header bar
            h_col1, h_col2 = st.columns([5, 1])
            with h_col1:
                st.markdown(f"<div style='font-size:15px;font-weight:700;color:#f1f5f9;padding:4px 0'>"
                            f"💬 {conv.get('title','New Chat')}</div>", unsafe_allow_html=True)
            with h_col2:
                if st.button("🗑️", key=f"del_conv_{conv['id']}", help="Delete this chat"):
                    st.session_state["coach_convs"] = [
                        c for c in st.session_state["coach_convs"] if c["id"] != conv["id"]
                    ]
                    st.session_state["coach_active_id"] = None
                    st.rerun()

            st.markdown("<hr style='border-color:#1e293b;margin:6px 0 12px 0'>", unsafe_allow_html=True)

            messages = conv.get("messages", [])

            # ── If no messages yet — show first-message setup ──
            if not messages:
                st.markdown("<div style='color:#94a3b8;font-size:14px;margin-bottom:12px'>"
                            "💡 Optionally attach a chart image before sending your first message.</div>",
                            unsafe_allow_html=True)
                img_counter = st.session_state.get("coach_img_counter", 0)
                first_img = st.file_uploader(
                    "📷 Attach chart (optional)",
                    type=["png", "jpg", "jpeg", "webp"],
                    key=f"coach_first_img_{img_counter}",
                )
                if first_img:
                    try:
                        _fp = Image.open(first_img)
                        _fb = io.BytesIO()
                        _fr = _fp.copy()
                        if _fr.mode in ("RGBA", "P"):
                            _fr = _fr.convert("RGB")
                        _fr.save(_fb, format="JPEG", quality=92)
                        conv["img_bytes"] = _fb.getvalue()
                        conv["img_b64"]   = base64.b64encode(conv["img_bytes"]).decode()
                        st.image(conv["img_bytes"], caption="✅ Chart attached", width=260)
                    except Exception:
                        pass

            # ── Render existing messages ───────────────────
            for mi, msg in enumerate(messages):
                role = msg.get("role", "user")
                with st.chat_message(role):
                    # Show chart thumbnail on first user message
                    if mi == 0 and role == "user" and conv.get("img_bytes"):
                        with st.expander("📷 Chart attached", expanded=False):
                            st.image(conv["img_bytes"], use_container_width=True)
                    # Show extra attached image if any
                    if role == "user" and msg.get("extra_img_b64"):
                        try:
                            st.image(base64.b64decode(msg["extra_img_b64"]),
                                     width=220,
                                     caption="📡 Live chart (auto-fetched)" if msg.get("live_data") else "📎 Chart")
                        except Exception:
                            pass
                    st.markdown(msg.get("content", ""))
                    if msg.get("live_note"):
                        st.caption(msg["live_note"])

            # ── Extra image attachment for follow-up ──────
            if messages:
                _ec = st.session_state.get("coach_img_counter", 0)
                extra_img_file = st.file_uploader(
                    "📎 Attach a new chart (optional)",
                    type=["png", "jpg", "jpeg", "webp"],
                    key=f"coach_extra_{conv['id']}_{_ec}",
                )
            else:
                extra_img_file = None

            # ── Chat input ─────────────────────────────────
            user_input = st.chat_input(
                "Message Chee AI…  ·  问我任何交易问题…",
                key=f"coach_input_{conv['id']}",
            )

            # Consume question forwarded from Home quick-ask
            if not user_input and st.session_state.get("pending_question"):
                user_input = st.session_state.pop("pending_question")

            if user_input:
                if not api_key:
                    st.warning("👈 Enter your API key in the sidebar first.")
                else:
                    # Encode extra image if any
                    extra_b64 = None
                    extra_bytes = None
                    if extra_img_file:
                        try:
                            _ep = Image.open(extra_img_file)
                            _eb = io.BytesIO()
                            _er = _ep.copy()
                            if _er.mode in ("RGBA", "P"):
                                _er = _er.convert("RGB")
                            _er.save(_eb, format="JPEG", quality=92)
                            extra_bytes = _eb.getvalue()
                            extra_b64   = base64.b64encode(extra_bytes).decode()
                            st.session_state["coach_img_counter"] = _ec + 1
                        except Exception:
                            pass

                    # ── AUTO-FETCH live market data (ChatGPT-style) ──
                    _live_digest = None
                    _live_note   = None
                    _detected    = detect_symbols_in_text(user_input)
                    if _detected:
                        _td_sym, _td_lbl = _detected[0]
                        try:
                            with st.spinner(f"📡 Fetching live {_td_lbl} data…"):
                                _df_h1, _src_h1 = fetch_candles_any(_td_lbl, "H1", twelve_data_key)
                                _df_d1, _src_d1 = fetch_candles_any(_td_lbl, "D1", twelve_data_key)
                            _live_digest = (
                                "[LIVE MARKET DATA — fetched seconds ago, treat as ground truth]\n"
                                + build_market_digest(_df_d1, _td_lbl, "D1") + "\n"
                                + build_market_digest(_df_h1, _td_lbl, "H1")
                            )
                            _sources = " / ".join(dict.fromkeys((_src_d1, _src_h1)))
                            _live_note = f"📡 Live data fetched: {_td_lbl} · D1 + H1 · {_sources} · 已自动获取实时数据"
                            # Attach an auto-generated live chart (if user didn't attach one)
                            if not extra_b64:
                                try:
                                    _chart_pil = generate_chart_image_from_df(
                                        _df_h1.tail(140), _td_lbl, "H1")
                                    _cb = io.BytesIO()
                                    _chart_pil.convert("RGB").save(_cb, format="JPEG", quality=90)
                                    extra_bytes = _cb.getvalue()
                                    extra_b64   = base64.b64encode(extra_bytes).decode()
                                except Exception:
                                    pass
                        except Exception as _fe:
                            _live_note = f"⚠️ Could not fetch live data ({_fe}) — answering from the chart/context only."

                    # Append user message
                    user_entry = {"role": "user", "content": user_input}
                    if extra_b64:
                        user_entry["extra_img_b64"] = extra_b64
                    if _live_digest:
                        user_entry["live_data"] = _live_digest
                    if _live_note:
                        user_entry["live_note"] = _live_note
                    conv["messages"].append(user_entry)

                    # Auto-set title from first user message
                    if len(conv["messages"]) == 1:
                        conv["title"] = _coach_title(conv["messages"])

                    with st.chat_message("user"):
                        if extra_b64:
                            try:
                                st.image(base64.b64decode(extra_b64), width=220,
                                         caption="📡 Live chart (auto-fetched)" if _live_digest else "📎 Chart")
                            except Exception:
                                pass
                        st.markdown(user_input)
                        if _live_note:
                            st.caption(_live_note)

                    with st.chat_message("assistant"):
                        with st.spinner("Thinking…"):
                            try:
                                if model_choice.startswith("gemini"):
                                    client_c = google_genai.Client(api_key=api_key)
                                    # Build history text
                                    hist_txt = "\n\n".join([
                                        f"{'Student' if m['role']=='user' else 'Coach'}:\n{m['content']}"
                                        + (("\n\n" + m["live_data"]) if m.get("live_data") else "")
                                        for m in conv["messages"]
                                    ])
                                    full_prompt = COACH_SYSTEM + "\n\n---\nConversation:\n" + hist_txt
                                    gemini_parts = [full_prompt]
                                    # Original chart
                                    if conv.get("img_bytes"):
                                        gemini_parts.append(
                                            google_types.Part.from_bytes(
                                                data=conv["img_bytes"], mime_type="image/jpeg"))
                                    # Extra chart
                                    if extra_bytes:
                                        gemini_parts.append(
                                            google_types.Part.from_bytes(
                                                data=extra_bytes, mime_type="image/jpeg"))
                                        gemini_parts[0] += "\n\n[Student attached a NEW chart image above.]"
                                    resp_c = client_c.models.generate_content(
                                        model=model_choice, contents=gemini_parts)
                                    answer = resp_c.text

                                else:
                                    client_c = anthropic.Anthropic(api_key=api_key)
                                    api_msgs = []
                                    for i, m in enumerate(conv["messages"]):
                                        content_parts = []
                                        # First message: include original chart
                                        if i == 0 and conv.get("img_b64"):
                                            content_parts.append({
                                                "type": "image",
                                                "source": {"type": "base64",
                                                           "media_type": "image/jpeg",
                                                           "data": conv["img_b64"]},
                                            })
                                        # Any message with extra image
                                        if m.get("extra_img_b64"):
                                            content_parts.append({
                                                "type": "image",
                                                "source": {"type": "base64",
                                                           "media_type": "image/jpeg",
                                                           "data": m["extra_img_b64"]},
                                            })
                                        _txt = m["content"] + (("\n\n" + m["live_data"]) if m.get("live_data") else "")
                                        content_parts.append({"type": "text", "text": _txt})
                                        role_msg = m["role"]
                                        if len(content_parts) == 1:
                                            api_msgs.append({"role": role_msg,
                                                             "content": content_parts[0]["text"]})
                                        else:
                                            api_msgs.append({"role": role_msg,
                                                             "content": content_parts})

                                    resp_c = client_c.messages.create(
                                        model=model_choice,
                                        max_tokens=2000,
                                        system=COACH_SYSTEM,
                                        messages=api_msgs[-20:],
                                    )
                                    answer = claude_text(resp_c)

                                conv["messages"].append({"role": "assistant", "content": answer})
                                st.markdown(answer)

                            except Exception as e:
                                err_msg = f"❌ Error: {str(e)}"
                                conv["messages"].append({"role": "assistant", "content": err_msg})
                                st.error(err_msg)

                    st.rerun()


# ════════════════════════════════════════════════════════════
# TOOL 5 — PDF REPORT GENERATOR
# ════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════
# TOOL 6 — CURRENCY STRENGTH METER
# ════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════
# TOOL 7 — LIVE DATA ANALYSIS
# ════════════════════════════════════════════════════════════
if _nav == "Markets":
    st.divider()
    st.markdown("### 📈 Live Data Analysis 实时数据分析")
    st.caption("Fetch live candles directly — no chart upload needed. 直接拉取实时K线，无需上传图表。")

    # ── Lazy imports ──────────────────────────────────────────
    try:
        import plotly.graph_objects as go
        _LIVE_AVAILABLE = True
    except ImportError:
        _LIVE_AVAILABLE = False
        st.error("📦 Live Data requires `plotly`. Please redeploy after updating requirements.txt.")

    if _LIVE_AVAILABLE:
        # ── Data source indicator ─────────────────────────────
        if twelve_data_key:
            st.success("🟢 **Real-time data** via Twelve Data · No delay")
        else:
            st.warning("⚡ **Twelve Data API key required.** Add your key in the sidebar under '📡 Live Data Key' to use this feature.")

        # ── Ticker presets ────────────────────────────────────
        # Each entry: display_name → (twelve_data_symbol, _unused)
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
                    "Twelve Data symbol",
                    placeholder="e.g. EUR/USD  or  AAPL",
                    key="ld_custom_ticker",
                ).strip().upper()
                td_sym  = custom_input
                yf_sym  = custom_input
                ticker_sym = custom_input
            else:
                ticker_sym = td_sym
                st.caption(f"Twelve Data symbol: `{ticker_sym}`")

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
                            raise ValueError("⚡ Twelve Data API key required. Add it in the sidebar under '📡 Live Data Key'.")

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
                            ld_context = ld_extra_context or f"Live {tf_label} data — {n} candles fetched automatically via Twelve Data."

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

# ════════════════════════════════════════════════════════════
# TOOL 9 — AI DEBATE (BOARD OF DIRECTORS)
# ════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════
# TOOL 10 — SIGNAL FEED (TradingView → Google Sheets)
# ════════════════════════════════════════════════════════════

# ── Footer ─────────────────────────────────────────────────
st.divider()
st.caption(
    "⚠️ **Disclaimer**: Chee AI is for **educational and informational purposes only**. "
    "It does NOT constitute financial advice. Trading involves substantial risk of loss. "
    "Always conduct your own research and manage your risk responsibly."
)
