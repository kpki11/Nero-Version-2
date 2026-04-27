# =============================================================================
#  NERO v2 – nero_ui.py  ·  FULL UI OVERHAUL
#  Neural Equity Ranking & Optimization System
#  Author: kppan  |  NSE India
#
#  Usage:  python nero_ui.py
#  Needs:  nero_v2.py in same directory (or on PYTHONPATH)
# =============================================================================

import os, sys, queue, threading, traceback, datetime, csv, io
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont

# ── Import NERO engine ────────────────────────────────────────────────────────
try:
    import nero_v2 as engine
    ENGINE_OK = True
except ImportError as _e:
    ENGINE_OK = False
    _ENGINE_IMPORT_ERR = str(_e)


# =============================================================================
#  DESIGN TOKENS
# =============================================================================

# Dark, premium trading-terminal palette
C = {
    # Backgrounds
    "bg":        "#0d0f14",   # deep void
    "surface":   "#13161e",   # cards / panels
    "surface2":  "#1a1e29",   # elevated surfaces
    "surface3":  "#21263a",   # inputs / entries
    "border":    "#2a3050",   # subtle borders
    "border2":   "#3a4060",   # hover borders

    # Accent system
    "blue":      "#3b82f6",   # primary CTA
    "blue_d":    "#2563eb",   # CTA hover
    "blue_glow": "#1d4ed8",
    "teal":      "#14b8a6",   # success / green signal
    "amber":     "#f59e0b",   # warning
    "red":       "#ef4444",   # danger
    "purple":    "#8b5cf6",   # secondary accent

    # Text
    "text":      "#e2e8f0",   # primary
    "text2":     "#94a3b8",   # secondary
    "text3":     "#64748b",   # muted
    "white":     "#ffffff",

    # Row bucket tints (dark theme)
    "asym_bg":   "#1e1a0a",
    "def_bg":    "#0a1a12",
    "bal_bg":    "#0a1020",

    # Chart / regime
    "bull":      "#22c55e",
    "bear":      "#ef4444",
    "neutral":   "#94a3b8",
}

FONT_MONO  = ("Consolas",    9)
FONT_BODY  = ("Segoe UI",    9)
FONT_BOLD  = ("Segoe UI",    9, "bold")
FONT_SM    = ("Segoe UI",    8)
FONT_LG    = ("Segoe UI",   12, "bold")
FONT_TITLE = ("Segoe UI",   14, "bold")
FONT_XL    = ("Segoe UI",   20, "bold")

APP_TITLE = "NERO v2  ·  Neural Equity Ranking & Optimization"
WIN_W, WIN_H = 1440, 860
LEFT_W       = 300
CENTRE_W     = 310
# right panel gets remaining space

TV_COLS = [
    ("rank",     "Rank",       40,  "center"),
    ("symbol",   "Symbol",    100,  "w"),
    ("score",    "Score",      58,  "center"),
    ("fund",     "Fund",       52,  "center"),
    ("tech",     "Tech",       52,  "center"),
    ("vol",      "Vol",        48,  "center"),
    ("bucket",   "Bucket",     90,  "w"),
    ("weight",   "Wt%",        52,  "center"),
    ("capital",  "Capital Rs", 110,  "e"),
    ("strategy", "Strategy",  130,  "w"),
    ("ev",       "EV",         60,  "center"),
    ("cvar",     "CVaR",       60,  "center"),
]

REGIME_COLORS = {
    "Bull Trend":        C["bull"],
    "Bear Trend":        C["bear"],
    "High Volatility":   C["amber"],
    "Recovery":          C["teal"],
    "Sideways":          C["neutral"],
}


# =============================================================================
#  HELPERS
# =============================================================================

def _env(var, default=""): return os.environ.get(var, default)
def _set_env(var, val):    os.environ[var] = str(val)

def _fmt_capital(v):
    try:    return f"{float(v):>12,.0f}"
    except: return str(v)

def _fmt_pct(v):
    try:    return f"{float(v)*100:.2f}%"
    except: return str(v)

def _fmt_f(v, d=2):
    try:    return f"{float(v):.{d}f}"
    except: return str(v)

def _regime_color(label):
    for k, c in REGIME_COLORS.items():
        if k.lower() in str(label).lower():
            return c
    return C["neutral"]


# =============================================================================
#  CUSTOM WIDGETS
# =============================================================================

class FlatButton(tk.Button):
    """Flat styled button with hover feedback."""
    def __init__(self, parent, text="", command=None,
                 bg=None, fg=C["white"], hover=None,
                 font=FONT_BOLD, width=None, height=1, **kw):
        bg    = bg    or C["blue"]
        hover = hover or C["blue_d"]
        super().__init__(parent, text=text, command=command,
                         bg=bg, fg=fg, activebackground=hover,
                         activeforeground=fg, font=font,
                         relief="flat", bd=0, cursor="hand2",
                         height=height, padx=14, pady=6, **kw)
        if width:
            self.config(width=width)
        self._bg, self._hover = bg, hover
        self.bind("<Enter>", lambda e: self.config(bg=self._hover))
        self.bind("<Leave>", lambda e: self.config(bg=self._bg))

    def set_colors(self, bg, hover):
        self._bg, self._hover = bg, hover
        self.config(bg=bg, activebackground=hover)


class SectionLabel(ttk.Label):
    def __init__(self, parent, text, **kw):
        super().__init__(parent, text=text.upper(),
                         foreground=C["text3"], background=C["surface"],
                         font=("Segoe UI", 7, "bold"), **kw)


class Divider(tk.Frame):
    def __init__(self, parent, padx=0, pady=6):
        super().__init__(parent, height=1, bg=C["border"])
        self.pack(fill="x", padx=padx, pady=pady)


class ScoreBadge(tk.Label):
    """Coloured score pill."""
    def __init__(self, parent, value=0, **kw):
        self._update_color(value)
        super().__init__(parent, text=f"{value:.1f}",
                         bg=self._bg, fg=self._fg,
                         font=("Consolas", 8, "bold"),
                         padx=4, pady=1, relief="flat", **kw)

    def _update_color(self, v):
        v = float(v) if v else 0
        if   v >= 70: self._bg, self._fg = C["teal"],   C["bg"]
        elif v >= 50: self._bg, self._fg = C["blue"],   C["white"]
        elif v >= 30: self._bg, self._fg = C["amber"],  C["bg"]
        else:          self._bg, self._fg = C["red"],    C["white"]

    def update_value(self, v):
        self._update_color(v)
        self.config(text=f"{float(v):.1f}", bg=self._bg, fg=self._fg)


class AnimatedProgressBar(tk.Canvas):
    """Animated indeterminate progress bar with glow."""
    def __init__(self, parent, height=4, **kw):
        super().__init__(parent, height=height, bg=C["surface3"],
                         highlightthickness=0, **kw)
        self._pos   = 0
        self._dir   = 1
        self._width = 0
        self._anim  = None
        self._active = False

    def start(self):
        self._active = True
        self._width  = self.winfo_width() or 300
        self._animate()

    def stop(self):
        self._active = False
        if self._anim:
            self.after_cancel(self._anim)
            self._anim = None
        self.delete("all")

    def _animate(self):
        if not self._active:
            return
        self._width = self.winfo_width() or 300
        bar_w = self._width // 3
        x0    = self._pos
        x1    = min(x0 + bar_w, self._width)
        self.delete("all")
        # Glow layers
        for alpha, extra in [(C["blue_glow"], 8), (C["blue"], 4), (C["teal"], 0)]:
            self.create_rectangle(max(0, x0-extra), 0, x1+extra,
                                  self.winfo_height() or 4,
                                  fill=alpha, outline="")
        self._pos += self._dir * 8
        if self._pos + bar_w >= self._width or self._pos <= 0:
            self._dir *= -1
        self._anim = self.after(30, self._animate)


class LogPane(tk.Frame):
    """Scrollable terminal-style log output."""
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=C["bg"], **kw)
        self._text = tk.Text(
            self, wrap="word",
            bg=C["bg"], fg=C["text2"],
            font=FONT_MONO, relief="flat", bd=0,
            state="disabled", selectbackground=C["surface3"],
        )
        sb = tk.Scrollbar(self, orient="vertical",
                          command=self._text.yview, bg=C["surface2"],
                          troughcolor=C["bg"], width=8)
        self._text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._text.pack(side="left", fill="both", expand=True)

        # Tag colours
        self._text.tag_configure("info",   foreground=C["text2"])
        self._text.tag_configure("ok",     foreground=C["teal"])
        self._text.tag_configure("warn",   foreground=C["amber"])
        self._text.tag_configure("err",    foreground=C["red"])
        self._text.tag_configure("head",   foreground=C["blue"],
                                 font=("Consolas", 9, "bold"))
        self._text.tag_configure("dim",    foreground=C["text3"])

    def append(self, line, tag="info"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._text.configure(state="normal")
        self._text.insert("end", f"[{ts}] ", "dim")
        self._text.insert("end", line + "\n", tag)
        self._text.configure(state="disabled")
        self._text.see("end")

    def clear(self):
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")


class RegimeGauge(tk.Canvas):
    """Compact circular gauge for a single metric."""
    def __init__(self, parent, label="", size=72, **kw):
        super().__init__(parent, width=size, height=size,
                         bg=C["surface"], highlightthickness=0, **kw)
        self._size  = size
        self._label = label
        self._value = 0.0
        self._draw()

    def set_value(self, v):
        self._value = max(-1.0, min(1.0, float(v) if v not in (None,"--","") else 0.0))
        self._draw()

    def _draw(self):
        s = self._size
        self.delete("all")
        pad = 6
        # Track
        self.create_oval(pad, pad, s-pad, s-pad,
                         outline=C["border2"], width=4)
        # Arc: map -1..1 to 0..270 degrees
        extent = (self._value + 1) / 2 * 270
        color  = C["teal"] if self._value >= 0 else C["red"]
        if extent > 0:
            self.create_arc(pad, pad, s-pad, s-pad,
                            start=135+270-extent, extent=extent,
                            outline=color, width=4, style="arc")
        # Value text
        pct = f"{self._value:+.2f}"
        self.create_text(s//2, s//2 - 4, text=pct,
                         fill=color, font=("Consolas", 8, "bold"))
        # Label
        self.create_text(s//2, s//2 + 10, text=self._label,
                         fill=C["text3"], font=("Segoe UI", 7))


# =============================================================================
#  TOOLTIP
# =============================================================================

class Tooltip:
    def __init__(self, widget, text):
        self._tip = None
        widget.bind("<Enter>", lambda e: self._show(e, text))
        widget.bind("<Leave>", self._hide)

    def _show(self, event, text):
        x = event.widget.winfo_rootx() + 20
        y = event.widget.winfo_rooty() + 20
        self._tip = tk.Toplevel()
        self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip, text=text,
                 bg=C["surface3"], fg=C["text"],
                 font=FONT_SM, padx=8, pady=4,
                 relief="flat", bd=1).pack()

    def _hide(self, _=None):
        if self._tip:
            self._tip.destroy()
            self._tip = None


# =============================================================================
#  MAIN APPLICATION
# =============================================================================

class NeroApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.minsize(1200, 720)
        self.configure(bg=C["bg"])
        self.resizable(True, True)

        # State
        self._q:          queue.Queue          = queue.Queue()
        self._thread:     threading.Thread     = None
        self._sort_col:   str                  = "rank"
        self._sort_rev:   bool                 = False
        self._portfolio                        = None
        self._results_df                       = None
        self._regime                           = "--"
        self._vector:     dict                 = {}

        # Redirect print() to log pane
        self._log_queue: queue.Queue = queue.Queue()
        sys.stdout = _QueueWriter(self._log_queue, sys.__stdout__)

        # Build UI
        self._apply_style()
        self._build_titlebar()
        self._build_body()
        self._build_statusbar()

        # Startup
        self.after(300, self._startup_checks)
        self._poll()

    # -------------------------------------------------------------------------
    #  TTK Style
    # -------------------------------------------------------------------------

    def _apply_style(self):
        s = ttk.Style(self)
        s.theme_use("clam")

        s.configure("TFrame",      background=C["surface"])
        s.configure("BG.TFrame",   background=C["bg"])
        s.configure("TLabel",      background=C["surface"], foreground=C["text"],   font=FONT_BODY)
        s.configure("Dim.TLabel",  background=C["surface"], foreground=C["text3"],  font=FONT_SM)
        s.configure("Mono.TLabel", background=C["surface"], foreground=C["teal"],   font=FONT_MONO)
        s.configure("Head.TLabel", background=C["surface"], foreground=C["text"],   font=FONT_BOLD)
        s.configure("BG.TLabel",   background=C["bg"],      foreground=C["text2"],  font=FONT_SM)

        s.configure("TEntry",
                    fieldbackground=C["surface3"],
                    foreground=C["text"],
                    insertcolor=C["text"],
                    bordercolor=C["border"],
                    lightcolor=C["border"],
                    darkcolor=C["border"])
        s.map("TEntry", bordercolor=[("focus", C["blue"])])

        s.configure("TCombobox",
                    fieldbackground=C["surface3"],
                    background=C["surface2"],
                    foreground=C["text"],
                    selectbackground=C["blue"],
                    selectforeground=C["white"],
                    arrowcolor=C["text2"],
                    bordercolor=C["border"])
        s.map("TCombobox",
              fieldbackground=[("readonly", C["surface3"])],
              foreground=[("readonly", C["text"])],
              bordercolor=[("focus", C["blue"])])

        s.configure("TCheckbutton",
                    background=C["surface"],
                    foreground=C["text"],
                    font=FONT_BODY,
                    focuscolor=C["surface"])
        s.map("TCheckbutton",
              background=[("active", C["surface"])],
              indicatorcolor=[("selected", C["blue"]), ("!selected", C["surface3"])])

        s.configure("TRadiobutton",
                    background=C["surface"],
                    foreground=C["text"],
                    font=FONT_BODY,
                    focuscolor=C["surface"])
        s.map("TRadiobutton",
              background=[("active", C["surface"])],
              indicatorcolor=[("selected", C["blue"])])

        s.configure("TNotebook",     background=C["bg"], borderwidth=0)
        s.configure("TNotebook.Tab", background=C["surface2"],
                    foreground=C["text3"],
                    padding=[16, 6],
                    font=FONT_BOLD)
        s.map("TNotebook.Tab",
              background=[("selected", C["surface"])],
              foreground=[("selected", C["text"])])

        s.configure("TSeparator", background=C["border"])
        s.configure("TScrollbar",
                    background=C["surface2"],
                    troughcolor=C["bg"],
                    arrowcolor=C["text3"],
                    bordercolor=C["bg"],
                    lightcolor=C["bg"],
                    darkcolor=C["bg"],
                    gripcount=0,
                    arrowsize=10)

        # Treeview
        s.configure("Treeview",
                    background=C["surface"],
                    fieldbackground=C["surface"],
                    foreground=C["text"],
                    rowheight=24,
                    font=FONT_MONO,
                    borderwidth=0,
                    relief="flat")
        s.configure("Treeview.Heading",
                    background=C["surface2"],
                    foreground=C["text3"],
                    font=("Segoe UI", 8, "bold"),
                    relief="flat",
                    borderwidth=0)
        s.map("Treeview",
              background=[("selected", C["blue_glow"])],
              foreground=[("selected", C["white"])])
        s.map("Treeview.Heading",
              background=[("active", C["border"])])

    # -------------------------------------------------------------------------
    #  Title bar
    # -------------------------------------------------------------------------

    def _build_titlebar(self):
        bar = tk.Frame(self, bg=C["surface"], height=52)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        # Logo + Title
        tk.Label(bar, text="*", bg=C["surface"], fg=C["blue"],
                 font=("Segoe UI", 20)).pack(side="left", padx=(18, 6), pady=10)
        tk.Label(bar, text="NERO", bg=C["surface"], fg=C["white"],
                 font=("Segoe UI", 16, "bold")).pack(side="left", pady=10)
        tk.Label(bar, text="  v2  |  Neural Equity Ranking & Optimization  |  NSE India",
                 bg=C["surface"], fg=C["text3"],
                 font=("Segoe UI", 9)).pack(side="left", pady=10)

        # Right side - clock + engine status
        self._clock_var = tk.StringVar()
        tk.Label(bar, textvariable=self._clock_var,
                 bg=C["surface"], fg=C["text3"],
                 font=FONT_MONO).pack(side="right", padx=18)

        eng_color = C["teal"] if ENGINE_OK else C["red"]
        eng_text  = "ENGINE OK" if ENGINE_OK else "ENGINE MISSING"
        tk.Label(bar, text=eng_text,
                 bg=C["surface"], fg=eng_color,
                 font=("Segoe UI", 8, "bold")).pack(side="right", padx=12)

        # Separator
        tk.Frame(self, height=1, bg=C["border"]).pack(fill="x")
        self._tick_clock()

    def _tick_clock(self):
        self._clock_var.set(datetime.datetime.now().strftime("%a %d %b %Y  %H:%M:%S"))
        self.after(1000, self._tick_clock)

    # -------------------------------------------------------------------------
    #  Body - three columns
    # -------------------------------------------------------------------------

    def _build_body(self):
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True)

        # Left
        left = tk.Frame(body, bg=C["surface"], width=LEFT_W)
        left.pack(side="left", fill="y", padx=(0, 1))
        left.pack_propagate(False)

        # Scrollable left interior
        left_canvas = tk.Canvas(left, bg=C["surface"],
                                highlightthickness=0, width=LEFT_W - 10)
        left_sb = tk.Scrollbar(left, orient="vertical",
                               command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_sb.set)
        left_sb.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)

        self._left_inner = tk.Frame(left_canvas, bg=C["surface"])
        left_canvas.create_window((0, 0), window=self._left_inner,
                                  anchor="nw", width=LEFT_W - 18)
        self._left_inner.bind("<Configure>",
            lambda e: left_canvas.configure(
                scrollregion=left_canvas.bbox("all")))
        left_canvas.bind("<MouseWheel>",
            lambda e: left_canvas.yview_scroll(-1*(e.delta//120), "units"))

        # Centre
        centre = tk.Frame(body, bg=C["surface2"], width=CENTRE_W)
        centre.pack(side="left", fill="y", padx=(1, 1))
        centre.pack_propagate(False)
        self._centre = centre

        # Right - notebook
        right = tk.Frame(body, bg=C["bg"])
        right.pack(side="left", fill="both", expand=True, padx=(1, 0))
        self._right = right

        self._build_left(self._left_inner)
        self._build_centre(centre)
        self._build_right(right)

    # -------------------------------------------------------------------------
    #  Status bar
    # -------------------------------------------------------------------------

    def _build_statusbar(self):
        tk.Frame(self, height=1, bg=C["border"]).pack(fill="x")
        bar = tk.Frame(self, bg=C["surface"], height=26)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(bar, textvariable=self._status_var,
                 bg=C["surface"], fg=C["text3"],
                 font=("Segoe UI", 8, "italic"),
                 anchor="w").pack(side="left", padx=12, fill="y")

        self._rowcount_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._rowcount_var,
                 bg=C["surface"], fg=C["text3"],
                 font=FONT_SM).pack(side="right", padx=12)


    # =========================================================================
    #  LEFT PANEL - Configuration
    # =========================================================================

    def _sec(self, parent, title):
        """Section header row."""
        f = tk.Frame(parent, bg=C["surface"])
        f.pack(fill="x", padx=0, pady=(12, 4))
        tk.Label(f, text=title.upper(),
                 bg=C["surface"], fg=C["text3"],
                 font=("Segoe UI", 7, "bold")).pack(side="left", padx=12)
        tk.Frame(f, bg=C["border"], height=1).pack(
            side="left", fill="x", expand=True, padx=(4, 12))

    def _field(self, parent, label, widget_fn, tip=None):
        """Two-row label + widget field."""
        wrap = tk.Frame(parent, bg=C["surface"])
        wrap.pack(fill="x", padx=12, pady=2)
        lbl = tk.Label(wrap, text=label, bg=C["surface"], fg=C["text3"],
                       font=("Segoe UI", 8))
        lbl.pack(anchor="w")
        w = widget_fn(wrap)
        w.pack(fill="x", pady=(1, 0))
        if tip:
            Tooltip(w, tip)
        return w

    def _combo(self, parent, var, values, **kw):
        return ttk.Combobox(parent, textvariable=var,
                            values=values, state="readonly",
                            font=FONT_BODY, **kw)

    def _entry(self, parent, var, **kw):
        e = ttk.Entry(parent, textvariable=var, font=FONT_MONO, **kw)
        return e

    def _build_left(self, p):
        # Header
        tk.Label(p, text="Configuration",
                 bg=C["surface"], fg=C["text"],
                 font=FONT_LG).pack(anchor="w", padx=12, pady=(14, 0))
        tk.Label(p, text="Engine parameters & paths",
                 bg=C["surface"], fg=C["text3"],
                 font=FONT_SM).pack(anchor="w", padx=12, pady=(0, 4))

        # Engine params
        self._sec(p, "Engine")

        self._v_candle  = tk.StringVar(value=_env("NERO_CANDLE_MIN",    "5"))
        self._v_risk    = tk.StringVar(value=_env("NERO_RISK_MODE",     "medium"))
        self._v_years   = tk.StringVar(value=_env("NERO_SWING_YEARS",   "3"))
        self._v_capital = tk.StringVar(value="100000")
        self._v_minvol  = tk.StringVar(value=_env("NERO_MIN_VOLUME",    "0"))
        self._v_corr    = tk.StringVar(value=_env("NERO_CORR_LOOKBACK", "1Y"))

        self._field(p, "Candle Interval (min)",
                    lambda par: self._combo(par, self._v_candle, ["1","3","5"]),
                    tip="Minute granularity used for OHLCV resampling")
        self._field(p, "Risk Mode",
                    lambda par: self._combo(par, self._v_risk, ["low","medium","high"]),
                    tip="Controls position sizing aggressiveness")
        self._field(p, "Swing Lookback (years)",
                    lambda par: self._entry(par, self._v_years),
                    tip="Years of history for swing backtest")
        self._field(p, "Capital (Rs)",
                    lambda par: self._entry(par, self._v_capital),
                    tip="Total portfolio capital to allocate")
        self._field(p, "Min Volume Filter",
                    lambda par: self._entry(par, self._v_minvol),
                    tip="Skip stocks below this average daily volume")
        self._field(p, "Correlation Window",
                    lambda par: self._combo(par, self._v_corr, ["6M","1Y"]),
                    tip="Lookback window for correlation pruning")

        # Features
        self._sec(p, "Features")

        self._v_per_stock = tk.BooleanVar(value=(_env("NERO_PER_STOCK_STRAT","ON")=="ON"))
        self._v_news      = tk.BooleanVar(value=(_env("NERO_NEWS_ENABLED","OFF")=="ON"))
        self._v_ic        = tk.BooleanVar(value=(_env("NERO_IC_ADAPTIVE","ON")=="ON"))

        for label, var, tip in [
            ("Per-Stock Strategy Selection", self._v_per_stock,
             "Each stock gets individually backtested strategy (slower first run)"),
            ("News Filter  (NSE API)",       self._v_news,
             "Pull NSE announcements and filter stocks with negative news"),
            ("IC-Adaptive Factor Weights",   self._v_ic,
             "Dynamic weights using rolling Information Coefficient (AQR style)"),
        ]:
            row = tk.Frame(p, bg=C["surface"])
            row.pack(fill="x", padx=10, pady=2)
            cb = ttk.Checkbutton(row, text=label, variable=var)
            cb.pack(side="left")
            if tip:
                Tooltip(cb, tip)

        # Paths
        self._sec(p, "Paths")

        self._v_archive = tk.StringVar(value=_env("NERO_ARCHIVE_PATH", "C:/NERO/archive"))
        self._v_funda   = tk.StringVar(value=_env("NERO_FUNDA_PATH",   "C:/NERO/data/Stock_Funda_2000.csv"))

        self._path_field(p, "Archive Directory", self._v_archive, mode="dir")
        self._path_field(p, "Fundamentals CSV",  self._v_funda,  mode="file")

        # Telegram
        self._sec(p, "Telegram Alerts (optional)")

        self._v_tg_token = tk.StringVar(value=_env("NERO_TG_BOT_TOKEN",""))
        self._v_tg_chat  = tk.StringVar(value=_env("NERO_TG_CHAT_ID",  ""))

        self._field(p, "Bot Token",
                    lambda par: ttk.Entry(par, textvariable=self._v_tg_token,
                                          show="*", font=FONT_MONO))
        self._field(p, "Chat ID",
                    lambda par: ttk.Entry(par, textvariable=self._v_tg_chat,
                                          font=FONT_MONO))

        # Spacer at bottom
        tk.Frame(p, bg=C["surface"], height=20).pack()

    def _path_field(self, parent, label, var, mode="file"):
        wrap = tk.Frame(parent, bg=C["surface"])
        wrap.pack(fill="x", padx=12, pady=2)
        tk.Label(wrap, text=label, bg=C["surface"], fg=C["text3"],
                 font=("Segoe UI", 8)).pack(anchor="w")
        row = tk.Frame(wrap, bg=C["surface"])
        row.pack(fill="x")
        ttk.Entry(row, textvariable=var, font=FONT_MONO).pack(
            side="left", fill="x", expand=True)

        def _browse():
            if mode == "dir":
                path = filedialog.askdirectory(title=f"Select {label}")
            else:
                path = filedialog.askopenfilename(
                    title=f"Select {label}",
                    filetypes=[("CSV files","*.csv"),("All files","*.*")])
            if path:
                var.set(path)

        FlatButton(row, text="...", command=_browse,
                   bg=C["surface3"], hover=C["border2"],
                   font=("Segoe UI", 9), width=3).pack(side="left", padx=(4,0))


    # =========================================================================
    #  CENTRE PANEL - Run Controls
    # =========================================================================

    def _build_centre(self, p):
        tk.Label(p, text="Run Engine",
                 bg=C["surface2"], fg=C["text"],
                 font=FONT_LG).pack(anchor="w", padx=14, pady=(14, 2))
        tk.Label(p, text="Configure mode and execute",
                 bg=C["surface2"], fg=C["text3"],
                 font=FONT_SM).pack(anchor="w", padx=14)

        tk.Frame(p, height=1, bg=C["border"]).pack(fill="x", pady=10)

        # Mode tabs
        tk.Label(p, text="TRADING MODE",
                 bg=C["surface2"], fg=C["text3"],
                 font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=14)

        self._v_mode   = tk.StringVar(value="swing")
        tab_bar        = tk.Frame(p, bg=C["surface3"], padx=2, pady=2)
        tab_bar.pack(fill="x", padx=14, pady=(4, 0))

        self._mode_btns = {}
        for mode, label in [("swing","SWING"), ("intraday","INTRADAY")]:
            btn = tk.Button(
                tab_bar, text=label,
                bg=C["blue"] if mode=="swing" else C["surface3"],
                fg=C["white"] if mode=="swing" else C["text3"],
                font=FONT_BOLD, relief="flat", bd=0,
                cursor="hand2", padx=8, pady=5,
                command=lambda m=mode: self._select_mode(m))
            btn.pack(side="left", fill="x", expand=True, padx=2, pady=2)
            self._mode_btns[mode] = btn

        # Mode description
        self._mode_desc = tk.Label(
            p,
            text="Multi-week positions - Trend + Momentum\nFundamentals = primary ranking driver",
            bg=C["surface2"], fg=C["text3"],
            font=FONT_SM, justify="left")
        self._mode_desc.pack(anchor="w", padx=14, pady=(6, 0))

        tk.Frame(p, height=1, bg=C["border"]).pack(fill="x", pady=10)

        # Portfolio mode
        tk.Label(p, text="PORTFOLIO MODE",
                 bg=C["surface2"], fg=C["text3"],
                 font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=14)

        self._v_pmode = tk.StringVar(value="fresh")
        pm_row = tk.Frame(p, bg=C["surface2"])
        pm_row.pack(fill="x", padx=14, pady=(4, 0))

        self._pmode_btns = {}
        for pm, label in [("fresh","Fresh Portfolio"), ("update","Update Existing")]:
            btn = tk.Button(
                pm_row, text=label,
                bg=C["teal"] if pm=="fresh" else C["surface3"],
                fg=C["bg"]   if pm=="fresh" else C["text3"],
                font=FONT_BOLD, relief="flat", bd=0,
                cursor="hand2", padx=8, pady=4,
                command=lambda m=pm: self._select_pmode(m))
            btn.pack(side="left", fill="x", expand=True, padx=(0,4) if pm=="fresh" else 0)
            self._pmode_btns[pm] = btn

        # Holdings input (hidden by default)
        self._holdings_frame = tk.Frame(p, bg=C["surface2"])
        tk.Label(self._holdings_frame,
                 text="Current holdings  (SYMBOL:WEIGHT%, ...)",
                 bg=C["surface2"], fg=C["text3"],
                 font=("Segoe UI", 7)).pack(anchor="w", padx=14, pady=(6,2))
        self._holdings_text = tk.Text(
            self._holdings_frame,
            height=4, bg=C["surface3"], fg=C["text"],
            font=FONT_MONO, relief="flat", bd=4,
            insertbackground=C["text"], wrap="word")
        self._holdings_text.pack(fill="x", padx=14)

        tk.Frame(p, height=1, bg=C["border"]).pack(fill="x", pady=10)

        # RUN button
        self._btn_run = FlatButton(
            p, text="RUN NERO",
            bg=C["blue"], hover=C["blue_d"],
            font=("Segoe UI", 13, "bold"),
            height=2,
            command=self._on_run)
        self._btn_run.pack(fill="x", padx=14, pady=4)

        # Progress bar
        self._prog = AnimatedProgressBar(p, height=4)
        self._prog.pack(fill="x", padx=14, pady=(0, 2))

        # Status text
        self._status_var2 = tk.StringVar(value="Ready.")
        tk.Label(p, textvariable=self._status_var2,
                 bg=C["surface2"], fg=C["text3"],
                 font=("Segoe UI", 8, "italic"),
                 anchor="w").pack(fill="x", padx=14)

        tk.Frame(p, height=1, bg=C["border"]).pack(fill="x", pady=10)

        # Last run summary
        tk.Label(p, text="LAST RUN SUMMARY",
                 bg=C["surface2"], fg=C["text3"],
                 font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=14)

        self._info_vars = {}
        for key in ("Mode", "Regime", "Positions", "Capital"):
            row = tk.Frame(p, bg=C["surface2"])
            row.pack(fill="x", padx=14, pady=2)
            tk.Label(row, text=f"{key}",
                     bg=C["surface2"], fg=C["text3"],
                     font=("Segoe UI", 8), width=10, anchor="w").pack(side="left")
            v = tk.StringVar(value="--")
            self._info_vars[key] = v
            tk.Label(row, textvariable=v,
                     bg=C["surface2"], fg=C["text"],
                     font=("Consolas", 8, "bold")).pack(side="left")

        tk.Frame(p, height=1, bg=C["border"]).pack(fill="x", pady=10)

        # Action buttons
        tk.Label(p, text="EXPORT & ALERTS",
                 bg=C["surface2"], fg=C["text3"],
                 font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=14)

        self._btn_export = FlatButton(
            p, text="Export CSV",
            bg=C["surface3"], hover=C["border2"],
            fg=C["text"], command=self._on_export)
        self._btn_export.pack(fill="x", padx=14, pady=(4, 2))
        self._btn_export.config(state="disabled")

        self._btn_tg = FlatButton(
            p, text="Send Telegram",
            bg=C["surface3"], hover=C["border2"],
            fg=C["text"], command=self._on_send_telegram)
        self._btn_tg.pack(fill="x", padx=14, pady=2)
        self._btn_tg.config(state="disabled")

        # Flags warning + View Flags button (Bug #6 fix)
        self._flags_var = tk.StringVar(value="")
        flags_row = tk.Frame(p, bg=C["surface2"])
        flags_row.pack(fill="x", padx=14, pady=4)
        tk.Label(flags_row, textvariable=self._flags_var,
                 bg=C["surface2"], fg=C["amber"],
                 font=("Segoe UI", 7, "italic"),
                 wraplength=200, justify="left").pack(side="left", anchor="w")

        def _view_flags():
            flags_path = os.path.join(
                os.environ.get("NERO_OUTPUT_PATH", "C:/NERO/results"), "flags.csv")
            if os.path.isfile(flags_path):
                try:
                    os.startfile(flags_path)
                except Exception as exc:
                    messagebox.showerror("View Flags", f"Cannot open flags.csv:\n{exc}")
            else:
                messagebox.showinfo("View Flags", "No flags.csv found yet.\nRun NERO first.")

        self._btn_flags = FlatButton(
            flags_row, text="View Flags",
            bg=C["surface3"], hover=C["border2"],
            fg=C["amber"], font=("Segoe UI", 7, "bold"),
            command=_view_flags)
        self._btn_flags.pack(side="right", padx=(6, 0))

    def _select_mode(self, mode):
        self._v_mode.set(mode)
        descs = {
            "swing":    "Multi-week positions - Trend + Momentum\nFundamentals = primary ranking driver",
            "intraday": "Short-term setups - VWAP + ORB + VolumeShock\nFundamentals = hard gate only",
        }
        self._mode_desc.config(text=descs[mode])
        for m, btn in self._mode_btns.items():
            if m == mode:
                btn.config(bg=C["blue"], fg=C["white"])
            else:
                btn.config(bg=C["surface3"], fg=C["text3"])

    def _select_pmode(self, mode):
        self._v_pmode.set(mode)
        for m, btn in self._pmode_btns.items():
            if m == mode:
                btn.config(bg=C["teal"], fg=C["bg"])
            else:
                btn.config(bg=C["surface3"], fg=C["text3"])
        if mode == "update":
            self._holdings_frame.pack(fill="x", pady=(0, 6))
        else:
            self._holdings_frame.pack_forget()


    # =========================================================================
    #  RIGHT PANEL - Tabbed results
    # =========================================================================

    def _build_right(self, p):
        nb = ttk.Notebook(p)
        nb.pack(fill="both", expand=True)
        self._nb = nb

        # Tab 1: Portfolio table
        tab_port = tk.Frame(nb, bg=C["surface"])
        nb.add(tab_port, text="  Portfolio  ")
        self._build_tab_portfolio(tab_port)

        # Tab 2: Regime dashboard
        tab_regime = tk.Frame(nb, bg=C["surface"])
        nb.add(tab_regime, text="  Regime  ")
        self._build_tab_regime(tab_regime)

        # Tab 3: Scores breakdown
        tab_scores = tk.Frame(nb, bg=C["surface"])
        nb.add(tab_scores, text="  Score Detail  ")
        self._build_tab_scores(tab_scores)

        # Tab 4: Analysis (MVO / Monte Carlo / Backtest)
        tab_analysis = tk.Frame(nb, bg=C["bg"])
        nb.add(tab_analysis, text="  Analysis  ")
        self._build_tab_analysis(tab_analysis)

        # Tab 5: Log
        tab_log = tk.Frame(nb, bg=C["bg"])
        nb.add(tab_log, text="  Engine Log  ")
        self._build_tab_log(tab_log)

    # Tab 1: Portfolio

    def _build_tab_portfolio(self, p):
        # Header row
        hdr = tk.Frame(p, bg=C["surface"])
        hdr.pack(fill="x", pady=(8, 4), padx=10)

        tk.Label(hdr, text="Portfolio Results",
                 bg=C["surface"], fg=C["text"],
                 font=FONT_LG).pack(side="left")

        self._regime_badge = tk.Label(
            hdr, text="Regime: --",
            bg=C["surface3"], fg=C["neutral"],
            font=("Segoe UI", 10, "bold"),
            padx=10, pady=3)
        self._regime_badge.pack(side="right")

        # Summary cards row
        cards = tk.Frame(p, bg=C["surface"])
        cards.pack(fill="x", padx=10, pady=(0, 6))
        self._summary_cards = {}
        for key in ("Total Capital", "Positions", "Avg Score", "Avg CVaR"):
            card = tk.Frame(cards, bg=C["surface2"], padx=12, pady=6)
            card.pack(side="left", padx=(0, 6))
            tk.Label(card, text=key, bg=C["surface2"], fg=C["text3"],
                     font=("Segoe UI", 7)).pack(anchor="w")
            v = tk.StringVar(value="--")
            self._summary_cards[key] = v
            tk.Label(card, textvariable=v,
                     bg=C["surface2"], fg=C["white"],
                     font=("Consolas", 11, "bold")).pack(anchor="w")

        # Treeview
        tv_wrap = tk.Frame(p, bg=C["surface"])
        tv_wrap.pack(fill="both", expand=True, padx=10, pady=(0, 4))

        vsb = ttk.Scrollbar(tv_wrap, orient="vertical")
        hsb = ttk.Scrollbar(tv_wrap, orient="horizontal")

        self._tree = ttk.Treeview(
            tv_wrap,
            columns=[c[0] for c in TV_COLS],
            show="headings",
            selectmode="browse",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set)
        vsb.config(command=self._tree.yview)
        hsb.config(command=self._tree.xview)

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._tree.pack(side="left", fill="both", expand=True)

        for col_id, heading, width, anchor in TV_COLS:
            self._tree.heading(col_id, text=heading,
                               command=lambda c=col_id: self._sort_by(c))
            self._tree.column(col_id, width=width, minwidth=width,
                              anchor=anchor,
                              stretch=(col_id in ("symbol","strategy")))

        # Row tags
        self._tree.tag_configure("Asymmetric", background=C["asym_bg"], foreground=C["amber"])
        self._tree.tag_configure("Defensive",  background=C["def_bg"],  foreground=C["teal"])
        self._tree.tag_configure("Balanced",   background=C["bal_bg"],  foreground=C["blue"])
        self._tree.tag_configure("Flag",       font=("Consolas", 8, "italic"))

        # Row detail on click
        self._tree.bind("<<TreeviewSelect>>", self._on_row_select)

    # Tab 2: Regime dashboard

    def _build_tab_regime(self, p):
        top = tk.Frame(p, bg=C["surface"])
        top.pack(fill="x", padx=16, pady=14)

        tk.Label(top, text="Market Regime Dashboard",
                 bg=C["surface"], fg=C["text"],
                 font=FONT_LG).pack(anchor="w")
        tk.Label(top, text="Dynamic factor weights and regime classification",
                 bg=C["surface"], fg=C["text3"],
                 font=FONT_SM).pack(anchor="w")

        tk.Frame(p, height=1, bg=C["border"]).pack(fill="x", padx=16)

        # Regime label + description
        reg_row = tk.Frame(p, bg=C["surface"])
        reg_row.pack(fill="x", padx=16, pady=14)

        self._regime_big = tk.Label(
            reg_row, text="--",
            bg=C["surface"], fg=C["neutral"],
            font=("Segoe UI", 28, "bold"))
        self._regime_big.pack(side="left")

        self._regime_desc = tk.Label(
            reg_row,
            text="Run NERO to detect current market regime.",
            bg=C["surface"], fg=C["text3"],
            font=FONT_BODY, justify="left", wraplength=340)
        self._regime_desc.pack(side="left", padx=20)

        # Gauges row
        gauges_frame = tk.Frame(p, bg=C["surface"])
        gauges_frame.pack(fill="x", padx=16, pady=(0, 14))

        self._gauges = {}
        for key in ("Trend", "Volatility", "Breadth", "Correlation"):
            col = tk.Frame(gauges_frame, bg=C["surface"])
            col.pack(side="left", padx=(0, 24))
            g = RegimeGauge(col, label=key, size=80)
            g.pack()
            v = tk.StringVar(value="--")
            self._gauges[key] = (g, v)
            tk.Label(col, textvariable=v,
                     bg=C["surface"], fg=C["text3"],
                     font=("Consolas", 8)).pack()

        tk.Frame(p, height=1, bg=C["border"]).pack(fill="x", padx=16)

        # Factor weights table
        tk.Label(p, text="ACTIVE FACTOR WEIGHTS",
                 bg=C["surface"], fg=C["text3"],
                 font=("Segoe UI", 7, "bold")).pack(anchor="w", padx=16, pady=(12, 4))

        weights_frame = tk.Frame(p, bg=C["surface"])
        weights_frame.pack(fill="x", padx=16)

        self._weight_bars = {}
        for factor in ("Fundamental", "Technical", "Volatility", "Sentiment"):
            row = tk.Frame(weights_frame, bg=C["surface"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=factor, bg=C["surface"], fg=C["text3"],
                     font=FONT_SM, width=14, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row, bg=C["surface3"], height=12)
            bar_bg.pack(side="left", fill="x", expand=True)
            bar_fill = tk.Frame(bar_bg, bg=C["blue"], height=12)
            bar_fill.place(x=0, y=0, relwidth=0.0, relheight=1.0)
            val_lbl = tk.Label(row, text="--%", bg=C["surface"], fg=C["text"],
                                font=("Consolas", 8), width=6)
            val_lbl.pack(side="left", padx=(6, 0))
            self._weight_bars[factor] = (bar_fill, val_lbl)

    # Tab 3: Score detail

    def _build_tab_scores(self, p):
        top = tk.Frame(p, bg=C["surface"])
        top.pack(fill="x", padx=16, pady=14)

        tk.Label(top, text="Stock Score Breakdown",
                 bg=C["surface"], fg=C["text"],
                 font=FONT_LG).pack(anchor="w")
        tk.Label(top, text="Click a row in the Portfolio tab to inspect",
                 bg=C["surface"], fg=C["text3"],
                 font=FONT_SM).pack(anchor="w")

        tk.Frame(p, height=1, bg=C["border"]).pack(fill="x", padx=16)

        # Selected stock header
        self._sel_symbol = tk.StringVar(value="No selection")
        self._sel_bucket = tk.StringVar(value="")
        hdr = tk.Frame(p, bg=C["surface"])
        hdr.pack(fill="x", padx=16, pady=10)
        tk.Label(hdr, textvariable=self._sel_symbol,
                 bg=C["surface"], fg=C["white"],
                 font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(hdr, textvariable=self._sel_bucket,
                 bg=C["surface3"], fg=C["teal"],
                 font=("Segoe UI", 9, "bold"),
                 padx=8, pady=3).pack(side="left", padx=12)

        # Score bars
        bars_frame = tk.Frame(p, bg=C["surface"])
        bars_frame.pack(fill="x", padx=16, pady=6)

        self._score_rows = {}
        labels = [
            ("Combined Score", C["blue"],   100),
            ("Fund Score",     C["teal"],   100),
            ("Tech Score",     C["purple"], 100),
            ("Vol Score",      C["amber"],  100),
        ]
        for name, color, max_v in labels:
            row = tk.Frame(bars_frame, bg=C["surface"])
            row.pack(fill="x", pady=4)
            tk.Label(row, text=name, bg=C["surface"], fg=C["text3"],
                     font=FONT_SM, width=16, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row, bg=C["surface3"], height=16)
            bar_bg.pack(side="left", fill="x", expand=True)
            bar_fill = tk.Frame(bar_bg, bg=color, height=16)
            bar_fill.place(x=0, y=0, relwidth=0.0, relheight=1.0)
            val_lbl = tk.Label(row, text="--", bg=C["surface"], fg=color,
                                font=("Consolas", 10, "bold"), width=8)
            val_lbl.pack(side="left", padx=(8, 0))
            self._score_rows[name] = (bar_fill, val_lbl, max_v)

        tk.Frame(p, height=1, bg=C["border"]).pack(fill="x", padx=16, pady=8)

        # Details grid
        detail_frame = tk.Frame(p, bg=C["surface"])
        detail_frame.pack(fill="x", padx=16)

        self._detail_vars = {}
        detail_keys = [
            ("Symbol",    "symbol"),   ("Strategy",    "Strategy"),
            ("Bucket",    "Bucket"),   ("Weight",      "Weight"),
            ("Capital Rs", "CapitalAllocated"), ("EV", "EV"),
            ("CVaR 95%",  "CVaR95"),   ("Funda Missing","FundaMissing"),
        ]
        for i, (label, key) in enumerate(detail_keys):
            row_i = i // 2
            col_i = i  % 2
            cell = tk.Frame(detail_frame, bg=C["surface2"], padx=10, pady=6)
            cell.grid(row=row_i, column=col_i, padx=(0,6), pady=(0,6), sticky="ew")
            detail_frame.columnconfigure(col_i, weight=1)
            tk.Label(cell, text=label, bg=C["surface2"], fg=C["text3"],
                     font=("Segoe UI", 7)).pack(anchor="w")
            v = tk.StringVar(value="--")
            self._detail_vars[key] = v
            tk.Label(cell, textvariable=v,
                     bg=C["surface2"], fg=C["white"],
                     font=("Consolas", 9, "bold")).pack(anchor="w")

    # Tab 4: Analysis

    def _build_tab_analysis(self, p):
        """
        Analysis tab — displays MVO, Monte Carlo, and Portfolio Backtest results
        loaded from C:/NERO/results after a run.  Layout:
          - Top: sub-tabs for each analysis section
          - Each sub-tab: metric cards + embedded PNG chart
        """
        import tkinter.font as tkfont

        # ── Outer header ────────────────────────────────────────────────────
        hdr = tk.Frame(p, bg=C["bg"])
        hdr.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(hdr, text="Portfolio Analysis",
                 bg=C["bg"], fg=C["text"], font=FONT_LG).pack(side="left")
        self._btn_refresh_analysis = FlatButton(
            hdr, text="↻  Refresh",
            bg=C["surface3"], hover=C["border2"],
            fg=C["text2"], font=FONT_SM,
            command=self._refresh_analysis_tab)
        self._btn_refresh_analysis.pack(side="right")

        # ── Sub-notebook (MVO | Monte Carlo | Backtest | Walk-Forward) ──────
        sub_nb = ttk.Notebook(p)
        sub_nb.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._analysis_sub_nb = sub_nb

        # Frames for each sub-tab
        self._tab_mvo   = tk.Frame(sub_nb, bg=C["surface"])
        self._tab_mc    = tk.Frame(sub_nb, bg=C["surface"])
        self._tab_bt    = tk.Frame(sub_nb, bg=C["surface"])
        self._tab_wf    = tk.Frame(sub_nb, bg=C["surface"])

        sub_nb.add(self._tab_mvo, text="  Efficient Frontier  ")
        sub_nb.add(self._tab_mc,  text="  Monte Carlo  ")
        sub_nb.add(self._tab_bt,  text="  Portfolio Backtest  ")
        sub_nb.add(self._tab_wf,  text="  Walk-Forward  ")

        # Place-holder messages — replaced by _refresh_analysis_tab()
        for frame, label in [
            (self._tab_mvo,  "Run NERO to generate MVO / Efficient Frontier"),
            (self._tab_mc,   "Run NERO to generate Monte Carlo simulation"),
            (self._tab_bt,   "Run NERO to generate Portfolio Backtest"),
            (self._tab_wf,   "Run NERO to generate Walk-Forward OOS results"),
        ]:
            tk.Label(frame, text=label,
                     bg=C["surface"], fg=C["text3"],
                     font=FONT_BODY).pack(expand=True)

        # Store refs to photo images so GC doesn't collect them
        self._analysis_images = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _analysis_clear(self, frame):
        """Destroy all children of a tab frame."""
        for w in frame.winfo_children():
            w.destroy()

    def _analysis_metric_row(self, parent, metrics):
        """
        Render a horizontal strip of (label, value, colour) metric cards.
        metrics = list of (label, value_str, color_key_or_hex)
        """
        row = tk.Frame(parent, bg=C["surface"])
        row.pack(fill="x", padx=8, pady=(6, 4))
        for label, value, color in metrics:
            card = tk.Frame(row, bg=C["surface2"], padx=14, pady=8)
            card.pack(side="left", padx=(0, 6), fill="y")
            tk.Label(card, text=label,
                     bg=C["surface2"], fg=C["text3"],
                     font=("Segoe UI", 7)).pack(anchor="w")
            tk.Label(card, text=value,
                     bg=C["surface2"], fg=color,
                     font=("Consolas", 11, "bold")).pack(anchor="w")

    def _analysis_png(self, parent, png_path, label):
        """Embed a PNG into a scrollable canvas inside parent."""
        try:
            from PIL import Image as PILImage, ImageTk
            img = PILImage.open(png_path)
            # Scale to fit ~900px wide while keeping aspect
            max_w = 900
            w, h  = img.size
            if w > max_w:
                img = img.resize((max_w, int(h * max_w / w)), PILImage.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._analysis_images[label] = photo  # keep reference
            lbl = tk.Label(parent, image=photo, bg=C["surface"])
            lbl.pack(pady=6)
        except ImportError:
            # Pillow not installed — show open-file button instead
            tk.Label(parent,
                     text="Install Pillow (pip install pillow) to view charts inline.",
                     bg=C["surface"], fg=C["amber"], font=FONT_SM).pack(pady=4)
            FlatButton(parent, text=f"Open {os.path.basename(png_path)} externally",
                       command=lambda p=png_path: os.startfile(p) if os.path.isfile(p) else None,
                       bg=C["surface3"], hover=C["border2"],
                       fg=C["text2"], font=FONT_SM).pack(pady=2)
        except Exception as exc:
            tk.Label(parent, text=f"Chart error: {exc}",
                     bg=C["surface"], fg=C["red"], font=FONT_SM).pack(pady=4)

    def _scrollable(self, parent):
        """Return a scrollable Frame inside parent."""
        canvas = tk.Canvas(parent, bg=C["surface"], highlightthickness=0)
        vsb    = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas, bg=C["surface"])
        win   = canvas.create_window((0, 0), window=inner, anchor="nw")
        def _on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win, width=canvas.winfo_width())
        inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        # Mouse-wheel scrolling
        def _on_wheel(e):
            canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_wheel)
        return inner

    def _refresh_analysis_tab(self):
        """
        Read output CSVs / PNGs from NERO_OUTPUT_PATH and populate
        the MVO, Monte Carlo, Backtest, and Walk-Forward sub-tabs.
        Called automatically after every run and manually via Refresh button.
        """
        import pandas as pd
        out = os.environ.get("NERO_OUTPUT_PATH", "C:/NERO/results")

        # ── 1. MVO / Efficient Frontier ──────────────────────────────────────
        self._analysis_clear(self._tab_mvo)
        ef_csv = os.path.join(out, "efficient_frontier.csv")
        ef_png = os.path.join(out, "efficient_frontier.png")
        inner  = self._scrollable(self._tab_mvo)
        tk.Label(inner, text="Mean-Variance Optimization — Efficient Frontier",
                 bg=C["surface"], fg=C["text"], font=FONT_LG).pack(anchor="w", padx=10, pady=(10, 2))
        tk.Label(inner, text="Max-Sharpe and Min-Vol portfolios on the efficient frontier",
                 bg=C["surface"], fg=C["text3"], font=FONT_SM).pack(anchor="w", padx=10)
        if os.path.isfile(ef_csv):
            try:
                ef_df = pd.read_csv(ef_csv)
                # Pull best Sharpe row
                best  = ef_df.loc[ef_df["Sharpe"].idxmax()]
                worst_vol = ef_df.loc[ef_df["Volatility"].idxmin()]
                metrics = [
                    ("Max Sharpe",         f"{best['Sharpe']:.3f}",                   C["teal"]),
                    ("Return @ Max Sharpe",f"{best['Return']*100:.2f}%",               C["teal"]),
                    ("Vol @ Max Sharpe",   f"{best['Volatility']*100:.2f}%",           C["amber"]),
                    ("Min Volatility",     f"{worst_vol['Volatility']*100:.2f}%",      C["blue"]),
                    ("Return @ Min Vol",   f"{worst_vol['Return']*100:.2f}%",          C["text"]),
                    ("Frontier Points",    f"{len(ef_df)}",                            C["text3"]),
                ]
                self._analysis_metric_row(inner, metrics)
            except Exception as exc:
                tk.Label(inner, text=f"CSV read error: {exc}",
                         bg=C["surface"], fg=C["red"], font=FONT_SM).pack(padx=10)
        else:
            tk.Label(inner, text="efficient_frontier.csv not found — run NERO first.",
                     bg=C["surface"], fg=C["text3"], font=FONT_SM).pack(padx=10, pady=4)
        if os.path.isfile(ef_png):
            self._analysis_png(inner, ef_png, "mvo")
        else:
            FlatButton(inner, text="Open efficient_frontier.png folder",
                       command=lambda: os.startfile(out),
                       bg=C["surface3"], hover=C["border2"],
                       fg=C["text2"], font=FONT_SM).pack(pady=6, padx=10)

        # ── 2. Monte Carlo ────────────────────────────────────────────────────
        self._analysis_clear(self._tab_mc)
        mc_csv = os.path.join(out, "monte_carlo.csv")
        mc_png = os.path.join(out, "monte_carlo.png")
        inner  = self._scrollable(self._tab_mc)
        tk.Label(inner, text="Monte Carlo Simulation — 10,000 Paths",
                 bg=C["surface"], fg=C["text"], font=FONT_LG).pack(anchor="w", padx=10, pady=(10, 2))
        tk.Label(inner, text="Cholesky-correlated forward simulation over 252 trading days",
                 bg=C["surface"], fg=C["text3"], font=FONT_SM).pack(anchor="w", padx=10)
        if os.path.isfile(mc_csv):
            try:
                mc_df = pd.read_csv(mc_csv).set_index("Metric")["Value"]
                def _mc(key, default="--"):
                    v = mc_df.get(key, default)
                    return str(v) if isinstance(v, str) else v
                def _fmt_rs(v):
                    try:    return f"₹{float(v):,.0f}"
                    except: return str(v)
                def _fmt_pct2(v):
                    try:    return f"{float(v)*100:.2f}%"
                    except: return str(v)

                metrics1 = [
                    ("Starting Capital",         _fmt_rs(_mc("Starting Capital")),        C["text3"]),
                    ("Expected Terminal Value",   _fmt_rs(_mc("Expected Terminal Value")), C["teal"]),
                    ("Median Terminal Value",     _fmt_rs(_mc("Median Terminal Value")),   C["teal"]),
                    ("Best Case (95th pct)",      _fmt_rs(_mc("Best Case (95th pct)")),   C["bull"]),
                    ("Worst Case (5th pct)",      _fmt_rs(_mc("Worst Case (5th pct)")),   C["red"]),
                ]
                metrics2 = [
                    ("VaR 95% (loss)",            _fmt_rs(_mc("VaR 95% (loss)")),         C["red"]),
                    ("CVaR 95% (exp loss)",       _fmt_rs(_mc("CVaR 95% (exp loss)")),    C["red"]),
                    ("Annualised Return",          _fmt_pct2(_mc("Annualised Return")),    C["teal"]),
                    ("Annualised Volatility",      _fmt_pct2(_mc("Annualised Volatility")),C["amber"]),
                    ("Simulations",               str(int(float(_mc("Simulations","10000")))), C["text3"]),
                    ("Horizon (days)",            str(int(float(_mc("Horizon (days)","252")))),C["text3"]),
                ]
                self._analysis_metric_row(inner, metrics1)
                self._analysis_metric_row(inner, metrics2)
            except Exception as exc:
                tk.Label(inner, text=f"CSV read error: {exc}",
                         bg=C["surface"], fg=C["red"], font=FONT_SM).pack(padx=10)
        else:
            tk.Label(inner, text="monte_carlo.csv not found — run NERO first.",
                     bg=C["surface"], fg=C["text3"], font=FONT_SM).pack(padx=10, pady=4)
        if os.path.isfile(mc_png):
            self._analysis_png(inner, mc_png, "mc")
        else:
            FlatButton(inner, text="Open results folder",
                       command=lambda: os.startfile(out),
                       bg=C["surface3"], hover=C["border2"],
                       fg=C["text2"], font=FONT_SM).pack(pady=6, padx=10)

        # ── 3. Portfolio Backtest ─────────────────────────────────────────────
        self._analysis_clear(self._tab_bt)
        bt_csv = os.path.join(out, "portfolio_backtest.csv")
        bt_png = os.path.join(out, "portfolio_backtest.png")
        inner  = self._scrollable(self._tab_bt)
        tk.Label(inner, text="Portfolio Backtest vs Nifty50",
                 bg=C["surface"], fg=C["text"], font=FONT_LG).pack(anchor="w", padx=10, pady=(10, 2))
        tk.Label(inner, text="Weighted portfolio return series vs Nifty50 benchmark",
                 bg=C["surface"], fg=C["text3"], font=FONT_SM).pack(anchor="w", padx=10)
        if os.path.isfile(bt_csv):
            try:
                bt_df = pd.read_csv(bt_csv)
                def _bt(metric, col="Portfolio"):
                    row = bt_df[bt_df["Metric"]==metric]
                    if row.empty: return "--"
                    return str(row.iloc[0][col])
                def _color_ret(s):
                    try:    return C["teal"] if float(s.strip("%")) >= 0 else C["red"]
                    except: return C["text"]

                ann_ret = _bt("Annualised Return")
                sharpe  = _bt("Sharpe Ratio")
                max_dd  = _bt("Max Drawdown")
                calmar  = _bt("Calmar Ratio")
                cum_ret = _bt("Cumulative Return")
                nifty_r = _bt("Annualised Return", "Nifty50")
                nifty_sr= _bt("Sharpe Ratio",      "Nifty50")

                metrics1 = [
                    ("Cumulative Return",   cum_ret,    _color_ret(cum_ret)),
                    ("Ann. Return (NERO)",  ann_ret,    _color_ret(ann_ret)),
                    ("Ann. Return (Nifty)", nifty_r,    _color_ret(nifty_r)),
                    ("Sharpe (NERO)",       sharpe,     C["teal"]),
                    ("Sharpe (Nifty)",      nifty_sr,   C["text"]),
                ]
                metrics2 = [
                    ("Max Drawdown",        max_dd,     C["red"]),
                    ("Calmar Ratio",        calmar,     C["purple"]),
                    ("Period Start",        _bt("Period Start"),  C["text3"]),
                    ("Period End",          _bt("Period End"),    C["text3"]),
                    ("Trading Days",        _bt("Trading Days"),  C["text3"]),
                ]
                self._analysis_metric_row(inner, metrics1)
                self._analysis_metric_row(inner, metrics2)

                # OOS Sharpe table (portfolio_with_oos.csv)
                oos_path = os.path.join(out, "portfolio_with_oos.csv")
                if os.path.isfile(oos_path):
                    try:
                        oos_df = pd.read_csv(oos_path)[["Symbol","OOS_Sharpe"]].dropna()
                        if not oos_df.empty:
                            tk.Label(inner,
                                     text="Per-Stock Out-of-Sample Sharpe",
                                     bg=C["surface"], fg=C["text2"],
                                     font=FONT_BOLD).pack(anchor="w", padx=10, pady=(10, 2))
                            tbl = tk.Frame(inner, bg=C["surface"])
                            tbl.pack(fill="x", padx=10, pady=2)
                            # Header
                            for col, txt, w in [("symbol","Symbol",100),("oos","OOS Sharpe",90)]:
                                tk.Label(tbl, text=txt, bg=C["surface2"], fg=C["text3"],
                                         font=("Segoe UI",7,"bold"), width=w//7,
                                         anchor="w").grid(row=0, column=list("so").index(col[0]),
                                                          sticky="ew", padx=2, pady=1)
                            for r, (_, row) in enumerate(oos_df.iterrows(), 1):
                                sh = float(row["OOS_Sharpe"])
                                col = C["teal"] if sh >= 0.5 else (C["amber"] if sh >= 0 else C["red"])
                                tk.Label(tbl, text=str(row["Symbol"]),
                                         bg=C["surface"], fg=C["text"],
                                         font=FONT_MONO).grid(row=r, column=0, sticky="w", padx=4)
                                tk.Label(tbl, text=f"{sh:.3f}",
                                         bg=C["surface"], fg=col,
                                         font=("Consolas",9,"bold")).grid(row=r, column=1, sticky="w", padx=4)
                    except Exception:
                        pass
            except Exception as exc:
                tk.Label(inner, text=f"CSV read error: {exc}",
                         bg=C["surface"], fg=C["red"], font=FONT_SM).pack(padx=10)
        else:
            tk.Label(inner, text="portfolio_backtest.csv not found — run NERO first.",
                     bg=C["surface"], fg=C["text3"], font=FONT_SM).pack(padx=10, pady=4)
        if os.path.isfile(bt_png):
            self._analysis_png(inner, bt_png, "bt")
        else:
            FlatButton(inner, text="Open results folder",
                       command=lambda: os.startfile(out),
                       bg=C["surface3"], hover=C["border2"],
                       fg=C["text2"], font=FONT_SM).pack(pady=6, padx=10)

        # ── 4. Walk-Forward OOS ───────────────────────────────────────────────
        self._analysis_clear(self._tab_wf)
        wf_csv = os.path.join(out, "walkforward_oos.csv")
        inner  = self._scrollable(self._tab_wf)
        tk.Label(inner, text="Walk-Forward Out-of-Sample Validation",
                 bg=C["surface"], fg=C["text"], font=FONT_LG).pack(anchor="w", padx=10, pady=(10, 2))
        tk.Label(inner, text="6-fold sequential OOS Sharpe — detects overfitting",
                 bg=C["surface"], fg=C["text3"], font=FONT_SM).pack(anchor="w", padx=10)
        if os.path.isfile(wf_csv):
            try:
                wf_df = pd.read_csv(wf_csv)
                mean_oos = wf_df["OOS_Sharpe"].mean()
                color = C["teal"] if mean_oos >= 0.5 else (C["amber"] if mean_oos >= 0 else C["red"])
                fit_label = "Good" if mean_oos >= 0.5 else ("Borderline" if mean_oos >= 0 else "Overfit ⚠")
                metrics = [
                    ("Mean OOS Sharpe", f"{mean_oos:.3f}", color),
                    ("Folds",           str(len(wf_df)),   C["text3"]),
                    ("Fit Assessment",  fit_label,          color),
                ]
                self._analysis_metric_row(inner, metrics)

                # Fold bar chart (pure tkinter)
                chart_frame = tk.Frame(inner, bg=C["surface2"], padx=12, pady=10)
                chart_frame.pack(fill="x", padx=10, pady=6)
                tk.Label(chart_frame, text="OOS Sharpe per Fold",
                         bg=C["surface2"], fg=C["text2"],
                         font=FONT_BOLD).pack(anchor="w")
                bar_area = tk.Frame(chart_frame, bg=C["surface2"])
                bar_area.pack(fill="x", pady=4)
                max_abs = max(abs(wf_df["OOS_Sharpe"].max()),
                              abs(wf_df["OOS_Sharpe"].min()), 0.01)
                for _, row in wf_df.iterrows():
                    sh   = float(row["OOS_Sharpe"])
                    fold = int(row["Fold"])
                    pct  = min(abs(sh) / max_abs, 1.0)
                    col  = C["teal"] if sh >= 0 else C["red"]
                    r    = tk.Frame(bar_area, bg=C["surface2"])
                    r.pack(fill="x", pady=1)
                    tk.Label(r, text=f"Fold {fold}", bg=C["surface2"],
                             fg=C["text3"], font=FONT_SM, width=6).pack(side="left")
                    bar_bg = tk.Frame(r, bg=C["surface3"], height=14, width=300)
                    bar_bg.pack(side="left", padx=4)
                    bar_bg.pack_propagate(False)
                    tk.Frame(bar_bg, bg=col, height=14,
                             width=int(300*pct)).place(x=0, y=0)
                    tk.Label(r, text=f"{sh:+.3f}", bg=C["surface2"],
                             fg=col, font=("Consolas",8,"bold")).pack(side="left", padx=4)
            except Exception as exc:
                tk.Label(inner, text=f"CSV read error: {exc}",
                         bg=C["surface"], fg=C["red"], font=FONT_SM).pack(padx=10)
        else:
            tk.Label(inner, text="walkforward_oos.csv not found — run NERO first.",
                     bg=C["surface"], fg=C["text3"], font=FONT_SM).pack(padx=10, pady=4)

    # Tab 5: Log

    def _build_tab_log(self, p):
        top = tk.Frame(p, bg=C["bg"])
        top.pack(fill="x", padx=10, pady=(8, 4))
        tk.Label(top, text="Engine Log",
                 bg=C["bg"], fg=C["text"],
                 font=FONT_LG).pack(side="left")
        FlatButton(top, text="Clear",
                   bg=C["surface3"], hover=C["border2"],
                   fg=C["text3"], font=FONT_SM,
                   command=lambda: self._log.clear()).pack(side="right")

        self._log = LogPane(p)
        self._log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self._log.append("NERO v2 UI started.", "head")
        if not ENGINE_OK:
            self._log.append(f"ENGINE IMPORT FAILED: {_ENGINE_IMPORT_ERR}", "err")
        else:
            self._log.append("nero_v2 engine loaded successfully.", "ok")


    # =========================================================================
    #  SORTING
    # =========================================================================

    def _sort_by(self, col):
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = False

        rows = [(self._tree.set(iid, col), iid)
                for iid in self._tree.get_children("")]

        def _key(item):
            v = item[0]
            try:
                return (0, float(v.replace("%","").replace(",","")
                                  .replace("Rs","").replace(" ","")))
            except Exception:
                return (1, str(v).lower())

        rows.sort(key=_key, reverse=self._sort_rev)
        for idx, (_, iid) in enumerate(rows):
            self._tree.move(iid, "", idx)
            self._tree.set(iid, "rank", str(idx + 1))


    # =========================================================================
    #  ROW SELECTION - Score Detail tab
    # =========================================================================

    def _on_row_select(self, _=None):
        sel = self._tree.selection()
        if not sel or self._portfolio is None:
            return
        iid   = sel[0]
        vals  = self._tree.item(iid, "values")
        if not vals:
            return

        # Robust strip: handles trailing " *", extra spaces, unicode variants
        symbol = str(vals[1]).strip().rstrip("*").strip()
        bucket = str(vals[6])

        self._sel_symbol.set(symbol)
        self._sel_bucket.set(f"  {bucket}  ")

        # Switch tab first — ensures Score Detail is visible even if data lookup fails
        self._nb.select(2)

        row = self._portfolio[self._portfolio["Symbol"]==symbol]
        if row.empty:
            return
        row = row.iloc[0]

        # Score bars
        for name, col, max_v in [
            ("Combined Score", "CombinedScore", 100),
            ("Fund Score",     "FundScore",     100),
            ("Tech Score",     "TechScore",     100),
            ("Vol Score",      "VolScore",      100),
        ]:
            v = float(row.get(col, 0) or 0)
            rel = max(0.0, min(1.0, v / max_v)) if max_v else 0
            bar_fill, val_lbl, _ = self._score_rows[name]
            bar_fill.place_configure(relwidth=rel)
            val_lbl.config(text=f"{v:.1f}")

        # Detail cells
        for key, v in [
            ("symbol",          symbol),
            ("Strategy",        row.get("Strategy","--")),
            ("Bucket",          bucket),
            ("Weight",          _fmt_pct(row.get("Weight",0))),
            ("CapitalAllocated",_fmt_capital(row.get("CapitalAllocated",0))),
            ("EV",              _fmt_f(row.get("EV",0),4)),
            ("CVaR95",          _fmt_f(row.get("CVaR95",0),3)),
            ("FundaMissing",    "Yes *" if row.get("FundaMissing") else "No"),
        ]:
            if key in self._detail_vars:
                self._detail_vars[key].set(str(v))


    # =========================================================================
    #  RUN LOGIC
    # =========================================================================

    def _collect(self):
        try:
            capital = float(self._v_capital.get() or 100000)
        except ValueError:
            raise ValueError("Capital must be a number.")
        return {
            "mode":      self._v_mode.get(),
            "risk":      self._v_risk.get(),
            "capital":   capital,
            "candle":    self._v_candle.get(),
            "years":     self._v_years.get(),
            "minvol":    self._v_minvol.get(),
            "corr":      self._v_corr.get(),
            "per_stock": "ON" if self._v_per_stock.get() else "OFF",
            "news":      "ON" if self._v_news.get()      else "OFF",
            "ic":        "ON" if self._v_ic.get()        else "OFF",
            "archive":   self._v_archive.get(),
            "funda":     self._v_funda.get(),
            "tg_token":  self._v_tg_token.get(),
            "tg_chat":   self._v_tg_chat.get(),
            "pmode":     self._v_pmode.get(),
            "holdings":  self._holdings_text.get("1.0","end").strip()
                         if self._v_pmode.get()=="update" else "",
        }

    def _on_run(self):
        if self._thread and self._thread.is_alive():
            messagebox.showwarning("Running", "NERO is already running. Please wait.")
            return
        if not ENGINE_OK:
            messagebox.showerror("Engine Missing",
                f"Cannot import nero_v2.py:\n{_ENGINE_IMPORT_ERR}\n\n"
                "Ensure nero_v2.py is in the same folder.")
            return
        try:
            s = self._collect()
        except ValueError as exc:
            messagebox.showerror("Settings Error", str(exc))
            return

        existing = {}
        if s["pmode"]=="update" and s["holdings"]:
            for pair in s["holdings"].split(","):
                pair = pair.strip()
                if ":" in pair:
                    sym, wt = pair.rsplit(":",1)
                    try:
                        existing[sym.strip().upper().replace(".NS","")] = float(wt.strip())
                    except ValueError:
                        pass

        # Clear results
        for iid in self._tree.get_children(""):
            self._tree.delete(iid)
        self._rowcount_var.set("")
        self._flags_var.set("")
        self._log.clear()
        self._log.append(f"Starting NERO | Mode={s['mode'].upper()} | "
                         f"Risk={s['risk']} | Capital=Rs{s['capital']:,.0f}", "head")

        # Push env
        _set_env("NERO_CANDLE_MIN",      s["candle"])
        _set_env("NERO_RISK_MODE",       s["risk"])
        _set_env("NERO_SWING_YEARS",     s["years"])
        _set_env("NERO_MIN_VOLUME",      s["minvol"])
        _set_env("NERO_CORR_LOOKBACK",   s["corr"])
        _set_env("NERO_PER_STOCK_STRAT", s["per_stock"])
        _set_env("NERO_NEWS_ENABLED",    s["news"])
        _set_env("NERO_IC_ADAPTIVE",     s["ic"])
        _set_env("NERO_ARCHIVE_PATH",    s["archive"])
        _set_env("NERO_FUNDA_PATH",      s["funda"])
        _set_env("NERO_TG_BOT_TOKEN",    s["tg_token"])
        _set_env("NERO_TG_CHAT_ID",      s["tg_chat"])

        # UI: running state
        self._btn_run.config(state="disabled", text="Running...")
        self._btn_export.config(state="disabled")
        self._btn_tg.config(state="disabled")
        self._prog.start()
        self._set_status("Loading data...")

        self._thread = threading.Thread(
            target=self._worker, args=(s, existing), daemon=True)
        self._thread.start()

    def _worker(self, s, existing):
        try:
            self._q.put(("status","Loading data..."))

            # Push settings to engine globals
            engine.NERO_ARCHIVE_PATH    = s["archive"]
            engine.NERO_FUNDA_PATH      = s["funda"]
            engine.NERO_CANDLE_MIN      = int(s["candle"])
            engine.NERO_SWING_YEARS     = int(s["years"])
            engine.NERO_MIN_VOLUME      = float(s["minvol"] or 0)
            engine.NERO_CORR_LOOKBACK   = s["corr"]
            engine.NERO_RISK_MODE       = s["risk"]
            engine.NERO_PER_STOCK_STRAT = s["per_stock"]
            engine.NERO_NEWS_ENABLED    = s["news"]
            engine.NERO_IC_ADAPTIVE     = s["ic"]
            engine.NERO_TG_BOT_TOKEN    = s["tg_token"]
            engine.NERO_TG_CHAT_ID      = s["tg_chat"]

            self._q.put(("status","Computing signals..."))
            results_df, regime_label, regime_vector, _bar_returns, _daily_returns = engine.run_engine(mode=s["mode"])

            if results_df.empty:
                self._q.put(("warn",
                    "No stocks passed all pruning gates.\n"
                    "Try: lowering MinVolume, or checking your archive path."))
                return

            if s["news"] == "ON":
                self._q.put(("status","Applying news filter..."))
                try:
                    results_df = engine.apply_news_filter(results_df)
                except Exception as exc:
                    self._q.put(("status",f"News filter error: {exc} -- skipped"))

            self._q.put(("status","Building portfolio..."))
            portfolio_df = engine.build_portfolio(
                results_df,
                total_capital=s["capital"],
                risk_mode=s["risk"],
                existing_holdings=existing or None,
            )

            # Auto-save
            try:
                out_dir = os.environ.get("NERO_OUTPUT_PATH","C:/NERO/results")
                os.makedirs(out_dir, exist_ok=True)
                portfolio_df.to_csv(os.path.join(out_dir,"portfolio.csv"), index=False)
            except Exception:
                pass

            # ── MVO / Monte Carlo / Portfolio Backtest (Bug #1 fix) ──────────
            self._q.put(("status","Running MVO (efficient frontier)..."))
            try:
                engine.run_mvo(
                    results_df=portfolio_df,
                    bar_returns_dict=_bar_returns,
                    risk_mode=s["risk"],
                )
            except Exception as exc:
                self._q.put(("status", f"MVO skipped: {exc}"))

            self._q.put(("status","Running Monte Carlo simulation..."))
            try:
                engine.run_monte_carlo(
                    portfolio_df=portfolio_df,
                    bar_returns_dict=_bar_returns,
                    total_capital=s["capital"],
                )
            except Exception as exc:
                self._q.put(("status", f"Monte Carlo skipped: {exc}"))

            self._q.put(("status","Running portfolio backtest..."))
            try:
                engine.run_portfolio_backtest(
                    portfolio_df=portfolio_df,
                    bar_returns_dict=_bar_returns,
                )
            except Exception as exc:
                self._q.put(("status", f"Portfolio backtest skipped: {exc}"))

            self._q.put(("status","Done!"))
            self._q.put(("result", {
                "portfolio":    portfolio_df,
                "results_df":   results_df,
                "regime":       regime_label,
                "vector":       regime_vector,
                "mode":         s["mode"],
                "capital":      s["capital"],
            }))

        except FileNotFoundError as exc:
            self._q.put(("error",
                f"File not found:\n{exc}\n\nCheck Archive Path and Funda CSV."))
        except Exception as exc:
            self._q.put(("error",
                f"Engine error:\n{exc}\n\n{traceback.format_exc()}"))


    # =========================================================================
    #  QUEUE POLLING
    # =========================================================================

    def _poll(self):
        # Drain log queue (engine print() output)
        try:
            while True:
                line = self._log_queue.get_nowait()
                if line.strip():
                    tag = ("err"  if "[ERROR]" in line or "Error" in line else
                           "warn" if "[FLAG]"  in line or "WARN"  in line else
                           "ok"   if "Done"    in line or "OK"    in line else
                           "info")
                    self._log.append(line.rstrip(), tag)
        except queue.Empty:
            pass

        # Drain result queue
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "status":
                    self._set_status(payload)
                elif kind == "result":
                    self._on_result(payload)
                elif kind == "error":
                    self._on_error(payload)
                elif kind == "warn":
                    self._on_warn(payload)
        except queue.Empty:
            pass

        self.after(100, self._poll)

    def _set_status(self, txt):
        self._status_var.set(txt)
        self._status_var2.set(txt)

    # -------------------------------------------------------------------------
    #  Result handlers
    # -------------------------------------------------------------------------

    def _on_result(self, p):
        self._prog.stop()
        self._btn_run.config(state="normal", text="RUN NERO")
        self._btn_export.config(state="normal")

        portfolio   = p["portfolio"]
        regime      = p["regime"]
        vector      = p["vector"]
        mode        = p["mode"]
        capital     = p["capital"]

        self._portfolio  = portfolio
        self._results_df = p["results_df"]
        self._regime     = regime
        self._vector     = vector

        # Regime displays
        rc = _regime_color(regime)
        self._regime_badge.config(text=f"  {regime}  ", fg=rc)
        self._regime_big.config(text=regime, fg=rc)

        regime_descs = {
            "Bull Trend":      "Strong uptrend: Fundamentals + Momentum weighted higher.",
            "Bear Trend":      "Downtrend detected: Defensive posture, lower position sizing.",
            "High Volatility": "Elevated volatility: Mean-reversion signals active.",
            "Recovery":        "Market recovering: Quality and value factors lead.",
            "Sideways":        "Range-bound: Cross-sectional momentum and mean reversion.",
        }
        desc = next((v for k,v in regime_descs.items()
                     if k.lower() in regime.lower()), "Regime detected.")
        self._regime_desc.config(text=desc)

        # Gauges
        for key, (gauge, v) in self._gauges.items():
            val = vector.get(key, 0)
            gauge.set_value(val)
            v.set(f"{float(val):+.4f}" if isinstance(val, float) else str(val))

        # Factor weight bars (from regime weights)
        try:
            rw = engine.get_regime_weights(regime)
            bar_map = {
                "Fundamental": rw.get("fund",0.4),
                "Technical":   rw.get("tech",0.4),
                "Volatility":  rw.get("vol", 0.1),
                "Sentiment":   rw.get("sentiment",0.1),
            }
            for factor, (bar_fill, val_lbl) in self._weight_bars.items():
                w = bar_map.get(factor, 0)
                bar_fill.place_configure(relwidth=float(w))
                val_lbl.config(text=f"{w*100:.0f}%")
        except Exception:
            pass

        # Info panel
        self._info_vars["Mode"].set(mode.upper())
        self._info_vars["Regime"].set(regime)
        self._info_vars["Positions"].set(str(len(portfolio)))
        self._info_vars["Capital"].set(f"Rs{capital:,.0f}")

        # Summary cards
        if not portfolio.empty:
            avg_score = portfolio.get("CombinedScore", portfolio.get("score",0)).mean()
            avg_cvar  = portfolio.get("CVaR95", 0).mean() if "CVaR95" in portfolio.columns else 0
            self._summary_cards["Total Capital"].set(f"Rs{capital:,.0f}")
            self._summary_cards["Positions"].set(str(len(portfolio)))
            self._summary_cards["Avg Score"].set(f"{avg_score:.1f}")
            self._summary_cards["Avg CVaR"].set(f"{avg_cvar:.3f}")

        # Populate tree
        self._populate_tree(portfolio)

        # Flags
        flags_path = os.path.join(
            os.environ.get("NERO_OUTPUT_PATH","C:/NERO/results"), "flags.csv")
        if os.path.isfile(flags_path):
            try:
                import pandas as pd
                n = len(pd.read_csv(flags_path))
                self._flags_var.set(f"* {n} flag(s) -- check results/flags.csv")
            except Exception:
                pass

        if self._v_tg_token.get().strip():
            self._btn_tg.config(state="normal")

        n = len(portfolio)
        self._set_status(f"Done! {n} position{'s' if n!=1 else ''} built.")
        self._rowcount_var.set(f"{n} position{'s' if n!=1 else ''}")
        self._log.append(f"Portfolio ready: {n} positions | Regime: {regime}", "ok")

        # Populate Analysis tab with MVO / Monte Carlo / Backtest results
        try:
            self._refresh_analysis_tab()
            self._log.append("Analysis tab updated (MVO · Monte Carlo · Backtest · Walk-Forward).", "ok")
        except Exception as _ae:
            self._log.append(f"Analysis tab refresh error (non-fatal): {_ae}", "warn")

        self._nb.select(0)

    def _on_error(self, msg):
        self._prog.stop()
        self._btn_run.config(state="normal", text="RUN NERO")
        self._set_status("Error -- see log.")
        self._log.append(msg, "err")
        self._nb.select(3)
        messagebox.showerror("NERO Engine Error", msg[:600])

    def _on_warn(self, msg):
        self._prog.stop()
        self._btn_run.config(state="normal", text="RUN NERO")
        self._set_status("Warning -- see log.")
        self._log.append(msg, "warn")
        messagebox.showwarning("NERO Warning", msg)

    # -------------------------------------------------------------------------
    #  Tree population
    # -------------------------------------------------------------------------

    def _populate_tree(self, df):
        for iid in self._tree.get_children(""):
            self._tree.delete(iid)

        if df is None or df.empty:
            self._rowcount_var.set("No results")
            return

        def _g(row, col, default="--"):
            v = row.get(col, default)
            return default if v is None else v

        for rank, (_, row) in enumerate(df.iterrows(), 1):
            bucket  = str(_g(row, "Bucket", "Balanced"))
            missing = bool(row.get("FundaMissing", False))
            sym     = str(_g(row,"Symbol","?")) + (" *" if missing else "")

            tags = [bucket]
            if missing:
                tags.append("Flag")

            self._tree.insert("", "end", values=(
                rank,
                sym,
                _fmt_f(_g(row,"CombinedScore",0), 1),
                _fmt_f(_g(row,"FundScore",    0), 1),
                _fmt_f(_g(row,"TechScore",    0), 1),
                _fmt_f(_g(row,"VolScore",     0), 1),
                bucket,
                _fmt_pct(_g(row,"Weight",     0)),
                _fmt_capital(_g(row,"CapitalAllocated",0)),
                str(_g(row,"Strategy","--"))[:18],
                _fmt_f(_g(row,"EV",           0), 4),
                _fmt_f(_g(row,"CVaR95",       0), 3),
            ), tags=tuple(tags))


    # =========================================================================
    #  EXPORT
    # =========================================================================

    def _on_export(self):
        if self._portfolio is None or self._portfolio.empty:
            messagebox.showinfo("Export","No portfolio to export.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("All","*.*")],
            initialfile="nero_portfolio.csv",
            title="Save Portfolio CSV")
        if not path:
            return
        try:
            self._portfolio.to_csv(path, index=False)
            self._log.append(f"Portfolio exported to {path}", "ok")
            messagebox.showinfo("Exported", f"Saved to:\n{path}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))


    # =========================================================================
    #  TELEGRAM
    # =========================================================================

    def _on_send_telegram(self):
        if not ENGINE_OK:
            messagebox.showerror("Error","Engine not loaded.")
            return
        token = self._v_tg_token.get().strip()
        chat  = self._v_tg_chat.get().strip()
        if not token or not chat:
            messagebox.showwarning("Telegram","Enter Bot Token and Chat ID first.")
            return
        if self._portfolio is None or self._portfolio.empty:
            messagebox.showinfo("Telegram","No portfolio to send.")
            return
        try:
            mode = self._info_vars.get("Mode", tk.StringVar()).get() or "swing"
            engine.send_portfolio_alert(
                self._portfolio, self._regime, mode,
                bot_token=token, chat_id=chat)
            self._log.append("Telegram alert sent successfully.", "ok")
            messagebox.showinfo("Telegram","Alert sent!")
        except Exception as exc:
            messagebox.showerror("Telegram Error", str(exc))


    # =========================================================================
    #  STARTUP CHECKS
    # =========================================================================

    def _startup_checks(self):
        warns = []
        if not ENGINE_OK:
            warns.append(
                f"nero_v2.py not found:\n  {_ENGINE_IMPORT_ERR}\n\n"
                "Place nero_v2.py in the same folder as nero_ui.py.")

        archive = self._v_archive.get()
        if not os.path.isdir(archive):
            warns.append(
                f"Archive directory missing:\n  {archive}\n\n"
                "Update Archive Dir in Configuration panel.")

        funda = self._v_funda.get()
        if not os.path.isfile(funda):
            warns.append(
                f"Fundamental CSV not found:\n  {funda}\n\n"
                "Update Funda CSV path in Configuration panel.")

        for w in warns:
            self._log.append(w.replace("\n"," "), "warn")

        if warns:
            messagebox.showwarning("NERO Setup Check",
                "\n\n------------------\n\n".join(warns))


# =============================================================================
#  STDOUT REDIRECT to Queue
# =============================================================================

class _QueueWriter:
    def __init__(self, q, fallback):
        self._q = q
        self._fallback = fallback
        self.encoding = "utf-8"

    def write(self, text):
        if text:
            self._q.put(text)
        try:
            self._fallback.write(text)
        except Exception:
            pass

    def flush(self):
        try:
            self._fallback.flush()
        except Exception:
            pass


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    app = NeroApp()
    app.mainloop()

if __name__ == "__main__":
    main()
