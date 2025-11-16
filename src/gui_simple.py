# src/gui_simple.py
# Simple, professional-looking GUI (Tkinter + Matplotlib) for your OR simulation.
# - Minimal controls, big KPIs, clean plots
# - No Streamlit / no pyarrow
# - Uses your existing run_baseline.py (SimConfig, run_once)

import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from run_baseline import SimConfig, run_once


# ---------------------- Styling helpers ----------------------
PRIMARY_BG = "#ffffff"
ACCENT = "#0F62FE"        # IBM blue-like
TEXT_DARK = "#1f2937"
TEXT_MUTED = "#6b7280"
CARD_BG = "#f8fafc"
WARN = "#d97706"

TITLE_FONT = ("Segoe UI", 16, "bold")
SUBTITLE_FONT = ("Segoe UI", 10)
LABEL_FONT = ("Segoe UI", 10)
KPI_LABEL_FONT = ("Segoe UI", 10)
KPI_VALUE_FONT = ("Segoe UI", 13, "bold")
BTN_FONT = ("Segoe UI", 10, "bold")

def apply_style(root: tk.Tk):
    style = ttk.Style(root)
    # Use a clean built-in theme, then override key elements
    try:
        style.theme_use("clam")
    except:
        pass

    style.configure("TFrame", background=PRIMARY_BG)
    style.configure("TLabel", background=PRIMARY_BG, foreground=TEXT_DARK, font=LABEL_FONT)
    style.configure("Card.TFrame", background=CARD_BG, relief="flat")
    style.configure("Accent.TButton", font=BTN_FONT, padding=6)
    style.map("Accent.TButton",
              foreground=[("active", "white")],
              background=[("!disabled", ACCENT), ("active", ACCENT)],
              relief=[("pressed", "sunken"), ("!pressed", "flat")])

# ---------------------- Main App ----------------------
class SimpleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Surgical Unit Simulation (Professional GUI)")
        self.geometry("1100x700")
        self.minsize(980, 640)
        apply_style(self)

        self._build_layout()
        self._set_defaults()

        self.df_pat = pd.DataFrame()
        self.df_mon = pd.DataFrame()

    # ---------- Layout ----------
    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Banner
        banner = ttk.Frame(self)
        banner.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 4))
        banner.columnconfigure(0, weight=1)

        title = ttk.Label(banner, text="Surgical Unit Discrete-Event Simulation",
                          font=TITLE_FONT, foreground=TEXT_DARK)
        subtitle = ttk.Label(
            banner,
            text="P prep rooms → 1 OR → R recovery rooms • No intermediate buffers • Per-patient service times • Optional light-surgery mix",
            font=SUBTITLE_FONT, foreground=TEXT_MUTED
        )
        title.grid(row=0, column=0, sticky="w")
        subtitle.grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Main content: left controls / right results
        main = ttk.Frame(self)
        main.grid(row=1, column=0, sticky="nsew", padx=14, pady=10)
        main.columnconfigure(0, weight=0)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)

        # Left: Controls
        self.controls = ttk.Frame(main)
        self.controls.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        for i in range(24):
            self.controls.rowconfigure(i, weight=0)

        self._build_controls(self.controls)

        # Right: KPIs + Plots
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_kpis(right)
        self._build_plots(right)

    def _build_controls(self, parent):
        r = 0
        section = ttk.Label(parent, text="Parameters", font=("Segoe UI", 11, "bold"))
        section.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 6)); r += 1

        # Time / Monitoring
        self.var_hours = tk.DoubleVar()
        self._labeled_spin(parent, "Simulation length (hours)", self.var_hours, 1, 24, 1.0, r); r += 1

        self.var_dt = tk.DoubleVar()
        self._labeled_spin(parent, "Monitoring Δt (minutes)", self.var_dt, 0.5, 10, 0.5, r); r += 1

        self._sep(parent, r); r += 1

        # Resources
        self.var_prep = tk.IntVar()
        self._labeled_spin(parent, "P (prep rooms)", self.var_prep, 1, 20, 1, r, int_mode=True); r += 1

        ttk.Label(parent, text="OR capacity").grid(row=r, column=0, sticky="w", pady=3)
        ttk.Label(parent, text="1 (fixed by assignment)", foreground=TEXT_MUTED).grid(row=r, column=1, sticky="e"); r += 1

        self.var_rec = tk.IntVar()
        self._labeled_spin(parent, "R (recovery rooms)", self.var_rec, 1, 20, 1, r, int_mode=True); r += 1

        self._sep(parent, r); r += 1

        # Arrivals & Services (baseline)
        self.var_iat = tk.DoubleVar()
        self._labeled_spin(parent, "Mean interarrival (min)", self.var_iat, 1, 120, 1, r); r += 1

        self.var_prep_mean = tk.DoubleVar()
        self._labeled_spin(parent, "Mean PREP (min)", self.var_prep_mean, 1, 240, 1, r); r += 1

        self.var_or_mean = tk.DoubleVar()
        self._labeled_spin(parent, "Mean OR (min)", self.var_or_mean, 1, 240, 1, r); r += 1

        self.var_rec_mean = tk.DoubleVar()
        self._labeled_spin(parent, "Mean REC (min)", self.var_rec_mean, 1, 240, 1, r); r += 1

        self._sep(parent, r); r += 1

        # Light mix (compact)
        section2 = ttk.Label(parent, text="Light-surgery mix", font=("Segoe UI", 10, "bold"))
        section2.grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 6)); r += 1

        self.var_frac_light = tk.DoubleVar()
        self._labeled_spin(parent, "Fraction light (0–1)", self.var_frac_light, 0, 1, 0.05, r); r += 1

        self.var_prep_light = tk.DoubleVar()
        self._labeled_spin(parent, "Light PREP mean", self.var_prep_light, 1, 240, 1, r); r += 1

        self.var_or_light = tk.DoubleVar()
        self._labeled_spin(parent, "Light OR mean", self.var_or_light, 1, 240, 1, r); r += 1

        self.var_rec_light = tk.DoubleVar()
        self._labeled_spin(parent, "Light REC mean", self.var_rec_light, 1, 240, 1, r); r += 1

        self._sep(parent, r); r += 1

        # Seed + label
        self.var_seed = tk.IntVar()
        self._labeled_spin(parent, "Random seed", self.var_seed, 0, 10000, 1, r, int_mode=True); r += 1

        self.var_label = tk.StringVar()
        ttk.Label(parent, text="Run label").grid(row=r, column=0, sticky="w", pady=3)
        ttk.Entry(parent, textvariable=self.var_label, width=14).grid(row=r, column=1, sticky="e"); r += 1

        # Save CSV
        self.var_save_csv = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Save CSVs to results/", variable=self.var_save_csv).grid(row=r, column=0, columnspan=2, sticky="w", pady=4); r += 1

        # Run button
        self.btn_run = ttk.Button(parent, text="▶  Run Simulation", style="Accent.TButton", command=self._on_run)
        self.btn_run.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(6, 2)); r += 1

        # Status
        self.lbl_status = ttk.Label(parent, text="Ready.", foreground=TEXT_MUTED)
        self.lbl_status.grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 0))

    def _labeled_spin(self, parent, text, var, frm, to, step, row, int_mode=False):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", pady=3)
        if int_mode:
            sb = ttk.Spinbox(parent, from_=int(frm), to=int(to), increment=int(step), textvariable=var, width=10)
        else:
            sb = ttk.Spinbox(parent, from_=frm, to=to, increment=step, textvariable=var, width=10)
        sb.grid(row=row, column=1, sticky="e")

    def _sep(self, parent, row):
        ttk.Separator(parent).grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)

    def _build_kpis(self, parent):
        kpi_frame = ttk.Frame(parent)
        kpi_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for i in range(8):
            kpi_frame.columnconfigure(i, weight=1)

        # Two rows of KPI "cards"
        self.kpi_values = {
            "throughput": tk.StringVar(value="—"),
            "or_util": tk.StringVar(value="—"),
            "prep_q": tk.StringVar(value="—"),
            "ttis": tk.StringVar(value="—"),
            "block1": tk.StringVar(value="—"),
            "block2": tk.StringVar(value="—"),
            "completed": tk.StringVar(value="—"),
        }

        def card(col, title, key, row=0):
            card = ttk.Frame(kpi_frame, style="Card.TFrame")
            card.grid(row=row, column=col, sticky="ew", padx=4, pady=2)
            ttk.Label(card, text=title, font=KPI_LABEL_FONT, foreground=TEXT_MUTED, background=CARD_BG).grid(row=0, column=0, sticky="w", padx=10, pady=(8, 0))
            ttk.Label(card, textvariable=self.kpi_values[key], font=KPI_VALUE_FONT, background=CARD_BG).grid(row=1, column=0, sticky="w", padx=10, pady=(0, 8))

        card(0, "Throughput / hour", "throughput", row=0)
        card(1, "Avg OR utilisation", "or_util", row=0)
        card(2, "Avg entrance queue", "prep_q", row=0)
        card(3, "Mean total time (min)", "ttis", row=0)

        card(0, "Mean block PREP→OR (min)", "block1", row=1)
        card(1, "Mean block OR→REC (min)", "block2", row=1)
        card(2, "Completed patients", "completed", row=1)

    def _build_plots(self, parent):
        plots = ttk.Frame(parent)
        plots.grid(row=1, column=0, sticky="nsew")
        plots.columnconfigure(0, weight=1)
        plots.columnconfigure(1, weight=1)
        plots.rowconfigure(0, weight=1)

        # Plot 1
        self.fig1 = Figure(figsize=(5, 3), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_title("Entrance queue length over time")
        self.ax1.set_xlabel("Time (min)")
        self.ax1.set_ylabel("Prep queue length")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plots)
        self.canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        # Plot 2
        self.fig2 = Figure(figsize=(5, 3), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_title("OR utilisation over time")
        self.ax2.set_xlabel("Time (min)")
        self.ax2.set_ylabel("Busy fraction")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plots)
        self.canvas2.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    # ---------- Defaults ----------
    def _set_defaults(self):
        self.var_hours.set(8.0)
        self.var_dt.set(1.0)
        self.var_prep.set(3)
        self.var_rec.set(3)
        self.var_iat.set(25.0)
        self.var_prep_mean.set(40.0)
        self.var_or_mean.set(20.0)
        self.var_rec_mean.set(40.0)
        self.var_frac_light.set(0.40)
        self.var_prep_light.set(30.0)
        self.var_or_light.set(12.0)
        self.var_rec_light.set(30.0)
        self.var_seed.set(42)
        self.var_label.set("demo")

    # ---------- Actions ----------
    def _on_run(self):
        self.btn_run.state(["disabled"])
        self.lbl_status.configure(text="Running…", foreground=ACCENT)
        threading.Thread(target=self._run_worker, daemon=True).start()

    def _run_worker(self):
        try:
            cfg = SimConfig(
                sim_time=self.var_hours.get() * 60.0,
                monitor_dt=self.var_dt.get(),
                p_prep=int(self.var_prep.get()),
                p_or=1,
                p_rec=int(self.var_rec.get()),
                mean_iat=self.var_iat.get(),
                mean_prep=self.var_prep_mean.get(),
                mean_or=self.var_or_mean.get(),
                mean_rec=self.var_rec_mean.get(),
                frac_light=self.var_frac_light.get(),
                mean_prep_light=self.var_prep_light.get(),
                mean_or_light=self.var_or_light.get(),
                mean_rec_light=self.var_rec_light.get(),
                rng_seed=int(self.var_seed.get()),
                results_dir="results",
                run_label=self.var_label.get(),
            )

            sys = run_once(cfg)  # handles CSV saving inside
            df_pat = pd.DataFrame(sys.records)
            df_mon = pd.DataFrame({
                "t": sys.t_samples,
                "prep_queue_len": sys.prep_q_samples,
                "or_utilisation": sys.or_util_samples
            })
            self.df_pat, self.df_mon = df_pat, df_mon

            self.after(0, self._update_outputs, cfg, df_pat, df_mon)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, lambda: (self.btn_run.state(["!disabled"]),
                                   self.lbl_status.configure(text="Ready.", foreground=TEXT_MUTED)))

    def _update_outputs(self, cfg: SimConfig, df_pat: pd.DataFrame, df_mon: pd.DataFrame):
        # KPIs
        sim_hours = cfg.sim_time / 60.0
        completed = len(df_pat)
        throughput = completed / sim_hours if sim_hours > 0 else float("nan")
        avg_util = float(df_mon["or_utilisation"].mean()) if not df_mon.empty else float("nan")
        avg_q = float(df_mon["prep_queue_len"].mean()) if not df_mon.empty else float("nan")
        mean_ttis = float(df_pat["total_time_in_system"].mean()) if "total_time_in_system" in df_pat else float("nan")
        mean_b1 = float(df_pat["block_prep_to_or"].mean()) if "block_prep_to_or" in df_pat else float("nan")
        mean_b2 = float(df_pat["block_or_to_rec"].mean()) if "block_or_to_rec" in df_pat else float("nan")

        self.kpi_values["throughput"].set(f"{throughput:.2f}")
        self.kpi_values["or_util"].set(f"{avg_util:.2f}")
        self.kpi_values["prep_q"].set(f"{avg_q:.2f}")
        self.kpi_values["ttis"].set(f"{mean_ttis:.1f}")
        self.kpi_values["block1"].set(f"{mean_b1:.2f}")
        self.kpi_values["block2"].set(f"{mean_b2:.2f}")
        self.kpi_values["completed"].set(str(completed))

        # Plots
        # Queue
        self.ax1.clear()
        self.ax1.set_title("Entrance queue length over time")
        self.ax1.set_xlabel("Time (min)")
        self.ax1.set_ylabel("Prep queue length")
        if not df_mon.empty:
            self.ax1.plot(df_mon["t"], df_mon["prep_queue_len"])
        self.canvas1.draw()

        # OR util
        self.ax2.clear()
        self.ax2.set_title("OR utilisation over time")
        self.ax2.set_xlabel("Time (min)")
        self.ax2.set_ylabel("Busy fraction")
        if not df_mon.empty:
            self.ax2.plot(df_mon["t"], df_mon["or_utilisation"])
        self.canvas2.draw()


if __name__ == "__main__":
    app = SimpleGUI()
    app.mainloop()
