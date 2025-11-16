# src/gui_tk.py
# Desktop GUI for your OR simulation using Tkinter + Matplotlib (no Streamlit, no pyarrow).
# - Non-blocking (runs simulation in a background thread)
# - Parameter inputs for all key settings (incl. light-surgery mix)
# - KPI panel + two time-series charts embedded in the window
# - Uses your existing run_baseline.py (SimConfig, run_once)

import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")  # Tk backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from run_baseline import SimConfig, run_once

APP_TITLE = "Surgical Unit Simulation (Tk GUI)"

class SimulationGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x720")
        self.minsize(1000, 640)

        self._build_layout()
        self._set_defaults()

        # placeholders for last results
        self.df_pat = pd.DataFrame()
        self.df_mon = pd.DataFrame()

    def _build_layout(self):
        # Main panes: left = controls; right = outputs (KPIs + plots)
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # --- Left: Controls frame ---
        ctrl = ttk.LabelFrame(self, text="Parameters")
        ctrl.grid(row=0, column=0, sticky="nsw", padx=8, pady=8)
        for i in range(0, 30):
            ctrl.rowconfigure(i, weight=0)
        ctrl.columnconfigure(0, weight=0)
        ctrl.columnconfigure(1, weight=0)

        r = 0
        # Time & monitor
        ttk.Label(ctrl, text="Simulation length (hours)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_hours = tk.DoubleVar()
        ttk.Spinbox(ctrl, from_=1, to=24, increment=1, textvariable=self.var_hours, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Label(ctrl, text="Monitoring Δt (minutes)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_dt = tk.DoubleVar()
        ttk.Spinbox(ctrl, from_=0.5, to=10, increment=0.5, textvariable=self.var_dt, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Separator(ctrl).grid(row=r, column=0, columnspan=2, sticky="ew", padx=4, pady=6); r += 1

        # Resources
        ttk.Label(ctrl, text="P (prep rooms)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_prep = tk.IntVar()
        ttk.Spinbox(ctrl, from_=1, to=20, increment=1, textvariable=self.var_prep, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Label(ctrl, text="R (recovery rooms)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_rec = tk.IntVar()
        ttk.Spinbox(ctrl, from_=1, to=20, increment=1, textvariable=self.var_rec, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Label(ctrl, text="OR capacity").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        ttk.Label(ctrl, text="1 (fixed by assignment)").grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Separator(ctrl).grid(row=r, column=0, columnspan=2, sticky="ew", padx=4, pady=6); r += 1

        # Arrivals & baseline services
        ttk.Label(ctrl, text="Mean interarrival (min)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_iat = tk.DoubleVar()
        ttk.Spinbox(ctrl, from_=1, to=120, increment=1, textvariable=self.var_iat, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Label(ctrl, text="Mean PREP (min)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_prep_mean = tk.DoubleVar()
        ttk.Spinbox(ctrl, from_=1, to=240, increment=1, textvariable=self.var_prep_mean, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Label(ctrl, text="Mean OR (min)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_or_mean = tk.DoubleVar()
        ttk.Spinbox(ctrl, from_=1, to=240, increment=1, textvariable=self.var_or_mean, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Label(ctrl, text="Mean REC (min)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_rec_mean = tk.DoubleVar()
        ttk.Spinbox(ctrl, from_=1, to=240, increment=1, textvariable=self.var_rec_mean, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Separator(ctrl).grid(row=r, column=0, columnspan=2, sticky="ew", padx=4, pady=6); r += 1

        # Light mix
        ttk.Label(ctrl, text="Fraction light (0–1)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_frac_light = tk.DoubleVar()
        ttk.Spinbox(ctrl, from_=0.0, to=1.0, increment=0.05, textvariable=self.var_frac_light, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Label(ctrl, text="Light PREP mean (min)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_prep_light = tk.DoubleVar()
        ttk.Spinbox(ctrl, from_=1, to=240, increment=1, textvariable=self.var_prep_light, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Label(ctrl, text="Light OR mean (min)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_or_light = tk.DoubleVar()
        ttk.Spinbox(ctrl, from_=1, to=240, increment=1, textvariable=self.var_or_light, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Label(ctrl, text="Light REC mean (min)").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_rec_light = tk.DoubleVar()
        ttk.Spinbox(ctrl, from_=1, to=240, increment=1, textvariable=self.var_rec_light, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Separator(ctrl).grid(row=r, column=0, columnspan=2, sticky="ew", padx=4, pady=6); r += 1

        # Seed + label
        ttk.Label(ctrl, text="Random seed").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_seed = tk.IntVar()
        ttk.Spinbox(ctrl, from_=0, to=10000, increment=1, textvariable=self.var_seed, width=8).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        ttk.Label(ctrl, text="Run label").grid(row=r, column=0, sticky="w", padx=6, pady=4)
        self.var_label = tk.StringVar()
        ttk.Entry(ctrl, textvariable=self.var_label, width=14).grid(row=r, column=1, sticky="e", padx=6, pady=4); r += 1

        # Save CSV
        self.var_save_csv = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Save CSVs to results/", variable=self.var_save_csv).grid(row=r, column=0, columnspan=2, sticky="w", padx=6, pady=6); r += 1

        # Run button + status
        self.btn_run = ttk.Button(ctrl, text="▶ Run Simulation", command=self._on_run)
        self.btn_run.grid(row=r, column=0, columnspan=2, sticky="ew", padx=6, pady=6); r += 1

        self.lbl_status = ttk.Label(ctrl, text="Idle.", foreground="gray")
        self.lbl_status.grid(row=r, column=0, columnspan=2, sticky="w", padx=6, pady=4); r += 1

        # --- Right: Output frame ---
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right.rowconfigure(0, weight=0)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # KPIs header
        self.kpi_frame = ttk.LabelFrame(right, text="Key Performance Indicators")
        self.kpi_frame.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        for i in range(8):
            self.kpi_frame.columnconfigure(i, weight=1)

        self.kpi_vars = {
            "throughput": tk.StringVar(value="-"),
            "or_util": tk.StringVar(value="-"),
            "prep_q": tk.StringVar(value="-"),
            "mean_ttis": tk.StringVar(value="-"),
            "block1": tk.StringVar(value="-"),
            "block2": tk.StringVar(value="-"),
            "completed": tk.StringVar(value="-"),
        }

        def add_kpi(col, label, key):
            ttk.Label(self.kpi_frame, text=label).grid(row=0, column=col*2, sticky="w", padx=6)
            ttk.Label(self.kpi_frame, textvariable=self.kpi_vars[key], font=("Segoe UI", 10, "bold")).grid(row=0, column=col*2+1, sticky="w", padx=6)

        add_kpi(0, "Throughput/hr:", "throughput")
        add_kpi(1, "Avg OR util:", "or_util")
        add_kpi(2, "Avg prep queue:", "prep_q")
        add_kpi(3, "Mean total time (min):", "mean_ttis")

        # second line of KPIs
        def add_kpi2(col, label, key):
            ttk.Label(self.kpi_frame, text=label).grid(row=1, column=col*2, sticky="w", padx=6)
            ttk.Label(self.kpi_frame, textvariable=self.kpi_vars[key], font=("Segoe UI", 10, "bold")).grid(row=1, column=col*2+1, sticky="w", padx=6)

        add_kpi2(0, "Mean block PREP→OR (min):", "block1")
        add_kpi2(1, "Mean block OR→REC (min):", "block2")
        add_kpi2(2, "Completed:", "completed")

        # Plot area
        plots = ttk.Frame(right)
        plots.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        plots.rowconfigure(0, weight=1)
        plots.columnconfigure(0, weight=1)
        plots.columnconfigure(1, weight=1)

        # Figure 1: Queue length
        self.fig1 = Figure(figsize=(5, 3), dpi=100)
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_xlabel("Time (min)")
        self.ax1.set_ylabel("Prep queue length")
        self.ax1.set_title("Entrance queue length over time")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=plots)
        self.canvas1.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Figure 2: OR utilisation
        self.fig2 = Figure(figsize=(5, 3), dpi=100)
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_xlabel("Time (min)")
        self.ax2.set_ylabel("OR utilisation (busy fraction)")
        self.ax2.set_title("OR utilisation over time")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=plots)
        self.canvas2.get_tk_widget().grid(row=0, column=1, sticky="nsew", padx=4, pady=4)

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
        self.var_label.set("tkdemo")

    def _on_run(self):
        # Disable button during run
        self.btn_run.config(state=tk.DISABLED)
        self.lbl_status.config(text="Running...", foreground="darkgreen")

        # Spin worker thread
        t = threading.Thread(target=self._run_simulation, daemon=True)
        t.start()

    def _run_simulation(self):
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
                results_dir="../results",
                run_label=self.var_label.get(),
            )
            sys = run_once(cfg)  # saves CSVs internally
            # Prepare DataFrames for plots/KPIs
            df_pat = pd.DataFrame(sys.records)
            df_mon = pd.DataFrame({
                "t": sys.t_samples,
                "prep_queue_len": sys.prep_q_samples,
                "or_utilisation": sys.or_util_samples
            })

            self.df_pat, self.df_mon = df_pat, df_mon
            self.after(0, self._update_outputs, cfg, df_pat, df_mon)

            # Optionally save copies from GUI flag
            if self.var_save_csv.get():
                Path("../results").mkdir(parents=True, exist_ok=True)
                df_pat.to_csv(Path("../results") / f"gui_pat_{cfg.run_label}.csv", index=False)
                df_mon.to_csv(Path("../results") / f"gui_mon_{cfg.run_label}.csv", index=False)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, lambda: (self.btn_run.config(state=tk.NORMAL),
                                   self.lbl_status.config(text="Idle.", foreground="gray")))

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

        self.kpi_vars["throughput"].set(f"{throughput:.2f}")
        self.kpi_vars["or_util"].set(f"{avg_util:.2f}")
        self.kpi_vars["prep_q"].set(f"{avg_q:.2f}")
        self.kpi_vars["mean_ttis"].set(f"{mean_ttis:.1f}")
        self.kpi_vars["block1"].set(f"{mean_b1:.2f}")
        self.kpi_vars["block2"].set(f"{mean_b2:.2f}")
        self.kpi_vars["completed"].set(f"{completed}")

        # Plots
        # 1) Queue length
        self.ax1.clear()
        self.ax1.set_xlabel("Time (min)")
        self.ax1.set_ylabel("Prep queue length")
        self.ax1.set_title("Entrance queue length over time")
        if not df_mon.empty:
            self.ax1.plot(df_mon["t"], df_mon["prep_queue_len"])
        self.canvas1.draw()

        # 2) OR utilisation
        self.ax2.clear()
        self.ax2.set_xlabel("Time (min)")
        self.ax2.set_ylabel("OR utilisation (busy fraction)")
        self.ax2.set_title("OR utilisation over time")
        if not df_mon.empty:
            self.ax2.plot(df_mon["t"], df_mon["or_utilisation"])
        self.canvas2.draw()


if __name__ == "__main__":
    app = SimulationGUI()
    app.mainloop()
