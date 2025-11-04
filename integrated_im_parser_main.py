
import sys, json
import numpy as np
from pathlib import Path
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QScrollArea, QTabWidget, QMessageBox, QPushButton
)

# Import your baseline module (must be in the same folder)
import im_desktop_app_pro_sparam_polar2 as base

class MainWindowSidebar(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IM & S‑parameters Viewer — Sidebar + Scroll")
        self.sources = []  # List[base.DataSource]

        # Left: Series + Controls
        self.series_panel = base.SeriesPanel()
        self.controls = base.ControlsPanel()
        self.controls.on_changed.connect(self._on_controls_changed)

        # Tabs
        self.tabs = QTabWidget(self)

        # --- IM Tab (uses base.PlotGrid) ---
        self.plot_grid = base.PlotGrid()
        self.im_tab = QWidget()
        im_v = QVBoxLayout(self.im_tab)

        top_row = QHBoxLayout()
        self.btn_load_im = QPushButton("Open .im…")
        self.btn_add_im  = QPushButton("Add files…")
        self.btn_clear   = QPushButton("Clear")
        self.btn_save_im = QPushButton("Save IM Figure")
        self.btn_reset_im = QPushButton("Reset View")
        for b in (self.btn_load_im, self.btn_add_im, self.btn_clear, self.btn_save_im, self.btn_reset_im):
            top_row.addWidget(b)
        top_row.addStretch(1)
        im_v.addLayout(top_row)
        im_v.addWidget(self.plot_grid, 1)

        # --- S‑parameters Tab (reuse baseline tab; it manages its own sources/UI) ---
        self.sparam_tab = base.SParamTab(self.controls)

        self.tabs.addTab(self.im_tab, "IM (1‑tone)")
        self.tabs.addTab(self.sparam_tab, "S‑parameters")

        # --- Left sidebar ---
        left_sidebar = QWidget()
        left_v = QVBoxLayout(left_sidebar)
        left_v.setContentsMargins(8, 8, 8, 8)
        left_v.addWidget(self.series_panel)
        left_v.addWidget(self.controls)
        left_v.addStretch(1)

        # --- Scrollable charts area ---
        scroll = QScrollArea(self); scroll.setWidgetResizable(True)
        tabs_container = QWidget(); tabs_layout = QVBoxLayout(tabs_container)
        tabs_layout.setContentsMargins(0,0,0,0); tabs_layout.addWidget(self.tabs)
        self.tabs.setMinimumSize(1400, 900)
        scroll.setWidget(tabs_container)

        splitter = QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        splitter.addWidget(left_sidebar); splitter.addWidget(scroll)
        splitter.setStretchFactor(0,0); splitter.setStretchFactor(1,1)
        left_sidebar.setMinimumWidth(300)
        self.setCentralWidget(splitter)

        # Wire actions
        self.btn_load_im.clicked.connect(self.open_files)
        self.btn_add_im.clicked.connect(self.add_files)
        self.btn_clear.clicked.connect(self.clear_all)
        self.btn_save_im.clicked.connect(lambda: self.plot_grid.save_figure(self))
        self.btn_reset_im.clicked.connect(self.plot_grid.reset_view)

        # Tab sync
        self.controls.set_mode('im')
        self.tabs.currentChanged.connect(self._on_tab_changed)

    # ---------- IM file IO (replicates baseline logic) ----------
    def open_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Open .im file(s)", "", "IM files (*.im);;All files (*)")
        if not paths: return
        try:
            self.sources = []
            for i, p in enumerate(paths):
                self._ingest_file(Path(p), i)
            self._refresh_after_ingest(status_prefix="Loaded")
        except Exception as e:
            base.log_ex(e); QtWidgets.QMessageBox.critical(self, "Failed to load", f"{e}\n\nSee log: {getattr(base, 'LOGFILE', 'im_app_error.log')}")

    def add_files(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add .im file(s)", "", "IM files (*.im);;All files (*)")
        if not paths: return
        try:
            start = len(self.sources)
            for k, p in enumerate(paths):
                self._ingest_file(Path(p), start + k)
            self._refresh_after_ingest(status_prefix="Added")
        except Exception as e:
            base.log_ex(e); QtWidgets.QMessageBox.critical(self, "Failed to add files", f"{e}\n\nSee log: {getattr(base, 'LOGFILE', 'im_app_error.log')}")

    def clear_all(self):
        self.sources = []
        self.series_panel.set_series([])
        self.plot_grid.clear_axes(); self.plot_grid.draw()
        self.statusBar().showMessage("Cleared")

    def _ingest_file(self, path: Path, index: int):
        parser = base.pick_parser(path)
        records = parser.parse(path)
        self.sources.append(base.DataSource(path, parser.name(), records, index))

    def _refresh_after_ingest(self, status_prefix="Loaded"):
        # Mirror baseline's left panel update
        self.series_panel.set_series(self.sources)
        self.statusBar().showMessage(f"{status_prefix} {len(self.sources)} file(s)")
        # Apply current controls into plot grid (legend/grid/spacing/toggles/scales)
        try:
            self.plot_grid.set_show_legends(self.controls.show_legends())
            self.plot_grid.set_hpad(self.controls.hpad())
            self.plot_grid.set_wpad(self.controls.wpad())
            self.plot_grid.set_show_grid(self.controls.show_grid())
            self.plot_grid.legend_mode = self.controls.legend_mode()
            self.plot_grid.legend_width = self.controls.legend_width()
            self.plot_grid.legend_fs = self.controls.legend_fontsize()
            self.plot_grid.legend_ncol = self.controls.legend_ncol()
            self.plot_grid.compact_labels = self.controls.compact_labels()
            self.plot_grid.set_axis_toggle(*self.controls.axis_toggles())
            self.plot_grid.set_axis_scales(*self.controls.axis_scales())
        except Exception:
            pass
        self.update_plots()

    # ---------- Plotting (replicates baseline) ----------
    def _build_mapping_for_plots(self, selected):
        mapping = {}; labels = []
        for si, lab in selected:
            if si < 0 or si >= len(self.sources): 
                continue
            src = self.sources[si]
            recs = src.freq_to_records.get(lab, [])
            if not recs: 
                continue
            label = f"{src.file_label} • {lab}"
            mapping[label] = (src.index, recs)
            labels.append(label)
        return mapping, labels

    def update_plots(self):
        if not self.sources:
            return
        selected = self.series_panel.selected_series()
        if not selected:
            self.plot_grid.clear_axes(); self.plot_grid.draw()
            self.statusBar().showMessage("No series selected")
            return
        curve_pref = self.controls.curve_pref()
        ignore_a2 = self.controls.ignore_a2()
        mapping, labels = self._build_mapping_for_plots(selected)

        # Check if ACPR present in any selected record (copy baseline logic)
        show_acpr_any = False
        for label in labels:
            _, recs = mapping.get(label, (0, []))
            rec = base.PlotGrid._choose_record(recs, curve_pref)
            if rec is None:
                continue
            df = base.compute_metrics(rec, use_gamma_source=False, ignore_a2=ignore_a2)
            sL = df.get("ACPR Lower [dBc] @ f0"); sU = df.get("ACPR Upper [dBc] @ f0")
            if sL is not None and np.isfinite(sL.values).any(): 
                show_acpr_any = True; break
            if sU is not None and np.isfinite(sU.values).any(): 
                show_acpr_any = True; break

        try:
            self.plot_grid.overlay_frequency_curves(mapping, labels, curve_pref, ignore_a2, show_acpr_any)
            msg_tail = "Pout=|B2|^2" if ignore_a2 else "Pout=|B2|^2−|A2|^2"
            self.statusBar().showMessage(f"Plotted {len(labels)} series • Pref: {curve_pref.upper()} • {msg_tail}")
        except Exception as e:
            base.log_ex(e)
            QtWidgets.QMessageBox.critical(self, "Plot error", f"{e}")

    # ---------- Controls & tabs ----------
    def _on_tab_changed(self, idx: int):
        w = self.tabs.widget(idx)
        if w is self.sparam_tab:
            self.controls.set_mode('sparam')
        else:
            self.controls.set_mode('im')
        self.update_plots()

    def _on_controls_changed(self):
        # Theme
        try:
            self.plot_grid.apply_theme(
                dark=self.controls.dark_theme(),
                show_grid=self.controls.show_grid(),
                lw=self.controls.linewidth(),
                alpha=self.controls.alpha()
            )
        except Exception:
            pass
        # Plot config
        try:
            self.plot_grid.set_show_legends(self.controls.show_legends())
            self.plot_grid.set_hpad(self.controls.hpad())
            self.plot_grid.set_wpad(self.controls.wpad())
            self.plot_grid.set_show_grid(self.controls.show_grid())
            self.plot_grid.legend_mode = self.controls.legend_mode()
            self.plot_grid.legend_width = self.controls.legend_width()
            self.plot_grid.legend_fs = self.controls.legend_fontsize()
            self.plot_grid.legend_ncol = self.controls.legend_ncol()
            self.plot_grid.compact_labels = self.controls.compact_labels()
            self.plot_grid.set_axis_toggle(*self.controls.axis_toggles())
            self.plot_grid.set_axis_scales(*self.controls.axis_scales())
        except Exception:
            pass
        self.update_plots()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindowSidebar()
    w.resize(1400, 900)
    w.show()
    sys.exit(app.exec())
