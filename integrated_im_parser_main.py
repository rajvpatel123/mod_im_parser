
from __future__ import annotations

from pathlib import Path
import re
import sys
import traceback
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from PySide6 import QtWidgets, QtCore
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QCheckBox, QGroupBox, QScrollArea, QRadioButton, QButtonGroup,
    QTabWidget, QComboBox, QTableWidget, QTableWidgetItem
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

LOGFILE = Path(__file__).with_suffix(".log")

def log_ex(ex: BaseException):
    try:
        LOGFILE.write_text("".join(traceback.format_exception(ex)), encoding="utf-8")
    except Exception:
        pass

# --------------------------------- Data Model ---------------------------------

class CurveRecord:
    def __init__(self, dataset_name: str, curve_name: str, cols: dict, meta: dict, rows: int,
                 freq_hz: Optional[float], freq_label: str):
        self.dataset_name = dataset_name
        self.curve_name = curve_name
        self.cols = cols
        self.meta = meta
        self.rows = rows
        self.freq_hz = freq_hz
        self.freq_label = freq_label


# --------------------------- Helper Functions ----------------------------------

PAIR_RE = re.compile(r'^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s+([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$')
FREQ_NAME_RE = re.compile(r'(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>GHz|MHz|kHz|Hz)', re.IGNORECASE)

def to_complex_any(val) -> complex:
    if isinstance(val, complex): return val
    if isinstance(val, (tuple, list, np.ndarray)) and len(val) == 2:
        try: return complex(float(val[0]), float(val[1]))
        except Exception: return complex(np.nan, np.nan)
    if isinstance(val, (int, float)) and np.isfinite(val): return complex(val, 0.0)
    return complex(np.nan, np.nan)

def arr_complex(cols, n, cid):
    if cid is None: return np.full(n, complex(np.nan, np.nan), dtype=complex)
    raw = np.asarray(cols[cid], dtype=object)
    out = np.empty(n, dtype=complex)
    for i, v in enumerate(raw): out[i] = to_complex_any(v)
    return out

def arr_float(cols, n, cid):
    if cid is None: return np.full(n, np.nan)
    raw = np.asarray(cols[cid], dtype=object)
    out = np.full(n, np.nan)
    for i, v in enumerate(raw):
        if isinstance(v, (int,float)) and np.isfinite(v): out[i] = float(v)
    return out

def pick_col(meta, prefix):
    pl = prefix.lower()
    for cid, m in meta.items():
        nm = (m.get("name") or cid).lower()
        if nm.startswith(pl):
            return cid
    return None

def human_hz(hz: float):
    if hz is None or not np.isfinite(hz): return 'Freq ?'
    if hz >= 1e9: return f'{hz/1e9:.3g} GHz'
    if hz >= 1e6: return f'{hz/1e6:.3g} MHz'
    if hz >= 1e3: return f'{hz/1e3:.3g} kHz'
    return f'{hz:.3g} Hz'


# -------------------------- Existing XML Parser --------------------------------

class ExistingXMLParser:
    def name(self) -> str:
        return "ExistingXMLParser"

    def can_parse(self, path: Path) -> float:
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(str(path)).getroot()
            if root.findall(".//dataset") and root.findall(".//curve") and root.findall(".//data"):
                return 0.9
        except Exception:
            return 0.0
        return 0.6

    def parse(self, path: Path) -> List[CurveRecord]:
        import xml.etree.ElementTree as ET
        root = ET.parse(str(path)).getroot()
        out: List[CurveRecord] = []
        for ds in root.findall(".//dataset"):
            dname = ds.get("name", "")
            for cv in ds.findall("./curve"):
                cname = cv.get("name", "")
                cols, meta, n = self._parse_curve(cv)
                f_hz, f_label = self._detect_frequency(dname, cname, cols, meta, n)
                out.append(CurveRecord(dname, cname, cols, meta, n, f_hz, f_label))
        return out

    def _parse_curve(self, curve):
        cols, meta, n = {}, {}, 0
        for data in curve.findall(".//data"):
            cid = data.get("id") or data.get("name") or f"col_{len(cols)}"
            name = data.get("name") or cid
            unit = data.get("unit") or ""
            text = (data.text or "").strip() if data.text else ""
            if text.startswith("[") and text.endswith("]"): text = text[1:-1]
            items = [s.strip() for s in text.split(",")] if text else []
            vals = []
            for s in items:
                m = PAIR_RE.match(s)
                if m:
                    try:
                        vals.append((float(m.group(1)), float(m.group(2)))); continue
                    except Exception: pass
                try: vals.append(float(s))
                except Exception: vals.append(np.nan)
            cols[cid] = vals
            meta[cid] = {"name": name, "unit": unit}
            n = max(n, len(vals))
        for k, v in list(cols.items()):
            if len(v) < n: cols[k] = v + [np.nan] * (n - len(v))
        return cols, meta, n

    def _detect_frequency(self, dname: str, cname: str, cols: dict, meta: dict, n: int):
        freq_cid = None
        for key in meta.keys():
            nm = (meta[key].get("name") or key).lower()
            if nm.startswith("f0") or nm.startswith("freq") or nm.startswith("frequency"):
                freq_cid = key; break
        if freq_cid:
            raw = np.asarray(cols[freq_cid], dtype=object)
            nums = [float(v) for v in raw if isinstance(v,(int,float)) and np.isfinite(v)]
            if nums:
                med = float(np.median(nums))
                return med, human_hz(med)
        text = f"{dname} {cname}"
        m = re.search(r'(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>GHz|MHz|kHz|Hz)', text, flags=re.IGNORECASE)
        if m:
            val = float(m.group("val")); unit = m.group("unit").lower()
            mult = {"ghz":1e9,"mhz":1e6,"khz":1e3,"hz":1.0}[unit]
            hz = val*mult
            return hz, f"{val:g} {unit.upper()}"
        return None, "Freq ?"


# -------------------------- New Variant Parser ---------------------------------

class NewVariantParser:
    def name(self) -> str:
        return "NewVariantParser"

    def can_parse(self, path: Path) -> float:
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(str(path)).getroot()
            if root.tag.lower() == "datasource" and root.findall(".//curve[@id='IqPowMod']"):
                return 0.95
        except Exception:
            pass
        try:
            head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:5]
            text = "\n".join(head).lower()
            if ("b2_re" in text or "b2,real" in text) and ("pdc" in text or "vdd" in text):
                return 0.6
            if "<imvariant" in text or "<sweep" in text:
                return 0.5
        except Exception:
            pass
        return 0.0

    def parse(self, path: Path) -> List[CurveRecord]:
        import xml.etree.ElementTree as ET
        root = ET.parse(str(path)).getroot()
        if root.tag.lower() == "datasource":
            records: List[CurveRecord] = []
            for c in root.findall(".//curve[@id='IqPowMod']"):
                dname = c.get("id") or "IqPowMod"
                cname = c.get("name") or (c.get("level") or "")
                cols: Dict[str, list] = {}
                meta: Dict[str, dict] = {}
                nmax = 0
                for d in c.findall(".//data"):
                    did = d.get("id") or d.get("data") or f"col_{len(cols)}"
                    txt = (d.text or "").strip()
                    if txt.startswith("[") and txt.endswith("]"): txt = txt[1:-1]
                    vals = []
                    for token in txt.split(",") if txt else []:
                        token = token.strip()
                        try: vals.append(float(token))
                        except Exception: vals.append(np.nan)
                    cols[did] = vals
                    meta[did] = {"name": d.get("name") or did, "unit": d.get("unit") or ""}
                    nmax = max(nmax, len(vals))
                for k,v in list(cols.items()):
                    if len(v) < nmax: cols[k] = v + [np.nan]*(nmax-len(v))
                # Frequency
                f_hz = None; f_label = "Freq ?"
                m = re.search(r"(\d+(?:\.\d+)?)\s*(GHz|MHz|kHz|Hz)", f"{dname} {cname}", flags=re.IGNORECASE)
                if m:
                    val = float(m.group(1)); unit = m.group(2).lower()
                    mult = {"ghz":1e9, "mhz":1e6, "khz":1e3, "hz":1.0}[unit]
                    f_hz = val*mult; f_label = f"{val:g} {unit.upper()}"
                records.append(CurveRecord(dname, cname, cols, meta, nmax, f_hz, f_label))
            if records:
                return records
        raise RuntimeError("NewVariantParser: could not parse this file; adjust mappings for the new format.")


# --------------------------- Parser Registry -----------------------------------

PARSERS = [ExistingXMLParser(), NewVariantParser()]

def pick_parser(path: Path):
    scores = []
    for p in PARSERS:
        try:
            scores.append((p, p.can_parse(path)))
        except Exception as e:
            scores.append((p, 0.0))
    best = max(scores, key=lambda t: t[1])
    if best[1] <= 0.0:
        raise RuntimeError("No parser recognized this .im file")
    return best[0]


# -------------------------------- Metrics --------------------------------------

def compute_metrics(record: CurveRecord, use_gamma_source: bool = False, ignore_a2: bool = False) -> pd.DataFrame:
    # Try wave-based path first
    A1 = pick_col(record.meta, "A1"); A2 = pick_col(record.meta, "A2")
    B1 = pick_col(record.meta, "B1"); B2 = pick_col(record.meta, "B2")
    Pdc_cid = next((cid for cid, m in record.meta.items() if (m.get("name","").lower().startswith("pdc"))), None)
    Gs  = next((cid for cid, m in record.meta.items() if "gamma source" in (m.get("name","").lower())), None)

    a1 = arr_complex(record.cols, record.rows, A1)
    a2 = arr_complex(record.cols, record.rows, A2)
    b1 = arr_complex(record.cols, record.rows, B1)
    b2 = arr_complex(record.cols, record.rows, B2)
    pdc = arr_float(record.cols, record.rows, Pdc_cid)
    gamma_s = arr_complex(record.cols, record.rows, Gs) if use_gamma_source else np.full(record.rows, complex(0.0,0.0))

    # Detect potential ACPR fields for new variant
    def find_col_by_name(substrs):
        for cid, m in record.meta.items():
            nm = (m.get("name") or cid).lower()
            if all(s in nm for s in substrs):
                return cid
        return None
    acpr_lower_cid = find_col_by_name(["acpr","lower"]) or find_col_by_name(["aclr","lower"])
    acpr_upper_cid = find_col_by_name(["acpr","upper"]) or find_col_by_name(["aclr","upper"])
    acpr_lower = arr_float(record.cols, record.rows, acpr_lower_cid)
    acpr_upper = arr_float(record.cols, record.rows, acpr_upper_cid)

    have_waves = np.isfinite(np.real(b2)).any() or np.isfinite(np.real(a1)).any()

    if have_waves:
        has_a2 = np.isfinite(np.real(a2)).any()
        if ignore_a2 or not has_a2:
            pout_w = np.abs(b2)**2
        else:
            pout_w = np.maximum(np.abs(b2)**2 - np.abs(a2)**2, 0.0)
        pavs_w = np.abs(a1)**2 * (1.0 - np.minimum(np.abs(gamma_s)**2, 0.999999))
        gt_db = 10*np.log10(np.maximum(pout_w / np.where(pavs_w > 1e-18, pavs_w, np.nan), 1e-18))
        phase_rel = np.unwrap(np.angle(b2) - np.angle(a1))
        finite = np.isfinite(phase_rel)
        ref = phase_rel[finite][0] if np.any(finite) else 0.0
        ampm_deg = (phase_rel - ref) * 180/np.pi
        has_b1 = np.isfinite(np.real(b1)).any()
        if has_b1:
            gamma_in = np.full(record.rows, complex(np.nan, np.nan), dtype=complex)
            denom_ok = (np.abs(a1) > 0) & np.isfinite(a1) & np.isfinite(b1)
            np.divide(b1, a1, out=gamma_in, where=denom_ok)
            irl_db = -20*np.log10(np.clip(np.abs(gamma_in), 1e-12, 1.0))
        else:
            irl_db = np.full(record.rows, np.nan)
    else:
        # Fallback path for NewVariant: use Pout/PinAvail (Watts) if provided
        Pout_cid = next((cid for cid, m in record.meta.items() if (m.get("name","Pout").lower().startswith("pout")) or cid.lower()=="pout"), None)
        PinAvail_cid = next((cid for cid, m in record.meta.items() if (m.get("name","PinAvail").lower().startswith("pin avail")) or cid.lower()=="pinavail"), None)
        pout_w = arr_float(record.cols, record.rows, Pout_cid)
        pin_av_w = arr_float(record.cols, record.rows, PinAvail_cid)
        gt_db = 10*np.log10(np.maximum(pout_w / np.where(pin_av_w > 1e-18, pin_av_w, np.nan), 1e-18))
        ampm_deg = np.full(record.rows, np.nan)
        irl_db = np.full(record.rows, np.nan)

    pout_dbm = 10*np.log10(np.maximum(pout_w, 1e-12)/1e-3)
    drain_eff = np.where(pdc > 0, (pout_w / pdc) * 100.0, np.nan)

    df = pd.DataFrame({
        "Pout [dBm] @ f0": pout_dbm,
        "Gt [dB] @ f0": gt_db,
        "AM/PM offset [deg] @ f0": ampm_deg,
        "Drain Efficiency [%] @ f0": drain_eff,
        "Input Return Loss [dB] @ f0": irl_db,
        "ACPR Lower [dBc] @ f0": acpr_lower,
        "ACPR Upper [dBc] @ f0": acpr_upper,
    })
    return df

def record_to_dataframe_with_metrics(record: CurveRecord, use_gamma_source: bool = False, ignore_a2: bool = False) -> pd.DataFrame:
    n = record.rows
    cols_out = {}
    for cid, vals in record.cols.items():
        name = (record.meta.get(cid, {}).get("name") or cid)
        if len(vals) and isinstance(vals[0], (tuple, list, complex)):
            re_arr = np.full(n, np.nan, dtype=float)
            im_arr = np.full(n, np.nan, dtype=float)
            for i, v in enumerate(vals):
                if isinstance(v, complex):
                    re_arr[i] = np.real(v); im_arr[i] = np.imag(v)
                elif isinstance(v, (tuple, list)) and len(v) == 2:
                    try:
                        re_arr[i] = float(v[0]); im_arr[i] = float(v[1])
                    except Exception:
                        pass
                elif isinstance(v, (int,float)) and np.isfinite(v):
                    re_arr[i] = float(v); im_arr[i] = 0.0
            cols_out[f"{name}_re"] = re_arr
            cols_out[f"{name}_im"] = im_arr
        else:
            arr = np.full(n, np.nan, dtype=float)
            for i, v in enumerate(vals):
                if isinstance(v, (int,float)) and np.isfinite(v):
                    arr[i] = float(v)
            cols_out[name] = arr

    m = compute_metrics(record, use_gamma_source=use_gamma_source, ignore_a2=ignore_a2)
    for col in m.columns:
        cols_out[col] = m[col].values

    df = pd.DataFrame(cols_out)
    return df


# ----------------------------------- UI ----------------------------------------

class PlotGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        gs = self.fig.add_gridspec(2, 2)
        self.ax_gain = self.fig.add_subplot(gs[0,0])
        self.ax_ampm_or_acpr = self.fig.add_subplot(gs[0,1])
        self.ax_eff  = self.fig.add_subplot(gs[1,0])
        self.ax_irl  = self.fig.add_subplot(gs[1,1])

    def clear_axes(self):
        for ax in [self.ax_gain, self.ax_ampm_or_acpr, self.ax_eff, self.ax_irl]:
            ax.cla()

    def draw(self):
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def overlay_frequency_curves(self, freq_to_records: Dict[str, List[CurveRecord]], selected_freq_labels: List[str], curve_pref: str, ignore_a2: bool):
        self.clear_axes()
        acpr_present_any = False

        # Detect presence
        for flabel in selected_freq_labels:
            records = freq_to_records.get(flabel, [])
            rec = self._choose_record(records, curve_pref)
            if rec is None: 
                continue
            df = compute_metrics(rec, use_gamma_source=False, ignore_a2=ignore_a2)
            sL = df.get("ACPR Lower [dBc] @ f0")
            sU = df.get("ACPR Upper [dBc] @ f0")
            if sL is not None and np.isfinite(sL.values).any(): acpr_present_any = True; break
            if sU is not None and np.isfinite(sU.values).any(): acpr_present_any = True; break

        for flabel in selected_freq_labels:
            records = freq_to_records.get(flabel, [])
            rec = self._choose_record(records, curve_pref)
            if rec is None:
                continue
            df = compute_metrics(rec, use_gamma_source=False, ignore_a2=ignore_a2)
            x = df['Pout [dBm] @ f0'].values

            # Gain
            y = df['Gt [dB] @ f0'].values
            m = np.isfinite(x) & np.isfinite(y)
            if m.any(): self.ax_gain.plot(x[m], y[m], label=flabel)

            # AM/PM or ACPR
            if acpr_present_any:
                yL = df.get('ACPR Lower [dBc] @ f0')
                yU = df.get('ACPR Upper [dBc] @ f0')
                if yL is not None:
                    mL = np.isfinite(x) & np.isfinite(yL.values)
                    if mL.any(): self.ax_ampm_or_acpr.plot(x[mL], yL.values[mL], label=f'{flabel} (L)')
                if yU is not None:
                    mU = np.isfinite(x) & np.isfinite(yU.values)
                    if mU.any(): self.ax_ampm_or_acpr.plot(x[mU], yU.values[mU], label=f'{flabel} (U)')
            else:
                y = df['AM/PM offset [deg] @ f0'].values
                m = np.isfinite(x) & np.isfinite(y)
                if m.any(): self.ax_ampm_or_acpr.plot(x[m], y[m], label=flabel)

            # Drain Eff
            y = df['Drain Efficiency [%] @ f0'].values
            m = np.isfinite(x) & np.isfinite(y)
            if m.any(): self.ax_eff.plot(x[m], y[m], label=flabel)

            # Input RL
            y = df['Input Return Loss [dB] @ f0'].values
            m = np.isfinite(x) & np.isfinite(y)
            if m.any(): self.ax_irl.plot(x[m], y[m], label=flabel)

        self.ax_gain.set_title('Gain @ f0 vs Pout')
        self.ax_gain.set_xlabel('Pout [dBm] @ f0'); self.ax_gain.set_ylabel('Gt [dB] @ f0'); self.ax_gain.legend()

        if acpr_present_any:
            self.ax_ampm_or_acpr.set_title('ACPR @ f0 vs Pout')
            self.ax_ampm_or_acpr.set_xlabel('Pout [dBm] @ f0'); self.ax_ampm_or_acpr.set_ylabel('ACPR [dBc] @ f0'); self.ax_ampm_or_acpr.legend()
        else:
            self.ax_ampm_or_acpr.set_title('AM/PM offset @ f0 vs Pout')
            self.ax_ampm_or_acpr.set_xlabel('Pout [dBm] @ f0'); self.ax_ampm_or_acpr.set_ylabel('AM/PM offset [deg] @ f0'); self.ax_ampm_or_acpr.legend()

        self.ax_eff.set_title('Drain Efficiency @ f0 vs Pout')
        self.ax_eff.set_xlabel('Pout [dBm] @ f0'); self.ax_eff.set_ylabel('Drain Efficiency [%] @ f0'); self.ax_eff.legend()

        self.ax_irl.set_title('Input Return Loss @ f0 vs Pout')
        self.ax_irl.set_xlabel('Pout [dBm] @ f0'); self.ax_irl.set_ylabel('Input Return Loss [dB] @ f0'); self.ax_irl.legend()

        self.draw()

    @staticmethod
    def _choose_record(records: List[CurveRecord], curve_pref: str) -> Optional[CurveRecord]:
        if not records: return None
        s = curve_pref.lower()
        for r in records:
            name = f"{r.dataset_name} {r.curve_name}".lower()
            if "1-tone" in name and s in name: return r
        for r in records:
            name = f"{r.dataset_name} {r.curve_name}".lower()
            if "1-tone" in name: return r
        return records[0]


class FrequencyPanel(QGroupBox):
    on_changed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__("Frequencies", parent)
        self.box = QWidget()
        self.v = QVBoxLayout(self.box); self.v.setContentsMargins(0,0,0,0)

        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setWidget(self.box)

        self.sel_all = QPushButton("Select All"); self.sel_none = QPushButton("Select None")
        self.sel_all.clicked.connect(self.select_all); self.sel_none.clicked.connect(self.select_none)

        outer = QVBoxLayout(self); outer.addWidget(self.scroll)
        btns = QHBoxLayout(); btns.addWidget(self.sel_all); btns.addWidget(self.sel_none); outer.addLayout(btns)

        self._checks: Dict[str, QCheckBox] = {}

    def set_frequencies(self, labels: List[str]):
        for i in reversed(range(self.v.count())):
            item = self.v.itemAt(i); w = item.widget() if item else None
            if w: w.setParent(None)
        self._checks.clear()
        for lab in labels:
            cb = QCheckBox(lab); cb.setChecked(True)
            cb.stateChanged.connect(lambda _st, lab=lab: self.on_changed.emit())
            self.v.addWidget(cb); self._checks[lab] = cb
        self.v.addStretch(1); self.on_changed.emit()

    def selected_labels(self) -> List[str]:
        return [lab for lab, cb in self._checks.items() if cb.isChecked()]

    def select_all(self):
        for cb in self._checks.values(): cb.setChecked(True)
        self.on_changed.emit()

    def select_none(self):
        for cb in self._checks.values(): cb.setChecked(False)
        self.on_changed.emit()


class ControlsPanel(QGroupBox):
    on_changed = QtCore.Signal()
    def __init__(self, parent=None):
        super().__init__("Controls", parent)
        v = QVBoxLayout(self)

        v.addWidget(QLabel("1‑Tone family"))
        self.grp = QButtonGroup(self)
        self.rb_s1 = QRadioButton("S1"); self.rb_s3 = QRadioButton("S3")
        self.rb_s1.setChecked(True)
        self.grp.addButton(self.rb_s1); self.grp.addButton(self.rb_s3)
        v.addWidget(self.rb_s1); v.addWidget(self.rb_s3)
        self.rb_s1.toggled.connect(lambda _ch: self.on_changed.emit())
        self.rb_s3.toggled.connect(lambda _ch: self.on_changed.emit())

        self.cb_ignore_a2 = QCheckBox("Disregard A2 (assume Pout = |B2|²)")
        self.cb_ignore_a2.setChecked(False)
        self.cb_ignore_a2.stateChanged.connect(lambda _st: self.on_changed.emit())
        v.addWidget(self.cb_ignore_a2)

        v.addStretch(1)

    def curve_pref(self) -> str: return "s1" if self.rb_s1.isChecked() else "s3"
    def ignore_a2(self) -> bool: return self.cb_ignore_a2.isChecked()


class TablesTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.freq_to_records: Dict[str, List[CurveRecord]] = {}
        self.ignore_a2 = False
        self.curve_pref = "s1"

        v = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.cmb_freq = QComboBox()
        self.cmb_curve = QComboBox()
        self.btn_export = QPushButton("Export CSV…")
        controls.addWidget(QLabel("Frequency:"))
        controls.addWidget(self.cmb_freq)
        controls.addSpacing(12)
        controls.addWidget(QLabel("Curve:"))
        controls.addWidget(self.cmb_curve)
        controls.addStretch(1)
        controls.addWidget(self.btn_export)
        v.addLayout(controls)

        self.table = QTableWidget()
        v.addWidget(self.table, 1)

        self.cmb_freq.currentIndexChanged.connect(self._on_freq_changed)
        self.cmb_curve.currentIndexChanged.connect(self._on_curve_changed)
        self.btn_export.clicked.connect(self._on_export)

    def set_data(self, freq_to_records: Dict[str, List[CurveRecord]], labels_sorted: List[str], curve_pref: str, ignore_a2: bool):
        self.freq_to_records = freq_to_records
        self.curve_pref = curve_pref
        self.ignore_a2 = ignore_a2

        self.cmb_freq.blockSignals(True)
        self.cmb_freq.clear()
        for lab in labels_sorted:
            self.cmb_freq.addItem(lab)
        self.cmb_freq.blockSignals(False)

        if labels_sorted:
            self.cmb_freq.setCurrentIndex(0)
            self._populate_curves_for_selected_freq()
            self._populate_table()

    def _choose_record(self, records: List[CurveRecord]) -> Optional[CurveRecord]:
        if not records: return None
        s = self.curve_pref.lower()
        for r in records:
            name = f"{r.dataset_name} {r.curve_name}".lower()
            if "1-tone" in name and s in name: return r
        for r in records:
            name = f"{r.dataset_name} {r.curve_name}".lower()
            if "1-tone" in name: return r
        return records[0]

    def _on_freq_changed(self, _idx):
        self._populate_curves_for_selected_freq()
        self._populate_table()

    def _on_curve_changed(self, _idx):
        self._populate_table()

    def _populate_curves_for_selected_freq(self):
        lab = self.cmb_freq.currentText()
        recs = self.freq_to_records.get(lab, [])
        self.cmb_curve.blockSignals(True)
        self.cmb_curve.clear()
        for r in recs:
            self.cmb_curve.addItem(r.curve_name)
        self.cmb_curve.blockSignals(False)
        if recs:
            self.cmb_curve.setCurrentIndex(0)

    def _current_record(self) -> Optional[CurveRecord]:
        lab = self.cmb_freq.currentText()
        recs = self.freq_to_records.get(lab, [])
        if not recs: return None
        wanted = self.cmb_curve.currentText()
        for r in recs:
            if r.curve_name == wanted:
                return r
        return self._choose_record(recs)

    def _populate_table(self):
        rec = self._current_record()
        if rec is None:
            self.table.clear()
            self.table.setRowCount(0); self.table.setColumnCount(0)
            return
        try:
            df = record_to_dataframe_with_metrics(rec, ignore_a2=self.ignore_a2)
        except Exception as e:
            log_ex(e)
            QtWidgets.QMessageBox.critical(self, "Compute error", f"{e}\n\nSee log: {LOGFILE}")
            return

        self.table.clear()
        self.table.setRowCount(len(df.index))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for j, col in enumerate(df.columns):
            col_vals = df[col].values
            for i, val in enumerate(col_vals):
                txt = ""
                if isinstance(val, (int,float)) and np.isfinite(val):
                    txt = f"{val:.6g}"
                item = QTableWidgetItem(txt)
                self.table.setItem(i, j, item)

        self.table.resizeColumnsToContents()

    def _on_export(self):
        rec = self._current_record()
        if rec is None: return
        try:
            df = record_to_dataframe_with_metrics(rec, ignore_a2=self.ignore_a2)
            suggested = f"table_{(rec.freq_label or 'Freq')}_{re.sub(r'[^A-Za-z0-9_\\-]+','_',rec.curve_name)}.csv"
            path, _ = QFileDialog.getSaveFileName(self, "Export CSV", suggested, "CSV files (*.csv)")
            if not path: return
            df.to_csv(path, index=False)
        except Exception as e:
            log_ex(e)
            QtWidgets.QMessageBox.critical(self, "Export failed", f"{e}\n\nSee log: {LOGFILE}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IM Desktop — Multi‑Format (Safe)")

        self.curves: List[CurveRecord] = []
        self.freq_to_records: Dict[str, List[CurveRecord]] = {}
        self.labels_sorted: List[str] = []

        central = QWidget(); self.setCentralWidget(central)
        outer = QHBoxLayout(central)

        left = QVBoxLayout()
        btn_open = QPushButton("Open .im…"); btn_open.clicked.connect(self.open_file)
        self.controls = ControlsPanel()
        self.freq_panel = FrequencyPanel()
        left.addWidget(btn_open); left.addWidget(self.controls); left.addWidget(self.freq_panel); left.addStretch(1)

        self.tabs = QTabWidget()
        self.plot_grid = PlotGrid()
        plot_tab = QWidget(); plot_tab_layout = QVBoxLayout(plot_tab); plot_tab_layout.addWidget(self.plot_grid)
        self.tabs.addTab(plot_tab, "Plots")
        self.tables_tab = TablesTab(); self.tabs.addTab(self.tables_tab, "Tables")

        outer.addLayout(left, 0); outer.addWidget(self.tabs, 1)

        self.controls.on_changed.connect(self.on_controls_changed)
        self.freq_panel.on_changed.connect(self.on_freq_selection_changed)

        self.statusBar().showMessage('Open a .im file to begin')

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open .im file", "", "IM files (*.im);;All files (*)")
        if not path: return
        try:
            self.load_file(Path(path))
        except Exception as e:
            log_ex(e)
            QtWidgets.QMessageBox.critical(self, "Failed to load", f"{e}\n\nSee log: {LOGFILE}")

    def load_file(self, path: Path):
        parser = pick_parser(path)
        records = parser.parse(path)

        freq_map: Dict[str, List[CurveRecord]] = {}
        for rec in records:
            lab = rec.freq_label or "Freq ?"
            freq_map.setdefault(lab, []).append(rec)

        def label_to_hz(lbl: str) -> float:
            m = FREQ_NAME_RE.search(lbl)
            if not m: return float("inf")
            val = float(m.group("val")); unit = m.group("unit").lower()
            mult = {"ghz":1e9,"mhz":1e6,"khz":1e3,"hz":1.0}.get(unit, 1.0)
            return val * mult
        labels_sorted = sorted(freq_map.keys(), key=label_to_hz)

        self.curves = records
        self.freq_to_records = freq_map
        self.labels_sorted = labels_sorted
        self.freq_panel.set_frequencies(labels_sorted)

        self.statusBar().showMessage(f"Loaded {len(records)} curves via {parser.name()} — Frequencies: {', '.join(labels_sorted)}")
        self.update_plots()
        self.update_tables_tab()

    def on_controls_changed(self):
        self.update_plots(); self.update_tables_tab()

    def on_freq_selection_changed(self):
        self.update_plots(); self.update_tables_tab()

    def update_tables_tab(self):
        if not self.curves: return
        try:
            self.tables_tab.set_data(self.freq_to_records, self.labels_sorted, self.controls.curve_pref(), self.controls.ignore_a2())
        except Exception as e:
            log_ex(e)

    def update_plots(self):
        if not self.curves: return
        selected = self.freq_panel.selected_labels()
        if not selected:
            self.plot_grid.clear_axes(); self.plot_grid.draw()
            self.statusBar().showMessage('No frequencies selected')
            return
        curve_pref = self.controls.curve_pref()
        ignore_a2 = self.controls.ignore_a2()
        try:
            self.plot_grid.overlay_frequency_curves(self.freq_to_records, selected, curve_pref, ignore_a2)
            self.statusBar().showMessage(f'Plotted {len(selected)} frequencies • Pref: {curve_pref.upper()} • {"Pout=|B2|^2" if ignore_a2 else "Pout=|B2|^2−|A2|^2"}')
        except Exception as e:
            log_ex(e)
            QtWidgets.QMessageBox.critical(self, "Plot error", f"{e}\n\nSee log: {LOGFILE}")


class ControlsPanel(QGroupBox):
    on_changed = QtCore.Signal()
    def __init__(self, parent=None):
        super().__init__("Controls", parent)
        v = QVBoxLayout(self)

        v.addWidget(QLabel("1‑Tone family"))
        self.grp = QButtonGroup(self)
        self.rb_s1 = QRadioButton("S1"); self.rb_s3 = QRadioButton("S3")
        self.rb_s1.setChecked(True)
        self.grp.addButton(self.rb_s1); self.grp.addButton(self.rb_s3)
        v.addWidget(self.rb_s1); v.addWidget(self.rb_s3)
        self.rb_s1.toggled.connect(lambda _ch: self.on_changed.emit())
        self.rb_s3.toggled.connect(lambda _ch: self.on_changed.emit())

        self.cb_ignore_a2 = QCheckBox("Disregard A2 (assume Pout = |B2|²)")
        self.cb_ignore_a2.setChecked(False)
        self.cb_ignore_a2.stateChanged.connect(lambda _st: self.on_changed.emit())
        v.addWidget(self.cb_ignore_a2)

        v.addStretch(1)

    def curve_pref(self) -> str: return "s1" if self.rb_s1.isChecked() else "s3"
    def ignore_a2(self) -> bool: return self.cb_ignore_a2.isChecked()


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        w = MainWindow(); w.resize(1200, 800); w.show()
        sys.exit(app.exec())
    except Exception as e:
        log_ex(e)
        raise
