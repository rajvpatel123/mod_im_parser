IM & S-Parameters Desktop Viewer

A lightweight offline desktop app (PySide6 + Matplotlib) for parsing and visualizing .im files (1-tone) and Touchstone .s2p S-parameters, with multi-file comparison, a data Tables view, and an intuitive color policy for comparisons.

Single-file build you have: im_app_single_sidebar_SCROLL_TABLES_COLORFILE_FINAL_fix.py

✨ Features

.im (1-tone) viewer

Plots vs Pout [dBm] @ f0:

Gain (Gt [dB] @ f0)

AM/PM or ACPR (auto-detects ACPR columns: Lower/Upper)

Drain Efficiency [%]

Input Return Loss [dB]

Proper Pout using RF wave variables (|B2|² or |B2|²−|A2|² when A2 available)

S-parameter viewer (.s2p)

Magnitude/phase plots

Smith & polar views

Unwrap option

Multi-file comparison

Color policy:

Multiple files selected → color by file (one color per file; line style varies per file)

Single file selected → color by frequency/series

Works for both IM and S-params

Tables tab

Shows the computed metric table for your current selection

Handy for cross-checking against lab/exported data

Nice-to-use UI

Left sidebar: file/frequency selection & controls

Plots are in a scrollable area (mouse wheel scroll works over the charts)

Light theme by default (offline-friendly)

🧩 Requirements

Python 3.9+ (tested with 3.10/3.11)

Windows (desktop use). Mac/Linux should work with PySide6 but not required for your team.

Python packages:

pip install PySide6 matplotlib numpy pandas

🚀 Quick Start

Clone/download the repo and place:

im_app_single_sidebar_SCROLL_TABLES_COLORFILE_FINAL_fix.py (main app)

Install dependencies:

pip install PySide6 matplotlib numpy pandas


Run:

python im_app_single_sidebar_SCROLL_TABLES_COLORFILE_FINAL_fix.py


In the app:

IM tab → “Open .im…” to load one or more .im files

S-parameters tab → load .s2p/Touchstone

Use the left sidebar to select files and series/frequencies

Switch to Tables tab for a numeric view (auto-refreshes on selection and when you switch to the tab)

📁 Supported File Types
.im files (1-tone)

Must include the usual wave variables for f0 processing (e.g., A1, B1, A2, B2 — stored as sqrt-Watts, with real/imag columns like B2re, B2im).

The app computes:

Pout [W] = |B2|² (or |B2|² − |A2|² if A2 present)

Pout [dBm]

Gt [dB] = 10·log10( Pout / Pin )

AM/PM [deg] from phase difference of transfer function around f0

Drain Efficiency [%] = 100·Pout / Pdc (expects Pdc in Watts)

IRL [dB] = −20·log10(|Γ_in|) (derived from A1/B1 when available)

ACPR [dB] if present (Lower/Upper) — shown in place of AM/PM if detected

If your .im schema differs, adjust the parser in the same file (look for the .im parser class) to map column names.

.s2p files (Touchstone v1)

Plots magnitude/phase; offers Smith and polar views.

Multi-file comparisons follow the color policy above.

🎛️ Controls & Behavior

Color policy (default)

Multiple files selected → color by file; line style (solid, dashed, dotted, etc.) differs per file.

Single file selected → per-frequency colors (i.e., each series gets its own color).

Legend

Compact and positioned to avoid covering data.

Mouse wheel

Scrolls the plot area vertically (event forwarding from the Matplotlib canvases).

📊 Tables Tab

Displays a combined table of computed metrics for your current selection (from the left sidebar).

Updates when:

You change selection

You toggle controls (e.g., curve preference)

You switch to the Tables tab

If you want always-live refresh on every checkbox click without switching tabs, it’s easy to enable—just ask.

🧪 Typical Workflow

Load multiple .im files (e.g., from different devices/runs).

In the left panel:

Check specific files → color-by-file kicks in.

Within a single file, check multiple frequencies → per-frequency coloring.

Review IM plots:

Gain vs Pout [dBm] @ f0

AM/PM or ACPR (auto-detected) vs Pout [dBm] @ f0

Drain Efficiency [%] vs Pout [dBm] @ f0

Input Return Loss [dB] vs Pout [dBm] @ f0

Switch to S-parameters tab for .s2p review (magnitude/phase/Smith/polar).

Switch to Tables to capture numeric values for the selected curves.

🛠️ Build a One-File EXE (Windows)

For easy distribution to test engineers:

pip install pyinstaller
pyinstaller --noconsole --onefile im_app_single_sidebar_SCROLL_TABLES_COLORFILE_FINAL_fix.py


Output EXE will be in dist/.

You can share the EXE alone—no internet connection required on target machines.

If you encounter missing Qt plugins at runtime, use:

pyinstaller --noconsole --onefile --add-data "C:\Python\Lib\site-packages\PySide6\plugins;PySide6\plugins" im_app_single_sidebar_SCROLL_TABLES_COLORFILE_FINAL_fix.py


(Adjust the PySide6 path to your environment.)

🔎 Troubleshooting

Tables tab is empty

Ensure you have files loaded and series selected in the left panel.

Switch to the Tables tab to force a refresh.

All lines look the same color when comparing multiple files

That’s the intended color-by-file behavior. If you only select a single file, you’ll get per-frequency colors.

IndentationError / unexpected syntax

If you hand-edited the file, ensure you didn’t duplicate the same method definitions (especially overlay_frequency_curves).

Matplotlib/Qt errors on import

Confirm dependencies:

pip install PySide6 matplotlib numpy pandas


No internet

Fully offline; no network required after installation.

🧰 Project Layout

Single-file app:

im_app_single_sidebar_SCROLL_TABLES_COLORFILE_FINAL_fix.py
README.md

📄 License

MIT (or your preferred license) — add your chosen text here.

🤝 Contributing

PRs welcome. If you change .im schemas or add new metrics, include a sample file and note the column mappings.

📬 Support / Questions

Open a GitHub issue with:

OS + Python version

The command you ran

A small sample .im / .s2p (if possible)

A screenshot/log of the error
