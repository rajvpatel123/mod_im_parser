IM & S‑Parameters Desktop Viewer (Single‑File App)
====================================================

This is a lightweight OFFLINE desktop app (PySide6 + Matplotlib) for parsing
and visualizing .im (1‑tone) and Touchstone .s2p files. It supports multi‑file
comparison, a numeric Tables view, and a color policy that is designed for
comparing different devices/runs.

Main file you have:
   parser_main.py

---------------------------------------------------------------------
FEATURES
---------------------------------------------------------------------
- .im (1‑tone) viewer
  Plots vs Pout [dBm] @ f0:
    • Gain (Gt [dB] @ f0)
    • AM/PM [deg] @ f0 OR ACPR [dB] (auto‑detected if Lower/Upper ACPR present)
    • Drain Efficiency [%]
    • Input Return Loss [dB]
  Proper Pout using wave variables:
    Pout[W] = |B2|^2  (or |B2|^2 − |A2|^2 if A2 present)
    Pout[dBm] = 10*log10(Pout[W] / 1e‑3)

- S‑parameter viewer (.s2p)
  Magnitude/phase, Smith, and Polar views with unwrap option.

- Multi‑file comparison
  COLOR POLICY (default):
    • Multiple files selected  -> color BY FILE (one color per file; linestyle varies per file)
    • Single file selected     -> color BY FREQUENCY/SERIES (each frequency gets a color)
  This policy applies to BOTH IM and S‑parameters views.

- Tables tab
  Shows computed metrics for the current selection (left side panel).
  Good for cross‑checking against lab/exported data.

- UI niceties
  Left sidebar for file/series selection; plot area is scrollable; mouse‑wheel
  scroll works over charts; offline‑friendly.

---------------------------------------------------------------------
REQUIREMENTS
---------------------------------------------------------------------
- Python 3.9+ (tested on 3.10/3.11)
- Windows desktop (Mac/Linux should work with PySide6 but not required for team)
- Python packages:
    pip install PySide6 matplotlib numpy pandas

---------------------------------------------------------------------
QUICK START
---------------------------------------------------------------------
1) Place the main app file somewhere convenient:
       im_app_single_sidebar_SCROLL_TABLES_COLORFILE_FINAL_fix.py

2) Install dependencies:
       pip install PySide6 matplotlib numpy pandas

3) Run:
       python parser_main.py

4) In the app:
   - IM tab:  “Open .im…”             (load one or more .im files)
   - S‑params: load .s2p files        (Touchstone v1)
   - Choose files and series/frequencies in the left sidebar.
   - Tables tab: numeric view of your current selection.

---------------------------------------------------------------------
SUPPORTED FILES
---------------------------------------------------------------------
.im files (1‑tone)
  Expected columns (or map accordingly in the internal parser):
    freq_label, A1re, A1im, B1re, B1im, A2re, A2im, B2re, B2im, PdcW
  Notes:
    • A*/B* are sqrt‑Watts wave quantities (real/imag parts).
    • PdcW is already in Watts.
    • The app computes Pout[W], Pout[dBm], Gt[dB], AM/PM, IRL, and ACPR (if present).

.s2p files (Touchstone v1)
  Uses “# MHz S MA R 50” style. Magnitude/angle pairs are supported.
  Multi‑file comparisons follow the color policy above.

---------------------------------------------------------------------
COLOR POLICY (IMPORTANT)
---------------------------------------------------------------------
- Multi‑file selected:  COLOR BY FILE (one color per file) + different linestyles.
- Single file selected: COLOR BY FREQUENCY/SERIES (each frequency gets a color).
- Applies to both IM and S‑parameters plotting.

---------------------------------------------------------------------
TABLES TAB
---------------------------------------------------------------------
- Displays computed metrics for the CURRENT SELECTION (what you’ve checked in
  the left sidebar). It auto‑refreshes when switching to the tab and on most
  selection/controls changes.

---------------------------------------------------------------------
BUILD A ONE‑FILE EXE (WINDOWS)
---------------------------------------------------------------------
- Easiest way for field/test‑bench machines:
    pip install pyinstaller
    pyinstaller --noconsole --onefile parser_main.py

- The EXE will be in “dist/”. You may share the single EXE with coworkers.
- If Qt plugins are missing in the EXE environment, try:
    pyinstaller --noconsole --onefile --add-data "C:\Python\Lib\site-packages\PySide6\plugins;PySide6\plugins" parser_main.py
  (Adjust the path to your PySide6 installation)

---------------------------------------------------------------------
TROUBLESHOOTING
---------------------------------------------------------------------
- Tables tab looks empty
    * Load files and check at least one series/frequency on the left.
    * Switching to the Tables tab forces a refresh.

- All lines are the same color when comparing files
    * That’s the intended “color‑by‑file” behavior for multi‑file comparison.
      Select a single file to see per‑frequency colors.

- Import or indentation errors
    * Ensure you’re running the unmodified single‑file app and that Python
      has the required packages installed.



