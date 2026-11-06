# Dataset Explorer

A small Streamlit app for looking at a CSV file. It shows:

- row and column counts, and a preview of the first 100 rows
- column names and pandas types
- missing values per column (count and %), and a button that lists rows with a missing value
- the number of extra exact duplicate rows
- a bar chart and table: the sum of a numeric column for each category
- a comparison of two versions of a CSV: rows, columns, missing cells, added and removed columns and records
- a comparison of two training-run records (JSON, format_version 1): F1, precision and recall A → B, recorded settings and whether the conditions match

The uploaded file is only read. It is never changed or saved.

File rules: `.csv` file, UTF-8, first row is the header, comma or semicolon separator, up to 5 MB.

## Versions checked (2026-10-22)

| Item | Version |
|---|---|
| OS | Omarchy Linux (x86_64), bash |
| Python | 3.13.16 (installed for this user with mise; system Python 3.14 is not used) |
| pandas | 3.0.6 |
| numpy | 2.5.4 |
| matplotlib | 3.11.2 |
| scikit-learn | 1.9.1 |
| streamlit | 1.66.1 |

## Start the app

```bash
cd /home/amir/Projects/AI_ML/Dataset_Explorer
.venv/bin/streamlit run app.py --server.port 8502
```

Open http://localhost:8502. Press `Ctrl+C` in the terminal to stop it.
If port 8502 is busy, use another port, for example `--server.port 8503`.

## Set up again (only if `.venv` is missing or broken)

```bash
cd /home/amir/Projects/AI_ML/Dataset_Explorer
mise install python@3.13.16
"$(mise where python@3.13.16)/bin/python3.13" -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Files

| File | Purpose |
|---|---|
| `app.py` | The app |
| `requirements.txt` | Package versions |
| `.streamlit/config.toml` | Dark theme: periwinkle `#8B97FF` for the app, amber for things to inspect, mint for all good. The glass widgets and glowing backdrop are CSS in `app.py` |
| `.python-version` | Python version for this folder (3.13) |
| `.vscode/settings.json` | Makes VS Code use `.venv/bin/python` |
| `check_dataset.csv` | Test file: A must total 30, B must total 10 |
| `run_A_example.json`, `run_B_example.json` | Example training-run records for "Compare two training runs" |
| `checks.txt` | Check results (automated, browser, and manual checks not yet done) |
| `setup-check.txt` | Record of the first environment setup |
| `.venv/` | Python environment (can be rebuilt with the commands above) |
