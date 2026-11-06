"""Dataset Explorer: upload a CSV file and see basic facts about it.

The uploaded file is only read. It is never changed or saved.
"""

import io
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

MAIN_COLOR = "#8B97FF"   # same as primaryColor in .streamlit/config.toml
FLAG_COLOR = "#F5A524"   # same as orangeColor: something that needs a look
TEXT_COLOR = "#E7EAF6"   # same as textColor
MUTED_COLOR = "#9AA3C2"
LINE_COLOR = "#3A4772"   # same as borderColor
MAX_FILE_MB = 5
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
PREVIEW_ROWS = 100
SEPARATORS = {"Comma ( , )": ",", "Semicolon ( ; )": ";"}

# Cell values that count as "missing". This is the pandas default list,
# written out here so the rule is visible and does not change silently.
MISSING_MARKERS = [
    "", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan",
    "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA", "NULL", "NaN", "None",
    "n/a", "nan", "null",
]


class ReadError(Exception):
    """A problem with the file, with a message the user can understand."""


def parse_csv(data: bytes, sep: str, **options) -> pd.DataFrame:
    """pd.read_csv with file problems turned into messages the user can understand."""
    if not data.strip():
        raise ReadError("The file is empty. Please upload a CSV file that has a header row and data.")
    try:
        # utf-8-sig reads normal UTF-8 and also UTF-8 files that start with a BOM.
        return pd.read_csv(io.BytesIO(data), sep=sep, header=0, encoding="utf-8-sig", **options)
    except UnicodeDecodeError:
        raise ReadError("The file is not UTF-8 text. Save it as \"CSV UTF-8\" and upload it again.")
    except pd.errors.EmptyDataError:
        raise ReadError("The file is empty. Please upload a CSV file that has a header row and data.")
    except pd.errors.ParserError as e:
        raise ReadError(
            "The file could not be read as a CSV. Check that you chose the right separator "
            f"and that every row has the same number of values.\n\nDetails: {e}"
        )
    except Exception as e:
        raise ReadError(f"The file could not be read.\n\nDetails: {e}")


@st.cache_data(show_spinner="Reading file...")
def read_csv(data: bytes, sep: str) -> tuple[pd.DataFrame, int]:
    """Read the CSV bytes. Returns the table and the number of extra exact duplicate rows."""
    df = parse_csv(data, sep, keep_default_na=False, na_values=MISSING_MARKERS)
    # Duplicates are checked on the original cell text, so "1" and "1.0" count as different.
    raw_text = parse_csv(data, sep, dtype=str, keep_default_na=False)
    duplicates = int(raw_text.duplicated(keep="first").sum())
    return df, duplicates


@st.cache_data(show_spinner="Reading file...")
def read_text_table(data: bytes, sep: str) -> pd.DataFrame:
    """Every cell as its original text, for comparing two versions of a file.

    "007" stays "007" and "NA" stays "NA". Only an empty cell or a cell made of
    whitespace counts as missing here. This is a separate view; the main table is not changed.
    """
    return parse_csv(data, sep, dtype=str, keep_default_na=False)


def blank_cells(text: pd.DataFrame) -> pd.DataFrame:
    """True where a cell is empty or only whitespace."""
    return text.map(lambda v: str(v).strip() == "")


def id_problem(text: pd.DataFrame, col: str) -> str | None:
    """Why `col` cannot be used as a record ID in this file, or None if it can."""
    values = text[col].astype(str)
    blanks = int((values.str.strip() == "").sum())
    if blanks:
        return f"{blanks:,} blank value{'s' if blanks != 1 else ''}"
    dups = int(values.duplicated().sum())
    if dups:
        return f"{dups:,} repeated value{'s' if dups != 1 else ''}"
    return None


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Columns with numbers (not True/False) and at least one value that is not missing."""
    return [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and df[c].notna().any()
    ]


def category_totals(df: pd.DataFrame, cat: str, num: str) -> pd.DataFrame:
    """Sum of `num` for each value of `cat`, sorted by total (largest first).

    Missing numbers are skipped, not treated as zero. A category whose numbers
    are all missing gets a total of NaN ("No data"). Rows with a missing
    category are left out here and reported separately.
    """
    groups = df.groupby(cat, dropna=True, sort=False)[num]
    table = pd.DataFrame({
        "total": groups.sum(min_count=1),
        "rows": groups.size(),
        "missing_numbers": groups.size() - groups.count(),
    })
    table = table.sort_values("total", ascending=False, na_position="last")
    table.index = table.index.map(str)
    return table


def format_total(value) -> str:
    if pd.isna(value):
        return "No data"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{round(float(value), 6):,}"


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    percent = (missing / len(df) * 100) if len(df) else missing * 0.0
    return pd.DataFrame({
        "Column": df.columns,
        "Missing values": missing.values,
        "Missing %": percent.round(2).values,
    })


def is_space_only(value) -> bool:
    """True for text made only of one or more spaces, like " " or "   ".

    Empty text, missing values (NaN), numbers, 0 and "0" are not counted.
    Only the space character counts, not tabs or line breaks.
    """
    return isinstance(value, str) and value != "" and value.strip(" ") == ""


def iqr_check(values: pd.Series) -> dict:
    """Tukey's rule on a Series of finite numbers (a copy; the table is not changed).

    Q1 and Q3 use pandas' default linear interpolation. IQR = Q3 - Q1.
    A value needs inspection when it is strictly below Q1 - 1.5 * IQR or
    strictly above Q3 + 1.5 * IQR. A constant column gives IQR = 0 and no flags.
    """
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "q1": q1, "q3": q3, "iqr": iqr, "lower": lower, "upper": upper,
        "median": values.median(),
        "flagged": values[(values < lower) | (values > upper)],
        "inside": values[(values >= lower) & (values <= upper)],
    }


def space_only_counts(df: pd.DataFrame) -> pd.Series:
    """Number of cells in each column whose value is text made only of spaces.

    This only reads the table. The values themselves are not trimmed or changed.
    """
    return pd.Series([int(df[c].map(is_space_only).sum()) for c in df.columns], index=df.columns)


# Training-run records (format_version 1). The IDs in a record are labels written by whoever
# ran the training; the app does not check them against any data.
RUN_FIELDS = ["run_id", "dataset_id", "split_id", "features", "target", "positive_class", "model", "penalty",
              "C", "seed", "threshold", "evaluation_split", "metrics", "note"]
CONTEXT_FIELDS = ["dataset_id", "split_id", "target", "positive_class", "features", "threshold", "seed", "evaluation_split"]
INTENDED_FIELDS = ["model", "penalty", "C"]  # settings chosen on purpose; a difference here is not a mismatch
METRIC_FIELDS = ["f1", "precision", "recall", "train_f1"]


def load_run(data: bytes) -> tuple[dict | None, list[str]]:
    """Parse one run record. Returns (record, problems). The record is None when the file is unusable."""
    try:
        obj = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError:
        return None, ["The file is not UTF-8 text."]
    except json.JSONDecodeError as e:
        return None, [f"The file is not valid JSON: {e.msg} at line {e.lineno}, column {e.colno}."]
    if not isinstance(obj, dict):
        return None, ["The JSON must be an object with the run fields, not a list or a single value."]
    if obj.get("format_version") != 1:
        return None, [f"format_version must be 1. Found {obj.get('format_version')!r}."]
    problems = []
    missing = [f for f in RUN_FIELDS if f not in obj]
    if missing:
        problems.append("Missing fields: " + ", ".join(missing) + ".")
    if "metrics" in obj and not isinstance(obj["metrics"], dict):
        problems.append("metrics must be an object with train_f1, f1, precision and recall.")
    return obj, problems


def metric_value(run: dict, name: str) -> tuple[float | None, str | None]:
    """A metric as a float, or None and the reason it cannot be used. 0 is a valid value."""
    metrics = run.get("metrics")
    if not isinstance(metrics, dict) or name not in metrics:
        return None, "missing"
    v = metrics[name]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None, f"not a number ({v!r})"
    if not math.isfinite(v):
        return None, f"not finite ({v!r})"
    if not 0 <= v <= 1:
        return None, f"outside 0 to 1 ({v!r})"
    return float(v), None


def context_value(run: dict, field: str):
    """A context field in comparable form, or None when it is absent, empty or the wrong type."""
    v = run.get(field)
    if field == "features":
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return tuple(sorted(v))  # order does not matter
        return None
    if v is None or v == "" or isinstance(v, bool):
        return None
    return v


def show_value(v) -> str:
    if v is None:
        return "not recorded"
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return str(v)


def run_comparison_section() -> None:
    """Two run records, A and B: metric cards, recorded settings and their differences. No winner is chosen."""
    st.write("Upload two training-run records (JSON, format_version 1) to see their F1, precision and recall side by side, "
             "and whether they were recorded under the same conditions. Nothing is retrained or re-evaluated here.")
    ca, cb = st.columns(2)
    file_a = ca.file_uploader("Run A (JSON)", type=["json"], key="run_a", max_upload_size=MAX_FILE_MB)
    file_b = cb.file_uploader("Run B (JSON)", type=["json"], key="run_b", max_upload_size=MAX_FILE_MB)
    if file_a is None or file_b is None:
        st.info("Upload both run records to compare them. "
                f"{'Run A is loaded.' if file_a else ''}{'Run B is loaded.' if file_b else ''}")
        return

    runs, usable = {}, True
    for label, f, col in (("A", file_a, ca), ("B", file_b, cb)):
        run, problems = load_run(f.getvalue())
        if run is None:
            col.error(f"Run {label} ({f.name}) cannot be read. {problems[0]}")
            usable = False
            continue
        col.markdown(f"**Run {label}:** `{f.name}` · run_id **{show_value(run.get('run_id'))}**")
        for msg in problems:
            col.warning(f"Run {label}: {msg}")
        runs[label] = run
    if not usable:
        st.info("Fix the file above and upload it again. The comparison is not shown while a file cannot be read.")
        return
    a, b = runs["A"], runs["B"]

    # Three cards. The arrow shows direction only; a higher value is not a verdict.
    with st.container(key="run-cards"):
        cards = st.columns(3)
    invalid = []
    for col, name, title in zip(cards, ("f1", "precision", "recall"), ("F1", "Precision", "Recall")):
        va, why_a = metric_value(a, name)
        vb, why_b = metric_value(b, name)
        if va is None or vb is None:
            col.metric(title, "n/a", border=True)
            for label, why in (("A", why_a), ("B", why_b)):
                if why:
                    invalid.append(f"{title} in run {label} is {why}")
            continue
        d = vb - va
        col.metric(title, f"{va:.4f} → {vb:.4f}", delta=f"{d:+.4f}" if d else "0 (same)",
                   delta_color="normal" if d else "off", border=True)
    st.caption("A → B, delta = B − A. Green is up, red is down, grey is no change. "
               "Direction is not a quality verdict: the right metric depends on what the model is for.")
    if invalid:
        st.error("Invalid metrics are not used in any calculation: " + "; ".join(invalid) + ". "
                 "Metrics must be finite numbers from 0 to 1.")

    # Recorded conditions: the fields that must be known and equal for the metrics to be comparable.
    not_recorded = [f for f in CONTEXT_FIELDS for r in (a, b) if context_value(r, f) is None]
    not_recorded = list(dict.fromkeys(not_recorded))
    differing = [f for f in CONTEXT_FIELDS if f not in not_recorded and context_value(a, f) != context_value(b, f)]
    not_validation = [label for label, r in (("A", a), ("B", b))
                      if context_value(r, "evaluation_split") not in (None, "validation")]
    if not_recorded:
        st.warning("Insufficient context: " + ", ".join(not_recorded) + " not recorded in one or both runs. "
                   "Without it the metrics cannot be said to come from the same conditions. No value is assumed.")
    elif differing or not_validation:
        reasons = [f"{f} differs" for f in differing]
        reasons += [f"run {label} was evaluated on \"{context_value(runs[label], 'evaluation_split')}\" rather than "
                    "validation" for label in not_validation]
        st.warning("Conditions differ: " + "; ".join(reasons) + ". The metrics were recorded under different "
                   "conditions, so a direct comparison is not supported by these records.")
    else:
        st.success("Same recorded conditions: dataset, split, target, positive class, features, threshold and seed "
                   "match, and both runs were evaluated on the validation split.")
    intended_diffs = [f for f in INTENDED_FIELDS if a.get(f) != b.get(f)]
    if intended_diffs:
        st.caption("Intended settings that differ: " + ", ".join(intended_diffs) + ". "
                   "These are choices, not a mismatch in conditions.")

    st.dataframe(pd.DataFrame({
        "Setting": CONTEXT_FIELDS + INTENDED_FIELDS,
        "Run A": [show_value(context_value(a, f)) for f in CONTEXT_FIELDS] + [show_value(a.get(f)) for f in INTENDED_FIELDS],
        "Run B": [show_value(context_value(b, f)) for f in CONTEXT_FIELDS] + [show_value(b.get(f)) for f in INTENDED_FIELDS],
        "Status": ["Not recorded" if f in not_recorded else "Differs" if f in differing else "Same" for f in CONTEXT_FIELDS]
                  + ["Differs (intended)" if f in intended_diffs else "Same" for f in INTENDED_FIELDS],
    }), hide_index=True, width="stretch")
    st.caption("IDs are labels written into the record. They are not checked against the data.")

    with st.expander("Details: training gap and notes"):
        for label, r in (("A", a), ("B", b)):
            split = context_value(r, "evaluation_split")
            tf, why_t = metric_value(r, "train_f1")
            f1, why_f = metric_value(r, "f1")
            if split != "validation":
                gap = (f"train_f1 − f1 unavailable: evaluation split is {show_value(split)}, "
                       "and the gap is only meaningful against the validation split.")
            elif tf is None or why_f:
                gap = "train_f1 − f1 unavailable: " + "; ".join(
                    f"{n} is {w}" for n, w in (("train_f1", why_t), ("f1", why_f)) if w) + "."
            else:
                gap = f"train_f1 − f1 = {tf:.4f} − {f1:.4f} = {tf - f1:+.4f}"
            note = r.get("note")
            st.markdown(f"**Run {label}** ({show_value(r.get('run_id'))}): {gap}  \n"
                        f"Note: {note if isinstance(note, str) and note.strip() else 'none'}")


st.set_page_config(page_title="Dataset Explorer", page_icon=":material/table_view:", layout="wide")

# Layout touches the theme cannot set: the glowing backdrop, glass widgets, page width, display numbers
# and the specimen table. Glass = a translucent fill over a blurred copy of the backdrop behind it.
GLASS = "background: rgba(255, 255, 255, 0.045); border: 1px solid rgba(255, 255, 255, 0.18); " \
        "backdrop-filter: blur(18px) saturate(150%); -webkit-backdrop-filter: blur(18px) saturate(150%);"
st.html(f"""
<style>
  /* Backdrop: a night sky with two soft glows, fixed so it stays put while the page scrolls. */
  [data-testid="stAppViewContainer"] {{
    background-color: #0B1020;
    background-image:
      radial-gradient(70rem 42rem at 10% -12%, rgba(139, 151, 255, 0.46), transparent 62%),
      radial-gradient(50rem 34rem at 100% 100%, rgba(245, 165, 36, 0.22), transparent 62%),
      radial-gradient(36rem 28rem at 72% 38%, rgba(95, 212, 196, 0.12), transparent 60%);
    background-attachment: fixed;
  }}
  [data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ max-width: 70rem; padding-top: 3.5rem; }}
  [data-testid="stMarkdownContainer"] p {{ max-width: 44rem; }}
  h1 {{ letter-spacing: -0.025em; line-height: 1.05; max-width: 24ch; text-wrap: balance; }}
  .brand {{ font-family: "Bricolage Grotesque", sans-serif; font-weight: 700; font-size: 1.25rem; color: {MUTED_COLOR}; margin: 0 0 0.5rem; }}

  /* Strong glass: the number cards. A bright top edge makes them read as a pane. */
  [data-testid="stMetric"] {{
    {GLASS} border-radius: 16px; padding: 1rem 1.25rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14), 0 12px 32px rgba(0, 0, 0, 0.28);
  }}
  [data-testid="stMetricValue"] {{
    font-family: "Bricolage Grotesque", sans-serif; font-weight: 700; font-size: 2.25rem;
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
  }}
  [data-testid="stMetricLabel"] p {{ color: {MUTED_COLOR}; }}
  [data-testid="stMetricDelta"] {{ background: rgba(255, 255, 255, 0.08); border-radius: 999px; padding: 0.1rem 0.5rem; }}
  /* Small supporting numbers (the IQR boundaries) are plain, not cards. */
  .st-key-boundaries [data-testid="stMetric"] {{ background: none; border: 0; box-shadow: none; backdrop-filter: none; -webkit-backdrop-filter: none; padding: 0; }}
  .st-key-boundaries [data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
  /* Run cards hold two numbers and an arrow, so the value is set smaller to fit three across. */
  .st-key-run-cards [data-testid="stMetricValue"] {{ font-size: 1.5rem; }}

  /* Quiet glass: panels you open and things you fill in. */
  [data-testid="stExpander"] details {{ {GLASS} border-radius: 12px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22); }}
  [data-testid="stExpander"] summary {{ padding: 0.75rem 1rem; }}
  [data-testid="stFileUploaderDropzone"] {{ {GLASS} border-radius: 12px; border-style: dashed; }}
  [data-testid="stFileUploaderDropzone"] button {{ background: rgba(255, 255, 255, 0.08); border-color: rgba(255, 255, 255, 0.16); }}
  [data-baseweb="select"] > div, [data-testid="stCheckbox"] span:first-child {{
    background: rgba(255, 255, 255, 0.05); border-color: rgba(255, 255, 255, 0.14);
  }}
  [data-baseweb="popover"] [data-baseweb="menu"] {{ background: rgba(20, 27, 49, 0.85); backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); }}
  [data-testid="stAlert"] {{ {GLASS} border-radius: 12px; }}
  [data-testid="stButton"] button[kind="primary"] {{
    background: rgba(139, 151, 255, 0.22); border: 1px solid rgba(139, 151, 255, 0.55); color: {TEXT_COLOR};
    backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  }}
  [data-testid="stButton"] button[kind="primary"]:hover {{ background: rgba(139, 151, 255, 0.34); border-color: {MAIN_COLOR}; }}
  [data-testid="stCaptionContainer"] {{ font-variant-numeric: tabular-nums; }}
  [data-testid="stTabs"] button[role="tab"] p {{ font-size: 1rem; }}
  /* Tables stay solid: a blurred grid is hard to read. A soft edge keeps them in the same family. */
  [data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.16), 0 8px 24px rgba(0, 0, 0, 0.22); }}
  /* Without backdrop blur, use a slightly denser fill so text stays readable. */
  @supports not (backdrop-filter: blur(1px)) {{
    [data-testid="stMetric"], [data-testid="stExpander"] details, [data-testid="stFileUploaderDropzone"], [data-testid="stAlert"], .specimen {{
      background: rgba(20, 27, 49, 0.92);
    }}
  }}

  /* Specimen table on the start page: the four things the explorer points out. */
  .specimen {{ {GLASS} border-radius: 16px; padding: 1.5rem 1.5rem 1.25rem;
               box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.14), 0 16px 40px rgba(0, 0, 0, 0.32); }}
  .specimen table {{ border-collapse: collapse; width: 100%; font-family: "JetBrains Mono", monospace; font-size: 0.95rem; }}
  .specimen th {{ text-align: left; color: {MUTED_COLOR}; font-weight: 400; padding: 0.4rem 0.75rem; border-bottom: 1px solid rgba(255, 255, 255, 0.14); }}
  .specimen td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid rgba(255, 255, 255, 0.06); color: {TEXT_COLOR}; }}
  .specimen td:first-child, .specimen th:first-child {{ color: {MUTED_COLOR}; width: 2rem; text-align: right; }}
  .specimen .num {{ text-align: right; }}
  .specimen .missing {{ outline: 1px dashed {MUTED_COLOR}; outline-offset: -4px; color: {MUTED_COLOR}; text-align: center; }}
  .specimen .space {{ background: rgba(139, 151, 255, 0.16); color: {MAIN_COLOR}; letter-spacing: 0.15em; }}
  .specimen .dup td {{ color: {MUTED_COLOR}; font-style: italic; }}
  .specimen .flag {{ color: {FLAG_COLOR}; font-weight: 600; }}
  .specimen .key {{ margin: 1.25rem 0 0; padding: 0; list-style: none; display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.4rem 2rem; color: {MUTED_COLOR}; font-size: 0.95rem; }}
  .specimen .key b {{ color: {TEXT_COLOR}; font-weight: 400; }}
  @media (max-width: 40rem) {{ .specimen .key {{ grid-template-columns: 1fr; }} }}
</style>
""")

# The big headline is for the start page. Once a file is loaded the heading shrinks to a small name line.
hero = st.container()

uploaded = st.file_uploader(
    f"CSV file (UTF-8, first row is the header, up to {MAX_FILE_MB} MB)",
    max_upload_size=MAX_FILE_MB,
)
sep_label = st.radio("Values are separated by", list(SEPARATORS), horizontal=True)

with hero:
    if uploaded is None:
        st.title("See what's inside a CSV before you trust it.")
        st.write("Size, columns, empty and space-only cells, duplicate rows, unusual numbers, and totals by group. "
                 "Your file is only read, never changed.")
    else:
        st.html('<p class="brand">Dataset Explorer</p>')

if uploaded is None:
    st.markdown(":material/arrow_upward: Choose a CSV file to start. "
                "For a quick test, use `check_dataset.csv` from the project folder.")
    st.html("""
<div class="specimen">
  <table>
    <tr><th></th><th>customer</th><th>branch</th><th class="num">days</th><th class="num">amount</th></tr>
    <tr><td>0</td><td>Lena K.</td><td>Airport</td><td class="num">3</td><td class="num">210</td></tr>
    <tr><td>1</td><td>Omar S.</td><td class="missing">empty</td><td class="num">5</td><td class="num">355</td></tr>
    <tr><td>2</td><td>Ana P.</td><td class="space">&middot;&middot;&middot;</td><td class="num">2</td><td class="num">140</td></tr>
    <tr class="dup"><td>3</td><td>Lena K.</td><td>Airport</td><td class="num">3</td><td class="num">210</td></tr>
    <tr><td>4</td><td>Jon M.</td><td>Downtown</td><td class="num">4</td><td class="num flag">28,000</td></tr>
  </table>
  <ul class="key">
    <li><b>Row 1</b> has an empty cell. Counted as missing.</li>
    <li><b>Row 2</b> has a cell made only of spaces. Counted separately.</li>
    <li><b>Row 3</b> is an exact copy of row 0. Counted as a duplicate.</li>
    <li><b>Row 4</b> has an amount far from the rest. Marked for inspection.</li>
  </ul>
</div>
""")
    st.subheader("Compare two training runs", divider="blue")
    run_comparison_section()
    st.stop()

if not uploaded.name.lower().endswith(".csv"):
    st.error(f"\"{uploaded.name}\" is not a CSV file. Please upload a file whose name ends in .csv.")
    st.stop()

if uploaded.size > MAX_FILE_BYTES:
    st.error(f"This file is {uploaded.size / 1024 / 1024:.1f} MB. The limit is {MAX_FILE_MB} MB per file.")
    st.stop()

try:
    df, duplicate_rows = read_csv(uploaded.getvalue(), SEPARATORS[sep_label])
except ReadError as e:
    st.error(str(e))
    st.stop()

if len(df.columns) == 1:
    other = ";" if SEPARATORS[sep_label] == "," else ","
    if other in str(df.columns[0]):
        st.warning(f"Only one column was found, and its name contains \"{other}\". "
                   "You may have chosen the wrong separator.")

if len(df) == 0:
    st.warning("The file has a header row but no data rows.")

st.header(uploaded.name, divider="blue")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}", border=True)
c2.metric("Columns", f"{len(df.columns):,}", border=True)
c3.metric("Extra duplicate rows", f"{duplicate_rows:,}", border=True)
st.caption("All counts use the whole file. The header row is not counted as a row. "
           "\"Extra duplicate rows\" counts rows that exactly match an earlier row; "
           "the first copy is not counted.")

# Plain-language summary. Problems are marked in orange; everything else stays neutral.
missing_info = missing_table(df)
cols_with_missing = missing_info[missing_info["Missing values"] > 0]
num_cols = numeric_columns(df)
findings = []
if duplicate_rows:
    findings.append(f":orange[:material/content_copy: **{duplicate_rows:,} "
                    f"{'row is an exact copy' if duplicate_rows == 1 else 'rows are exact copies'}** of an earlier row.]")
else:
    findings.append(":green[:material/check: No duplicate rows.]")
if len(cols_with_missing):
    worst = cols_with_missing.sort_values("Missing %", ascending=False).iloc[0]
    findings.append(f":orange[:material/help: **{len(cols_with_missing):,} of {len(df.columns):,} columns** "
                    f"have missing values. The most is in \"{worst['Column']}\" ({worst['Missing %']:g}%).]")
else:
    findings.append(":green[:material/check: No missing values.]")
if num_cols:
    shown = ", ".join(f"\"{c}\"" for c in num_cols[:5]) + (" and more" if len(num_cols) > 5 else "")
    findings.append(f":material/functions: {len(num_cols):,} numeric "
                    f"{'column' if len(num_cols) == 1 else 'columns'} can be added up: {shown}.")
else:
    findings.append(":material/functions: No numeric columns, so totals by category are not available.")

st.subheader("What we found")
st.markdown("\n".join(f"- {f}" for f in findings))

preview_tab, columns_tab, inspect_tab, compare_tab, runs_tab, totals_tab = st.tabs(
    ["Preview", "Columns and missing values", "Inspect numeric values", "Compare versions", "Compare runs",
     "Totals by category"], key="section")

with preview_tab:
    shown_rows = min(len(df), PREVIEW_ROWS)
    st.caption(f"The first {shown_rows:,} of {len(df):,} rows. Row numbers start at 0.")
    st.dataframe(df.head(PREVIEW_ROWS), width="stretch")

with columns_tab:
    space_only = space_only_counts(df)
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "pandas type": df.dtypes.astype(str).values,
            "Missing values": missing_info["Missing values"].values,
            "Missing %": missing_info["Missing %"].values,
            "Space-only cells": space_only.values,
        }),
        hide_index=True, width="stretch",
        column_config={
            "Missing %": st.column_config.ProgressColumn(
                "Missing %", min_value=0, max_value=100, format="%.2f%%", color=MAIN_COLOR),
        },
    )
    total_space_only = int(space_only.sum())
    st.caption(f"Space-only cells in the whole file: **{total_space_only:,}**. "
               "These are cells whose text is only one or more spaces. They are counted separately "
               "and are not included in \"Missing values\". The values are left exactly as they are in the file.")
    if total_space_only:
        # Optional view only. The full table is still used everywhere else on the page.
        if st.checkbox("Show only rows with a space-only cell", key="show_space_only_rows"):
            rows_with_spaces = df[df.map(is_space_only).any(axis=1)]
            st.write(f"{len(rows_with_spaces):,} of {len(df):,} rows have at least one space-only cell. "
                     "Row numbers match the preview. The values are shown as they are in the file.")
            st.dataframe(rows_with_spaces, width="stretch")
    with st.expander("How missing values are counted"):
        st.write("A cell counts as missing when it is empty, or when its whole text is exactly one of these values:")
        st.code(", ".join(repr(v) for v in MISSING_MARKERS if v) + ", and an empty cell ''")
        st.write("Matching is case-sensitive: for example `NULL` and `null` count, but `Null` does not. "
                 "A cell with only spaces, a dash (`-`), `0` or `?` is **not** counted as missing. "
                 "Missing % = missing values in the column ÷ number of rows × 100.")
        st.write("\"Space-only cells\" counts cells whose text is made only of spaces (for example `\" \"` or `\"   \"`). "
                 "Empty cells, missing values, `0` and `\"0\"` are not counted. Tabs are not counted as spaces.")

    # Show/hide rows that have at least one missing value. The choice is reset when a new file is uploaded.
    file_id = (uploaded.name, uploaded.size, sep_label)
    if st.session_state.get("missing_rows_file") != file_id:
        st.session_state.missing_rows_file = file_id
        st.session_state.show_missing_rows = False

    rows_with_missing = df[df.isna().any(axis=1)]
    if rows_with_missing.empty:
        st.success("No rows have missing values.")
    else:
        showing = st.session_state.show_missing_rows
        label = "Hide rows with missing values" if showing else f"Show rows with missing values ({len(rows_with_missing):,})"
        if st.button(label, type="primary", icon=":material/visibility_off:" if showing else ":material/visibility:"):
            st.session_state.show_missing_rows = not showing
            st.rerun()
        if showing:
            st.write(f"{len(rows_with_missing):,} of {len(df):,} rows have at least one missing value. "
                     "Row numbers match the preview. Missing cells are shown as \"None\".")
            st.dataframe(rows_with_missing, width="stretch")

with inspect_tab:
    # This tab comes before "Totals by category" in the code because that tab can call st.stop().
    if not num_cols:
        st.info("There is no numeric column to inspect. A column only counts as numeric when every value "
                "in it can be read as a number (missing values are allowed). The rest of the page still works.")
    else:
        inspect_col = st.selectbox("Numeric column to inspect", num_cols)
        # A copy of the column as floats. The uploaded table itself is not changed.
        numbers = df[inspect_col].astype(float)
        usable = numbers[np.isfinite(numbers)]
        excluded = len(df) - len(usable)

        m1, m2, m3 = st.columns(3)
        m1.metric("Usable values", f"{len(usable):,}", border=True)
        m2.metric("Missing or excluded", f"{excluded:,}", border=True)
        if len(usable) == 0:
            m3.metric("Needs inspection", "–", border=True)
            st.info(f"\"{inspect_col}\" has no usable value: every cell is missing or not a finite number, "
                    "so there is nothing to inspect.")
        elif len(usable) < 4:
            m3.metric("Needs inspection", "–", border=True)
            st.info(f"\"{inspect_col}\" has only {len(usable):,} usable value{'s' if len(usable) > 1 else ''}. "
                    "At least 4 are needed for the IQR rule, so no verdict is shown. These are the rows:")
            st.dataframe(df.loc[usable.index, [inspect_col]], width="stretch")
        else:
            r = iqr_check(usable)
            flagged = r["flagged"]
            m3.metric("Needs inspection", f"{len(flagged):,}", border=True)

            # The plot is drawn from the same numbers as the count, so the two always agree.
            fig, ax = plt.subplots(figsize=(8, 2.2), facecolor="none")
            ax.set_facecolor("none")
            ax.bxp(
                [{"med": r["median"], "q1": r["q1"], "q3": r["q3"],
                  "whislo": r["inside"].min(), "whishi": r["inside"].max(), "fliers": flagged.to_numpy()}],
                orientation="horizontal", showfliers=True, widths=0.5,
                boxprops={"color": MAIN_COLOR}, medianprops={"color": MAIN_COLOR, "linewidth": 2},
                whiskerprops={"color": MAIN_COLOR}, capprops={"color": MAIN_COLOR},
                flierprops={"marker": "o", "markerfacecolor": FLAG_COLOR, "markeredgecolor": FLAG_COLOR, "markersize": 6},
            )
            ax.set_yticks([])
            ax.set_xlabel(inspect_col, color=MUTED_COLOR)
            ax.tick_params(axis="x", colors=MUTED_COLOR)
            ax.spines["bottom"].set_color(LINE_COLOR)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            fig.tight_layout()
            png = io.BytesIO()
            fig.savefig(png, format="png", dpi=200, transparent=True, bbox_inches="tight")
            plt.close(fig)
            st.image(png, width="stretch")
            st.caption("Amber points are values that need inspection. They may be correct, but they are far from the rest.")

            with st.container(key="boundaries"):
                b1, b2, b3, b4, b5 = st.columns(5)
                b1.metric("Q1", format_total(r["q1"]))
                b2.metric("Q3", format_total(r["q3"]))
                b3.metric("IQR", format_total(r["iqr"]))
                b4.metric("Lower boundary", format_total(r["lower"]))
                b5.metric("Upper boundary", format_total(r["upper"]))
            st.caption("Q1 and Q3 are the 25% and 75% quantiles (pandas default, linear interpolation). "
                       "IQR = Q3 − Q1. Lower = Q1 − 1.5 × IQR. Upper = Q3 + 1.5 × IQR. "
                       "A value needs inspection only when it is strictly below the lower boundary "
                       "or strictly above the upper boundary. Values exactly on a boundary are not flagged.")

            if flagged.empty:
                st.success(f"No value in \"{inspect_col}\" needs inspection.")
            else:
                st.subheader(f"Needs inspection: {len(flagged):,} row{'s' if len(flagged) > 1 else ''}")
                st.write("These are the original rows. Row numbers match the preview. Nothing has been changed or removed.")
                st.dataframe(df.loc[flagged.sort_values().index], width="stretch")

with compare_tab:
    # The file loaded above is the old version. A second upload here is the new version.
    # Both are read again as plain text so IDs keep leading zeros and "NA" stays text.
    st.write(f"Compare \"{uploaded.name}\" (old version) with a newer version of the same file. "
             "Both files are only read, never changed.")
    new_file = st.file_uploader("New version (CSV, same separator)", key="compare_file", max_upload_size=MAX_FILE_MB)
    if new_file is None:
        st.info("Upload the new version to see what changed: rows, columns, missing cells and, "
                "with an ID column, which records were added or removed.")
    elif not new_file.name.lower().endswith(".csv"):
        st.error(f"\"{new_file.name}\" is not a CSV file. Please upload a file whose name ends in .csv.")
    elif new_file.size > MAX_FILE_BYTES:
        st.error(f"This file is {new_file.size / 1024 / 1024:.1f} MB. The limit is {MAX_FILE_MB} MB per file.")
    else:
        try:
            old_text = read_text_table(uploaded.getvalue(), SEPARATORS[sep_label])
            new_text = read_text_table(new_file.getvalue(), SEPARATORS[sep_label])
        except ReadError as e:
            st.error(f"The new version could not be compared. {e}")
            old_text = new_text = None

        if new_text is not None:
            old_missing, new_missing = blank_cells(old_text), blank_cells(new_text)
            summary = [
                ("Rows", len(old_text), len(new_text)),
                ("Columns", len(old_text.columns), len(new_text.columns)),
                ("Missing cells", int(old_missing.values.sum()), int(new_missing.values.sum())),
            ]
            for card, (label, old, new) in zip(st.columns(3), summary):
                card.metric(label, f"{old:,} → {new:,}", delta=f"{new - old:+,}" if new != old else "0",
                            delta_color="off", border=True)
                card.caption(":orange[Changed]" if new != old else "Unchanged")
            st.caption(f"Old: \"{uploaded.name}\". New: \"{new_file.name}\". Counts use every row of both files. "
                       "For this comparison a cell is missing only when it is empty or only whitespace; "
                       "0, NA, null and similar text are kept as text.")

            old_cols, new_cols = list(old_text.columns), list(new_text.columns)
            added_cols = [c for c in new_cols if c not in old_cols]
            removed_cols = [c for c in old_cols if c not in new_cols]
            common_cols = [c for c in old_cols if c in new_cols]

            with st.expander(f"Columns: {len(added_cols):,} added, {len(removed_cols):,} removed"):
                if not added_cols and not removed_cols:
                    st.write("Unchanged: both versions have the same column names.")
                if added_cols:
                    st.markdown(":green[Added] " + ", ".join(f"`{c}`" for c in added_cols))
                if removed_cols:
                    st.markdown(":red[Removed] " + ", ".join(f"`{c}`" for c in removed_cols))

            with st.expander(f"Missing cells in the {len(common_cols):,} common columns"):
                if not common_cols:
                    st.write("The two versions have no column name in common, so there is nothing to align.")
                else:
                    old_by_col, new_by_col = old_missing[common_cols].sum(), new_missing[common_cols].sum()
                    delta = new_by_col - old_by_col
                    st.dataframe(pd.DataFrame({
                        "Column": common_cols,
                        "Old missing": old_by_col.values,
                        "New missing": new_by_col.values,
                        "Delta": delta.values,
                        "Status": ["Changed" if d else "Unchanged" for d in delta.values],
                    }), hide_index=True, width="stretch")

            with st.expander("Records added or removed", expanded=True):
                st.write("Choose a column that identifies each record, like an order number. It must be filled in "
                         "and unique in both versions. Records are matched by the exact text of that ID, "
                         "so reordering rows does not count as a change.")
                NO_ID = "No ID column chosen"
                id_key = f"compare_id:{uploaded.name}:{uploaded.size}:{new_file.name}:{new_file.size}:{sep_label}"
                id_col = st.selectbox("ID column (must exist in both versions)", [NO_ID] + common_cols, key=id_key)
                if id_col == NO_ID:
                    st.info("Record matching is not available until an ID column is chosen."
                            if common_cols else
                            "Record matching is not available: the two versions have no column in common.")
                else:
                    problems = [f"the old version has {p}" for p in [id_problem(old_text, id_col)] if p] + \
                               [f"the new version has {p}" for p in [id_problem(new_text, id_col)] if p]
                    if problems:
                        st.warning(f"Record matching is not available with \"{id_col}\": "
                                   + " and ".join(problems) + " in that column. Choose a column that is filled in "
                                   "and unique in both versions.")
                    else:
                        old_ids, new_ids = set(old_text[id_col]), set(new_text[id_col])
                        added = new_text[new_text[id_col].isin(new_ids - old_ids)]
                        removed = old_text[old_text[id_col].isin(old_ids - new_ids)]
                        a, r = st.columns(2)
                        a.metric("Added records", f"{len(added):,}", border=True)
                        r.metric("Removed records", f"{len(removed):,}", border=True)
                        if added.empty and removed.empty:
                            st.write(f"Unchanged: every \"{id_col}\" in the old version is in the new one, and the other way round.")
                        if not added.empty:
                            st.markdown(f":green[Added] {len(added):,} record{'s' if len(added) != 1 else ''} "
                                        f"in the new version. Row numbers are from the new file.")
                            st.dataframe(added, width="stretch")
                        if not removed.empty:
                            st.markdown(f":red[Removed] {len(removed):,} record{'s' if len(removed) != 1 else ''} "
                                        f"from the old version. Row numbers are from the old file.")
                            st.dataframe(removed, width="stretch")

with runs_tab:
    run_comparison_section()

with totals_tab:
    TOP_N = 20
    if not num_cols:
        st.info("The chart is not available: this file has no numeric column with at least one value. "
                "A column only counts as numeric when every value in it can be read as a number "
                "(missing values are allowed). The rest of the page still works.")
        st.stop()

    text_cols = [c for c in df.columns if c not in num_cols]
    c1, c2 = st.columns(2)
    cat_col = c1.selectbox("Group by (category column)", list(df.columns),
                           index=list(df.columns).index(text_cols[0]) if text_cols else 0)
    num_choices = [c for c in num_cols if c != cat_col]
    if not num_choices:
        st.info("The chart is not available: the only numeric column is the one chosen as the category. "
                "Choose a different category column.")
        st.stop()
    num_col = c2.selectbox("Add up (numeric column)", num_choices)

    totals = category_totals(df, cat_col, num_col)
    with_data = totals[totals["total"].notna()]
    y_label = f"Sum of {num_col}"

    if with_data.empty:
        st.info(f"The chart is not available: every value in \"{num_col}\" is missing for the rows that have a category.")
    else:
        chart_data = with_data.head(TOP_N)
        if len(with_data) > TOP_N:
            st.warning(f"The chart shows only the top {TOP_N} of {len(with_data):,} categories, by total. "
                       "The table below lists every category.")
        else:
            if len(with_data) == 1:
                st.caption("The chart shows the 1 category that has data.")
            else:
                st.caption(f"The chart shows all {len(with_data):,} categories that have data, sorted by total.")
        st.bar_chart(
            chart_data.reset_index().rename(columns={cat_col: "Category", "total": y_label}),
            x="Category", y=y_label, x_label=cat_col, y_label=y_label, sort=f"-{y_label}",
            color=MAIN_COLOR,
        )

    no_data = len(totals) - len(with_data)
    if no_data:
        st.caption(f"{no_data:,} categor{'y has' if no_data == 1 else 'ies have'} only missing values "
                   f"in \"{num_col}\". They are listed as \"No data\" and are not in the chart.")

    st.subheader(f"{y_label} for each {cat_col}")
    st.dataframe(
        pd.DataFrame({
            cat_col: totals.index,
            y_label: [format_total(v) for v in totals["total"]],
            "Rows": totals["rows"].values,
            f"Rows with missing {num_col}": totals["missing_numbers"].values,
        }),
        hide_index=True, width="stretch",
    )
    st.caption(f"Totals use the whole file. Missing values in \"{num_col}\" are skipped, not counted as 0. "
               "\"No data\" means every value in that group is missing.")

    missing_cat = df[cat_col].isna()
    if missing_cat.any():
        rest = df.loc[missing_cat, num_col]
        st.info(f"{int(missing_cat.sum()):,} row(s) have a missing \"{cat_col}\" and are not in the chart or table above. "
                f"Their {y_label.lower()} is {format_total(rest.sum(min_count=1))} "
                f"(missing \"{num_col}\" in {int(rest.isna().sum()):,} of these rows).", icon=":material/info:")
