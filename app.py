# =============================================================================
# J.A.W.I.R. — Joint Agent Workflow Integrity & Reliability
# Team STEIJanggal | Hackathon MVP
# =============================================================================
# Architecture: Actor-Critic dual-agent orchestration on IBM watsonx.ai
#   ASEP  → Actor  : Automated Script Engineering & Parser
#   CECEP → Critic : Contract-schema Evaluation & Critical Error Preventer
# =============================================================================

import base64
import os
import re

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Environment / Credentials
# ---------------------------------------------------------------------------

load_dotenv()  # Pull variables from .env into os.environ at startup


def load_credentials() -> dict:
    """Return watsonx credentials from environment variables."""
    return {
        "api_key": os.getenv("IBM_CLOUD_API_KEY", ""),
        "project_id": os.getenv("WATSONX_PROJECT_ID", ""),
        "url": os.getenv("WATSONX_URL", "https://jp-tok.ml.cloud.ibm.com"),
    }


CREDS = load_credentials()

# ---------------------------------------------------------------------------
# 2. Governance Contracts (static, dataset-agnostic baseline)
# ---------------------------------------------------------------------------

SCHEMA_EXPECTATIONS = (
    "STRICT RULES:\n\n"
    "1. All numeric columns found in this dataset are critical features for downstream "
    "Machine Learning models and MUST NOT be dropped.\n\n"
    "2. Data types must not be cast to incompatible formats (e.g., numeric to string) "
    "without explicit fallback logic.\n\n"
    "3. Primary identifier columns (if any) must remain intact.\n\n"
    "4. Column names are strictly immutable. Renaming ANY existing column is strictly forbidden."
)

# LINEAGE_MAP is rendered as custom HTML in the sidebar — this string is kept
# as a fallback for the CECEP prompt builder which still references it as plain text.
LINEAGE_MAP = (
    "Uploaded Dataset -> Target ETL Script -> Global Data Warehouse "
    "-> Downstream ML Prediction Pipelines & Executive Dashboards"
)


_LINEAGE_MAP_HTML = """
<div style="font-size:0.88rem;line-height:2;">
  <div><span style="color:#39FF14;font-weight:700;">-&gt;</span> Uploaded Dataset</div>
  <div><span style="color:#39FF14;font-weight:700;">-&gt;</span> Target ETL Script</div>
  <div><span style="color:#39FF14;font-weight:700;">-&gt;</span> Global Data Warehouse</div>
  <div><span style="color:#39FF14;font-weight:700;">-&gt;</span> Downstream ML Prediction Pipelines<br>&nbsp;&nbsp;&nbsp;&nbsp; &amp; Executive Dashboards</div>
</div>
"""

# ---------------------------------------------------------------------------
# 3. Watsonx API Helper
# ---------------------------------------------------------------------------

# Model IDs — both agents use the same available model to prevent WMLClientError
MODEL_ASEP = "meta-llama/llama-3-3-70b-instruct"
MODEL_CECEP = "meta-llama/llama-3-3-70b-instruct"

# Watsonx generation parameters
_GEN_PARAMS = {
    "max_new_tokens": 350,          # ASEP default — enough for PR note + concise code
    "temperature": 0.2,
    "top_p": 0.9,
    "repetition_penalty": 1.25,     # prevents looping / repeated paragraphs
}

# CECEP gets a tighter token budget — 3-point audit only
_CECEP_MAX_TOKENS = 150


def call_watsonx(prompt: str, model_id: str, max_new_tokens: int | None = None) -> str:
    """
    Sends `prompt` to the specified watsonx model and returns the generated text.
    `max_new_tokens` overrides the default limit when supplied (used for CECEP).
    Wraps the API call in a try-except so a bad key / timeout never crashes the UI.
    """
    try:
        # Import here so the app still loads even if the package isn't installed yet
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        credentials = {
            "url": CREDS["url"],
            "apikey": CREDS["api_key"],
        }

        effective_max_tokens = max_new_tokens if max_new_tokens is not None else _GEN_PARAMS["max_new_tokens"]

        model = ModelInference(
            model_id=model_id,
            credentials=credentials,
            project_id=CREDS["project_id"],
            params={
                GenParams.MAX_NEW_TOKENS: effective_max_tokens,
                GenParams.TEMPERATURE: _GEN_PARAMS["temperature"],
                GenParams.TOP_P: _GEN_PARAMS["top_p"],
                GenParams.REPETITION_PENALTY: _GEN_PARAMS["repetition_penalty"],
            },
        )

        response = model.generate_text(prompt=prompt)
        return response

    except Exception as exc:  # noqa: BLE001
        # Surface a clean error instead of a raw traceback
        error_type = type(exc).__name__
        return (
            f"[WATSONX API ERROR] {error_type}: {exc}\n\n"
            "Check your IBM_CLOUD_API_KEY, WATSONX_PROJECT_ID, and WATSONX_URL in .env."
        )


# ---------------------------------------------------------------------------
# 4. Data Profiling Helper
# ---------------------------------------------------------------------------


def process_uploaded_file(uploaded_file) -> tuple[pd.DataFrame, str]:
    """
    Reads an uploaded CSV into a DataFrame and builds the ACTIVE_DATA_CONTEXT
    string (column names + dtypes + first 2 rows as markdown).
    Capped at 2 sample rows to minimise token consumption.
    Returns (df, context_string).
    """
    df = pd.read_csv(uploaded_file)

    col_info_lines = [f"  - {col} ({str(dtype)})" for col, dtype in df.dtypes.items()]
    col_info = "\n".join(col_info_lines)

    # 2 rows only — keeps prompt lean
    sample_md = df.head(2).to_markdown(index=False)

    context = (
        "DATA CONTEXT:\n"
        f"Columns & dtypes:\n{col_info}\n\n"
        f"Sample (2 rows):\n{sample_md}"
    )
    return df, context


# ---------------------------------------------------------------------------
# 5. Prompt Builders
# ---------------------------------------------------------------------------


def build_asep_prompt(user_request: str, data_context: str) -> str:
    """Construct the full prompt for ASEP."""
    return (
        "You are ASEP (Automated Script Engineering & Parser), a highly efficient Data Engineer AI. Your task is to generate a pandas ETL script based on the user request.\n\n"
        "CRITICAL RULES:\n"
        "1. You MUST wrap all transformation logic inside a main function explicitly named `def etl(df):` and it MUST return `df`.\n"
        "2. Write pure pandas code only. NO SQL. NO markdown explanations outside the code block.\n"
        "3. ZERO conversational filler. Do not write any introduction or greeting.\n"
        "4. ZERO-TRUST COMPLIANCE: DO NOT use `.drop()`, `.dropna()`, or `.rename()`. NEVER overwrite existing column values and NEVER delete rows. If the user asks to modify or rename something, create a NEW column instead.\n\n"
        "MANDATORY OUTPUT FORMAT:\n"
        "Line 1: A single-sentence PR justification explaining the change.\n"
        "Line 2 onwards: The Python code block enclosed in ```python and ``` tags.\n\n"
        "=== DATA CONTEXT ===\n"
        f"{data_context}\n\n"
        "=== USER REQUEST ===\n"
        f"{user_request}\n\n"
        "YOUR OUTPUT (Strictly follow the MANDATORY OUTPUT FORMAT):"
    )


def build_cecep_prompt(asep_output: str) -> str:
    """Construct the concise prompt for CECEP (Critic), isolating ASEP's output."""
    return (
        "You are CECEP (Contract Evaluation & Critical Error Preventer), an aggressive and uncompromising Data Governance Inspector AI.\n"
        "Your sole task is to audit the provided Python pandas code (ASEP OUTPUT) against the SCHEMA_EXPECTATIONS and LINEAGE_MAP.\n\n"
        "CRITICAL AUDIT RULES (ZERO-TRUST POLICY):\n"
        "1. If you see `.drop()`, `.dropna()`, or `.rename()` anywhere in the code -> IMMEDIATELY [STATUS: REJECTED].\n"
        "2. If the code alters existing rows (filtering) or overwrites the values of existing numeric columns -> [STATUS: REJECTED].\n"
        "3. If alphanumeric string IDs are cast to float -> [STATUS: REJECTED].\n"
        "4. Column names are strictly immutable. Renaming ANY existing column is strictly forbidden and MUST result in a [STATUS: REJECTED] due to downstream lineage breakage.\n"
        "5. If the code safely appends NEW columns or modifies text formats safely without breaking the schema -> [STATUS: APPROVED].\n\n"
        "MANDATORY OUTPUT FORMAT (You MUST choose one of these exact templates):\n\n"
        "TEMPLATE 1: IF REJECTED\n"
        "[STATUS: REJECTED]\n\n"
        "### 📄 Violating Code Snippet\n"
        "```python\n"
        "# Insert specific violating code lines here\n"
        "```\n\n"
        "### 📋 Audit Findings\n"
        "- Risk Level: CRITICAL\n"
        "- Reason: [1 sentence explaining the violation]\n"
        "- Required Action: [1 sentence on how to fix it]\n\n"
        "TEMPLATE 2: IF APPROVED\n"
        "[STATUS: APPROVED]\n\n"
        "### 📄 Violating Code Snippet\n"
        "No schema violations detected.\n\n"
        "### 📋 Audit Findings\n"
        "- Risk Level: NONE\n"
        "- Reason: Code is fully compliant with schema and lineage.\n"
        "- Required Action: None\n\n"
        "=== CONTEXT FOR AUDIT ===\n"
        f"SCHEMA_EXPECTATIONS:\n{SCHEMA_EXPECTATIONS}\n\n"
        f"LINEAGE_MAP:\n{LINEAGE_MAP}\n\n"
        "=== ASEP OUTPUT TO REVIEW ===\n"
        f"{asep_output}\n\n"
        "FINAL INSTRUCTION: Evaluate the ASEP OUTPUT above. You MUST start your response immediately with either '[STATUS: APPROVED]' or '[STATUS: REJECTED]'. Do not output any JSON, arrays, or filler words before the status tag. Begin your audit now:"
    )


# ---------------------------------------------------------------------------
# 6. UI Asset Helpers
# ---------------------------------------------------------------------------


def _load_logo_b64(path: str) -> str:
    """
    Read an image file and return its base64-encoded data URI string.
    Resolves the path relative to the directory containing app.py using
    os.path.abspath so the file is always found regardless of the working
    directory Streamlit is launched from.
    Returns an empty string if the file is missing or unreadable.
    """
    try:
        abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        with open(abs_path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("utf-8")
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
        return f"data:{mime};base64,{encoded}"
    except (FileNotFoundError, OSError):
        return ""


def _agent_heading_html(
    light_path: str,
    dark_path: str,
    short_name: str,
    full_name: str,
) -> str:
    """
    Build a self-contained HTML block for an agent header:
      - Line 1: a <span> logo using CSS background-image (pure CSS dark/light swap
                via @media prefers-color-scheme) placed inline before the short name.
      - Line 2: full agent name as a muted <p> subtitle.

    The logo uses background-image so transparency is preserved without <img> artefacts.
    If both asset files are missing the logo span is omitted gracefully.
    Each call generates unique CSS class names keyed on short_name to avoid collisions.
    """
    light_src = _load_logo_b64(light_path)
    dark_src = _load_logo_b64(dark_path)

    # INVERTED assignment: _dark filenames contain white strokes (visible on dark BG)
    #                      _light filenames contain black strokes (visible on light BG)
    # Default (light theme) -> use dark_src (black strokes on white page)
    # @media dark           -> use light_src (white strokes on dark page)
    _default = dark_src or light_src   # shown in light mode
    _on_dark  = light_src or dark_src  # shown in dark mode

    # Build a safe CSS identifier from the short name (e.g. "asep", "cecep")
    css_key = re.sub(r"[^a-z0-9]", "", short_name.lower())

    if _default:
        logo_span = f'<span class="jawir-agent-logo jawir-logo-{css_key}"></span>'
        logo_css = f"""
.jawir-agent-logo {{
  display: inline-block;
  width: 1.5em;
  height: 1.5em;
  background-size: contain;
  background-repeat: no-repeat;
  background-position: center;
  vertical-align: middle;
  margin-right: 8px;
  border: none;
  background-color: transparent;
  flex-shrink: 0;
}}
.jawir-logo-{css_key} {{
  background-image: url('{_default}');
}}
@media (prefers-color-scheme: dark) {{
  .jawir-logo-{css_key} {{
    background-image: url('{_on_dark}');
  }}
}}"""
    else:
        logo_span = ""
        logo_css = ""

    style_block = f"<style>{logo_css}</style>" if logo_css else ""

    # Tight single-container layout: both lines share one wrapper with line-height 1.0
    return (
        f'{style_block}'
        f'<div style="line-height:1.2; margin:0 0 20px 0; padding:0; color: var(--text-color);">'
        f'<div style="font-size:1.5em; font-weight:bold; display:flex; align-items:center;">{logo_span}{short_name}</div>'
        f'<div style="font-size:1.5em; font-weight:bold; margin-top:5px; color: var(--text-color);">{full_name}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# 7. Response Parsers & Code Cleaner
# ---------------------------------------------------------------------------


def _clean_python_code(raw_code: str) -> str:
    """
    Extract pure executable Python from an LLM response string.

    Priority order:
    1. Fenced ```python ... ``` block  → return inner content only.
    2. Fenced generic ``` ... ``` block → return inner content only.
    3. No fences → drop narrative lines that appear before the first
       Python keyword (import / def / class / #) and after the last
       code line, then strip any remaining stray backtick lines.
    """
    # 1. Explicit python fence
    match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Generic fence
    match = re.search(r"```(.*?)```", raw_code, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 3. No fences — find the first line that looks like real Python code
    #    (starts with import / from / def / class / # / a word followed by '=')
    code_start_pattern = re.compile(
        r"^(import |from |def |class |#|[A-Za-z_]\w*\s*[=(])", re.MULTILINE
    )
    first_code = code_start_pattern.search(raw_code)
    if first_code:
        trimmed = raw_code[first_code.start():]
    else:
        trimmed = raw_code

    # Remove stray backtick-only lines and trailing whitespace
    cleaned_lines = [
        line for line in trimmed.splitlines()
        if not line.strip().startswith("```")
    ]
    return "\n".join(cleaned_lines).strip()


def parse_asep_response(raw: str) -> tuple[str, str]:
    """
    Split ASEP's raw text into (pr_justification, python_code).
    Extracts the first ```python ... ``` block; everything before it is the PR note.
    The returned python_code is the raw extracted block (not yet cleaned for exec).
    """
    pattern = r"```python\s*(.*?)```"
    match = re.search(pattern, raw, re.DOTALL)
    if match:
        code = match.group(1).strip()
        justification = raw[: match.start()].strip()
    else:
        code = raw.strip()
        justification = "No PR justification found in ASEP's response."
    return justification, code


# ---------------------------------------------------------------------------
# 7. Split-Screen Renderer
# ---------------------------------------------------------------------------


# Dangerous pandas operations that indicate a schema violation risk
_DANGEROUS_KEYWORDS = [".drop(", ".astype(", "del ", ".pop(", "dropna("]


def _extract_violating_snippet(cecep_raw: str) -> str:
    """
    Pull the content under '### 📄 Violating Code Snippet' from CECEP's response.
    First tries a fenced ```python ... ``` block inside that section.
    Falls back to the raw text of the section if no fenced block is found.
    Returns an empty string if the section is absent entirely.
    """
    # Locate the section heading
    section_pattern = r"###\s*📄\s*Violating Code Snippet\s*\n(.*?)(?=\n###|\Z)"
    section_match = re.search(section_pattern, cecep_raw, re.DOTALL | re.IGNORECASE)
    if not section_match:
        return ""
    section_body = section_match.group(1).strip()

    # Prefer the content inside a fenced code block within that section
    fence_match = re.search(r"```(?:python)?\s*(.*?)```", section_body, re.DOTALL)
    if fence_match:
        extracted = fence_match.group(1).strip()
        return extracted if extracted else section_body

    return section_body


def _fallback_dangerous_lines(python_code: str) -> str:
    """
    Scan ASEP's Python code line-by-line for known dangerous operations.
    Returns the matching lines joined by newline, or an empty string if none found.
    """
    flagged = [
        line.rstrip()
        for line in python_code.splitlines()
        if any(kw in line for kw in _DANGEROUS_KEYWORDS)
    ]
    return "\n".join(flagged)


# Variable names ASEP might assign the transformed DataFrame to
_RESULT_CANDIDATES = ["processed_df", "updated_df", "result_df", "output_df", "df_out", "df"]


def _execute_asep_code(python_code: str, df: pd.DataFrame) -> tuple[pd.DataFrame | None, str]:
    clean_code = _clean_python_code(python_code)

    safe_builtins = {
        "print": print, "range": range, "len": len, "list": list,
        "dict": dict, "str": str, "int": int, "float": float,
        "bool": bool, "enumerate": enumerate, "zip": zip,
        "min": min, "max": max, "sum": sum, "abs": abs,
        "isinstance": isinstance, "type": type, "round": round,
        "__import__": __import__
    }

    exec_globals = {"pd": pd, "__builtins__": safe_builtins}
    local_ns: dict = {"df": df.copy(), "pd": pd}

    try:
        exec(clean_code, exec_globals, local_ns)

        # Prioritas 1: Cari fungsi buatan ASEP duluan biar data CSV asli yang diproses
        for name, obj in local_ns.items():
            if callable(obj) and name not in ("pd",):
                try:
                    result = obj(df.copy())
                    if isinstance(result, pd.DataFrame):
                        return result, ""
                except Exception:
                    pass

        # Prioritas 2: Kalau fungsi gagal, baru cari dari variabel
        for candidate in _RESULT_CANDIDATES:
            result = local_ns.get(candidate)
            if isinstance(result, pd.DataFrame):
                return result, ""

        return None, "Gagal menemukan hasil eksekusi dataframe."

    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def render_split_screen(asep_raw: str, cecep_raw: str, df: pd.DataFrame | None = None) -> None:
    """
    Render the dual-agent results in a side-by-side 2-column layout.
    Each column is wrapped in st.container(border=True) for visual clarity.
    If CECEP approves and `df` is provided, executes ASEP's code and shows
    an output preview + CSV download button below the review columns.
    """
    st.markdown("---")
    st.subheader("J.A.W.I.R. Dual-Agent Review Results")

    col1, col2 = st.columns(2)

    # Parse ASEP output once — shared by both columns
    pr_justification, python_code = parse_asep_response(asep_raw)

    # Determine CECEP verdict once for reuse
    is_rejected = "[STATUS: REJECTED]" in cecep_raw
    is_approved = "[STATUS: APPROVED]" in cecep_raw

    # Strip status tags and collapse excessive blank lines from CECEP's raw output
    audit_text = re.sub(
        r"\n{3,}", "\n\n",
        cecep_raw
        .replace("[STATUS: REJECTED]", "")
        .replace("[STATUS: APPROVED]", "")
    ).strip()

    # Extract only the violating snippet section from CECEP's response
    violating_snippet = _extract_violating_snippet(cecep_raw)

    # --- Left Column: ASEP (Actor) ---
    with col1:
        with st.container(border=True):
            asep_heading = _agent_heading_html(
                "assets/asep_light.png", "assets/asep_dark.png",
                short_name="ASEP",
                full_name="Automated Script Engineering & Parser",
            )
            st.markdown(asep_heading, unsafe_allow_html=True)

            # PR Justification block
            st.info(pr_justification if pr_justification else "*(No justification returned)*")

            # Generated code block
            if python_code:
                st.code(python_code, language="python")
            else:
                st.warning("ASEP did not return a valid Python code block.")

    # --- Right Column: CECEP (Critic) ---
    with col2:
        with st.container(border=True):
            cecep_heading = _agent_heading_html(
                "assets/cecep_light.png", "assets/cecep_dark.png",
                short_name="CECEP",
                full_name="Contract Evaluation & Error Preventer",
            )
            st.markdown(cecep_heading, unsafe_allow_html=True)

            # 1. Status banner — formal, enterprise style
            if is_rejected:
                st.error("STATUS: REJECTED | SCHEMA VIOLATION DETECTED")
            elif is_approved:
                st.success("STATUS: APPROVED | SAFE FOR PRODUCTION")
            else:
                st.warning("CECEP returned an unrecognised status. Raw output shown below.")

            # 2. Flagged Code Snippet
            st.caption("Flagged Code Snippet:")
            if is_approved:
                st.code("No schema violations detected.", language="text")
            else:
                # Use CECEP's isolated snippet; fall back to auto-detected dangerous lines
                display_snippet = violating_snippet
                if not display_snippet:
                    display_snippet = _fallback_dangerous_lines(python_code)
                if not display_snippet:
                    display_snippet = "(No dangerous lines detected by static analysis)"
                st.code(display_snippet, language="python")

            # 3. CECEP's full audit response — copyable text block
            st.caption("CECEP Responses:")
            st.code(audit_text if audit_text else "(No audit text returned)", language="text")

    # --- Post-review: Execute & Download (APPROVED path only) ---
    st.markdown("---")
    if is_approved and df is not None:
        # Attempt to exec ASEP's code; fall back gracefully on any failure
        exec_succeeded = False
        updated_df = df.copy()  # safe default — always a valid DataFrame

        if python_code:
            result, exec_error = _execute_asep_code(python_code, df)
            if result is not None:
                updated_df = result
                exec_succeeded = True
            # On failure: updated_df stays as df.copy() — no red error banner shown

        with st.container(border=True):
            st.subheader("Applied Modification Preview")

            if exec_succeeded:
                st.success("Script executed successfully against the uploaded dataset.")
            else:
                # Graceful fallback — inform without alarming
                st.info(
                    "Script execution encountered an issue. "
                    "Displaying the original dataset as a safe fallback."
                )

            # Preview and download are always rendered when APPROVED
            st.dataframe(updated_df.head(), use_container_width=True)

            # Unified export gate: banner + download button share one themed container
            with st.container():
                st.markdown(
                    """
<style>
.jawir-export-banner {
  background-color: #0a1f0a;
  border: 1px solid #39FF14;
  border-radius: 8px 8px 0 0;
  padding: 12px 16px;
  margin-top: 12px;
  margin-bottom: 0;
  font-family: 'Courier New', Courier, monospace;
}
.jawir-export-prefix {
  color: #39FF14;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  display: block;
  margin-bottom: 2px;
  opacity: 0.7;
}
.jawir-export-msg {
  color: #39FF14;
  font-size: 0.88rem;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
/* Target the download button rendered immediately after this banner */
div[data-testid="stDownloadButton"] > button {
  border: 1px solid #1b5e20 !important;
  border-radius: 8px !important;
  color: #ffffff !important;
  background-color: #2e7d32 !important;
  font-family: 'Courier New', Courier, monospace !important;
  font-weight: 600 !important;
  width: 100%;
  margin-top: 10px;
  transition: all 0.3s ease;
}
div[data-testid="stDownloadButton"] > button:hover {
  background-color: #1b5e20 !important;
  border-color: #39FF14 !important;
}
</style>
<div class="jawir-export-banner">
  <span class="jawir-export-prefix">[ JAWIR GOVERNANCE GATE: PASSED ]</span>
  <span class="jawir-export-msg">AI-Reviewed Output Ready for Export</span>
</div>
""",
                    unsafe_allow_html=True,
                )
                st.download_button(
                    label="Download AI-Reviewed Dataset (.csv)",
                    data=updated_df.to_csv(index=False).encode("utf-8"),
                    file_name="updated_dataset.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    elif is_rejected:
        # Governance gate blocked execution — do not run the code
        st.warning("Script execution blocked due to governance violations.")


# ---------------------------------------------------------------------------
# 8. Main Application Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    # ---- Page Config --------------------------------------------------------
    st.set_page_config(
        page_title="J.A.W.I.R. — Dual-Agent ETL Review",
        page_icon="[J]",
        layout="wide",
    )

    # ---- CSS: fix canvas scroll & column overflow ---------------------------
    st.markdown(
        """
        <style>
          .main { overflow-y: auto !important; }
          div[data-testid="stColumn"] { overflow: visible !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---- Session State Bootstrap --------------------------------------------
    # Initialise keys on first run to prevent KeyError & avoid re-firing API calls
    if "is_processed" not in st.session_state:
        st.session_state.is_processed = False
    if "asep_result" not in st.session_state:
        st.session_state.asep_result = ""
    if "cecep_result" not in st.session_state:
        st.session_state.cecep_result = ""
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None

    # ---- Sidebar ------------------------------------------------------------
    with st.sidebar:
        # Adaptive sidebar logo — no leading whitespace on HTML lines to avoid
        # Streamlit treating indented content as a Markdown code block.
        _img_white = _load_logo_b64("assets/jawir_logo_white.png")  # white strokes -> dark mode
        _img_black = _load_logo_b64("assets/jawir_logo_black.png")  # black strokes -> light mode

        _sidebar_html = f"""<style>
.jawir-logo-light {{ display: block; width: 100%; height: auto; }}
.jawir-logo-dark {{ display: none; width: 100%; height: auto; }}
@media (prefers-color-scheme: dark) {{
.jawir-logo-light {{ display: none !important; }}
.jawir-logo-dark {{ display: block !important; }}
}}
.team-title {{ font-size: 1.2rem; font-weight: 700; margin-top: -45px; margin-bottom: 16px; color: inherit; display: block; }}
.stei-blue {{ color: #0062FF; font-weight: 800; }}
</style>
<div><img class="jawir-logo-light" src="{_img_black}" /><img class="jawir-logo-dark" src="{_img_white}" /><div class="team-title">Team <span class="stei-blue">STEI</span>Janggal</div></div>"""

        st.sidebar.markdown(_sidebar_html, unsafe_allow_html=True)

        st.divider()

        # --- Box 1 (static): System always online once the app is running ----
        st.markdown(
    """
<div style="background-color:#1b5e20; border: 1px solid #39FF14; border-radius:8px; padding:10px 14px; margin-bottom:10px;">
  <span style="font-size:0.82rem; font-weight:700; color:#39FF14; letter-spacing:0.02em;">
    System Status: Securely Connected to IBM watsonx.ai Enterprise Network
  </span>
</div>
""",
    unsafe_allow_html=True,
)

        # --- Box 2 (dynamic): probe Watsonx API with a lightweight credentials check
        # Attempt a minimal API initialisation; success = connected, exception = disconnected
        watsonx_ok = False
        if CREDS["api_key"] and CREDS["project_id"]:
            try:
                from ibm_watsonx_ai.foundation_models import ModelInference  # noqa: PLC0415
                ModelInference(
                    model_id=MODEL_CECEP,
                    credentials={"url": CREDS["url"], "apikey": CREDS["api_key"]},
                    project_id=CREDS["project_id"],
                )
                watsonx_ok = True
            except Exception:  # noqa: BLE001
                watsonx_ok = False

        api_bg   = "transparent" if watsonx_ok else "transparent"
        api_text = "#39FF14" if watsonx_ok else "#ff5252"
        api_label = "Watsonx API: Connected" if watsonx_ok else "Watsonx API: Disconnected"

        st.markdown(
            f"""
<style>
.jawir-dot-tegas {{
  display: inline-block;
  width: 12px; height: 12px;
  border-radius: 50%;
  background-color: {api_text};
  vertical-align: middle;
  margin-right: 10px;
}}
</style>
<div style="background-color:{api_bg}; border: 1px solid {api_text}; border-radius:8px; padding:10px 14px; margin-bottom:4px;">
  <span class="jawir-dot-tegas"></span>
  <span style="font-size:0.82rem;font-weight:700;color:{api_text};letter-spacing:0.02em;">
    {api_label}
  </span>
</div>
""",
            unsafe_allow_html=True,
        )

        st.divider()

        # --- Governance contract panels — card-styled expanders --------------
        st.markdown("#### Active Data Governance")

        # Shared card wrapper injected once; expander content floats inside it
        st.markdown(
            """
<style>
.jawir-gov-card {
  background-color: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.10);
  border-radius: 8px;
  padding: 15px;
  margin-bottom: 8px;
  font-size: 0.88rem;
  line-height: 1.7;
  transition: all 0.3s ease-in-out;
}
.jawir-gov-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 6px 12px rgba(57, 255, 20, 0.15);
  border-color: #39FF14;
}
</style>
""",
            unsafe_allow_html=True,
        )

        with st.expander("Schema Rules & Expectations", expanded=False):
            st.markdown(
                f'<div class="jawir-gov-card">{SCHEMA_EXPECTATIONS}</div>',
                unsafe_allow_html=True,
            )
        with st.expander("Data Lineage Map", expanded=False):
            st.markdown(
                f'<div class="jawir-gov-card">{_LINEAGE_MAP_HTML}</div>',
                unsafe_allow_html=True,
            )

    # ---- Main Header --------------------------------------------------------
    st.title("J.A.W.I.R")
    st.markdown(
        "**Joint Agent Workflow Integrity & Reliability** | "
        "Actor-Critic Dual-Agent ETL Code Review powered by IBM watsonx.ai"
    )
    st.markdown(
        "> Eliminate silent breaking changes in enterprise data pipelines "
        "before they reach production."
    )
    st.markdown("---")

    # ---- Step 1: CSV Uploader & Data Profiling ------------------------------
    st.subheader("Step 1 : Upload Target Dataset")
    uploaded_file = st.file_uploader(
        "Upload your target dataset (.csv)", type=["csv"]
    )

    df = None
    active_data_context = ""

    if uploaded_file is not None:
        # Process and profile the uploaded file; persist df for post-review execution
        df, active_data_context = process_uploaded_file(uploaded_file)
        st.session_state.uploaded_df = df

        st.success(f"{uploaded_file.name} loaded — {df.shape[0]} rows x {df.shape[1]} columns")
        st.markdown("**Data Preview:**")
        st.dataframe(df.head(5), use_container_width=True)

        with st.expander("Full Column Profile", expanded=False):
            profile = pd.DataFrame(
                {"dtype": df.dtypes, "null_count": df.isnull().sum(), "unique": df.nunique()}
            )
            st.dataframe(profile, use_container_width=True)

    # ---- Step 2: Request Input — only visible after upload ------------------
    if df is not None:
        st.markdown("---")
        st.subheader("Step 2 : Define Your ETL Modification Request")
        user_request = st.text_area(
            "Dataset loaded. What modifications do you want to apply to this data structure or ETL script?",
            height=140,
            placeholder=(
                "e.g. 'Drop all rows where age < 18 and normalise the salary column to z-scores.'"
            ),
        )

        execute_clicked = st.button(
            "Execute J.A.W.I.R. Dual-Agent Review",
            type="primary",
            use_container_width=True,
        )

        # ---- Input Validation -----------------------------------------------
        if execute_clicked:
            if not user_request.strip():
                st.warning(
                    "Please upload a CSV dataset and enter your modification request first."
                )
                st.stop()

            # ---- Sequential Dual-Agent Execution ----------------------------
            # API is only called on button press — session_state prevents re-fire
            with st.spinner("J.A.W.I.R. agents are analyzing the request via Watsonx..."):

                # Step 1 → Build & call ASEP (Actor)
                asep_prompt = build_asep_prompt(user_request, active_data_context)
                asep_raw = call_watsonx(asep_prompt, MODEL_ASEP)

                # Step 2 → Save ASEP output to session state
                st.session_state.asep_result = asep_raw

                # Step 3 → Guard: if ASEP failed, abort before calling CECEP
                # This prevents burning CECEP tokens on an error message
                if "[WATSONX API ERROR]" in asep_raw:
                    st.session_state.cecep_result = ""
                    st.session_state.is_processed = True
                    # Surface the ASEP error immediately; CECEP is not called
                    st.error(f"ASEP API call failed. Review error details in the results panel below.")
                else:
                    # Step 4 → Pass ASEP output exclusively to CECEP (Critic)
                    # max_new_tokens capped at 150 to enforce concise 3-point audit
                    cecep_prompt = build_cecep_prompt(asep_raw)
                    cecep_raw = call_watsonx(cecep_prompt, MODEL_CECEP, max_new_tokens=_CECEP_MAX_TOKENS)

                    # Step 5 → Save CECEP output to session state
                    st.session_state.cecep_result = cecep_raw

                # Step 6 → Mark as processed so layout persists on rerender
                st.session_state.is_processed = True

        # ---- Render Split-Screen once results are available -----------------
        if st.session_state.is_processed:
            render_split_screen(
                st.session_state.asep_result,
                st.session_state.cecep_result,
                df=st.session_state.uploaded_df,
            )

    else:
        # Friendly placeholder when no file has been uploaded yet
        st.info(
            "Upload a CSV file above to activate the J.A.W.I.R. dual-agent review pipeline."
        )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
