# =============================================================================
# J.A.W.I.R. — Joint Agent Workflow Integrity & Reliability
# Team STEIJanggal | Hackathon MVP
# =============================================================================
# Architecture: Actor-Critic dual-agent orchestration on IBM watsonx.ai
#   ASEP  → Actor  : Automated Script Engineering & Parser
#   CECEP → Critic : Contract-schema Evaluation & Critical Error Preventer
# =============================================================================

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
    "STRICT RULES: "
    "1. All numeric columns found in this dataset are critical features for downstream "
    "Machine Learning models and MUST NOT be dropped. "
    "2. Data types must not be cast to incompatible formats (e.g., numeric to string) "
    "without explicit fallback logic. "
    "3. Primary identifier columns (if any) must remain intact."
)

LINEAGE_MAP = (
    "[Uploaded Dataset] -> [Target ETL Script] -> [Global Data Warehouse] "
    "-> [Downstream ML Prediction Pipelines & Executive Dashboards]"
)

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
    """Construct the full prompt for ASEP (Actor). Kept concise to conserve tokens."""
    return (
        "You are ASEP, a Data Engineer AI. Given the dataset context below, "
        "generate a pandas ETL script for the user's request.\n\n"
        "Response format (no filler text):\n"
        "1. One-sentence PR justification.\n"
        "2. Python code block in ```python ``` tags.\n\n"
        f"{data_context}\n\n"
        f"Request: {user_request}"
    )


def build_cecep_prompt(asep_output: str) -> str:
    """Construct the concise prompt for CECEP (Critic), isolating ASEP's output."""
    return (
        "You are CECEP, a Data Governance Inspector. "
        "Review the ASEP Python pandas code against SCHEMA_EXPECTATIONS and LINEAGE_MAP.\n\n"
        "STRICT RULE: DO NOT OUTPUT ANY SQL CODE OR SQL EQUIVALENT. "
        "Focus ONLY on Python pandas.\n\n"
        "Rules: REJECT if numeric columns are dropped, types are changed recklessly, "
        "or schema rules are violated. APPROVE if code is compliant.\n\n"
        "MANDATORY OUTPUT FORMAT (follow exactly, no deviations):\n\n"
        "If REJECTED:\n"
        "[STATUS: REJECTED]\n\n"
        "### 📄 Violating Code Snippet\n"
        "```python\n"
        "<Copy ONLY 1-2 lines of the specific Python pandas code that causes the violation. "
        "Do NOT rewrite or paraphrase. Do NOT output SQL.>\n"
        "```\n\n"
        "### 📋 Audit Findings\n"
        "- Risk Level: [CRITICAL / HIGH / NONE]\n"
        "- Reason: [1 sentence]\n"
        "- Required Action: [1 sentence]\n\n"
        "If APPROVED:\n"
        "[STATUS: APPROVED]\n\n"
        "### 📄 Violating Code Snippet\n"
        "No schema violations detected.\n\n"
        "### 📋 Audit Findings\n"
        "- Risk Level: NONE\n"
        "- Reason: Code is fully compliant.\n"
        "- Required Action: None\n\n"
        "RULE: First characters MUST be the status tag. No filler. No SQL.\n\n"
        f"SCHEMA_EXPECTATIONS: {SCHEMA_EXPECTATIONS}\n"
        f"LINEAGE_MAP: {LINEAGE_MAP}\n\n"
        f"ASEP OUTPUT:\n{asep_output}"
    )


# ---------------------------------------------------------------------------
# 6. Response Parsers & Code Cleaner
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
    """
    Extract pure Python, exec it safely, and return the transformed DataFrame.

    Execution strategy:
    1. Run _clean_python_code() to strip all markdown artefacts.
    2. Seed the exec namespace with `df` (copy), `pd`, and common builtins.
    3. Look for a resulting DataFrame in _RESULT_CANDIDATES (named variables).
    4. If none found, scan for any callable defined in the code (def ...) and
       call it with df.copy() — handles ETL scripts that wrap logic in a function.
    5. Return (updated_df, "") on success or (None, error_message) on failure.
    """
    clean_code = _clean_python_code(python_code)

    # Provide a safe subset of builtins needed for typical pandas scripts
    safe_builtins = {
        "print": print, "range": range, "len": len, "list": list,
        "dict": dict, "str": str, "int": int, "float": float,
        "bool": bool, "enumerate": enumerate, "zip": zip,
        "min": min, "max": max, "sum": sum, "abs": abs,
        "isinstance": isinstance, "type": type,
    }

    exec_globals = {"pd": pd, "__builtins__": safe_builtins}
    local_ns: dict = {"df": df.copy(), "pd": pd}

    try:
        exec(clean_code, exec_globals, local_ns)  # noqa: S102

        # Strategy A: check well-known result variable names
        for candidate in _RESULT_CANDIDATES:
            result = local_ns.get(candidate)
            if isinstance(result, pd.DataFrame):
                return result, ""

        # Strategy B: find any function defined by the script and call it with df
        for name, obj in local_ns.items():
            if callable(obj) and name not in ("pd",):
                try:
                    result = obj(df.copy())
                    if isinstance(result, pd.DataFrame):
                        return result, ""
                except Exception:  # noqa: BLE001
                    pass  # Try next callable if this one fails

        return None, (
            "Executed code did not assign a result DataFrame. "
            "Ensure the script assigns the output to one of: "
            + ", ".join(f"`{c}`" for c in _RESULT_CANDIDATES)
        )

    except SyntaxError as exc:
        return None, (
            f"SyntaxError (line {exc.lineno}): {exc.msg}. "
            "This usually means markdown backticks were not fully stripped — "
            "check the ASEP code panel above."
        )
    except Exception as exc:  # noqa: BLE001
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

    # Strip status tags from body text — banner carries the status visually
    audit_text = (
        cecep_raw
        .replace("[STATUS: REJECTED]", "")
        .replace("[STATUS: APPROVED]", "")
        .strip()
    )

    # Extract only the violating snippet section from CECEP's response
    violating_snippet = _extract_violating_snippet(cecep_raw)

    # --- Left Column: ASEP (Actor) ---
    with col1:
        with st.container(border=True):
            st.markdown("### [ASEP] Automated Script Engineering & Parser")

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
            st.markdown("### [CECEP] Contract Evaluation & Error Preventer")

            # 1. Status banner — formal, enterprise style
            if is_rejected:
                st.error("STATUS: REJECTED | SCHEMA VIOLATION DETECTED")
            elif is_approved:
                st.success("STATUS: APPROVED | SAFE FOR PRODUCTION")
            else:
                st.warning("CECEP returned an unrecognised status. Raw output shown below.")

            # 2. Flagged Code Snippet
            st.caption("📄 Flagged Code Snippet:")
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
            st.caption("📋 CECEP Responses:")
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
            st.subheader("🎉 Applied Modification Preview")

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

            st.download_button(
                label="📥 Download Updated Dataset (.csv)",
                data=updated_df.to_csv(index=False).encode("utf-8"),
                file_name="updated_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )

    elif is_rejected:
        # Governance gate blocked execution — do not run the code
        st.warning("⚠️ Script execution blocked due to governance violations.")


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
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg",
            width=80,
        )
        st.title("J.A.W.I.R. Control Panel")
        st.markdown("**Team STEIJanggal**")
        st.markdown("---")

        # Watsonx connection status
        st.markdown("#### Watsonx Connection")
        if CREDS["api_key"] and CREDS["project_id"]:
            st.success("Credentials loaded")
        else:
            st.warning("Credentials missing in .env!")

        st.markdown(f"**Endpoint:** `{CREDS['url']}`")
        st.markdown("---")

        # Active governance contract summary
        st.markdown("#### Active Governance Contract")
        with st.expander("View SCHEMA_EXPECTATIONS", expanded=False):
            st.markdown(SCHEMA_EXPECTATIONS)
        with st.expander("View LINEAGE_MAP", expanded=False):
            st.markdown(LINEAGE_MAP)

    # ---- Main Header --------------------------------------------------------
    st.title("J.A.W.I.R.")
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
        st.markdown("**Data Preview (first 5 rows):**")
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
