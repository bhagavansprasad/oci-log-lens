"""
prompts.py
----------
All LLM prompts for OIC-LogLens are maintained here.
Each prompt is a function that takes string inputs and returns the final built prompt.
Schemas and static instructions are defined as module-level globals for easy maintenance.
"""

# ── OUTPUT CONTRACT ────────────────────────────────────────────────────────────
# This is the exact JSON structure the LLM must always return.
# - Every field must be present in the response.
# - If a field has no data in the source log, return null — never omit the field.
# - For informational logs, the entire "error" block must be null.

NORMALIZED_LOG_SCHEMA = """
{
  "log_type": "error | informational",

  "flow": {
    "code":         "string | null",
    "version":      "string | null",
    "type":         "number | null",
    "trigger_type": "rest | soap | scheduled | null",
    "operation":    "string | null",
    "timestamp":    "ISO 8601 string | null"
  },

  "user": {
    "id": "string | null"
  },

  "tracking_variables": {
    "primary_key": { "name": "string | null", "value": "string | null" },
    "secondary":   [ { "name": "string", "value": "string" } ]
  },

  "error": {
    "code":           "string | null",
    "state":          "number | null",
    "summary":        "string | null",
    "message_parsed": {
      "http_status":       "number | null",
      "root_cause":        "string | null",
      "failed_url":        "string | null",
      "error_description": "string | null"
    },
    "endpoint_name":       "string | null",
    "endpoint_type":       "string | null",
    "operation":           "string | null",
    "milestone":           "string | null",
    "retry_count":         "number | null",
    "auto_retriable":      "boolean | null",
    "business_error_name": "string | null"
  }
}
"""

# ── NORMALIZATION RULES ────────────────────────────────────────────────────────
# Static rules injected into the normalization prompt.
# Update this global when rules change — no need to touch the function.

NORMALIZATION_RULES = """
RULES:
1. Read the entire JSON array and understand the full execution story of the workflow.
2. Classify the log as "error" or "informational":
   - "error"         : if any object contains errorCode, errorMessage, or errorState = 500
   - "informational" : if no error indicators are present
3. Extract the output fields from the relevant objects:
   - flow.*              : from the object where automationRoot = true or flowCode is present
   - user.id             : from the userId field in the root trigger object
   - tracking_variables  : from the object that contains the "variables" array
       - Strip the _Oo_VarName_Oo_ wrapper from values to extract the clean value
       - The variable with trackingPkVarName is the primary_key
       - All others are secondary
   - error.*             : from the object that contains errorCode AND errorMessage
4. For the error.message_parsed block:
   - The errorMessage field often contains raw XML (APIInvocationError format)
   - Extract http_status, root_cause, failed_url, and a clean error_description from it
   - Do not include raw XML in the output
5. Convert flowEventCreationDate (epoch milliseconds) to ISO 8601 timestamp format.
6. For trigger_type:
   - If the root trigger object has no endpointType and has scheduleId or schedulerJobState → "scheduled"
   - If endpointType = "rest"  → "rest"
   - If endpointType = "soap"  → "soap"
7. For informational logs, return the entire "error" block as null.
8. Never add fields that are not in the output schema.
9. Never omit fields — always return every field in the schema, use null if no data exists.
10. Return only valid JSON — no explanation, no markdown, no code fences.
"""


# ── PROMPT FUNCTIONS ───────────────────────────────────────────────────────────

def get_normalization_prompt(raw_log: str) -> str:
    """
    Builds the normalization prompt for a raw OIC log.

    Args:
        raw_log: The raw log file content as a JSON string.

    Returns:
        The complete prompt string to send to the LLM.
    """
    return f"""You are a log normalization engine for Oracle Integration Cloud (OIC) logs.

Your job is to read a raw OIC log and extract a normalized JSON object that strictly follows the output schema below.

OUTPUT SCHEMA:
{NORMALIZED_LOG_SCHEMA}

{NORMALIZATION_RULES}

RAW LOG:
{raw_log}

Return only the normalized JSON object. No explanation. No markdown. No code fences.
"""

# ── EMBEDDING FIELDS ──────────────────────────────────────────────────────────
# Fields selected from the normalized log to build the embedding text.
# Only semantically meaningful fields are included.
# Update these globals to change what gets embedded — no need to touch the function.

EMBEDDING_FIELDS = [
    ("flow",  "code"),
    ("flow",  "trigger_type"),
    ("error", "code"),
    ("error", "summary"),
    ("error", "endpoint_name"),
    ("error", "endpoint_type"),
]

EMBEDDING_NESTED_FIELDS = [
    ("error", "message_parsed", "root_cause"),
    ("error", "message_parsed", "error_description"),
]

def get_embedding_text(normalized_log: dict) -> str:
    """
    Builds a single concatenated text string from selected fields
    of the normalized log for use in embedding generation.

    Args:
        normalized_log: Normalized log dict (output of normalize_log).

    Returns:
        A clean concatenated string of the selected fields.
    """
    parts = []

    for parent, field in EMBEDDING_FIELDS:
        value = normalized_log.get(parent, {})
        if isinstance(value, dict):
            val = value.get(field)
        else:
            val = None
        if val:
            parts.append(str(val))

    for parent, child, field in EMBEDDING_NESTED_FIELDS:
        val = normalized_log.get(parent, {})
        if isinstance(val, dict):
            val = val.get(child, {})
        if isinstance(val, dict):
            val = val.get(field)
        if val:
            parts.append(str(val))

    return " ".join(parts)