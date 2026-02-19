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


# ── RE-RANKING PROMPT ──────────────────────────────────────────────────────────

RERANK_SYSTEM_PROMPT = """You are an expert at analyzing Oracle Integration Cloud (OIC) error logs and determining if errors are duplicates or related issues.

Your task is to analyze a query log and a list of candidate similar logs, then classify each candidate and provide reasoning.

Classification categories:
- EXACT_DUPLICATE (90-100% match): Same root cause, same error, same fix applicable
- SIMILAR_ROOT_CAUSE (70-89% match): Same underlying issue, solution likely transferable
- RELATED (50-69% match): Some overlap, may provide useful context
- NOT_RELATED (0-49% match): Different issues, not helpful

Focus on:
1. Root cause analysis (not just error codes)
2. Error patterns and chains
3. Business context (workflow, endpoint, integration)
4. Whether the same fix would apply

Return your analysis as JSON only, no markdown, no preamble."""

RERANK_USER_PROMPT = """Query Log (New Error):
{query_log}

Candidate Similar Logs:
{candidates}

Analyze each candidate and return as a JSON object with this structure:
{{
  "results": [
    {{
      "jira_id": "OLL-XXX",
      "rank": 1,
      "classification": "EXACT_DUPLICATE",
      "confidence": 95,
      "reasoning": "Brief explanation why"
    }}
  ]
}}

Order by rank (1 = best match). Include all candidates.
Classification must be one of: EXACT_DUPLICATE, SIMILAR_ROOT_CAUSE, RELATED, NOT_RELATED"""


# ── RE-RANKING RESPONSE SCHEMA ─────────────────────────────────────────────────

RERANK_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "jira_id": {
                        "type": "string",
                        "description": "Jira ticket ID (e.g., OLL-4FF0674A)"
                    },
                    "rank": {
                        "type": "integer",
                        "description": "Ranking position (1 = best match)",
                        "minimum": 1
                    },
                    "classification": {
                        "type": "string",
                        "enum": ["EXACT_DUPLICATE", "SIMILAR_ROOT_CAUSE", "RELATED", "NOT_RELATED"],
                        "description": "Classification category"
                    },
                    "confidence": {
                        "type": "integer",
                        "description": "Confidence score 0-100",
                        "minimum": 0,
                        "maximum": 100
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of classification"
                    }
                },
                "required": ["jira_id", "rank", "classification", "confidence", "reasoning"]
            }
        }
    },
    "required": ["results"]
}


def get_rerank_prompt(query_log: dict, candidates: list) -> tuple[str, str, dict]:
    """
    Build the LLM re-ranking prompt with structured output schema.
    
    Args:
        query_log: Normalized query log
        candidates: List of candidate logs from vector search
        
    Returns:
        Tuple of (system_prompt, user_prompt, response_schema)
    """
    # Format query log - only key fields
    query_summary = f"""
Flow: {query_log.get('flow', {}).get('code', 'N/A')}
Trigger: {query_log.get('flow', {}).get('trigger_type', 'N/A')}
Error Code: {query_log.get('error', {}).get('code', 'N/A')}
Error Summary: {query_log.get('error', {}).get('summary', 'N/A')[:200]}
Root Cause: {query_log.get('error', {}).get('message_parsed', {}).get('root_cause', 'N/A')}
"""
    
    # Format candidates with full normalized data
    import json
    from config import logger
    candidates_text = ""
    for i, candidate in enumerate(candidates, 1):
        norm_data = candidate.get('normalized_json', {})
        
        # If still a string, parse it
        if isinstance(norm_data, str):
            try:
                norm_data = json.loads(norm_data)
                logger.info(f"Candidate {i}: Successfully parsed JSON string")
            except Exception as e:
                logger.warning(f"Candidate {i}: Failed to parse JSON: {e}")
                norm_data = {}
        
        flow_info = norm_data.get('flow', {})
        error_info = norm_data.get('error', {})
        message_parsed = error_info.get('message_parsed', {})
        root_cause = message_parsed.get('root_cause', 'N/A')
        
        candidates_text += f"""
Candidate {i}:
  Jira ID: {candidate.get('jira_id', 'N/A')}
  Similarity: {candidate.get('similarity_score', 0)}%
  Flow: {flow_info.get('code', candidate.get('flow_code', 'N/A'))}
  Trigger: {flow_info.get('trigger_type', candidate.get('trigger_type', 'N/A'))}
  Error Code: {error_info.get('code', candidate.get('error_code', 'N/A'))}
  Error Summary: {error_info.get('summary', candidate.get('error_summary', 'N/A'))[:200]}
  Root Cause: {error_info.get('message_parsed', {}).get('root_cause', 'N/A')}

"""
    
    user_prompt = RERANK_USER_PROMPT.format(
        query_log=query_summary.strip(),
        candidates=candidates_text.strip()
    )
    
    return RERANK_SYSTEM_PROMPT, user_prompt, RERANK_RESPONSE_SCHEMA
