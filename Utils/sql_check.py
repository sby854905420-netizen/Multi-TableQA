import re
from typing import Tuple

# 允许的起始关键字（只允许查询）
START_QUERY_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


# 明确禁止的关键字/语法（按需可增减）
# 说明：这里是“保守拒绝”，宁可误杀也不放过非查询。
BANNED_TOKENS_RE = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|MERGE|UPSERT|REPLACE|"
    r"CREATE|ALTER|DROP|TRUNCATE|RENAME|"
    r"GRANT|REVOKE|"
    r"ATTACH|DETACH|"
    r"VACUUM|ANALYZE|REINDEX|"
    r"PRAGMA|"
    r"CALL|EXEC|EXECUTE|"
    r"BEGIN|COMMIT|ROLLBACK|SAVEPOINT|"
    r"SET|USE|"
    r"COPY|LOAD|IMPORT|EXPORT"
    r")\b",
    re.IGNORECASE
)

def strip_code_fences(s: str) -> str:
    t = s.strip()
    # remove leading ```sql or ``` and trailing ```
    t = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def single_statement_only(sql: str) -> Tuple[bool, str, str]:
    """
    Ensure exactly one statement:
    - allow at most one trailing semicolon
    - reject if semicolon separates multiple non-empty segments
    """
    t = sql.strip()
    if ";" not in t:
        return True, "ok", t

    parts = [p.strip() for p in t.split(";")]
    non_empty = [p for p in parts if p]
    if len(non_empty) != 1:
        return False, "multiple statements detected", ""
    return True, "ok", non_empty[0]


def looks_like_query_sql(
    s: str,
    require_from: bool = True
) -> Tuple[bool, str, str]:
    """
    Returns: (is_valid, reason, normalized_sql)
    """
    if not s or not s.strip():
        return False, "empty input", ""

    t = strip_code_fences(s)

    # must start with SELECT/WITH
    if not START_QUERY_RE.match(t):
        return False, "must start with SELECT or WITH", ""

    # single statement
    ok, reason, t = single_statement_only(t)
    if not ok:
        return False, reason, ""

    # ban dangerous / non-query tokens anywhere
    if BANNED_TOKENS_RE.search(t):
        return False, "contains banned non-query token", ""

    # must contain SELECT
    if not re.search(r"\bSELECT\b", t, re.IGNORECASE):
        return False, "missing SELECT", ""

    # optional: require FROM for stronger realism
    if require_from:
        # For WITH queries, still require a FROM somewhere (conservative)
        if not re.search(r"\bFROM\b", t, re.IGNORECASE):
            return False, "missing FROM", ""

    return True, "ok", t



