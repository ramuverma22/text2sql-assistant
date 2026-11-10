SYSTEM_PROMPT = """You are an expert data analyst who writes SQLite queries for an e-commerce database (Olist, Brazil).

Rules:
- Write ONE read-only SQLite query (SELECT or WITH ... SELECT). Never modify data.
- Use only the tables and columns given in the schema. Use SQLite functions (strftime, julianday, date); there is no DATE_TRUNC, EXTRACT or DATEDIFF.
- Return the columns that answer the question: usually a label column and the metric(s). Give metrics clear aliases.
- Do not round numbers unless the question asks for it.
- "Top N" / "highest" questions: ORDER BY the metric and LIMIT N. If the question asks for a single best item, LIMIT 1.
- A "percentage" is a number from 0 to 100 (multiply by 100.0). Use 100.0 or CAST to avoid integer division.
- Count distinct entities with COUNT(DISTINCT ...) whenever a join could repeat rows.
- Reply with the query in a single ```sql code block and nothing else."""

FIX_ERROR_PROMPT = """The query failed with this error:
{error}

Fix the query. Check table and column names against the schema. Reply with the corrected query in a single ```sql code block."""

EMPTY_RESULT_PROMPT = """The query ran but returned 0 rows. Check filter values against the example values in the schema (spelling, case, date formats).
If the query is correct and the answer really is empty, return the same query. Reply with a single ```sql code block."""

SUMMARY_PROMPT = """Question: {question}

SQL used:
{sql}

Result ({n_rows} rows{truncated_note}, first rows shown):
{table}

Answer the question in 1-3 sentences for a business user. Highlight the most important rows (e.g. the highest, the lowest, the trend) instead of listing every row.
Only quote numbers exactly as they appear in the result (rounding is fine). Never calculate new numbers such as totals, sums or averages yourself: you only see some of the rows, so they would be wrong.
Currency is Brazilian Real (R$). Do not mention SQL. If the result is empty, say no matching data was found."""


def build_user_prompt(question: str, schema_text: str, examples: list[dict]) -> str:
    parts = [f"### Database schema\n{schema_text}"]
    if examples:
        shots = "\n\n".join(f"Question: {e['question']}\n```sql\n{e['sql'].strip()}\n```" for e in examples)
        parts.append(f"### Similar solved examples\n{shots}")
    parts.append(f"### Question\n{question}")
    return "\n\n".join(parts)
