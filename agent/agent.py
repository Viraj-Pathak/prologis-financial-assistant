import json
import os
from typing import Any
from pathlib import Path

# Must be set BEFORE importing google.genai — the SDK reads this at Client() instantiation
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"

from dotenv import load_dotenv, dotenv_values
from google import genai
from google.genai import types

from agent.tools import query_postgres, query_sec_edgar, query_press_releases
from agent.bedrock import summarize_with_bedrock

_ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_FILE, override=True)
_raw = dotenv_values(_ENV_FILE)
_GOOGLE_API_KEY = _raw.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=_GOOGLE_API_KEY, vertexai=False)
MODEL = "gemini-2.0-flash"

SYSTEM_INSTRUCTION = """You are a financial intelligence assistant for Prologis, Inc.
(NYSE: PLD), a global industrial REIT. You have access to four tools:

1. query_postgres — filter properties and financials stored in our database
2. query_sec_edgar — look up official financial metrics from SEC EDGAR filings
3. query_press_releases — search recent Prologis press releases
4. summarize_with_bedrock — condense long text into a short summary

Routing rules:
- Questions about specific properties, metro areas, sq footage, or property revenue → query_postgres
- Questions about company-level financials (revenue, net income, total assets) → query_sec_edgar
- Questions about news, acquisitions, expansions, earnings announcements → query_press_releases
- When a press release answer needs condensing → summarize_with_bedrock
- Multi-source questions: call multiple tools and synthesize the answers

Always cite specific figures (dollar amounts, dates, property counts) in your answer.
Format numbers with commas and dollar signs. Keep answers concise and fact-driven."""

TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="query_postgres",
            description="Query the Prologis properties and financials database. Filter by metro area, property type, or minimum revenue.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "metro_area": types.Schema(
                        type=types.Type.STRING,
                        description="City name, e.g. 'Chicago', 'Dallas', 'Los Angeles'",
                    ),
                    "property_type": types.Schema(
                        type=types.Type.STRING,
                        description="One of: Industrial, Logistics, Warehouse",
                    ),
                    "min_revenue": types.Schema(
                        type=types.Type.NUMBER,
                        description="Minimum annual revenue in USD (e.g. 5000000 for $5M)",
                    ),
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="Max number of properties to return (default 20)",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="query_sec_edgar",
            description="Look up Prologis financial metrics from SEC EDGAR filings (10-K annual, 10-Q quarterly).",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "metric": types.Schema(
                        type=types.Type.STRING,
                        description="One of: revenue, net_income, operating_expenses, total_assets, total_liabilities. Leave blank for all.",
                    ),
                    "period": types.Schema(
                        type=types.Type.STRING,
                        description="'annual' (10-K) or 'quarterly' (10-Q). Default: annual",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="query_press_releases",
            description="Search Prologis press releases by keyword and/or category.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "keywords": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="Keywords to search for, e.g. ['acquisition', 'Dallas']",
                    ),
                    "category": types.Schema(
                        type=types.Type.STRING,
                        description="One of: earnings, acquisition, expansion, sustainability",
                    ),
                    "limit": types.Schema(
                        type=types.Type.INTEGER,
                        description="Max number of press releases to return (default 5)",
                    ),
                },
            ),
        ),
        types.FunctionDeclaration(
            name="summarize_with_bedrock",
            description="Use AWS Bedrock (Claude Haiku) to summarize a block of text into a short summary.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "text": types.Schema(
                        type=types.Type.STRING,
                        description="The text to summarize",
                    ),
                    "max_words": types.Schema(
                        type=types.Type.INTEGER,
                        description="Maximum number of words in the summary (default 50)",
                    ),
                },
                required=["text"],
            ),
        ),
    ]
)


def _dispatch(name: str, args: dict) -> Any:
    if name == "query_postgres":
        return query_postgres(**args)
    if name == "query_sec_edgar":
        return query_sec_edgar(**args)
    if name == "query_press_releases":
        return query_press_releases(**args)
    if name == "summarize_with_bedrock":
        return summarize_with_bedrock(**args)
    return {"error": f"Unknown tool: {name}"}


def ask(question: str, max_turns: int = 6) -> dict:
    """Send a natural-language question to the Gemini agent.

    Returns:
        {"answer": str, "tool_calls": [{"tool", "args", "result"}, ...]}
    """
    history = [types.Content(role="user", parts=[types.Part(text=question)])]
    tool_calls_log = []

    for _ in range(max_turns):
        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[TOOLS],
                temperature=0.1,
            ),
        )

        candidate = response.candidates[0]
        history.append(candidate.content)

        function_calls = [
            p.function_call
            for p in candidate.content.parts
            if p.function_call is not None
        ]

        if not function_calls:
            answer = "".join(
                p.text for p in candidate.content.parts if hasattr(p, "text") and p.text
            )
            return {"answer": answer, "tool_calls": tool_calls_log}

        function_responses = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            result = _dispatch(fc.name, args)
            tool_calls_log.append({"tool": fc.name, "args": args, "result": result})
            function_responses.append(
                types.Part.from_function_response(name=fc.name, response={"result": result})
            )

        history.append(types.Content(role="user", parts=function_responses))

    return {
        "answer": "I reached the maximum number of reasoning steps. Please try a more specific question.",
        "tool_calls": tool_calls_log,
    }
