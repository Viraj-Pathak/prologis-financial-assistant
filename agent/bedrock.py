import json
import os
import boto3
from dotenv import load_dotenv

load_dotenv()

REGION = os.getenv("AWS_REGION", "us-east-1")
# Cross-region inference profile required for Claude Haiku 4.5
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def summarize_with_bedrock(text: str, max_words: int = 50) -> str:
    """Summarize text using Claude Haiku via AWS Bedrock. Falls back to truncation on error."""
    if not text:
        return ""
    client = boto3.client("bedrock-runtime", region_name=REGION)
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "messages": [{
            "role": "user",
            "content": (
                f"Summarize the following text in at most {max_words} words. "
                f"Be specific and concrete. Return ONLY the summary, no preamble.\n\n"
                f"Text:\n{text}"
            ),
        }],
    }
    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        return json.loads(response["body"].read())["content"][0]["text"].strip()
    except Exception as e:
        return text[:300] + ("..." if len(text) > 300 else "") + f" [Bedrock error: {e}]"
