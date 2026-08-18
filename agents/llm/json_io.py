"""Small JSON helper shared by all LLM roles."""

import json
import re


def parse_json(output):
    if not isinstance(output, str):
        return output
    text = output.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return json.loads(match.group(1) if match else text)
