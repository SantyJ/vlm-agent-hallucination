# This file builds the multi agent map reading workflow
# Four sub agents look at quadrants and a root agent decides the path
import os
import json
import base64
from typing import TypedDict, Dict, List
from dotenv import load_dotenv
from anthropic import Anthropic
from langgraph.graph import StateGraph, END

# Loads the API key and starts the Anthropic client
load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Names of the two Claude models used
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-5"

# Shape of the data passed between steps in the workflow
class FlowState(TypedDict):
    city_id: int
    condition: str
    arch_type: str
    quadrant_paths: Dict[str, str]
    subagent_reports: List[Dict]
    root_decision: Dict
    ground_truth: Dict

# Reads an image and turns it into base64 text
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

# Strips markdown fences so the model reply parses as JSON
def clean_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# The prompt every quadrant subagent receives
SUBAGENT_PROMPT = (
    "You are observing the {sector} quadrant of a city map.\n"
    "Identify all visible landmarks and intersection labels in the image.\n"
    "IMPORTANT: Only list intersection labels (e.g. i08, i19) that you can "
    "EXPLICITLY SEE printed on the image. Do NOT infer, guess, or complete "
    "sequences. If you see i08 and i18, do NOT assume i09-i17 exist.\n"
    "Return ONLY valid JSON. No markdown, no backticks, no extra text.\n"
    "Schema:\n"
    '{{"sector": "{sector}", '
    '"landmarks_seen": ["..."], '
    '"intersections_seen": ["..."], '
    '"confidence": <integer 0-100>}}'
)

# Builds one subagent for the named quadrant
def make_subagent(sector: str):
    def node(state: FlowState) -> Dict:
        path = state["quadrant_paths"][sector]
        b64 = encode_image(path)
        prompt = SUBAGENT_PROMPT.format(sector=sector)
        try:
            resp = client.messages.create(
                model=HAIKU_MODEL,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            raw = resp.content[0].text
            report = json.loads(clean_json_text(raw))
        except Exception:
            report = {
                "sector": sector,
                "landmarks_seen": [],
                "roads_seen": [],
                "confidence": 0,
                "parse_error": True,
            }
        existing = state.get("subagent_reports", [])
        return {"subagent_reports": existing + [report]}
    return node

# The prompt the root agent receives
ROOT_PROMPT = (
    "You are a navigation coordinator reviewing reports from 4 quadrant subagents.\n"
    "Determine the shortest path from Zorblax Tower to Qintu Market.\n\n"
    "Subagent reports:\n{reports}\n\n"
    "Return ONLY valid JSON. No markdown, no backticks, no extra text.\n"
    "Schema:\n"
    '{{"recommended_path": ["..."], '
    '"reasoning": "...", '
    '"agents_trusted": ["..."], '
    '"confidence": <integer 0-100>, '
    '"contradictions_found": ["..."]}}'
)

# Sends content to the root agent and parses its reply
def call_root(content_blocks) -> Dict:
    thinking_trace = ""
    text_out = ""
    try:
        resp = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=4096,
            thinking={"type": "enabled", "budget_tokens": 1024},
            messages=[{"role": "user", "content": content_blocks}],
        )
        for block in resp.content:
            if block.type == "thinking":
                thinking_trace = block.thinking
            elif block.type == "text":
                text_out = block.text
                print("RAW ROOT OUTPUT:", repr(text_out))
        decision = json.loads(clean_json_text(text_out))
        decision["thinking_trace"] = thinking_trace
        return decision
    except Exception as e:
        print("ROOT ERROR:", e)
        return {
            "recommended_path": [],
            "reasoning": "",
            "agents_trusted": [],
            "confidence": 0,
            "contradictions_found": [],
            "thinking_trace": thinking_trace,
            "raw_text": text_out,
            "parse_error": True,
        }

# Variant A where the root agent only sees text reports
def root_agent_arch_a(state: FlowState) -> Dict:
    reports_str = json.dumps(state["subagent_reports"], indent=2)
    prompt = ROOT_PROMPT.format(reports=reports_str)
    decision = call_root([{"type": "text", "text": prompt}])
    return {"root_decision": decision}

# Variant B where the root agent sees text plus all four images
def root_agent_arch_b(state: FlowState) -> Dict:
    reports_str = json.dumps(state["subagent_reports"], indent=2)
    prompt = ROOT_PROMPT.format(reports=reports_str)
    content = []
    for sector in ["NW", "NE", "SW", "SE"]:
        b64 = encode_image(state["quadrant_paths"][sector])
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        })
    content.append({"type": "text", "text": prompt})
    decision = call_root(content)
    return {"root_decision": decision}

# Picks variant A or B based on the run setting
def root_node(state: FlowState) -> Dict:
    if state["arch_type"] == "B":
        return root_agent_arch_b(state)
    return root_agent_arch_a(state)

# Wires the four subagents and the root agent into a sequence
def build_graph():
    g = StateGraph(FlowState)
    g.add_node("NW", make_subagent("NW"))
    g.add_node("NE", make_subagent("NE"))
    g.add_node("SW", make_subagent("SW"))
    g.add_node("SE", make_subagent("SE"))
    g.add_node("root", root_node)
    g.set_entry_point("NW")
    g.add_edge("NW", "NE")
    g.add_edge("NE", "SW")
    g.add_edge("SW", "SE")
    g.add_edge("SE", "root")
    g.add_edge("root", END)
    return g.compile()
