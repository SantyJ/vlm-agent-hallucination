import os
import json
import importlib.util
from PIL import Image

# load module with leading-digit filename
spec = importlib.util.spec_from_file_location(
    "flow_module",
    os.path.join(os.path.dirname(__file__), "4_langgraph_flow.py"),
)
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)

# dummy image setup
DUMMY_PATH = os.path.join("data", "cities", "city_141_NW.png")

# initial state
initial_state = {
    "city_id": 0,
    "condition": "baseline",
    "arch_type": "A",
    "quadrant_paths": {
        "NW": DUMMY_PATH,
        "NE": DUMMY_PATH,
        "SW": DUMMY_PATH,
        "SE": DUMMY_PATH,
    },
    "subagent_reports": [],
    "root_decision": {},
    "ground_truth": {},
}

# run pipeline
graph = flow.build_graph()
result = graph.invoke(initial_state)
output_path = "test_pipeline_output.json"
with open(output_path, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"Test output successfully written to {output_path}")
