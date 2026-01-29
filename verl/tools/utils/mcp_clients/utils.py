# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import threading
import time

from mcp import Tool

logger = logging.getLogger(__file__)


class TokenBucket:
    def __init__(self, rate_limit: float):
        self.rate_limit = rate_limit  # tokens per second
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        with self.lock:
            now = time.time()
            # Add new tokens based on time elapsed
            new_tokens = (now - self.last_update) * self.rate_limit
            self.tokens = min(self.rate_limit, self.tokens + new_tokens)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


def _transform_anyof_schema(schema: dict) -> dict:
    """Transform anyOf schemas to simple type format.

    MCP servers may return schemas with anyOf for optional types like:
    {"anyOf": [{"type": "string"}, {"type": "null"}]}

    This transforms them to simple format:
    {"type": "string"}
    """
    if not isinstance(schema, dict):
        return schema

    # Handle properties in parameters
    if "properties" in schema:
        for prop_name, prop_def in schema["properties"].items():
            if isinstance(prop_def, dict) and "anyOf" in prop_def:
                # Extract first non-null type from anyOf
                for type_def in prop_def["anyOf"]:
                    if isinstance(type_def, dict) and type_def.get("type") != "null":
                        prop_def["type"] = type_def.get("type")
                        # Copy other fields like description if present
                        if "description" in type_def:
                            prop_def["description"] = type_def["description"]
                        break
                # Remove anyOf after extracting type
                del prop_def["anyOf"]

    return schema


def mcp2openai(mcp_tool: Tool) -> dict:
    """Convert a MCP Tool to an OpenAI ChatCompletionTool."""
    # Transform anyOf schemas before creating OpenAI format
    parameters = _transform_anyof_schema(dict(mcp_tool.inputSchema))

    openai_format = {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": parameters,
            "strict": False,
        },
    }
    if not openai_format["function"]["parameters"].get("required", None):
        openai_format["function"]["parameters"]["required"] = []
    return openai_format
