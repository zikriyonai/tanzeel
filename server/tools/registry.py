"""
Tool registry — exposes Zikriyon WRC tools to Qwen via function calling.
"""

from . import zikriyon_wrc


# OpenAI-compatible tool definitions (Qwen3 uses same format)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information. Use this when the user "
                "asks about recent events, news, facts you're not sure about, or "
                "anything requiring up-to-date information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query in the user's language",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_page",
            "description": (
                "Fetch a specific URL and extract its readable text content. "
                "Use this when you have a URL from a previous search and need "
                "to read its full content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full URL starting with http:// or https://",
                    }
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "research",
            "description": (
                "Deep research: searches the web AND fetches top pages, returning "
                "combined context with sources. Use this for factual questions, "
                "news, or anything needing reliable current information. This is "
                "the preferred tool for most knowledge questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research query",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "How many pages to fetch (default 2, max 4)",
                        "default": 2,
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# Name → function mapping
TOOL_FUNCTIONS = {
    "web_search": zikriyon_wrc.web_search,
    "fetch_page": zikriyon_wrc.fetch_page,
    "research": zikriyon_wrc.research,
}


def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a tool by name and return its result."""
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool: {tool_name}"}

    try:
        return TOOL_FUNCTIONS[tool_name](**arguments)
    except Exception as e:
        return {"error": f"Tool '{tool_name}' failed: {str(e)}"}
