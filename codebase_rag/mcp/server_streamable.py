import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from loguru import logger
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent, Tool
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Route

from codebase_rag import constants as cs
from codebase_rag import logs as lg
from codebase_rag import tool_errors as te
from codebase_rag.config import settings
from codebase_rag.mcp.tools import create_mcp_tools_registry
from codebase_rag.services.graph_service import MemgraphIngestor
from codebase_rag.services.llm import CypherGenerator
from codebase_rag.types_defs import MCPToolArguments


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=cs.MCP_LOG_LEVEL_INFO,
        format=cs.MCP_LOG_FORMAT,
    )


def get_project_root() -> Path:
    # First priority: Command line argument
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
        logger.info(lg.MCP_SERVER_INFERRED_ROOT.format(path=repo_path))
    else:
        # Second priority: Environment variables
        repo_path: str | None = (
            os.environ.get(cs.MCPEnvVar.TARGET_REPO_PATH) or settings.TARGET_REPO_PATH
        )

        if not repo_path:
            repo_path = os.environ.get(
                cs.MCPEnvVar.CLAUDE_PROJECT_ROOT
            ) or os.environ.get(cs.MCPEnvVar.PWD)

            if repo_path:
                logger.info(lg.MCP_SERVER_INFERRED_ROOT.format(path=repo_path))
            else:
                repo_path = str(Path.cwd())
                logger.info(lg.MCP_SERVER_NO_ROOT.format(path=repo_path))

    project_root = Path(repo_path).resolve()

    if not project_root.exists():
        raise ValueError(te.MCP_PATH_NOT_EXISTS.format(path=project_root))

    if not project_root.is_dir():
        raise ValueError(te.MCP_PATH_NOT_DIR.format(path=project_root))

    logger.info(lg.MCP_SERVER_ROOT_RESOLVED.format(path=project_root))
    return project_root


def remove_none(obj: Any) -> Any:
    if isinstance(obj, list):
        return [remove_none(x) for x in obj if x is not None]
    elif isinstance(obj, dict):
        return {k: remove_none(v) for k, v in obj.items() if v is not None}
    else:
        return obj


def create_mcp_server() -> tuple[Server, MemgraphIngestor]:
    setup_logging()

    try:
        project_root = get_project_root()
        logger.info(lg.MCP_SERVER_USING_ROOT.format(path=project_root))
    except ValueError as e:
        logger.error(lg.MCP_SERVER_CONFIG_ERROR.format(error=e))
        raise

    logger.info(lg.MCP_SERVER_INIT_SERVICES)

    ingestor = MemgraphIngestor(
        host=settings.MEMGRAPH_HOST,
        port=settings.MEMGRAPH_PORT,
        batch_size=settings.MEMGRAPH_BATCH_SIZE,
    )

    cypher_generator = CypherGenerator()

    tools = create_mcp_tools_registry(
        project_root=str(project_root),
        ingestor=ingestor,
        cypher_gen=cypher_generator,
    )

    logger.info(lg.MCP_SERVER_INIT_SUCCESS)

    server = Server(cs.MCP_SERVER_NAME)

    def _create_error_content(message: str) -> list[TextContent]:
        return [
            TextContent(
                type=cs.MCP_CONTENT_TYPE_TEXT,
                text=te.ERROR_WRAPPER.format(message=message),
            )
        ]

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        schemas = tools.get_tool_schemas()
        return [
            Tool(
                name=schema.name,
                description=schema.description,
                inputSchema={**schema.inputSchema},
            )
            for schema in schemas
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: MCPToolArguments) -> list[TextContent]:
        logger.info(lg.MCP_SERVER_CALLING_TOOL.format(name=name))

        try:
            handler_info = tools.get_tool_handler(name)
            if not handler_info:
                error_msg = cs.MCP_UNKNOWN_TOOL_ERROR.format(name=name)
                logger.error(lg.MCP_SERVER_UNKNOWN_TOOL.format(name=name))
                return _create_error_content(error_msg)

            handler, returns_json = handler_info

            result = await handler(**arguments)

            # Filter out None types recursively
            result = remove_none(result)

            if returns_json:
                result_text = json.dumps(result, indent=cs.MCP_JSON_INDENT)
            else:
                if result is None:
                    result_text = ""
                else:
                    result_text = str(result)

            return [TextContent(type=cs.MCP_CONTENT_TYPE_TEXT, text=result_text)]

        except Exception as e:
            error_msg = cs.MCP_TOOL_EXEC_ERROR.format(name=name, error=e)
            logger.exception(lg.MCP_SERVER_TOOL_ERROR.format(name=name, error=e))
            return _create_error_content(error_msg)

    return server, ingestor


@asynccontextmanager
async def lifespan(app: Starlette):
    server, ingestor = create_mcp_server()
    app.state.server = server
    app.state.manager = StreamableHTTPSessionManager(server)
    
    # Manually manage ingestor lifecycle
    logger.info(
        lg.MCP_SERVER_CONNECTED.format(
            host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT
        )
    )
    with ingestor:
        async with app.state.manager.run():
            yield


async def handle_request(request: Request):
    return request.app.state.manager.handle_request


def create_app() -> Starlette:
    routes = [
        Route("/", endpoint=handle_request, methods=["GET", "POST"]),
    ]
    return Starlette(debug=True, routes=routes, lifespan=lifespan)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=8000)
