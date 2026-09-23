"""
MCP (Model Context Protocol) layer of dSGP4 (EXPERIMENTAL).

This optional subpackage exposes the library to MCP clients (Claude Desktop, Claude
Code, IDEs, agents...) as a server with tools, resources and prompts. It requires the
`mcp` extra (`pip install dsgp4[mcp]`, Python >= 3.10) and is never imported by the
rest of the library.

This layer is experimental: the library functionality behind it is stable, but the
MCP Python SDK is still evolving quickly, so the server surface (tool names, output
fields) may be adjusted in future releases.

Run the server from the command line:

    dsgp4-mcp                       # all domains, stdio transport
    dsgp4-mcp --domains tle,propagation
    dsgp4-mcp --transport streamable-http --port 8000

or create it programmatically:

    from dsgp4.mcp import create_server
    server = create_server()
    server.run(transport='stdio')
"""
_INSTALL_HINT = ("The MCP support of dsgp4 requires the optional 'mcp' dependency, version 2 or later (and Python >= 3.10): "
                 "install or upgrade it with `pip install -U dsgp4[mcp]` or `pip install -U mcp`.")


def create_server(domains=None, name='dsgp4'):
    """
    Creates the dSGP4 MCP server (see `dsgp4.mcp.server.create_server`). Raises a
    `ModuleNotFoundError` with an installation hint when the `mcp` dependency is
    missing.
    """
    try:
        from .server import create_server as _create_server
    except ModuleNotFoundError as error:
        if getattr(error, 'name', '') and error.name.split('.')[0] == 'mcp' and error.name != 'dsgp4.mcp':
            raise ModuleNotFoundError(_INSTALL_HINT) from error
        raise
    return _create_server(domains=domains, name=name)


def main(argv=None):
    """Command line entry point of the server (see `dsgp4.mcp.__main__`)."""
    from .__main__ import main as _main
    return _main(argv)
