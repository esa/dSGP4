"""
Factory of the dSGP4 MCP server.

The server exposes the library to MCP clients (Claude Desktop, Claude Code, IDEs,
agents...) as a single process organized in tool domains; each domain can be enabled
or disabled independently, so a narrow server (e.g. propagation only) can be run when
the full toolset is not wanted:

- `tle`: parse/validate/build/convert element sets (TLE and CCSDS OMM formats);
- `propagation`: dSGP4 propagation (single and batched), state and time conversions;
- `gradients`: autodiff Jacobians of the state and covariance transformations;
- `estimation`: differentiable Newton-Raphson TLE determination;
- `ml`: forecasts with the hybrid ML-dSGP4 model;
- `plot`: rendered PNG visualizations.

Reference resources (`dsgp4://reference/...`, `dsgp4://examples/...`) and workflow
prompts are always registered.
"""
from mcp.server.mcpserver import MCPServer

from .. import __version__
from . import estimation_tools, gradient_tools, ml_tools, plot_tools, prompts, propagation_tools, resources, tle_tools

#the registrable tool domains, in the order they are attached to the server:
DOMAINS = {
    'tle': tle_tools,
    'propagation': propagation_tools,
    'gradients': gradient_tools,
    'estimation': estimation_tools,
    'ml': ml_tools,
    'plot': plot_tools,
}

INSTRUCTIONS = """\
This server exposes dSGP4, the differentiable PyTorch implementation of the SGP4
satellite orbit propagator (https://github.com/esa/dSGP4). Element set inputs accept
both TLEs (two or three lines in one string) and CCSDS OMM messages (JSON/XML/KVN/CSV);
times are minutes since the element set epoch or ISO 8601 UTC dates; output states are
in the TEME frame, km and km/s. Read the `dsgp4://reference/overview` resource for the
conventions and the available tool domains (TLE handling, propagation, autodiff
gradients, TLE estimation, ML-corrected forecasts, plots).
"""


def create_server(domains=None, name='dsgp4'):
    """
    Creates the dSGP4 `MCPServer`.

    Parameters:
    ----------------
    domains (``list`` of ``str``): tool domains to register, out of 'tle',
        'propagation', 'gradients', 'estimation', 'ml' and 'plot' (default: all).
    name (``str``): name the server advertises to clients.

    Returns:
    ----------------
    ``mcp.server.mcpserver.MCPServer``: the configured server, to be run e.g. with
        ``server.run(transport='stdio')``.
    """
    if domains is None:
        selected = list(DOMAINS)
    else:
        selected = list(domains)
        unknown = [domain for domain in selected if domain not in DOMAINS]
        if unknown:
            raise ValueError('Unknown MCP tool domain(s) {}: available domains are {}.'.format(
                ', '.join(sorted(unknown)), ', '.join(DOMAINS)))
    server = MCPServer(name=name,
                       version=__version__,
                       website_url='https://github.com/esa/dSGP4',
                       instructions=INSTRUCTIONS)
    resources.register(server)
    prompts.register(server)
    for domain in DOMAINS:
        if domain in selected:
            DOMAINS[domain].register(server)
    return server
