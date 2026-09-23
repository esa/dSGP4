"""
Command line entry point of the dSGP4 MCP server (`dsgp4-mcp`, or
`python -m dsgp4.mcp`).
"""
import argparse
import sys

_INSTALL_HINT = ("The MCP support of dsgp4 requires the optional 'mcp' dependency, version 2 or later (and Python >= 3.10): "
                 "install or upgrade it with `pip install -U dsgp4[mcp]` or `pip install -U mcp`.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='dsgp4-mcp',
        description='Run the dSGP4 MCP server, exposing the differentiable SGP4 library '
                    '(TLE/OMM handling, propagation, autodiff gradients, TLE estimation, '
                    'ML-corrected forecasts and plots) to MCP clients.')
    parser.add_argument('--domains', default=None,
                        help='comma-separated tool domains to enable (default: all). '
                             "Available: 'tle', 'propagation', 'gradients', 'estimation', 'ml', 'plot'.")
    parser.add_argument('--transport', default='stdio', choices=['stdio', 'sse', 'streamable-http'],
                        help='MCP transport (default: stdio, the one desktop clients use).')
    parser.add_argument('--host', default='127.0.0.1',
                        help='bind address for the sse/streamable-http transports (default: 127.0.0.1).')
    parser.add_argument('--port', type=int, default=8000,
                        help='port for the sse/streamable-http transports (default: 8000).')
    parser.add_argument('--name', default='dsgp4', help="name the server advertises (default: 'dsgp4').")
    args = parser.parse_args(argv)

    try:
        from .server import create_server
    except ModuleNotFoundError as error:
        if getattr(error, 'name', '') and error.name.split('.')[0] == 'mcp' and error.name != 'dsgp4.mcp':
            sys.exit(_INSTALL_HINT)
        raise  # pragma: no cover - unrelated import failure

    domains = [domain.strip() for domain in args.domains.split(',') if domain.strip()] if args.domains else None
    server = create_server(domains=domains, name=args.name)
    if args.transport == 'stdio':
        server.run(transport='stdio')
    else:
        server.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == '__main__':  # pragma: no cover - exercised via the stdio subprocess
    main()
