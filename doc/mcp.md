# MCP server (LLM/agent integration)

`dsgp4` ships an optional [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server
that exposes the library to large language models and agents: once connected, an LLM client
(Claude Desktop, Claude Code, IDEs, agent frameworks) can call dSGP4 directly — parse and
validate TLEs, propagate orbits, compute autodiff Jacobians and covariances, fit TLEs, run
ML-dSGP4 forecasts and render orbit plots — instead of writing and executing Python code.

```{note}
This layer is **experimental**. The library functionality behind it is stable, but the MCP
Python SDK is still evolving quickly, so the server surface (tool names, output fields) may be
adjusted in future releases.
```

## Installation

The server is an optional extra (it requires Python >= 3.10):

```console
$ pip install dsgp4[mcp]
```

This installs the `dsgp4-mcp` command (equivalently: `python -m dsgp4.mcp`). Running it starts
an MCP server; on its own it just waits for a client, so the interesting part is registering it
with an LLM client, below.

## Connecting it to an LLM

An MCP client is the bridge: you register the server with the client once, and from then on the
LLM sees the dSGP4 tools and calls them on its own when a conversation needs them (you can then
simply paste a TLE in the chat and ask for an analysis).

**Claude Code** (terminal):

```console
$ claude mcp add dsgp4 -- dsgp4-mcp
```

**Claude Desktop** (Settings → Developer → Edit Config, i.e. `claude_desktop_config.json`) and
most other command-based MCP clients (Cursor, VS Code, ...):

```json
{
  "mcpServers": {
    "dsgp4": {
      "command": "dsgp4-mcp"
    }
  }
}
```

With this configuration the client launches `dsgp4-mcp` itself and talks to it over
stdin/stdout (the default `stdio` transport); there is nothing to keep running manually.

```{note}
The client must be able to find `dsgp4-mcp` in the environment where `dsgp4[mcp]` is installed.
If you installed it in a conda/virtual environment, either use the absolute path of the
`dsgp4-mcp` script in the `command` field, or use the environment's interpreter explicitly:
`"command": "/path/to/env/bin/python", "args": ["-m", "dsgp4.mcp"]`.
```

By default all the tool domains are enabled; a narrower server can be registered by adding
arguments, e.g. `"args": ["--domains", "tle,propagation"]`.

### URL-based clients (HTTP transport)

Clients that connect to a URL instead of launching a command (web-based agents, remote setups,
several clients sharing one server) can use the HTTP transport:

```console
$ dsgp4-mcp --transport streamable-http --port 8000
```

and point the client at `http://127.0.0.1:8000/mcp`. Note that this endpoint speaks the MCP
JSON-RPC protocol — it is an API for MCP clients, not a web page for a browser. The server
binds `127.0.0.1` (local machine only) by default and has no authentication: do not expose it
on a network (`--host 0.0.0.0`) unless it is behind a reverse proxy or on a trusted network.

### Trying it without an LLM

The [MCP Inspector](https://github.com/modelcontextprotocol/inspector) provides a browser UI to
explore and call the tools by hand:

```console
$ npx @modelcontextprotocol/inspector dsgp4-mcp
```

## What the LLM gets

The tools are organized in six domains (each can be disabled via `--domains`):

| Domain | Tools |
|---|---|
| `tle` | `parse_element_set`, `validate_tle`, `fix_tle_checksum`, `build_tle`, `convert_element_sets` (TLE ↔ CCSDS OMM in JSON/XML/KVN/CSV) |
| `propagation` | `propagate`, `propagate_batch`, `cartesian_to_keplerian`, `convert_time`, `orbital_elements_summary` |
| `gradients` | `state_partials_wrt_tle` (6×9 autodiff Jacobians), `state_partials_wrt_time`, `transform_covariance` |
| `estimation` | `update_tle_epoch`, `fit_tle_to_state` (differentiable Newton-Raphson TLE determination) |
| `ml` | `mldsgp4_propagate` (trained ML-dSGP4 checkpoints), `mldsgp4_inspect` |
| `plot` | `plot_orbits`, `plot_element_distributions` (returned as PNG images) |

Conventions shared by every tool: element-set inputs accept a TLE (two or three lines in one
string) *or* a CCSDS OMM message; times are minutes since the element-set epoch or ISO 8601 UTC
dates; output states are in the TEME frame, in km and km/s.

The server also exposes reference **resources** the model can read (`dsgp4://reference/...`:
conventions, the TLE and OMM formats, the nine differentiable SGP4 parameters and their
internal units, gravity constants, example element sets) and workflow **prompts**
(`characterize_orbit`, `compare_orbits`, `uncertainty_analysis`, `tle_determination`,
`mldsgp4_training_guide`) that guide the model through multi-tool analyses.

Once connected, things you can ask the LLM directly:

- *"Here is a TLE: … — validate it, describe the orbit and plot one revolution."*
- *"Propagate these three TLEs to 2024-03-27T12:00:00 UTC and compare their positions."*
- *"How sensitive is the position after one day to the mean motion? Show the Jacobian."*
- *"Map this covariance of the mean elements to a Cartesian covariance at epoch + 60 min."*
- *"Re-epoch this TLE to next Monday 00:00 UTC."*

## Programmatic use

The server can also be created and run from Python:

```python
from dsgp4.mcp import create_server

server = create_server()                      # or create_server(domains=["tle", "propagation"])
server.run(transport="stdio")
```
