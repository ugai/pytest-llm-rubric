"""Anti-corruption layer: normalize external system conventions for internal use."""

from __future__ import annotations

import ipaddress
import urllib.parse

_OLLAMA_DEFAULT_PORT = 11434


def parse_ollama_host(host: str | None) -> str:
    """Normalize an OLLAMA_HOST value into a full ``scheme://host:port[/path]`` URL.

    Handles the same formats as the Ollama CLI/server: bare hostnames, host:port,
    scheme://host, :port shorthand, trailing slashes, paths, and IPv6 brackets.
    """
    host, port = host or "", _OLLAMA_DEFAULT_PORT
    scheme, _, hostport = host.partition("://")
    if not hostport:
        scheme, hostport = "http", host
    elif scheme == "http":
        port = 80
    elif scheme == "https":
        port = 443

    split = urllib.parse.urlsplit(f"{scheme}://{hostport}")
    host = split.hostname or "127.0.0.1"
    port = split.port or port

    try:
        if isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address):
            host = f"[{host}]"
    except ValueError:
        ...

    if path := split.path.strip("/"):
        return f"{scheme}://{host}:{port}/{path}"

    return f"{scheme}://{host}:{port}"
