"""Tests for OLLAMA_HOST parsing/normalization.

Behavior is derived from the ollama Python package's _parse_host function.
These tests pin the expected behavior so we can implement a clean-room version.
"""

from __future__ import annotations

from pytest_llm_rubric.utils import parse_ollama_host


class TestParseOllamaHostDefaults:
    """None or empty string should return the default URL."""

    def test_none(self):
        assert parse_ollama_host(None) == "http://127.0.0.1:11434"

    def test_empty_string(self):
        assert parse_ollama_host("") == "http://127.0.0.1:11434"


class TestParseOllamaHostIPv4:
    """IPv4 addresses with various formats."""

    def test_ip_only(self):
        assert parse_ollama_host("1.2.3.4") == "http://1.2.3.4:11434"

    def test_port_only(self):
        assert parse_ollama_host(":56789") == "http://127.0.0.1:56789"

    def test_ip_and_port(self):
        assert parse_ollama_host("1.2.3.4:56789") == "http://1.2.3.4:56789"

    def test_http_scheme(self):
        assert parse_ollama_host("http://1.2.3.4") == "http://1.2.3.4:80"

    def test_https_scheme(self):
        assert parse_ollama_host("https://1.2.3.4") == "https://1.2.3.4:443"

    def test_https_with_port(self):
        assert parse_ollama_host("https://1.2.3.4:56789") == "https://1.2.3.4:56789"


class TestParseOllamaHostDomain:
    """Domain names with various formats."""

    def test_domain_only(self):
        assert parse_ollama_host("example.com") == "http://example.com:11434"

    def test_domain_with_port(self):
        assert parse_ollama_host("example.com:56789") == "http://example.com:56789"

    def test_http_domain(self):
        assert parse_ollama_host("http://example.com") == "http://example.com:80"

    def test_https_domain(self):
        assert parse_ollama_host("https://example.com") == "https://example.com:443"

    def test_https_domain_with_port(self):
        assert parse_ollama_host("https://example.com:56789") == "https://example.com:56789"


class TestParseOllamaHostTrailingSlash:
    """Trailing slashes should be stripped."""

    def test_domain_trailing_slash(self):
        assert parse_ollama_host("example.com/") == "http://example.com:11434"

    def test_domain_port_trailing_slash(self):
        assert parse_ollama_host("example.com:56789/") == "http://example.com:56789"


class TestParseOllamaHostWithPath:
    """URLs with path components should preserve the path (without trailing slash)."""

    def test_domain_with_path(self):
        assert parse_ollama_host("example.com/path") == "http://example.com:11434/path"

    def test_domain_port_with_path(self):
        assert parse_ollama_host("example.com:56789/path") == "http://example.com:56789/path"

    def test_https_domain_port_with_path(self):
        assert (
            parse_ollama_host("https://example.com:56789/path") == "https://example.com:56789/path"
        )

    def test_domain_port_path_trailing_slash(self):
        assert parse_ollama_host("example.com:56789/path/") == "http://example.com:56789/path"


class TestParseOllamaHostIPv6:
    """IPv6 addresses in bracket notation."""

    def test_ipv6_only(self):
        assert parse_ollama_host("[0001:002:003:0004::1]") == "http://[0001:002:003:0004::1]:11434"

    def test_ipv6_with_port(self):
        assert (
            parse_ollama_host("[0001:002:003:0004::1]:56789")
            == "http://[0001:002:003:0004::1]:56789"
        )

    def test_http_ipv6(self):
        assert (
            parse_ollama_host("http://[0001:002:003:0004::1]") == "http://[0001:002:003:0004::1]:80"
        )

    def test_https_ipv6(self):
        assert (
            parse_ollama_host("https://[0001:002:003:0004::1]")
            == "https://[0001:002:003:0004::1]:443"
        )

    def test_https_ipv6_with_port(self):
        assert (
            parse_ollama_host("https://[0001:002:003:0004::1]:56789")
            == "https://[0001:002:003:0004::1]:56789"
        )

    def test_ipv6_trailing_slash(self):
        assert parse_ollama_host("[0001:002:003:0004::1]/") == "http://[0001:002:003:0004::1]:11434"

    def test_ipv6_port_trailing_slash(self):
        assert (
            parse_ollama_host("[0001:002:003:0004::1]:56789/")
            == "http://[0001:002:003:0004::1]:56789"
        )

    def test_ipv6_with_path(self):
        assert (
            parse_ollama_host("[0001:002:003:0004::1]/path")
            == "http://[0001:002:003:0004::1]:11434/path"
        )

    def test_ipv6_port_with_path(self):
        assert (
            parse_ollama_host("[0001:002:003:0004::1]:56789/path")
            == "http://[0001:002:003:0004::1]:56789/path"
        )

    def test_https_ipv6_port_with_path(self):
        assert (
            parse_ollama_host("https://[0001:002:003:0004::1]:56789/path")
            == "https://[0001:002:003:0004::1]:56789/path"
        )

    def test_ipv6_port_path_trailing_slash(self):
        assert (
            parse_ollama_host("[0001:002:003:0004::1]:56789/path/")
            == "http://[0001:002:003:0004::1]:56789/path"
        )

    def test_ipv6_all_interfaces(self):
        assert parse_ollama_host("[::]:11434") == "http://[::1]:11434"

    def test_ipv6_all_interfaces_no_port(self):
        assert parse_ollama_host("[::]") == "http://[::1]:11434"


class TestParseOllamaHostLocalhostVariants:
    """Common localhost variations users might set."""

    def test_localhost(self):
        assert parse_ollama_host("localhost") == "http://localhost:11434"

    def test_localhost_with_port(self):
        assert parse_ollama_host("localhost:11434") == "http://localhost:11434"

    def test_http_localhost(self):
        assert parse_ollama_host("http://localhost:11434") == "http://localhost:11434"

    def test_all_interfaces(self):
        assert parse_ollama_host("0.0.0.0:11434") == "http://127.0.0.1:11434"

    def test_all_interfaces_no_port(self):
        assert parse_ollama_host("0.0.0.0") == "http://127.0.0.1:11434"

    def test_all_interfaces_with_scheme(self):
        assert parse_ollama_host("http://0.0.0.0:11434") == "http://127.0.0.1:11434"

    def test_all_interfaces_https(self):
        assert parse_ollama_host("https://0.0.0.0:56789") == "https://127.0.0.1:56789"
