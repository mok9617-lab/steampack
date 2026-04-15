from __future__ import annotations

import json
import os
import socket
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import load_settings


@dataclass
class CheckResult:
    target: str
    ok: bool
    detail: str


def _http_check(url: str, timeout: float = 6.0) -> CheckResult:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return CheckResult(target=url, ok=True, detail=f"status={getattr(resp, 'status', 'ok')}")
    except urllib.error.HTTPError as exc:
        # HTTP errors still mean network path is reachable.
        return CheckResult(target=url, ok=True, detail=f"http_error={exc.code}")
    except Exception as exc:
        return CheckResult(target=url, ok=False, detail=str(exc))


def _socket_tls_check(host: str, port: int = 443, timeout: float = 6.0) -> CheckResult:
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            context = ssl.create_default_context()
            with context.wrap_socket(sock, server_hostname=host):
                return CheckResult(target=f"{host}:{port}", ok=True, detail="tls_ok")
    except Exception as exc:
        return CheckResult(target=f"{host}:{port}", ok=False, detail=str(exc))


def _openai_api_check(api_key: str | None, timeout: float = 8.0) -> CheckResult:
    if not api_key:
        return CheckResult(target="openai_api_key", ok=False, detail="missing")
    req = urllib.request.Request(
        "https://api.openai.com/v1/models",
        method="GET",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            return CheckResult(target="openai_api", ok=(200 <= status < 300), detail=f"status={status}")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            return CheckResult(target="openai_api", ok=False, detail=f"auth_error={exc.code}")
        return CheckResult(target="openai_api", ok=True, detail=f"http_error={exc.code}")
    except Exception as exc:
        return CheckResult(target="openai_api", ok=False, detail=str(exc))


def run_network_diagnosis() -> dict:
    settings = load_settings()
    results: list[CheckResult] = []

    results.append(_socket_tls_check("huggingface.co"))
    results.append(_http_check("https://huggingface.co"))
    results.append(_socket_tls_check("api.openai.com"))
    results.append(_http_check("https://api.openai.com"))
    results.append(_openai_api_check(settings.openai_api_key))

    hf_offline = os.getenv("HF_HUB_OFFLINE", "")
    tf_offline = os.getenv("TRANSFORMERS_OFFLINE", "")
    summary = {
        "all_ok": all(r.ok for r in results),
        "results": [r.__dict__ for r in results],
        "env": {
            "HF_HUB_OFFLINE": hf_offline,
            "TRANSFORMERS_OFFLINE": tf_offline,
            "OPENAI_MODEL": settings.openai_model,
            "HAS_OPENAI_KEY": bool(settings.openai_api_key),
        },
    }
    return summary


def print_network_diagnosis() -> None:
    report = run_network_diagnosis()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_ok"]:
        print("\n권장 조치:")
        print("1) 방화벽/보안프로그램에서 python.exe 아웃바운드 허용")
        print("2) 회사/학교망이면 개인 핫스팟으로 재시도")
        print("3) HF 오프라인 모드 필요 시 .env에 HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 설정")
        print("4) OpenAI 401이면 API 키 재발급/재설정 확인")
