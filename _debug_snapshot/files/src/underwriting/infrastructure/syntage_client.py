# Ruta: src/underwriting/infrastructure/syntage_client.py
# Archivo: syntage_client.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List
import random
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings


def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=200, pool_maxsize=200)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


@dataclass
class SyntageClient:
    settings: Settings
    timeout_sec: int = 30

    def __post_init__(self) -> None:
        self._session = _build_session()

    def _headers(self) -> Dict[str, str]:
        return {
            "X-API-Key": self.settings.syntage_api_key,
            "Accept": "application/ld+json",
        }

    def _url(self, path: str) -> str:
        base = self.settings.syntage_base_url.rstrip("/")
        return f"{base}/{path.lstrip('/')}"

    def _get_json(self, path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        url = self._url(path)
        resp = self._session.get(url, headers=self._headers(), params=params, timeout=self.timeout_sec)
        if resp.status_code >= 400:
            raise RuntimeError(f"Syntage error {resp.status_code}: {resp.text}")
        return resp.json()

    def _looks_like_xml(self, content_type: str | None, body_preview: str) -> bool:
        ct = (content_type or "").lower()
        b = (body_preview or "").lstrip()

        if "xml" in ct:
            return True
        if b.startswith("<?xml"):
            return True
        if b.startswith("<cfdi:Comprobante") or b.startswith("<Comprobante"):
            return True
        if b.startswith("<") and not b.lower().startswith("<!doctype html") and not b.lower().startswith("<html"):
            return True
        return False

    def _normalize_invoice_id(self, id_or_path: str) -> str:
        s = str(id_or_path).strip()
        if not s:
            return ""
        if s.startswith("http://") or s.startswith("https://"):
            if "/invoices/" in s:
                s = s.split("/invoices/")[-1]
            s = s.strip("/")
        if "/invoices/" in s:
            s = s.split("/invoices/")[-1].strip("/")
        return s.strip("/")

    def _get_xml_bytes_with_jitter(
        self,
        url: str,
        headers: Dict[str, str],
        *,
        timeout: int,
        max_tries: int = 5,
    ) -> bytes:
        """
        Devuelve BYTES (resp.content) para NO corromper el XML con decoding de resp.text.
        """
        for i in range(1, max_tries + 1):
            try:
                resp = self._session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
                st = resp.status_code

                if 200 <= st < 300:
                    content = resp.content or b""
                    # preview seguro
                    preview = ""
                    try:
                        preview = content[:400].decode("utf-8", errors="ignore")
                    except Exception:
                        preview = ""
                    if self._looks_like_xml(resp.headers.get("content-type"), preview):
                        return content

                    # 200 pero NO parece XML => tratar como transitorio y reintentar
                    sleep_s = random.uniform(0.8, 1.4) * (2 ** (i - 1))
                    time.sleep(min(sleep_s, 12.0))
                    continue

                if st in (429, 500, 502, 503, 504):
                    sleep_s = random.uniform(0.8, 1.4) * (2 ** (i - 1))
                    time.sleep(min(sleep_s, 12.0))
                    continue

                return b""
            except requests.RequestException:
                sleep_s = random.uniform(0.8, 1.4) * (2 ** (i - 1))
                time.sleep(min(sleep_s, 12.0))

        return b""

    # ──────────────────────────────────────────────────────────────────────────
    # SAT
    # ──────────────────────────────────────────────────────────────────────────
    def get_tax_status_by_rfc(self, rfc: str) -> Dict[str, Any]:
        return self._get_json(f"/taxpayers/{rfc}/tax-status")

    # ──────────────────────────────────────────────────────────────────────────
    # CFDI: list + xml
    # ──────────────────────────────────────────────────────────────────────────
    def list_invoices(
        self,
        rfc: str,
        is_issuer: bool = True,
        date_from: date | None = None,
        date_to: date | None = None,
        items_per_page: int = 500,
        max_pages: int = 200,
    ) -> List[Dict[str, Any]]:
        base = f"/taxpayers/{rfc}/invoices"
        q0: Dict[str, Any] = {
            "itemsPerPage": min(int(items_per_page), 1000),
            "isIssuer": "true" if is_issuer else "false",
            "fields[*]": "*",
            "fields[issuer]": "*",
            "fields[receiver]": "*",
        }
        if date_from is not None:
            q0["issuedAt[after]"] = f"{date_from:%Y-%m-%d}T00:00:00Z"
        if date_to is not None:
            q0["issuedAt[before]"] = f"{date_to:%Y-%m-%d}T23:59:59Z"

        acc: List[Dict[str, Any]] = []
        next_lt: str | None = None

        for _ in range(int(max_pages)):
            q = dict(q0)
            if next_lt:
                q["id[lt]"] = next_lt

            raw = self._get_json(base, params=q)
            rows = raw.get("hydra:member", [])
            if not isinstance(rows, list) or not rows:
                break

            for r in rows:
                if isinstance(r, dict):
                    acc.append(r)

            last = rows[-1]
            cand = None
            if isinstance(last, dict):
                cand = last.get("id") or last.get("@id")
                if isinstance(cand, str) and "/invoices/" in cand:
                    cand = cand.split("/invoices/")[-1].strip("/")
            if not cand:
                break

            next_lt = str(cand)

            if len(rows) < q0["itemsPerPage"]:
                break

        return acc

    def get_cfdi_xml(self, id_or_path: str) -> bytes:
        """
        Devuelve BYTES del XML.
        """
        inv_id = self._normalize_invoice_id(id_or_path)
        if not inv_id:
            return b""

        path = f"/invoices/{inv_id}/cfdi"
        url = self._url(path)

        headers = dict(self._headers())
        headers["Accept"] = "application/xml, text/xml;q=0.9, */*;q=0.5"

        content = self._get_xml_bytes_with_jitter(
            url,
            headers,
            timeout=max(self.timeout_sec, 45),
            max_tries=5,
        )
        return content if content and b"<" in content else b""
