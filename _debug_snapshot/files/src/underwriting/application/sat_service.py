#sat_service.py

from __future__ import annotations

from typing import Any, Dict, Iterable

from underwriting.domain.models import TaxStatus, EconomicActivity, TaxRegime
from underwriting.infrastructure.syntage_client import SyntageClient


def _to_float_percentage(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _iter_members(raw: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    Normaliza respuesta JSON-LD:
    - Si viene como colección Hydra: itera hydra:member
    - Si viene como objeto simple: itera solo ese objeto
    """
    if isinstance(raw, dict) and isinstance(raw.get("hydra:member"), list):
        for item in raw["hydra:member"]:
            if isinstance(item, dict):
                yield item
    else:
        if isinstance(raw, dict):
            yield raw


def _extract_tax_status_uuid(raw: Dict[str, Any]) -> str | None:
    iri = raw.get("@id") or ""
    if isinstance(iri, str) and "/tax-status/" in iri:
        uuid = iri.split("/tax-status/")[-1].strip("/")
        return uuid or None
    return None


class SatService:
    def __init__(self, client: SyntageClient):
        self.client = client

    def get_tax_status(self, rfc: str) -> TaxStatus:
        raw: Dict[str, Any] = self.client.get_tax_status_by_rfc(rfc)

        # Si la respuesta no trae economicActivities en ningún member,
        # intentamos fallback por UUID (si existe) y volvemos a intentar.
        members = list(_iter_members(raw))
        has_any_activities = any(
            (m.get("economicActivities") or (m.get("company") or {}).get("economicActivities") or []) for m in members
        )
        if not has_any_activities:
            uuid = _extract_tax_status_uuid(members[0]) if members else _extract_tax_status_uuid(raw)
            if uuid:
                raw = self.client.get_tax_status_by_uuid(uuid)
                members = list(_iter_members(raw))

        # Extraer y aplanar activities/regimes desde todos los members
        econ: list[EconomicActivity] = []
        regimes: list[TaxRegime] = []

        for m in members:
            activities_raw = (
                m.get("economicActivities")
                or (m.get("company") or {}).get("economicActivities")
                or (m.get("person") or {}).get("economicActivities")
                or (m.get("taxStatus") or {}).get("economicActivities")
                or []
            )

            regimes_raw = (
                m.get("taxRegimes")
                or (m.get("company") or {}).get("taxRegimes")
                or (m.get("person") or {}).get("taxRegimes")
                or (m.get("taxStatus") or {}).get("taxRegimes")
                or []
            )

            for a in activities_raw or []:
                if not isinstance(a, dict):
                    continue
                econ.append(
                    EconomicActivity(
                        name=a.get("name", ""),
                        order=int(a["order"]) if a.get("order") is not None else None,
                        percentage=_to_float_percentage(a.get("percentage")),
                        startDate=a.get("startDate") or a.get("startAt") or a.get("startedAt"),
                        endDate=a.get("endDate") or a.get("endAt"),
                    )
                )

            for r in regimes_raw or []:
                if not isinstance(r, dict):
                    continue
                regimes.append(
                    TaxRegime(
                        code=str(r.get("code")) if r.get("code") is not None else None,
                        name=r.get("name"),
                        startDate=r.get("startDate") or r.get("startAt"),
                        endDate=r.get("endDate") or r.get("endAt"),
                    )
                )

        # Deduplicar (por campos estables)
        def econ_key(x: EconomicActivity) -> tuple:
            return (x.order, x.name, x.startDate, x.endDate, x.percentage)

        def regime_key(x: TaxRegime) -> tuple:
            return (x.code, x.name, x.startDate, x.endDate)

        econ_dedup = list({econ_key(x): x for x in econ}.values())
        regimes_dedup = list({regime_key(x): x for x in regimes}.values())

        econ_sorted = sorted(econ_dedup, key=lambda x: (x.order is None, x.order))
        # Para regímenes suele ser útil por fecha de inicio
        regimes_sorted = sorted(regimes_dedup, key=lambda x: (x.startDate is None, x.startDate))

        # Elegir rfc/status desde el primer member que lo tenga
        rfc_out = rfc
        status_out = None
        for m in members:
            if isinstance(m.get("rfc"), str) and m["rfc"].strip():
                rfc_out = m["rfc"].strip()
                break

        for m in members:
            if m.get("status") is not None:
                status_out = m.get("status")
                break

        return TaxStatus(
            rfc=rfc_out,
            status=status_out,
            economicActivities=econ_sorted,
            taxRegimes=regimes_sorted,
        )
