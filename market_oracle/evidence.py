from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_REGISTRY_PATH = ROOT / "configs" / "evidence_registry_v1.json"
REGISTRY_HASH_METHOD = "sha256(canonical_json(registry_without_top_level_registry_hash))"


class EvidenceRegistryError(ValueError):
    """Raised when the versioned evidence registry is invalid or ambiguous."""


class ForecastAvailability(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class HistoricalEvidence(str, Enum):
    UNTESTED = "UNTESTED"
    NO_EDGE = "NO_EDGE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"


class ForwardEvidence(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED_INCONCLUSIVE = "COMPLETED_INCONCLUSIVE"
    COMPLETED_SUPPORTIVE = "COMPLETED_SUPPORTIVE"


class EvidenceScope(str, Enum):
    AGGREGATE_UNIVERSE = "AGGREGATE_UNIVERSE"
    SYMBOL_SPECIFIC = "SYMBOL_SPECIFIC"


@dataclass(frozen=True)
class EvidenceAssessment:
    symbol: str
    horizon: int
    forecast_availability: ForecastAvailability
    historical_evidence: HistoricalEvidence
    forward_evidence: ForwardEvidence
    evidence_scope: EvidenceScope | None
    forward_scope: EvidenceScope | None
    historical_protocol_id: str | None
    forward_protocol_id: str | None
    provenance: dict[str, Any]


@dataclass(frozen=True)
class EvidenceCopy:
    icon: str
    title: str
    summary: str
    detail: str


def canonical_registry_json(registry: dict[str, Any]) -> str:
    """Canonical JSON used by the registry hash contract.

    Only the top-level ``registry_hash`` field is omitted. Nested provenance fields
    remain part of the protected payload.
    """
    payload = dict(registry)
    payload.pop("registry_hash", None)
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceRegistryError(f"Evidence registry is not canonical JSON: {exc}") from exc


def registry_hash(registry: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_registry_json(registry).encode("utf-8")).hexdigest()


def _normalized_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _claim_keys(
    claims: list[dict[str, Any]],
    *,
    status_enum: type[Enum],
    section: str,
) -> set[tuple[str, int]]:
    seen: set[tuple[str, int]] = set()
    claim_ids: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict):
            raise EvidenceRegistryError(f"{section} claim must be an object.")
        claim_id = str(claim.get("claim_id") or "").strip()
        protocol_id = str(claim.get("protocol_id") or "").strip()
        if not claim_id or not protocol_id:
            raise EvidenceRegistryError(f"{section} claim requires claim_id and protocol_id.")
        if claim_id in claim_ids:
            raise EvidenceRegistryError(f"Duplicate {section} claim_id: {claim_id}")
        claim_ids.add(claim_id)
        try:
            horizon = int(claim.get("horizon"))
        except (TypeError, ValueError) as exc:
            raise EvidenceRegistryError(f"Invalid horizon in {claim_id}.") from exc
        if horizon <= 0:
            raise EvidenceRegistryError(f"Invalid horizon in {claim_id}: {horizon}")
        try:
            status_enum(str(claim.get("status")))
            scope = EvidenceScope(str(claim.get("evidence_scope")))
        except ValueError as exc:
            raise EvidenceRegistryError(f"Invalid status or evidence_scope in {claim_id}.") from exc
        symbols = claim.get("exact_symbols")
        if not isinstance(symbols, list) or not symbols:
            raise EvidenceRegistryError(f"{claim_id} requires non-empty exact_symbols.")
        normalized = [_normalized_symbol(symbol) for symbol in symbols]
        if any(not symbol for symbol in normalized) or normalized != symbols:
            raise EvidenceRegistryError(f"{claim_id} exact_symbols must be canonical uppercase symbols.")
        if len(set(normalized)) != len(normalized):
            raise EvidenceRegistryError(f"Duplicate exact_symbols inside {claim_id}.")
        if scope is EvidenceScope.SYMBOL_SPECIFIC and len(normalized) != 1:
            raise EvidenceRegistryError(f"SYMBOL_SPECIFIC claim {claim_id} must contain exactly one symbol.")
        if not str(claim.get("evidence_updated_at") or "").strip():
            raise EvidenceRegistryError(f"{claim_id} requires evidence_updated_at provenance.")
        if section == "historical":
            artifact_refs = claim.get("artifact_refs")
            if not isinstance(artifact_refs, list) or not artifact_refs:
                raise EvidenceRegistryError(f"{claim_id} requires artifact_refs provenance.")
            for artifact in artifact_refs:
                if not isinstance(artifact, dict) or not str(artifact.get("path") or "").strip():
                    raise EvidenceRegistryError(f"{claim_id} has an artifact without a path.")
                if not _is_sha256(artifact.get("sha256")):
                    raise EvidenceRegistryError(f"{claim_id} artifact requires a canonical sha256.")
        else:
            required = (
                "candidate_manifest_hash",
                "forward_universe_hash",
                "ledger_id",
                "ledger_path",
                "first_event_hash",
                "started_at",
            )
            missing = [field for field in required if not str(claim.get(field) or "").strip()]
            if missing:
                raise EvidenceRegistryError(f"{claim_id} missing Forward start provenance: {missing}")
            for field in ("candidate_manifest_hash", "forward_universe_hash", "first_event_hash"):
                if not _is_sha256(claim.get(field)):
                    raise EvidenceRegistryError(f"{claim_id} requires canonical {field}.")
        for symbol in normalized:
            key = (symbol, horizon)
            if key in seen:
                raise EvidenceRegistryError(
                    f"Overlapping {section} claims for exact symbol+horizon: {symbol} {horizon}"
                )
            seen.add(key)
    return seen


def validate_evidence_registry(registry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(registry, dict):
        raise EvidenceRegistryError("Evidence registry root must be an object.")
    if registry.get("schema_version") != 1:
        raise EvidenceRegistryError(f"Unsupported evidence registry schema: {registry.get('schema_version')}")
    if registry.get("hash_method") != REGISTRY_HASH_METHOD:
        raise EvidenceRegistryError(f"Unsupported evidence registry hash method: {registry.get('hash_method')}")
    expected = registry.get("registry_hash")
    actual = registry_hash(registry)
    if not isinstance(expected, str) or expected != actual:
        raise EvidenceRegistryError(f"Evidence registry hash mismatch: expected={expected} actual={actual}")
    historical_claims = registry.get("historical_claims")
    forward_claims = registry.get("forward_claims")
    if not isinstance(historical_claims, list) or not isinstance(forward_claims, list):
        raise EvidenceRegistryError("Evidence registry claims must be lists.")
    _claim_keys(
        historical_claims,
        status_enum=HistoricalEvidence,
        section="historical",
    )
    _claim_keys(
        forward_claims,
        status_enum=ForwardEvidence,
        section="forward",
    )
    return registry


def load_evidence_registry(path: Path = EVIDENCE_REGISTRY_PATH) -> dict[str, Any]:
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceRegistryError(f"Cannot load evidence registry {path}: {exc}") from exc
    return validate_evidence_registry(registry)


def verify_evidence_sources(
    registry: dict[str, Any] | None = None,
    *,
    root: Path = ROOT,
) -> dict[str, int]:
    """Explicitly audit source artifacts where local research outputs exist.

    This verifier is intentionally not called by the application or the standard
    registry loader because research outputs and the live ledger are gitignored.
    """
    resolved_registry = load_evidence_registry() if registry is None else validate_evidence_registry(registry)
    verified_artifacts = 0
    for claim in resolved_registry["historical_claims"]:
        for artifact in claim["artifact_refs"]:
            path = root / artifact["path"]
            if not path.is_file():
                raise EvidenceRegistryError(f"Evidence source artifact is unavailable: {artifact['path']}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != artifact["sha256"]:
                raise EvidenceRegistryError(
                    f"Evidence source artifact hash mismatch: {artifact['path']}"
                )
            verified_artifacts += 1

    verified_forward_checkpoints = 0
    for claim in resolved_registry["forward_claims"]:
        path = root / claim["ledger_path"]
        if not path.is_file():
            raise EvidenceRegistryError(f"Forward source ledger is unavailable: {claim['ledger_path']}")
        try:
            first_line = next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            first_event = json.loads(first_line)
        except (StopIteration, OSError, json.JSONDecodeError) as exc:
            raise EvidenceRegistryError(f"Cannot read Forward start checkpoint: {claim['ledger_path']}") from exc
        expected = {
            "event_hash": claim["first_event_hash"],
            "event_time_utc": claim["started_at"],
            "candidate_manifest_hash": claim["candidate_manifest_hash"],
        }
        if any(first_event.get(field) != value for field, value in expected.items()):
            raise EvidenceRegistryError(f"Forward start checkpoint mismatch: {claim['ledger_path']}")
        verified_forward_checkpoints += 1
    return {
        "artifacts": verified_artifacts,
        "forward_checkpoints": verified_forward_checkpoints,
    }


def _matching_claim(
    claims: list[dict[str, Any]],
    symbol: str,
    horizon: int,
) -> dict[str, Any] | None:
    return next(
        (
            claim
            for claim in claims
            if int(claim["horizon"]) == horizon and symbol in claim["exact_symbols"]
        ),
        None,
    )


def resolve_evidence(
    symbol: str,
    horizon: int,
    available_horizons: Iterable[int],
    registry: dict[str, Any] | None = None,
) -> EvidenceAssessment:
    """Resolve evidence solely from the reviewed registry and exact symbol scope."""
    resolved_registry = load_evidence_registry() if registry is None else validate_evidence_registry(registry)
    normalized_symbol = _normalized_symbol(symbol)
    requested_horizon = int(horizon)
    available = {int(value) for value in available_horizons}
    historical = _matching_claim(
        resolved_registry["historical_claims"],
        normalized_symbol,
        requested_horizon,
    )
    forward = _matching_claim(
        resolved_registry["forward_claims"],
        normalized_symbol,
        requested_horizon,
    )
    historical_status = HistoricalEvidence.UNTESTED
    evidence_scope = None
    historical_protocol_id = None
    if historical is not None:
        historical_status = HistoricalEvidence(historical["status"])
        evidence_scope = EvidenceScope(historical["evidence_scope"])
        historical_protocol_id = str(historical["protocol_id"])
    forward_status = ForwardEvidence.NOT_STARTED
    forward_scope = None
    forward_protocol_id = None
    if forward is not None:
        forward_status = ForwardEvidence(forward["status"])
        forward_scope = EvidenceScope(forward["evidence_scope"])
        forward_protocol_id = str(forward["protocol_id"])
    return EvidenceAssessment(
        symbol=normalized_symbol,
        horizon=requested_horizon,
        forecast_availability=(
            ForecastAvailability.AVAILABLE
            if requested_horizon in available
            else ForecastAvailability.UNAVAILABLE
        ),
        historical_evidence=historical_status,
        forward_evidence=forward_status,
        evidence_scope=evidence_scope,
        forward_scope=forward_scope,
        historical_protocol_id=historical_protocol_id,
        forward_protocol_id=forward_protocol_id,
        provenance={
            "registry_id": resolved_registry.get("registry_id"),
            "registry_hash": resolved_registry.get("registry_hash"),
            "historical_claim": None if historical is None else dict(historical),
            "forward_claim": None if forward is None else dict(forward),
        },
    )


def evidence_copy(assessment: EvidenceAssessment) -> EvidenceCopy:
    """Build trust copy from evidence status and scope, never from live forecast metrics."""
    symbol = assessment.symbol
    horizon = assessment.horizon
    historical = assessment.historical_evidence
    scope = assessment.evidence_scope
    historical_claim = assessment.provenance.get("historical_claim") or {}

    if assessment.forecast_availability is ForecastAvailability.UNAVAILABLE:
        icon = "⚪"
        title = "Prognoza niedostępna"
        summary = f"Bieżąca analiza nie zawiera forecastu {horizon}d."
        detail = f"Bieżąca analiza nie zawiera forecastu dla horyzontu {horizon}."
    elif historical is HistoricalEvidence.NO_EDGE and scope is EvidenceScope.AGGREGATE_UNIVERSE:
        icon = "⚪"
        title = "Brak wykazanej przewagi w protokole zbiorczym"
        summary = f"{symbol} należało do badanego koszyka; status nie jest indywidualną oceną instrumentu."
        detail = (
            f"{symbol} było częścią badanego koszyka. Wynik całego protokołu {horizon}d "
            f"nie wykazał przewagi; nie jest to osobna ocena skuteczności {symbol}."
        )
    elif historical is HistoricalEvidence.NO_EDGE and scope is EvidenceScope.SYMBOL_SPECIFIC:
        icon = "⚪"
        title = "Brak wykazanej przewagi dla instrumentu"
        summary = f"Osobny protokół {symbol} nie wykazał przewagi na horyzoncie {horizon}d."
        detail = f"Osobny protokół dla {symbol} na horyzoncie {horizon}d nie wykazał przewagi."
    elif historical is HistoricalEvidence.INSUFFICIENT_EVIDENCE and scope is EvidenceScope.AGGREGATE_UNIVERSE:
        icon = "🧪"
        title = "Za mało dowodów w badanym zakresie"
        summary = f"{symbol} należało do badanego koszyka, ale niezależna próbka była zbyt mała."
        detail = (
            f"{symbol} było częścią badanego koszyka. Liczba niezależnych przypadków "
            "była zbyt mała do wiarygodnego wniosku; nie jest to ocena skuteczności samego instrumentu."
        )
    elif historical is HistoricalEvidence.INSUFFICIENT_EVIDENCE:
        icon = "🧪"
        title = "Za mało dowodów dla instrumentu"
        summary = f"Osobna próbka {symbol} była zbyt mała do wiarygodnego wniosku."
        detail = f"Osobna próbka {symbol} była zbyt mała do wiarygodnego wniosku."
    elif historical is HistoricalEvidence.RESEARCH_CANDIDATE and scope is EvidenceScope.AGGREGATE_UNIVERSE:
        icon = "🔬"
        title = "Kandydat badawczy"
        summary = f"Wynik pochodzi z protokołu zbiorczego, nie z indywidualnej walidacji {symbol}."
        detail = (
            f"{symbol} należy do universe Candidate v1. Historyczny wynik protokołu zbiorczego "
            "był obiecujący na ograniczonej próbce; nie jest to indywidualnie potwierdzony edge."
        )
    elif historical is HistoricalEvidence.RESEARCH_CANDIDATE:
        icon = "🔬"
        title = "Kandydat badawczy dla instrumentu"
        summary = f"Osobny protokół {symbol} był obiecujący, ale wymaga dowodu prospektywnego."
        detail = f"Osobny historyczny protokół {symbol} był obiecujący, ale nadal wymaga dowodu prospektywnego."
    elif str(historical_claim.get("protocol_state") or "") == "PREREGISTERED_NOT_RUN":
        icon = "🧪"
        title = "Eksperymentalny forecast · protokół jeszcze nieuruchomiony"
        summary = f"{symbol} jest w prerejestrowanym koszyku, ale właściwa walidacja jeszcze nie ruszyła."
        detail = (
            f"{symbol} należy do prerejestrowanego koszyka {horizon}d, ale właściwa walidacja "
            "nie została wykonana. Nie ma jeszcze wyniku evidence."
        )
    else:
        icon = "🧪"
        title = "Eksperymentalny forecast"
        summary = f"Brak zarejestrowanej walidacji dla {symbol} na horyzoncie {horizon}d."
        detail = f"Brak zarejestrowanej walidacji dla {symbol} na horyzoncie {horizon}d."

    if assessment.forward_evidence is ForwardEvidence.IN_PROGRESS:
        title += " · Forward trwa"
        detail += " Prospektywny test Forward tego koszyka trwa i nie stanowi jeszcze potwierdzonej przewagi."
    elif assessment.forward_evidence is ForwardEvidence.COMPLETED_INCONCLUSIVE:
        detail += " Zakończony test Forward nie dał rozstrzygającego wyniku."
    elif assessment.forward_evidence is ForwardEvidence.COMPLETED_SUPPORTIVE:
        detail += " Zakończony test Forward wsparł hipotezę, ale nie gwarantuje przyszłych wyników."
    return EvidenceCopy(icon=icon, title=title, summary=summary, detail=detail)
