from __future__ import annotations

import json
import math
import re
import sqlite3
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DATASTORE_DIR, DEFAULT_VERSION
from .logging_utils import get_logger
from .lmstudio import LMStudioError, embeddings as lm_embeddings, rerank as lm_rerank

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "new",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    # Product boilerplate terms (present in most headings).
    "luxriot",
    "evo",
}

_GRANULARITY_WEIGHTS = {"p": 1.05, "s": 1.0, "t": 0.9}
_DUPLICATE_GUIDE_DOCS = {
    "luxriot-evo-global-administration-guide",
    "luxriot-evo-standalone",
}


def _has_any(terms: set[str], candidates: set[str]) -> bool:
    return bool(terms.intersection(candidates))


def _text_has_any(text: str, candidates: tuple[str, ...]) -> bool:
    hay = str(text or "").lower()
    return any(candidate in hay for candidate in candidates)


def expand_luxriot_query(query: str) -> str:
    """Add Luxriot domain aliases for retrieval without changing the user prompt.

    The manuals describe cameras primarily as devices/channels. Short support
    questions often say "add camera", which otherwise ranks mobile "Stream camera"
    pages above the actual Console device setup pages.
    """
    base = str(query or "").strip()
    terms = set(tokenize(base))
    additions: list[str] = []

    camera_terms = {"camera", "cameras", "cam", "cams", "ipcamera", "webcam"}
    device_action_terms = {
        "add",
        "adding",
        "connect",
        "connecting",
        "configure",
        "configuring",
        "setup",
        "set",
        "install",
        "register",
        "attach",
    }
    license_terms = {
        "activate",
        "activation",
        "deactivate",
        "evaluation",
        "license",
        "licence",
        "licensing",
        "trial",
    }
    install_terms = {
        "install",
        "installation",
        "initialize",
        "initialization",
        "migration",
        "prerequisite",
        "prerequisites",
        "uninstall",
        "update",
        "upgrade",
    }
    storage_terms = {
        "archive",
        "backup",
        "disk",
        "drive",
        "record",
        "recording",
        "recordings",
        "retention",
        "schedule",
        "storage",
    }
    user_terms = {
        "access",
        "administrator",
        "admin",
        "ldap",
        "oauth",
        "permission",
        "permissions",
        "role",
        "roles",
        "user",
        "users",
    }
    integration_terms = {
        "gallagher",
        "integriti",
        "modbus",
        "mqtt",
        "opc",
        "satel",
    }
    export_terms = {
        "audit",
        "bookmark",
        "bookmarks",
        "export",
        "report",
        "reports",
        "snapshot",
        "snapshots",
        "vca",
    }
    layout_terms = {"layout", "layouts", "map", "maps", "videowall", "webpage", "webpages"}

    if _has_any(terms, camera_terms):
        additions.append("device devices channel channels video source IP camera network camera")
    if _has_any(terms, camera_terms) and _has_any(terms, device_action_terms):
        additions.append(
            "Add Devices Manually Add Devices Using Autodiscovery Add single device "
            "Device Autodiscovery + New device Configuration Devices"
        )
    if "onvif" in terms:
        additions.append("ONVIF Compatible ONVIF driver device model generic autodiscovery")
    if _has_any(terms, {"show", "view", "display", "live"}):
        additions.append("Luxriot EVO Monitor Resources channels live view drag and drop")
    if _has_any(terms, {"folder", "folders"}) or _text_has_any(
        base,
        ("папк", "фолдер", "магазин", "магазине", "админ", "администратор", "доступ"),
    ):
        additions.append(
            "Purpose and Operation Description of Folders folder folders folder tree "
            "Creating Folders Moving resources Granting access rights Folder access "
            "User access Configuration Users parent folder subfolders Link folder"
        )
    if _text_has_any(base, ("лиценз", "активац", "пробн")):
        additions.append(
            "License Activation Online Activation Offline Activation Evaluation License "
            "Trial License Activation Management license key activation file"
        )
    if _text_has_any(base, ("установ", "обнов", "миграц", "инициал", "предвар", "удал")):
        additions.append(
            "Getting Started Prerequisites installation initialization Console installation "
            "software update uninstall remote upgrade migration database import"
        )
    if _has_any(terms, {"disk", "drive", "retention"}) or _text_has_any(
        base,
        ("диск", "хранилищ", "архив", "запис", "бэкап", "резерв", "retention"),
    ):
        additions.append(
            "Prerequisites recording locations RAID free space system disk not recommended "
            "indexing defragmentation Storage archive recording configurations retention disk space"
        )
    if _has_any(terms, storage_terms) and _has_any(
        terms,
        {"defragmentation", "disk", "free", "indexing", "minimum", "raid", "recommendations", "recommended", "space", "system"},
    ):
        additions.append(
            "Prerequisites recording storage recommendations RAID free space recording location "
            "system disk not recommended indexing defragmentation"
        )
    if "installation" in terms and "wizard" in terms and _has_any(terms, {"component", "optional"}):
        additions.append("Standalone Installation installation wizard optional component Luxriot EVO Monitor")
    if _has_any(terms, {"license", "licence", "licensing"}) and _has_any(
        terms,
        {"channel", "channels", "discontinued", "free", "type", "types"},
    ):
        additions.append(
            "License Activation license channel types video channels VA CrossLink data channels "
            "free Luxriot EVO license discontinued version 1.22.0"
        )
    if "offline" in terms and "activation" in terms:
        additions.append("Offline Activation three steps system.bin license.dat activation file")
    if _has_any(terms, {"upgrade", "upgrading", "uninstall"}) and _has_any(
        terms,
        {"files", "locked", "process", "retry", "use"},
    ):
        additions.append("Software Update and Uninstall files in use locked process Abort Retry Ignore")
    if _has_any(terms, {"upgrade", "upgrading"}) and _has_any(terms, {"before", "checks", "recommended"}):
        additions.append("Software Update and Uninstall before starting software upgrade checks Windows updates backup")
    if _has_any(terms, {"administrator", "ldap", "oauth", "permission", "permissions", "role", "roles"}) or _text_has_any(
        base,
        ("пользоват", "админ", "доступ", "прав", "роль", "ldap", "oauth"),
    ):
        additions.append(
            "User Management Permissions and Membership Active Directory LDAP "
            "Anonymous User Two Factor Authentication OAuth Folder access Temporary Permissions"
        )
    if "modbus" in terms:
        additions.append("Modbus setup Modbus functionality Modbus TCP register connection device data")
    elif "mqtt" in terms:
        additions.append("MQTT setup MQTT broker topic connection external services data source")
    elif "opc" in terms:
        additions.append("OPC Client OPC functionality OPC server connection external services data source")
    elif "gallagher" in terms:
        additions.append("Gallagher security system connection controller access control configuration")
    elif "integriti" in terms:
        additions.append("Inner Range Integriti security system access control configuration")
    elif "satel" in terms:
        additions.append("Satel INTEGRA security system integration configuration")
    elif _has_any(terms, integration_terms) or _text_has_any(
        base,
        ("модбас", "интеграц", "сател", "охран", "безопасн"),
    ):
        additions.append(
            "External Services Data Sources Security Systems Modbus MQTT OPC Gallagher "
            "Inner Range Integriti Satel external metadata"
        )
    if _has_any(terms, export_terms) or _text_has_any(
        base,
        ("отчет", "отчёт", "экспорт", "снимок", "заклад", "аудит"),
    ):
        additions.append(
            "Reports Audit VCA reports Video Snapshot Export Case Export Bookmarks "
            "library export report generation"
        )
    if _has_any(terms, layout_terms) or _text_has_any(base, ("расклад", "карта", "видеостен", "веб-страниц")):
        additions.append(
            "Layouts layout templates manage layouts Maps manage maps Webpages Video Wall "
            "visual groups user buttons live view"
        )

    if not additions:
        return base

    seen = set(terms)
    extra_tokens: list[str] = []
    for text in additions:
        for token in tokenize(text):
            if token in seen:
                continue
            seen.add(token)
            extra_tokens.append(token)
    if not extra_tokens:
        return base
    return f"{base}\nRelated Luxriot terms: {' '.join(extra_tokens)}"


def luxriot_summary_queries(query: str, retrieval_query: str) -> list[str]:
    """Build facet queries for the page-level summary router.

    The summary index is useful as a coarse page gate, but a single short user
    question can contain several workflow facets: add the camera in Console,
    choose ONVIF/model settings, then view the channel in Monitor. Separate
    queries keep one facet from crowding out the others before chunk retrieval.
    """
    base = str(retrieval_query or query or "").strip()
    if not base:
        return []

    terms = set(tokenize(f"{query}\n{retrieval_query}"))
    out: list[str] = [base]
    camera_terms = {"camera", "cameras", "cam", "cams", "ipcamera", "webcam"}
    device_action_terms = {
        "add",
        "adding",
        "connect",
        "connecting",
        "configure",
        "configuring",
        "setup",
        "set",
        "install",
        "register",
        "attach",
    }
    monitor_terms = {"show", "view", "display", "live"}
    license_terms = {
        "activate",
        "activation",
        "deactivate",
        "evaluation",
        "license",
        "licence",
        "licensing",
        "trial",
    }
    install_terms = {
        "install",
        "installation",
        "initialize",
        "initialization",
        "migration",
        "prerequisite",
        "prerequisites",
        "uninstall",
        "update",
        "upgrade",
    }
    storage_terms = {
        "archive",
        "backup",
        "disk",
        "drive",
        "record",
        "recording",
        "recordings",
        "retention",
        "schedule",
        "storage",
    }
    user_terms = {
        "administrator",
        "ldap",
        "oauth",
        "permission",
        "permissions",
        "role",
        "roles",
    }
    integration_terms = {
        "gallagher",
        "integriti",
        "modbus",
        "mqtt",
        "opc",
        "satel",
    }
    export_terms = {
        "audit",
        "bookmark",
        "bookmarks",
        "export",
        "report",
        "reports",
        "snapshot",
        "snapshots",
        "vca",
    }
    layout_terms = {"layout", "layouts", "map", "maps", "videowall", "webpage", "webpages"}

    if _has_any(terms, camera_terms) and _has_any(terms, device_action_terms):
        out.append(
            "Luxriot EVO Console Add Devices Manually Add single device "
            "Add Devices Using Autodiscovery Device Autodiscovery + New device "
            "Configuration Devices channels"
        )
    if "onvif" in terms:
        out.append("Manage Devices Device Drivers Models ONVIF Driver ONVIF Compatible")
    if _has_any(terms, monitor_terms):
        out.append(
            "Luxriot EVO Monitor Live View Section Left Resources channels "
            "layout templates drag and drop double-click connected servers"
        )
    if _has_any(terms, {"folder", "folders"}):
        out.append(
            "Purpose and Operation Description of Folders Creating Folders "
            "Moving resources Granting access rights Folder access User access "
            "Configuration Users folder tree subfolders link folder"
        )
    if _has_any(terms, license_terms):
        out.append(
            "License Activation Online Activation Offline Activation Evaluation License "
            "Trial License Activation Management"
        )
    if _has_any(terms, install_terms):
        out.append(
            "Getting Started Prerequisites installation initialization Console installation "
            "software update uninstall remote upgrade migration database import"
        )
    if _has_any(terms, storage_terms):
        out.append(
            "Storage archive recording profiles policies assign recording configurations "
            "Archive Backup retention schedule disk space"
        )
    if _has_any(terms, user_terms):
        out.append(
            "User Management Permissions and Membership Active Directory LDAP "
            "Anonymous User Two Factor Authentication OAuth Folder access Temporary Permissions"
        )
    if _has_any(terms, integration_terms):
        out.append(
            "External Services Data Sources Security Systems Modbus MQTT OPC Gallagher "
            "Inner Range Integriti Satel external metadata"
        )
    if _has_any(terms, export_terms):
        out.append(
            "Reports Audit VCA reports Video Snapshot Export Case Export Bookmarks "
            "library export report generation"
        )
    if _has_any(terms, layout_terms):
        out.append(
            "Layouts layout templates manage layouts Maps manage maps Webpages Video Wall "
            "visual groups user buttons live view"
        )

    seen: set[str] = set()
    deduped: list[str] = []
    for q in out:
        key = " ".join(tokenize(q))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    return deduped


def _workflow_page_cap(query_terms: set[str], configured_max_per_page: int) -> int:
    if _has_any(query_terms, {"folder", "folders"}):
        return max(int(configured_max_per_page or 0), 12)

    camera_terms = {"camera", "cameras", "cam", "cams", "ipcamera", "webcam"}
    workflow_terms = {
        "add",
        "adding",
        "connect",
        "connecting",
        "configure",
        "configuring",
        "setup",
        "install",
        "register",
        "attach",
        "monitor",
        "show",
        "view",
        "display",
        "live",
    }
    if not (_has_any(query_terms, camera_terms) and _has_any(query_terms, workflow_terms)):
        return int(configured_max_per_page or 0)
    if configured_max_per_page <= 0:
        return 3
    return min(int(configured_max_per_page), 3)


def _intent_page_multiplier(page_id: str, heading_path: list[str], query_terms: set[str]) -> float:
    camera_terms = {"camera", "cameras", "cam", "cams", "ipcamera", "webcam"}
    device_action_terms = {
        "add",
        "adding",
        "connect",
        "connecting",
        "configure",
        "configuring",
        "setup",
        "install",
        "register",
        "attach",
    }
    mobile_terms = {"mobile", "phone", "ios", "android", "stream", "streaming"}
    event_terms = {"event", "events", "rule", "rules", "trigger", "triggers", "analytics", "vca"}
    monitor_terms = {"show", "view", "display", "live"}
    folder_terms = {"folder", "folders"}
    license_terms = {
        "activate",
        "activation",
        "deactivate",
        "evaluation",
        "license",
        "licence",
        "licensing",
        "trial",
    }
    install_terms = {
        "credentials",
        "initialized",
        "install",
        "installation",
        "initialize",
        "initialization",
        "login",
        "migration",
        "port",
        "prerequisite",
        "prerequisites",
        "server",
        "uninstall",
        "update",
        "upgrade",
        "wizard",
    }
    storage_terms = {
        "archive",
        "backup",
        "disk",
        "drive",
        "record",
        "recording",
        "recordings",
        "retention",
        "schedule",
        "storage",
    }
    user_terms = {
        "administrator",
        "ldap",
        "oauth",
        "permission",
        "permissions",
        "role",
        "roles",
    }
    integration_terms = {
        "gallagher",
        "integriti",
        "modbus",
        "mqtt",
        "opc",
        "satel",
    }
    export_terms = {
        "audit",
        "bookmark",
        "bookmarks",
        "export",
        "report",
        "reports",
        "snapshot",
        "snapshots",
        "vca",
    }
    layout_terms = {"layout", "layouts", "map", "maps", "videowall", "webpage", "webpages"}

    wants_camera_setup = _has_any(query_terms, camera_terms) and _has_any(query_terms, device_action_terms)
    wants_monitor_display = _has_any(query_terms, monitor_terms)
    wants_folder_access = _has_any(query_terms, folder_terms)
    wants_license = _has_any(query_terms, license_terms)
    wants_install = _has_any(query_terms, install_terms)
    wants_storage = _has_any(query_terms, storage_terms)
    wants_users = _has_any(query_terms, user_terms)
    wants_integration = _has_any(query_terms, integration_terms)
    wants_export = _has_any(query_terms, export_terms)
    wants_layout = _has_any(query_terms, layout_terms)
    if not (
        wants_camera_setup
        or wants_monitor_display
        or wants_folder_access
        or wants_license
        or wants_install
        or wants_storage
        or wants_users
        or wants_integration
        or wants_export
        or wants_layout
    ):
        return 1.0

    page = str(page_id or "").lower()
    heading = " ".join(heading_path or []).lower()
    mult = 1.0

    if wants_camera_setup:
        if page in {"adddevicesmanually", "adddevicesusingautodiscovery", "overviewofdevicesandchannels"}:
            mult *= 1.45
        elif page == "managedevices":
            mult *= 1.25

    if wants_monitor_display:
        if page == "liveviewsection":
            mult *= 1.8
        elif page in {"interfaceoverview", "interfaceelements"}:
            mult *= 1.6
        elif page in {"layouts", "layouttemplates", "connections"}:
            mult *= 1.3

    if wants_folder_access:
        if page == "wtffolders":
            mult *= 2.5
        elif page in {"modbus", "externalmetadata", "editionoverview", "prerequisites", "hardwarerequirements"}:
            mult *= 0.25

    if wants_license:
        if page == "licenseactivation":
            mult *= 1.75
        if page == "offlineactivation" and "offline" in query_terms:
            mult *= 2.2
        if page == "onlineactivation" and "online" in query_terms:
            mult *= 1.9
        if page == "evaluationlicense" and _has_any(query_terms, {"evaluation", "trial"}):
            mult *= 1.8
        elif page == "evaluationlicense":
            mult *= 0.55
        if page == "evotriallicenseluxriot" and "trial" in query_terms:
            mult *= 1.8
        elif page == "evotriallicenseluxriot":
            mult *= 0.7
        if page == "activationmanagement" and _has_any(
            query_terms,
            {"manage", "management", "renew", "subscription", "upgrade", "wizard"},
        ):
            mult *= 1.45

    if wants_install:
        install_wizard_terms = {"component", "install", "installation", "optional", "path", "paths", "wizard"}
        init_terms = {"credentials", "default", "initialized", "initialization", "login", "port", "server", "wizard"}
        upgrade_terms = {"files", "renew", "subscription", "uninstall", "update", "upgrade"}
        migration_terms = {"database", "db", "import", "migration", "vms", "xml"}
        if page == "standaloneinstallation" and _has_any(query_terms, install_wizard_terms):
            mult *= 2.0
        elif page in {"globalinstallation", "recinstallation", "consoleinstallation"} and _has_any(
            query_terms,
            install_wizard_terms,
        ):
            mult *= 1.45
        if page in {"standaloneinitialization", "globalinitialization", "recinitialization"} and _has_any(
            query_terms,
            init_terms,
        ):
            mult *= 1.65
        if page == "softwareupdateuninstall" and _has_any(query_terms, upgrade_terms):
            mult *= 1.8
        if page == "remoteupgrade" and _has_any(query_terms, {"remote", "upgrade"}):
            mult *= 1.9
        if page in {"migrationfromvms", "dbimport"} and _has_any(query_terms, migration_terms):
            mult *= 1.7
        if page in {"gettingstarted", "prerequisites"} and _has_any(
            query_terms,
            {"getting", "prerequisite", "prerequisites", "requirements", "started"},
        ):
            mult *= 1.5

    if wants_storage:
        storage_recommendation_terms = {
            "defrag",
            "defragmentation",
            "disk",
            "free",
            "indexing",
            "minimum",
            "raid",
            "recommendation",
            "recommendations",
            "recommended",
            "space",
            "system",
        }
        wants_storage_recommendations = _has_any(query_terms, storage_recommendation_terms)
        if page == "prerequisites" and wants_storage_recommendations:
            mult *= 2.1
        elif page in {
            "storage",
            "policies",
            "freerecordingprofiles",
            "assignrecordingconfigurations",
            "archivebackup",
            "create-schedules",
        }:
            mult *= 1.35
            if wants_storage_recommendations:
                mult *= 0.75

    if wants_users:
        if page in {
            "standaloneusermanagement",
            "usermanagement",
            "activedirectory-ldap",
            "permissionsandmembership",
            "anonymoususer",
            "2fauth",
            "oauth",
            "temporarypermissions",
            "wtffolders",
        }:
            mult *= 1.75

    if wants_integration:
        exact_integration_pages = {"modbus", "mqtt", "opc", "gallagher", "inner-range-integriti"}
        exact_page = None
        for token, candidate_page in (
            ("modbus", "modbus"),
            ("mqtt", "mqtt"),
            ("opc", "opc"),
            ("gallagher", "gallagher"),
            ("integriti", "inner-range-integriti"),
        ):
            if token in query_terms:
                exact_page = candidate_page
                break
        if exact_page and page == exact_page:
            mult *= 3.0
        elif exact_page and page in exact_integration_pages:
            mult *= 0.45
        if page in {
            "modbus",
            "mqtt",
            "opc",
            "securitysystems",
            "securitysystemsmonitor",
            "gallagher",
            "inner-range-integriti",
            "externalservices",
            "externalservicesoperation",
            "externalmetadata",
            "datasources",
            "datasourcesdbs",
            "camio",
            "vcaevents",
            "externalva",
        }:
            mult *= 1.8

    if wants_export:
        if page in {
            "reports",
            "audit",
            "vcareports",
            "vcareporting",
            "videosnapshotexport",
            "caseexport",
            "bookmarks",
            "librarysection",
        }:
            mult *= 1.75

    if wants_layout:
        if page in {
            "layouts",
            "layouttemplates",
            "managelayouts",
            "maps",
            "managemaps",
            "webpages",
            "videowall",
            "visual-groups",
            "manageuserbuttons",
            "userbuttons",
            "liveviewsection",
        }:
            mult *= 1.6

    if wants_camera_setup:
        if page == "mobileapplicationforstreamingserver" and not _has_any(query_terms, mobile_terms):
            mult *= 0.25
        if page == "onvifgenericevents" and not _has_any(query_terms, event_terms):
            mult *= 0.35
        if page in {"add-events", "add-rules"} and not _has_any(query_terms, event_terms):
            mult *= 0.4
        if "external video analytics" in heading and not _has_any(query_terms, event_terms):
            mult *= 0.5

    return float(mult)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())



@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    doc_id: str
    page_id: str
    heading_path: list[str]
    text: str
    source_path: str
    anchor: str | None
    images: list[str]
    length: int


@dataclass(frozen=True)
class SummaryRow:
    summary_id: str
    doc_id: str
    page_id: str
    heading_path: list[str]
    text: str
    source_path: str
    anchor: str | None
    length: int


class SearchEngine:
    def __init__(self, version: str = DEFAULT_VERSION, datastore_dir: Path = DATASTORE_DIR) -> None:
        self.version = version
        self.datastore_dir = datastore_dir
        self.index_path = datastore_dir / version / "index.sqlite"
        self._conn: sqlite3.Connection | None = None
        self._meta: dict[str, Any] | None = None
        self._embedding_vectors: dict[str, array] | None = None
        self._embedding_dim: int | None = None
        self._embedding_model_id: str | None = None
        self._summary_ready: bool | None = None

    def is_ready(self) -> bool:
        return self.index_path.exists()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self.index_path.exists():
                raise FileNotFoundError(f"Index not found: {self.index_path}")
            conn = sqlite3.connect(str(self.index_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON;")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._meta = None
        self._embedding_vectors = None
        self._embedding_dim = None
        self._embedding_model_id = None
        self._summary_ready = None

    def _load_meta(self) -> dict[str, Any]:
        if self._meta is not None:
            return self._meta
        conn = self._connect()
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
        meta = {r["key"]: r["value"] for r in rows}
        self._meta = meta
        return meta

    def _get_stat_floats(self) -> tuple[int, float]:
        meta = self._load_meta()
        try:
            n_chunks = int(meta["n_chunks"])
            avgdl = float(meta["avgdl"])
        except Exception as e:
            raise RuntimeError(f"Invalid meta in index: {e}") from e
        return n_chunks, avgdl

    def _get_summary_stat_floats(self) -> tuple[int, float]:
        meta = self._load_meta()
        try:
            n_units = int(meta.get("summary_units", 0) or 0)
            avgdl = float(meta.get("summary_avgdl", 0) or 0.0)
        except Exception as e:
            raise RuntimeError(f"Invalid summary meta in index: {e}") from e
        return n_units, avgdl

    def embeddings_ready(self) -> bool:
        try:
            meta = self._load_meta()
            enabled = str(meta.get("embeddings_enabled", "0"))
            dim = int(meta.get("embedding_dim", "0") or 0)
            model_id = str(meta.get("embedding_model_id", "") or "")
        except Exception:
            return False

        if enabled != "1" or dim <= 0 or not model_id:
            return False

        conn = self._connect()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='embeddings' LIMIT 1"
        ).fetchone()
        return bool(row)

    def summary_ready(self) -> bool:
        if self._summary_ready is not None:
            return self._summary_ready
        try:
            meta = self._load_meta()
            enabled = str(meta.get("summary_enabled", "0"))
            units = int(meta.get("summary_units", "0") or 0)
        except Exception:
            self._summary_ready = False
            return False
        if enabled != "1" or units <= 0:
            self._summary_ready = False
            return False
        conn = self._connect()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='summary_chunks' LIMIT 1"
        ).fetchone()
        self._summary_ready = bool(row)
        return self._summary_ready

    def _load_embeddings(self) -> None:
        if self._embedding_vectors is not None:
            return
        if not self.embeddings_ready():
            raise RuntimeError(
                "Embeddings are not available for this datastore. Re-run ingestion without --no-embeddings."
            )

        meta = self._load_meta()
        model_id = str(meta.get("embedding_model_id", "") or "")
        dim = int(meta.get("embedding_dim", "0") or 0)
        if not model_id or dim <= 0:
            raise RuntimeError("Embedding metadata missing in index; re-run ingestion.")

        conn = self._connect()
        rows = conn.execute("SELECT chunk_id, dim, vector FROM embeddings").fetchall()
        if not rows:
            raise RuntimeError("Embeddings table is empty; re-run ingestion.")

        vectors: dict[str, array] = {}
        for r in rows:
            cid = str(r["chunk_id"])
            rdim = int(r["dim"])
            if rdim != dim:
                raise RuntimeError("Embedding dimension mismatch in DB.")
            buf = r["vector"]
            if not isinstance(buf, (bytes, bytearray, memoryview)):
                raise RuntimeError("Invalid embedding vector type in DB.")
            a = array("f")
            a.frombytes(bytes(buf))
            if len(a) != dim:
                raise RuntimeError("Embedding vector length mismatch in DB.")
            vectors[cid] = a

        self._embedding_vectors = vectors
        self._embedding_dim = dim
        self._embedding_model_id = model_id

    def _fetch_chunk_rows(self, chunk_ids: list[str]) -> dict[str, ChunkRow]:
        if not chunk_ids:
            return {}
        conn = self._connect()
        qmarks = ",".join(["?"] * len(chunk_ids))
        rows = conn.execute(
            f"""
            SELECT chunk_id, doc_id, page_id, heading_path_json, text, source_path, anchor, images_json, length
            FROM chunks
            WHERE chunk_id IN ({qmarks})
            """,
            chunk_ids,
        ).fetchall()
        out: dict[str, ChunkRow] = {}
        for r in rows:
            out[r["chunk_id"]] = ChunkRow(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                page_id=r["page_id"],
                heading_path=json.loads(r["heading_path_json"]) if r["heading_path_json"] else [],
                text=r["text"],
                source_path=r["source_path"],
                anchor=r["anchor"],
                images=json.loads(r["images_json"]) if r["images_json"] else [],
                length=int(r["length"] or 0),
            )
        return out

    def _fetch_summary_rows(self, summary_ids: list[str]) -> dict[str, SummaryRow]:
        if not summary_ids:
            return {}
        conn = self._connect()
        qmarks = ",".join(["?"] * len(summary_ids))
        rows = conn.execute(
            f"""
            SELECT summary_id, doc_id, page_id, heading_path_json, text, source_path, anchor, length
            FROM summary_chunks
            WHERE summary_id IN ({qmarks})
            """,
            summary_ids,
        ).fetchall()
        out: dict[str, SummaryRow] = {}
        for r in rows:
            out[r["summary_id"]] = SummaryRow(
                summary_id=r["summary_id"],
                doc_id=r["doc_id"],
                page_id=r["page_id"],
                heading_path=json.loads(r["heading_path_json"]) if r["heading_path_json"] else [],
                text=r["text"],
                source_path=r["source_path"],
                anchor=r["anchor"],
                length=int(r["length"] or 0),
            )
        return out

    def _bm25_rank(
        self,
        query: str,
        k: int,
        allowed_pages: set[tuple[str, str]] | None = None,
    ) -> tuple[list[tuple[str, float]], dict[str, ChunkRow]]:
        query_terms = tokenize(query)
        if not query_terms:
            return ([], {})

        conn = self._connect()
        n_chunks, avgdl = self._get_stat_floats()

        # BM25 parameters (reasonable defaults).
        k1 = 1.5
        b = 0.75

        unique_terms = list(dict.fromkeys(query_terms))

        # Gather postings for each term so we only hit SQLite once per term.
        postings_by_term: dict[str, list[tuple[str, int]]] = {}
        candidate_chunk_ids: set[str] = set()
        idf_by_term: dict[str, float] = {}

        for term in unique_terms:
            df_row = conn.execute("SELECT df FROM terms WHERE term = ?", (term,)).fetchone()
            if not df_row:
                continue
            df = int(df_row["df"])
            idf_by_term[term] = math.log((n_chunks - df + 0.5) / (df + 0.5) + 1.0)

            posts = conn.execute("SELECT chunk_id, tf FROM postings WHERE term = ?", (term,)).fetchall()
            if not posts:
                continue
            lst: list[tuple[str, int]] = []
            for post in posts:
                chunk_id = str(post["chunk_id"])
                if allowed_pages is not None:
                    page_key = self._chunk_page_key(chunk_id)
                    if page_key is None or page_key not in allowed_pages:
                        continue
                tf = int(post["tf"])
                candidate_chunk_ids.add(chunk_id)
                lst.append((chunk_id, tf))
            postings_by_term[term] = lst

        if not candidate_chunk_ids:
            return ([], {})

        chunk_rows = self._fetch_chunk_rows(list(candidate_chunk_ids))
        if allowed_pages is not None:
            chunk_rows = {
                cid: row
                for cid, row in chunk_rows.items()
                if (row.doc_id, row.page_id) in allowed_pages
            }
            if not chunk_rows:
                return ([], {})
        scores: dict[str, float] = {}
        for term, posts in postings_by_term.items():
            idf = idf_by_term.get(term)
            if idf is None:
                continue
            for chunk_id, tf in posts:
                row = chunk_rows.get(chunk_id)
                if not row:
                    continue
                dl = max(row.length, 1)
                denom = tf + k1 * (1.0 - b + b * (dl / max(avgdl, 1e-9)))
                score = idf * (tf * (k1 + 1.0) / denom)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return top, chunk_rows

    def _summary_bm25_rank(self, query: str, k: int) -> tuple[list[tuple[str, float]], dict[str, SummaryRow]]:
        query_terms = tokenize(query)
        if not query_terms:
            return ([], {})

        conn = self._connect()
        n_units, avgdl = self._get_summary_stat_floats()
        if n_units <= 0:
            return ([], {})

        k1 = 1.5
        b = 0.75
        unique_terms = list(dict.fromkeys(query_terms))

        postings_by_term: dict[str, list[tuple[str, int]]] = {}
        candidate_ids: set[str] = set()
        idf_by_term: dict[str, float] = {}

        for term in unique_terms:
            df_row = conn.execute("SELECT df FROM summary_terms WHERE term = ?", (term,)).fetchone()
            if not df_row:
                continue
            df = int(df_row["df"])
            idf_by_term[term] = math.log((n_units - df + 0.5) / (df + 0.5) + 1.0)

            posts = conn.execute(
                "SELECT summary_id, tf FROM summary_postings WHERE term = ?", (term,)
            ).fetchall()
            if not posts:
                continue
            lst: list[tuple[str, int]] = []
            for post in posts:
                sid = str(post["summary_id"])
                tf = int(post["tf"])
                candidate_ids.add(sid)
                lst.append((sid, tf))
            postings_by_term[term] = lst

        if not candidate_ids:
            return ([], {})

        rows = self._fetch_summary_rows(list(candidate_ids))
        scores: dict[str, float] = {}
        for term, posts in postings_by_term.items():
            idf = idf_by_term.get(term)
            if idf is None:
                continue
            for sid, tf in posts:
                row = rows.get(sid)
                if not row:
                    continue
                dl = max(row.length, 1)
                denom = tf + k1 * (1.0 - b + b * (dl / max(avgdl, 1e-9)))
                score = idf * (tf * (k1 + 1.0) / denom)
                scores[sid] = scores.get(sid, 0.0) + score

        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return top, rows

    async def _embedding_rank(
        self,
        query: str,
        k: int,
        allowed_pages: set[tuple[str, str]] | None = None,
    ) -> list[tuple[str, float]]:
        self._load_embeddings()
        assert self._embedding_vectors is not None
        assert self._embedding_dim is not None
        assert self._embedding_model_id is not None

        try:
            q_vec = (await lm_embeddings([query], model=self._embedding_model_id))[0]
        except LMStudioError as e:
            raise RuntimeError(str(e)) from e

        norm = math.sqrt(sum(x * x for x in q_vec)) or 1.0
        q = array("f", (float(x / norm) for x in q_vec))
        if len(q) != self._embedding_dim:
            raise RuntimeError("Query embedding dimension mismatch; check embedding model and datastore.")

        scores: list[tuple[str, float]] = []
        dim = self._embedding_dim
        for cid, v in self._embedding_vectors.items():
            if allowed_pages is not None:
                page_key = self._chunk_page_key(cid)
                if page_key is None or page_key not in allowed_pages:
                    continue
            s = 0.0
            for i in range(dim):
                s += q[i] * v[i]
            scores.append((cid, float(s)))

        scores.sort(key=lambda kv: (-kv[1], kv[0]))
        return scores[:k]

    def _heading_match_multiplier(self, heading_path: list[str], query_terms: set[str], boost: float) -> float:
        if boost <= 0.0 or not query_terms:
            return 1.0
        hay = " ".join(heading_path or [])
        heading_terms = set(tokenize(hay))
        if not heading_terms:
            return 1.0
        overlap = query_terms.intersection(heading_terms)
        if not overlap:
            return 1.0
        frac = float(len(overlap)) / float(len(query_terms) or 1)
        return 1.0 + float(boost) * frac

    def _granularity_code(self, chunk_id: str) -> str:
        try:
            idx_part = chunk_id.rsplit(":", 1)[1]
        except Exception:
            return ""
        if idx_part and idx_part[0].isalpha():
            return idx_part[0].lower()
        return ""

    def _granularity_multiplier(self, chunk_id: str) -> float:
        code = self._granularity_code(chunk_id)
        return float(_GRANULARITY_WEIGHTS.get(code, 1.0))

    def _doc_priority_multiplier(self, doc_id: str, priority: list[str], boost: float, prio_map: dict[str, int]) -> float:
        if boost <= 0.0:
            return 1.0
        idx = prio_map.get(doc_id)
        if idx is None:
            return 1.0
        if len(priority) <= 1:
            rank_factor = 1.0
        else:
            rank_factor = 1.0 - (idx / (len(priority) - 1))
        return 1.0 + boost * rank_factor

    def _duplicate_guide_key(self, row: ChunkRow) -> tuple[str, str] | None:
        if row.doc_id not in _DUPLICATE_GUIDE_DOCS:
            return None
        if not row.page_id:
            return None
        return ("evo-admin-vs-standalone", row.page_id)

    def _duplicate_doc_preference(
        self,
        doc_id: str,
        query_terms: set[str],
        prio_map: dict[str, int],
    ) -> float:
        if "standalone" in query_terms and doc_id == "luxriot-evo-standalone":
            return 1000.0
        if "global" in query_terms and doc_id == "luxriot-evo-global-administration-guide":
            return 1000.0
        idx = prio_map.get(doc_id)
        if idx is None:
            return 0.0
        return 100.0 - float(idx)

    def _preferred_duplicate_docs(
        self,
        ranked: list[tuple[str, float]],
        chunk_rows: dict[str, ChunkRow],
        query_terms: set[str],
        prio_map: dict[str, int],
    ) -> dict[tuple[str, str], str]:
        candidates: dict[tuple[str, str], tuple[float, float, str]] = {}
        for cid, score in ranked:
            row = chunk_rows.get(cid)
            if not row:
                continue
            key = self._duplicate_guide_key(row)
            if key is None:
                continue
            pref = self._duplicate_doc_preference(row.doc_id, query_terms, prio_map)
            item = (pref, float(score), row.doc_id)
            prev = candidates.get(key)
            if prev is None or item > prev:
                candidates[key] = item
        return {key: doc_id for key, (_, _, doc_id) in candidates.items()}

    def _apply_dedupe(
        self,
        ranked: list[tuple[str, float]],
        chunk_rows: dict[str, ChunkRow],
        k: int,
        max_per_page: int,
        max_per_doc: int,
        preferred_duplicate_docs: dict[tuple[str, str], str] | None = None,
    ) -> list[tuple[str, float]]:
        preferred_duplicate_docs = preferred_duplicate_docs or {}
        if max_per_page <= 0 and max_per_doc <= 0 and not preferred_duplicate_docs:
            return ranked[:k]
        page_counts: Counter[tuple[str, str]] = Counter()
        doc_counts: Counter[str] = Counter()
        out: list[tuple[str, float]] = []
        for cid, score in ranked:
            row = chunk_rows.get(cid)
            if not row:
                continue
            dup_key = self._duplicate_guide_key(row)
            if dup_key is not None:
                preferred_doc = preferred_duplicate_docs.get(dup_key)
                if preferred_doc and row.doc_id != preferred_doc:
                    continue
            page_key = (row.doc_id, row.page_id)
            if max_per_page > 0 and page_counts[page_key] >= max_per_page:
                continue
            if max_per_doc > 0 and doc_counts[row.doc_id] >= max_per_doc:
                continue
            page_counts[page_key] += 1
            doc_counts[row.doc_id] += 1
            out.append((cid, score))
            if len(out) >= k:
                break
        return out

    def _truncate_for_rerank(self, text: str, max_chars: int) -> str:
        s = str(text or "").strip()
        if max_chars > 0 and len(s) > max_chars:
            return s[:max_chars]
        return s

    async def _rerank_candidates(
        self,
        query: str,
        ranked: list[tuple[str, float]],
        chunk_rows: dict[str, ChunkRow],
        *,
        model: str,
        top_k: int,
        min_score: float,
        max_chars: int,
        debug_out: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        model_id = str(model or "").strip()
        if not model_id:
            return ranked
        if not ranked or top_k <= 0:
            return ranked
        take = min(len(ranked), max(1, int(top_k)))
        docs: list[str] = []
        cids: list[str] = []
        for cid, _ in ranked[:take]:
            row = chunk_rows.get(cid)
            text = row.text if row else ""
            docs.append(self._truncate_for_rerank(text, max_chars=max_chars))
            cids.append(cid)
        try:
            scores = await lm_rerank(query, docs, model=model_id)
        except LMStudioError as e:
            log.warning("Reranker failed: %s", e)
            if debug_out is not None:
                debug_out["reranker"] = {
                    "enabled": True,
                    "model": model_id,
                    "top_k": int(top_k),
                    "min_score": float(min_score),
                    "max_chars": int(max_chars),
                    "applied": False,
                    "error": str(e),
                }
            return ranked

        score_map: dict[str, float] = {}
        for item in scores:
            try:
                idx = int(item.get("index"))
            except Exception:
                continue
            if 0 <= idx < len(cids):
                try:
                    score_map[cids[idx]] = float(item.get("score"))
                except Exception:
                    continue

        if not score_map:
            if debug_out is not None:
                debug_out["reranker"] = {
                    "enabled": True,
                    "model": model_id,
                    "top_k": int(top_k),
                    "min_score": float(min_score),
                    "max_chars": int(max_chars),
                    "applied": False,
                    "error": "Reranker returned no scores.",
                }
            return ranked

        reranked_top: list[tuple[str, float]] = []
        for cid, _ in ranked[:take]:
            if cid not in score_map:
                continue
            score = float(score_map[cid])
            if min_score and score < min_score:
                continue
            reranked_top.append((cid, score))
        reranked_top.sort(key=lambda kv: (-kv[1], kv[0]))

        remainder = ranked[take:]
        combined = reranked_top + remainder
        if debug_out is not None:
            debug_out["reranker"] = {
                "enabled": True,
                "model": model_id,
                "top_k": int(top_k),
                "min_score": float(min_score),
                "max_chars": int(max_chars),
                "applied": True,
                "scores": [
                    {"chunk_id": cid, "score": float(score_map.get(cid, 0.0))}
                    for cid in cids
                    if cid in score_map
                ][: min(50, len(score_map))],
            }
        return combined

    def _dot(self, a: array, b: array) -> float:
        if len(a) != len(b):
            return 0.0
        s = 0.0
        for i in range(len(a)):
            s += float(a[i]) * float(b[i])
        return float(s)

    def _jaccard_similarity(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = a.intersection(b)
        if not inter:
            return 0.0
        union = a.union(b)
        return float(len(inter)) / float(len(union) or 1)

    def _mmr_select(
        self,
        ranked: list[tuple[str, float]],
        chunk_rows: dict[str, ChunkRow],
        *,
        k: int,
        mmr_lambda: float,
        use_embeddings: bool,
        max_per_page: int,
        max_per_doc: int,
        preferred_duplicate_docs: dict[tuple[str, str], str] | None = None,
        trace_out: list[dict[str, Any]] | None = None,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        if not ranked:
            return []

        # Clamp lambda to [0, 1]. Higher values favor relevance; lower values favor diversity.
        lam = float(mmr_lambda)
        if lam < 0.0:
            lam = 0.0
        if lam > 1.0:
            lam = 1.0

        # Pre-filter to candidates we can actually return.
        candidates: list[tuple[str, float]] = [(cid, float(score)) for cid, score in ranked if cid in chunk_rows]
        if not candidates:
            return []

        # Normalize relevance to [0, 1] using min-max scores for stable MMR arithmetic.
        scores = [score for _, score in candidates]
        min_score = min(scores)
        max_score = max(scores)
        if max_score > min_score:
            rel = {cid: (score - min_score) / (max_score - min_score) for cid, score in candidates}
        else:
            rel = {cid: 1.0 for cid, _ in candidates}

        use_emb = bool(use_embeddings) and self.embeddings_ready()
        cand_vec: dict[str, array] = {}
        cand_tokens: dict[str, set[str]] = {}
        if use_emb:
            self._load_embeddings()
            assert self._embedding_vectors is not None
            for cid, _ in candidates:
                v = self._embedding_vectors.get(cid)
                if v is not None:
                    cand_vec[cid] = v

        if not use_emb or len(cand_vec) < len(candidates):
            for cid, _ in candidates:
                if cid in cand_tokens:
                    continue
                row = chunk_rows.get(cid)
                if not row:
                    continue
                toks = tokenize(row.text)
                # Bound token-set size to keep it cheap and stable.
                cand_tokens[cid] = set(toks[:256])

        selected: list[tuple[str, float]] = []
        selected_ids: list[str] = []
        page_counts: Counter[tuple[str, str]] = Counter()
        doc_counts: Counter[str] = Counter()
        preferred_duplicate_docs = preferred_duplicate_docs or {}

        def allowed(cid: str) -> bool:
            row = chunk_rows.get(cid)
            if not row:
                return False
            dup_key = self._duplicate_guide_key(row)
            if dup_key is not None:
                preferred_doc = preferred_duplicate_docs.get(dup_key)
                if preferred_doc and row.doc_id != preferred_doc:
                    return False
            page_key = (row.doc_id, row.page_id)
            if max_per_page > 0 and page_counts[page_key] >= max_per_page:
                return False
            if max_per_doc > 0 and doc_counts[row.doc_id] >= max_per_doc:
                return False
            return True

        def update_counts(cid: str) -> None:
            row = chunk_rows.get(cid)
            if not row:
                return
            page_counts[(row.doc_id, row.page_id)] += 1
            doc_counts[row.doc_id] += 1

        def similarity(a_id: str, b_id: str) -> float:
            if a_id == b_id:
                return 1.0
            if use_emb and a_id in cand_vec and b_id in cand_vec:
                s = self._dot(cand_vec[a_id], cand_vec[b_id])
                if s < 0.0:
                    s = 0.0
                if s > 1.0:
                    s = 1.0
                return float(s)
            a_toks = cand_tokens.get(a_id) or set()
            b_toks = cand_tokens.get(b_id) or set()
            return self._jaccard_similarity(a_toks, b_toks)

        # Deterministic MMR selection.
        while len(selected) < k:
            best_cid: str | None = None
            best_mmr: float = -1e18

            for cid, orig_score in candidates:
                if cid in selected_ids:
                    continue
                if not allowed(cid):
                    continue

                relevance = float(rel.get(cid, 0.0))
                if not selected_ids:
                    mmr_score = relevance
                else:
                    max_sim = 0.0
                    for sid in selected_ids:
                        sim = similarity(cid, sid)
                        if sim > max_sim:
                            max_sim = sim
                    mmr_score = lam * relevance - (1.0 - lam) * max_sim

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_cid = cid
                    best_orig_score = orig_score
                    best_relevance = relevance
                    best_max_sim = float(max_sim) if selected_ids else 0.0

            if best_cid is None:
                break
            selected.append((best_cid, float(best_orig_score)))
            selected_ids.append(best_cid)
            update_counts(best_cid)
            if trace_out is not None:
                trace_out.append(
                    {
                        "step": len(selected_ids),
                        "chunk_id": best_cid,
                        "relevance": float(best_relevance),
                        "max_similarity": float(best_max_sim),
                        "mmr_score": float(best_mmr),
                    }
                )

        return selected

    def _neighbor_chunk_ids(self, chunk_id: str, neighbors: int) -> list[str]:
        if neighbors <= 0:
            return [chunk_id]
        try:
            doc_part, page_part, idx_part = chunk_id.rsplit(":", 2)
            prefix = ""
            num_part = idx_part
            if idx_part and idx_part[0].isalpha():
                prefix = idx_part[0]
                num_part = idx_part[1:]
            idx = int(num_part)
        except Exception:
            return [chunk_id]
        start = max(0, idx - neighbors)
        end = idx + neighbors
        return [f"{doc_part}:{page_part}:{prefix}{i:03d}" for i in range(start, end + 1)]

    def _chunk_page_key(self, chunk_id: str) -> tuple[str, str] | None:
        try:
            doc_part, page_part, _ = chunk_id.rsplit(":", 2)
            return (doc_part, page_part)
        except Exception:
            return None

    def _expand_chunk_text(
        self,
        chunk_id: str,
        chunk_rows: dict[str, ChunkRow],
        *,
        neighbors: int,
        max_chars: int,
    ) -> tuple[str, list[str]]:
        if neighbors <= 0:
            row = chunk_rows.get(chunk_id)
            return (row.text, row.images) if row else ("", [])

        wanted = self._neighbor_chunk_ids(chunk_id, neighbors)
        missing = [cid for cid in wanted if cid not in chunk_rows]
        if missing:
            fetched = self._fetch_chunk_rows(missing)
            chunk_rows.update(fetched)

        segments: list[tuple[str, str, list[str]]] = []
        for cid in wanted:
            row = chunk_rows.get(cid)
            if not row:
                continue
            segments.append((cid, row.text, row.images))

        if not segments:
            return ("", [])

        center_idx = 0
        for i, (cid, _, _) in enumerate(segments):
            if cid == chunk_id:
                center_idx = i
                break

        # Always include the center segment; expand outward until we hit max_chars.
        include: set[int] = {center_idx}
        text_total = len(segments[center_idx][1])
        if max_chars <= 0:
            include = set(range(len(segments)))
        else:
            remaining = max_chars - text_total
            if remaining <= 0:
                text = segments[center_idx][1][:max_chars]
                images = segments[center_idx][2]
                return text, images

            left = center_idx - 1
            right = center_idx + 1
            while remaining > 0 and (left >= 0 or right < len(segments)):
                progressed = False
                if left >= 0:
                    seg_len = len(segments[left][1]) + 2
                    if seg_len <= remaining:
                        include.add(left)
                        remaining -= seg_len
                        progressed = True
                    left -= 1
                if right < len(segments):
                    seg_len = len(segments[right][1]) + 2
                    if seg_len <= remaining:
                        include.add(right)
                        remaining -= seg_len
                        progressed = True
                    right += 1
                if not progressed:
                    break

        texts: list[str] = []
        images_out: list[str] = []
        seen_img: set[str] = set()
        for i in sorted(include):
            t = segments[i][1].strip()
            if t:
                texts.append(t)
            for url in segments[i][2]:
                if url in seen_img:
                    continue
                seen_img.add(url)
                images_out.append(url)

        return "\n\n".join(texts).strip(), images_out

    def _rows_to_results(
        self,
        selected: list[tuple[str, float]],
        chunk_rows: dict[str, ChunkRow],
        *,
        expand_neighbors: int,
        expand_max_chars: int,
        expand_include_images: bool,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cid, score in selected:
            row = chunk_rows.get(cid)
            if not row:
                continue
            if expand_neighbors > 0:
                text, expanded_images = self._expand_chunk_text(
                    cid,
                    chunk_rows,
                    neighbors=expand_neighbors,
                    max_chars=expand_max_chars,
                )
                images = expanded_images if expand_include_images else row.images
            else:
                text, images = row.text, row.images
            out.append(
                {
                    "chunk_id": cid,
                    "doc_id": row.doc_id,
                    "page_id": row.page_id,
                    "heading_path": row.heading_path,
                    "text": text,
                    "score": float(score),
                    "source_path": row.source_path,
                    "anchor": row.anchor,
                    "images": images,
                }
            )
        return out

    async def search(
        self,
        query: str,
        k: int = 8,
        mode: str = "bm25",
        *,
        mmr_enabled: bool = False,
        mmr_lambda: float = 0.7,
        mmr_candidates: int | None = None,
        mmr_use_embeddings: bool = True,
        expand_neighbors: int = 0,
        expand_max_chars: int = 0,
        expand_include_images: bool = False,
        heading_boost: float = 0.0,
        bm25_candidates: int | None = None,
        embedding_candidates: int | None = None,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        embedding_weight: float = 1.0,
        doc_priority: list[str] | None = None,
        doc_priority_boost: float = 0.0,
        summary_enabled: bool | None = None,
        summary_k: int | None = None,
        summary_max_pages: int | None = None,
        reranker_enabled: bool = False,
        reranker_model: str | None = None,
        reranker_top_k: int | None = None,
        reranker_min_score: float = 0.0,
        reranker_max_chars: int | None = None,
        max_per_page: int = 0,
        max_per_doc: int = 0,
        debug_out: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        mode = (mode or "bm25").lower().strip()
        if mode not in ("bm25", "embedding", "hybrid"):
            raise ValueError(f"Unknown retrieval mode: {mode}")

        retrieval_query = expand_luxriot_query(query)
        doc_priority = doc_priority or []
        prio_map = {d: i for i, d in enumerate(doc_priority)}
        mmr_enabled = bool(mmr_enabled)
        use_mmr = mmr_enabled and k > 1
        cand_limit = int(mmr_candidates or 0) or 0
        if cand_limit > 0:
            cand_limit = max(cand_limit, k)
        heading_boost = float(heading_boost or 0.0)
        query_terms = set(tokenize(retrieval_query))
        query_terms_for_heading = {t for t in query_terms if t not in _STOPWORDS}
        configured_max_per_page = int(max_per_page or 0)
        max_per_page = _workflow_page_cap(query_terms, configured_max_per_page)
        want_debug = debug_out is not None
        reranker_enabled = bool(reranker_enabled)
        reranker_top_k_val = int(reranker_top_k or 0) or 0
        reranker_min_score_val = float(reranker_min_score or 0.0)
        reranker_max_chars_val = int(reranker_max_chars or 0) or 0
        if want_debug:
            debug_out.clear()
            debug_out.update(
                {
                    "query": query,
                    "retrieval_query": retrieval_query,
                    "mode": mode,
                    "k": int(k),
                    "doc_priority_boost": float(doc_priority_boost or 0.0),
                    "heading_boost": float(heading_boost or 0.0),
                    "doc_priority": list(doc_priority),
                    "reranker": {
                        "enabled": bool(reranker_enabled),
                        "model": str(reranker_model or ""),
                        "top_k": int(reranker_top_k_val),
                        "min_score": float(reranker_min_score_val),
                        "max_chars": int(reranker_max_chars_val),
                    },
                    "dedupe": {
                        "max_per_page": int(max_per_page or 0),
                        "configured_max_per_page": int(configured_max_per_page or 0),
                        "max_per_doc": int(max_per_doc or 0),
                    },
                    "mmr": {
                        "enabled": bool(mmr_enabled),
                        "lambda": float(mmr_lambda),
                        "candidates": int(cand_limit or 0),
                        "use_embeddings": bool(mmr_use_embeddings),
                    },
                    "expand": {
                        "neighbors": int(expand_neighbors or 0),
                        "max_chars": int(expand_max_chars or 0),
                        "include_images": bool(expand_include_images),
                    },
                }
            )

        summary_enabled_flag = bool(summary_enabled) if summary_enabled is not None else False
        summary_k_val = int(summary_k or 0)
        summary_max_pages_val = int(summary_max_pages or 0)
        summary_ready = False
        summary_k_eff = 0
        allowed_pages: set[tuple[str, str]] | None = None
        summary_top: list[tuple[str, float]] = []
        summary_rows: dict[str, SummaryRow] = {}
        summary_pages: list[tuple[str, str]] = []

        summary_queries: list[str] = []
        if summary_enabled_flag:
            summary_ready = self.summary_ready()
            if summary_ready:
                summary_k_eff = summary_k_val if summary_k_val > 0 else max(k, 10)
                summary_queries = luxriot_summary_queries(query, retrieval_query)
                rrf_scores: dict[str, float] = {}
                for q in summary_queries:
                    top, rows = self._summary_bm25_rank(q, summary_k_eff)
                    for sid, row in rows.items():
                        summary_rows.setdefault(sid, row)
                    for rank, (sid, _) in enumerate(top, start=1):
                        row = rows.get(sid)
                        intent_mult = (
                            _intent_page_multiplier(row.page_id, row.heading_path, query_terms)
                            if row is not None
                            else 1.0
                        )
                        rrf_scores[sid] = rrf_scores.get(sid, 0.0) + (
                            float(intent_mult) / float(rrf_k + rank)
                        )
                summary_top = sorted(rrf_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:summary_k_eff]
                seen_pages: set[tuple[str, str]] = set()
                for sid, score in summary_top:
                    row = summary_rows.get(sid)
                    if not row:
                        continue
                    key = (row.doc_id, row.page_id)
                    if key in seen_pages:
                        continue
                    seen_pages.add(key)
                    summary_pages.append(key)
                    if summary_max_pages_val > 0 and len(summary_pages) >= summary_max_pages_val:
                        break
                if summary_pages:
                    allowed_pages = set(summary_pages)

        if want_debug:
            summary_debug: dict[str, Any] = {
                "enabled": bool(summary_enabled_flag),
                "ready": bool(summary_ready),
                "applied": bool(allowed_pages),
                "k": int(summary_k_eff or summary_k_val or 0),
                "max_pages": int(summary_max_pages_val or 0),
                "candidates": int(len(summary_top)),
                "queries": list(summary_queries),
                "selected_pages": [{"doc_id": d, "page_id": p} for d, p in summary_pages],
            }
            if summary_top:
                top_rows: list[dict[str, Any]] = []
                for rank, (sid, score) in enumerate(summary_top, start=1):
                    row = summary_rows.get(sid)
                    if not row:
                        continue
                    top_rows.append(
                        {
                            "rank": int(rank),
                            "summary_id": sid,
                            "doc_id": row.doc_id,
                            "page_id": row.page_id,
                            "heading_path": row.heading_path,
                            "score": float(score),
                        }
                    )
                    if len(top_rows) >= 50:
                        break
                summary_debug["top"] = top_rows
            debug_out["summary"] = summary_debug

        if mode == "bm25":
            top, chunk_rows = self._bm25_rank(
                retrieval_query,
                cand_limit if use_mmr and cand_limit else k,
                allowed_pages=allowed_pages,
            )
            adjusted: list[tuple[str, float]] = []
            cand_debug: dict[str, dict[str, Any]] = {}
            for rank, (cid, bm25_score) in enumerate(top, start=1):
                row = chunk_rows.get(cid)
                if not row:
                    continue
                doc_mult = self._doc_priority_multiplier(row.doc_id, doc_priority, doc_priority_boost, prio_map)
                heading_mult = self._heading_match_multiplier(row.heading_path, query_terms_for_heading, heading_boost)
                gran_mult = self._granularity_multiplier(cid)
                intent_mult = _intent_page_multiplier(row.page_id, row.heading_path, query_terms)
                score = (
                    float(bm25_score)
                    * float(doc_mult)
                    * float(heading_mult)
                    * float(gran_mult)
                    * float(intent_mult)
                )
                adjusted.append((cid, float(score)))
                if want_debug:
                    cand_debug[cid] = {
                        "chunk_id": cid,
                        "doc_id": row.doc_id,
                        "page_id": row.page_id,
                        "heading_path": row.heading_path,
                        "score": float(score),
                        "bm25": {"rank": int(rank), "score": float(bm25_score)},
                        "doc_priority_mult": float(doc_mult),
                        "heading_mult": float(heading_mult),
                        "intent_mult": float(intent_mult),
                        "granularity": self._granularity_code(cid) or "legacy",
                        "granularity_mult": float(gran_mult),
                    }
            adjusted.sort(key=lambda kv: (-kv[1], kv[0]))
            if reranker_enabled:
                adjusted = await self._rerank_candidates(
                    retrieval_query,
                    adjusted,
                    chunk_rows,
                    model=str(reranker_model or ""),
                    top_k=reranker_top_k_val or len(adjusted),
                    min_score=reranker_min_score_val,
                    max_chars=reranker_max_chars_val or 0,
                    debug_out=debug_out if want_debug else None,
                )
            preferred_duplicate_docs = self._preferred_duplicate_docs(
                adjusted,
                chunk_rows,
                query_terms,
                prio_map,
            )
            mmr_trace: list[dict[str, Any]] = []
            if use_mmr:
                selected = self._mmr_select(
                    adjusted,
                    chunk_rows,
                    k=k,
                    mmr_lambda=mmr_lambda,
                    use_embeddings=mmr_use_embeddings,
                    max_per_page=max_per_page,
                    max_per_doc=max_per_doc,
                    preferred_duplicate_docs=preferred_duplicate_docs,
                    trace_out=mmr_trace if want_debug else None,
                )
            else:
                selected = self._apply_dedupe(
                    adjusted,
                    chunk_rows,
                    k,
                    max_per_page,
                    max_per_doc,
                    preferred_duplicate_docs=preferred_duplicate_docs,
                )
            results = self._rows_to_results(
                selected,
                chunk_rows,
                expand_neighbors=int(expand_neighbors or 0),
                expand_max_chars=int(expand_max_chars or 0),
                expand_include_images=bool(expand_include_images),
            )
            if want_debug:
                debug_out["candidates_count"] = int(len(adjusted))
                debug_out["candidates_top"] = [cand_debug[cid] for cid, _ in adjusted[: min(50, len(adjusted))] if cid in cand_debug]
                debug_out["duplicate_dedupe"] = [
                    {"group": list(key), "preferred_doc_id": doc_id}
                    for key, doc_id in sorted(preferred_duplicate_docs.items())
                ][:50]
                debug_out["mmr_trace"] = mmr_trace
                debug_out["selected"] = []
                for i, r in enumerate(results, start=1):
                    info = dict(cand_debug.get(r["chunk_id"], {}))
                    info.update(
                        {
                            "rank": int(i),
                            "returned_score": float(r.get("score", 0.0)),
                            "text_chars": int(len(r.get("text") or "")),
                            "images_count": int(len(r.get("images") or [])),
                        }
                    )
                    debug_out["selected"].append(info)
            return results

        if mode == "embedding":
            if not self.embeddings_ready():
                raise RuntimeError(
                    "Embeddings mode requested but embeddings are not available. Re-run ingestion to build embeddings."
                )
            top = await self._embedding_rank(
                retrieval_query,
                k=max((cand_limit if use_mmr and cand_limit else k), 1),
                allowed_pages=allowed_pages,
            )
            chunk_ids = [cid for cid, _ in top]
            chunk_rows = self._fetch_chunk_rows(chunk_ids)
            adjusted: list[tuple[str, float]] = []
            cand_debug: dict[str, dict[str, Any]] = {}
            for rank, (cid, emb_score) in enumerate(top, start=1):
                row = chunk_rows.get(cid)
                if not row:
                    continue
                doc_mult = self._doc_priority_multiplier(row.doc_id, doc_priority, doc_priority_boost, prio_map)
                heading_mult = self._heading_match_multiplier(row.heading_path, query_terms_for_heading, heading_boost)
                gran_mult = self._granularity_multiplier(cid)
                intent_mult = _intent_page_multiplier(row.page_id, row.heading_path, query_terms)
                score = (
                    float(emb_score)
                    * float(doc_mult)
                    * float(heading_mult)
                    * float(gran_mult)
                    * float(intent_mult)
                )
                adjusted.append((cid, float(score)))
                if want_debug:
                    cand_debug[cid] = {
                        "chunk_id": cid,
                        "doc_id": row.doc_id,
                        "page_id": row.page_id,
                        "heading_path": row.heading_path,
                        "score": float(score),
                        "embedding": {"rank": int(rank), "score": float(emb_score)},
                        "doc_priority_mult": float(doc_mult),
                        "heading_mult": float(heading_mult),
                        "intent_mult": float(intent_mult),
                        "granularity": self._granularity_code(cid) or "legacy",
                        "granularity_mult": float(gran_mult),
                    }
            adjusted.sort(key=lambda kv: (-kv[1], kv[0]))
            if reranker_enabled:
                adjusted = await self._rerank_candidates(
                    retrieval_query,
                    adjusted,
                    chunk_rows,
                    model=str(reranker_model or ""),
                    top_k=reranker_top_k_val or len(adjusted),
                    min_score=reranker_min_score_val,
                    max_chars=reranker_max_chars_val or 0,
                    debug_out=debug_out if want_debug else None,
                )
            preferred_duplicate_docs = self._preferred_duplicate_docs(
                adjusted,
                chunk_rows,
                query_terms,
                prio_map,
            )
            mmr_trace: list[dict[str, Any]] = []
            if use_mmr:
                selected = self._mmr_select(
                    adjusted,
                    chunk_rows,
                    k=k,
                    mmr_lambda=mmr_lambda,
                    use_embeddings=mmr_use_embeddings,
                    max_per_page=max_per_page,
                    max_per_doc=max_per_doc,
                    preferred_duplicate_docs=preferred_duplicate_docs,
                    trace_out=mmr_trace if want_debug else None,
                )
            else:
                selected = self._apply_dedupe(
                    adjusted,
                    chunk_rows,
                    k,
                    max_per_page,
                    max_per_doc,
                    preferred_duplicate_docs=preferred_duplicate_docs,
                )
            results = self._rows_to_results(
                selected,
                chunk_rows,
                expand_neighbors=int(expand_neighbors or 0),
                expand_max_chars=int(expand_max_chars or 0),
                expand_include_images=bool(expand_include_images),
            )
            if want_debug:
                debug_out["candidates_count"] = int(len(adjusted))
                debug_out["candidates_top"] = [cand_debug[cid] for cid, _ in adjusted[: min(50, len(adjusted))] if cid in cand_debug]
                debug_out["duplicate_dedupe"] = [
                    {"group": list(key), "preferred_doc_id": doc_id}
                    for key, doc_id in sorted(preferred_duplicate_docs.items())
                ][:50]
                debug_out["mmr_trace"] = mmr_trace
                debug_out["selected"] = []
                for i, r in enumerate(results, start=1):
                    info = dict(cand_debug.get(r["chunk_id"], {}))
                    info.update(
                        {
                            "rank": int(i),
                            "returned_score": float(r.get("score", 0.0)),
                            "text_chars": int(len(r.get("text") or "")),
                            "images_count": int(len(r.get("images") or [])),
                        }
                    )
                    debug_out["selected"].append(info)
            return results

        # hybrid
        if not self.embeddings_ready():
            raise RuntimeError(
                "Hybrid mode requested but embeddings are not available. Re-run ingestion to build embeddings."
            )

        bm25_candidates = int(bm25_candidates or max(50, k))
        embedding_candidates = int(embedding_candidates or max(50, k))
        if want_debug:
            debug_out["hybrid"] = {
                "rrf_k": int(rrf_k),
                "bm25_weight": float(bm25_weight),
                "embedding_weight": float(embedding_weight),
                "bm25_candidates": int(bm25_candidates),
                "embedding_candidates": int(embedding_candidates),
            }

        bm25_top, _bm25_rows = self._bm25_rank(retrieval_query, bm25_candidates, allowed_pages=allowed_pages)
        emb_top = await self._embedding_rank(retrieval_query, embedding_candidates, allowed_pages=allowed_pages)

        bm25_rank = {cid: i + 1 for i, (cid, _) in enumerate(bm25_top)}
        emb_rank = {cid: i + 1 for i, (cid, _) in enumerate(emb_top)}
        bm25_score = {cid: float(score) for cid, score in bm25_top}
        emb_score = {cid: float(score) for cid, score in emb_top}

        candidate_ids = set(bm25_rank.keys()) | set(emb_rank.keys())
        chunk_rows = self._fetch_chunk_rows(list(candidate_ids))

        combined: list[tuple[str, float]] = []
        cand_debug: dict[str, dict[str, Any]] = {}
        for cid in candidate_ids:
            base = 0.0
            r_b = bm25_rank.get(cid)
            r_e = emb_rank.get(cid)
            if r_b is not None:
                base += float(bm25_weight) / float(rrf_k + r_b)
            if r_e is not None:
                base += float(embedding_weight) / float(rrf_k + r_e)
            row = chunk_rows.get(cid)
            doc_mult = 1.0
            heading_mult = 1.0
            intent_mult = 1.0
            if row:
                doc_mult = self._doc_priority_multiplier(row.doc_id, doc_priority, doc_priority_boost, prio_map)
                heading_mult = self._heading_match_multiplier(row.heading_path, query_terms_for_heading, heading_boost)
                intent_mult = _intent_page_multiplier(row.page_id, row.heading_path, query_terms)
            gran_mult = self._granularity_multiplier(cid)
            score = float(base) * float(doc_mult) * float(heading_mult) * float(gran_mult) * float(intent_mult)
            combined.append((cid, float(score)))
            if want_debug and row:
                cand_debug[cid] = {
                    "chunk_id": cid,
                    "doc_id": row.doc_id,
                    "page_id": row.page_id,
                    "heading_path": row.heading_path,
                    "score": float(score),
                    "rrf_base": float(base),
                    "bm25": {"rank": int(r_b) if r_b is not None else None, "score": bm25_score.get(cid)},
                    "embedding": {"rank": int(r_e) if r_e is not None else None, "score": emb_score.get(cid)},
                    "doc_priority_mult": float(doc_mult),
                    "heading_mult": float(heading_mult),
                    "intent_mult": float(intent_mult),
                    "granularity": self._granularity_code(cid) or "legacy",
                    "granularity_mult": float(gran_mult),
                }

        combined.sort(key=lambda kv: (-kv[1], kv[0]))
        if reranker_enabled:
            combined = await self._rerank_candidates(
                retrieval_query,
                combined,
                chunk_rows,
                model=str(reranker_model or ""),
                top_k=reranker_top_k_val or len(combined),
                min_score=reranker_min_score_val,
                max_chars=reranker_max_chars_val or 0,
                debug_out=debug_out if want_debug else None,
            )
        mmr_trace: list[dict[str, Any]] = []
        if use_mmr:
            trimmed = combined[: (cand_limit if cand_limit else len(combined))]
            preferred_duplicate_docs = self._preferred_duplicate_docs(
                trimmed,
                chunk_rows,
                query_terms,
                prio_map,
            )
            selected = self._mmr_select(
                trimmed,
                chunk_rows,
                k=k,
                mmr_lambda=mmr_lambda,
                use_embeddings=mmr_use_embeddings,
                max_per_page=max_per_page,
                max_per_doc=max_per_doc,
                preferred_duplicate_docs=preferred_duplicate_docs,
                trace_out=mmr_trace if want_debug else None,
            )
        else:
            preferred_duplicate_docs = self._preferred_duplicate_docs(
                combined,
                chunk_rows,
                query_terms,
                prio_map,
            )
            selected = self._apply_dedupe(
                combined,
                chunk_rows,
                k,
                max_per_page,
                max_per_doc,
                preferred_duplicate_docs=preferred_duplicate_docs,
            )
        results = self._rows_to_results(
            selected,
            chunk_rows,
            expand_neighbors=int(expand_neighbors or 0),
            expand_max_chars=int(expand_max_chars or 0),
            expand_include_images=bool(expand_include_images),
        )
        if want_debug:
            debug_out["candidates_count"] = int(len(combined))
            debug_out["candidates_top"] = [cand_debug[cid] for cid, _ in combined[: min(50, len(combined))] if cid in cand_debug]
            debug_out["duplicate_dedupe"] = [
                {"group": list(key), "preferred_doc_id": doc_id}
                for key, doc_id in sorted(preferred_duplicate_docs.items())
            ][:50]
            debug_out["mmr_trace"] = mmr_trace
            debug_out["selected"] = []
            for i, r in enumerate(results, start=1):
                info = dict(cand_debug.get(r["chunk_id"], {}))
                info.update(
                    {
                        "rank": int(i),
                        "returned_score": float(r.get("score", 0.0)),
                        "text_chars": int(len(r.get("text") or "")),
                        "images_count": int(len(r.get("images") or [])),
                    }
                )
                debug_out["selected"].append(info)
        return results
