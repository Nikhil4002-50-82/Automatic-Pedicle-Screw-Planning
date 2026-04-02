from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _unique_paths(paths: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for raw_path in paths:
        path = str(Path(raw_path).expanduser().resolve(strict=False))
        key = _normalize_path(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return tuple(unique)


def default_recents_path() -> Path:
    base_dir = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA")
    if base_dir:
        root = Path(base_dir)
    else:
        root = Path.home()
    return root / "Automatic-Pedicle-Screw-Planning" / "ct_viewer_recents.json"


@dataclass(frozen=True, slots=True)
class RecentStudy:
    ct_path: str
    mask_paths: tuple[str, ...] = ()
    updated_at: float = 0.0
    label: str = ""

    @property
    def display_label(self) -> str:
        return self.label or Path(self.ct_path).name

    @property
    def mask_count(self) -> int:
        return len(self.mask_paths)

    def to_dict(self) -> dict[str, object]:
        return {
            "ct_path": self.ct_path,
            "mask_paths": list(self.mask_paths),
            "updated_at": self.updated_at,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "RecentStudy":
        ct_path = str(payload.get("ct_path", "")).strip()
        if not ct_path:
            raise ValueError("Recent entry is missing a CT path")

        raw_mask_paths = payload.get("mask_paths", [])
        mask_paths = _unique_paths([str(path) for path in raw_mask_paths if str(path)])
        updated_at = float(payload.get("updated_at", 0.0) or 0.0)
        label = str(payload.get("label", "")).strip()
        return cls(ct_path=ct_path, mask_paths=mask_paths, updated_at=updated_at, label=label)


class RecentStudiesStore:
    def __init__(self, path: Path | None = None, max_entries: int = 12) -> None:
        self.path = path or default_recents_path()
        self.max_entries = max_entries
        self._entries = self._load()

    @property
    def entries(self) -> tuple[RecentStudy, ...]:
        return tuple(self._entries)

    def _load(self) -> list[RecentStudy]:
        if not self.path.exists():
            return []

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

        studies: list[RecentStudy] = []
        if not isinstance(payload, list):
            return studies

        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                studies.append(RecentStudy.from_dict(item))
            except Exception:
                continue
        return studies[: self.max_entries]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [study.to_dict() for study in self._entries[: self.max_entries]]
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def clear(self) -> None:
        self._entries = []
        self._save()

    def upsert(self, ct_path: str, mask_paths: list[str] | tuple[str, ...] = (), label: str = "") -> RecentStudy:
        study = RecentStudy(
            ct_path=str(Path(ct_path).expanduser().resolve(strict=False)),
            mask_paths=_unique_paths(list(mask_paths)),
            label=label.strip() or Path(ct_path).name,
        )
        key = _normalize_path(study.ct_path)
        self._entries = [entry for entry in self._entries if _normalize_path(entry.ct_path) != key]
        self._entries.insert(0, study)
        self._entries = self._entries[: self.max_entries]
        self._save()
        return study

    def remove(self, ct_path: str) -> None:
        key = _normalize_path(ct_path)
        new_entries = [entry for entry in self._entries if _normalize_path(entry.ct_path) != key]
        if len(new_entries) == len(self._entries):
            return
        self._entries = new_entries
        self._save()
