"""Entity resolution (master data management) for team and player names.

Every source spells clubs differently ("Man Utd", "Manchester United",
"Manchester Utd"). Names are normalised, looked up in a curated alias table
and otherwise fuzzy-matched with a Levenshtein similarity against the
canonical register. Each canonical entity gets a deterministic UUIDv5 so the
same club receives the same key on every run and in every table, which makes
SQL joins across sources safe.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable

# Fixed namespace: changing it would re-key every table in the database.
NAMESPACE = uuid.UUID("6f1c2a9e-4b7d-5e8f-9a0b-1c2d3e4f5a6b")


def entity_uuid(kind: str, canonical_name: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{kind}:{canonical_name}"))


def team_uuid(canonical_name: str) -> str:
    return entity_uuid("team", canonical_name)


def levenshtein(a: str, b: str) -> int:
    """Classic edit distance (insert, delete, substitute) in O(len(a) * len(b))."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ca != cb),  # substitution
                )
            )
        previous = current
    return previous[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    """1 - distance / max length, in [0, 1]."""
    if not a and not b:
        return 1.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def prefix_token_similarity(a: str, b: str, score: float = 0.9) -> float:
    """Score short forms such as "west brom" / "west bromwich albion" or "b fernandes".

    Returns ``score`` when every token of the shorter name is a prefix of the
    token at the same position in the longer name, otherwise 0.
    """
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0
    if len(ta) > len(tb):
        ta, tb = tb, ta
    if all(long.startswith(short) for short, long in zip(ta, tb)):
        return score
    # Allow a missing given name or initial: "fernandes" vs "bruno fernandes".
    if len(ta) < len(tb) and ta[-1] == tb[-1] and all(
        long.startswith(short) for short, long in zip(ta[:-1], tb[: len(ta) - 1])
    ):
        return score
    return 0.0


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def basic_normalize(name: str) -> str:
    text = strip_accents(str(name)).lower().replace("&", " and ")
    text = re.sub(r"['’`.]", "", text)  # "Nott'm" -> "nottm", "B." -> "b"
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


TEAM_ABBREVIATIONS = {
    "man": "manchester",
    "utd": "united",
    "nottm": "nottingham",
    "nottham": "nottingham",
    "notts": "nottingham",
    "sheff": "sheffield",
    "weds": "wednesday",
    "wed": "wednesday",
    "qpr": "queens park rangers",
    "wba": "west bromwich albion",
}
TEAM_STOPWORDS = {"fc", "afc", "the", "cf"}


def normalize_team_name(name: str) -> str:
    tokens = []
    for token in basic_normalize(name).split():
        if token in TEAM_STOPWORDS:
            continue
        tokens.extend(TEAM_ABBREVIATIONS.get(token, token).split())
    return " ".join(tokens)


# Clubs that have played in the Premier League since 2000/01, canonical spelling.
CANONICAL_TEAMS = [
    "Arsenal", "Aston Villa", "Barnsley", "Birmingham City", "Blackburn Rovers",
    "Blackpool", "Bolton Wanderers", "Bournemouth", "Bradford City", "Brentford",
    "Brighton and Hove Albion", "Burnley", "Cardiff City", "Charlton Athletic",
    "Chelsea", "Coventry City", "Crystal Palace", "Derby County", "Everton",
    "Fulham", "Huddersfield Town", "Hull City", "Ipswich Town", "Leeds United",
    "Leicester City", "Liverpool", "Luton Town", "Manchester City",
    "Manchester United", "Middlesbrough", "Newcastle United", "Norwich City",
    "Nottingham Forest", "Portsmouth", "Queens Park Rangers", "Reading",
    "Sheffield United", "Sheffield Wednesday", "Southampton", "Stoke City",
    "Sunderland", "Swansea City", "Tottenham Hotspur", "Watford",
    "West Bromwich Albion", "West Ham United", "Wigan Athletic",
    "Wolverhampton Wanderers",
]

# Nicknames and abbreviations that no string metric can recover.
TEAM_ALIASES = {
    "Wolverhampton Wanderers": ["Wolves"],
    "Tottenham Hotspur": ["Spurs"],
    "Manchester United": ["Man U", "MUFC"],
    "Manchester City": ["MCFC"],
    "Brighton and Hove Albion": ["Brighton Hove"],
    "West Bromwich Albion": ["West Brom"],
    "Nottingham Forest": ["Forest"],
    "Wigan Athletic": ["Wigan"],
}


class UnresolvedEntityError(LookupError):
    pass


@dataclass(frozen=True)
class Resolution:
    query: str
    canonical: str
    entity_id: str
    score: float
    method: str  # "exact" | "alias" | "fuzzy" | "created"


class FuzzyMatcher:
    """Resolve free-text names against a register of canonical entities."""

    def __init__(
        self,
        canonical: Iterable[str],
        aliases: dict[str, Iterable[str]] | None = None,
        *,
        kind: str = "entity",
        normalizer: Callable[[str], str] = basic_normalize,
        threshold: float = 0.85,
        ambiguity_margin: float = 0.03,
    ) -> None:
        self.kind = kind
        self.normalizer = normalizer
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin
        self._keys: dict[str, str] = {}  # normalised key -> canonical name
        self._canonical: dict[str, str] = {}  # canonical name -> entity id
        self._cache: dict[str, Resolution] = {}
        for name in canonical:
            self.add_entity(name)
        for name, names in (aliases or {}).items():
            for alias in names:
                self.add_alias(alias, name)

    @property
    def entities(self) -> dict[str, str]:
        return dict(self._canonical)

    def add_entity(self, canonical: str) -> str:
        entity_id = self._canonical.setdefault(canonical, entity_uuid(self.kind, canonical))
        self._keys[self.normalizer(canonical)] = canonical
        self._cache.clear()
        return entity_id

    def add_alias(self, alias: str, canonical: str) -> None:
        if canonical not in self._canonical:
            raise KeyError(f"Unknown canonical {self.kind}: {canonical!r}")
        self._keys[self.normalizer(alias)] = canonical
        self._cache.clear()

    def candidates(self, name: str, limit: int = 3) -> list[tuple[str, float]]:
        """Best canonical candidates with their similarity score, highest first."""
        key = self.normalizer(name)
        best: dict[str, float] = {}
        for candidate_key, canonical in self._keys.items():
            score = max(
                levenshtein_similarity(key, candidate_key),
                prefix_token_similarity(key, candidate_key),
            )
            if score > best.get(canonical, -1.0):
                best[canonical] = score
        return sorted(best.items(), key=lambda kv: -kv[1])[:limit]

    def resolve(self, name: str, *, create_missing: bool = False) -> Resolution:
        if name in self._cache:
            return self._cache[name]
        key = self.normalizer(name)
        if key in self._keys:
            canonical = self._keys[key]
            method = "exact" if self.normalizer(canonical) == key else "alias"
            result = Resolution(name, canonical, self._canonical[canonical], 1.0, method)
        else:
            ranked = self.candidates(name, limit=2)
            top_name, top_score = ranked[0] if ranked else ("", 0.0)
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
            if top_score >= self.threshold and top_score - runner_up > self.ambiguity_margin:
                result = Resolution(name, top_name, self._canonical[top_name], top_score, "fuzzy")
                # Remember the spelling so later lookups are exact.
                self._keys[key] = top_name
            elif create_missing:
                entity_id = self.add_entity(name)
                result = Resolution(name, name, entity_id, 1.0, "created")
            else:
                raise UnresolvedEntityError(
                    f"Cannot resolve {self.kind} {name!r}; best candidates: {ranked}"
                )
        self._cache[name] = result
        return result

    def resolve_id(self, name: str, *, create_missing: bool = False) -> str:
        return self.resolve(name, create_missing=create_missing).entity_id


def team_resolver(**kwargs) -> FuzzyMatcher:
    """Matcher pre-loaded with every Premier League club and known aliases."""
    return FuzzyMatcher(
        CANONICAL_TEAMS,
        TEAM_ALIASES,
        kind="team",
        normalizer=normalize_team_name,
        **kwargs,
    )


def player_resolver(names: Iterable[str], **kwargs) -> FuzzyMatcher:
    kwargs.setdefault("threshold", 0.8)
    return FuzzyMatcher(names, kind="player", **kwargs)
