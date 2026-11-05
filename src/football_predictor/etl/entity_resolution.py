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
PREMIER_LEAGUE_TEAMS = [
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

# Other Championship clubs since 2005/06, needed to rate promoted sides.
CHAMPIONSHIP_TEAMS = [
    "Bristol City", "Burton Albion", "Colchester United", "Crewe Alexandra",
    "Doncaster Rovers", "Gillingham", "Milton Keynes Dons", "Millwall",
    "Oxford United", "Peterborough United", "Plymouth Argyle", "Preston North End",
    "Rotherham United", "Scunthorpe United", "Southend United", "Wrexham",
    "Wycombe Wanderers", "Yeovil Town",
]

# League One and Two clubs whose Football-Data spelling is a near-miss of a
# Premier League club's ("Northampton" scores 0.82 against Southampton):
# fuzzy matching would neither accept nor create them.
LOWER_LEAGUE_TEAMS = ["Northampton Town"]

CANONICAL_TEAMS = PREMIER_LEAGUE_TEAMS + CHAMPIONSHIP_TEAMS + LOWER_LEAGUE_TEAMS

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
    "Milton Keynes Dons": ["MK Dons"],
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
        create_below: float = 0.75,
    ) -> None:
        """
        ``create_below``: with ``create_missing=True`` a new entity is only
        registered when no existing one scores at least this much, so a
        near-miss or ambiguous spelling raises instead of creating a duplicate.
        """
        self.kind = kind
        self.normalizer = normalizer
        self.threshold = threshold
        self.ambiguity_margin = ambiguity_margin
        self.create_below = create_below
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
            elif create_missing and top_score < self.create_below:
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

    def register(self, name: str) -> Resolution:
        """Resolve ``name`` by its exact (normalised) spelling only, registering it when new.

        For a source whose names are consistent and self-contained, such as
        Football-Data's files for the leagues outside England, a fuzzy match
        against the English register does harm: "Rangers" is not Queens Park
        Rangers, "Dundee" is not Dundee United and "Gent" is a different club
        from "Genk". Once registered, the name resolves exactly ever after.
        """
        key = self.normalizer(name)
        if key in self._keys:
            canonical = self._keys[key]
            method = "exact" if self.normalizer(canonical) == key else "alias"
            return Resolution(name, canonical, self._canonical[canonical], 1.0, method)
        return Resolution(name, name, self.add_entity(name), 1.0, "created")


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


# -- the clubs of one league under another source's spelling --------------------

# Legal forms, city prefixes and other tokens one source writes and another
# drops ("FC Augsburg" / "Augsburg", "Stade de Reims" / "Reims").
CLUB_STOPWORDS = TEAM_STOPWORDS | {
    "sc", "ac", "as", "us", "ss", "ssc", "rc", "rcd", "cd", "ud", "sd", "ca", "cp", "bc", "bk", "fk", "sk", "if",
    "sv", "vfl", "vfb", "tsg", "tsv", "fsv", "bsc", "kv", "kvc", "krc", "kaa", "ksv", "rsc", "royal", "ogc", "aj",
    "ea", "es", "usl", "hsc", "sco", "af", "foot", "calcio", "spa", "ssd", "asd", "de", "del", "di", "la", "le",
    "les", "fcv", "eh", "ev", "of",
}
CLUB_ABBREVIATIONS = TEAM_ABBREVIATIONS | {"nott": "nottingham", "saint": "st", "sint": "st", "sporting": "sp",
                                           "atletico": "ath", "athletic": "ath"}


def club_tokens(name: str) -> list[str]:
    """The informative tokens of a club name: accents stripped, apostrophes and
    hyphens split ("M'gladbach" -> gladbach), abbreviations expanded, legal
    forms, numbers and single letters dropped."""
    text = strip_accents(str(name)).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    tokens = []
    for token in text.split():
        for part in CLUB_ABBREVIATIONS.get(token, token).split():
            if part in CLUB_STOPWORDS or part.isdigit() or len(part) < 2:
                continue
            tokens.append(part)
    return tokens


def _club_tokens_match(a: str, b: str) -> bool:
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 3 and long.startswith(short):  # "ath" / "atletico", "lyon" / "lyonnais"
        return True
    if len(short) >= 5 and long.endswith(short):  # "gladbach" / "monchengladbach"
        return True
    return len(short) >= 5 and levenshtein_similarity(a, b) >= 0.8  # "olympiakos" / "olympiacos"


def _initials(tokens: list[str], others: list[str]) -> dict[int, list[int]]:
    """Positions of short tokens that spell the initials of consecutive ``others`` ("sg" / "saint germain")."""
    found = {}
    for i, token in enumerate(tokens):
        if not 2 <= len(token) <= 3:
            continue
        for j in range(len(others) - len(token) + 1):
            if "".join(o[0] for o in others[j:j + len(token)]) == token:
                found[i] = list(range(j, j + len(token)))
                break
    return found


def club_overlap(query: list[str], candidate: list[str], weight: dict[str, float] | None = None) -> float:
    """How much of each of two club spellings the other covers, averaged, in [0, 1].

    Tokens pair up when equal, when one is a prefix or a long suffix of the
    other, when they are near-identical, or when one spells the initials of
    consecutive tokens of the other. Each side's coverage is the ``weight``
    (default 1) of its matched tokens over that of all its tokens; an
    unmatched token of the query that no candidate carries weighs half, since
    it says nothing about which club is meant ("Hellas", "Borussia"). The
    candidate's own coverage counts as much as the query's, so "Real Betis"
    covers "Betis" whole and "Real Madrid" by half.
    """
    weight = weight or {}
    if not query or not candidate:
        return 0.0
    pairs = {(i, j) for i, q in enumerate(query) for j, c in enumerate(candidate) if _club_tokens_match(q, c)}
    for i, js in _initials(query, candidate).items():
        pairs |= {(i, j) for j in js}
    for j, is_ in _initials(candidate, query).items():
        pairs |= {(i, j) for i in is_}
    matched_q = {i: weight.get(candidate[j], weight.get(query[i], 1.0)) for i, j in pairs}
    matched_c = {j: weight.get(candidate[j], 1.0) for _, j in pairs}
    total_q = sum(matched_q.get(i, weight.get(q, 0.5)) for i, q in enumerate(query))
    total_c = sum(matched_c.get(j, weight.get(c, 1.0)) for j, c in enumerate(candidate))
    return 0.5 * (sum(matched_q.values()) / total_q + sum(matched_c.values()) / total_c)


class ClubMatcher:
    """Match another source's spellings against the clubs of one league.

    ``clubs`` maps each team id to its known names (canonical and aliases).
    A name is scored against every club by :func:`club_overlap`, with each
    token weighted by the inverse of the number of clubs that carry it, so
    "city" in a league of six Cities counts for little and "hull" for
    everything. A name resolves when the best club scores at least
    ``threshold`` and beats the runner-up by ``margin``; agreeing on the first
    token adds ``first_token_bonus``, which settles "Inter Milan" for Inter
    rather than Milan. A spelling no token can bridge ("Stade Rennais" for
    Rennes) stays unresolved and is left to an alias.
    """

    def __init__(
        self,
        clubs: dict[str, Iterable[str]],
        *,
        threshold: float = 0.5,
        margin: float = 0.05,
        first_token_bonus: float = 0.05,
    ) -> None:
        self.threshold, self.margin, self.first_token_bonus = threshold, margin, first_token_bonus
        self._raw = {team: [str(n) for n in names] for team, names in clubs.items()}
        self._names = {team: [club_tokens(n) for n in names] for team, names in clubs.items()}
        self._names = {team: [t for t in tokens if t] for team, tokens in self._names.items()}
        carriers: dict[str, set[str]] = {}
        for team, spellings in self._names.items():
            for tokens in spellings:
                for token in tokens:
                    carriers.setdefault(token, set()).add(team)
        self._weight = {token: 1.0 / len(teams) for token, teams in carriers.items()}

    def candidates(self, name: str, limit: int = 3) -> list[tuple[str, float]]:
        """Best clubs with their score, highest first."""
        query = club_tokens(name)
        scored = []
        for team, spellings in self._names.items():
            best = 0.0
            for tokens in spellings:
                score = club_overlap(query, tokens, self._weight)
                if score and query and tokens and _club_tokens_match(query[0], tokens[0]):
                    score += self.first_token_bonus
                best = max(best, score)
            if best > 0:
                scored.append((team, best))
        return sorted(scored, key=lambda kv: -kv[1])[:limit]

    def match(self, name: str) -> tuple[str, float] | None:
        """(team id, score) of the club ``name`` denotes, or None when no club clearly does.

        A tie between clubs is settled for the one spelt exactly like
        ``name``, if there is one: a club a source has registered twice
        ("Mechelen", then "KV Mechelen") resolves to its current spelling.
        """
        ranked = self.candidates(name, limit=2)
        if not ranked:
            return None
        team, score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if score >= self.threshold and score - runner_up >= self.margin - 1e-9:
            return team, min(score, 1.0)
        key = basic_normalize(name)
        exact = [t for t, s in ranked if s >= self.threshold and any(basic_normalize(n) == key for n in self._raw[t])]
        if len(exact) == 1:
            return exact[0], 1.0
        return None
