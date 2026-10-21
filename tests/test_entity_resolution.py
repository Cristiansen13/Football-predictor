import pytest

from football_predictor.etl.entity_resolution import (
    UnresolvedEntityError,
    levenshtein,
    levenshtein_similarity,
    normalize_team_name,
    player_resolver,
    team_resolver,
    team_uuid,
)


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [("kitten", "sitting", 3), ("", "abc", 3), ("same", "same", 0), ("flaw", "lawn", 2)],
)
def test_levenshtein(a, b, distance):
    assert levenshtein(a, b) == distance
    assert levenshtein(b, a) == distance


def test_similarity_bounds():
    assert levenshtein_similarity("abc", "abc") == 1.0
    assert levenshtein_similarity("abc", "xyz") == 0.0


@pytest.mark.parametrize(
    ("variant", "canonical"),
    [
        ("Man Utd", "Manchester United"),
        ("Manchester Utd", "Manchester United"),
        ("Man United", "Manchester United"),
        ("Man City", "Manchester City"),
        ("Nott'm Forest", "Nottingham Forest"),
        ("Nott'ham Forest", "Nottingham Forest"),
        ("Wolves", "Wolverhampton Wanderers"),
        ("Spurs", "Tottenham Hotspur"),
        ("Tottenham", "Tottenham Hotspur"),
        ("Brighton", "Brighton and Hove Albion"),
        ("Brighton & Hove Albion", "Brighton and Hove Albion"),
        ("West Brom", "West Bromwich Albion"),
        ("Sheffield Weds", "Sheffield Wednesday"),
        ("Sheffield Utd", "Sheffield United"),
        ("AFC Bournemouth", "Bournemouth"),
        ("QPR", "Queens Park Rangers"),
        ("Newcastle Utd", "Newcastle United"),
        ("Leicester", "Leicester City"),
        ("Totenham Hotspur", "Tottenham Hotspur"),  # typo, plain Levenshtein
    ],
)
def test_team_variants_resolve(variant, canonical):
    resolution = team_resolver().resolve(variant)
    assert resolution.canonical == canonical
    assert resolution.entity_id == team_uuid(canonical)


def test_ambiguous_names_are_rejected():
    resolver = team_resolver()
    for name in ("Manchester", "Sheffield", "United"):
        with pytest.raises(UnresolvedEntityError):
            resolver.resolve(name)


def test_unknown_team_can_be_registered():
    resolver = team_resolver()
    with pytest.raises(UnresolvedEntityError):
        resolver.resolve("Real Madrid")
    created = resolver.resolve("Real Madrid", create_missing=True)
    assert created.method == "created"
    assert resolver.resolve("Real Madrid").entity_id == created.entity_id


def test_uuid_is_deterministic():
    assert team_uuid("Arsenal") == team_uuid("Arsenal")
    assert team_uuid("Arsenal") != team_uuid("Chelsea")


def test_normalization():
    assert normalize_team_name("  Man. Utd FC ") == "manchester united"


def test_player_matching_handles_initials_and_accents():
    resolver = player_resolver(["Bruno Fernandes", "Kevin De Bruyne", "Martin Ødegaard", "André Onana"])
    assert resolver.resolve("B. Fernandes").canonical == "Bruno Fernandes"
    assert resolver.resolve("K. De Bruyne").canonical == "Kevin De Bruyne"
    assert resolver.resolve("Andre Onana").canonical == "André Onana"
    assert resolver.resolve("Fernandes").canonical == "Bruno Fernandes"
