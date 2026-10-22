"""A small Premier League match predictor: Poisson goal model, score grid, evaluation."""

from .data import download_season, load_seasons, parse_csv
from .evaluate import backtest, summarize
from .model import PoissonModel

__all__ = ["PoissonModel", "backtest", "download_season", "load_seasons", "parse_csv", "summarize"]
