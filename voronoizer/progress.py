"""Verbose/progress helpers. Quiet by default."""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Iterable, Iterator, TypeVar

_VERBOSE = False


def set_verbose(verbose: bool) -> None:
    global _VERBOSE
    _VERBOSE = verbose


def is_verbose() -> bool:
    return _VERBOSE


def log(msg: str) -> None:
    if _VERBOSE:
        print(msg, file=sys.stderr, flush=True)


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr, flush=True)


@contextmanager
def step(name: str) -> Iterator[None]:
    """Time a pipeline stage; only prints when verbose."""
    if _VERBOSE:
        print(f"[..] {name}", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    try:
        yield
    except Exception:
        if _VERBOSE:
            dt = time.perf_counter() - t0
            print(f"[!!] {name} failed after {dt:.2f}s", file=sys.stderr, flush=True)
        raise
    dt = time.perf_counter() - t0
    if _VERBOSE:
        print(f"[ok] {name} ({dt:.2f}s)", file=sys.stderr, flush=True)


T = TypeVar("T")


def progress(iterable: Iterable[T], desc: str, total: int | None = None) -> Iterable[T]:
    """tqdm wrapper that's a no-op when not verbose."""
    if not _VERBOSE:
        return iterable
    from tqdm import tqdm
    return tqdm(iterable, desc=desc, total=total, leave=False)
