from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Literal, Self, TypeAlias, TypedDict

from pydantic import BaseModel, ConfigDict

Worker: TypeAlias = Callable[[str], Awaitable[str]]
ResultProcessor: TypeAlias = Callable[[str], str]

JobStatus: TypeAlias = Literal["pending", "inflight", "done", "failed"]
TerminalJobStatus: TypeAlias = Literal["done", "failed"]
ReturnDType: TypeAlias = Literal[
    "list[dict]",
    "list[str]",
    "list[tuple[str,str]]",
    "polars",
]
OutputShape: TypeAlias = Literal["unique", "original"]

if TYPE_CHECKING:
    import polars as pl

    PolarsDataFrame: TypeAlias = pl.DataFrame
else:
    PolarsDataFrame: TypeAlias = object


class ResultRowDict(TypedDict):
    idx: int
    key: str
    prompt: str
    status: TerminalJobStatus
    result: str | None
    error: str | None


class JobRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    idx: int
    key: str
    prompt: str
    status: JobStatus
    result: str | None = None
    error: str | None = None
    attempts: int = 0


class ResultRowBase(BaseModel):
    model_config = ConfigDict(frozen=True)

    idx: int
    key: str
    prompt: str

    def with_idx(self, idx: int) -> Self:
        return self.model_copy(update={"idx": idx})


class DoneResultRow(ResultRowBase):
    status: Literal["done"] = "done"
    result: str
    error: None = None


class FailedResultRow(ResultRowBase):
    status: Literal["failed"] = "failed"
    result: None = None
    error: str


ResultRow: TypeAlias = DoneResultRow | FailedResultRow


def result_row_from_job(job: JobRow) -> ResultRow:
    if job.status == "done":
        if job.result is None:
            raise ValueError(f"Done job {job.key!r} is missing a result.")
        if job.error is not None:
            raise ValueError(f"Done job {job.key!r} unexpectedly has an error.")
        return DoneResultRow(
            idx=job.idx,
            key=job.key,
            prompt=job.prompt,
            result=job.result,
        )

    if job.status == "failed":
        if not job.error:
            raise ValueError(f"Failed job {job.key!r} is missing an error message.")
        if job.result is not None:
            raise ValueError(f"Failed job {job.key!r} unexpectedly has a result.")
        return FailedResultRow(
            idx=job.idx,
            key=job.key,
            prompt=job.prompt,
            error=job.error,
        )

    raise ValueError(
        f"Job {job.key!r} has non-terminal status {job.status!r}; terminal results must be done or failed."
    )


def result_row_to_dict(row: ResultRow) -> ResultRowDict:
    if isinstance(row, DoneResultRow):
        return {
            "idx": row.idx,
            "key": row.key,
            "prompt": row.prompt,
            "status": row.status,
            "result": row.result,
            "error": row.error,
        }

    if isinstance(row, FailedResultRow):
        return {
            "idx": row.idx,
            "key": row.key,
            "prompt": row.prompt,
            "status": row.status,
            "result": row.result,
            "error": row.error,
        }

    raise TypeError(f"Unsupported result row type: {type(row).__name__}")


__all__ = [
    "DoneResultRow",
    "FailedResultRow",
    "JobRow",
    "JobStatus",
    "OutputShape",
    "PolarsDataFrame",
    "ResultRow",
    "ResultRowBase",
    "ResultRowDict",
    "ResultProcessor",
    "ReturnDType",
    "TerminalJobStatus",
    "Worker",
    "result_row_from_job",
    "result_row_to_dict",
]
