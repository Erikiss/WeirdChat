from collections.abc import Awaitable, Sequence
from typing import TypeVar, cast

import anyio
from tqdm.auto import tqdm

R = TypeVar("R")


async def run_bounded(
    futures: Sequence[Awaitable[R]],
    concurrency: int,
    label: str | None = None,
    position: int | None = None,
) -> list[R]:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    results: list[R | None] = [None] * len(futures)
    pending = iter(enumerate(futures))

    async def run() -> None:
        for index, future in pending:
            try:
                results[index] = await future
            finally:
                bar.update()

    with tqdm(total=len(futures), desc=label, unit="task", position=position) as bar:
        async with anyio.create_task_group() as group:
            for _ in range(min(concurrency, len(futures))):
                group.start_soon(run)

    return cast(list[R], results)
