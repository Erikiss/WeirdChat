from typing import Literal

import anyio
import pytest

from weirdchat.concurrency import run_bounded


@pytest.mark.parametrize("backend", ["asyncio", "trio"])
def test_run_bounded(backend: Literal["asyncio", "trio"]) -> None:
    async def check() -> None:
        active = 0
        peak = 0

        async def double(value: int) -> int:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await anyio.sleep(value / 100)
            active -= 1
            return value * 2

        futures = [double(value) for value in [4, 3, 2, 1]]
        results = await run_bounded(futures, 2, "double")

        assert results == [8, 6, 4, 2]
        assert peak == 2

    anyio.run(check, backend=backend)


def test_run_bounded_rejects_invalid_concurrency() -> None:
    async def check() -> None:
        await run_bounded([], 0)

    with pytest.raises(ValueError, match="concurrency must be at least 1"):
        anyio.run(check)
