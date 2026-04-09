import asyncio
import time
from contextlib import suppress

class MockConn:
    async def cancel_order(self, oid, symbol):
        await asyncio.sleep(0.1) # Simulate network delay
        if oid == "error":
            raise ValueError("Mock error")

class MockConfig:
    paper_mode = False

CFG = MockConfig()

class Pos:
    def __init__(self):
        self.tp_order_id = "1"
        self.sl_order_id = "2"
        self.symbol = "BTCUSDT"
        self.status = "open"
        self.side = "long"
        self.qty = 1.0

async def sequential_cancel(pos, conn):
    if not CFG.paper_mode:
        for oid in [pos.tp_order_id, pos.sl_order_id]:
            if oid:
                with suppress(Exception):
                    await conn.cancel_order(oid, pos.symbol)

async def parallel_cancel(pos, conn):
    if not CFG.paper_mode:
        tasks = []
        for oid in [pos.tp_order_id, pos.sl_order_id]:
            if oid:
                tasks.append(conn.cancel_order(oid, pos.symbol))
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

async def run_benchmark():
    conn = MockConn()
    pos = Pos()

    t0 = time.time()
    await sequential_cancel(pos, conn)
    t1 = time.time()
    print(f"Sequential time: {t1 - t0:.4f}s")

    t0 = time.time()
    await parallel_cancel(pos, conn)
    t1 = time.time()
    print(f"Parallel time: {t1 - t0:.4f}s")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
