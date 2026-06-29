"""
MediRAG Pro — Performance Benchmark Script.

Measures end-to-end latency at different concurrency levels.
Requires the API to be running and documents to be ingested.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --url http://localhost:8000 --users 10 --requests 100
    python scripts/benchmark.py --output evaluation/benchmark_results.json
"""
import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

try:
    import aiohttp
except ImportError:
    print("Run: pip install aiohttp")
    raise

SAMPLE_QUERIES = [
    "What is hypertension and how is it classified?",
    "What are the first-line medications for treating hypertension?",
    "What is diabetes mellitus type 2?",
    "What are the diagnostic criteria for diabetes?",
    "What is the mechanism of action of aspirin?",
    "What are the symptoms of myocardial infarction?",
    "What is asthma and how is it treated?",
    "What are ACE inhibitors used for?",
    "What is the difference between Type 1 and Type 2 diabetes?",
    "What is heart failure?",
]


async def single_request(session: aiohttp.ClientSession, url: str, query: str) -> dict:
    """Make a single /chat request and return timing + metadata."""
    t0 = time.perf_counter()
    try:
        async with session.post(
            f"{url}/api/v1/chat",
            json={"query": query},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json()
            latency_ms = (time.perf_counter() - t0) * 1000
            return {
                "latency_ms": latency_ms,
                "status": resp.status,
                "cache_hit": data.get("cache_hit", False),
                "confidence": data.get("confidence", 0),
                "sources_count": len(data.get("sources", [])),
                "success": resp.status == 200,
            }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "latency_ms": latency_ms,
            "status": 0,
            "cache_hit": False,
            "confidence": 0,
            "sources_count": 0,
            "success": False,
            "error": str(e),
        }


async def run_concurrency_test(
    url: str, concurrent_users: int, total_requests: int
) -> dict:
    """Run benchmark at a specific concurrency level."""
    print(f"\n  Testing {concurrent_users} concurrent users, {total_requests} requests...")

    results = []
    semaphore = asyncio.Semaphore(concurrent_users)

    async def bounded_request(session, query):
        async with semaphore:
            return await single_request(session, url, query)

    async with aiohttp.ClientSession() as session:
        tasks = [
            bounded_request(session, SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)])
            for i in range(total_requests)
        ]
        t_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - t_start

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    latencies = sorted([r["latency_ms"] for r in successful])
    cache_hits = sum(1 for r in successful if r["cache_hit"])

    if not latencies:
        return {"error": "All requests failed", "concurrent_users": concurrent_users}

    n = len(latencies)
    return {
        "concurrent_users": concurrent_users,
        "total_requests": total_requests,
        "successful": len(successful),
        "failed": len(failed),
        "throughput_rps": round(len(successful) / total_time, 2),
        "cache_hit_rate": round(cache_hits / len(successful), 3) if successful else 0,
        "latency": {
            "p50_ms": round(latencies[int(n * 0.50)], 1),
            "p75_ms": round(latencies[int(n * 0.75)], 1),
            "p95_ms": round(latencies[int(n * 0.95)], 1),
            "p99_ms": round(latencies[int(n * 0.99)], 1),
            "min_ms": round(latencies[0], 1),
            "max_ms": round(latencies[-1], 1),
            "mean_ms": round(statistics.mean(latencies), 1),
        },
    }


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 75)
    print("MEDIRAG PRO — BENCHMARK RESULTS")
    print("=" * 75)
    print(f"{'Users':>6} {'P50':>8} {'P75':>8} {'P95':>8} {'P99':>8} {'RPS':>6} {'Cache%':>7} {'OK':>5}")
    print("-" * 75)
    for r in results:
        if "error" in r:
            print(f"{r['concurrent_users']:>6}  ERROR: {r['error']}")
            continue
        lat = r["latency"]
        print(
            f"{r['concurrent_users']:>6} "
            f"{lat['p50_ms']:>7.0f}ms "
            f"{lat['p75_ms']:>7.0f}ms "
            f"{lat['p95_ms']:>7.0f}ms "
            f"{lat['p99_ms']:>7.0f}ms "
            f"{r['throughput_rps']:>6.1f} "
            f"{r['cache_hit_rate']:>6.0%} "
            f"{r['successful']}/{r['total_requests']:>3}"
        )
    print("=" * 75)


async def main(url: str, max_users: int, requests_per_level: int, output: str | None) -> None:
    # Check API is reachable
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                health = await r.json()
                if health.get("status") != "healthy":
                    print(f"⚠️  API health: {health.get('status')} — proceed with caution")
                else:
                    print(f"✅ API healthy at {url}")
        except Exception as e:
            print(f"❌ Cannot reach API at {url}: {e}")
            return

    # Run at increasing concurrency levels
    concurrency_levels = [1, 5, max_users]
    all_results = []

    for users in concurrency_levels:
        if users > max_users:
            break
        result = await run_concurrency_test(url, users, requests_per_level)
        all_results.append(result)

    print_table(all_results)

    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "api_url": url,
        "results": all_results,
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Results saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediRAG Pro benchmark")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--users", type=int, default=10, help="Max concurrent users")
    parser.add_argument("--requests", type=int, default=30, help="Requests per concurrency level")
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    asyncio.run(main(args.url, args.users, args.requests, args.output))
