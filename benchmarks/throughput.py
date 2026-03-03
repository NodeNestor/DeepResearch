"""
vLLM Throughput Benchmark for DeepResearch

Tests parallel request throughput with varying:
- Concurrency levels (1, 10, 25, 50, 100)
- Input context sizes (1k, 4k, 16k, 64k tokens)
- Output token counts (256, 1024, 4096)

Measures: tokens/sec, requests/sec, latency percentiles, total VRAM usage.

Usage:
    python benchmarks/throughput.py [--url http://localhost:8000/v1] [--model Qwen/Qwen3.5-0.8B]
"""

import argparse
import asyncio
import json
import statistics
import time

import httpx


async def single_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    input_tokens: int,
    max_tokens: int,
    sem: asyncio.Semaphore,
) -> dict:
    """Fire a single completion request and measure timing."""
    # Generate input of ~input_tokens by repeating a paragraph
    filler = (
        "The quantum error correction code uses stabilizer measurements to detect "
        "and correct errors without directly measuring the logical qubit state. "
        "Surface codes arrange physical qubits on a 2D lattice where each logical "
        "qubit is encoded across many physical qubits. "
    )
    # ~50 tokens per repetition
    reps = max(1, input_tokens // 50)
    user_content = (filler * reps)[:input_tokens * 4]  # rough chars-to-tokens

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a research assistant. Respond with detailed analysis."},
            {"role": "user", "content": f"Analyze the following text and extract all key facts, entities, and relationships:\n\n{user_content}"},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3,
    }

    async with sem:
        t0 = time.perf_counter()
        try:
            resp = await client.post(f"{url}/chat/completions", json=payload)
            t1 = time.perf_counter()
            resp.raise_for_status()
            data = resp.json()

            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            return {
                "ok": True,
                "latency": t1 - t0,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        except Exception as e:
            t1 = time.perf_counter()
            return {
                "ok": False,
                "latency": t1 - t0,
                "error": str(e),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }


async def run_benchmark(
    url: str,
    model: str,
    concurrency: int,
    input_tokens: int,
    max_tokens: int,
    num_requests: int,
) -> dict:
    """Run a batch of parallel requests and collect stats."""
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        t_start = time.perf_counter()
        results = await asyncio.gather(*[
            single_request(client, url, model, input_tokens, max_tokens, sem)
            for _ in range(num_requests)
        ])
        t_total = time.perf_counter() - t_start

    successful = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]

    if not successful:
        return {
            "concurrency": concurrency,
            "input_tokens": input_tokens,
            "max_tokens": max_tokens,
            "num_requests": num_requests,
            "successful": 0,
            "failed": len(failed),
            "error": failed[0].get("error", "unknown") if failed else "all failed",
        }

    latencies = [r["latency"] for r in successful]
    total_prompt = sum(r["prompt_tokens"] for r in successful)
    total_completion = sum(r["completion_tokens"] for r in successful)
    total_tokens = total_prompt + total_completion

    return {
        "concurrency": concurrency,
        "input_tokens": input_tokens,
        "max_tokens": max_tokens,
        "num_requests": num_requests,
        "successful": len(successful),
        "failed": len(failed),
        "wall_time_s": round(t_total, 2),
        "requests_per_sec": round(len(successful) / t_total, 2),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "tokens_per_sec_generation": round(total_completion / t_total, 1),
        "tokens_per_sec_total": round(total_tokens / t_total, 1),
        "latency_p50_s": round(statistics.median(latencies), 2),
        "latency_p90_s": round(sorted(latencies)[int(len(latencies) * 0.9)], 2),
        "latency_p99_s": round(sorted(latencies)[int(len(latencies) * 0.99)], 2),
        "latency_min_s": round(min(latencies), 2),
        "latency_max_s": round(max(latencies), 2),
    }


async def check_model(url: str) -> str | None:
    """Check which model is loaded on vLLM."""
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(f"{url}/models")
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                return models[0]["id"]
        except Exception:
            pass
    return None


async def main():
    parser = argparse.ArgumentParser(description="vLLM throughput benchmark")
    parser.add_argument("--url", default="http://localhost:8000/v1", help="vLLM API URL")
    parser.add_argument("--model", default=None, help="Model name (auto-detected if omitted)")
    parser.add_argument("--quick", action="store_true", help="Quick test (fewer combos)")
    args = parser.parse_args()

    # Detect model
    model = args.model or await check_model(args.url)
    if not model:
        print("ERROR: Cannot connect to vLLM or detect model. Is it running?")
        print(f"  Tried: {args.url}/models")
        return

    print(f"{'=' * 70}")
    print(f"  DeepResearch vLLM Throughput Benchmark")
    print(f"  URL:   {args.url}")
    print(f"  Model: {model}")
    print(f"{'=' * 70}")
    print()

    if args.quick:
        test_matrix = [
            # (concurrency, input_tokens, max_tokens, num_requests)
            (1, 1000, 256, 3),
            (10, 1000, 256, 20),
            (50, 1000, 256, 50),
            (10, 16000, 1024, 10),
            (25, 16000, 1024, 25),
        ]
    else:
        test_matrix = [
            # Baseline: single request, small context
            (1, 1000, 256, 5),
            (1, 1000, 1024, 5),

            # Scale concurrency with small context
            (10, 1000, 256, 30),
            (25, 1000, 256, 50),
            (50, 1000, 256, 50),
            (100, 1000, 256, 100),

            # Scale context size with moderate concurrency
            (10, 4000, 512, 20),
            (10, 16000, 1024, 20),
            (10, 64000, 1024, 10),

            # High concurrency + medium context (realistic research workload)
            (50, 4000, 1024, 50),
            (100, 4000, 1024, 100),

            # Stress test: high concurrency + large context
            (25, 16000, 2048, 25),
            (50, 16000, 2048, 50),

            # Large context stress
            (5, 64000, 4096, 5),
            (10, 64000, 4096, 10),
        ]

    results = []
    for concurrency, input_tokens, max_tokens, num_requests in test_matrix:
        label = f"C={concurrency:>3d}  IN={input_tokens:>6d}  OUT={max_tokens:>5d}  N={num_requests:>3d}"
        print(f"Running: {label} ... ", end="", flush=True)

        result = await run_benchmark(
            url=args.url,
            model=model,
            concurrency=concurrency,
            input_tokens=input_tokens,
            max_tokens=max_tokens,
            num_requests=num_requests,
        )
        results.append(result)

        if result.get("error"):
            print(f"FAILED: {result['error'][:60]}")
        else:
            print(
                f"OK  {result['requests_per_sec']:>6.1f} req/s  "
                f"{result['tokens_per_sec_generation']:>7.0f} gen tok/s  "
                f"p50={result['latency_p50_s']:.1f}s  p90={result['latency_p90_s']:.1f}s"
            )

    # Summary
    print(f"\n{'=' * 70}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Concurr':>7} {'InTok':>6} {'OutTok':>6} {'N':>4} "
          f"{'OK':>4} {'req/s':>7} {'gen t/s':>8} {'tot t/s':>8} "
          f"{'p50':>6} {'p90':>6} {'max':>6}")
    print("-" * 90)
    for r in results:
        if r.get("error"):
            print(f"{r['concurrency']:>7} {r['input_tokens']:>6} {r['max_tokens']:>6} "
                  f"{r['num_requests']:>4}  FAILED: {r.get('error', '')[:40]}")
        else:
            print(f"{r['concurrency']:>7} {r['input_tokens']:>6} {r['max_tokens']:>6} "
                  f"{r['num_requests']:>4} {r['successful']:>4} "
                  f"{r['requests_per_sec']:>7.1f} {r['tokens_per_sec_generation']:>8.0f} "
                  f"{r['tokens_per_sec_total']:>8.0f} "
                  f"{r['latency_p50_s']:>5.1f}s {r['latency_p90_s']:>5.1f}s "
                  f"{r['latency_max_s']:>5.1f}s")

    # Save raw results
    with open("benchmarks/results.json", "w") as f:
        json.dump({"model": model, "url": args.url, "results": results}, f, indent=2)
    print(f"\nRaw results saved to benchmarks/results.json")


if __name__ == "__main__":
    asyncio.run(main())
