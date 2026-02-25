#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试脚本（服务于客观评价：与 NotebookLM 对比时的速度与耗能）

产出指标：T_full、可选 T_first_token、CPU 利用率、内存峰值。
供 evaluation_objective 结果表使用。NotebookLM 需另行测量。
默认在项目根目录执行：python -m test.performance.run_performance_tests
"""

import os
import sys
import time
import threading
from typing import Any, Callable, Dict

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_pythonpath() -> None:
    root = project_root()
    if root not in sys.path:
        sys.path.insert(0, root)


setup_pythonpath()


def measure_with_resources(
    func: Callable[[], Any],
    repeat: int = 3,
) -> Dict[str, Any]:
    """
    多次执行 func()，统计耗时与资源占用（需 psutil）。
    返回: avg_time_s, total_time_s, t_full_s, cpu_avg_pct, cpu_peak_pct, mem_peak_mb
    """
    times: list[float] = []
    all_cpu: list[float] = []
    all_mem: list[float] = []

    for _ in range(repeat):
        start = time.perf_counter()
        if _HAS_PSUTIL:
            proc = psutil.Process(os.getpid())
            cpu_samples: list[float] = []
            mem_samples: list[float] = []
            end_evt = threading.Event()
            def sampler():
                while not end_evt.is_set():
                    try:
                        cpu_samples.append(proc.cpu_percent())
                        mem_samples.append(proc.memory_info().rss / (1024 * 1024))
                    except Exception:
                        break
                    time.sleep(0.05)
            t = threading.Thread(target=sampler, daemon=True)
            t.start()
        try:
            func()
        finally:
            if _HAS_PSUTIL:
                end_evt.set()
                time.sleep(0.1)
                all_cpu.extend(cpu_samples)
                all_mem.extend(mem_samples)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    total = sum(times)
    avg = total / len(times) if times else 0.0
    out: Dict[str, Any] = {
        "runs": repeat,
        "total_time_s": round(total, 4),
        "avg_time_s": round(avg, 4),
        "t_full_s": round(avg, 4),
    }
    if _HAS_PSUTIL and all_cpu and all_mem:
        out["cpu_avg_pct"] = round(sum(all_cpu) / len(all_cpu), 2)
        out["cpu_peak_pct"] = round(max(all_cpu), 2)
        out["mem_peak_mb"] = round(max(all_mem), 2)
    else:
        out["cpu_avg_pct"] = None
        out["cpu_peak_pct"] = None
        out["mem_peak_mb"] = None
    return out


def test_system_pipeline() -> Dict[str, Any]:
    """
    系统级性能测试：模拟或调用真实「用户请求 → 完整响应」的 pipeline。
    默认用一次简单后端调用或 sleep 占位；你可替换为真实 pipeline 入口。
    """
    def _run() -> None:
        try:
            from backend.core import recommender as rec
            user_docs = ["Machine learning and neural networks."] * 2
            resources = [
                {"title": f"R{i}", "content": f"Content {i} about ML."}
                for i in range(20)
            ]
            rec.recommend_best_resources(
                user_docs=user_docs,
                resources=resources,
                top_k=5,
            )
        except Exception:
            time.sleep(0.3)

    result = measure_with_resources(_run, repeat=3)
    result["module"] = "system_pipeline"
    result["status"] = "ok"
    result["note"] = "可替换为真实 pipeline（如含 LLM 的完整请求）以测 T_full 与资源"
    return result


def main() -> None:
    print("=== 性能测试（客观评价：速度与耗能）===")
    if not _HAS_PSUTIL:
        print("(未安装 psutil，仅输出耗时；安装后可得到 CPU/内存: pip install psutil)")

    results: list[Dict[str, Any]] = []
    try:
        res = test_system_pipeline()
        results.append(res)
        print("\n--- system_pipeline ---")
        print(res)
    except Exception as e:
        results.append({
            "module": "system_pipeline",
            "status": "error",
            "error": str(e),
        })
        print(f"\n[ERROR] system_pipeline: {e}")

    print("\n=== 性能测试结束 ===")
    print("将上述 t_full_s / cpu_avg_pct / mem_peak_mb 填入 evaluation_objective 结果表。")


if __name__ == "__main__":
    main()
