## 性能测试（performance）

本目录为 **客观评价** 中的「与 NotebookLM 对比」提供 **本系统** 的性能与资源占用数据，用于填写 `evaluation_objective` 中的结果表（响应时间、CPU/内存等）。

### 定位

- **本脚本只测本系统**：NotebookLM 的延迟与耗能需在浏览器/官方环境中另行测量并填入客观评价表格。
- **产出指标**（与 `evaluation_objective/README.md` 对齐）：
  - **T_full**：单次请求从发起到完整响应结束的耗时（秒）。
  - **T_first_token**（若支持流式）：从发起到首个 token 出现的时间（秒）。
  - **CPU_avg / CPU_peak**：运行期间 CPU 利用率（%）。
  - **Mem_peak**：进程内存占用峰值（MB）。

### 目录内容

- `run_performance_tests.py`：统一入口。内含通用计时与资源统计工具，以及可配置的「系统级」性能测试（可替换为你的真实 pipeline 或 LLM 调用）。

### 使用方式

在项目根目录执行：

```bash
python -m test.performance.run_performance_tests
```

输出示例（可抄入客观评价表格）：

- `t_full_s`：完整响应耗时
- `t_first_token_s`：首 token 耗时（若为流式）
- `cpu_avg_pct` / `cpu_peak_pct`
- `mem_peak_mb`

### 可选依赖

- **psutil**：用于统计 CPU、内存。未安装时脚本仍可运行，但资源相关字段为 `null`。安装：`pip install psutil`。

### 与客观评价的衔接

1. 用本脚本对本系统做多轮测试，记录平均/中位数填入「本系统」行。
2. NotebookLM 在相同或相近任务下人工测一次（或多次取平均），填入「NotebookLM」行。
3. 对比两者在 T_full、资源占用上的差异，写入 `evaluation_objective` 的结论与图表。
