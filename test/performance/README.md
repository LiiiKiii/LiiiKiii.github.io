# AI-Pedia 性能测试（performance）

本目录包含 **当前项目实际使用的本地性能基准脚本**，用于衡量 AI-Pedia 中确定性本地阶段的运行开销。

它不是旧版“外部系统对比”草案，而是为当前论文中的性能部分直接提供可复现输出。

## 主要用途

`run_performance_tests.py` 主要评估以下本地阶段：

- 文档读取
- 关键词提取
- 资源排序 / 推荐
- 本地组合 pipeline 的总体耗时

## 运行方式

在项目根目录执行：

```bash
python test/performance/run_performance_tests.py
```

## 生成结果

脚本会在 `test/performance/results/` 下生成：

- `performance_results.json`
- `performance_summary.csv`
- `performance_table.tex`

并为论文图表导出对应结果。

## 当前定位

这部分性能测试的目标是回答：

- 本地确定性阶段是否足够轻量
- 哪些步骤最耗时
- 当前实现是否适合作为可运行、可部署的课程项目原型

它只依赖当前仓库中的性能脚本与结果目录，不依赖任何已经移除的历史评估草案。

## 使用建议

如果你要在论文或答辩中解释这一部分，最准确的说法是：

> `test/performance/` contains reproducible local benchmarks for the deterministic stages of the AI-Pedia pipeline, and its outputs are exported directly into paper-ready tables and figures.
