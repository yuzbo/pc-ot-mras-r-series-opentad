# Research Wiki Lint Report

生成时间：2026-04-09T12:55:01+08:00

## P0（必须修复）
1. **图谱长期为空的问题刚被补齐**
   - 之前 graph/edges.jsonl 为空，导致所有页面都是 orphan，Connections 机制失效。
   - 当前已补入第一批基础 edges，但仍然远不完整。

2. **实验覆盖与 wiki 覆盖不一致**
   - wiki 中只有 7 个实验页，但项目里实际已有大量后续实验、诊断配置和 projection 变体。
   - 这会导致后续 query / ideation 读到的是“过时状态”。

## P1（强烈建议）
1. **缺少近期关键实验页面**
   - 建议补齐：R019 / R020 / R022 / oracle_50pct / coarse_inject / projection 优化系列。

2. **gap_map 之前过于空泛**
   - 已更新为 G1-G5，但还缺少与每个实验页的显式双向连接。

3. **部分页面存在内容过短或编码显示异常现象**
   - 现有旧页面中有一些中文显示在 PowerShell 下有乱码痕迹；建议统一 UTF-8 并在后续编辑中逐步清洗。

## P2（可选优化）
1. 为每个 experiment / idea / claim 自动生成 ## Connections 区块。
2. 给 experiment 页增加统一字段：vg_map, map_05, map_07, confidence。
3. 增加一页 efficiency.md 记录 FLOPs / latency / throughput。

## 当前统计
- Papers: 3
- Ideas: 6
- Experiments: 8（已新增状态快照页）
- Claims: 6
- Edges: 18

## 建议修复顺序
1. 先补近期关键实验页（oracle / R020 / coarse inject / projection 优化）
2. 再把这些实验与 G1-G5、claims、ideas 建立 edges
3. 最后做自动化 Connections / stats / query_pack 生成
