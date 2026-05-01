---
type: experiment
node_id: exp:ORACLE50
title: "Oracle 50% Upper-Bound Test"
date: 2026-04-09
status: prepared
config: v3_oracle_50pct.py
---

# Oracle 50% Upper-Bound Test

## Purpose
验证“完美 50% 采样策略”在当前 V3 backbone + detector 框架下的理论上界，用来区分：
1. scorer 没学会；还是
2. 50% intelligent sampling 本身就没有太大空间。

## Status
- 配置已准备
- 需要正式结果写回

## Expected Use
如果 Oracle 结果接近或超过 uniform 50%，说明 scorer 学习仍有空间；反之说明主瓶颈可能不在 scorer，而在 detector / projection / sparse compatibility。
