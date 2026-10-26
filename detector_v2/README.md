# Detector V2

Detector V2 是弱扰动优化阶段的独立工作区。

第一版 `dataset/`、`tcn_detector/` 和 `gru_detector/` 已完成 TCN/GRU 对比实验，保留为
可复现实验快照。V2 不修改或依赖旧版 Detector 源码；后续从数据采集开始逐模块重新实现。

目录职责：

```text
dataset/       数据采集、标签与数据版本
tcn_detector/  TCN 的独立开发与实验产物
gru_detector/  GRU 的独立开发与实验产物
```

TCN 和 GRU 可以使用相同的原始数据集，但各自维护特征处理、窗口构建、训练、预测和评估代码。
