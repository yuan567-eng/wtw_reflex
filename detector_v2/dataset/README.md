# Detector V2 数据模块

此目录将保存 Detector V2 的专用采集脚本、数据格式说明和本地实验数据。

```text
scripts/       采集、归一化、窗口构建和绘图脚本
raw/           原始 episode 数据
normalized/    归一化后的数据
metadata/      数据版本、采集配置和字段说明
ablation/      弱扰动与特征消融实验数据
```

本目录不会修改或覆盖根目录 `dataset/` 中的第一版数据。
