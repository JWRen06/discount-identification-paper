# 公开来源与离线审计

获取日期2026-10-06，市场文件内部持仓日2026-10-02。

`MTUM_latest.csv`与`ishares_product.html`是未改写原始文件。结果JSON及逐行CSV保留全部129条记录的计算。它们不是独立行情、成交日志或重复随机冲击，横截面记录数不能替代实验重复次数。

父目录统一复现入口于临时目录重算本审计与此前理论/统计结果，逐字节比较归档输出。单独运行`audit_public_holdings.py`则只更新派生表，不改原始来源。

`historical_endpoint_response.html`是失败历史接口返回的网页，不是历史持仓。SEC原始XML直取403，未归档XML；Orland–Roos出版页直取也403，未取得实验数据。

Cao文件为正式补充材料；Komarova为作者接受稿；Shehab为正式公开正文；Skalse–Abate两篇及Rodrigues为固定arXiv版。本地原文用于研究核查，外部传播应链接来源并遵守原文许可。不把预印本命题号称为已经核实的后续期刊编号。

下载记录见`source_download_log.json`；字节指纹及来源见`source_manifest.json`。线上latest链接会变化，复现使用当前归档字节。
