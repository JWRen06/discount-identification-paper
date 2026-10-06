# 外部输入与完整复现

本上传包保留实验归档结果，但不是完整离线数据包。全部暂未打包的运行输入在 [third_party_inputs.json](third_party_inputs.json) 中列出，包含相对路径、字节数、SHA256和可用的来源链接。未核实这些第三方原件的再分发授权，因此不直接放入上传包；这不是对其法律授权状态的结论。

## 外部策略代码

来源：https://github.com/dynamic-trading-RL/dynamic-trading

固定提交：718825316bc7569dab591fde95b1df8c13fa379b。

固定提交树保存在 `experiments/paper_breakthrough_20261006/external/tree.json`，源路径及精确文件指纹见输入清单。按清单从该提交取回文件并放在清单路径。运行本论文回放无需安装整个外部项目。

## EIA价格与公开持仓

EIA下载网址、时间范围、提取规则及原始表/CSV指纹保存在 `experiments/paper_breakthrough_20261006/external/eia_source_manifest.json`。CSV提取使用xlrd 2.0.2；已经取得归档CSV时，离线实验不需要xlrd。

持仓快照与产品网页来源在 `experiments/public_audit_20261006/source_manifest.json`。网站最新下载会更新，不能保证重新下载得到2026-10-06归档的相同字节。EIA归档截至2026-09-29；最新表也可能不同。精确复现必须取得对应快照并通过SHA256核对，不能将新文件替换成旧版本后声称精确复现。

## 已有本地归档的读者

以下助手从指定的原始 `experiments` 目录复制清单中的输入，复制前核对指纹：

```sh
python restore_local_inputs.py "原项目的experiments目录"
python experiments/paper_upgrade_reproduce_20261006.py
```

助手不会联网。恢复文件均已列入 `.gitignore`；网页手动上传不会应用 `.gitignore`，不要手动把这些恢复文件加进公开仓库。公开包自身可复现六项代数证书和早期字段计算；外部规则、完整持仓审计及完整市场路径重算须补齐输入。
