# near_dup_dedup — 可续跑的近重复清洗

对「每行一个 JSON 对象、含 token 字符串数组」的 JSONL 采集结果做近重复清洗，
候选来自 datasketch 的 `MinHashLSH`（沿用其近似召回），判定前用**原始 token
集合的精确 Jaccard** 复核。所有状态落在本地 SQLite，进程异常退出后可从最后一个
已提交批次无损续跑。

## 判定语义

1. 逐条读入，对 token **集合**（自动去重、忽略顺序）建 `MinHash`。
2. `MinHashLSH.query()` 给出候选，候选只可能来自**已接受**记录：
   - 持久索引：此前已提交批次的接受记录；
   - 批内临时索引：当前批次中更早被接受的记录（批次结束即弃，不进持久索引）。
3. 对每个候选计算原始 token 集合的精确 Jaccard，`>= jaccard_threshold` 才算合格。
   因此 LSH 的假阳性候选不会造成误判，LSH 只影响召回。
4. 合格候选中**源行号最小（最早接受）**的记录作为 `duplicate_of`；没有合格候选
   则接受。空 token 集合无法在 LSH 中碰撞，一律保留。
5. 精确重复（集合相同）的 MinHash 签名完全一致、在每个 band 都碰撞，召回率为 1。

判定结果与 batch_size 无关：批内临时索引保证「单批全量」与「逐行提交」结论一致
（有测试 `test_batch_size_does_not_change_decisions` 覆盖）。

## 崩溃安全契约

每个批次的处理顺序：

```
BEGIN 事务 → 逐条判定（只查索引，不改持久 LSH）→ 写 records/evidence/progress
→ COMMIT（WAL + synchronous=FULL，fsync 后返回）→ 接受记录插入持久 LSH
```

- **不漏记**：提交是单事务，批次内任一步失败整体回滚，重开后该批重新处理。
- **不重复接受**：LSH 键为源行号且有主键约束；重开时索引完全由已提交的
  `records` 重建，插入前还有 `key in lsh` 防御检查。
- **未提交记录不进索引**：持久 LSH 只在 COMMIT 成功后更新；重开不从任何内存
  状态恢复，索引内容与已提交接受记录严格一致（`verify` 会复核键集合）。

## 续跑时的核对

重开数据库时强制校验，任一不符抛 `ParameterMismatchError` 且不处理数据：

- 参数指纹（SHA-256，覆盖全部**判定相关**参数与输入绝对路径，存
  `meta.fingerprint`）；`batch_size` 只决定事务粒度、不影响判定，不在指纹内，
  续跑时可以调整；
- 输入文件内容身份：字节大小、SHA-256、总行数；
- 实际生效的 LSH `(b, r)` 布局；
- 已提交字节偏移必须落在当前文件的行边界上。

## SQLite 表结构

| 表 | 内容 |
|---|---|
| `meta` | 参数 JSON、指纹、输入身份、LSH 布局、完成标记 |
| `batches` | 每批的起止行、接受/重复计数、提交时间 |
| `records` | 每行的原始行、token 集合、判定、`duplicate_of`、最佳 Jaccard |
| `evidence` | 该行全部 LSH 候选的来源（committed/batch）、精确 Jaccard、是否合格、排名 |
| `progress` | 最后提交行号、字节偏移、已提交行数（单行） |

从已提交状态即可导出原记录（`records.raw_line`）与重复依据（`evidence`），
无需重新读取或重新计算输入。

## 命令行用法

```bash
# 处理或续跑（参数必须与首次完全一致）
python -m near_dup_dedup run \
    --input docs.jsonl --db state.db \
    --tokens-key tokens --num-perm 128 \
    --lsh-threshold 0.9 --jaccard-threshold 0.9 \
    --batch-size 1000

python -m near_dup_dedup status --db state.db
python -m near_dup_dedup verify --db state.db --input docs.jsonl   # 重算全部判定
python -m near_dup_dedup export --db state.db --output clean.jsonl --mode accepted
python -m near_dup_dedup evidence --db state.db                    # 审计轨迹 JSONL
```

`export` 默认 `--mode all`，保留全部记录并加 `_dedup` 注解块；`accepted` 为
清洗后流，`duplicate` 为被剔除记录。

## 库用法

```python
from near_dup_dedup import DedupConfig, Deduplicator
from near_dup_dedup.verify import verify

cfg = DedupConfig(input_path="docs.jsonl", jaccard_threshold=0.9, num_perm=128, batch_size=1000)
with Deduplicator(cfg, "state.db") as dedup:  # 重开即续跑
    stats = dedup.run()
    report = verify(dedup.store, cfg)  # 可选：重算全部判定
```

只使用 `MinHashLSH` 的公开接口（`insert` / `query` / `__contains__` /
`is_empty` / `keys`），未修改 datasketch 任何代码。

## 测试

```bash
python -m pytest tests/ -q
```

28 个测试覆盖：判定正确性、批大小不变性、批内去重、SIGKILL/钩子模拟崩溃后的
无损续跑、多次崩溃一致性、参数与输入篡改拒绝、完整性校验（含人为篡改库的检出）、
精确重复召回率为 1、无误报、7200 行级别的中断续跑与 CLI 全路径。
