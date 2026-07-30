# Ex7 分层提示：过滤与去重

先读指南 §4.2。数据处理题和模型题不同：自动测试只能检查结构，过滤与 fuzzy
dedup 的误杀必须人工审计。

```text
normalize
  → quality filter
  → exact dedup
  → optional fuzzy dedup
  → token count
  → 5-record audit
```

先在四五条手写记录上跑通，不要第一次就下载大数据。

## EX07_QUALITY_FILTER

### 函数契约

- 输入是一条已经规范化、至少含 `text` 的 record；
- 输出必须严格是 `bool`；
- 一次实验只引入一条可解释规则；
- 规则边界和误杀案例要写入 `notes.md`。

### 先设计判定表

先选择一个规则，例如最短字符数或字符/词重复率，并明确阈值。不要同时加两条。

| id | text 特征 | 你预期 keep/drop | 理由 |
|---|---|---|---|
| empty | 空文本 | ____ | ____ |
| just_below | 刚低于阈值 | ____ | ____ |
| boundary | 恰好在阈值 | ____ | ____ |
| just_above | 刚高于阈值 | ____ | ____ |
| useful_short | 短但可能有用 | ____ | ____ |
| long_spam | 长但高度重复 | ____ | ____ |

### 提示 1

先把规则写成一句自然语言：“当 ______ 时丢弃”。若这句话无法由一个简单测试
验证，规则可能一次改得太多。

### 提示 2

从 `record["text"]` 计算一个可打印的统计量，再与阈值比较。临时输出
`id / statistic / decision`，确认边界方向后删除调试打印。

### 提示 3

公开 check 只要求所有样例返回 bool，随后给 `MANUAL`。你必须自己准备至少
3 个保留和 3 个剔除案例；自动测试不会替你判断语料质量。

### 常见错误

- 返回 `None`、分数或匹配对象，而不是 bool；
- 在函数内修改 record；
- 用未经说明的多条规则一起过滤，无法归因；
- 只看被删除比例，不抽查 false positive；
- 把“长文本”直接等同于“高质量文本”。

## EX07_EXACT_DEDUP

### 函数契约

- 相等键是规范化后的 `record["text"]`；
- 首次出现者保留；
- 输出顺序与首次出现顺序一致；
- 不修改输入列表；
- 返回 `(kept_records, removed_count)`。

### 先手算

```text
输入 id/text:
A/"same", B/"other", C/"same", D/"same", E/"last"

保留 id: __________________
removed_count: ____________
```

### 提示 1：不变量

从左到右扫描时，需要知道“这个文本以前是否见过”。第一次见到时同时做两件事：
记录键、把原 record 放入结果；以后再见只增加删除计数。

### 提示 2：数据结构

一个 `set` 负责 O(1) 平均成员检查，一个 `list` 负责保持顺序。不要把所有 records
直接转成 set：dict 不可哈希，而且即使只存文本也会丢失原 record 与稳定顺序。

### 提示 3：自检不变量

完成后断言：

```text
len(kept) + removed == len(records)
输入 records 的长度和内容未改变
kept 中 text 不重复
重复运行得到同样顺序
```

### 常见错误

- 按 `id` 去重而不是按规范化文本；
- 保留最后一次出现；
- 用排序去重，改变语料顺序；
- 原地删除导致跳过相邻重复；
- removed 返回保留数或 unique 数。

## EX07_FUZZY_DEDUP（可选）

exact dedup 通过并完成人工抽查后再做。见指南 §4.2 的 MinHash/LSH 方向。

### 先定义实验，不先写算法

在 `notes.md` 写清：

1. 文本怎样切成 shingles；
2. 相似度怎样计算；
3. 阈值是多少；
4. 候选对怎样产生；
5. 两条近似文本中保留哪条；
6. 如何审计误杀。

小切片可以先用直接两两比较验证语义；规模化前再换 MinHash/LSH。

### audit 契约

每个删除对至少要导出：

```text
kept_text
removed_text
similarity
```

不要只返回被删 records；没有配对证据就无法人工判断阈值是否合理。

### 常见错误

- 先写 MinHash 代码，尚未定义“近似”的文本语义；
- 规范化不足导致标点/空白主导距离；
- 阈值太低，把同主题但不同事实的文章删掉；
- 一个 record 与多条候选重复计数；
- audit 的 similarity 不是实际做删除决策时的数值。

## 小步运行顺序

1. 先运行 `check`，让 normalization wiring 通过；
2. 只填 quality filter，记录 `MANUAL` 的边界决策；
3. 只填 exact dedup，直到 `A,B,A,A,C → A,B,C` 通过；
4. 用本地小 JSONL 跑完整 pipeline；
5. 最后才安装/下载 Ex7 真实数据所需依赖与小切片；
6. fuzzy 未做时显示 `SKIP`，不阻塞必做验收。
