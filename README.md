# RAG Challenge 冠军方案

- 使用 Docling 进行自定义 PDF 解析
- 使用父文档检索的向量搜索
- 使用 LLM 重排提高上下文相关性
- 使用带有思维链推理的结构化输出提示
- 用于多公司比较的查询路由

## 快速开始

python -m venv .venv
.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e . -r requirements.txt

在 Windows 上，后续运行项目时也请使用 `.\.venv\Scripts\python.exe`，确保使用同一个虚拟环境。

安装并启动 [Ollama](https://ollama.com/)，然后下载所需模型：
ollama pull qwen2.5:7b
ollama pull bge-m3

配置项目根目录 `.env` 文件的模型配置

运行前确保 Ollama 服务已启动，且上述两个模型已下载。

## 测试数据集

该仓库包含两个数据集：

1. 一个小型测试集（位于 `data/test_set/`），包含 5 份年报和问题
2. 完整的 ERC2 竞赛数据集（位于 `data/erc2_set/`），包含所有竞赛问题和报告

每个数据集目录都包含自己的 README，说明具体设置步骤和可用文件。你可以使用任一数据集来：

- 学习示例问题、报告和系统输出
- 使用提供的 PDF 从头运行流水线
- 使用预处理数据直接跳到特定的流水线阶段

请参阅相应的 README 文件以获取详细的数据集内容和设置说明：

- `data/test_set/README.md` - 用于小型测试数据集
- `data/erc2_set/README.md` - 用于完整竞赛数据集

## 使用方法

可以通过注释 `src/pipeline.py` 的方法来运行流水线的任意部分，并执行：
.\.venv\Scripts\python.exe .\src\pipeline.py

也可以使用 `main.py` 运行任意流水线阶段，但需要在包含数据的目录中执行：
cd .\data\test_set\
..\..\.venv\Scripts\python.exe ..\..\main.py process-questions --config max_nst_o3m

### CLI 命令

获取可用命令的帮助：
.\.venv\Scripts\python.exe main.py --help

可用命令：

- `download-models` - 下载所需的 Docling 模型
- `parse-pdfs` - 使用并行处理选项解析 PDF 报告
- `serialize-tables` - 处理已解析报告中的表格
- `process-reports` - 对已解析报告运行完整流水线
- `process-questions` - 使用指定配置处理问题

每个命令都有自己的选项。例如：
.\.venv\Scripts\python.exe main.py parse-pdfs --help

##   显示类似 --parallel/--sequential, --chunk-size, --max-workers 的选项

.\.venv\Scripts\python.exe main.py process-reports --config ser_tab

## 使用序列化表格配置处理报告

## 部分配置

- 各问答配置均使用 `.env` 中配置的 Ollama 模型；保留配置名以兼容原有命令

切换到 `bge-m3` 后，旧数据集提供的 FAISS 索引不能直接复用。确保已解压 `debug_data` 中的解析报告，然后在数据集目录重新生成向量库：
cd .\data\test_set
python ..\..\main.py process-reports --config no_ser_tab
python ..\..\main.py process-questions --config max_nst_o3m

完整 ERC2 数据集请将 `test_set` 替换为 `erc2_set`。重建索引会调用本地 `bge-m3`，首次运行前请先确认 Ollama 服务已启动。

查看 `pipeline.py` 以获取更多配置及其详细信息。

## 项目详细介绍

main.py文件：PDF 原始文件 → Docling 解析 → 表格 LLM 序列化（可选）→ 文本规整合并 → Markdown 导出（人工核对）→ 文本分块 → 构建向量库 / BM25 词库 → 问题检索 + LLM 问答 → 输出 answers.json 答案文件

- `PipelineConfig`：**路径配置**（所有输入输出文件夹路径，跟着 `use_serialized_tables` 自动加后缀 `_ser_tab`）
- `RunConfig`：**算法策略配置**（要不要序列化表格、要不要父文档检索、要不要 LLM 重排、并发数、模型、是否全文上下文等）
✅ 这是这个 RAG 比较关键的增强点：
docling 只能识别出表格单元格，但是原始表格 json 机器检索效果差；这个步骤调用 LLM，把表格转成自然语言描述（序列化），写入 json 新增字段 `serialized_table`。
后续分块的时候，可以把序列化后的表格文本一起放进 chunk，**表格类问题召回准确率明显提升**。
注意：这个步骤本身就要调用 LLM，会消耗 token，而且是可选步骤。
