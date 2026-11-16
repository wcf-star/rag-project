# 把 PDF 解析 → 表格序列化 → 规整 → 分块 → 建索引 → 问答 串成可调度的流水线。
# Qwen-Turbo API的基础限流设置为每分钟不超过500次API调用（QPM）。同时，Token消耗限流为每分钟不超过500,000 Tokens
# 这个文件是整个 RAG 流水线的总入口：
# 1) 准备数据路径和配置
# 2) 解析 PDF 报告
# 3) 处理/清洗/分块文本
# 4) 创建向量数据库和 BM25 索引
# 5) 根据问题进行检索和回答
# 6) 输出结果文件和提交文件
from dataclasses import dataclass, field
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import torch  # noqa: F401
import pandas as pd

from src.pdf_parsing import PDFParser
from src.parsed_reports_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.questions_processing import QuestionsProcessor
from src.tables_serialization import TableSerializer
from src.ollama_client import get_llm_provider, get_ollama_model

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        # 路径配置，支持不同流程和数据目录。
        # 这里保存了从数据集根目录到各中间产物、数据库、答案文件的路径。
        self.root_path = root_path
        suffix = "_ser_tab" if serialized else ""

        # 原始输入数据
        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name

        # 输出结果文件
        self.answers_file_path = root_path / f"answers{config_suffix}.json"
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / f"databases{suffix}"

        # 检索相关目录
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        # PDF 解析后的中间目录（按步骤编号）
        self.parsed_reports_dirname = "01_parsed_reports"
        self.parsed_reports_debug_dirname = "01_parsed_reports_debug"
        self.merged_reports_dirname = f"02_merged_reports{suffix}"
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}"

        # 这里最终对应真实磁盘路径
        self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname

@dataclass
class RunConfig:
    # 运行流程参数配置。
    # 这些参数决定使用哪种检索/重排/回答策略，属于“实验配置”。
    use_serialized_tables: bool = False
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 1
    team_email: str = "79250515615@yandex.com"
    submission_name: str = "Ilia_Ris vDB + SO CoT"
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    api_provider: str = field(default_factory=get_llm_provider)
    answering_model: str = field(default_factory=get_ollama_model)
    config_suffix: str = ""

class Pipeline:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        # 初始化主流程，加载路径和配置。
        # 通过 run_config 决定是使用向量DB、父文档检索、重排器等功能。
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        """根据配置初始化所有路径。

        这里的对象会在 Pipeline 启动时创建，用于统一管理数据集目录、调试目录、数据库目录和答案输出路径。
        """
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        """
        检查是否存在 subset.json 且无 subset.csv；若是，则自动转换为 CSV。

        这一步是为了兼容不同数据集格式，避免后续读取时因为文件后缀不同导致问题。
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"

        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                df = pd.DataFrame(data)
                df.to_csv(csv_path, index=False)

            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

    @staticmethod
    def download_docling_models():
        # 下载 Docling 所需模型，避免首次运行时出现对应模型未下载的问题。
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_sequential(self):
        # 顺序解析 PDF 报告，输出结构化 JSON。
        # 适合调试或小规模数据集，简单可靠。
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        pdf_parser.parse_and_export(doc_dir=self.paths.pdf_reports_dir)
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """多进程并行解析 PDF 报告，提升处理效率。

        参数：
            chunk_size: 每个 worker 处理的 PDF 数量
            max_workers: 并发 worker 数量
        """
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        # 先拿到所有 pdf 文件路径，再按照 chunk_size 分组并行处理。
        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))

        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def serialize_tables(self, max_workers: int = 1):
        """并行处理所有报告中的表格，使用 LLM 对表格进行结构化序列化。"""
        # 目标是把 PDF 中原始表格提升成更适合检索和回答的结构化文本。
        serializer = TableSerializer()
        serializer.process_directory_parallel(
            self.paths.parsed_reports_path,
            max_workers=max_workers
        )

    def merge_reports(self):
        """将复杂 JSON 报告规整为每页结构化文本，便于后续分块和人工审查。"""
        # PageTextPreparation 会把每个页面的文本、表格等整理成更标准的中间格式。
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        _ = ptp.process_reports(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.merged_reports_path
        )
        print(f"Reports saved to {self.paths.merged_reports_path}")

    def export_reports_to_markdown(self):
        """导出规整后报告为 Markdown，便于人工复核。"""
        # 这一步输出的是更易读的文本版本，适合检查是否有提取错误或缺失。
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        ptp.export_to_markdown(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.reports_markdown_path
        )
        print(f"Reports saved to {self.paths.reports_markdown_path}")

    def chunk_reports(self, include_serialized_tables: bool = False):
        """将规整后报告分块，便于后续向量化和检索。"""
        # 分块后，每一小段文本都能作为独立的候选检索片段，提高召回和速度。
        text_splitter = TextSplitter()

        serialized_tables_dir = None
        if include_serialized_tables:
            # 可选：将表格序列化结果也加入 chunk 语义中，增强表格类问题召回。
            serialized_tables_dir = self.paths.parsed_reports_path

        text_splitter.split_all_reports(
            self.paths.merged_reports_path,
            self.paths.documents_dir,
            serialized_tables_dir
        )
        print(f"Chunked reports saved to {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """从分块报告创建向量数据库。"""
        # 向量数据库主要用于语义搜索：相似度比较，找出与问题最相关的文本片段。
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir

        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")

    def create_bm25_db(self):
        """从分块报告创建 BM25 数据库。"""
        # BM25 是传统词频检索算法，适合关键词匹配场景，常和向量检索一起使用。
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path

        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")

    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        # 统一入口，控制并行/顺序解析模式
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers)
        else:
            self.parse_pdf_reports_sequential()

    def process_parsed_reports(self):
        """处理已解析的 PDF 报告，执行完整预处理流水线：
        1. 合并为更简洁的结构化格式
        2. 导出为 Markdown
        3. 分块文本
        4. 建立向量数据库
        """
        print("Starting reports processing pipeline...")

        print("Step 1: Merging reports...")
        self.merge_reports()

        print("Step 2: Exporting reports to markdown...")
        self.export_reports_to_markdown()

        print("Step 3: Chunking reports...")
        self.chunk_reports()

        print("Step 4: Creating vector databases...")
        self.create_vector_dbs()

        print("Reports processing pipeline completed successfully!")

    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        如果目标文件已存在，则自动生成下一个编号的文件名，避免覆盖历史结果。
        例如：answers.json -> answers_01.json
        """
        if not base_path.exists():
            return base_path

        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent

        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename

            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        # 实际问答阶段：从数据库中检索相关文档，再交给 LLM 生成答案。
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )

        output_path = self._get_next_available_filename(self.paths.answers_file_path)

        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            team_email=self.run_config.team_email,
            submission_name=self.run_config.submission_name,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")


# 预处理阶段的配置：主要影响 PDF 处理前的表格兼容方式。
# 这些配置是 "pipeline.process_parsed_reports()" 的基础输入参数，
# 不直接参与 LLM 回答，但会影响后续索引内容和检索质量。
preprocess_configs = {"ser_tab": RunConfig(use_serialized_tables=True),
                      "no_ser_tab": RunConfig(use_serialized_tables=False)}

# 基础配置：最简版 RAG 流程。
# 一般用于快速跑通，验证数据、索引和本文检索逻辑是否正常。
base_config = RunConfig(
    parallel_requests=1,
    submission_name="Ilia Ris v.0",
    pipeline_details="Custom pdf parsing + vDB + Router + SO CoT; local Ollama model",
    config_suffix="_base"
)

# 父文档检索配置：先找最相关父文档，再在父文档中检索更细粒度的段落。
# 这通常能提高长文档问答的召回质量。
parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=1,
    submission_name="Ilia Ris v.1",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; local Ollama model",
    config_suffix="_pdr"
)

# 混合增强配置：表格序列化 + 父文档检索 + LLM 重排。
max_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=1,
    submission_name="Ilia Ris v.2",
    pipeline_details="Custom pdf parsing + table serialization + vDB + Router + Parent Document Retrieval + reranking + SO CoT; local Ollama model",
    config_suffix="_max"
)

# 同上，但不使用表格序列化版本，用来对比表格结构化是否提升性能。
max_no_ser_tab_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=1,
    submission_name="Ilia Ris v.3",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; local Ollama model",
    config_suffix="_max_no_ser_tab"
)

# 这是项目中经过实验验证效果较好的配置。 
# 代码中注释说明它返回最佳结果，说明它是比较成熟的基线配置之一。
max_nst_o3m_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=1,
    submission_name="Ilia Ris v.4",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; local Ollama model",
    config_suffix="_max_nst_o3m"
)

# 表格序列化 + 父文档检索 + 重排，但使用另一套模型/版本。
max_st_o3m_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=1,
    submission_name="Ilia Ris v.5",
    pipeline_details="Custom pdf parsing + tables serialization + Router + vDB + Parent Document Retrieval + reranking + SO CoT; local Ollama model",
    config_suffix="_max_st_o3m"
)

# IBM Llama 70B 相关配置；注释说明这类 API 仅在比赛当年可用，后续不再可用。
ibm_llama70b_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=1,
    submission_name="Ilia Ris v.6",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT + SO reparser; local Ollama model",
    config_suffix="_ibm_llama70b"
)

# IBM Llama 8B 配置，同上为历史模型配置。
ibm_llama8b_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=1,
    submission_name="Ilia Ris v.7",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT + SO reparser; local Ollama model",
    config_suffix="_ibm_llama8b"
)

# Gemini 全文上下文配置：将整份文档直接纳入模型上下文，减少召回依赖。
# 这种方式对长文档很有帮助，但会增加 token 成本。
gemini_thinking_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=1,
    full_context=True,
    submission_name="Ilia Ris v.8",
    pipeline_details="Custom pdf parsing + Full Context + Router + SO CoT + SO reparser; local Ollama model",
    config_suffix="_gemini_thinking_fc"
)

# 同样的全文上下文思路，用于不同的 Gemini 版本/模型.
gemini_flash_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=1,
    full_context=True,
    submission_name="Ilia Ris v.9",
    pipeline_details="Custom pdf parsing + Full Context + Router + SO CoT + SO reparser; local Ollama model",
    config_suffix="_gemini_flash_fc"
)

# 更大上下文下的重排/检索配置：增加 top_n 和 rerank 样本量，提升召回覆盖面。
max_nst_o3m_config_big_context = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=1,
    llm_reranking_sample_size=36,
    top_n_retrieval=14,
    submission_name="Ilia Ris v.10",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; local Ollama model; top_n = 14; topn for rerank = 36",
    config_suffix="_max_nst_o3m_bc"
)

# 同样目标的另一套大上下文实验配置。
ibm_llama70b_config_big_context = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=1,
    llm_reranking_sample_size=36,
    top_n_retrieval=14,
    submission_name="Ilia Ris v.11",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; local Ollama model; top_n = 14; topn for rerank = 36",
    config_suffix="_ibm_llama70b_bc"
)

# Gemini 的大上下文版本，扩大检索候选数量。
gemini_thinking_config_big_context = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    parallel_requests=1,
    top_n_retrieval=30,
    submission_name="Ilia Ris v.12",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; local Ollama model; top_n = 30;",
    config_suffix="_gemini_thinking_bc"
)

# 所有实验配置的统一字典入口。
# 通过 configs["base"]、configs["max"] 等方式灵活切换不同的流水线设置。
configs = {"base": base_config,
           "pdr": parent_document_retrieval_config,
           "max": max_config,
           "max_no_ser_tab": max_no_ser_tab_config,
           "max_nst_o3m": max_nst_o3m_config,  # 该配置经实验验证效果较好
           "max_st_o3m": max_st_o3m_config,
           "ibm_llama70b": ibm_llama70b_config,  # 此配置仅在比赛期间可用
           "ibm_llama8b": ibm_llama8b_config,  # 同上，历史接口已不可用
           "gemini_thinking": gemini_thinking_config}


# 你可以直接在本文件中运行任意方法：
# python .\src\pipeline.py
# 只需取消你想运行的方法的注释即可
# 你也可以修改 run_config 以尝试不同的配置
if __name__ == "__main__":
    # 设置数据集根目录（此处以 test_set 为例）
    root_path = here() / "data" / "test_set"
    # root_path = here("d:") / "test_set-rag"
    print('root_path:', root_path)
    # print(type(root_path))

    # 初始化主流程，使用推荐的最佳配置
    pipeline = Pipeline(root_path, run_config=max_nst_o3m_config)

    # 以下方法可按需取消注释，逐步运行各流程：
    # 1. 解析 PDF 报告为结构化 JSON，输出到 debug/data_01_parsed_reports
    #    同时保存 docling 原始输出到 debug/data_01_parsed_reports_debug（含大量元数据，后续流程不使用）
    print('1. 解析PDF报告为结构化JSON，输出到 debug/data_01_parsed_reports')
    pipeline.parse_pdf_reports_sequential()

    # 2. 仅在需要表格序列化配置时调用。
    #    这一步会在 debug/data_01_parsed_reports 的每个表格中新增 "serialized_table" 字段。
    print('2. 序列化表格，输出到 debug/data_01_parsed_reports')
    pipeline.serialize_tables(max_workers=5)

    # 3. 将解析后的 JSON 规整为更简单的每页 markdown 结构，输出到 debug/data_02_merged_reports
    print('3. 将解析后的JSON规整为更简单的每页markdown结构，输出到 debug/data_02_merged_reports')
    pipeline.merge_reports()

    # 4. 导出规整后报告为纯 markdown 文本，仅用于人工复核或全文检索（如 gemini_thinking_config）
    #    新文件在 debug/data_03_reports_markdown
    print('4. 导出规整后报告为纯markdown文本，仅用于人工复核或全文检索（如 gemini_thinking_config）')
    pipeline.export_reports_to_markdown()

    # 5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports
    print('5. 将规整后报告分块，便于后续向量化，输出到 databases/chunked_reports')
    pipeline.chunk_reports()

    # 6. 从分块报告创建向量数据库，输出到 databases/vector_dbs
    print('6. 从分块报告创建向量数据库，输出到 databases/vector_dbs')
    pipeline.create_vector_dbs()

    # 7. 处理问题并生成答案，具体逻辑取决于 run_config
    print('7. 处理问题并生成答案，具体逻辑取决于 run_config')
    pipeline.process_questions()

    print('完成')
