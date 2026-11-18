# 把 PDF 解析（MinerU）→ 分块 → 建索引 → 问答 串成可调度的流水线。
# 这个文件是整个 RAG 流水线的总入口：
# 1) 准备数据路径和配置
# 2) 使用 MinerU 云 API 将 PDF 解析为 Markdown
# 3) 将 Markdown 分块
# 4) 创建向量数据库（可选 BM25 索引）
# 5) 根据问题进行检索和回答
# 6) 输出结果文件
from dataclasses import dataclass, field
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import shutil
import time
import pandas as pd

from src import pdf_mineru
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.questions_processing import QuestionsProcessor
from src.model_client import get_llm_provider, get_default_model

@dataclass
class PipelineConfig:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", config_suffix: str = ""):
        # 路径配置，支持不同流程和数据目录。
        # 这里保存了从数据集根目录到各中间产物、数据库、答案文件的路径。
        self.root_path = root_path

        # 原始输入数据
        self.subset_path = root_path / subset_name
        self.questions_file_path = root_path / questions_file_name
        self.pdf_reports_dir = root_path / pdf_reports_dir_name

        # 输出结果文件
        self.answers_file_path = root_path / f"answers{config_suffix}.json"
        self.debug_data_path = root_path / "debug_data"
        self.databases_path = root_path / "databases"

        # 检索相关目录
        self.vector_db_dir = self.databases_path / "vector_dbs"
        self.documents_dir = self.databases_path / "chunked_reports"
        self.bm25_db_path = self.databases_path / "bm25_dbs"

        # MinerU 解析后的 Markdown 目录
        self.reports_markdown_dirname = "03_reports_markdown"
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname

@dataclass
class RunConfig:
    # 运行流程参数配置。
    # 这些参数决定使用哪种检索/重排/回答策略，属于“实验配置”。
    parent_document_retrieval: bool = False
    use_vector_dbs: bool = True
    use_bm25_db: bool = False
    llm_reranking: bool = False
    llm_reranking_sample_size: int = 30
    top_n_retrieval: int = 10
    parallel_requests: int = 1  # Ollama 本地模型建议串行，避免内存/显存超限
    pipeline_details: str = ""
    submission_file: bool = True
    full_context: bool = False
    api_provider: str = field(default_factory=get_llm_provider)
    answering_model: str = field(default_factory=get_default_model)
    config_suffix: str = ""

class Pipeline:
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        # 初始化主流程，加载路径和配置。
        # 通过 run_config 决定是使用向量DB、父文档检索、重排器等功能。
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        """根据配置初始化所有路径。"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        """
        检查是否存在 subset.json 且无 subset.csv；若是，则自动转换为 CSV。
        这一步是为了兼容不同数据集格式。
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

    def export_reports_to_markdown(self, file_name: str):
        """
        使用 MinerU 云 API，将指定 PDF 文件转换为 Markdown，并放到 reports_markdown_dirname 目录下。
        :param file_name: PDF 文件名（如 '【财报】中芯国际：中芯国际2024年年度报告.pdf'）
        """
        print(f"开始处理: {file_name}")
        task_id = pdf_mineru.get_task_id(file_name)
        print(f"task_id: {task_id}")
        pdf_mineru.get_result(task_id)

        # 解压后目录名与 task_id 相同
        extract_dir = f"{task_id}"
        md_path = os.path.join(extract_dir, "full.md")
        if not os.path.exists(md_path):
            print(f"未找到 markdown 文件: {md_path}")
            return
        # 目标目录
        os.makedirs(self.paths.reports_markdown_path, exist_ok=True)
        # 目标文件名为原始 file_name，扩展名改为 .md
        base_name = os.path.splitext(file_name)[0]
        target_path = os.path.join(self.paths.reports_markdown_path, f"{base_name}.md")
        shutil.move(md_path, target_path)
        print(f"已将 {md_path} 移动到 {target_path}")

    def chunk_reports(self):
        """
        将 MinerU 解析出的 Markdown 报告分块，便于后续向量化和检索。
        """
        text_splitter = TextSplitter()
        print(f"开始分割 {self.paths.reports_markdown_path} 目录下的 markdown 文件...")
        text_splitter.split_markdown_reports(
            all_md_dir=self.paths.reports_markdown_path,
            output_dir=self.paths.documents_dir,
            subset_csv=self.paths.subset_path
        )
        print(f"分割完成，结果已保存到 {self.paths.documents_dir}")

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

    def process_parsed_reports(self):
        """处理已解析的 PDF 报告，执行完整预处理流水线：
        1. 分块文本
        2. 建立向量数据库
        """
        print("开始处理报告流程...")

        print("步骤1：报告分块...")
        self.chunk_reports()

        print("步骤2：创建向量数据库...")
        self.create_vector_dbs()

        print("报告处理流程已成功完成！")

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
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")

    def answer_single_question(self, question: str, kind: str = "string"):
        """
        单条问题即时推理，返回结构化答案（dict）。
        kind: 支持 'string'、'number'、'boolean'、'names' 等
        """
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=None,  # 单问无需文件
            new_challenge_pipeline=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=1,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )
        return processor.process_single_question(question, kind=kind)


def _active_model_desc() -> str:
    """返回当前 provider 与模型名，用于写入提交文件 details。"""
    provider = get_llm_provider()
    if provider == "qwen":
        return f"cloud Qwen ({get_default_model()})"
    return f"local Ollama ({get_default_model()})"


# 基础配置：最简版 RAG 流程。
base_config = RunConfig(
    parallel_requests=1,
    pipeline_details=f"MinerU pdf parsing + vDB + Router + SO CoT; {_active_model_desc()}",
    config_suffix="_base"
)

# 父文档检索配置：先找最相关父文档，再在父文档中检索更细粒度的段落。
parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=1,
    pipeline_details=f"MinerU pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; {_active_model_desc()}",
    config_suffix="_pdr"
)

# 混合增强配置：父文档检索 + LLM 重排。
max_config = RunConfig(
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=1,
    pipeline_details=f"MinerU pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; {_active_model_desc()}",
    config_suffix="_max"
)

# 所有实验配置的统一字典入口。
configs = {"base": base_config,
           "pdr": parent_document_retrieval_config,
           "max": max_config}


# 你可以直接在本文件中运行任意方法：
# python .\src\pipeline.py
# 只需取消你想运行的方法的注释即可
if __name__ == "__main__":
    # 设置数据集根目录（中芯国际财报研报集）
    root_path = here() / "data" / "stock_data"
    print('root_path:', root_path)

    # 初始化主流程，使用推荐配置
    pipeline = Pipeline(root_path, run_config=max_config)

    # 1. 使用 MinerU 云 API 将 PDF 解析为 Markdown（需要有效的 MinerU API key，
    #    且 PDF 需先上传到对应的 OSS；本项目已附带预生成的 markdown，通常无需重新运行）
    # print('1. 将 PDF 解析为 Markdown')
    # pipeline.export_reports_to_markdown('【财报】中芯国际：中芯国际2024年年度报告.pdf')

    # 2. 将 Markdown 报告分块，输出到 databases/chunked_reports
    # print('2. 将 Markdown 报告分块，输出到 databases/chunked_reports')
    # pipeline.chunk_reports()

    # 3. 从分块报告创建向量数据库，输出到 databases/vector_dbs
    #    （使用 .env 中 EMBEDDING_PROVIDER 指定的嵌入模型，云端千问或本地 Ollama）
    # print('3. 从分块报告创建向量数据库，输出到 databases/vector_dbs')
    # pipeline.create_vector_dbs()

    # 4. 处理问题并生成答案，具体逻辑取决于 run_config
    print('4. 处理问题并生成答案')
    pipeline.process_questions()

    print('完成')
