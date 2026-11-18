# 把分块文本建成向量库 / BM25 索引。
import json
import pickle
from typing import List, Optional, Union
from pathlib import Path
from tqdm import tqdm

from rank_bm25 import BM25Okapi
import faiss
import numpy as np
from tenacity import retry, wait_fixed, stop_after_attempt
from src.model_client import get_embeddings


# BM25Ingestor：BM25索引构建与保存工具
class BM25Ingestor:
    def __init__(self):
        pass

    def create_bm25_index(self, chunks: List[str]) -> BM25Okapi:
        """从文本块列表创建BM25索引"""
        tokenized_chunks = [chunk.split() for chunk in chunks]
        return BM25Okapi(tokenized_chunks)
    
    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        """
        批量处理所有报告，生成并保存BM25索引。
        参数：
            all_reports_dir (Path): 存放JSON报告的目录
            output_dir (Path): 保存BM25索引的目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_report_paths = list(all_reports_dir.glob("*.json"))

        for report_path in tqdm(all_report_paths, desc="Processing reports for BM25"):
            # 加载报告
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # 提取文本块并创建BM25索引
            text_chunks = [chunk['text'] for chunk in report_data['content']['chunks']]
            bm25_index = self.create_bm25_index(text_chunks)
            
            # 保存BM25索引，文件名用 sha1
            sha1 = report_data["metainfo"]["sha1"]
            output_file = output_dir / f"{sha1}.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(bm25_index, f)
                
        print(f"Processed {len(all_report_paths)} reports")

# VectorDBIngestor：向量库构建与保存工具
class VectorDBIngestor:
    @retry(wait=wait_fixed(20), stop=stop_after_attempt(2))
    def _get_embeddings(
        self, text: Union[str, List[str]], model: Optional[str] = None
    ) -> List[List[float]]:
        # 获取文本或文本块的嵌入向量，分批降低本地模型的内存占用
        if isinstance(text, str) and not text.strip():
            raise ValueError("Input text cannot be an empty string.")
        
        # 保证 input 为一维字符串列表或单个字符串
        if isinstance(text, list):
            text_chunks = text
        else:
            text_chunks = [text]

        # 类型检查，确保每一项都是字符串
        if not all(isinstance(x, str) for x in text_chunks):
            raise ValueError("所有待嵌入文本必须为字符串类型！实际类型: {}".format([type(x) for x in text_chunks]))

        # 过滤空字符串
        text_chunks = [x for x in text_chunks if x.strip()]
        if not text_chunks:
            raise ValueError("所有待嵌入文本均为空字符串！")
        embeddings = []
        max_batch_size = 25
        for i in range(0, len(text_chunks), max_batch_size):
            embeddings.extend(
                get_embeddings(text_chunks[i:i + max_batch_size], model=model)
            )
        return embeddings

    def _create_vector_db(self, embeddings: List[float]):
        # 用faiss构建向量库，采用内积（余弦距离）
        embeddings_array = np.array(embeddings, dtype=np.float32)
        dimension = len(embeddings[0])
        index = faiss.IndexFlatIP(dimension)  # Cosine distance
        index.add(embeddings_array)
        return index
    
    def _process_report(self, report: dict):
        # 针对单份报告，提取文本块并生成向量库
        text_chunks = [chunk['text'] for chunk in report['content']['chunks']]
        # 过滤空内容，超长内容截断到 2048 字符（MinerU 按行分块可能产生超长块，
        # 超出嵌入模型的 token 上下文，需截断后再嵌入）
        max_len = 2048
        text_chunks = [t[:max_len] for t in text_chunks if len(t) > 0]
        embeddings = self._get_embeddings(text_chunks)
        index = self._create_vector_db(embeddings)
        return index

    def process_reports(self, all_reports_dir: Path, output_dir: Path):
        # 批量处理所有报告，生成并保存faiss向量库
        all_report_paths = list(all_reports_dir.glob("*.json"))
        output_dir.mkdir(parents=True, exist_ok=True)

        for report_path in tqdm(all_report_paths, desc="Processing reports"):
            with open(report_path, 'r', encoding='utf-8') as file:
                report_data = json.load(file)
            index = self._process_report(report_data)
            sha1 = report_data["metainfo"]["sha1"]
            faiss_file_path = output_dir / f"{sha1}.faiss"
            faiss.write_index(index, str(faiss_file_path))

        print(f"Processed {len(all_report_paths)} reports")