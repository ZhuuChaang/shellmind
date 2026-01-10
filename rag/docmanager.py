import os
import json
import hashlib
from typing import List, Dict
import PyPDF2
from loguru import logger

class DocumentManager:
    """
    只处理 PDF 文档的 DocumentManager
    功能：
      - 加载 PDF
      - 文本提取
      - chunk 划分
      - 保存 chunk + 元数据
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        :param chunk_size: 每个文本块最大字符数
        :param chunk_overlap: 块之间重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks: List[str] = []
        self.metadata: List[Dict] = []  # 每个 chunk 对应元数据

    @staticmethod
    def _extract_text_from_pdf(file_path: str) -> List[str]:
        """提取 PDF 每页文本"""
        contents = []
        with open(file_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page_idx, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    contents.append(text.strip())
        return contents

    def _split_text(self, text: str) -> List[str]:
        """
        按 chunk_size 和 chunk_overlap 切分文本
        """
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    @staticmethod
    def _hash_file(file_path: str) -> str:
        """简单 md5 前 1MB 文件内容作为文件 id"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            chunk = f.read(1024 * 1024)
            hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def load_pdfs(self, pdf_files: List[str]):
        """加载 PDF 文件并生成 chunks + metadata"""
        for file_path in pdf_files:
            if not os.path.isfile(file_path):
                logger.warning(f"文件不存在: {file_path}")
                continue
            file_id = self._hash_file(file_path)
            pages = self._extract_text_from_pdf(file_path)
            for page_idx, page_text in enumerate(pages):
                page_chunks = self._split_text(page_text)
                for chunk_idx, chunk in enumerate(page_chunks):
                    self.chunks.append(chunk)
                    self.metadata.append({
                        'file_id': file_id,
                        'file_path': file_path,
                        'page': page_idx + 1,
                        'chunk_idx': chunk_idx
                    })
        logger.info(f"总共生成 {len(self.chunks)} chunks")

    def save_metadata(self, save_dir: str):
        """保存 chunks 和 metadata"""
        os.makedirs(save_dir, exist_ok=True)
        # 保存 chunks
        with open(os.path.join(save_dir, 'chunks.json'), 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        # 保存 metadata
        with open(os.path.join(save_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info(f"chunks 和 metadata 已保存到 {save_dir}")

    def load_metadata(self, save_dir: str):
        """加载已有 chunks 和 metadata"""
        with open(os.path.join(save_dir, 'chunks.json'), 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        with open(os.path.join(save_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        logger.info(f"已加载 {len(self.chunks)} chunks 和 metadata")

    def get_chunks(self) -> List[str]:
        return self.chunks

    def get_metadata(self) -> List[Dict]:
        return self.metadata
