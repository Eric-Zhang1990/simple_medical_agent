import os
# --- 【新增】设置国内镜像环境变量 ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import torch

class AdvanceRetriever:
    def __init__(self, db_path="./pdfs/faiss_index"):
        # 在线下载模型
        # self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        # 已下载模型，本地加载
        self.embeddings = HuggingFaceEmbeddings(
            model_name=r"E:\Disk\bigmodel\demo2\BAAI-bge-small-zh-v1.5",
            # 推荐加上这个参数，显式指定设备，防止报错
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}, # 如果你有显卡且装好了CUDA，改成 'cuda'
            encode_kwargs={'normalize_embeddings': True} # BGE模型推荐开启归一化
        )
        self.vector_store = FAISS.load_local(db_path, self.embeddings, allow_dangerous_deserialization=True)
        print("Loading Reranker...")
        # 在线下载模型
        # self.reranker = CrossEncoder("BAAI/bge-reranker-base", device='cuda' if torch.cuda.is_available() else 'cpu')
        # 已下载模型，本地加载
        self.reranker = CrossEncoder(r"/demo2\BAAI-bge-reranker-base",
                                     device='cuda' if torch.cuda.is_available() else 'cpu',
                                     automodel_args={"torch_dtype": torch.float16})

    def get_relevant_context(self, query, top_k=50, final_k=3):
        # Step 1: Initial retrieval using vector store
        initial_docs = self.vector_store.similarity_search(query, k=top_k)
        if not initial_docs:
            return ""
        # Step 2: Rerank using cross-encoder
        pairs = [(query, doc.page_content) for doc in initial_docs]
        scores = self.reranker.predict(pairs)
        
        # Combine documents with their scores
        doc_score_pairs = list(zip(initial_docs, scores))
        
        # Sort documents based on scores in descending order
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top_k documents
        top_docs = doc_score_pairs[:final_k]
        for doc, score in top_docs:
            print(f"Score: {score:.4f}, Content: {doc.page_content[:20]}...")
        
        context = "\n\n".join([f"[出处：{d[0].metadata.get('source', 'unknow')}] {d[0].page_content}" for d in top_docs])
        return context


if __name__ == "__main__":
    # 模拟运行
    retriever = AdvanceRetriever(db_path="./pdfs/faiss_index")
    context = retriever.get_relevant_context("什么是人工湿地？", top_k=50, final_k=3)
    print("Final Context:\n", context)