import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class KnowledgaeBaseBuilder:
    # 在线下载模型：embedding_model_name="BAAI/bge-small-zh-v1.5"
    # 加载已下载的本地模型：embedding_model_name=r"E:\Disk\bigmodel\demo2\BAAI-bge-small-zh-v1.5"
    def __init__(self, embedding_model_name=r"E:\Disk\bigmodel\demo2\BAAI-bge-small-zh-v1.5"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！","，"]
        )

    def load_and_process_pdfs(self, pdf_folder):    
        all_docs = []
        for file in os.listdir(pdf_folder):
            if file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, file)
                print(f"Processing file: {pdf_path}")
                loader = PDFPlumberLoader(pdf_path)
                documents = loader.load()
                for doc in documents:
                    doc.page_content = doc.page_content.replace("\n", " ").strip()
                    doc.metadata["source"] = file  # 添加文件名作为来源
                all_docs.extend(documents)
        
        splitted_docs = self.text_splitter.split_documents(all_docs)
        print(f"Total chunks created: {len(splitted_docs)}")
        return splitted_docs

    def build_vector_store(self, docs, save_path="faiss_index"):
        print("Building vector store (FAISS)...")
        vector_store = FAISS.from_documents(docs, self.embeddings)
        vector_store.save_local(save_path)
        print(f"Vector store saved to {save_path}")
        return vector_store


if __name__ == "__main__":
    # 模拟运行
    kb = KnowledgaeBaseBuilder()
    docs = kb.load_and_process_pdfs("./pdfs")
    vector_store = kb.build_vector_store(docs, save_path="./pdfs/faiss_index")