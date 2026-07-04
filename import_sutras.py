import os
import glob
import uuid
import json
import docx
from langchain_text_splitters import RecursiveCharacterTextSplitter
import database
from embedder import EmbeddingService

def read_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    return '\n\n'.join(full_text)

def process_and_embed_sutras():
    # 初始化與清除資料庫 (清除 default notebook 舊資料)
    database.init_db()
    database.clear_db("default-notebook-uuid")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sutras_dir = os.path.join(BASE_DIR, "佛經", "金剛經")
    docx_files = glob.glob(os.path.join(sutras_dir, "*.docx"))
    
    print(f"Found {len(docx_files)} .docx files in {sutras_dir}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=120,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    
    embed_service = EmbeddingService()
    chroma_collection = database.get_chroma_collection()
    
    default_notebook_id = "default-notebook-uuid"
    BATCH_SIZE = 15
    all_chunks_to_embed = []
    
    print("Chunking sutras...")
    for idx, file_path in enumerate(docx_files):
        filename = os.path.basename(file_path)
        # title e.g., 《金剛經》第一章：法會應有分.docx -> 《金剛經》第一章：法會應有分
        title = os.path.splitext(filename)[0]
        
        content = read_docx(file_path)
        if not content.strip():
            print(f"Warning: {filename} is empty or unreadable.")
            continue
            
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"sutra://jingangjing/{filename}"))
        
        # 1. 儲存原始文件到 SQLite
        database.save_document(
            doc_id=doc_id,
            notebook_id=default_notebook_id,
            title=title,
            author="佛陀 / 鳩摩羅什譯",
            publish_date="姚秦",
            source_url=f"local://佛經/金剛經/{filename}",
            category_path=["佛經", "金剛經"],
            raw_content=content
        )
        
        # 2. 切分文本
        chunks = text_splitter.split_text(content)
        
        for c_idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{c_idx}"
            
            enriched_text = f"章節標題：{title}\n\n{chunk_text}"
            
            # 儲存 chunk 到 SQLite
            database.save_chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=enriched_text,
                chunk_index=c_idx
            )
            
            # 加入待 embedding 列表
            all_chunks_to_embed.append({
                "id": chunk_id,
                "document_id": doc_id,
                "text": enriched_text,
                "metadata": {
                    "document_id": doc_id,
                    "notebook_id": default_notebook_id,
                    "title": title,
                    "author": "佛陀 / 鳩摩羅什譯",
                    "category_path": json.dumps(["佛經", "金剛經"]),
                    "chunk_index": c_idx
                }
            })
            
    print(f"Total chunks generated: {len(all_chunks_to_embed)}")
    print("Computing embeddings and writing to ChromaDB in batches...")
    
    # 批次進行 Embedding 與寫入 ChromaDB
    total_chunks = len(all_chunks_to_embed)
    if total_chunks == 0:
        print("No chunks to process. Exiting.")
        return

    for i in range(0, total_chunks, BATCH_SIZE):
        batch = all_chunks_to_embed[i:i+BATCH_SIZE]
        batch_texts = [item['text'] for item in batch]
        batch_ids = [item['id'] for item in batch]
        batch_metas = [item['metadata'] for item in batch]
        
        print(f"Processing chunk batch {i+1} to {min(i+BATCH_SIZE, total_chunks)} of {total_chunks}...")
        
        # 呼叫 Embedding API
        embeddings = embed_service.get_embeddings_batch(batch_texts)
        
        # 寫入 ChromaDB
        chroma_collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=batch_metas,
            documents=batch_texts
        )
        
    print(f"Successfully embedded and stored {total_chunks} chunks into ChromaDB and SQLite!")

if __name__ == "__main__":
    process_and_embed_sutras()
