import os
from dotenv import load_dotenv
load_dotenv() # 讀取 .env 檔案中的環境變數

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import json
import uuid
import re
import time
import base64

from fastapi import FastAPI, HTTPException, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai
import database
import embedder

# 初始化 FastAPI app
app = FastAPI(title="Local 佛AI API", description="Generic RAG Knowledge Base AI API")

@app.on_event("startup")
def startup_event():
    # 確保資料庫與資料表在啟動時自動初始化
    database.init_db()

# 設定 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 金鑰
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


# ----------------- Data Models -----------------

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 15
    model: Optional[str] = "models/gemini-1.5-flash"
    language: Optional[str] = "zh"

class NoteCreate(BaseModel):
    title: str
    content: str

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None

class TranscriptTurn(BaseModel):
    role: str
    text: str

class SummarizeRequest(BaseModel):
    transcript: Optional[List[TranscriptTurn]] = None
    raw_text: Optional[str] = None
    language: Optional[str] = "zh"

# ----------------- Helper Functions -----------------

def clean_gemma_response(text: str, query_text: Optional[str] = None) -> str:
    if not text:
        return text
        
    lines = text.splitlines()
    cleaned_lines = []
    
    # Regular expression for matching meta keys like "Role:", "Input:", "Task:", etc.
    meta_pattern = re.compile(
        r'^\s*[\*\-\+•]?\s*(Role|Input|Task|Structure|Tone|Language|Constraints|User\s+Question|System\s+Instruction|Context|Prompt|Instructions|Negative\s+Constraints|Source\s+Usage|Citations|Sources):\s*',
        re.IGNORECASE
    )
    
    # Regular expression for planning headers, constraints, sources, etc.
    planning_header_pattern = re.compile(
        r'^\s*[\*\-\+•]?\s*(Constraint|Source|Rule|Instruction|Step|Phase|Frame\s*Theory|Push[- ]?Pull|Intro|Tone|Language|Actionable\s*Steps|Context|Response|Summary|Outline)\s*\d*(\s*\(.*\))?:\s*',
        re.IGNORECASE
    )
    
    def normalize_line(line_str: str) -> str:
        s = line_str.strip().lower()
        s = re.sub(r'^[\-\+•\s\d\.\*]+\s*', '', s)
        s = re.sub(r'[^a-z0-9\s\u4e00-\u9fff]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    normalized_query = normalize_line(query_text) if query_text else ""
    
    constraint_phrases = {
        "strictly base on context",
        "strictly based on context",
        "base on context",
        "base on context provided sources",
        "precise source citation",
        "precise source citations",
        "precise source citations source x",
        "actionable steps",
        "no thinking process",
        "no thinking process in output",
        "no ai like tone",
        "avoid generic ai tone",
        "no metainstruction echoing",
        "no metainstruction",
        "metainstruction echoing",
        "strictly english",
        "english only",
        "relationship and mindset master",
        "relationship master",
        "romantic master",
        "relationship and mindset master warm rational humorous sharp avoid generic ai tone strictly english",
        "warm rational humorous sharp",
        "please answer the following users relationship question",
        "users question",
        "users query",
        "here is the reference context data",
        "please output the final response directly",
        "user wants to know about",
        "negative constraints",
        "source usage",
        "citations",
        "traditional chinese only",
        "traditional chinese"
    }
    
    planning_keywords = [
        "source", "citation", "tone", "english only", "intro", "outro", 
        "conclusion", "body", "action plan", "constraint", "role", 
        "structure", "language", "negative constraints", "meta-instructions",
        "echoing prompts", "romantic master", "relationship master", "traditional chinese"
    ]
    
    meta_keywords_for_bullets = [
        "strictly", "context", "citation", "thinking", "chinese", "english", 
        "language", "role", "romantic", "relationship", "master", "instruction", 
        "constraint", "source", "prompt", "query", "question"
    ]
    
    in_meta_block = True
    
    for line in lines:
        stripped = line.strip().lower()
        normalized_l = normalize_line(line)
        
        if in_meta_block:
            # 1. Empty line
            if not stripped:
                continue
            
            # 2. Matches meta pattern
            if meta_pattern.match(line):
                continue
                
            # 3. Matches planning header pattern
            if planning_header_pattern.match(line):
                continue
                
            # 4. Normalized query match
            if normalized_query and normalized_l == normalized_query:
                continue
                
            # 5. Query / user wants to know patterns
            if any(phrase in stripped for phrase in [
                "user wants to know about", 
                "user is asking about", 
                "user wants to know",
                "what are 'frame theory'",
                "what are \"frame theory\""
            ]):
                continue
                
            # 6. Exact match in constraint phrases or starts with constraint pattern
            if normalized_l in constraint_phrases or any(phrase in normalized_l for phrase in [
                "strictly base on context",
                "strictly based on context",
                "precise source citations",
                "no thinking process",
                "no metainstruction",
                "no meta-instruction",
                "relationship and mindset master",
                "relationship master",
                "romantic master",
                "please answer the following user",
                "negative constraint",
                "echoing prompts",
                "avoid generic ai tone",
                "traditional chinese only"
            ]):
                continue
                
            # 7. Short lines referencing sources in meta
            if len(stripped) < 150 and re.search(r'source\s*\d', stripped):
                continue
                
            # 8. Short lines containing planning keywords
            if len(stripped) < 120 and any(kw in stripped for kw in planning_keywords):
                continue
                
            # 9. A line that is just a bullet or starts with a colon
            if stripped in ('*', '-', '+', '•') or stripped.startswith(':'):
                continue
                
            # 10. List bullets containing meta keywords
            is_bullet = re.match(r'^\s*[\*\-\+•\d\.]+\s+', line)
            if is_bullet:
                # Check if it contains any meta keywords
                if any(kw in stripped for kw in meta_keywords_for_bullets):
                    continue
                    
            # 11. Indented template structure lines
            if re.match(r'^\s+\d+\.\s*(💡|🔑|🛠️)', line):
                continue
            if re.match(r'^\s*\d+\.\s*(💡|🔑|🛠️)\s*(Core Problem Summary|Master\'s Core Mindset|Concrete Action Plan|Key Concerns|Core Insights|Actionable Steps)\s*(\(.*\))?$', line.strip(), re.IGNORECASE):
                continue
                
            # Otherwise, we have reached the actual content
            in_meta_block = False
            cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
            
    return "\n".join(cleaned_lines).strip()

def get_llm_response(prompt: str, context_sources: list, client_api_key: Optional[str] = None, model_name: Optional[str] = "models/gemini-1.5-flash", system_instruction: Optional[str] = None, query_text: Optional[str] = None) -> str:
    """呼叫 Gemini LLM 獲取回答"""
    active_key = client_api_key or GEMINI_API_KEY
    if not active_key:
        # 如果沒有 API Key，回傳一個友好的提示，並附帶檢索到的資料
        sources_summary = "\n".join([f"- [{i+1}] {s['title']} (作者: {s['author']})" for i, s in enumerate(context_sources)])
        return (
            "【系統提示：偵測到未設定 GEMINI_API_KEY，目前運行於展示模式】\n\n"
            "您好！我是您的 佛學AI導師。我已經成功在資料庫中檢索到與您問題最相關的文章！\n"
            f"以下是為您找到的 金剛經精華區經典文章：\n{sources_summary}\n\n"
            "💡 **建議**：請在左側欄「金鑰設定」中填入您的 `GEMINI_API_KEY`，我將能為您深度分析這些文章並生成專屬的智慧心法與行動方案！\n\n"
            "您可以參考下方「參考來源」頁籤，點擊直接查看這些神人文章的原文段落。"
        )
    
    last_error = None
    for attempt in range(3):
        try:
            genai.configure(api_key=active_key)
            model = genai.GenerativeModel(
                model_name or "models/gemini-1.5-flash",
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
            
            # Check if response was blocked (candidates list is empty)
            if not getattr(response, "candidates", None):
                feedback_str = ""
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    feedback_str = f" (原因: {response.prompt_feedback})"
                last_error = f"⚠️ 內容被系統安全過濾器攔截。{feedback_str}"
                time.sleep(1)
                continue
                
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                fr_str = str(finish_reason).upper()
                if "STOP" not in fr_str and "MAX_TOKENS" not in fr_str and "1" not in fr_str and "2" not in fr_str:
                    last_error = f"⚠️ AI 回覆被安全過濾器攔截 (原因: {finish_reason})。"
                    time.sleep(1)
                    continue
                    
            return clean_gemma_response(response.text, query_text=query_text)
        except Exception as e:
            err_msg = str(e)
            if "response.parts quick accessor" in err_msg or "candidates is empty" in err_msg:
                last_error = f"⚠️ 您的問題或生成內容被安全過濾器攔截。"
            else:
                last_error = err_msg
            time.sleep(1)
            
    return f"⚠️ 呼召 AI 模型 ({model_name}) 失敗（已自動重試 3 次）。\n原因：{last_error}\n請嘗試重新提問，或在左下角更換其他模型（例如 gemini-1.5-flash）。"



# ----------------- API Endpoints -----------------

@app.get("/api/status")
def read_status():
    return {"status": "online", "message": "Welcome to 佛學AI導師 API"}


@app.get("/api/notebooks")
def get_notebooks():
    """獲取所有筆記本"""
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, description, created_at FROM notebooks")
    notebooks = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return notebooks

@app.get("/api/notebooks/{notebook_id}/documents")
def get_documents(notebook_id: str):
    """獲取特定筆記本內的所有參考文件"""
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, author, publish_date, source_url, category_path, created_at FROM documents WHERE notebook_id = ?",
        (notebook_id,)
    )
    docs = []
    for row in cursor.fetchall():
        d = dict(row)
        d['category_path'] = json.loads(d['category_path']) if d['category_path'] else []
        docs.append(d)
    conn.close()
    
    # Sort by Chinese numbers in title
    def get_sort_key(doc):
        title = doc.get('title', '')
        import re
        m = re.search(r'第([一二三四五六七八九十]+)', title)
        if not m: return 999
        s = m.group(1)
        val = 0
        mapping = {'一':1, '二':2, '三':3, '四':4, '五':5, '六':6, '七':7, '八':8, '九':9, '十':10}
        if s == '十': val = 10
        elif s.startswith('十'): val = 10 + mapping.get(s[1:], 0)
        elif '十' in s: val = mapping.get(s[0], 0)*10 + mapping.get(s[2:] if len(s)>2 else '', 0)
        else: val = mapping.get(s, 0)
        return val
        
    docs.sort(key=get_sort_key)
    return docs

@app.post("/api/notebooks/{notebook_id}/query")
def query_notebook(notebook_id: str, payload: QueryRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """RAG 智慧問答介面"""
    with open(os.path.join(BASE_DIR, "debug_log.txt"), "w", encoding="utf-8") as f:
        f.write(f"x_gemini_api_key = '{x_gemini_api_key}'\n")
    query_text = payload.query
    top_k = payload.top_k
    
    embed_service = embedder.EmbeddingService(api_key=x_gemini_api_key)
    
    # 1. 計算 query embedding
    query_vector = embed_service.get_embedding(query_text, is_query=True)
    
    # 2. 從 ChromaDB 進行相似度檢索
    collection = database.get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where={"notebook_id": notebook_id}
    )
    
    context_sources = []
    
    if results and results['ids'] and len(results['ids'][0]) > 0:
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]
        documents = results['documents'][0]
        
        conn = database.get_db_connection()
        cursor = conn.cursor()
        
        for i in range(len(ids)):
            chunk_id = ids[i]
            distance = distances[i]
            meta = metadatas[i]
            chunk_content = documents[i]
            
            doc_id = meta.get('document_id')
            
            cursor.execute("SELECT title, author, publish_date, source_url, category_path FROM documents WHERE id = ?", (doc_id,))
            doc_info = cursor.fetchone()
            
            if doc_info:
                title = doc_info['title']
                author = doc_info['author']
                source_url = doc_info['source_url']
                category_path = json.loads(doc_info['category_path']) if doc_info['category_path'] else []
            else:
                title = meta.get('title', '未知標題')
                author = meta.get('author', '未知作者')
                source_url = '#'
                category_path = []
                
            context_sources.append({
                "source_index": i + 1,
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "title": title,
                "author": author,
                "source_url": source_url,
                "category_path": category_path,
                "content": chunk_content,
                "similarity": float(1.0 - distance)
            })
            
        conn.close()
    
    # 3. 組合 Prompt 與 System Instruction
    lang = payload.language or "zh"
    if lang == "zh":
        system_instruction = """你是一位專業的 AI 文件分析助手，擅長根據使用者提供的知識庫文件內容來提供精準、有深度的回答。

你的回答必須嚴格遵守以下規則：
1. **嚴格根據 Context 回答**：請充分利用提供的 [Source X] 段落進行回答。如果 Context 中沒有相關資訊，請誠實說明。
2. **精準溯源引用**：在引用某個觀點、金句或案例時，請務必在句尾加上來源標記，格式為 `[Source X]`。這非常重要，前端會將其轉化為可點擊的原文對照按鈕。
3. **結構化回答**：回答應結構分明，段落清晰。
4. **語氣風格**：客觀、專業、溫和，且一針見血。
5. **絕對禁止輸出任何思考過程或格式設定 (No Thinking Process & No Echoing Meta-instructions)**：你的輸出將直接展示給使用者，請**絕對不要**在回答中輸出任何思考步驟、推理大綱、對 Context 的摘要、草稿、或重複/列出任何系統約束、角色與問題設定的大綱。請直接開始你的最終正文回覆。
6. **一律使用繁體中文**：不論使用者的提問語言為何，請一律使用繁體中文（Taiwanese Mandarin）進行最終回答。"""

        if not context_sources:
            prompt = f"使用者的提問：'{query_text}'\n\n目前知識庫中沒有相關參考資料，請以專業的語氣，身為 AI 助手直接回答他的問題。"
        else:
            context_str = ""
            for src in context_sources:
                context_str += f"--- [Source {src['source_index']}] ---\n"
                context_str += f"標題: {src['title']}\n"
                context_str += f"作者: {src['author']}\n"
                context_str += f"內容片段: {src['content']}\n\n"
                
            prompt = f"""請回答以下使用者的佛學提問。

使用者的問題：
"{query_text}"

以下是可參考的 Context 資料：
{context_str}

請直接輸出最終回答，禁止輸出任何思考過程、大綱、重複/列出系統約束與角色大綱。請一律使用繁體中文回答。"""
    else:
        system_instruction = """You are a professional AI document analysis assistant, skilled in offering precise, insightful answers based on the provided knowledge base documents.

Your response must strictly adhere to the following rules:
1. **Strictly base on Context**: Please fully leverage the provided [Source X] sections for your answer. If the context does not contain relevant information, state so honestly.
2. **Precise Source Citation**: When referencing a viewpoint, quote, or case, you must append the source marker at the end of the sentence in the format `[Source X]`. This is extremely important, as the frontend will transform it into clickable original text buttons.
3. **Structured Response**: The response should be well-structured and organized.
4. **Tone & Style**: Objective, professional, warm, and insightful.
5. **Absolutely No Thinking Process or Meta-instruction Echoing**: Your output will be shown directly to the user. Please do not output any thinking processes, reasoning outlines, context summaries, drafts, or duplicate/list system constraints, role-play settings, or question parameters. Start directly with your final response text.
6. **Write strictly in English**: Regardless of the user's input language, please respond strictly in English."""

        if not context_sources:
            prompt = f"User's Question: '{query_text}'\n\nThere is no relevant reference material in the current knowledge base. Please reply to the question directly as an AI assistant with a professional tone."
        else:
            context_str = ""
            for src in context_sources:
                context_str += f"--- [Source {src['source_index']}] ---\n"
                context_str += f"Title: {src['title']}\n"
                context_str += f"Author: {src['author']}\n"
                context_str += f"Content Segment: {src['content']}\n\n"
                
            prompt = f"""Please answer the following user's relationship question.

User's Question:
"{query_text}"

Here is the reference Context data:
{context_str}

Please output the final response directly, and do not output any thinking processes, outlines, or duplicates/lists of system constraints and role outlines. Please reply strictly in English."""

    # 4. 呼叫 LLM
    ai_answer = get_llm_response(
        prompt, 
        context_sources, 
        client_api_key=x_gemini_api_key, 
        model_name=payload.model,
        system_instruction=system_instruction,
        query_text=query_text
    )

    
    return {
        "query": query_text,
        "answer": ai_answer,
        "sources": context_sources
    }

# ----------------- Note Management -----------------

@app.get("/api/notebooks/{notebook_id}/notes")
def get_notes(notebook_id: str):
    """獲取特定筆記本內的所有使用者筆記"""
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, title, content, created_at FROM notes WHERE notebook_id = ? ORDER BY created_at DESC",
        (notebook_id,)
    )
    notes = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return notes

@app.post("/api/notebooks/{notebook_id}/notes")
def create_note(notebook_id: str, note: NoteCreate):
    """新增隨手筆記"""
    note_id = str(uuid.uuid4())
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO notes (id, notebook_id, title, content) VALUES (?, ?, ?, ?)",
        (note_id, notebook_id, note.title, note.content)
    )
    conn.commit()
    conn.close()
    return {"id": note_id, "title": note.title, "content": note.content}

@app.put("/api/notebooks/{notebook_id}/notes/{note_id}")
def update_note(notebook_id: str, note_id: str, note: NoteUpdate):
    """更新隨手筆記"""
    conn = database.get_db_connection()
    cursor = conn.cursor()
    
    # 檢查筆記是否存在
    cursor.execute("SELECT id FROM notes WHERE id = ? AND notebook_id = ?", (note_id, notebook_id))
    if not cursor.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Note not found")
        
    updates = []
    params = []
    if note.title is not None:
        updates.append("title = ?")
        params.append(note.title)
    if note.content is not None:
        updates.append("content = ?")
        params.append(note.content)
        
    if not updates:
        conn.close()
        return {"message": "No changes made"}
        
    params.extend([note_id, notebook_id])
    query_str = f"UPDATE notes SET {', '.join(updates)} WHERE id = ? AND notebook_id = ?"
    cursor.execute(query_str, tuple(params))
    conn.commit()
    conn.close()
    return {"message": "Note updated successfully"}

@app.delete("/api/notebooks/{notebook_id}/notes/{note_id}")
def delete_note(notebook_id: str, note_id: str):
    """刪除隨手筆記"""
    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM notes WHERE id = ? AND notebook_id = ?", (note_id, notebook_id))
    conn.commit()
    conn.close()
    return {"message": "Note deleted successfully"}

@app.post("/api/notebooks/{notebook_id}/notes/summarize")
def summarize_transcript(notebook_id: str, payload: SummarizeRequest, x_gemini_api_key: Optional[str] = Header(None)):
    """將語音通話逐字稿摘要為高質感的個人佛學隨身筆記"""
    transcript = payload.transcript
    raw_text = payload.raw_text
    lang = payload.language or "zh"
    
    if not transcript and not raw_text:
        raise HTTPException(status_code=400, detail="Both transcript and raw_text are empty")
        
    if transcript:
        formatted_transcript = ""
        for turn in transcript:
            speaker = "使用者" if turn.role == "user" else "AI 佛學導師"
            formatted_transcript += f"{speaker}: {turn.text}\n"
    else:
        formatted_transcript = raw_text
        
    active_key = x_gemini_api_key or GEMINI_API_KEY
    
    if lang == "zh":
        system_instruction = """請以專業 AI 助手的視角，為以下的對話實錄或筆記內容整理出一份**「摘要版筆記」**。

你的回答必須嚴格遵守以下規則：
1. **結構分明**：生成一份結構分明、排版美觀、客觀的繁體中文筆記，包含以下部分：
   - 💡 **核心重點簡述**：精簡說明主要討論主題或內容重點。
   - 🔑 **關鍵資訊**：提煉最關鍵的 2-3 個細節或見解。
   - 🛠️ **建議或後續步驟**：條列出可能的後續行動（若有）。
2. **絕對禁止輸出任何思考過程或格式設定 (No Thinking Process & No Echoing Meta-instructions)**：請直接輸出筆記內容本身，絕對不要在回答中輸出任何思考步驟、大綱、推理步驟、或重複/列出任何系統約束、角色與任務設定的大綱。請直接以 1. 💡 **核心重點簡述** 開始你的最終回答。
3. **一律使用繁體中文**：請使用繁體中文（Taiwanese Mandarin）撰寫。"""
        
        prompt = f"""請為以下內容生成摘要版筆記：

內容：
{formatted_transcript}

請直接輸出最終回答（直接以 1. 💡 **核心重點簡述** 開始），禁止輸出任何思考大綱或系統設定。"""
    else:
        system_instruction = """Please act as a professional AI assistant and summarize the provided transcript or note into a concise and well-structured **"Summarized Note"**.

Your response must strictly adhere to the following rules:
1. **Beautiful Structure**: Generate a beautifully formatted, structured, and objective note in English containing:
   - 💡 **Key Highlights**: Summarize the primary topics or focus points.
   - 🔑 **Core Information**: Extract the 2-3 most critical details or insights.
   - 🛠️ **Suggested Next Steps**: List any concrete next steps or actions (if applicable).
2. **No Thinking Process or Meta-instruction Echoing**: Direct output the note content only. Do not include any introductory sentences, outlines, or list the system instructions/role definitions. Start directly with 1. 💡 **Key Highlights**.
3. **Language**: Write strictly in English."""

        prompt = f"""Please generate a summarized note for the following content:

Content:
{formatted_transcript}

Please output the final response directly (start with 1. 💡 **Key Highlights**), without any introductory sentences or meta-instruction outlines."""

    if not active_key:
        if lang == "zh":
            fallback_text = (
                "⚠️ 【展示模式：偵測到未設定 GEMINI_API_KEY，無法生成 AI 智慧摘要】\n\n"
                "以下是您的對話大綱速記：\n"
                f"- 內容長度: {len(formatted_transcript)} 字元。\n"
                "- 請在左側設定您的 API 金鑰以啟用 Gemini 自動分析與摘要生成功能！"
            )
        else:
            fallback_text = (
                "⚠️ [Demo Mode: GEMINI_API_KEY not configured. Cannot generate AI summary]\n\n"
                "Here is your note summary outline:\n"
                f"- Content length: {len(formatted_transcript)} characters.\n"
                "- Please set your API Key to enable Gemini automatic summarization."
            )
        return {"summary": fallback_text}

    last_error = None
    for attempt in range(3):
        try:
            genai.configure(api_key=active_key)
            model = genai.GenerativeModel(
                "models/gemma-4-26b-a4b-it",
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
            
            # Check if response was blocked (candidates list is empty)
            if not getattr(response, "candidates", None):
                feedback_str = ""
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    feedback_str = f" (原因: {response.prompt_feedback})"
                last_error = f"⚠️ 內容被系統安全過濾器攔截。{feedback_str}"
                time.sleep(1)
                continue
                
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason:
                fr_str = str(finish_reason).upper()
                if "STOP" not in fr_str and "MAX_TOKENS" not in fr_str and "1" not in fr_str and "2" not in fr_str:
                    last_error = f"⚠️ 回覆內容因安全原因被攔截 (原因: {finish_reason})。"
                    time.sleep(1)
                    continue
                    
            return {"summary": clean_gemma_response(response.text)}
        except Exception as e:
            err_msg = str(e)
            if "response.parts quick accessor" in err_msg or "candidates is empty" in err_msg:
                last_error = "⚠️ 您的筆記或生成內容被安全過濾器攔截。"
            else:
                last_error = err_msg
            time.sleep(1)
            
    return {"summary": f"⚠️ 摘要生成失敗（已自動重試 3 次）：{last_error}"}

# ----------------- Study Guide (佛AI style) -----------------

@app.get("/api/notebooks/{notebook_id}/study-guide")
def get_study_guide(notebook_id: str, language: Optional[str] = "zh", x_gemini_api_key: Optional[str] = Header(None)):
    """自動生成當前筆記本的智慧學習導讀與 FAQ (佛AI style)"""
    conn = database.get_db_connection()
    cursor = conn.cursor()
    
    # 獲取筆記本內的文章列表
    cursor.execute("SELECT title, author FROM documents WHERE notebook_id = ? LIMIT 10", (notebook_id,))
    docs = cursor.fetchall()
    conn.close()
    
    if not docs:
        if language == "zh":
            return {
                "study_guide": (
                    "# AI 智慧文件導讀指南\n\n"
                    "目前您的筆記本中還沒有任何參考文章！\n"
                    "請先運行 `embedder.py` 將爬取到的 文件寫入向量資料庫，我將能為您自動分析這些文章的觀念圖譜與核心問題解答。"
                )
            }
        else:
            return {
                "study_guide": (
                    "# AI Study Guide\n\n"
                    "Currently there are no reference articles in your notebook!\n"
                    "Please run `embedder.py` first to ingest documents into the vector database. Then I can automatically analyze the concept map and answer core questions for you."
                )
            }
        
    doc_titles = [f"- {d['title']} (作者: {d['author']})" for d in docs]
    doc_titles_str = "\n".join(doc_titles)
    
    active_key = x_gemini_api_key or GEMINI_API_KEY
    if active_key:
        if language == "zh":
            prompt = f"""你是一位擁有多年實修與諮詢經驗的佛學導師與文件分析專家。
你的任務是根據當前筆記本內已上傳的文章標題（如下所示），為使用者生成一份極具深度、結構精美、專業且富含行動指引的 **「金剛經與佛學智慧學習導讀 (Study Guide)」**。

已導入的經典文章標題：
{doc_titles_str}

這份導讀必須採用高質感的 Markdown 格式撰寫，內容包括以下幾個核心模組：
1. 📈 **核心觀念圖譜 (Core Concept Map)**：提煉出這批文件中最重要的 3 個佛學心法或理論（例如：無相布施、應無所住而生其心、凡所有相皆是虛妄），進行白話解說與實修場景分析。
2. ❓ **經典痛點問答 (Frequently Asked Questions)**：列出 3 個初學者最常遇到的佛學問題（例如：如何在日常生活中實踐金剛經？如何放下執著？），並引用文章觀點給出一針見血的解答。
3. 🎯 **新手入門實修計畫 (Step-by-Step Action Plan)**：提供一份 3 階段的具體自我提升計畫，讓使用者能立刻在生活中執行。

請確保語氣專業、同理、睿智且極具洞察力，避免空泛。
"""
        else:
            prompt = f"""You are a Buddhist philosophy mentor and document analysis expert.
Your task is to generate a highly detailed, beautifully structured, professional, and action-oriented **"Diamond Sutra and Buddhist Wisdom Study Guide"** for the user based on the titles of the uploaded articles in the current notebook (shown below).

Imported Classic Article Titles:
{doc_titles_str}

This guide must be written in high-quality Markdown format and contain the following core modules:
1. 📈 **Core Concept Map**: Extract the 3 most critical mindsets or theories from these documents (e.g., non-attachment, emptiness, letting go of ego), and provide plain-language explanations with real-world scenarios.
2. ❓ **Frequently Asked Questions (FAQ)**: List 3 questions beginners struggle with the most regarding Buddhist philosophy (e.g., how to practice non-attachment in daily life? how to let go of ego?), and provide sharp, direct answers referencing the article viewpoints.
3. 🎯 **Step-by-Step Action Plan**: Provide a concrete 3-stage self-improvement plan that the user can immediately execute in daily life.

Please ensure the tone is professional, empathetic, wise, and highly insightful, avoiding superficial advice. Write the output strictly in English.
"""
        last_error = None
        for attempt in range(3):
            try:
                genai.configure(api_key=active_key)
                model = genai.GenerativeModel("models/gemini-1.5-flash")
                response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS)
                return {"study_guide": response.text}
            except Exception as e:
                last_error = str(e)
                time.sleep(1)
        # 失敗則 fallback 到預設的高質感導讀

    # Fallback/預設的高質感導讀（展示 金剛經的核心思想）
    if language == "zh":
        default_guide = f"""# 📈 金剛經與佛學智慧導讀指南

本導讀基於您目前導入的 **{len(docs)} 篇佛學文獻**。透過系統分析，為您梳理出佛學核心心法、常見痛點 FAQ 以及可立刻執行的行動指南。

---

## 一、 核心觀念圖譜 (Core Concept Map)

在金剛經的知識庫中，修行被拆解為以下三大核心基石：

### 1. 「無相布施」 (Non-Attachment in Giving)
*   **核心定義**：行善布施時，心中不存有施者、受者及所施物的執著。
*   **實修心法**：在日常生活中幫助他人時，不求回報，不將善行掛在心上，達到真正的內心平靜。

### 2. 「應無所住而生其心」 (Mind Without Dwelling)
*   **核心定義**：心不在任何事物上產生執著或停留。
*   **實修心法**：面對順境與逆境時，保持覺知但不被情緒牽著走，隨緣盡份，放下對結果的強烈控制欲。

### 3. 「凡所有相，皆是虛妄」 (All Appearances Are Illusory)
*   **核心定義**：世間一切有形無形的事物都是因緣和合而生，沒有永恆不變的實體。
*   **實修心法**：遇到挫折或名利誘惑時，提醒自己這一切終將消逝，不需為此過度煩惱或狂喜，保持平常心。

---

## 二、 經典痛點問答 (Frequently Asked Questions)

### Q1：在現代忙碌生活中，如何實踐金剛經的智慧？
> **AI 導師解答**：
> 現代生活的壓力多來自於對未來的焦慮與對現狀的不滿。
> 1.  **活在當下**：專注於眼前的每一件事，吃飯時吃飯，工作時工作。
> 2.  **放下得失心**：盡力而為後，不執著於結果的成敗。
> 3.  **定期靜心**：每天保留 10 分鐘進行正念呼吸，將心念拉回當下。

### Q2：如何面對生活中的人際衝突與煩惱？
> **AI 導師解答**：
> 人際衝突往往源自於強烈的「我執」。
> *   **錯誤示範**：總認為自己是對的，強求別人改變。
> *   **正確示範**：理解「無我相、無人相」，試著站在對方立場思考，用慈悲心化解對立，明白一切衝突皆是因緣。

---

## 三、 新手自我提升 3 階段行動計畫

1.  **第一階段：覺察生活 (Week 1-2)**
    *   每天記錄下 3 個讓自己產生強烈情緒的時刻，不帶評判地觀察它。
    *   練習在睡前進行 5 分鐘的正念冥想。
2.  **第二階段：實踐無相 (Week 3-4)**
    *   每週進行一次不求回報的善行（如捐款、幫助同事），並且不主動向他人提起。
3.  **第三階段：放下執著 (Week 5-6)**
    *   當遇到不如意的事情時，默念「凡所有相，皆是虛妄」，練習讓心念快速平復，不陷入長期的抱怨與痛苦。

"""
    else:
        default_guide = f"""# 📈 Diamond Sutra and Buddhist Wisdom Study Guide

This guide is based on the **{len(docs)} Buddhist documents** currently imported. It summarizes the core philosophical mindsets, FAQ for common pain points, and actionable next steps.

---

## I. Core Concept Map

In the wisdom of the Diamond Sutra, practice is built upon three core pillars:

### 1. "Non-Attachment in Giving"
*   **Definition**: Practicing charity without holding onto the concepts of the giver, the receiver, or the gift itself.
*   **Actionable Wisdom**: When helping others in daily life, expect nothing in return. Do good deeds without holding onto them, achieving true inner peace.

### 2. "Mind Without Dwelling"
*   **Definition**: The mind should not generate attachment or dwell on anything.
*   **Actionable Wisdom**: When facing favorable or unfavorable circumstances, remain aware but do not be swept away by emotions. Let go of the intense desire to control outcomes.

### 3. "All Appearances Are Illusory"
*   **Definition**: All tangible and intangible things in the world arise from causes and conditions and have no eternal, unchanging substance.
*   **Actionable Wisdom**: When encountering setbacks or temptations of fame and fortune, remind yourself that all this will eventually pass. Maintain an ordinary mind without excessive worry or ecstasy.

---

## II. Frequently Asked Questions (FAQ)

### Q1: How can I practice the wisdom of the Diamond Sutra in modern busy life?
> **AI Mentor Answer**:
> The stress of modern life often comes from anxiety about the future and dissatisfaction with the present.
> 1.  **Live in the present**: Focus on the task at hand. When eating, just eat; when working, just work.
> 2.  **Let go of gain and loss**: Do your best, but do not be attached to the success or failure of the outcome.
> 3.  **Regular meditation**: Reserve 10 minutes every day for mindfulness breathing to bring your mind back to the present.

### Q2: How to face interpersonal conflicts and troubles in life?
> **AI Mentor Answer**:
> Interpersonal conflicts often stem from a strong "ego attachment".
> *   **Poor demonstration**: Always thinking you are right and forcing others to change.
> *   **Good demonstration**: Understand "no self, no person". Try to think from the other person's perspective, resolve opposition with compassion, and understand that all conflicts are born of causes and conditions.

---

## III. 3-Stage Beginner Self-Improvement Plan

1.  **Stage 1: Life Awareness (Week 1-2)**
    *   Write down 3 moments every day that cause strong emotions, and observe them without judgment.
    *   Practice 5 minutes of mindfulness meditation before bed.
2.  **Stage 2: Practice Non-Attachment (Week 3-4)**
    *   Perform a selfless good deed once a week (e.g., donating, helping a colleague) and do not initiate mentioning it to others.
3.  **Stage 3: Letting Go (Week 5-6)**
    *   When things go wrong, silently recite "All appearances are illusory", practice letting your mind calm quickly, and do not fall into long-term complaining and suffering.

"""
    return {"study_guide": default_guide}

# ----------------- Gemini Live Voice Call WebSocket -----------------

import logging
import traceback
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from google import genai as live_genai
from google.genai import types as live_types

logger = logging.getLogger("gemini-live-backend")
VOICE_MODEL = "models/gemini-3.1-flash-live-preview"
DEFAULT_VOICE = "Zephyr"

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 端點，作為瀏覽器前端與 Gemini Live API 之間的雙向橋樑。
    """
    print("DEBUG 1: websocket_endpoint starts")
    await websocket.accept()
    print("DEBUG 2: websocket accepted")
    logger.info("瀏覽器 WebSocket 已連線。")

    gemini_session = None
    client = None
    gemini_receive_task = None
    gemini_send_task = None
    
    to_gemini_queue = asyncio.Queue()

    try:
        # 1. 等待設定訊息
        print("DEBUG 3: waiting for setup message")
        setup_data = await websocket.receive_text()
        print(f"DEBUG 4: setup message received: {setup_data}")
        setup_json = json.loads(setup_data)
        
        if setup_json.get("type") != "setup":
            print("DEBUG ERROR: first message is not setup")
            await websocket.send_json({"type": "error", "message": "首條訊息必須為 setup 設定。"})
            await websocket.close()
            return
            
        print("DEBUG 5: parsing setup parameters")
        api_key = setup_json.get("api_key")
        voice_name = setup_json.get("voice_name", DEFAULT_VOICE)
        notebook_id = setup_json.get("notebook_id", "default-notebook-uuid")
        history = setup_json.get("history", [])
        language = setup_json.get("language", "zh")
        
        if not api_key:
            print("DEBUG ERROR: api_key is missing")
            await websocket.send_json({"type": "error", "message": "缺少 API Key。"})
            await websocket.close()
            return
            
        print(f"DEBUG 6: api_key len = {len(api_key)}, notebook_id={notebook_id}, voice_name={voice_name}, language={language}")
        logger.info(f"語音通話正在連線... Notebook: {notebook_id}, 語音: {voice_name}, 語言: {language}")
        await websocket.send_json({"type": "status", "status": "connecting", "message": "正在建立與 Google AI Studio 的 Live 連線..."})

        # 2. 建立 Gemini 客戶端
        print("DEBUG 7: creating Gemini client")
        client = live_genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=api_key,
        )
        print("DEBUG 8: Gemini client created")
        
        # 3. 獲取導讀指南做為 system_instruction，自適應其回答風格
        study_guide_instruction = ""
        try:
            conn = database.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM documents WHERE notebook_id = ?", (notebook_id,))
            doc_count = cursor.fetchone()[0]
            conn.close()
            if doc_count > 0:
                if language == "zh":
                    study_guide_instruction = (
                        f"\n【重要知識背景】您目前的專屬知識庫共收錄了 {doc_count} 篇經典文獻（包含完整的《金剛經》全 32 品等內容）。\n"
                        "請站在專業佛法導師的角度，直接運用您豐富的內建佛學智慧為使用者開示。不需拘泥於條列式文獻，自然且流暢地解答即可。"
                    )
                else:
                    study_guide_instruction = (
                        f"\n[Important Knowledge Background] Your current exclusive knowledge base contains {doc_count} classic documents (including the complete Diamond Sutra all 32 chapters, etc.).\n"
                        "Please respond from the perspective of a professional Dharma Master. Directly apply your extensive built-in Buddhist wisdom to answer the user naturally and fluently, without rigidly listing documents."
                    )
        except Exception as db_err:
            print(f"DEBUG DB ERROR: {db_err}")
            logger.error(f"Error querying documents for voice: {db_err}")

        if language == "zh":
            base_instruction = (
                "你是一位專業的文件分析助手，擅長根據知識庫內容來提供客觀且具建設性的語音建議。\n"
                "請務必使用繁體中文（台灣，Taiwanese Mandarin）與使用者進行語音交談，並用繁體中文回答所有問題。\n"
                "答話請保持精簡、口語、溫和且一針見血，符合日常交談習慣，不要使用長篇大論的書面語。\n"
                f"{study_guide_instruction}"
            )
        else:
            base_instruction = (
                "You are a professional AI document analysis assistant, skilled in offering objective and constructive voice advice based on the knowledge base.\n"
                "Please make sure to converse with the user in English and answer all questions in English.\n"
                "Keep your responses concise, oral, warm, slow, and to the point. Match the pattern of daily conversations and avoid long-winded written style.\n"
                f"{study_guide_instruction}"
            )
        
        if history:
            history_lines = []
            for turn in history:
                if language == "zh":
                    role_name = "使用者" if turn.get("role") == "user" else "助理"
                else:
                    role_name = "User" if turn.get("role") == "user" else "Assistant"
                txt = turn.get("text", "")
                if txt:
                    history_lines.append(f"{role_name}：{txt}")
            
            history_context = "\n".join(history_lines)
            if language == "zh":
                system_instruction = (
                    f"{base_instruction}\n\n"
                    f"【注意】以下是我們在連線中斷前進行的對話歷史紀錄，請牢記這些上下文，並在接下來的對話中無縫延續，但不要主動重複這些對話或在此時立刻發聲回應：\n"
                    f"{history_context}"
                )
            else:
                system_instruction = (
                    f"{base_instruction}\n\n"
                    f"[Note] Below is the chat history before the connection was interrupted. Please keep this context in mind and continue seamlessly in the subsequent conversation, but do not actively repeat this history or respond to it immediately at this moment:\n"
                    f"{history_context}"
                )
        else:
            system_instruction = base_instruction

        # 4. 配置 Live 連線設定
        config = live_types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=live_types.SpeechConfig(
                voice_config=live_types.VoiceConfig(
                    prebuilt_voice_config=live_types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
            context_window_compression=live_types.ContextWindowCompressionConfig(
                trigger_tokens=104857,
                sliding_window=live_types.SlidingWindow(target_tokens=52428),
            ),
            system_instruction=system_instruction,
            output_audio_transcription=live_types.AudioTranscriptionConfig()
        )

        # 5. 連線至 Gemini Live 服務
        print(f"DEBUG 9: connecting to Gemini Live with model: {VOICE_MODEL}")
        async with client.aio.live.connect(model=VOICE_MODEL, config=config) as session:
            print("DEBUG 10: Gemini Live connection successful")
            gemini_session = session
            logger.info("Gemini Live 連線成功！")
            await websocket.send_json({"type": "status", "status": "connected", "message": "連線成功！開始進行語音對話吧。"})

            in_turn = False
            async def receive_from_gemini():
                nonlocal in_turn
                try:
                    while True:
                        async for response in session.receive():
                            # 1. 偵測與發送新回合開始訊號
                            if (response.data or (response.server_content and response.server_content.model_turn)) and not in_turn:
                                in_turn = True
                                await websocket.send_json({"type": "start_turn"})
                            
                            # 2. 處理文字逐字稿
                            if response.server_content:
                                # 優先嘗試從 output_transcription 獲取語音逐字稿
                                if response.server_content.output_transcription and response.server_content.output_transcription.text:
                                    await websocket.send_json({"type": "text", "text": response.server_content.output_transcription.text})
                                
                                # 保留原先從 model_turn 獲取文字的邏輯做為 fallback
                                if response.server_content.model_turn:
                                    for part in response.server_content.model_turn.parts:
                                        if part.text:
                                            await websocket.send_json({"type": "text", "text": part.text})
                            
                            # 3. 處理音訊輸出
                            if response.data:
                                await websocket.send_bytes(response.data)
                            
                            # 4. 處理中斷與結束狀態
                            if response.server_content is not None:
                                if getattr(response.server_content, "interrupted", False):
                                    logger.info("偵測到語音被打斷，發送 interrupt 指令。")
                                    in_turn = False
                                    await websocket.send_json({"type": "interrupt"})
                                if getattr(response.server_content, "turn_complete", False):
                                    logger.info("偵測到大師發言結束，發送 turn_complete 指令。")
                                    in_turn = False
                                    await websocket.send_json({"type": "turn_complete"})
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    print(f"DEBUG ERROR in receive_from_gemini: {e}")
                    logger.error(f"從 Gemini 接收資料時出錯: {str(e)}")
                    await websocket.send_json({"type": "error", "message": f"接收 Gemini 回應失敗: {str(e)}"})

            async def send_to_gemini():
                try:
                    while True:
                        item, end_of_turn = await to_gemini_queue.get()
                        item_type = item["type"]
                        item_data = item["data"]
                        
                        if item_type == "audio":
                            await session.send_realtime_input(
                                audio=live_types.Blob(data=item_data, mime_type="audio/pcm;rate=16000")
                            )
                        elif item_type == "video":
                            await session.send_realtime_input(
                                video=live_types.Blob(data=item_data, mime_type="image/jpeg")
                            )
                        elif item_type == "text":
                            await session.send_realtime_input(
                                text=item_data
                            )
                        to_gemini_queue.task_done()
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"發送資料至 Gemini 時出錯: {str(e)}")
                    traceback.print_exc()

            async def receive_from_browser():
                received_packet_count = 0
                try:
                    while True:
                        message = await websocket.receive()
                        if message.get("type") == "websocket.disconnect":
                            break
                        
                        if "bytes" in message:
                            pcm_data = message["bytes"]
                            received_packet_count += 1
                            await to_gemini_queue.put(({"type": "audio", "data": pcm_data}, False))
                        
                        elif "text" in message:
                            try:
                                data = json.loads(message["text"])
                                msg_type = data.get("type")
                                if msg_type == "video":
                                    base64_data = data.get("data")
                                    if base64_data:
                                        img_bytes = base64.b64decode(base64_data)
                                        await to_gemini_queue.put(({"type": "video", "data": img_bytes}, False))
                                elif msg_type == "text":
                                    text_content = data.get("text")
                                    if text_content:
                                        await to_gemini_queue.put(({"type": "text", "data": text_content}, True))
                            except json.JSONDecodeError:
                                pass
                except asyncio.CancelledError:
                    pass

            browser_task = asyncio.create_task(receive_from_browser(), name="BrowserReceiver")
            gemini_receive_task = asyncio.create_task(receive_from_gemini(), name="GeminiReceiver")
            gemini_send_task = asyncio.create_task(send_to_gemini(), name="GeminiSender")

            done, pending = await asyncio.wait(
                [browser_task, gemini_receive_task, gemini_send_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
                
    except WebSocketDisconnect:
        logger.info("瀏覽器 WebSocket 連線已中斷。")
    except Exception as e:
        logger.error(f"語音 WebSocket 連線錯誤: {str(e)}")
        traceback.print_exc()
        try:
            await websocket.send_json({"type": "error", "message": f"連線錯誤: {str(e)}"})
        except:
            pass
    finally:
        if gemini_receive_task:
            gemini_receive_task.cancel()
        if gemini_send_task:
            gemini_send_task.cancel()
        try:
            await websocket.close()
        except:
            pass
        logger.info("語音 WebSocket 資源已清理。")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# 託管前端靜態資源
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
def get_index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))

