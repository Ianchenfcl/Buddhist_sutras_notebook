# 🚀 佛AI - 金剛經智慧開源知識庫 / Dharma AI Platform

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Database-SQLite%20%26%20ChromaDB-orange.svg" alt="Database">
  <img src="https://img.shields.io/badge/AI%20Brain-Gemini%201.5%2520Flash-purple.svg" alt="AI Brain">
</p>

---

## 🌐 Quick Links / 語言切換
* [繁體中文版說明](#繁體中文版說明-traditional-chinese)
* [English Version](#english-version)

---

# 繁體中文版說明 (Traditional Chinese)

## 📌 項目簡介
**佛AI** 是一款專為佛學經典（目前以《金剛經》為核心）深度訂製的 **開源智慧導讀與對話平台**。

本專案支援 **多主體知識庫 (Multi-Notebook) 切換**，預設提供精心整理的佛經知識庫：
1. **📚 金剛經 32 分**：完整收錄《金剛般若波羅蜜經》的 32 個章節，經過精確的章節標題增強與語意切分，保留最純粹的佛法心要。
2. **📖 未來擴充性**：支援隨時放入不同門派與經典的文件（如《心經》、《楞嚴經》等），讓各門各派的學徒都能針對喜歡的經文進行參悟。

系統利用先進的 **RAG（檢索增強生成）** 技術，讓您能與 AI 佛學導師進行深度對話，一鍵生成智慧學習導讀（Study Guide），並針對您生活中的煩惱、執念與修行疑問，提供極具智慧的客製化解惑方案！

---

## 🛠️ 快速開始 (Quick Start)

本專案經過自動化封裝，支援 Windows 與 macOS/Linux 系統！

### 步驟 1：取得免費的 Gemini API Key
1. 前往 **[Google AI Studio](https://aistudio.google.com/)**。
2. 點擊 **"Get API key"** -> **"Create API key"** 並複製您的金鑰（格式為 `AIzaSy...`）。

### 步驟 2：啟動系統
*   **Windows 系統**：在專案根目錄下，直接雙擊執行 **`start.bat`**。
*   **macOS / Linux 系統**：打開終端機（Terminal），切換至專案根目錄，執行以下指令：
    ```bash
    chmod +x start.sh
    ./start.sh
    ```
啟動腳本會**全自動於背景檢查並安裝**環境所需的 Python 套件（採用 `.venv` 獨立虛擬環境隔離，不影響您原本的系統設定），並自動開啟瀏覽器：
👉 **`http://localhost:8000`**

貼上您的 API Key 並選擇模型，即可開始與佛法 AI 對話！

---

## 🛡️ 設計理念與著作權尊重 (技術中立)

為了保障開源社群的健康發展，並推廣佛法經典，**佛AI** 在設計上導入了以下關鍵機制：

1. **🔗 溯源橋樑（Citation Bridge）**：
   - AI 的每一句回答都會強制帶上可點擊的 `[Source X]` 引用標籤，確保每一句開示皆有經典依據。
2. **🧠 技術中立與零幻覺防禦**：
   - 採用 **SQLite（目錄管理）+ ChromaDB（向量檢索）** 雙核心引擎，確保 AI 僅根據所選知識庫內的經文真實內容進行回答，防禦 AI 瞎編與幻覺。
3. **🔒 用戶隱私防護**：
   - API Key 僅儲存於用戶本機瀏覽器 `localStorage`，不經過任何後端伺服器，確保金鑰隱私絕對安全。

---

## 🌟 核心特色 (Features)
* **🗂️ 多主體知識庫自由切換**：可於左側控制面板無縫切換不同的佛經知識庫，載入不同經典並與 AI 對話。
* **📞 撥打語音給 AI 佛學導師**：點擊右上角「撥打語音導師」按鈕，即可與 AI 進行如同打電話般的即時語音對話探討佛理。
  * 支援 **Barge-in（插話打斷）** 功能，讓您的修行對話探討更加自然逼真。
* **📝 佛理參悟測驗 (Interactive Quizzes)**：內建多達 9 種不同難度與主題的實修測驗題庫（包含「四相觀念測試」、「無住生心挑戰」等），隨時檢驗您對空性與佛法的理解。
* **🎨 質感介面 (Tailwind CSS)**：支援左中右三欄面板、原文對照、個人修行筆記等，支援全螢幕無干擾閱讀。

---

# English Version

## 📌 About 佛AI (Dharma AI)
**佛AI (Dharma AI)** is an open-source **Buddhist Sutra RAG Platform** deeply tailored for contemplating the Diamond Sutra and exploring Buddhist philosophy. 

By indexing **all 32 chapters of the Diamond Sutra**, users can chat or **make live voice calls** with a highly intelligent Buddhist AI mentor. It leverages **RAG (Retrieval-Augmented Generation)** to generate custom guidance, insight reflections, and Study Guides to elevate your understanding of emptiness, non-attachment, and wisdom!

---

## 🛠️ Quick Start

This project is packaged with automation scripts for both Windows and macOS/Linux.

### Step 1: Obtain a Free Gemini API Key
1. Go to **[Google AI Studio](https://aistudio.google.com/)**.
2. Click **"Get API key"** -> **"Create API key"** and copy your `AIzaSy...` key.

### Step 2: Launch the System
*   **Windows**: Double-click **`start.bat`** in the project root folder.
*   **macOS / Linux**: Open your Terminal, navigate to the project root directory, and run:
    ```bash
    chmod +x start.sh
    ./start.sh
    ```
The launch script will **automatically create a Python virtual environment (`.venv`), install all requirements**, and open:
👉 **`http://localhost:8000`**

Input your API Key and choose a model to start your Dharma journey!

---

## 🛡️ Strategic Design
1. **🔗 Fair Use Citation Bridge**: Every AI-synthesized claim is strictly anchored via `[Source X]` badges.
2. **🧠 Advanced RAG to Prevent Hallucination**: Built on a **SQLite + ChromaDB** dual engine, ensuring AI only answers based on verified sutras.
3. **🔒 Client-Side Local Privacy**: API keys are secured purely within browser `localStorage`. No backend server logging is involved.

---

## 🌟 Key Features
- **🗂️ Multi-Notebook Support**: Swap between different sutras or philosophical texts on the fly.
- **📞 Phone Call with AI Mentors**: Click "Voice Tutor" to voice call the AI Buddhist mentor with real-time **barge-in / interruption** support.
- **📝 Interactive Dharma Quizzes**: Test your understanding of emptiness and non-attachment with multiple quiz modes.
- **🎨 Premium UI**: Clean responsive design with source indexing, interactive RAG chat window, and a multi-functional side panel.
