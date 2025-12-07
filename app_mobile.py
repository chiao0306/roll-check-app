import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import google.generativeai as genai
import json
import concurrent.futures # 引入平行處理套件

# --- 1. 頁面設定 ---
st.set_page_config(page_title="中鋼機械稽核", page_icon="🏭", layout="centered")

# --- 2. 秘密金鑰讀取 ---
try:
    DOC_ENDPOINT = st.secrets["DOC_ENDPOINT"]
    DOC_KEY = st.secrets["DOC_KEY"]
    GEMINI_KEY = st.secrets["GEMINI_KEY"]
except:
    st.error("找不到金鑰！請在 Streamlit Cloud 設定 Secrets。")
    st.stop()

# --- 3. 初始化 Session State ---
if 'photo_gallery' not in st.session_state:
    st.session_state.photo_gallery = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- 4. 核心函數：Azure 神之眼 (單頁處理邏輯) ---
def process_single_file(file_data, page_index, endpoint, key):
    """
    這是一個獨立函數，專門用來處理單張圖片，方便平行運算。
    """
    try:
        client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
        
        # 呼叫 Azure
        poller = client.begin_analyze_document(
            "prebuilt-layout", 
            file_data,
            content_type="application/octet-stream"
        )
        result: AnalyzeResult = poller.result()
        
        markdown_output = ""
        # 提取表格
        if result.tables:
            for idx, table in enumerate(result.tables):
                # 嘗試抓取頁碼，若抓不到就用傳入的 index
                p_num = table.bounding_regions[0].page_number if table.bounding_regions else (page_index + 1)
                
                markdown_output += f"\n### Table {idx + 1} (Page {p_num}):\n"
                rows = {}
                for cell in table.cells:
                    r, c = cell.row_index, cell.column_index
                    content = cell.content.replace("\n", " ").strip()
                    if r not in rows: rows[r] = {}
                    rows[r][c] = content
                for r in sorted(rows.keys()):
                    row_cells = []
                    if rows[r]:
                        max_col = max(rows[r].keys())
                        for c in range(max_col + 1): row_cells.append(rows[r].get(c, ""))
                        markdown_output += "| " + " | ".join(row_cells) + " |\n"
        
        # 【資料瘦身】：只取全文的前 1000 個字元 (通常表頭都在最上面)
        # 這樣可以大幅減少 Gemini 的閱讀負擔，加快速度
        header_text_snippet = result.content[:1000] if result.content else ""
        
        return {
            "page": page_index + 1,
            "table": markdown_output,
            "header_text": header_text_snippet,
            "success": True
        }
    except Exception as e:
        return {"page": page_index + 1, "error": str(e), "success": False}

# --- 5. 核心函數：Gemini 神之腦 ---
def audit_with_gemini(extracted_data_list, api_key):
    genai.configure(api_key=api_key)
    # 堅持使用 2.5 Pro 以確保精度，靠前面的平行化來補速度
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    
    combined_input = "以下是各頁資料：\n"
    for data in extracted_data_list:
        combined_input += f"\n--- Page {data['page']} ---\n"
        combined_input += f"【頁首文字片段 (檢查工令/日期)】:\n{data['header_text']}\n"
        combined_input += f"【表格數據】:\n{data['table']}\n"

    system_prompt = """
    你是一位極度嚴謹的中鋼機械品管稽核員。
    
    請執行以下 **全方位邏輯稽核**：

    ### 0. 跨頁一致性與格式檢查 (Header Consistency)：
    - **來源**：請從「頁首文字片段」中尋找資訊。
    - **目標**：
      1. **工令編號** (Job No)
      2. **預定交貨日期** (Scheduled Date)
      3. **實際交貨日期** (Actual Date)
    - **規則**：
      - 所有頁面的上述三個欄位內容必須「完全相同」。不同 -> **FAIL**。
      - 日期格式必須為 `YYY.MM.DD` (如 `114.10.30`)。格式錯誤 -> **FAIL**。

    ### 1. 製程判定邏輯 (Process Logic)：
    - **未再生/車修**：實測值 **<= (小於或等於)** 規格值。
    - **銲補 (Welding)**：實測值 **>= (大於或等於)** 規格值。
    - **再生車修**：實測值必須 **包含於 (Inclusive)** 上下限之間。

    ### 2. 數量一致性檢查 (Quantity Check)：
    - **步驟**：讀取項目名稱中的數量要求 `(10PC)` -> 清點該列實測數據個數 -> 比對。
    - **規則**：若 `實測個數 < 要求個數` -> **FAIL (數量不符)**。
    - **例外**：僅「熱處理」忽略數量。

    ### 3. 多重規格智慧歸類 (Multi-Spec Matching)：
    - 若項目有多種尺寸規格（如：一、157mm；二、127mm）。
    - 對每個實測值，自動判斷它接近哪一個規格，就套用該規格的判定標準。

    ### 4. 數學比對嚴謹度：
    - 進行 **小數點後兩位** 的精確比對。

    ### 輸出格式 (JSON Only)：
    {
      "job_no": "工令編號",
      "summary": "總結發現幾個異常",
      "issues": [
         {
           "page": 1,
           "item": "項目名稱",
           "spec_logic": "判定標準",
           "measured": "實測數據",
           "issue_type": "數值超規 / 數量不符 / 跨頁資訊不符 / 日期格式錯誤",
           "reason": "詳細說明"
         }
      ]
    }
    """
    
    try:
        response = model.generate_content(
            [system_prompt, combined_input],
            generation_config={"response_mime_type": "application/json"}
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --- 6. 手機版 UI ---
st.title("🏭 現場稽核助手")

# A. 檔案上傳區
with st.container(border=True):
    st.subheader("📂 新增頁面")
    uploaded_files = st.file_uploader(
        "點擊上傳 (手機可選直接拍照)", 
        type=['jpg', 'png', 'jpeg'], 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )
    if uploaded_files:
        for f in uploaded_files:
            st.session_state.photo_gallery.append(f)
        st.session_state.uploader_key += 1
        st.rerun()

# B. 預覽與管理區
if st.session_state.photo_gallery:
    st.divider()
    st.write(f"📊 已累積 **{len(st.session_state.photo_gallery)}** 頁文件")
    
    cols = st.columns(3)
    for idx, img in enumerate(st.session_state.photo_gallery):
        with cols[idx % 3]:
            st.image(img, caption=f"P.{idx+1}", use_container_width=True)
            if st.button("❌", key=f"del_{idx}"):
                st.session_state.photo_gallery.pop(idx)
                st.rerun()

    # C. 執行按鈕
    st.divider()
    
    if st.button("🚀 極速平行分析 (2.5 Pro)", type="primary", use_container_width=True):
        
        status = st.empty()
        progress_bar = st.progress(0)
        
        # 1. 平行處理 Azure OCR
        status.text("Azure 正在平行掃描所有頁面 (極速模式)...")
        
        extracted_data_list = []
        # 準備要傳入的參數列表
        files_to_process = []
        for i, img in enumerate(st.session_state.photo_gallery):
            # 必須讀取 bytes 才能傳入執行緒
            img.seek(0)
            files_to_process.append((img.read(), i))

        # 【關鍵加速點】：使用 ThreadPoolExecutor 同時發送請求
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 提交任務
            future_to_page = {
                executor.submit(process_single_file, f_data, idx, DOC_ENDPOINT, DOC_KEY): idx 
                for f_data, idx in files_to_process
            }
            
            results = []
            for future in concurrent.futures.as_completed(future_to_page):
                data = future.result()
                results.append(data)
                # 更新進度條
                progress_bar.progress(len(results) / (len(files_to_process) + 1))
        
        # 依照頁碼重新排序 (因為平行處理回來的順序是不固定的)
        results.sort(key=lambda x: x['page'])
        
        # 檢查是否有 OCR 錯誤
        valid_results = [r for r in results if r['success']]
        if len(valid_results) < len(results):
            st.error("部分頁面讀取失敗，請檢查網路或圖片。")
        
        # 2. Gemini 思考
        status.text("Gemini 2.5 Pro 正在進行邏輯稽核...")
        result_str = audit_with_gemini(valid_results, GEMINI_KEY)
        
        progress_bar.progress(100)
        status.text("完成！")

        # 3. 顯示結果
        try:
            result = json.loads(result_str)
            if isinstance(result, list): result = result[0] if len(result) > 0 else {}
            
            st.success(f"工令: {result.get('job_no', 'Unknown')}")
            
            issues = result.get('issues', [])
            if not issues:
                st.balloons()
                st.success("✅ 全數合格！")
            else:
                st.error(f"發現 {len(issues)} 個異常")
                for item in issues:
                    with st.container(border=True):
                        st.markdown(f"**{item.get('item')}**")
                        st.write(f"🚫 `{item.get('issue_type')}`")
                        st.caption(f"實測/內容: {item.get('measured')}")
                        st.caption(f"原因: {item.get('reason')}")
        except:
            st.error("分析錯誤")
            st.code(result_str)
            
    if st.button("🗑️ 清除所有照片"):
        st.session_state.photo_gallery = []
        st.rerun()

else:
    st.info("👆 請點擊上方按鈕開始新增照片")
