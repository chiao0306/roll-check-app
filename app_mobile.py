import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import google.generativeai as genai
import json

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

# --- 3. 初始化 Session State (存照片用) ---
if 'photo_gallery' not in st.session_state:
    st.session_state.photo_gallery = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- 4. 核心函數 (Azure OCR) ---
def extract_layout_with_azure(file_obj, endpoint, key):
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    file_content = file_obj.getvalue()
    poller = client.begin_analyze_document("prebuilt-layout", file_content, content_type="application/octet-stream")
    result: AnalyzeResult = poller.result()
    
    markdown_output = ""
    if result.tables:
        for idx, table in enumerate(result.tables):
            page_num = "Unknown"
            if table.bounding_regions: page_num = table.bounding_regions[0].page_number
            markdown_output += f"\n### Table {idx + 1} (Page {page_num}):\n"
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
    return markdown_output

# --- 5. 核心函數 (Gemini Logic) ---
def audit_with_gemini(extracted_text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    
    system_prompt = """
    你是一位極度嚴謹的中鋼機械品管稽核員。
    你的輸入是由 Azure OCR 提取的表格文字。請忽略簽名，專注於數據稽核。
    
    請執行以下 **深度邏輯稽核 (Deep Reasoning)**：

    ### 1. 製程判定邏輯 (Process Logic) - 【修正邊界定義】：
    - **未再生/車修**：
       - 判定規則：實測值 **<= (小於或等於)** 規格值。
    - **銲補 (Welding)**：
       - 判定規則：實測值 **>= (大於或等於)** 規格值。
    - **再生車修 (Finish Turning)**：
       - 規格通常為「區間」 (如 101.64~101.66)。
       - 判定規則：實測值 必須 **包含於 (Inclusive)** 上下限之間。

    ### 2. 數量一致性檢查 (Quantity Check) - 【強制執行】：
    - **步驟 A**：讀取項目名稱中的數量要求，例如 `(10PC)` 或 `(5PC)`。
    - **步驟 B**：**逐一清點** 該列提取到的實測數據個數 (Count)。
    - **步驟 C**：比對。若 `實測個數 < 要求個數` -> **FAIL (數量不符)**。
    - **注意**：請對「所有項目」（包含銲補、未再生、再生）都執行此檢查。
    - **例外**：僅「熱處理」項目忽略數量。

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
           "spec_logic": "說明使用的判定標準",
           "measured": "實測數據串",
           "issue_type": "數值超規 / 數量不符",
           "reason": "詳細說明"
         }
      ]
    }
    """
    
    try:
        response = model.generate_content(
            [system_prompt, f"表格數據:\n{extracted_text}"],
            generation_config={"response_mime_type": "application/json"}
        )
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# --- 6. 手機版 UI (移除相機元件版) ---
st.title("🏭 現場稽核助手")

# A. 檔案上傳區 (在手機上點這個按鈕，可以選擇「直接拍照」或「相簿」)
with st.container(border=True):
    st.subheader("📂 新增頁面")
    
    # 使用 uploader_key 來強制重置上傳元件，達到連續上傳的效果
    uploaded_files = st.file_uploader(
        "點擊上傳 (手機可選直接拍照)", 
        type=['jpg', 'png', 'jpeg'], 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_files:
        # 將新上傳的檔案加入暫存區
        for f in uploaded_files:
            st.session_state.photo_gallery.append(f)
        
        # 更新 key，強制清空上傳元件，方便下一輪上傳
        st.session_state.uploader_key += 1
        st.rerun()

# B. 預覽與管理區
if st.session_state.photo_gallery:
    st.divider()
    st.write(f"📊 已累積 **{len(st.session_state.photo_gallery)}** 頁文件")
    
    # 縮圖顯示
    cols = st.columns(3)
    for idx, img in enumerate(st.session_state.photo_gallery):
        with cols[idx % 3]:
            st.image(img, caption=f"P.{idx+1}", use_container_width=True)
            # 刪除按鈕
            if st.button("❌", key=f"del_{idx}"):
                st.session_state.photo_gallery.pop(idx)
                st.rerun()

    # C. 執行按鈕
    st.divider()
    if st.button("🚀 開始分析", type="primary", use_container_width=True):
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        # 1. OCR
        all_text = ""
        total_imgs = len(st.session_state.photo_gallery)
        
        for i, img in enumerate(st.session_state.photo_gallery):
            status.text(f"Azure 正在掃描第 {i+1}/{total_imgs} 頁...")
            try:
                txt = extract_layout_with_azure(img, DOC_ENDPOINT, DOC_KEY)
                all_text += f"\n--- Page {i+1} ---\n{txt}"
            except Exception as e:
                st.error(f"第 {i+1} 頁讀取失敗: {e}")
            progress_bar.progress((i + 1) / (total_imgs + 1))

        # 2. Gemini
        status.text("Gemini 2.5 Pro 正在進行邏輯稽核...")
        result_str = audit_with_gemini(all_text, GEMINI_KEY)
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
                        st.caption(f"實測: {item.get('measured')}")
                        st.caption(f"原因: {item.get('reason')}")
        except:
            st.error("分析錯誤")
            st.code(result_str)
            
    # 清空按鈕
    if st.button("🗑️ 清除所有照片"):
        st.session_state.photo_gallery = []
        st.rerun()

else:
    st.info("👆 請點擊上方按鈕開始新增照片")
