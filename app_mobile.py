import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import google.generativeai as genai
import json
from io import BytesIO

# --- 1. 頁面設定 ---
st.set_page_config(page_title="中鋼機械稽核", page_icon="🏭", layout="centered") # 手機版用 centered 比較好看

# --- 2. 秘密金鑰讀取 (從雲端設定讀取) ---
# 這樣就不用在介面上輸入了
try:
    DOC_ENDPOINT = st.secrets["DOC_ENDPOINT"]
    DOC_KEY = st.secrets["DOC_KEY"]
    GEMINI_KEY = st.secrets["GEMINI_KEY"]
except:
    st.error("找不到金鑰！請在 Streamlit Cloud 設定 Secrets。")
    st.stop()

# --- 3. 初始化 Session State (用來存照片) ---
if 'photo_gallery' not in st.session_state:
    st.session_state.photo_gallery = [] # 存放所有拍好的照片
if 'camera_key' not in st.session_state:
    st.session_state.camera_key = 0     # 用來重置相機

# --- 4. 核心函數 (維持不變，省略細節以節省版面，請直接用上一版的邏輯) ---
# ... (這裡放入 extract_layout_with_azure 函數) ...
def extract_layout_with_azure(file_obj, endpoint, key):
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    file_content = file_obj.getvalue() # 注意：Session State 裡的圖片要用 getvalue()
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

# ... (這裡放入 audit_with_gemini 函數，完全沿用上一版的邏輯) ...
def audit_with_gemini(extracted_text, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    
    # ... (Prompt 保持上一版最強的那個設定，這裡省略以節省篇幅) ...
    # 請務必把上一版完整的 system_prompt 貼在這裡
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
       - **重要範例**：若規格 101.64~101.66，實測 **101.66 為 PASS**，實測 **101.64 為 PASS**。只有超過這個範圍才算 FAIL。

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
    - 規格下限 203.52，實測 203.50 -> **FAIL** (因為 203.50 < 203.52)。

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
           "reason": "詳細說明 (例如: 應測10PC，實測僅8PC / 101.67 超出上限 101.66)"
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

# --- 5. 手機版專用 UI ---
st.title("🏭 現場稽核助手")

# A. 拍照區
with st.expander("📸 開啟相機 / 上傳照片", expanded=True):
    # 這是 Streamlit 的相機元件
    # 在手機上，它會直接呼叫前/後鏡頭
    img_file_buffer = st.camera_input("拍攝檢驗單", key=f"cam_{st.session_state.camera_key}")

    if img_file_buffer is not None:
        # 當拍下一張照片時
        timestamp = img_file_buffer.name
        # 存入列表
        st.session_state.photo_gallery.append(img_file_buffer)
        # 強制重置相機元件，讓使用者可以拍下一張
        st.session_state.camera_key += 1
        st.rerun()

    # 也可以保留「從相簿上傳」的選項
    uploaded_files = st.file_uploader("或從相簿選擇", accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            st.session_state.photo_gallery.append(f)
        # 清空上傳暫存
        st.rerun()

# B. 預覽與管理區 (實現你的「重拍/刪除」需求)
if st.session_state.photo_gallery:
    st.divider()
    st.write(f"📊 已累積 **{len(st.session_state.photo_gallery)}** 頁文件")
    
    # 顯示縮圖列
    cols = st.columns(3)
    for idx, img in enumerate(st.session_state.photo_gallery):
        with cols[idx % 3]:
            st.image(img, caption=f"第 {idx+1} 頁")
            # 刪除按鈕 (如果不滿意這張)
            if st.button("🗑️", key=f"del_{idx}"):
                st.session_state.photo_gallery.pop(idx)
                st.rerun()

    # C. 執行按鈕
    st.divider()
    if st.button("🚀 結束拍照，開始分析", type="primary", use_container_width=True):
        
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
                        c1, c2 = st.columns([1, 3])
                        c1.error(item.get('issue_type'))
                        c1.caption(f"第 {item.get('page')} 頁")
                        c2.markdown(f"**{item.get('item')}**")
                        c2.write(f"實測: `{item.get('measured')}`")
                        c2.caption(f"原因: {item.get('reason')}")
        except:
            st.error("分析錯誤")
            st.code(result_str)
            
    # 清空按鈕
    if st.button("清除所有照片，重新開始"):
        st.session_state.photo_gallery = []
        st.rerun()
else:
    st.info("👆 請使用上方相機拍攝第一頁")