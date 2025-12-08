import streamlit as st
import streamlit.components.v1 as components
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import google.generativeai as genai
import json
import time

# --- 1. 頁面設定 ---
st.set_page_config(page_title="中機交貨單稽核", page_icon="🏭", layout="centered")

# --- CSS 樣式：按鈕 + 標題優化 ---
st.markdown("""
<style>
/* 1. 針對 type="primary" 的按鈕 (開始分析) 進行樣式修改 */
button[kind="primary"] {
    height: 60px;          
    font-size: 20px;       
    font-weight: bold;     
    border-radius: 10px;   
    margin-top: 20px;
    margin-bottom: 20px;
}

/* 2. 讓圖片欄位間距變緊湊 */
div[data-testid="column"] {
    padding: 2px;
}

/* 3. 【新增】控制標題字體大小，強制一行顯示 */
h1 {
    font-size: 1.7rem !important;   /* 數字越小字越小 (原預設約 2.5rem) */
    white-space: nowrap !important; /* 強制不換行 */
    overflow: hidden !important;    /* 超出範圍隱藏 (預防萬一) */
    text-overflow: ellipsis !important;
}
</style>
""", unsafe_allow_html=True)

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

# --- 4. 核心函數：Azure 神之眼 ---
def extract_layout_with_azure(file_obj, endpoint, key):
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    file_content = file_obj.getvalue()
    
    poller = client.begin_analyze_document(
        "prebuilt-layout", 
        file_content,
        content_type="application/octet-stream"
    )
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
    
    header_snippet = result.content[:300] if result.content else ""
    return markdown_output, header_snippet

# --- 5. 核心函數：Gemini 神之腦 ---
def audit_with_gemini(extracted_data_list, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    
    combined_input = "以下是各頁資料：\n"
    for data in extracted_data_list:
        combined_input += f"\n=== Page {data['page']} ===\n"
        combined_input += f"【頁首文字片段】:\n{data['header_text']}\n"
        combined_input += f"【表格數據】:\n{data['table']}\n"

    system_prompt = """
    你是一位極度嚴謹的中鋼機械品管稽核員。
    請依據 Azure OCR 提取的表格文字進行稽核。

    ### ⛔️ 極重要排除指令 (Exclusion Rules)：
    - **完全無視簽名欄位**：請忽略頁面底部的主管/承辦人簽名、簽核日期。
    - 不論是否有簽名、日期是否正確、是否為 `0月`，**一律不檢查、不回報**。
    - 請將注意力 100% 集中在「數據表格」與「表頭資訊」。

    ### 0. 核心任務與數據清洗：
    - **識別滾輪編號 (Roll ID)**：找出每筆數據對應的編號 (如 `Y5612001`, `E30`)。
    - **分軌識別**：判斷該項目屬於「本體 (Body)」還是「軸頸 (Journal)」。
    - **數值容錯**：忽略數字間的空格 (如 `341 . 12` -> `341.12`)。

    ### 1. 數量一致性檢查 (Quantity Logic Split)：
    - **優先檢查：特例項目**
      - **熱處理 (Heat Treatment)**：
        - 規則：**忽略** 項目名稱中的數量要求 (PC)。
        - 判定：只要該欄位有填寫數據 (通常為重量 KG)，且筆數 >= 1，即視為 **PASS**。
    
    - **情境 A：軸頸 (Journal)**
      - 適用：項目名稱含「軸頸」或「軸位」。
      - 規則：允許同一編號出現最多 **2次**。
      - 數量：**總資料筆數** (含重複) 必須等於 要求數量。

    - **情境 B：本體 (Body) 與 其他項目 (預設)**
      - 適用：項目名稱含「本體」或 未包含上述關鍵字的項目。
      - 規則：實測數據的「編號」必須 **唯一**。若有重複 -> **FAIL (編號重複)**。
      - 數量：獨立編號總數 必須等於 要求數量。

    ### 2. 存在性依賴檢查 (Dependency Check)：
    - **規則**：軸頸出現的編號，必須曾經在本體相關項目出現過。
    - **異常**：若軸頸有編號 `X`，但本體完全沒出現過 `X` -> **FAIL (孤立軸頸)**。

    ### 3. 製程判定邏輯 (分軌制)：

    #### A. 【本體 (Body)】未再生/車修：
    - **規格解析 (Spec Parsing) - 【強制最大值】**：
      - 請掃描該項目列出的所有尺寸數字 (忽略「每次車修Xmm」或「無裂痕」等文字)。
      - **關鍵規則**：若出現多個目標尺寸 (例如 338mm 與 344mm)，**一律取數值最大者** 作為唯一比對標準 (Max_Spec)。
      - 範例：規範包含 338 與 344 -> Max_Spec = 344。
    - **邏輯分流**：
      1. **整數** (未完工)：實測值 **<=** Max_Spec。
      2. **小數** (已完工)：實測值 **>=** Max_Spec，且格式需為 `#.##`。

    #### B. 【軸頸 (Journal)】未再生/車修：
    - **步驟 1 (智慧歸類)**：若有多個規格 (如 157, 127)，請計算實測值與各規格的距離，選出 **數值最接近** 的那個當作「目標規格」。
    - **步驟 2 (數值比對)**：實測值 必須 **<= (小於等於)** 目標規格。
    - **步驟 3 (格式檢查)**：實測值必須為 **整數**。若有小數點 -> **FAIL** (軸頸未再生不可完工)。

    #### C. 銲補 (Welding) (通用)：
    - 規則：實測值 **>=** 規格。

    #### D. 再生車修 (Finish) (通用)：
    - 數值：**包含於 (Inclusive)** 上下限之間。
    - 格式：忽略空格後，必須精確到小數點後兩位。

    ### 4. 全域流程防呆 (Process Integrity) - 【補回尺寸邏輯】：
    - **前向檢查**：本體未再生已完工(小數) -> 不可出現在後續。
    - **後向檢查**：出現在銲補/再生 -> 前面必須有未再生紀錄。
    - **尺寸合理性檢查 (Dimension Continuity)**：
      - 檢查同一編號在 未再生 -> 銲補 -> 再生 過程中的尺寸變化。
      - 基準：各階段尺寸應在合理範圍內 (例如 350 ± 20mm)。
      - 若出現劇烈跳動 (如 350 -> 200) -> **FAIL (尺寸異常：數值不連貫)**。
    - **跨頁一致性**：工令、日期需一致 (日期格式 `YYY.MM.DD` 允許空格)。

    ### 輸出格式 (JSON Only)：
    {
      "job_no": "工令編號",
      "summary": "總結",
      "issues": [
         {
           "page": "頁碼",
           "item": "項目名稱",
           "issue_type": "數值超規 / 數量不符 / 流程異常 / 尺寸異常 / 格式錯誤 / 編號異常",
           "spec_logic": "判定標準",
           "common_reason": "錯誤原因概述",
           "failures": [
              {"id": "Y5612001", "val": "136"}
           ]
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
st.title("🏭 中機交貨單稽核")

# A. 檔案上傳區
with st.container(border=True):
    uploaded_files = st.file_uploader(
        "📂 新增頁面 (點擊拍照或上傳)", 
        type=['jpg', 'png', 'jpeg'], 
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}"
    )
    if uploaded_files:
        for f in uploaded_files:
            st.session_state.photo_gallery.append(f)
        st.session_state.uploader_key += 1
        
        # 自動捲動
        components.html(
            """
            <script>
                var docBody = window.parent.document.body;
                window.parent.scrollTo(0, docBody.scrollHeight);
            </script>
            """,
            height=0
        )
        st.rerun()

# B. 預覽與管理區
if st.session_state.photo_gallery:
    
    st.caption(f"已累積 {len(st.session_state.photo_gallery)} 頁文件")

    # --- 操作按鈕區 (置頂) ---
    col_btn1, col_btn2 = st.columns([3, 1])
    
    with col_btn1:
        start_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)
    with col_btn2:
        st.write("") 
        clear_btn = st.button("清除照片🗑️", help="清除所有", use_container_width=True)

    if clear_btn:
        st.session_state.photo_gallery = []
        st.rerun()

    # --- 執行分析邏輯 ---
    if start_btn:
        start_time = time.time()
        status = st.empty()
        progress_bar = st.progress(0)
        
        # 1. OCR
        extracted_data_list = []
        total_imgs = len(st.session_state.photo_gallery)
        
        for i, img in enumerate(st.session_state.photo_gallery):
            status.text(f"Azure 正在掃描第 {i+1}/{total_imgs} 頁...")
            img.seek(0)
            try:
                table_md, text_snippets = extract_layout_with_azure(img, DOC_ENDPOINT, DOC_KEY)
                extracted_data_list.append({
                    "page": i + 1,
                    "table": table_md,
                    "header_text": text_snippets 
                })
            except Exception as e:
                st.error(f"第 {i+1} 頁讀取失敗: {e}")
            progress_bar.progress((i + 1) / (total_imgs + 1))

        # 2. Gemini
        status.text(f"Gemini 2.5 Pro 正在進行邏輯稽核...")
        result_str = audit_with_gemini(extracted_data_list, GEMINI_KEY)
        
        progress_bar.progress(100)
        status.text("完成！")
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 3. 顯示結果
        try:
            result = json.loads(result_str)
            if isinstance(result, list): result = result[0] if len(result) > 0 else {}
            
            st.success(f"工令: {result.get('job_no', 'Unknown')} | ⏱️ 耗時: {elapsed_time:.1f} 秒")
            
            issues = result.get('issues', [])
            if not issues:
                st.balloons()
                st.success("✅ 全數合格！")
            else:
                st.error(f"發現 {len(issues)} 類異常項目")
                
                for item in issues:
                    with st.container(border=True):
                        col_head1, col_head2 = st.columns([3, 1])
                        page_str = str(item.get('page', '?'))
                        col_head1.markdown(f"**P.{page_str} | {item.get('item')}**")
                        
                        itype = item.get('issue_type', '異常')
                        if "流程" in itype or "尺寸" in itype or "編號" in itype:
                            col_head2.error(f"🛑 {itype}")
                        else:
                            col_head2.warning(f"⚠️ {itype}")
                        
                        st.caption(f"原因: {item.get('common_reason')}")
                        if item.get('spec_logic'):
                            st.caption(f"標準: {item.get('spec_logic')}")
                        
                        failures = item.get('failures', [])
                        if failures:
                            table_data = [{"滾輪編號": f.get('id', '未知'), "實測值": f.get('val', 'N/A')} for f in failures]
                            st.dataframe(table_data, use_container_width=True, hide_index=True)
                        else:
                             st.text(f"實測數據: {item.get('measured', 'N/A')}")
                            
        except Exception as e:
            st.error("分析錯誤")
            st.code(result_str)
            st.write(e)

    # --- 圖片縮圖區 ---
    st.divider()
    st.caption("已拍攝照片：")
    
    cols = st.columns(4)
    for idx, img in enumerate(st.session_state.photo_gallery):
        with cols[idx % 4]:
            st.image(img, caption=f"P.{idx+1}", use_container_width=True)
            if st.button("❌", key=f"del_{idx}"):
                st.session_state.photo_gallery.pop(idx)
                st.rerun()

else:
    st.info("👆 請點擊上方按鈕開始新增照片")

