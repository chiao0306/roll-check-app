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

# --- 5. 核心函數：Gemini 神之腦 (Prompt 優化版) ---
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

    ### 0. 核心任務與數據前處理：
    - **識別滾輪編號 (Roll ID)**：找出每筆數據對應的編號 (如 `Y5612001`, `E30`)。
    - **分軌識別**：區分該項目屬於「本體 (Body)」還是「軸頸 (Journal)」。
    - **數值容錯**：忽略數字間的空格 (如 `341 . 12` -> `341.12`)。
    - **跨頁一致性**：所有頁面的工令編號、交貨日期需完全相同。日期格式 `YYY.MM.DD` (允許空格)。

    ### 1. 全域流程與尺寸履歷檢查 (Process & Dimension Continuity) - 【最優先執行】：
    **請建立每一支滾輪編號的完整履歷，並執行以下比對：**
    
    #### A. 流程防呆 (Interlock)：
    - **流程順序**：未再生 -> 銲補 -> 再生車修 -> 研磨。
    - **前向鎖定**：若「本體未再生」階段已標記為「已完工」(有小數點)，則該編號 **不可出現** 在後續任何流程。
    - **後向溯源**：若編號出現在「銲補」、「再生」或「研磨」，則 **必須存在** 於該部位的「未再生」紀錄中 (防止幽靈工件)。

    #### B. 尺寸合理性檢查 (Dimension Jump) - 【嚴格執行】：
    - **研磨限制 (Grinding Check)**：
      - 若同一編號同時存在「再生車修」與「研磨」數據，**研磨尺寸 必須小於 再生車修尺寸**。
      - 若 研磨 >= 再生 -> **FAIL (邏輯異常：研磨後尺寸變大)**。
    - 以 **「最終完成尺寸」** (再生車修或研磨) 為基準 (Base)。
    - **本體 (Body)**：
      - 未再生 (往下跳)：`Base - 未再生` 必須 <= 20mm。
      - 銲補 (往上跳)：`銲補 - Base` 必須 <= 8mm。
    - **軸頸 (Journal)**：
      - 未再生 (往下跳)：`Base - 未再生` 必須 <= 5mm。
      - 銲補 (往上跳)：`銲補 - Base` 必須 <= 7mm。
    - **異常**：若跳動幅度超過上述範圍 -> **FAIL (尺寸異常：數值不連貫)**。

    ### 2. 數量與依賴性檢查 (Quantity & Dependency)：
    - **熱處理**：忽略數量 PC，有數據即 PASS。
    - **單位換算 (Unit Conversion)**：
      - `(1SET=4PCS)` -> 目標 = SET數 * 4。
      - `(SET)` -> 預設 目標 = SET數 * 2。
      - `(PC)` -> 目標 = PC數。
    - **本體 (Body)**：編號必須 **唯一**。實測總數需等於目標數量。
    - **軸頸 (Journal) / 內孔 (Inner Hole)**：允許單一編號出現最多 2 次。實測總數需等於目標數量。
    - **Keyway Cut / 內孔 (Inner Hole) 依賴**：
      - 依賴對象：**必須是** 有進行「軸位再生」的編號。
      - 數量限制：該項目的數量 <= 該編號的軸位再生數量。
      - 孤立檢查：若編號出現在 Keyway 或 內孔，但軸位再生沒做 -> **FAIL (依賴異常)**。
    - **根依賴 (Root Check)**：若編號出現在軸頸/Keyway/內孔，但本體完全沒出現 -> **FAIL (幽靈工件)**。

    ### 3. 製程判定邏輯 (分軌制)：

    #### A. 【本體 (Body)】未再生/車修：
    - **規格解析**：忽略「每次車修」，只看「至 Ymm」。多規格取 **最大值 (Max_Spec)**。
    - **邏輯**：
      - **整數** (未完工)：實測 <= Max_Spec。
      - **小數** (已完工)：實測 >= Max_Spec 且 格式為 `#.##`。

    #### B. 【軸頸 (Journal)】未再生/車修：
    - **規格比對**：採「智慧歸類」，與實測值最接近的規格為目標。
    - **邏輯**：實測 <= 目標規格。
    - **格式**：必須為 **整數**。出現小數 -> **FAIL**。

    #### C. 銲補 (Welding)：
    - **多重規格鎖定**：若再生車修確定為大尺寸，此處必須比對大尺寸規格。
    - **邏輯**：實測 >= 規格。

    #### D. 再生車修 (Finish)：
    - **多重規格**：符合任一規格區間即 PASS (同時鎖定該編號的規格身份)。
    - **數值**：**包含於 (Inclusive)** 上下限之間。
    - **格式**：忽略空格後，必須精確到小數點後兩位。

    #### E. 內孔車修 (Inner Hole)：
    - **規格對應**：軸頸~85 -> 孔50；軸頸~75 -> 孔45。
    - **邏輯**：實測值需在對應規範的公差範圍內。

    ### 5. 上方統計欄位稽核 (Summary Table Reconciliation) - 【新增】：
    **請核對頁面左上方「統計表格」中的「實交數量」與下方/全卷詳細項目的計數：**
    
    - **A. 雙軌聚合規則 (Aggregated Counting)**：
      - **適用項目**：項目名稱包含「ROLL 車修」、「ROLL 銲補」或「ROLL 拆裝」。
      - **車修總數** = 全工令 (本體未再生 + 本體再生 + 軸頸未再生 + 軸頸再生) 的項目總和。
      - **銲補總數** = 全工令 (本體銲補 + 軸頸銲補) 的項目總和。
      - **拆裝總數** = 全工令 (新品組裝 + 舊品拆裝) 的項目總和。
      - **檢查**：統計欄位的數值 必須等於 上述加總。
    
    - **B. 通用規則 (General Rule)**：
      - **適用項目**：不屬於上述 A 類的一般項目 (如「冷卻水管拆除」)。
      - **檢查**：統計欄位的數值 必須等於 該項目在下方列表的 (PC) 數量。
    
    - **C. 例外規則 (Exception)**：
      - **W3 #6 機 驅動輥輪**：該項目的車修/銲補 **不列入** A 類的聚合計算。請依據 B 類規則獨立核對。

    - **異常判定**：若上方統計數量 ≠ 計算出的對應數量 -> **FAIL (統計數量不符)**。

    ### 輸出格式 (JSON Only)：
    {
      "job_no": "工令編號",
      "summary": "總結",
      "issues": [
         {
           "page": "頁碼",
           "item": "項目名稱",
           "issue_type": "數值超規 / 數量不符 / 統計數量不符 / 流程異常 / 格式錯誤",
           "spec_logic": "判定標準",
           "common_reason": "錯誤原因概述",
           "failures": [
              {
                "id": "Y5612001", 
                "val": "136",
                "calc": "若為尺寸跳動或統計錯誤，請列出計算式 (如: 上方統計12 != 下方加總10)"
              }
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

    col_btn1, col_btn2 = st.columns([3, 1])
    
    with col_btn1:
        start_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)
    with col_btn2:
        st.write("") 
        clear_btn = st.button("清除照片🗑️", help="清除所有", use_container_width=True)

    if clear_btn:
        st.session_state.photo_gallery = []
        st.rerun()

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
                            # 動態生成表格：如果有 'calc' 欄位就顯示，沒有就不顯示
                            table_data = []
                            for f in failures:
                                row = {"滾輪編號": f.get('id', '未知'), "實測值": f.get('val', 'N/A')}
                                if f.get('calc'):
                                    row["差值/備註"] = f.get('calc')
                                table_data.append(row)
                                
                            st.dataframe(table_data, use_container_width=True, hide_index=True)
                        else:
                             st.text(f"實測數據: {item.get('measured', 'N/A')}")
                            
        except Exception as e:
            st.error("分析錯誤")
            st.code(result_str)
            st.write(e)

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




