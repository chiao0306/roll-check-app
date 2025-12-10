import streamlit as st
import streamlit.components.v1 as components
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import google.generativeai as genai
import json
import time
import concurrent.futures

# --- 1. 頁面設定 ---
st.set_page_config(page_title="中鋼機械稽核", page_icon="🏭", layout="centered")

# --- CSS 樣式 ---
st.markdown("""
<style>
button[kind="primary"] {
    height: 80px; font-size: 20px; font-weight: bold; border-radius: 10px;
    margin-top: 20px; margin-bottom: 20px;
}
div[data-testid="column"] { padding: 2px; }
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
if 'photo_gallery' not in st.session_state: st.session_state.photo_gallery = []
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

# --- 4. 核心函數：Azure 神之眼 ---
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
    
    header_snippet = result.content[:800] if result.content else "" # 稍微加長一點給會計師看
    return markdown_output, header_snippet

# --- 5.1 Agent A: 工程師 (負責製程與尺寸) ---
def agent_engineer_check(combined_input, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    
    system_prompt = """
    你是一位極度嚴謹的中鋼機械品管【工程師】。
    你的任務是專注於「數據規格」、「製程邏輯」與「尺寸合理性」。
    **請完全忽略** 數量計算與表頭統計，那不是你的工作。

    ### ⛔️ 排除指令：
    - 不檢查數量 PC/SET 是否相符。
    - 不檢查表頭統計欄位。
    - 不檢查簽名。

    ### 1. 核心邏輯 (Process & Dimension)：
    **請建立每一支滾輪編號 (Roll ID) 的完整履歷，並執行以下比對：**
    
    #### A. 流程防呆 (Interlock)：
    - **流程順序**：未再生 -> 銲補 -> 再生車修 -> 研磨。
    - **前向鎖定**：若「本體未再生」階段已標記為「已完工」(有小數點)，則該編號 **不可出現** 在後續任何流程。
    - **後向溯源**：若編號出現在「銲補」、「再生」或「研磨」，則 **必須存在** 於該部位的「未再生」紀錄中 (防止幽靈工件)。
    - **存在性依賴**：若編號出現在軸頸/Keyway/內孔，但本體完全沒出現 -> **FAIL (幽靈工件)**。
    - **Keyway/內孔依賴**：必須有「軸位再生」才能做。

    #### B. 尺寸合理性檢查 (Dimension Jump) - 【嚴格執行】：
    - **研磨限制**：研磨尺寸 必須小於 再生車修尺寸。
    - 以 **「最終完成尺寸」** (再生車修或研磨) 為基準 (Base)。
    - **本體 (Body)**：
      - 未再生 (往下跳)：`Base - 未再生` 必須 <= 20mm。
      - 銲補 (往上跳)：`銲補 - Base` 必須 <= 8mm。
    - **軸頸 (Journal)**：
      - 未再生 (往下跳)：`Base - 未再生` 必須 <= 5mm。
      - 銲補 (往上跳)：`銲補 - Base` 必須 <= 7mm。
    - **異常**：跳動幅度過大 -> **FAIL (尺寸異常)**。

    ### 2. 製程判定邏輯 (分軌制)：
    **數值容錯**：忽略數字間的空格 (如 `341 . 12` -> `341.12`)。

    #### A. 【本體 (Body)】未再生/車修：
    - **規格**：忽略「每次車修」，只看「至 Ymm」。多規格取 **最大值 (Max_Spec)**。
    - **邏輯**：整數(未完工) <= Max_Spec；小數(已完工) >= Max_Spec 且格式 `#.##`。

    #### B. 【軸頸 (Journal)】未再生/車修：
    - **規格**：採「智慧歸類」，比對最接近的規格。
    - **邏輯**：實測 <= 目標規格。
    - **格式**：必須為 **整數**。出現小數 -> **FAIL**。

    #### C. 銲補 (Welding) - 【加法邏輯】：
    - **邏輯防呆**：銲補是加肉，數值越大越好。
    - **規則**：實測值 **>=** 規格。嚴禁使用未再生的<=邏輯。

    #### D. 再生車修 (Finish) / E. 內孔 (Inner Hole)：
    - **多重規格**：符合任一規格區間即 PASS。
    - **內孔對應**：軸頸~85 -> 孔50；軸頸~75 -> 孔45。
    - **數值**：**包含於 (Inclusive)** 上下限之間。
    - **格式**：精確到小數點後兩位。

    ### 輸出格式 (JSON Only)：
    {
      "issues": [
         {
           "page": "頁碼",
           "item": "項目名稱",
           "issue_type": "數值超規 / 流程異常 / 尺寸異常 / 格式錯誤 / 依賴異常",
           "spec_logic": "判定標準",
           "common_reason": "錯誤原因概述",
           "failures": [{"id": "ID", "val": "Value", "calc": "計算式(若有)"}]
         }
      ]
    }
    """
    try:
        response = model.generate_content([system_prompt, combined_input], generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except:
        return {"issues": []}

# --- 5.2 Agent B: 會計師 (負責數量與統計) ---
def agent_accountant_check(combined_input, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    
    system_prompt = """
    你是一位極度嚴謹的中鋼機械品管【會計師】。
    你的任務是專注於「數量核對」、「表頭一致性」與「上方統計表格」。
    **請完全忽略** 尺寸公差與製程邏輯，那不是你的工作。

    ### ⛔️ 排除指令：
    - 不檢查尺寸是否超規。
    - 不檢查流程先後順序。
    - 不檢查簽名。

    ### 1. 跨頁一致性 (Header)：
    - 工令編號、交貨日期(預定/實際)：所有頁面必須相同。日期格式 `YYY.MM.DD` (允許空格)。

    ### 2. 數量一致性檢查 (Quantity)：
    - **單位換算**：`(1SET=4PCS)` -> *4；`(SET)` -> *2；`(PC)` -> *1。
    - **熱處理**：忽略數量，有數據即 PASS。
    - **本體 (Body)**：編號必須 **唯一**。實測總數 = 目標數量。
    - **軸頸 (Journal) / 內孔**：允許單一編號出現 2 次。實測總數 = 目標數量。
    - **Keyway**：Keyway 數量 <= 軸位再生數量。

    ### 3. 上方統計欄位稽核 (Summary Table Reconciliation)：
    **請核對左上角「統計表格」的「實交數量」與內文計數：**
    - **A. 雙軌聚合 (Aggregated)**：
      - 項目：含「ROLL 車修」、「ROLL 銲補」、「ROLL 拆裝」。
      - 車修總數 = 全卷 (本體未再生 + 本體再生 + 軸頸未再生 + 軸頸再生) 總和。
      - 銲補總數 = 全卷 (本體銲補 + 軸頸銲補) 總和。
      - 拆裝總數 = 全卷 (新品組裝 + 舊品拆裝) 總和。
    - **B. 通用規則**：其他項目 (如水管拆除) -> 統計數 = 下方列表數。
    - **C. 例外**：**W3 #6 機 驅動輥輪** 不列入聚合，採通用規則獨立核對。
    - **判定**：若 統計數量 != 計算數量 -> **FAIL**。

    ### 輸出格式 (JSON Only)：
    {
      "job_no": "工令編號",
      "issues": [
         {
           "page": "頁碼",
           "item": "項目名稱",
           "issue_type": "數量不符 / 統計數量不符 / 跨頁資訊不符 / 編號重複",
           "spec_logic": "判定標準",
           "common_reason": "錯誤原因概述",
           "failures": [{"id": "ID", "val": "Count", "calc": "統計X != 計算Y"}]
         }
      ]
    }
    """
    try:
        response = model.generate_content([system_prompt, combined_input], generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except:
        return {"job_no": "Error", "issues": []}

# --- 6. 手機版 UI ---
st.title("🏭 中鋼機械稽核")

with st.container(border=True):
    uploaded_files = st.file_uploader("📂 新增頁面", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
    if uploaded_files:
        for f in uploaded_files: st.session_state.photo_gallery.append(f)
        st.session_state.uploader_key += 1
        components.html("""<script>window.parent.document.body.scrollTo(0, window.parent.document.body.scrollHeight);</script>""", height=0)
        st.rerun()

if st.session_state.photo_gallery:
    st.caption(f"已累積 {len(st.session_state.photo_gallery)} 頁文件")
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1: start_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)
    with col_btn2: 
        st.write("")
        clear_btn = st.button("🗑️", help="清除", use_container_width=True)

    if clear_btn:
        st.session_state.photo_gallery = []
        st.rerun()

    if start_btn:
        start_time = time.time()
        status = st.empty()
        progress_bar = st.progress(0)
        
        # 1. OCR (依序掃描)
        extracted_data_list = []
        total_imgs = len(st.session_state.photo_gallery)
        
        for i, img in enumerate(st.session_state.photo_gallery):
            status.text(f"Azure 正在掃描第 {i+1}/{total_imgs} 頁...")
            img.seek(0)
            try:
                table_md, text_snippets = extract_layout_with_azure(img, DOC_ENDPOINT, DOC_KEY)
                extracted_data_list.append({"page": i + 1, "table": table_md, "header_text": text_snippets})
            except Exception as e:
                st.error(f"第 {i+1} 頁讀取失敗: {e}")
            progress_bar.progress((i + 1) / (total_imgs + 1))

        # 2. 雙軌平行稽核 (Parallel Execution)
        combined_input = "以下是各頁資料：\n"
        for data in extracted_data_list:
            combined_input += f"\n=== Page {data['page']} ===\n【頁首】:\n{data['header_text']}\n【表格】:\n{data['table']}\n"

        status.text("Gemini 雙代理人正在平行稽核 (工程師 & 會計師)...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_eng = executor.submit(agent_engineer_check, combined_input, GEMINI_KEY)
            future_acc = executor.submit(agent_accountant_check, combined_input, GEMINI_KEY)
            
            # 等待兩者完成
            res_eng = future_eng.result()
            res_acc = future_acc.result()
        
        progress_bar.progress(100)
        status.text("完成！")
        end_time = time.time()
        
        # 3. 合併結果
        job_no = res_acc.get("job_no", "Unknown") # 工令以會計師為準
        issues_eng = res_eng.get("issues", [])
        issues_acc = res_acc.get("issues", [])
        all_issues = issues_eng + issues_acc # 合併列表

        st.success(f"工令: {job_no} | ⏱️ 耗時: {end_time - start_time:.1f} 秒")
        
        if not all_issues:
            st.balloons()
            st.success("✅ 全數合格！")
        else:
            st.error(f"發現 {len(all_issues)} 類異常項目")
            for item in all_issues:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    c1.markdown(f"**P.{item.get('page', '?')} | {item.get('item')}**")
                    itype = item.get('issue_type', '異常')
                    if "流程" in itype or "尺寸" in itype or "統計" in itype: c2.error(f"🛑 {itype}")
                    else: c2.warning(f"⚠️ {itype}")
                    
                    st.caption(f"原因: {item.get('common_reason')}")
                    if item.get('spec_logic'): st.caption(f"標準: {item.get('spec_logic')}")
                    
                    failures = item.get('failures', [])
                    if failures:
                        # 動態表格：根據是否有 calc 欄位決定顯示內容
                        table_data = []
                        for f in failures:
                            row = {"滾輪編號": f.get('id', '未知'), "實測/計數": f.get('val', 'N/A')}
                            if f.get('calc'): row["差值/備註"] = f.get('calc')
                            table_data.append(row)
                        st.dataframe(table_data, use_container_width=True, hide_index=True)
                    else:
                        st.text(f"實測數據: {item.get('measured', 'N/A')}")

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