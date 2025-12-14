import streamlit as st
import streamlit.components.v1 as components
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import google.generativeai as genai
import json
import time
import concurrent.futures
import pandas as pd  # 【新增】這行一定要有，才能讀 Excel

# --- 1. 頁面設定 ---
st.set_page_config(page_title="中機交貨單稽核", page_icon="🏭", layout="centered")

# --- CSS 樣式 ---
st.markdown("""
<style>
button[kind="primary"] {
    height: 60px; font-size: 20px; font-weight: bold; border-radius: 10px;
    margin-top: 20px; margin-bottom: 20px;
}
div[data-testid="column"] { padding: 2px; }
h1 {
    font-size: 1.7rem !important; white-space: nowrap !important;
    overflow: hidden !important; text-overflow: ellipsis !important;
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
if 'photo_gallery' not in st.session_state: st.session_state.photo_gallery = []
if 'uploader_key' not in st.session_state: st.session_state.uploader_key = 0

# --- 【新增】側邊欄模型設定 ---
with st.sidebar:
    st.header("🧠 模型設定")
    
    model_options = {
        "Gemini 2.5 Pro (精準)": "models/gemini-2.5-pro",
        "Gemini 2.5 Flash (極速)": "models/gemini-2.5-flash"
    }
    
    st.subheader("👷 工程師 Agent")
    eng_selection = st.radio("負責：製程、尺寸、依賴", options=list(model_options.keys()), index=0, key="eng_model")
    eng_model_name = model_options[eng_selection]
    
    st.divider()
    
    st.subheader("👨‍💼 會計師 Agent")
    acc_selection = st.radio("負責：數量、統計、表頭", options=list(model_options.keys()), index=0, key="acc_model")
    acc_model_name = model_options[acc_selection]

# --- 【新增】Excel 規則讀取函數 ---
@st.cache_data
def get_dynamic_rules(ocr_text):
    try:
        # 讀取 Excel (GitHub 上的 rules.xlsx)
        df = pd.read_excel("rules.xlsx")
        matched_rules = []
        for index, row in df.iterrows():
            keyword = str(row.iloc[0]).strip()
            rule = str(row.iloc[1]).strip()
            # 如果 Excel 里的關鍵字出現在 OCR 內容中
            if keyword in ocr_text:
                matched_rules.append(f"- 項目: {keyword} -> 規範: {rule}")
        
        if not matched_rules: return "無特定對應規則，請依通用邏輯判斷。"
        return "\n".join(matched_rules)
    except:
        return "無外部規則檔 (rules.xlsx)，僅使用通用邏輯。"

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
    
    header_snippet = result.content[:800] if result.content else ""
    # 回傳全文以供規則比對
    return markdown_output, header_snippet, result.content

# --- 5.1 Agent A: 工程師 (動態規則版) ---
def agent_engineer_check(combined_input, full_text_for_search, api_key, model_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # 1. 先去 Excel 撈規則
    dynamic_rules = get_dynamic_rules(full_text_for_search)

    system_prompt = f"""
    你是一位極度嚴謹的中鋼機械品管【工程師】。
    
    ### 📂 專案特定規範 (Project Specs from Excel)：
    **以下是根據文件內容自動檢索到的標準答案，優先級最高：**
    {dynamic_rules}
    --------------------------------------------------
    任務：專注於「數據規格」、「製程邏輯」與「尺寸合理性」。
    
    ### ⛔️ 極重要原則 (Strict Rules)：
    1. **合格即PASS**：只要實測值落在規格區間內 (包含邊界值)，就是 **PASS**。
    2. **禁止雞婆**：絕對 **不要** 回報「接近上限」、「裕度不足」、「剛好達標」等主觀意見。這會干擾判斷。
    3. **排除無關項目**：不檢查數量、不檢查表頭、不檢查簽名。

    ### 0. 核心任務與數據前處理：
    - **識別滾輪編號 (Roll ID)**：找出每筆數據對應的編號 (如 `Y5612001`, `E30`)。
    - **分軌識別**：區分該項目屬於「本體 (Body)」還是「軸頸 (Journal)」。
    - **數值容錯**：忽略數字間的空格 (如 `341 . 12` -> `341.12`)。

    ### 1. 核心邏輯 (Process & Dimension)：
    **請建立每一支滾輪編號 (Roll ID) 的完整履歷，並執行以下比對：**
    
    #### ⚠️ 獨立項目豁免 (Standalone Exemption) - 【優先排除】：
    - **定義**：以下項目屬於獨立加工，**不參與** 下方的「流程防呆(A)」與「尺寸邏輯檢查(B)」：
      1. **組裝/拆裝** (包含新品組裝、舊品拆裝、真圓度測試)。
      2. **鍵槽 (Keyway)**。
      3. **內孔 (Inner Hole)**。
    - **規則**：針對上述項目，請 **僅執行** 「第 2 點：製程判定邏輯 (單項規格檢查)」，**忽略** 跨流程的前後對照。

    #### A. 流程防呆 (Interlock) - 【邏輯修正】：
    - **流程順序**：未再生 -> 銲補 -> 再生車修 -> 研磨。
    - **完工定義**：
      - **已完工**：本體未再生實測值為「小數」(有小數點) -> **不可出現** 在後續任何流程。
      - **未完工**：本體未再生實測值為「整數」 -> **必須進入** 後續的「銲補」與「再生車修」流程。
    - **異常判定**：
      1. 若「已完工」卻出現在後續 -> **FAIL (流程異常：已完工件重複加工)**。
      2. 若「未完工(整數)」卻 **沒有** 出現在「銲補」或「再生」 -> **FAIL (流程異常：未完工件中斷)**。
    - **後向溯源**：若編號出現在「銲補」、「再生」或「研磨」，則 **必須存在** 於該部位的「未再生」紀錄中 (防止幽靈工件)。
    - **存在性依賴**：若編號出現在軸頸/Keyway/內孔，但本體完全沒出現 -> **FAIL (幽靈工件)**。
    - **Keyway/內孔依賴**：必須有「軸位再生」才能做。

    #### B. 尺寸邏輯檢查 (Size Ordering) - 【嚴格執行】：
    - **核心原則**：針對同一編號，依據製程物理特性，尺寸大小必須符合以下順序：
      **`未再生 (Pre-repair) < 研磨 (Grinding) < 再生車修 (Finish) < 銲補 (Welding)`**
    - **詳細驗證規則** (若該階段有數據)：
      1. **未再生車修**：必須是該編號所有流程中的 **最小值**。
      2. **銲補**：必須是該編號所有流程中的 **最大值**。
      3. **研磨 vs 再生**：若兩者皆存在，**研磨 必須小於 再生車修**。
    - **異常判定**：若違反上述任何大小關係 (例如：未再生 > 再生，或 研磨 > 銲補) -> **FAIL (尺寸邏輯異常：違反製程大小順序)**。
    
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
    - **數值**：**包含於 (Inclusive)** 上下限之間。 `Min <= X <= Max` 均為合格。
    - **格式**：精確到小數點後兩位。

    #### F. 組裝/拆裝 (Assembly) - 【真圓度檢查】：
    - **適用項目**：項目名稱包含「舊品拆裝」或「新品組裝」者。
    - **規格識別**：尋找「真圓度」規範 (例如：真圓度 ±0.1mm)。
    - **數值邏輯**：真圓度數值必須 **<=** 規範上限 (取絕對值，例如 0.1)。
    - **格式規定**：
      - 實測值必須精確到 **小數點後兩位 (`#.##`)**。
      - 範例：`0.03` -> **PASS**；`0.1` -> **FAIL (位數不足)**；`0.0` -> **FAIL (位數不足)**。

    ### 輸出格式 (JSON Only) - 【請極度簡潔，節省成本】：
    - **common_reason**: 限制在 **15個中文字以內**。例如 "數值超規"、"流程異常"。**禁止** 在此欄位解釋計算過程。
    - **spec_logic**: 僅寫出標準即可，例如 ">= 233"。
    - **Excel 標記**：若該項目的判定標準是來自最上方的 **「專案特定規範 (Project Specs)」**，請務必在 `item` 名稱最後面加上 `(📚Excel)` 以供識別。
    {{
      "issues": [
         {{
           "page": "頁碼",
           "item": "項目名稱 (若來自Excel請加註標記)",
           "issue_type": "數值超規 / 流程異常 / 尺寸異常 / 格式錯誤 / 依賴異常",
           "spec_logic": "判定標準",
           "common_reason": "簡短錯誤原因(限15字)",
           "failures": [{{ "id": "ID", "val": "Value", "calc": "計算式(若有)" }}]
         }}
      ]
    }}
    """
    
    # 穩定參數
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }
    
    try:
        response = model.generate_content([system_prompt, combined_input], generation_config=generation_config)
        return json.loads(response.text)
    except:
        return {"issues": []}

# --- 5.2 Agent B: 會計師 (運費規則版) ---
def agent_accountant_check(combined_input, api_key, model_name):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
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
    - **本體 (Body)**：
      - **唯一性定義**：檢查範圍僅限於 **「單一項目內」**。
      - 規則：在同一個項目(如本體未再生)中，編號不可重複。
      - **注意**：同一編號出現在不同項目(如P2未再生、P3銲補)是正常流程，**不算** 重複。
      - 數量：該項目內的獨立編號總數 = 目標數量。
    - **軸頸 (Journal) / 內孔**：允許單一編號出現 2 次。實測總數 = 目標數量。
    - **Keyway**：Keyway 數量 <= 軸位再生數量。

    ### 3. 上方統計欄位稽核 (Summary Table Reconciliation) - 【邏輯修正】：
    **請核對左上角「統計表格」的「實交數量」與內文計數：**
    - **重要前提**：上方統計表格的數值代表 **「全卷總數」**。若在每一頁重複出現，**請勿累加**，取單一值即可。
    
    - **A. 運費規則 (Freight) - 【邏輯修正】：**
      - 適用項目：名稱包含「運費」者 (如「輥輪拆裝.車修或銲補運費」)。
      - **計數來源**：僅計算全卷 **「本體未再生車修」** 的項目數量。
      - **運費專用計數邏輯 (Freight Counting Logic)**：
        - **特例 (Exception)**：若項目名稱包含 `W3 #1~6號機 130~145 ROLL ROLL BODY車修加工`，該項目的 `1 SET` 在運費計算中視為 **1 個單位 (x1)**。
        - **情境 1**：若項目名稱包含 `(1SET=4PCS)` 關鍵字，該項目的 `1 SET` 在運費計算中視為 **1 個單位 (x1)**。
        - **情境 2**：若項目名稱僅標示 `(SET)` 且無上述特殊定義，該項目的 `1 SET` 在運費計算中視為 **2 個單位 (x2)**。
        - **情境 3**：若標示 `(PC)`，則直接累加數量。
      - **檢查**：統計欄位的數值 必須等於 上述邏輯計算出的總和。
    - **B. 雙軌聚合 (Aggregated)**：
      - 項目：含「ROLL 車修」、「ROLL 銲補」、「ROLL 拆裝」。
      - 車修總數 = 全卷 (本體未再生 + 本體再生 + 軸頸未再生 + 軸頸再生) 總和。
      - 銲補總數 = 全卷 (本體銲補 + 軸頸銲補) 總和。
      - 拆裝總數 = 全卷 (新品組裝 + 舊品拆裝) 總和。
    - **C. 通用規則**：其他項目 (如水管拆除) -> 統計數 = 下方列表數。
    - **D. 例外**：**W3 #6 機 驅動輥輪** 不列入聚合，採通用規則獨立核對。
    
    - **判定**：若 統計數量(單一值) != 計算出的總和 -> **FAIL**。

    ### 4. 執行步驟 (Step-by-Step Execution) - 【強制點名，不回傳】：
    為了確保數量準確，在判斷數量是否異常前，請執行以下內心思考 (不要輸出到 JSON)：
    1. **Extraction (提取)**：找出該項目所有相關的實測編號。
    2. **Counting (計數)**：計算這些編號的數量。
    3. **Comparison (比對)**：與目標數量比對。
    4. **Reporting (回報)**：只有當兩者不符時，才生成 Error。

    ### 輸出格式 (JSON Only) - 【請極度簡潔，節省成本】：
    - **common_reason**: 限制在 **15個中文字以內**。例如 "統計數量不符"、"數量不符"。**禁止** 在此欄位解釋計算過程。
    - **calc**: 在此欄位顯示簡單算式即可，例如 "統計32 != 計算26"。
    {
      "job_no": "工令編號",
      "issues": [
         {
           "page": "頁碼",
           "item": "項目名稱",
           "issue_type": "數量不符 / 統計數量不符 / 跨頁資訊不符 / 編號重複",
           "spec_logic": "判定標準",
           "common_reason": "簡短錯誤原因(限15字)",
           "failures": [{"id": "ID", "val": "Count", "calc": "統計X != 計算Y"}]
         }
      ]
    }
    """
    
    generation_config = {
        "response_mime_type": "application/json",
        "temperature": 0.0,
    }

    try:
        response = model.generate_content([system_prompt, combined_input], generation_config=generation_config)
        return json.loads(response.text)
    except:
        return {"job_no": "Error", "issues": []}

# --- 6. 手機版 UI ---
st.title("🏭 中機交貨單稽核")

with st.container(border=True):
    uploaded_files = st.file_uploader("📂 新增頁面", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True, key=f"uploader_{st.session_state.uploader_key}")
    if uploaded_files:
        for f in uploaded_files: 
            st.session_state.photo_gallery.append({
                'file': f, 
                'table_md': None, 
                'header_text': None,
                'full_text': None
            })
        st.session_state.uploader_key += 1
        components.html("""<script>window.parent.document.body.scrollTo(0, window.parent.document.body.scrollHeight);</script>""", height=0)
        st.rerun()

if st.session_state.photo_gallery:
    st.caption(f"已累積 {len(st.session_state.photo_gallery)} 頁文件")
    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1: start_btn = st.button("🚀 開始分析", type="primary", use_container_width=True)
    with col_btn2: 
        st.write("")
        clear_btn = st.button("🗑️照片清除", help="清除", use_container_width=True)

    if clear_btn:
        st.session_state.photo_gallery = []
        st.rerun()

    if start_btn:
        total_start = time.time()
        status = st.empty()
        progress_bar = st.progress(0)
        
        # 1. OCR (含快取機制)
        extracted_data_list = []
        full_text_for_search = ""
        total_imgs = len(st.session_state.photo_gallery)
        
        ocr_start = time.time()
        
        for i, item in enumerate(st.session_state.photo_gallery):
            img_file = item['file']
            
            if item['table_md'] and item['header_text'] and item.get('full_text'):
                status.text(f"讀取第 {i+1} 頁快取資料...")
                extracted_data_list.append({"page": i + 1, "table": item['table_md'], "header_text": item['header_text']})
                full_text_for_search += item['full_text']
                time.sleep(0.1) 
            else:
                status.text(f"Azure 正在掃描第 {i+1}/{total_imgs} 頁...")
                img_file.seek(0)
                try:
                    table_md, header_snippet, full_content = extract_layout_with_azure(img_file, DOC_ENDPOINT, DOC_KEY)
                    item['table_md'] = table_md
                    item['header_text'] = header_snippet
                    item['full_text'] = full_content
                    extracted_data_list.append({"page": i + 1, "table": table_md, "header_text": header_snippet})
                    full_text_for_search += full_content
                except Exception as e:
                    st.error(f"第 {i+1} 頁讀取失敗: {e}")
            
            progress_bar.progress((i + 1) / (total_imgs + 1))
        
        ocr_end = time.time()
        ocr_duration = ocr_end - ocr_start

        # 2. Gemini 雙軌計時
        combined_input = "以下是各頁資料：\n"
        for data in extracted_data_list:
            combined_input += f"\n=== Page {data['page']} ===\n【頁首】:\n{data['header_text']}\n【表格】:\n{data['table']}\n"

        status.text("Gemini 雙代理人正在平行稽核 (工程師 & 會計師)...")
        
        def run_with_timer(func, *args):
            t0 = time.time()
            res = func(*args)
            t1 = time.time()
            return res, t1 - t0

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # 傳入選定的模型名稱 (eng_model_name 和 acc_model_name)
            future_eng = executor.submit(run_with_timer, agent_engineer_check, combined_input, full_text_for_search, GEMINI_KEY, eng_model_name)
            future_acc = executor.submit(run_with_timer, agent_accountant_check, combined_input, GEMINI_KEY, acc_model_name)
            
            res_eng, time_eng = future_eng.result()
            res_acc, time_acc = future_acc.result()
        
        progress_bar.progress(100)
        status.text("完成！")
        
        total_end = time.time()
        total_duration = total_end - total_start
        
        # 3. 合併結果
        job_no = res_acc.get("job_no", "Unknown")
        issues_eng = res_eng.get("issues", [])
        issues_acc = res_acc.get("issues", [])
        all_issues = issues_eng + issues_acc

        st.success(f"工令: {job_no} | ⏱️ 總耗時: {total_duration:.1f}s")
        st.caption(f"細節耗時: Azure OCR {ocr_duration:.1f}s | 工程師 ({eng_selection}) {time_eng:.1f}s | 會計師 ({acc_selection}) {time_acc:.1f}s")
        
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
    for idx, item in enumerate(st.session_state.photo_gallery):
        with cols[idx % 4]:
            st.image(item['file'], caption=f"P.{idx+1}", use_container_width=True)
            if st.button("❌", key=f"del_{idx}"):
                st.session_state.photo_gallery.pop(idx)
                st.rerun()
else:
    st.info("👆 請點擊上方按鈕開始新增照片")
