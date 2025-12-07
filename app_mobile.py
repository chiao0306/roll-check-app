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

# --- 3. 初始化 Session State ---
if 'photo_gallery' not in st.session_state:
    st.session_state.photo_gallery = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- 4. 核心函數：Azure 神之眼 ---
def extract_layout_with_azure(file_obj, endpoint, key):
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    file_content = file_obj.getvalue()
    
    # 呼叫 Azure
    poller = client.begin_analyze_document(
        "prebuilt-layout", 
        file_content,
        content_type="application/octet-stream"
    )
    result: AnalyzeResult = poller.result()
    
    markdown_output = ""
    # A. 提取表格
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
    
    # B. 提取表頭 (前 300 字)
    header_snippet = result.content[:300] if result.content else ""
    
    return markdown_output, header_snippet

# --- 5. 核心函數：Gemini 神之腦 (歸類邏輯更新) ---
def audit_with_gemini(extracted_data_list, api_key):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-2.5-pro")
    
    combined_input = "以下是各頁資料：\n"
    for data in extracted_data_list:
        combined_input += f"\n--- Page {data['page']} ---\n"
        combined_input += f"【頁首文字片段】:\n{data['header_text']}\n"
        combined_input += f"【表格數據】:\n{data['table']}\n"

    system_prompt = """
    你是一位極度嚴謹的中鋼機械品管稽核員。
    請依據 Azure OCR 提取的表格文字進行稽核。

    ### 0. 核心任務：找出異常並「依類型歸類」
    - **識別滾輪編號 (Roll ID)**：在每一列數據中，請找出該數據對應的「滾輪編號」（通常在數據左側，格式如 `Y5612001` 或 `V103`）。
    - **分組回報**：若同一個項目有多個滾輪發生**相同類型的錯誤**，請將它們合併為一筆異常紀錄，並列出所有出問題的編號與數值。
    - **不要自動修正**：`129.` 就是 `129.`，請忠實呈現。

    ### 1. 跨頁一致性與格式檢查：
    - 工令編號、預定/實際交貨日期：所有頁面必須相同。
    - 日期格式：`YYY.MM.DD` (允許空格)，`/` 或 `-` 為 FAIL。

    ### 2. 製程判定邏輯 (Process Logic)：
    - **未再生/車修**：實測值 <= 規格。
    - **銲補 (Welding)**：實測值 >= 規格。
    - **再生車修**：
       - **數值**：實測值必須包含於上下限之間。
       - **格式**：必須精確到小數點後兩位 (如 `101.60` PASS, `101.6` FAIL)。

    ### 3. 數量一致性檢查：
    - 讀取項目名稱中的 `(10PC)` -> 清點該列實測數據個數。
    - 若個數不足 -> FAIL (數量不符)。
    - **例外**：「熱處理」忽略數量。

    ### 4. 多重規格智慧歸類：
    - 若一項有多個規格 (如 157mm, 127mm)，請自動歸類比對。

    ### 輸出格式 (JSON Only) - 【請嚴格遵守嵌套結構】：
    {
      "job_no": "工令編號",
      "summary": "總結",
      "issues": [
         {
           "page": 1,
           "item": "項目名稱",
           "issue_type": "數值超規 / 數量不符 / 小數點位數錯誤 / 日期錯誤",
           "spec_logic": "判定標準 (例如: 需 >= 163)",
           "common_reason": "錯誤原因概述 (例如: 實測值均小於下限)",
           "failures": [
              {"id": "Y5612001", "val": "136"},
              {"id": "Y5612002", "val": "136"}
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
    
    if st.button("🚀 開始分析 (歸類整合版)", type="primary", use_container_width=True):
        
        status = st.empty()
        progress_bar = st.progress(0)
        
        # 1. OCR
        extracted_data_list = []
        total_imgs = len(st.session_state.photo_gallery)
        
        for i, img in enumerate(st.session_state.photo_gallery):
            status.text(f"Azure 正在掃描第 {i+1}/{total_imgs} 頁...")
            img.seek(0)
            try:
                table_md, raw_txt = extract_layout_with_azure(img, DOC_ENDPOINT, DOC_KEY)
                extracted_data_list.append({
                    "page": i + 1,
                    "table": table_md,
                    "header_text": raw_txt 
                })
            except Exception as e:
                st.error(f"第 {i+1} 頁讀取失敗: {e}")
            progress_bar.progress((i + 1) / (total_imgs + 1))

        # 2. Gemini
        status.text(f"Gemini 2.5 Pro 正在進行邏輯歸類...")
        result_str = audit_with_gemini(extracted_data_list, GEMINI_KEY)
        
        progress_bar.progress(100)
        status.text("完成！")

        # 3. 顯示結果 (UI 升級：顯示詳細歸類表格)
        try:
            result = json.loads(result_str)
            if isinstance(result, list): result = result[0] if len(result) > 0 else {}
            
            st.success(f"工令: {result.get('job_no', 'Unknown')}")
            
            issues = result.get('issues', [])
            if not issues:
                st.balloons()
                st.success("✅ 全數合格！")
            else:
                st.error(f"發現 {len(issues)} 類異常項目")
                
                # 遍歷每一個「異常群組」
                for item in issues:
                    with st.container(border=True):
                        # 標題列：項目名稱 + 異常類型
                        col_head1, col_head2 = st.columns([3, 1])
                        col_head1.markdown(f"**{item.get('item')}**")
                        col_head2.error(f"{item.get('issue_type')}")
                        
                        # 說明列：原因 + 標準
                        st.caption(f"⚠️ {item.get('common_reason')}")
                        st.caption(f"📏 標準: {item.get('spec_logic')}")
                        
                        # 詳細清單 (如果有多支滾輪，列出表格)
                        failures = item.get('failures', [])
                        if failures:
                            st.write("🔻 **異常明細：**")
                            # 簡單表格呈現
                            # CSS hack 讓表格緊湊一點
                            table_data = [{"滾輪編號": f.get('id', '未知'), "實測值": f.get('val', 'N/A')} for f in failures]
                            st.dataframe(table_data, use_container_width=True, hide_index=True)
                        else:
                            # 像是數量不符這種，可能沒有個別 ID，就顯示 measured
                            st.text(f"實測數據: {item.get('measured', 'N/A')}")
                            
        except Exception as e:
            st.error("分析錯誤")
            st.code(result_str)
            st.write(e)
            
    if st.button("🗑️ 清除所有照片"):
        st.session_state.photo_gallery = []
        st.rerun()

else:
    st.info("👆 請點擊上方按鈕開始新增照片")
