import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
import google.generativeai as genai
import json
import time  # <--- 新增時間模組

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

    ### 0. 核心任務與數據清洗：
    - **識別滾輪編號 (Roll ID)**：找出每筆數據對應的編號 (如 `Y5612001`)。
    - **頁碼追蹤**：異常若涉及跨頁，請列出頁碼 (如 "1, 2")。
    - **數值容錯**：忽略數字間的空格 (如 `341 . 12` -> `341.12`)。

    ### 1. 跨頁一致性檢查 (Header Consistency)：
    - **檢查項目**：工令編號、預定交貨日期、實際交貨日期。
    - **規則**：所有頁面的上述欄位內容必須「完全相同」。
    - **日期格式**：`YYY.MM.DD` (允許空格)，`/` 或 `-` 為 FAIL。

    ### 2. 製程判定邏輯 (分軌制)：

    #### A. 【本體 (Body)】未再生/車修：
    - **規格解析**：忽略「每次車修Xmm」，只看「至 Ymm」。取最大值為 Max_Spec。
    - **邏輯分流**：
      1. **整數** (未完工)：實測值 **<=** 規格值。
      2. **小數** (已完工)：
         - 規則：實測值 **>=** 規格值。
         - 格式：忽略空格後，必須精確到小數點後兩位 (`#.##`)。
         - **標記**：此編號狀態為「本體已完工」。

    #### B. 【軸頸 (Journal)】未再生/車修：
    - 規格比對：智慧歸類 (多重規格)。
    - **強制規則**：實測值必須為 **整數**。
    - 若出現小數點 -> **FAIL** (軸頸未再生不可完工)。

    #### C. 銲補 (Welding) (通用)：
    - 規則：實測值 **>=** 規格。

    #### D. 再生車修 (Finish) (通用)：
    - 數值：**包含於 (Inclusive)** 上下限之間。
    - 格式：忽略空格後，必須精確到小數點後兩位。

    ### 3. 全域流程防呆 (Global Process Integrity)：
    - **前向檢查**：若某編號在「本體未再生」已標記為「已完工」，則 **不可出現** 在「本體銲補」或「本體再生車修」。
    - **後向檢查**：若某編號出現在「銲補」或「再生車修」，則 **必須出現** 在該部位的「未再生」階段。
    - **尺寸合理性**：檢查同一編號在各階段尺寸是否劇烈跳動 (如 350 -> 200 -> FAIL)。

    ### 4. 數量一致性檢查：
    - 讀取 `(10PC)` -> 清點實測個數。
    - 若 `實測個數 ≠ 要求個數` -> **FAIL (數量不符)**。
    - 例外：「熱處理」忽略數量。

    ### 輸出格式 (JSON Only)：
    {
      "job_no": "工令編號",
      "summary": "總結",
      "issues": [
         {
           "page": "頁碼",
           "item": "項目名稱",
           "issue_type": "數值超規 / 數量不符 / 流程異常 / 尺寸異常 / 格式錯誤 / 日期格式錯誤",
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
    
    if st.button("🚀 開始分析", type="primary", use_container_width=True):
        
        # --- 計時開始 ---
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
        
        # --- 計時結束 ---
        end_time = time.time()
        elapsed_time = end_time - start_time

        # 3. 顯示結果 (含計時)
        try:
            result = json.loads(result_str)
            if isinstance(result, list): result = result[0] if len(result) > 0 else {}
            
            # 在成功訊息旁邊顯示耗時
            st.success(f"工令: {result.get('job_no', 'Unknown')} | ⏱️ 耗時: {elapsed_time:.1f} 秒")
            
            issues = result.get('issues', [])
            if not issues:
                st.balloons()
                st.info("✅ 全數合格！數據邏輯與流程皆無異常。")
            else:
                st.error(f"發現 {len(issues)} 類異常項目")
                
                for item in issues:
                    with st.container(border=True):
                        # 標題
                        col_head1, col_head2 = st.columns([3, 1])
                        page_str = str(item.get('page', '?'))
                        col_head1.markdown(f"**P.{page_str} | {item.get('item')}**")
                        
                        itype = item.get('issue_type', '異常')
                        if "流程" in itype or "尺寸" in itype or "日期" in itype:
                            col_head2.error(f"🛑 {itype}")
                        else:
                            col_head2.warning(f"⚠️ {itype}")
                        
                        st.caption(f"原因: {item.get('common_reason')}")
                        if item.get('spec_logic'):
                            st.caption(f"標準: {item.get('spec_logic')}")
                        
                        # 明細表格
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
            
    if st.button("🗑️ 清除所有照片"):
        st.session_state.photo_gallery = []
        st.rerun()

else:
    st.info("👆 請點擊上方按鈕開始新增照片")
