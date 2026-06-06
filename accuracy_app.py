import streamlit as st
import pandas as pd
import os
from evaluate_accuracy import evaluate_routing_accuracy, save_accuracy_log

st.set_page_config(page_title="Routing Accuracy Validator", layout="wide", page_icon="📊")

st.title("📊 Routing Accuracy Validator (ระบบตรวจสอบความแม่นยำการจัดทริป)")
st.markdown("อัปโหลดไฟล์ Excel หรือ CSV ของ **ข้อมูลที่จัดจริง (Actual Data)** และ **ข้อมูลจากระบบ (System Data)** เพื่อเปรียบเทียบความแม่นยำของการจัดเส้นทางและการเลือกรถ")

# ---- ประวัติการทดสอบ (History) ----
history_file = 'accuracy_history.csv'
if os.path.exists(history_file):
    with st.expander("📈 ดูประวัติการพัฒนา (Accuracy Improvement History)", expanded=False):
        df_history = pd.read_csv(history_file)
        if not df_history.empty:
            st.dataframe(df_history, use_container_width=True)
            
            # วาดกราฟ
            st.markdown("#### แนวโน้มความแม่นยำ (Trend)")
            chart_data = df_history[['Timestamp', 'Vehicle_Accuracy_Pct', 'Avg_Routing_Similarity']].copy()
            chart_data.set_index('Timestamp', inplace=True)
            st.line_chart(chart_data)
            
            if st.button("🗑️ ล้างประวัติ (Clear History)"):
                os.remove(history_file)
                st.rerun()

# File Uploaders
col1, col2 = st.columns(2)

with col1:
    st.header("📂 1. ข้อมูลจัดจริง (Actual Data)")
    st.info("ไฟล์ประวัติการจัดรถของจริง เช่น จากหน้างานหรือระบบเดิม")
    actual_file = st.file_uploader("อัปโหลดไฟล์ Actual Data", type=['csv', 'xlsx'], key='actual')
    act_header_row = st.number_input("แถวที่เป็นชื่อคอลัมน์ (Actual)", min_value=1, value=1, help="ระบุหมายเลขแถวใน Excel ที่เป็นชื่อคอลัมน์ (เช่น แถว 3)")

with col2:
    st.header("💻 2. ข้อมูลจากระบบ (System Data)")
    st.info("ไฟล์ที่ส่งออกมาจากระบบ (Output ของโปรแกรม)")
    system_file = st.file_uploader("อัปโหลดไฟล์ System Data", type=['csv', 'xlsx'], key='system')
    sys_header_row = st.number_input("แถวที่เป็นชื่อคอลัมน์ (System)", min_value=1, value=1, help="ระบุหมายเลขแถวใน Excel ที่เป็นชื่อคอลัมน์")

if actual_file and system_file:
    try:
        # Load data
        with st.spinner("กำลังโหลดไฟล์ข้อมูล..."):
            act_header_idx = act_header_row - 1
            if actual_file.name.endswith('.csv'):
                actual_df = pd.read_csv(actual_file, header=act_header_idx)
            else:
                actual_df = pd.read_excel(actual_file, header=act_header_idx)
                
            sys_header_idx = sys_header_row - 1
            if system_file.name.endswith('.csv'):
                system_df = pd.read_csv(system_file, header=sys_header_idx)
            else:
                system_df = pd.read_excel(system_file, header=sys_header_idx)
                
        st.success("✅ โหลดไฟล์สำเร็จ! กรุณาจับคู่คอลัมน์ด้านล่างให้ตรงกับข้อมูลของคุณ")
        
        # Column matching
        st.markdown("---")
        st.subheader("⚙️ จับคู่คอลัมน์ (Column Mapping)")
        
        c1, c2 = st.columns(2)
        
        def find_best_match(columns, keywords):
            cols_lower = [str(c).lower() for c in columns]
            for kw in keywords:
                for idx, col in enumerate(cols_lower):
                    if kw in col:
                        return idx
            return 0

        with c1:
            st.markdown("#### คอลัมน์ของ Actual Data")
            act_cols = actual_df.columns.tolist()
            act_branch = st.selectbox("รหัสสาขา (Branch Code)", act_cols, index=find_best_match(act_cols, ['branch', 'สาขา', 'code', 'รหัส']), key='act_branch')
            act_trip = st.selectbox("เลขทริป (Trip ID)", act_cols, index=find_best_match(act_cols, ['trip', 'ทริป', 'route']), key='act_trip')
            act_vehicle = st.selectbox("ประเภทรถ (Vehicle Type)", act_cols, index=find_best_match(act_cols, ['vehicle', 'car', 'รถ', 'ประเภท']), key='act_veh')
            
        with c2:
            st.markdown("#### คอลัมน์ของ System Data")
            sys_cols = system_df.columns.tolist()
            sys_branch = st.selectbox("รหัสสาขา (Branch Code)", sys_cols, index=find_best_match(sys_cols, ['branch', 'สาขา', 'code', 'รหัส']), key='sys_branch')
            sys_trip = st.selectbox("เลขทริป (Trip ID)", sys_cols, index=find_best_match(sys_cols, ['trip', 'ทริป', 'route']), key='sys_trip')
            sys_vehicle = st.selectbox("ประเภทรถ (Vehicle Type)", sys_cols, index=find_best_match(sys_cols, ['vehicle', 'car', 'รถ', 'ประเภท']), key='sys_veh')
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 เริ่มการวิเคราะห์ (Analyze Accuracy)", type="primary", use_container_width=True):
            with st.spinner("กำลังเปรียบเทียบและวิเคราะห์ข้อมูล..."):
                metrics, details_df = evaluate_routing_accuracy(
                    actual_df, system_df,
                    act_branch_col=act_branch,
                    sys_branch_col=sys_branch,
                    actual_trip_col=act_trip,
                    actual_vehicle_col=act_vehicle,
                    sys_trip_col=sys_trip,
                    sys_vehicle_col=sys_vehicle
                )
                
                # Save the history log
                save_accuracy_log(metrics)
                
                st.markdown("---")
                st.header("🎯 ผลการประเมินความแม่นยำ (Accuracy Results)")
                
                # Metrics Row 1
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("จำนวนสาขาจริง (Actual Branches)", metrics['Total_Actual_Branches'])
                m2.metric("จำนวนสาขาในระบบ (System Branches)", metrics['Total_System_Branches'])
                m3.metric("จับคู่สำเร็จ (Matched)", metrics['Matched_Branches'], delta=f"ขาด {metrics['Missing_Branches']} / เกิน {metrics['Extra_Branches']}", delta_color="off")
                m4.metric("ความเหมือนของกลุ่มเพื่อนร่วมทริป (Routing Similarity)", f"{metrics['Avg_Routing_Similarity']}%")
                
                # Metrics Row 2
                st.markdown("<br>", unsafe_allow_html=True)
                m5, m6 = st.columns(2)
                m5.metric("🚚 เลือกรถถูกประเภท (Vehicle Match)", f"{metrics['Vehicle_Match_Count']} สาขา", delta=f"{metrics['Vehicle_Accuracy_Pct']}% ของสาขาที่จับคู่ได้", delta_color="normal")
                m6.info(f"**สรุปข้อมูล:** ระบบสามารถจับคู่สาขาได้ตรงกัน {metrics['Matched_Branches']} สาขา ขาดตกหล่นไป {metrics['Missing_Branches']} สาขา และเกินมา {metrics['Extra_Branches']} สาขา")
                
                st.markdown("---")
                st.subheader("🔍 วิเคราะห์เชิงลึกรายสาขา (Detailed Branch Comparison)")
                st.markdown("ตารางด้านล่างแสดงการเปรียบเทียบรายสาขา (อ้างอิงสาขาเป็นหลัก) ว่าแต่ละสาขาของจริงไปกับรถอะไร ทริปไหน และระบบจัดให้ไปกับรถอะไร ทริปไหน")
                
                # Color code match status
                def highlight_status(val):
                    color = 'green' if val == 'Matched' else 'red'
                    return f'color: {color}'
                
                styled_df = details_df.style.map(highlight_status, subset=['Status'])
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # CSV Download
                csv = details_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 ดาวน์โหลดผลวิเคราะห์ฉบับเต็ม (Download CSV)",
                    data=csv,
                    file_name='routing_accuracy_analysis.csv',
                    mime='text/csv',
                    use_container_width=True
                )
                
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการโหลดหรือประมวลผลข้อมูล: {str(e)}")
        st.info("กรุณาตรวจสอบว่าไฟล์มีคอลัมน์ที่จำเป็นครบถ้วน และข้อมูลในคอลัมน์ไม่เป็นค่าว่างทั้งหมด")
