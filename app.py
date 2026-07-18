import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import scipy.signal as signal
import matplotlib.pyplot as plt
from google import genai
from streamlit_autorefresh import st_autorefresh

# --- 1. การตั้งค่าหน้าเว็บและการรีเฟรชอัตโนมัติ ---
st.set_page_config(page_title="SmartVibe Layer Analysis", layout="wide", page_icon="🏢")

# รีเฟรชหน้าเว็บทุกๆ 3000 มิลลิวินาที (3 วินาที) เพื่อเปิดทางให้ Gemini ประมวลผลได้ทัน
st_autorefresh(interval=3000, limit=None, key="smartvibe_autorefresh")

# --- 2. การจัดการคีย์และการตั้งค่า Secrets ---
# เรียกใช้ข้อมูลคีย์ต่างๆ จากระบบตระกร้าความลับของ Streamlit (Secrets)
try:
    FIREBASE_URL = st.secrets["FIREBASE_URL"]
    TELEGRAM_BOT_TOKEN = st.secrets["TELEGRAM_BOT_TOKEN"]
    CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")  # ใส่ ID กลุ่ม/แชท Telegram ของคุณ
    GEMINI_API_KEY = st.secrets["AQ.Ab8RN6IznGVhDIaboHs6p6bCJaFh8Bx9CQFsxGnOTvw-wF0dAQ"]
except Exception:
    # Fallback เผื่อทดสอบในเครื่องและยังไม่ได้ตั้งค่า Secrets
    FIREBASE_URL = "https://example-default.firebaseio.com/.json"
    TELEGRAM_BOT_TOKEN = "MOCK_TOKEN"
    CHAT_ID = "MOCK_ID"
    GEMINI_API_KEY = "MOCK_KEY"

# --- 3. การเริ่มต้นระบบตัวแปรคงที่ (Session State) ---
if "baseline_locked" not in st.session_state:
    st.session_state.baseline_locked = False
if "baseline_rms" not in st.session_state:
    st.session_state.baseline_rms = {"Floor 1": 0.03, "Floor 2": 0.03, "Floor 3": 0.03}
if "last_status" not in st.session_state:
    st.session_state.last_status = "Green"

# --- 4. ฟังก์ชันสำหรับฟีเจอร์ต่างๆ ของระบบ ---

def send_telegram_alert(status, message):
    """ฟังก์ชันส่งการแจ้งเตือนเข้า Telegram ในกรณีสถานะเปลี่ยนแปลงเพื่อป้องกัน Spam"""
    if TELEGRAM_BOT_TOKEN == "MOCK_TOKEN" or not CHAT_ID:
        return
    
    emoji = "🟢" if status == "Green" else "🟡" if status == "Yellow" else "🔴"
    full_message = f"{emoji} [SmartVibe Alert]\nสถานะปัจจุบัน: {status}\nรายละเอียด: {message}"
    url = f"https://api.telegram.com/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": CHAT_ID, "text": full_message}, timeout=3)
    except Exception:
        pass

def calculate_welch_fft(signal_data, fs=100):
    """คำนวณความถี่เด่นและ Band Power รอบๆ ย่านวิกฤต 8.5 Hz ด้วย Welch's Method"""
    f, Pxx = signal.welch(signal_data, fs=fs, nperseg=128)
    peak_freq = f[np.argmax(Pxx)]
    
    # คำนวณ Band Power ย่าน 8.0 - 9.0 Hz (รอบๆ 8.5 Hz ที่เป็นความถี่เป้าหมาย)
    idx_band = (f >= 8.0) & (f <= 9.0)
    
    # 🔥 แก้ไขตรงนี้: เปลี่ยนจาก np.trapz เป็น np.trapezoid
    band_power = np.trapezoid(Pxx[idx_band], f[idx_band]) if np.any(idx_band) else 0.0
    
    return peak_freq, band_power, f, Pxx
    
# --- 5. แถบควบคุมด้านข้าง (Sidebar) & โหมดทดสอบระบบ ---
st.sidebar.title("🏢 SmartVibe Controller")
st.sidebar.write("ระบบวิเคราะห์ความสั่นสะเทือนโครงสร้าง 3 ชั้น")

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ โหมดทดสอบระบบ")
sim_mode = st.sidebar.checkbox("เปิดใช้งานระบบจำลองข้อมูล (Mock Data)", value=True)

# ปุ่มสำหรับล็อกค่า Baseline โครงสร้างอาคาร
st.sidebar.subheader("⚙️ การจัดการโครงสร้างอ้างอิง")
if st.sidebar.button("🔒 ล็อกค่า Baseline ปัจจุบัน"):
    st.session_state.baseline_locked = True
    st.sidebar.success("บันทึกค่าอ้างอิงของโครงสร้างเรียบร้อยแล้ว!")

if st.sidebar.button("🔓 รีเซ็ต Baseline"):
    st.session_state.baseline_locked = False
    st.sidebar.info("รีเซ็ตกลับไปใช้ค่าอ้างอิงมาตรฐาน")

# --- 6. กลไกจัดการข้อมูล (Firebase VS Simulation) ---
fs = 100  # อัตราการสุ่มตัวอย่าง 100 Hz
t = np.linspace(0, 5, 500)

if sim_mode:
    # ⏳ คำนวณลูปเวลาการสลับช่วงละ 15 วินาที (สลับ 3 สถานะ: 0, 1, 2 วนลูป)
    current_time_slot = int(time.time() // 15) % 3
    time_left = 15 - (int(time.time()) % 15)
    
    st.sidebar.info(f"🔄 โหมดจำลองจะเปลี่ยนสถานะในอีก {time_left} วินาที")
    
    if current_time_slot == 0:
        # 🟢 ช่วงสถานะ: ปกติ (Normal)
        current_global_status = "Green"
        status_text = "ปกติ (Normal)"
        f1_p, f2_p, f3_p = 2.5, 3.1, 1.8
        f1_amp, f2_amp, f3_amp = 0.01, 0.012, 0.009
        noise_amp = 0.015
    elif current_time_slot == 1:
        # 🟡 ช่วงสถานะ: เฝ้าระวัง (Warning)
        current_global_status = "Yellow"
        status_text = "เฝ้าระวัง (Warning)"
        f1_p, f2_p, f3_p = 6.2, 5.8, 7.0
        f1_amp, f2_amp, f3_amp = 0.18, 0.22, 0.15
        noise_amp = 0.04
    else:
        # 🔴 ช่วงสถานะ: ไม่ปลอดภัย (Danger - วิ่งเข้าหาเรโซแนนซ์ 8.5 Hz)
        current_global_status = "Red"
        status_text = "ไม่ปลอดภัย (Danger - ใกล้เคียง 8.5 Hz!)"
        f1_p, f2_p, f3_p = 8.45, 8.52, 8.48
        f1_amp, f2_amp, f3_amp = 0.65, 0.75, 0.60
        noise_amp = 0.08

    st.sidebar.markdown(f"**สถานะจำลองปัจจุบัน:** `{status_text}`")
    
    # สร้างสัญญาณสั่นสะเทือนจำลองทั้ง 3 ชั้น
    data_f1 = (f1_amp * np.sin(2 * np.pi * f1_p * t)) + np.random.normal(0, noise_amp, 500)
    data_f2 = (f2_amp * np.sin(2 * np.pi * f2_p * t)) + np.random.normal(0, noise_amp, 500)
    data_f3 = (f3_amp * np.sin(2 * np.pi * f3_p * t)) + np.random.normal(0, noise_amp, 500)
    
    watchdog_trigger = False  # โหมดจำลองไม่มีการค้างของเซ็นเซอร์

else:
    # 🌐 ดึงข้อมูลจริงจาก Firebase Realtime Database
    try:
        response = requests.get(FIREBASE_URL, timeout=2.5)
        fb_data = response.json()
        
        # สมมติโครงสร้างข้อมูลใน Firebase คือคีย์ย้อนหลัง 500 จุด
        # ในที่นี้ทำการแปลงหรือแตกค่าออกมาเป็นอาเรย์
        data_f1 = np.array(fb_data.get("Floor1_AccX", np.zeros(500)))
        data_f2 = np.array(fb_data.get("Floor2_AccX", np.zeros(500)))
        data_f3 = np.array(fb_data.get("Floor3_AccX", np.zeros(500)))
        
        # ระบบ Watchdog ตรวจจับค่าค้าง (ดึง Uptime ล่าสุดมาเทียบ)
        last_timestamp = fb_data.get("timestamp", time.time())
        if time.time() - last_timestamp > 7:  # เกิน 7 วินาทีไม่มีค่าใหม่ขยับ
            watchdog_trigger = True
        else:
            watchdog_trigger = False
            
    except Exception:
        st.error("❌ ไม่สามารถเชื่อมต่อกับ Firebase ได้ กำลังแสดงค่าว่างเพื่อความปลอดภัย")
        data_f1, data_f2, data_f3 = np.zeros(500), np.zeros(500), np.zeros(500)
        watchdog_trigger = True

# --- 7. การประมวลผลข้อมูลฟิสิกส์และการคำนวณดัชนีสุขภาพ (Health %) ---
# คำนวณค่า RMS ของแต่ละชั้น
rms_f1 = np.sqrt(np.mean(data_f1**2))
rms_f2 = np.sqrt(np.mean(data_f2**2))
rms_f3 = np.sqrt(np.mean(data_f3**2))

# ประมวลผล FFT ของแต่ละชั้น
peak_f1, bp_f1, freq_axis, psd_f1 = calculate_welch_fft(data_f1, fs)
peak_f2, bp_f2, _, psd_f2 = calculate_welch_fft(data_f2, fs)
peak_f3, bp_f3, _, psd_f3 = calculate_welch_fft(data_f3, fs)

# กำหนดระดับการคำนวณ Health Index (คำนวณโดยอิงจาก Baseline หรือระดับความรุนแรง)
if st.session_state.baseline_locked:
    # หากล็อก Baseline ไว้ จะเทียบความเสื่อมสภาพตามอัตราการโตของ RMS ยิ่ง RMS โต สุขภาพยิ่งแย่
    health_f1 = max(0.0, min(100.0, 100.0 - ((rms_f1 - st.session_state.baseline_rms["Floor 1"]) * 200)))
    health_f2 = max(0.0, min(100.0, 100.0 - ((rms_f2 - st.session_state.baseline_rms["Floor 2"]) * 200)))
    health_f3 = max(0.0, min(100.0, 100.0 - ((rms_f3 - st.session_state.baseline_rms["Floor 3"]) * 200)))
else:
    # หากไม่ได้ล็อก จะคำนวณโดยตรงจากอัตราความเข้าใกล้ระดับสั่นพ้อง 8.5 Hz
    health_f1 = max(0.0, min(100.0, 100.0 - (bp_f1 * 400)))
    health_f2 = max(0.0, min(100.0, 100.0 - (bp_f2 * 400)))
    health_f3 = max(0.0, min(100.0, 100.0 - (bp_f3 * 400)))

# คำนวณค่าเฉลี่ยสุขภาพรวมอาคารและกำหนดสถานะภาพรวม
avg_health = (health_f1 + health_f2 + health_f3) / 3
if not sim_mode:
    if watchdog_trigger:
        current_global_status = "Red"
    elif avg_health < 60:
        current_global_status = "Red"
    elif avg_health < 85:
        current_global_status = "Yellow"
    else:
        current_global_status = "Green"

# จัดการยิงการแจ้งเตือนเข้า Telegram เมื่อสถานะเปลี่ยนไปจากรอบก่อนหน้าเท่านั้น
if current_global_status != st.session_state.last_status:
    send_telegram_alert(current_global_status, f"สุขภาพเฉลี่ยของโครงสร้างอาคารเปลี่ยนเป็น {avg_health:.2f}%")
    st.session_state.last_status = current_global_status

# --- 8. ส่วนจัดวาง Layout หน้าจอแสดงผลหลัก (Dashboard UI) ---
st.title("📊 แดชบอร์ดตรวจสอบสุขภาพโครงสร้างอาคารแบบเรียลไทม์")

# แสดงการเตือนของระบบความปลอดภัยสูงสุด
if watchdog_trigger:
    st.critical("🚨 [WATCHDOG ALERT] ข้อมูลจากเซ็นเซอร์ ESP32 ขาดการติดต่อหรือเกิดอาการค้าง! โปรดตรวจสอบฮาร์ดแวร์")

# แบ่งคอลัมน์แสดงผลภาพรวมสถานะ 3 ชั้น
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="🏢 สุขภาพชั้นที่ 1 (Floor 1)", value=f"{health_f1:.2f} %", delta=f"Peak: {peak_f1:.2f} Hz")
    st.caption(f"RMS: {rms_f1:.4f} g")

with col2:
    st.metric(label="🏢 สุขภาพชั้นที่ 2 (Floor 2)", value=f"{health_f2:.2f} %", delta=f"Peak: {peak_f2:.2f} Hz")
    st.caption(f"RMS: {rms_f2:.4f} g")

with col3:
    st.metric(label="🏢 สุขภาพชั้นที่ 3 (Floor 3)", value=f"{health_f3:.2f} %", delta=f"Peak: {peak_f3:.2f} Hz")
    st.caption(f"RMS: {rms_f3:.4f} g")

st.markdown("---")

# ส่วนแสดงกราฟคลื่นความสั่นสะเทือนตามเวลา (Time Domain) และการแจกแจงความถี่ (Frequency Domain)
g_col1, g_col2 = st.columns(2)

with g_col1:
    st.subheader("📈 คลื่นความเร่งสั่นสะเทือน (Time Domain)")
    df_time = pd.DataFrame({"Floor 1": data_f1, "Floor 2": data_f2, "Floor 3": data_f3}, index=t)
    st.line_chart(df_time, height=280)

with g_col2:
    st.subheader("📊 การวิเคราะห์ความหนาแน่นพลังงานความถี่ (Welch FFT)")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(freq_axis, psd_f1, label="Floor 1", alpha=0.8)
    ax.plot(freq_axis, psd_f2, label="Floor 2", alpha=0.8)
    ax.plot(freq_axis, psd_f3, label="Floor 3", alpha=0.8)
    ax.axvline(8.5, color="red", linestyle="--", label="🎯 Target Freq (8.5 Hz)")
    ax.set_xlim(0, 20)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Density")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")

# --- 9. ระบบวิเคราะห์ความปลอดภัยเชิงลึกด้วย AI (Gemini Integration) ---
st.subheader("🧠 ระบบวิเคราะห์ความปลอดภัยเชิงลึกด้วย AI")
st.write("กดปุ่มด้านล่างเพื่อส่งข้อมูลสถิติฟิสิกส์ทั้งหมดจากเซ็นเซอร์ไปให้ AI ประมวลผลและออกบทวิเคราะห์สภาพสิ่งก่อสร้าง")

if st.button("✨ ให้ Gemini วิเคราะห์สถานะอาคารตอนนี้"):
    if GEMINI_API_KEY == "MOCK_KEY" or not GEMINI_API_KEY:
        st.warning("⚠️ ไม่พบข้อมูล `GEMINI_API_KEY` ในระบบ Secrets กรุณาตรวจสอบการตั้งค่าหลังบ้านก่อนใช้งานครับ")
    else:
        with st.spinner("🤖 กำลังรวบรวมข้อมูลเซ็นเซอร์และให้ Gemini เขียนบทวิเคราะห์วิศวกรรม..."):
            try:
                # การเขียนเรียบเรียง Prompt เชิงข้อมูลดิบส่งเข้าสมองกล AI
                prompt_input = f"""
                คุณคือวิศวกรโครงสร้างผู้เชี่ยวชาญด้านระบบ Structural Health Monitoring (SHM) 
                นี่คือข้อมูลที่วัดได้จากเซ็นเซอร์ความสั่นสะเทือน 3 ชั้นของสิ่งก่อสร้าง ณ วินาทีนี้:
                
                [ภาพรวมระบบ]
                - โหมดการทำงาน: {"โหมดจำลองสถานการณ์" if sim_mode else "ข้อมูลสดจากอุปกรณ์จริง"}
                - สถานะความปลอดภัยรวม: {current_global_status}
                - ค่าเฉลี่ยสุขภาพอาคาร: {avg_health:.2f}%
                
                [ข้อมูลดิบรายชั้น]
                1. ชั้นที่ 1:
                   - ค่าความถี่สูงสุดเด่น (Peak Frequency): {peak_f1:.2f} Hz
                   - ค่าพลังงานสั่นสะเทือนรวม (RMS): {rms_f1:.4f} g
                   - ดัชนีความแข็งแรง (Health Index): {health_f1:.2f}%
                2. ชั้นที่ 2:
                   - ค่าความถี่สูงสุดเด่น (Peak Frequency): {peak_f2:.2f} Hz
                   - ค่าพลังงานสั่นสะเทือนรวม (RMS): {rms_f2:.4f} g
                   - ดัชนีความแข็งแรง (Health Index): {health_f2:.2f}%
                3. ชั้นที่ 3:
                   - ค่าความถี่สูงสุดเด่น (Peak Frequency): {peak_f3:.2f} Hz
                   - ค่าพลังงานสั่นสะเทือนรวม (RMS): {rms_f3:.4f} g
                   - ดัชนีความแข็งแรง (Health Index): {health_f3:.2f}%
                   
                *เป้าหมายที่ต้องเฝ้าระวังสูงสุด: หากความถี่เด่นเข้าใกล้ความถี่ธรรมชาติวิกฤตที่ย่าน 8.5 Hz โครงสร้างจะเสี่ยงเกิดการสั่นพ้อง (Resonance)
                
                โปรดช่วยสรุปรายงานการประเมินความปลอดภัยโดยสรุป โดยระบุ:
                1. การวิเคราะห์สถานะปัจจุบันว่ามีความเสี่ยงในแง่ของฟิสิกส์ความสั่นสะเทือนและการเกิด Resonance หรือไม่?
                2. เปรียบเทียบอาการระหว่างชั้นทั้ง 3 ว่าชั้นไหนมีความเครียดสะสมสูงที่สุด?
                3. คำแนะนำเชิงวิศวกรรมที่ควรดำเนินการเร่งด่วนสำหรับผู้ดูแลตึก
                ตอบกลับเป็นภาษาไทย กระชับ ได้ใจความเชิงเทคนิควิศวกรรม
                """
                
                # เรียกใช้งานผ่านโครงสร้างใหม่ของคลังสินค้า google-genai SDK 
                client = genai.Client(api_key=GEMINI_API_KEY)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt_input,
                )
                
                # แสดงผลลัพธ์ที่เขียนส่งคืนกลับมาจากทาง AI
                st.success("📝 บทวิเคราะห์สภาพโครงสร้างโดย Gemini AI:")
                st.info(response.text)
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการสื่อสารกับ AI: {e}")
