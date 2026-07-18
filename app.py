import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import scipy.signal as signal
import matplotlib.pyplot as plt
from streamlit_autorefresh import st_autorefresh

# --- 1. การตั้งค่าหน้าเว็บและการรีเฟรชอัตโนมัติ ---
st.set_page_config(page_title="SmartVibe Layer Analysis", layout="wide", page_icon="🏢")

# รีเฟรชหน้าเว็บทุกๆ 3000 มิลลิวินาที (3 วินาที) เพื่อให้กราฟขยับและเว้นจังหวะให้ AI ประมวลผลได้ทัน
st_autorefresh(interval=3000, limit=None, key="smartvibe_autorefresh")

# --- 2. การจัดการคีย์และการตั้งค่าระบบหลังบ้าน ---
# ฝังคีย์ที่คุณส่งมาโดยตรงในโค้ด เพื่อให้ REST API เรียกใช้งานได้ทันที 
GEMINI_API_KEY = "GEMINI_API_KEY = "AIzaSy_Ab8RN6JKkNy4jBkY8yPTXUfOQcT44b8KwmxD6s6DeZlv5y1T-g"

# สำหรับค่าระบบอื่นๆ สามารถดึงผ่าน Secrets หรือใช้ค่าจำลอง (Mock) ไปก่อนได้ครับ
FIREBASE_URL = st.secrets.get("FIREBASE_URL", "https://smart-vibe-f944b-default-rtdb.asia-southeast1.firebasedatabase.app/SmartVibe/History3F.json")
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "MOCK_TOKEN")
CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "MOCK_ID")

# --- 3. การเริ่มต้นระบบตัวแปรคงที่ (Session State) ---
if "baseline_locked" not in st.session_state:
    st.session_state.baseline_locked = False
if "baseline_rms" not in st.session_state:
    st.session_state.baseline_rms = {"Floor 1": 0.03, "Floor 2": 0.03, "Floor 3": 0.03}
if "last_status" not in st.session_state:
    st.session_state.last_status = "Green"

# --- 4. ฟังก์ชันสำหรับฟีเจอร์ต่างๆ ของระบบ ---
def send_telegram_alert(status, message):
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
    
    # คำนวณ Band Power ย่าน 8.0 - 9.0 Hz รอบความถี่เป้าหมาย 8.5 Hz
    idx_band = (f >= 8.0) & (f <= 9.0)
    # ใช้ np.trapezoid เพื่อรองรับ NumPy 2.0+ บน Python เวอร์ชันใหม่บน Cloud
    band_power = np.trapezoid(Pxx[idx_band], f[idx_band]) if np.any(idx_band) else 0.0
    return peak_freq, band_power, f, Pxx

# --- 5. แถบควบคุมด้านข้าง (Sidebar) & โหมดทดสอบระบบ ---
st.sidebar.title("🏢 SmartVibe Controller")
st.sidebar.write("ระบบวิเคราะห์ความสั่นสะเทือนโครงสร้าง 3 ชั้น")

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ โหมดทดสอบระบบ")
sim_mode = st.sidebar.checkbox("เปิดใช้งานระบบจำลองข้อมูล (Mock Data)", value=True)

st.sidebar.subheader("⚙️ การจัดการโครงสร้างอ้างอิง")
if st.sidebar.button("🔒 ล็อกค่า Baseline ปัจจุบัน"):
    st.session_state.baseline_locked = True
    st.sidebar.success("บันทึกค่าอ้างอิงของโครงสร้างเรียบร้อยแล้ว!")

if st.sidebar.button("🔓 รีเซ็ต Baseline"):
    st.session_state.baseline_locked = False
    st.sidebar.info("รีเซ็ตกลับไปใช้ค่าอ้างอิงมาตรฐาน")

# --- 6. กลไกจัดการข้อมูล (Firebase VS Simulation) ---
fs = 100  
t = np.linspace(0, 5, 500)

if sim_mode:
    # ⏳ กลไกสลับช่วงสถานะละ 15 วินาที (0: ปกติ -> 1: เฝ้าระวัง -> 2: ไม่ปลอดภัย)
    current_time_slot = int(time.time() // 15) % 3
    time_left = 15 - (int(time.time()) % 15)
    
    st.sidebar.info(f"🔄 โหมดจำลองจะเปลี่ยนสถานะในอีก {time_left} วินาที")
    
    if current_time_slot == 0:
        current_global_status = "Green"
        status_text = "ปกติ (Normal)"
        f1_p, f2_p, f3_p = 2.5, 3.1, 1.8
        f1_amp, f2_amp, f3_amp = 0.01, 0.012, 0.009
        noise_amp = 0.015
    elif current_time_slot == 1:
        current_global_status = "Yellow"
        status_text = "เฝ้าระวัง (Warning)"
        f1_p, f2_p, f3_p = 6.2, 5.8, 7.0
        f1_amp, f2_amp, f3_amp = 0.18, 0.22, 0.15
        noise_amp = 0.04
    else:
        current_global_status = "Red"
        status_text = "ไม่ปลอดภัย (Danger - ใกล้เคียง 8.5 Hz!)"
        f1_p, f2_p, f3_p = 8.45, 8.52, 8.48
        f1_amp, f2_amp, f3_amp = 0.65, 0.75, 0.60
        noise_amp = 0.08

    st.sidebar.markdown(f"**สถานะจำลองปัจจุบัน:** `{status_text}`")
    
    data_f1 = (f1_amp * np.sin(2 * np.pi * f1_p * t)) + np.random.normal(0, noise_amp, 500)
    data_f2 = (f2_amp * np.sin(2 * np.pi * f2_p * t)) + np.random.normal(0, noise_amp, 500)
    data_f3 = (f3_amp * np.sin(2 * np.pi * f3_p * t)) + np.random.normal(0, noise_amp, 500)
    watchdog_trigger = False
else:
    try:
        response = requests.get(FIREBASE_URL, timeout=2.5)
        fb_data = response.json()
        data_f1 = np.array(fb_data.get("Floor1_AccX", np.zeros(500)))
        data_f2 = np.array(fb_data.get("Floor2_AccX", np.zeros(500)))
        data_f3 = np.array(fb_data.get("Floor3_AccX", np.zeros(500)))
        
        last_timestamp = fb_data.get("timestamp", time.time())
        watchdog_trigger = True if (time.time() - last_timestamp > 7) else False
    except Exception:
        st.error("❌ ไม่สามารถเชื่อมต่อกับ Firebase ได้ กำลังแสดงค่าว่างเพื่อความปลอดภัย")
        data_f1, data_f2, data_f3 = np.zeros(500), np.zeros(500), np.zeros(500)
        watchdog_trigger = True

# --- 7. การประมวลผลข้อมูลฟิสิกส์และการคำนวณดัชนีสุขภาพ (Health %) ---
rms_f1 = np.sqrt(np.mean(data_f1**2))
rms_f2 = np.sqrt(np.mean(data_f2**2))
rms_f3 = np.sqrt(np.mean(data_f3**2))

peak_f1, bp_f1, freq_axis, psd_f1 = calculate_welch_fft(data_f1, fs)
peak_f2, bp_f2, _, psd_f2 = calculate_welch_fft(data_f2, fs)
peak_f3, bp_f3, _, psd_f3 = calculate_welch_fft(data_f3, fs)

if st.session_state.baseline_locked:
    health_f1 = max(0.0, min(100.0, 100.0 - ((rms_f1 - st.session_state.baseline_rms["Floor 1"]) * 200)))
    health_f2 = max(0.0, min(100.0, 100.0 - ((rms_f2 - st.session_state.baseline_rms["Floor 2"]) * 200)))
    health_f3 = max(0.0, min(100.0, 100.0 - ((rms_f3 - st.session_state.baseline_rms["Floor 3"]) * 200)))
else:
    health_f1 = max(0.0, min(100.0, 100.0 - (bp_f1 * 400)))
    health_f2 = max(0.0, min(100.0, 100.0 - (bp_f2 * 400)))
    health_f3 = max(0.0, min(100.0, 100.0 - (bp_f3 * 400)))

avg_health = (health_f1 + health_f2 + health_f3) / 3
if not sim_mode:
    if watchdog_trigger: current_global_status = "Red"
    elif avg_health < 60: current_global_status = "Red"
    elif avg_health < 85: current_global_status = "Yellow"
    else: current_global_status = "Green"

if current_global_status != st.session_state.last_status:
    send_telegram_alert(current_global_status, f"สุขภาพเฉลี่ยของโครงสร้างอาคารเปลี่ยนเป็น {avg_health:.2f}%")
    st.session_state.last_status = current_global_status

# --- 8. ส่วนจัดวาง Layout หน้าจอแสดงผลหลัก (Dashboard UI) ---
st.title("📊 แดชบอร์ดตรวจสอบสุขภาพโครงสร้างอาคารแบบเรียลไทม์")

if watchdog_trigger:
    st.error("🚨 [WATCHDOG ALERT] ข้อมูลจากเซ็นเซอร์ ESP32 ขาดการติดต่อหรือเกิดอาการค้าง!")

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

# --- 9. ระบบวิเคราะห์ด้วย AI (ยิงตรงผ่าน REST API เพื่อแก้ปัญหาบั๊กสิทธิ์พาสบนคลาวด์) ---
st.subheader("🧠 ระบบวิเคราะห์ความปลอดภัยเชิงลึกด้วย AI")
st.write("กดปุ่มด้านล่างเพื่อส่งข้อมูลสถิติฟิสิกส์ทั้งหมดจากเซ็นเซอร์ไปให้ AI ประมวลผลและออกบทวิเคราะห์สภาพสิ่งก่อสร้าง")

if st.button("✨ ให้ Gemini วิเคราะห์สถานะอาคารตอนนี้"):
    with st.spinner("🤖 กำลังรวบรวมข้อมูลเซ็นเซอร์และให้ Gemini เขียนบทวิเคราะห์วิศวกรรม..."):
        try:
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
            
            # บังคับยิงตรงผ่าน REST API Endpoint เพื่อสลัดบั๊กการขอสิทธิ์ OAuth ของไลบรารีบนโครงสร้าง Google Cloud
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt_input}]
                    }
                ]
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                res_json = response.json()
                ai_text = res_json['candidates'][0]['content']['parts'][0]['text']
                st.success("📝 บทวิเคราะห์สภาพโครงสร้างโดย Gemini AI:")
                st.info(ai_text)
            else:
                st.error(f"❌ Gemini API Error ({response.status_code}): {response.text}")
                st.info("คำแนะนำ: หากยังพบข้อผิดพลาด รบกวนตรวจสอบอีกครั้งว่ามีอักขระหรือช่องว่างหลุดเข้าไปในตอนก๊อปปี้คีย์หรือไม่ครับ")
                
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดในการสื่อสารกับ AI: {e}")
