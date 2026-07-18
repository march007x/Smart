import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from scipy.signal import welch
from streamlit_autorefresh import st_autorefresh

# ==========================================================
# ⚙️ ส่วนตั้งค่าโปรเจกต์ และ Telegram Bot
# ==========================================================
FIREBASE_URL = 'https://smart-vibe-f944b-default-rtdb.asia-southeast1.firebasedatabase.app/SmartVibe/History3F.json'

STATE_URL = 'https://smart-vibe-f944b-default-rtdb.asia-southeast1.firebasedatabase.app/SmartVibe/State3F.json'

# --- ใส่ Token และ Chat ID ของคุณตรงนี้ ---
TELEGRAM_BOT_TOKEN = "8816324739:AAHZEKbjTyvLUORVd97t5kzFWy7pIxqFEhY"
TELEGRAM_CHAT_ID = "7360818672"

# --- 🔮 Gemini API (ใส่ API Key จาก https://aistudio.google.com/app/apikey ตรงนี้) ---
GEMINI_API_KEY = "AIzaSyB8Ouh0J6Vy6yrTuxrvL1oD_SSy0B3xTC0"  # <<< แก้บรรทัดนี้เป็น API Key จริงของคุณ
GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
# ==========================================================

st.set_page_config(page_title="SmartVibe Layer Analysis", layout="wide")
st.title("SmartVibe: ระบบวิเคราะห์ความสั่นสะเทือนแยก")

st_autorefresh(interval=850, limit=None, key="smartvibe_autorefresh")

# แก้ไขจุดที่ทำให้เกิด Error: ลบ ?auth={...} ออกทั้งหมด เพราะใช้ Test Mode
QUERY = '?orderBy="$key"&limitToLast=500'
STATE_QUERY = '' 

NOMINAL_FS = 50.0
FORCING_FREQ = 8.5
BAND_HZ = 1.5
HISTORY_SIZE = 7
MIN_CONSEC = 2

# ===== Session state =====
if 'http_session' not in st.session_state: st.session_state.http_session = requests.Session()
if 'last_uptime' not in st.session_state: st.session_state.last_uptime = 0
if 'stuck_counter' not in st.session_state: st.session_state.stuck_counter = 0
if 'gemini_result' not in st.session_state: st.session_state.gemini_result = None
if 'gemini_error' not in st.session_state: st.session_state.gemini_error = None

# ใช้ตรวจสอบเพื่อไม่ให้แจ้งเตือนซ้ำหากสถานะยังเหมือนเดิม
if 'prev_status' not in st.session_state: st.session_state.prev_status = {0: 'green', 1: 'green', 2: 'green'}

for i in range(3):
    if f'base_amp{i}' not in st.session_state: st.session_state[f'base_amp{i}'] = None
    if f'history_a{i}' not in st.session_state: st.session_state[f'history_a{i}'] = []
    if f'rms_ch{i}' not in st.session_state: st.session_state[f'rms_ch{i}'] = 0.0
    if f'status{i}' not in st.session_state: st.session_state[f'status{i}'] = 'green'
    if f'consec{i}' not in st.session_state: st.session_state[f'consec{i}'] = 0
    if f'consec_dir{i}' not in st.session_state: st.session_state[f'consec_dir{i}'] = None

# ===== Sidebar =====
with st.sidebar:
    st.header("⚙️ ปรับ Threshold")
    G2Y = st.slider("🟢→🟡", 50, 99, 80, 1)
    Y2R = st.slider("🟡→🔴", 50, 99, 65, 1)
    Y2G = st.slider("🟡→🟢", 50, 99, 87, 1)
    R2Y = st.slider("🔴→🟡", 50, 99, 70, 1)

    st.markdown("---")

# ===== Telegram Notification Function =====
def send_telegram_notification(message):
    """ส่งข้อความแจ้งเตือนผ่าน Telegram API"""
    if not TELEGRAM_BOT_TOKEN or "ใส่_" in TELEGRAM_BOT_TOKEN:
        return # ข้ามการส่งถ้ายังไม่ได้แก้ค่าโทเค็น
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        st.session_state.http_session.post(url, json=payload, timeout=3)
    except Exception as e:
        st.sidebar.warning(f"Telegram Send Error: {e}")

def fetch_data():
    try:
        res = st.session_state.http_session.get(FIREBASE_URL + QUERY, timeout=3)
        if res.status_code == 200:
            data = res.json()
            if not data: return pd.DataFrame()
            flat = {}
            for k, v in data.items():
                if not isinstance(v, dict): continue
                if 'uptime_ms' in v: flat[k] = v
                else:
                    for sk, sv in v.items():
                        if isinstance(sv, dict) and 'uptime_ms' in sv:
                            flat[sk] = sv
            if not flat: return pd.DataFrame()
            df = pd.DataFrame.from_dict(flat, orient='index')
            df['uptime_ms'] = pd.to_numeric(df['uptime_ms'], errors='coerce')
            df = df.dropna(subset=['uptime_ms'])
            return df.sort_values('uptime_ms').reset_index(drop=True)
    except Exception as e:
        st.sidebar.error(f"fetch error: {e}")
    return pd.DataFrame()

def push_baseline_to_firebase(amps):
    payload = {f"base_amp{i}": amps[i] for i in range(3)}
    try:
        res = st.session_state.http_session.patch(STATE_URL + STATE_QUERY, json=payload, timeout=3)
        return res.status_code == 200
    except Exception:
        return False

def fetch_remote_state():
    try:
        res = st.session_state.http_session.get(STATE_URL + STATE_QUERY, timeout=3)
        if res.status_code == 200: return res.json() or {}
    except Exception: pass
    return {}

def get_band_power(df, col, ch_idx, is_new_data):
    sig = df[col].values.astype(float)
    sig = sig - np.mean(sig)
    st.session_state[f'rms_ch{ch_idx}'] = float(np.sqrt(np.mean(sig**2)))
    
    fw, psd = welch(sig, fs=NOMINAL_FS, nperseg=min(256, len(sig)//2), window='hann')
    mask = (fw >= FORCING_FREQ - BAND_HZ) & (fw <= FORCING_FREQ + BAND_HZ)
    band_power = float(np.sum(psd[mask])) if mask.any() else 0.0
    
    hist = st.session_state[f'history_a{ch_idx}']
    if is_new_data:
        hist.append(band_power)
        if len(hist) > HISTORY_SIZE: hist.pop(0)
        st.session_state[f'history_a{ch_idx}'] = hist
        
    return float(np.median(hist)) if hist else band_power

def compute_health(amps):
    bases = [st.session_state[f'base_amp{i}'] for i in range(3)]
    if any(b is None for b in bases): return [None]*3
    return [min(amps[i]/bases[i]*100, 100.0) if bases[i] > 0 else 0.0 for i in range(3)]

def update_status(pct, ch_idx, is_new_data, floor_name):
    s = st.session_state[f'status{ch_idx}']
    c = st.session_state[f'consec{ch_idx}']
    
    if not is_new_data:
        return s, c
        
    new_s = s
    if s == 'green':
        c = c+1 if pct < G2Y else 0
        if c >= MIN_CONSEC: new_s, c = 'yellow', 0
    elif s == 'yellow':
        cur_dir = 'up' if pct >= Y2G else ('down' if pct < Y2R else None)
        prev_dir = st.session_state[f'consec_dir{ch_idx}']
        if cur_dir != prev_dir: c = 0
        st.session_state[f'consec_dir{ch_idx}'] = cur_dir
        if cur_dir is not None:
            c += 1
            if c >= MIN_CONSEC:
                new_s = 'green' if cur_dir == 'up' else 'red'
                c = 0
        else:
            c = 0
    elif s == 'red':
        c = c+1 if pct >= R2Y else 0
        if c >= MIN_CONSEC: new_s, c = 'yellow', 0

    if new_s != st.session_state.prev_status[ch_idx]:
        status_emojis = {'green': '🟢 ปกติ', 'yellow': '⚠️ เฝ้าระวัง', 'red': '🚨 อันตราย!'}
        old_status_text = status_emojis.get(st.session_state.prev_status[ch_idx], st.session_state.prev_status[ch_idx])
        new_status_text = status_emojis.get(new_s, new_s)
        
        msg = f"🔔 *[SmartVibe Alert]*\n📍 *{floor_name}*\n"
        msg += f"🔄 สถานะเปลี่ยน: {old_status_text} ➡️ *{new_status_text}*\n"
        msg += f"📉 Health % ล่าสุด: `{pct:.1f}%`"
        
        send_telegram_notification(msg)
        st.session_state.prev_status[ch_idx] = new_s

    st.session_state[f'status{ch_idx}'] = new_s
    st.session_state[f'consec{ch_idx}'] = c
    return new_s, c

def get_fft_graph_data(df):
    result_freqs, result_psds = None, []
    for col in ['AccX_CH0', 'AccX_CH1', 'AccX_CH2']:
        sig = df[col].values.astype(float) - df[col].mean()
        if len(sig) < 100: return None, None, None, None
        fw, psd = welch(sig, fs=NOMINAL_FS, nperseg=min(256, len(sig)//2), window='hann')
        valid = fw >= 0.5
        if result_freqs is None: result_freqs = fw[valid]
        result_psds.append(psd[valid])
    return result_freqs, result_psds[0], result_psds[1], result_psds[2]

# ==========================================================
# 🔮 Gemini AI deep-analysis
# ==========================================================
def build_gemini_prompt(floor_names, rms_list, amps, base_list, health, status_list, hist_lists, fft_data):
    """สรุปข้อมูลปัจจุบันของระบบให้เป็นข้อความ prompt สำหรับ Gemini"""
    lines = []
    lines.append("คุณคือวิศวกรผู้เชี่ยวชาญด้านการวิเคราะห์ความสั่นสะเทือนของโครงสร้างอาคาร (Structural Health Monitoring)")
    lines.append("ต่อไปนี้คือข้อมูลล่าสุดจากเซ็นเซอร์ความสั่นสะเทือน 3 ชั้นของอาคาร ระบบใช้การวิเคราะห์ band power รอบความถี่ excitation ที่ตั้งใจสั่นอาคาร (forced vibration) เพื่อประเมินสุขภาพโครงสร้างเทียบกับค่า baseline")
    lines.append(f"\nความถี่ที่ใช้กระตุ้น (forcing frequency): {FORCING_FREQ} Hz ± {BAND_HZ} Hz")
    lines.append("\nข้อมูลแต่ละชั้น:")
    for i, name in enumerate(floor_names):
        base_txt = f"{base_list[i]:.5f}" if base_list[i] else "ยังไม่ได้ล็อก baseline"
        health_txt = f"{health[i]:.1f}%" if health[i] is not None else "N/A"
        hist = hist_lists[i]
        cv_txt = "N/A"
        if len(hist) >= 3 and np.mean(hist) > 0:
            cv_txt = f"{np.std(hist)/np.mean(hist)*100:.1f}%"
        lines.append(
            f"- {name}: RMS={rms_list[i]:.4f}, Band Power={amps[i]:.5f}, "
            f"Baseline={base_txt}, Health%={health_txt}, สถานะปัจจุบัน={status_list[i]}, "
            f"ความแปรปรวนของค่าอ่าน (CV)={cv_txt}"
        )

    if fft_data and fft_data[0] is not None:
        xf, m0, m1, m2 = fft_data
        for i, m in enumerate([m0, m1, m2]):
            peak_idx = int(np.argmax(m))
            lines.append(f"- {floor_names[i]}: ความถี่ที่มีพลังงาน PSD สูงสุด ≈ {xf[peak_idx]:.2f} Hz")

    lines.append(
        "\nโปรดวิเคราะห์เชิงลึกเป็นภาษาไทย ครอบคลุมประเด็นต่อไปนี้:\n"
        "1) ภาพรวมสุขภาพโครงสร้างของแต่ละชั้น และแนวโน้มความผิดปกติ (ถ้ามี)\n"
        "2) ความสัมพันธ์ระหว่างชั้น (เช่น ชั้นบนสั่นมากกว่าชั้นล่างผิดปกติหรือไม่)\n"
        "3) ความเสี่ยงที่ควรเฝ้าระวัง และคำแนะนำเชิงปฏิบัติ (ตรวจสอบน็อต/จุดยึด/อื่นๆ)\n"
        "4) สรุปสั้นๆ 1-2 ประโยคปิดท้าย\n"
        "ตอบในรูปแบบ Markdown ใช้หัวข้อย่อยให้อ่านง่าย ไม่ต้องทวนข้อมูลตัวเลขดิบซ้ำทั้งหมด เน้นการตีความ"
    )
    return "\n".join(lines)


def call_gemini_analysis(prompt_text, api_key):
    """เรียก Gemini API (REST) เพื่อขอบทวิเคราะห์เชิงลึก ต้องออนไลน์"""
    api_key = (api_key or "").strip()
    if not api_key:
        return None, "ยังไม่ได้ใส่ Gemini API Key ในแถบด้านซ้าย — ขอฟรีได้ที่ https://aistudio.google.com/app/apikey (ต้องเป็น API Key จาก AI Studio ไม่ใช่ OAuth Client จาก Cloud Console)"

    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt_text}]}
        ]
    }
    try:
        res = requests.post(
            f"{GEMINI_URL}?key={api_key}",
            headers=headers,
            data=json.dumps(body),
            timeout=25
        )
        if res.status_code == 401 or res.status_code == 403:
            return None, (
                f"Gemini API error {res.status_code}: API Key ไม่ถูกต้องหรือไม่มีสิทธิ์ใช้งาน "
                f"กรุณาตรวจสอบว่าใส่ API Key จาก https://aistudio.google.com/app/apikey ถูกต้องครบถ้วน "
                f"(ไม่มีช่องว่างเกิน ไม่ใช่ OAuth Client) — รายละเอียด: {res.text[:200]}"
            )
        if res.status_code != 200:
            return None, f"Gemini API error {res.status_code}: {res.text[:300]}"

        data = res.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return None, "Gemini ไม่ส่งผลลัพธ์กลับมา (ไม่มี candidates)"

        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        if not text:
            return None, "Gemini ส่งผลลัพธ์ว่างเปล่า"
        return text, None
    except requests.exceptions.Timeout:
        return None, "หมดเวลาเชื่อมต่อ Gemini (timeout) — ตรวจสอบอินเทอร์เน็ต"
    except requests.exceptions.ConnectionError:
        return None, "เชื่อมต่ออินเทอร์เน็ตไม่ได้ — ระบบต้องออนไลน์เพื่อใช้ Gemini"
    except Exception as e:
        return None, f"เกิดข้อผิดพลาด: {e}"


# ==========================================================
# Main Execution
# ==========================================================
df = fetch_data()

if not df.empty and len(df) > 50:
    cur = df['uptime_ms'].iloc[-1]
    is_new_data = (cur != st.session_state.last_uptime)
    
    if is_new_data:
        st.session_state.stuck_counter = 0
        st.session_state.last_uptime = cur
    else:
        st.session_state.stuck_counter += 1
        
    if st.session_state.stuck_counter >= 10:
        st.error("🚨 ข้อมูลหยุดนิ่ง — เซ็นเซอร์อาจเน็ตหลุด หรือบอร์ดค้าง")

    amps = [get_band_power(df, f'AccX_CH{i}', i, is_new_data) for i in range(3)]
    health = compute_health(amps)
    floor_names = ["ชั้น 1 (ฐานราก)", "ชั้น 2 (กลาง)", "ชั้น 3 (ยอด)"]

    st.info(f"🔊 Forcing: **{FORCING_FREQ} Hz** ±{BAND_HZ} Hz")
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔒 ล็อก Baseline (ลำโพงเปิด + น็อตครบ)", type="primary", key="btn_lock"):
            for i in range(3):
                st.session_state[f'base_amp{i}'] = amps[i]
                st.session_state[f'status{i}'] = 'green'
                st.session_state[f'consec{i}'] = 0
                st.session_state[f'consec_dir{i}'] = None
                st.session_state.prev_status[i] = 'green'
            ok = push_baseline_to_firebase(amps)
            if ok: st.success("✅ ล็อก baseline และส่งขึ้น Firebase แล้ว")
            st.rerun()
    with c2:
        if st.button("ล้างค่าทั้งหมด", key="btn_reset"):
            for i in range(3):
                st.session_state[f'base_amp{i}'] = None
                st.session_state[f'history_a{i}'] = []
                st.session_state[f'status{i}'] = 'green'
                st.session_state[f'consec{i}'] = 0
                st.session_state[f'consec_dir{i}'] = None
                st.session_state.prev_status[i] = 'green'
            st.rerun()

    st.markdown("---")
    cols = st.columns(3)

    for i in range(3):
        with cols[i]:
            st.subheader(floor_names[i])
            rms_now = st.session_state[f'rms_ch{i}']
            hist = st.session_state[f'history_a{i}']
            base = st.session_state[f'base_amp{i}']

            st.markdown(f"RMS: `{rms_now:.4f}`")
            st.progress(min(int(rms_now / 0.15 * 100), 100))

            if base and base > 0:
                delta_pct = (amps[i] - base) / base * 100
                st.metric(f"Band Power ({FORCING_FREQ}±{BAND_HZ} Hz)", f"{amps[i]:.5f}", delta=f"{delta_pct:+.1f}%")
            else:
                st.metric(f"Band Power ({FORCING_FREQ}±{BAND_HZ} Hz)", f"{amps[i]:.5f}")

            if len(hist) >= 3:
                cv = np.std(hist)/np.mean(hist)*100 if np.mean(hist) > 0 else 0
                st.caption(f"readings: {len(hist)}/{HISTORY_SIZE}  CV={cv:.1f}%  {'✅' if cv < 15 else '⚠️'}")

            if base and base > 0 and health[i] is not None:
                pct = health[i]
                status, cnt = update_status(pct, i, is_new_data, floor_names[i]) 
                st.metric("Health %", f"{pct:.1f}%")
                st.progress(min(int(pct), 100))

                if status == 'green': st.success(f"🟢 ปกติ: {pct:.1f}%")
                elif status == 'yellow': st.warning(f"🟡 เฝ้าระวัง: {pct:.1f}%  [{cnt}/{MIN_CONSEC}]")
                else: st.error(f"🔴 อันตราย: {pct:.1f}%  [{cnt}/{MIN_CONSEC}]")
            else:
                st.info("กด 🔒 ล็อก Baseline")

    st.markdown("---")
    st.subheader("กราฟ FFT แยกตามชั้น")
    result = get_fft_graph_data(df)
    if result[0] is not None:
        xf, m0, m1, m2 = result
        chart_df = pd.DataFrame({"ชั้น 1": m0, "ชั้น 2": m1, "ชั้น 3": m2}, index=xf)
        st.line_chart(chart_df[chart_df.index <= 20], x_label="Frequency (Hz)", y_label="PSD")

        with st.expander("ℹ️ debug"):
            dts = df['uptime_ms'].diff().dropna()
            nd = dts[(dts >= 15) & (dts <= 40)]
            st.write("ช่วงดิฟของ Uptime (ms):", nd.describe())

    # ===== Gemini AI deep analysis section =====
    st.markdown("---")
    st.subheader("🤖 การวิเคราะห์เชิงลึกด้วย Gemini AI")

    gemini_clicked = st.button("🤖 วิเคราะห์เชิงลึกด้วย Gemini AI", key="btn_gemini", type="primary")

    if gemini_clicked:
        rms_list = [st.session_state[f'rms_ch{i}'] for i in range(3)]
        base_list = [st.session_state[f'base_amp{i}'] for i in range(3)]
        status_list = [st.session_state[f'status{i}'] for i in range(3)]
        hist_lists = [st.session_state[f'history_a{i}'] for i in range(3)]

        prompt_text = build_gemini_prompt(
            floor_names, rms_list, amps, base_list, health, status_list, hist_lists, result
        )
        with st.spinner("กำลังเชื่อมต่อ Gemini AI เพื่อวิเคราะห์ข้อมูล... (ต้องออนไลน์)"):
            answer, err = call_gemini_analysis(prompt_text, GEMINI_API_KEY)
        st.session_state.gemini_result = answer
        st.session_state.gemini_error = err

    if st.session_state.gemini_error:
        st.error(f"❌ {st.session_state.gemini_error}")
    if st.session_state.gemini_result:
        st.markdown(st.session_state.gemini_result)
    if not gemini_clicked and not st.session_state.gemini_result and not st.session_state.gemini_error:
        st.caption("กดปุ่ม '🤖 วิเคราะห์เชิงลึกด้วย Gemini AI' ด้านบนเพื่อขอบทวิเคราะห์แบบเชิงลึก (ต้องเชื่อมต่ออินเทอร์เน็ต)")

    st.markdown("---")
    with st.expander("🤖 สถานะ Cloud Function (ฝั่งแจ้งเตือน Telegram)"):
        remote_state = fetch_remote_state()
        if not remote_state:
            st.caption("ยังไม่มีข้อมูลจาก Cloud Function")
        else:
            cols2 = st.columns(3)
            for i in range(3):
                with cols2[i]:
                    st.caption(floor_names[i])
                    st.write(f"status: `{remote_state.get(f'status{i}', '-')}`")
                    st.write(f"last_pct: `{remote_state.get(f'last_pct{i}', '-')}`")
