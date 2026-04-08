from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Literal, Optional
from datetime import datetime
from uuid import uuid4
import os
import requests

from dotenv import load_dotenv

# ==============#
#  CONFIG & ENV #
# ==============#

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI(title="Social Bot – CRM + Meta DM Webhook", version="1.2")

# Allow dashboard.html (file://) to call http://127.0.0.1:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # For local dev. Lock this in production.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Admin login ---
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password123")

print("[AUTH] Admin credentials are:")
print("  EMAIL:", repr(ADMIN_EMAIL))
print("  PASSWORD:", repr(ADMIN_PASSWORD))

security = HTTPBearer()
CURRENT_TOKEN: Optional[str] = None

# --- Global settings (simple, per-install) ---
SETTINGS = {
    "auto_reply": True,
    "ai_assistant_enabled": True,
}

# --- Meta / Facebook / Instagram config ---
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "my_secret_token")

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")  # generic
FB_PAGE_ID = os.getenv("FB_PAGE_ID")
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN") or ACCESS_TOKEN
INSTAGRAM_ID = os.getenv("INSTAGRAM_ID")

if not FB_PAGE_ACCESS_TOKEN:
    print("[META] WARNING: No FB_PAGE_ACCESS_TOKEN or ACCESS_TOKEN found. "
          "Auto-replies to Instagram/Facebook DMs will NOT work.")

# --- X / Twitter (not used yet, but loaded) ---
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")
X_WEBHOOK_SECRET = os.getenv("X_WEBHOOK_SECRET")
X_BOT_USER_ID = os.getenv("X_BOT_USER_ID")

# --- Optional AI (OpenAI) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

openai_client = None
if OPENAI_API_KEY and OpenAI is not None:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)


# =========#
#  MODELS  #
# =========#

class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class AiChatRequest(BaseModel):
    message: str
    history: List[ChatHistoryItem] = []


class TestMessageRequest(BaseModel):
    sender: str = "test_user"
    message: str = "Hello"
    channel: str = "test"   # "Instagram", "Facebook", "X", etc.


class KeywordCreateRequest(BaseModel):
    keyword: str
    reply: str


class SettingsUpdateRequest(BaseModel):
    auto_reply: Optional[bool] = None
    ai_assistant_enabled: Optional[bool] = None


# =============#
#  AUTH LOGIC  #
# =============#

def get_current_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """
    Simple single-session bearer token auth.
    """
    global CURRENT_TOKEN
    if not CURRENT_TOKEN or credentials.credentials != CURRENT_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return credentials.credentials


# =============#
#  UTILITIES   #
# =============#

def load_replies():
    """
    Load keyword-based replies from replies.txt.
    Format per line:
      keyword|Reply text...
    """
    replies = {}
    path = os.path.join(BASE_DIR, "replies.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "|" in line:
                    key, rep = line.split("|", 1)
                    replies[key.lower()] = rep
    except FileNotFoundError:
        pass
    return replies


def generate_reply(message: str) -> str:
    """
    Simple DM reply:
      - If auto_reply disabled -> static message.
      - Else, keyword matching from replies.txt.
      - Fallback generic message.
    """
    if not SETTINGS.get("auto_reply", True):
        return "Auto-replies are currently disabled. A human will respond soon."

    msg = (message or "").lower()
    replies = load_replies()
    for k, rep in replies.items():
        if k in msg:
            return rep

    return "Hello! How can I help you today?"


def save_message(sender: str, message: str, reply: str, channel: str = "Unknown"):
    """
    Append a message + bot reply to messages.txt.
    Format:
      --- YYYY-MM-DD HH:MM:SS ---
      From: sender
      Message: message
      Reply: reply
      Channel: channel
    (Dashboard reads this file.)
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(BASE_DIR, "messages.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n--- {ts} ---\n")
        f.write(f"From: {sender}\n")
        f.write(f"Message: {message}\n")
        f.write(f"Reply: {reply}\n")
        f.write(f"Channel: {channel}\n")


def resolve_meta_name(user_id: str) -> str:
    """
    Try to resolve a human-friendly name/username for a Facebook/Instagram user.
    If it fails, just returns the raw ID.
    """
    if not FB_PAGE_ACCESS_TOKEN:
        return user_id

    url = f"https://graph.facebook.com/v20.0/{user_id}"
    params = {
        "fields": "name,username,ig_username",
        "access_token": FB_PAGE_ACCESS_TOKEN,
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        if not r.ok:
            print("[META] name lookup failed:", r.status_code, r.text)
            return user_id
        data = r.json()
        for field in ("ig_username", "username", "name"):
            if data.get(field):
                return data[field]
        return data.get("id", user_id)
    except Exception as e:
        print("[META] name lookup error:", e)
        return user_id


def send_meta_dm(recipient_id: str, text: str, channel_label: str):
    """
    Send a DM via Meta Graph:
      - Works for Facebook Page DMs and Instagram DMs (since both use the Page token).
      - Uses /me/messages with page access token.
    """
    if not FB_PAGE_ACCESS_TOKEN:
        print(f"[{channel_label}] No FB_PAGE_ACCESS_TOKEN configured. Cannot send reply.")
        return

    url = "https://graph.facebook.com/v20.0/me/messages"
    params = {"access_token": FB_PAGE_ACCESS_TOKEN}
    payload = {
        "recipient": {"id": recipient_id},
        "message": {"text": text},
    }

    try:
        r = requests.post(url, params=params, json=payload, timeout=10)
        print(f"[{channel_label}] send reply status:", r.status_code, r.text)
    except Exception as e:
        print(f"[{channel_label}] error sending reply:", e)


# =================#
#  BASIC ENDPOINTS #
# =================#

@app.get("/")
def home():
    return {"status": "ok", "message": "Social Bot backend is running"}


# ============#
#  AUTH API   #
# ============#

@app.post("/api/login")
async def login(request: Request):
    """
    Single admin login.
    Credentials come from .env:
      ADMIN_EMAIL, ADMIN_PASSWORD
    """
    global CURRENT_TOKEN
    data = await request.json()
    email = (data.get("email") or "").strip()
    password = (data.get("password") or "").strip()

    print("[LOGIN] received:", repr(email), repr(password))

    if email != ADMIN_EMAIL or password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = str(uuid4())
    CURRENT_TOKEN = token
    print("[LOGIN] success, token:", token)
    return {"token": token, "email": email}


@app.post("/api/logout")
async def logout(token: str = Depends(get_current_token)):
    """
    Clear the current auth token.
    """
    global CURRENT_TOKEN
    CURRENT_TOKEN = None
    print("[LOGOUT] token cleared")
    return {"success": True}


# ==============#
#  SETTINGS API #
# ==============#

@app.get("/api/settings")
def get_settings(token: str = Depends(get_current_token)):
    return SETTINGS


@app.post("/api/settings")
async def update_settings(
    request: Request,
    token: str = Depends(get_current_token),
):
    data = await request.json()
    payload = SettingsUpdateRequest(**data)

    if payload.auto_reply is not None:
        SETTINGS["auto_reply"] = payload.auto_reply
        print("[SETTINGS] auto_reply:", payload.auto_reply)

    if payload.ai_assistant_enabled is not None:
        SETTINGS["ai_assistant_enabled"] = payload.ai_assistant_enabled
        print("[SETTINGS] ai_assistant_enabled:", payload.ai_assistant_enabled)

    return SETTINGS


# =============#
#  STATUS API  #
# =============#

@app.get("/api/status")
def get_status(token: str = Depends(get_current_token)):
    """
    Status used by the dashboard.
    """
    return {
        "server": "online",
        "webhook": "active",  # If webhook is set on Meta app, this is effectively true
        "instagram_id": INSTAGRAM_ID,
        "facebook_page_id": FB_PAGE_ID,
        "x_bot_user_id": X_BOT_USER_ID,
        "auto_reply": SETTINGS.get("auto_reply", True),
        "ai_assistant_enabled": SETTINGS.get("ai_assistant_enabled", True),
    }


# ===============#
#  MESSAGES API  #
# ===============#

@app.get("/api/messages")
def get_messages(token: str = Depends(get_current_token)):
    """
    Read messages.txt and return structured DM history
    for the CRM dashboard & analytics.
    """
    messages = []
    path = os.path.join(BASE_DIR, "messages.txt")

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except FileNotFoundError:
        return {"messages": []}

    if not content:
        return {"messages": []}

    entries = content.split("\n--- ")
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue

        lines = entry.split("\n")
        if len(lines) < 4:
            continue

        raw_ts = lines[0].replace("---", "").strip()
        from_line = lines[1].replace("From:", "").strip()
        msg_line = lines[2].replace("Message:", "").strip()
        reply_line = lines[3].replace("Reply:", "").strip()
        channel = "Unknown"
        if len(lines) >= 5 and "Channel:" in lines[4]:
            channel = lines[4].replace("Channel:", "").strip()

        messages.append(
            {
                "timestamp": raw_ts,
                "from": from_line,
                "message": msg_line,
                "reply": reply_line,
                "channel": channel,
            }
        )

    return {"messages": messages}


@app.post("/api/test-message")
async def test_message(
    payload: TestMessageRequest,
    token: str = Depends(get_current_token),
):
    """
    Sandbox endpoint: simulate a DM coming in from any channel.
    Used by the "DM Test Sandbox" card on the dashboard.
    """
    sender = payload.sender or "test_user"
    message = payload.message or "Hello"
    channel = payload.channel or "test"

    reply = generate_reply(message)
    save_message(sender, message, reply, channel)

    print(f"[TEST] {channel} | {sender}: {message} -> {reply}")

    return {
        "success": True,
        "from": sender,
        "message": message,
        "reply": reply,
        "channel": channel,
    }


# ==============#
#  KEYWORDS API #
# ==============#

@app.get("/api/keywords")
def get_keywords(token: str = Depends(get_current_token)):
    """
    List all keyword-based replies (auto DM rules).
    """
    path = os.path.join(BASE_DIR, "replies.txt")
    keywords = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "|" in line:
                    k, rep = line.split("|", 1)
                    keywords.append({"keyword": k, "reply": rep})
    except FileNotFoundError:
        pass
    return {"keywords": keywords}


@app.post("/api/keywords")
async def add_keyword(
    request: Request,
    token: str = Depends(get_current_token),
):
    """
    Add a keyword → reply rule.
    """
    data = await request.json()
    keyword = (data.get("keyword") or "").strip().lower()
    reply = (data.get("reply") or "").strip()

    if not keyword or not reply:
        return JSONResponse(
            {"error": "Keyword and reply required"},
            status_code=status.HTTP_400_BAD_REQUEST,
        )

    path = os.path.join(BASE_DIR, "replies.txt")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{keyword}|{reply}\n")

    print("[KEYWORD] added:", keyword)
    return {"success": True, "keyword": keyword}


@app.delete("/api/keywords/{keyword}")
def delete_keyword(keyword: str, token: str = Depends(get_current_token)):
    """
    Remove a keyword → reply rule.
    """
    path = os.path.join(BASE_DIR, "replies.txt")
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return JSONResponse({"error": "File not found"}, status_code=404)

    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            if not line.startswith(f"{keyword}|"):
                f.write(line)

    print("[KEYWORD] deleted:", keyword)
    return {"success": True}


# =================#
#  AI CHAT API     #
# =================#

@app.post("/api/ai/chat")
async def ai_chat(
    payload: AiChatRequest,
    token: str = Depends(get_current_token),
):
    """
    Generic AI assistant used by the right-hand AI drawer.
    """
    user_message = payload.message.strip()
    history = payload.history

    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    if not SETTINGS.get("ai_assistant_enabled", True):
        return {
            "reply": "The AI assistant is currently disabled in settings.",
            "used_ai": False,
        }

    if not openai_client:
        # Simple local fallback (no external API call)
        txt_history = "\n".join([f"{h.role}: {h.content}" for h in history])
        reply = (
            "AI backend is not fully configured yet.\n\n"
            f"You said: {user_message}\n\n"
            "History:\n" + (txt_history or "(no history)")
        )
        return {"reply": reply, "used_ai": False}

    # Real OpenAI call (if configured)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful AI assistant inside a Social Media Bot dashboard "
                "called 'Social Bot'. You help with reply ideas, content, and support. "
                "Keep answers concise and actionable."
            ),
        }
    ]
    for h in history:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": user_message})

    try:
        completion = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.4,
        )
        reply = completion.choices[0].message.content
        return {"reply": reply, "used_ai": True}
    except Exception as e:
        print("[AI] error:", e)
        return {
            "reply": "Error calling AI backend. Check your OpenAI API key/plan.",
            "used_ai": False,
        }


# =====================#
#  META WEBHOOK (IG/FB)
# =====================#

@app.get("/webhook")
def verify_webhook(
    hub_mode: Optional[str] = None,
    hub_verify_token: Optional[str] = None,
    hub_challenge: Optional[str] = None,
):
    """
    GET /webhook is called by Meta when you set the webhook URL.
    It must echo back hub_challenge if the token matches.
    """
    print("[WEBHOOK][GET]", hub_mode, hub_verify_token, hub_challenge)
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return PlainTextResponse(hub_challenge or "")
    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def handle_webhook(request: Request):
    """
    POST /webhook is called for every Instagram/Facebook event.
    We:
      - Parse DM messages
      - Ignore echo messages (is_echo = true)
      - Generate reply using our DM logic
      - Save to messages.txt
      - Send reply via Meta API
    """
    data = await request.json()
    print("[WEBHOOK][POST] raw:", data)

    obj = data.get("object")

    # Instagram DMs (Instagram Messaging)
    if obj == "instagram":
        for entry in data.get("entry", []):
            messaging_events = entry.get("messaging", [])
            for event in messaging_events:
                message = event.get("message") or {}
                # Ignore echo messages (our own replies)
                if message.get("is_echo"):
                    print("[META] Ignoring Instagram echo message")
                    continue

                sender_id = event.get("sender", {}).get("id")
                text = message.get("text", "")

                if not sender_id or not text:
                    continue

                # Try to resolve sender name (username) for dashboard
                sender_name = resolve_meta_name(sender_id)

                print(f"[META] Incoming Instagram DM from {sender_id} ({sender_name}): {text}")

                reply = generate_reply(text)
                save_message(sender_name, text, reply, channel="Instagram")
                send_meta_dm(sender_id, reply, channel_label="Instagram")

        return {"status": "ok"}

    # Facebook Page DMs (Messenger)
    if obj == "page":
        for entry in data.get("entry", []):
            messaging_events = entry.get("messaging", [])
            for event in messaging_events:
                message = event.get("message") or {}
                if message.get("is_echo"):
                    print("[META] Ignoring Facebook echo message")
                    continue

                sender_id = event.get("sender", {}).get("id")
                text = message.get("text", "")

                if not sender_id or not text:
                    continue

                sender_name = resolve_meta_name(sender_id)

                print(f"[META] Incoming Facebook DM from {sender_id} ({sender_name}): {text}")

                reply = generate_reply(text)
                save_message(sender_name, text, reply, channel="Facebook")
                send_meta_dm(sender_id, reply, channel_label="Facebook")

        return {"status": "ok"}

    print("[WEBHOOK] Unknown object type:", obj)
    return {"status": "ignored", "object": obj}
