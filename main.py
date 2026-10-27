import os, re, time, json, collections, secrets, hmac, hashlib, base64
from datetime import datetime, timedelta, timezone
import bcrypt, jwt, httpx
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import create_engine, String, Text, DateTime, ForeignKey, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, Session

DB = os.getenv("DATABASE_URL", "sqlite:///./voltsathi.db")
if DB.startswith("postgres://"): DB = DB.replace("postgres://", "postgresql+psycopg2://", 1)
elif DB.startswith("postgresql://"): DB = DB.replace("postgresql://", "postgresql+psycopg2://", 1)
SECRET = os.getenv("SECRET_KEY", "dev-only-change-me")
STATUSES = ["Submitted", "Assigned", "In progress", "Resolved"]
# --- Live AI (any OpenAI-compatible service; default Groq, free) ---
AI_KEY = os.getenv("GROQ_API_KEY") or os.getenv("AI_API_KEY", "")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
AI_MODEL = os.getenv("AI_MODEL", "openai/gpt-oss-120b")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "").strip()
BREVO_API_KEY = os.getenv("BREVO_API_KEY", "").strip()   # for password-reset emails
MAIL_FROM = os.getenv("MAIL_FROM", "").strip()           # a sender email verified in Brevo
AI_SYSTEM = (
    "You are VoltAI, a friendly, knowledgeable AI assistant inside VoltSathi, an electrical fault reporting and safety app used mostly in India. "
    "You can answer any question like a general assistant (studies, coding, daily life) and you are especially strong in electrical topics: appliances, bills, earthing, MCB/fuse, inverter/UPS, solar, motors, basic electronics and polytechnic electrical subjects. "
    "Reply in the same language the user writes in (English, Hindi, or Hinglish). Be clear and conversational. For simple questions answer briefly; for complex ones explain step by step with a small example. "
    "Use simple markdown when it helps: short paragraphs, bullet or numbered lists, **bold** for key terms, and code blocks for code. "
    "Safety first: never give step-by-step instructions for repairing live wiring, bypassing MCB/earthing/meters, or working on high voltage. "
    "For sparking, burning smell, exposed wires or shock, tell the person to stay away, switch off the main supply only if it is safe, and call a qualified electrician or the electricity helpline 1912, or 112 in an emergency. "
    "If you are not sure, say so instead of guessing."
)
ai_hits = collections.defaultdict(list)
engine = create_engine(DB, pool_pre_ping=True,
                       connect_args={"check_same_thread": False} if DB.startswith("sqlite") else {})
SessionLocal = sessionmaker(engine, expire_on_commit=False)
now = lambda: datetime.now(timezone.utc).replace(tzinfo=None)

class Base(DeclarativeBase): pass
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(80))
    email: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(12), default="USER")  # USER / TECHNICIAN / ADMIN
class Report(Base):
    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    reference: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    issue: Mapped[str] = mapped_column(String(80))
    priority: Mapped[str] = mapped_column(String(10), default="Normal")
    location: Mapped[str] = mapped_column(String(140))
    details: Mapped[str] = mapped_column(Text)
    contact: Mapped[str] = mapped_column(String(100), default="")
    gps: Mapped[str] = mapped_column(String(100), default="")
    status: Mapped[str] = mapped_column(String(20), default="Submitted")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=now, onupdate=now)

class PasswordReset(Base):
    __tablename__ = "pw_resets"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(120), index=True)
    code_hash: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    attempts: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)

def hash_pw(p): return bcrypt.hashpw(p.encode()[:72], bcrypt.gensalt()).decode()
def check_pw(p, h): return bcrypt.checkpw(p.encode()[:72], h.encode())

app = FastAPI(title="VoltSathi API")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    Base.metadata.create_all(engine)
    em, pw = os.getenv("ADMIN_EMAIL"), os.getenv("ADMIN_PASSWORD")
    if em and pw:
        with SessionLocal() as s:
            if not s.scalar(select(User).where(User.email == em.lower())):
                s.add(User(name="Admin", email=em.lower(), password_hash=hash_pw(pw), role="ADMIN")); s.commit()

def get_db():
    with SessionLocal() as s: yield s
bearer = HTTPBearer(auto_error=False)
def current_user(c: HTTPAuthorizationCredentials = Depends(bearer), db: Session = Depends(get_db)) -> User:
    try:
        uid = int(jwt.decode(c.credentials, SECRET, algorithms=["HS256"])["sub"])
    except Exception:
        raise HTTPException(401, "Please login again")
    u = db.get(User, uid)
    if not u: raise HTTPException(401, "Please login again")
    return u
def staff(u: User = Depends(current_user)) -> User:
    if u.role not in ("ADMIN", "TECHNICIAN"): raise HTTPException(403, "Staff only")
    return u
def admin(u: User = Depends(current_user)) -> User:
    if u.role != "ADMIN": raise HTTPException(403, "Admin only")
    return u

class AuthIn(BaseModel):
    email: str = Field(max_length=120)
    password: str = Field(min_length=8, max_length=72)
    name: str = Field(default="User", max_length=80)
    @field_validator("email")
    @classmethod
    def ok_email(cls, v):
        v = v.strip().lower()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v): raise ValueError("Invalid email")
        return v
class ReportIn(BaseModel):
    reference: str = Field(pattern=r"^VS-\d{4,10}$")
    issue: str = Field(min_length=1, max_length=80)
    priority: str = Field(default="Normal", pattern="^(Normal|Medium|High)$")
    location: str = Field(min_length=1, max_length=140)
    details: str = Field(min_length=1, max_length=1000)
    contact: str = Field(default="", max_length=100)
    gps: str = Field(default="", max_length=100, pattern=r"^(https://maps\.google\.com/\?q=-?[\d.]+,-?[\d.]+)?$")
class StatusIn(BaseModel):
    status: str
class RoleIn(BaseModel):
    email: str
    role: str = Field(pattern="^(USER|TECHNICIAN|ADMIN)$")
def _norm_email(v):
    v = v.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", v): raise ValueError("Invalid email")
    return v
class ForgotIn(BaseModel):
    email: str = Field(max_length=120)
    @field_validator("email")
    @classmethod
    def ok_email(cls, v): return _norm_email(v)
class ResetIn(BaseModel):
    email: str = Field(max_length=120)
    code: str = Field(min_length=4, max_length=12)
    password: str = Field(min_length=8, max_length=72)
    @field_validator("email")
    @classmethod
    def ok_email(cls, v): return _norm_email(v)
class GoogleIn(BaseModel):
    credential: str = Field(min_length=20, max_length=4000)
class AIMsg(BaseModel):
    role: str
    content: str = Field(max_length=4000)
class AIChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[AIMsg] = Field(default=[], max_length=30)

def user_out(u): return {"id": u.id, "name": u.name, "email": u.email, "role": u.role}
def token_for(u): return jwt.encode({"sub": str(u.id), "exp": datetime.now(timezone.utc) + timedelta(days=7)}, SECRET, algorithm="HS256")
def rep_out(r, name=None):
    d = {c: getattr(r, c) for c in ("reference", "issue", "priority", "location", "details", "contact", "gps", "status")}
    d["created_at"] = r.created_at.isoformat() + "Z"; d["updated_at"] = r.updated_at.isoformat() + "Z"
    if name is not None: d["user_name"] = name
    return d

@app.get("/api/health")
def health():
    f = ["auth"] + (["ai"] if AI_KEY else []) + (["google"] if GOOGLE_CLIENT_ID else []) + (["mail"] if BREVO_API_KEY and MAIL_FROM else [])
    return {"status": "ok", "features": f, "google_client_id": GOOGLE_CLIENT_ID}

def _ai_prepare(b: AIChatIn, request: Request, stream: bool):
    if not AI_KEY: raise HTTPException(503, "AI is not configured on the server")
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "x")).split(",")[0].strip()
    t = time.time(); ai_hits[ip] = [x for x in ai_hits[ip] if t - x < 60]
    if len(ai_hits[ip]) >= 20: raise HTTPException(429, "Too many questions. Please wait a minute.")
    ai_hits[ip].append(t)
    msgs = [{"role": "system", "content": AI_SYSTEM}]
    for h in b.history[-14:]:
        msgs.append({"role": "assistant" if h.role == "assistant" else "user", "content": h.content})
    msgs.append({"role": "user", "content": b.message})
    payload = {"model": AI_MODEL, "messages": msgs, "max_tokens": 2000, "temperature": 0.6}
    if AI_MODEL.startswith("openai/gpt-oss"): payload["reasoning_effort"] = "low"
    if stream: payload["stream"] = True
    return payload

@app.post("/api/ai/chat")
async def ai_chat(b: AIChatIn, request: Request):
    payload = _ai_prepare(b, request, False)
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(AI_BASE_URL + "/chat/completions", headers={"Authorization": "Bearer " + AI_KEY}, json=payload)
    except httpx.HTTPError:
        raise HTTPException(502, "AI service not reachable")
    if r.status_code != 200:
        print("AI error", r.status_code, r.text[:300], flush=True)
        raise HTTPException(503, "AI is busy right now. Try again shortly.")
    try: text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, ValueError): text = ""
    if not text: raise HTTPException(502, "AI gave no answer")
    return {"reply": text}

@app.post("/api/ai/stream")
async def ai_stream(b: AIChatIn, request: Request):
    payload = _ai_prepare(b, request, True)
    def ev(d): return "data: " + json.dumps(d) + "\n\n"
    async def gen():
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(90, connect=15)) as c:
                async with c.stream("POST", AI_BASE_URL + "/chat/completions", headers={"Authorization": "Bearer " + AI_KEY}, json=payload) as r:
                    if r.status_code != 200:
                        body = await r.aread(); print("AI error", r.status_code, body[:300], flush=True)
                        yield ev({"error": "AI is busy right now. Try again shortly."}); return
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"): continue
                        data = line[5:].strip()
                        if data == "[DONE]": break
                        try: tok = json.loads(data)["choices"][0]["delta"].get("content")
                        except (KeyError, IndexError, ValueError, TypeError): continue
                        if tok: yield ev({"t": tok})
        except httpx.HTTPError:
            yield ev({"error": "AI service not reachable"})
        yield ev({"done": True})
    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/api/ai/test")
async def ai_test():
    if not AI_KEY: return {"ok": False, "detail": "No AI key set on the server"}
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(AI_BASE_URL + "/chat/completions", headers={"Authorization": "Bearer " + AI_KEY},
                             json={"model": AI_MODEL, "messages": [{"role": "user", "content": "Say hi"}], "max_tokens": 60})
    except httpx.HTTPError:
        return {"ok": False, "detail": "AI service not reachable"}
    return {"ok": r.status_code == 200, "status": r.status_code, "model": AI_MODEL, "detail": "" if r.status_code == 200 else r.text[:300]}

@app.post("/api/auth/register")
def register(b: AuthIn, db: Session = Depends(get_db)):
    if db.scalar(select(User).where(User.email == b.email)): raise HTTPException(409, "Email already registered")
    u = User(name=b.name.strip() or "User", email=b.email, password_hash=hash_pw(b.password)); db.add(u); db.commit()
    return {"token": token_for(u), "user": user_out(u)}
@app.post("/api/auth/login")
def login(b: AuthIn, db: Session = Depends(get_db)):
    u = db.scalar(select(User).where(User.email == b.email))
    if not u or not check_pw(b.password, u.password_hash): raise HTTPException(401, "Wrong email or password")
    return {"token": token_for(u), "user": user_out(u)}
def _code_hash(email, code): return hmac.new(SECRET.encode(), (email + ":" + code).encode(), hashlib.sha256).hexdigest()

async def send_mail(to, subject, text):
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.post("https://api.brevo.com/v3/smtp/email", headers={"api-key": BREVO_API_KEY, "accept": "application/json"},
                             json={"sender": {"email": MAIL_FROM, "name": "VoltSathi"}, "to": [{"email": to}], "subject": subject, "textContent": text})
        if r.status_code not in (200, 201, 202): print("Mail error", r.status_code, r.text[:300], flush=True)
        return r.status_code in (200, 201, 202)
    except httpx.HTTPError as e:
        print("Mail error", repr(e), flush=True); return False

@app.post("/api/auth/forgot")
async def forgot(b: ForgotIn, db: Session = Depends(get_db)):
    if not (BREVO_API_KEY and MAIL_FROM): raise HTTPException(503, "Email service is not set up on the server yet")
    u = db.scalar(select(User).where(User.email == b.email))
    if u:
        last = db.scalar(select(PasswordReset).where(PasswordReset.email == b.email).order_by(PasswordReset.created_at.desc()))
        if not last or (now() - last.created_at).total_seconds() >= 60:
            code = f"{secrets.randbelow(10**6):06d}"
            db.add(PasswordReset(email=b.email, code_hash=_code_hash(b.email, code), expires_at=now() + timedelta(minutes=10))); db.commit()
            await send_mail(b.email, "VoltSathi password reset code",
                            f"Your VoltSathi reset code is {code}. It is valid for 10 minutes. If you did not ask for this, ignore this email.")
    return {"ok": True}

@app.post("/api/auth/reset")
def reset_password(b: ResetIn, db: Session = Depends(get_db)):
    bad = HTTPException(400, "Wrong or expired code")
    pr = db.scalar(select(PasswordReset).where(PasswordReset.email == b.email).order_by(PasswordReset.created_at.desc()))
    if not pr or pr.expires_at < now() or pr.attempts >= 5: raise bad
    pr.attempts += 1; db.commit()
    if not hmac.compare_digest(pr.code_hash, _code_hash(b.email, b.code.strip())): raise bad
    u = db.scalar(select(User).where(User.email == b.email))
    if not u: raise bad
    u.password_hash = hash_pw(b.password)
    for x in db.scalars(select(PasswordReset).where(PasswordReset.email == b.email)): db.delete(x)
    db.commit()
    return {"ok": True}

@app.post("/api/auth/google")
async def google_login(b: GoogleIn, db: Session = Depends(get_db)):
    if not GOOGLE_CLIENT_ID: raise HTTPException(503, "Google sign-in is not set up on the server yet")
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get("https://oauth2.googleapis.com/tokeninfo", params={"id_token": b.credential})
    except httpx.HTTPError:
        raise HTTPException(502, "Could not reach Google")
    if r.status_code != 200: raise HTTPException(401, "Google sign-in failed")
    d = r.json()
    if d.get("aud") != GOOGLE_CLIENT_ID or str(d.get("email_verified")).lower() != "true" or not d.get("email"):
        raise HTTPException(401, "Google sign-in failed")
    email = d["email"].strip().lower()
    u = db.scalar(select(User).where(User.email == email))
    if not u:
        u = User(name=(d.get("name") or "User")[:80], email=email, password_hash=hash_pw(secrets.token_urlsafe(32))); db.add(u); db.commit()
    return {"token": token_for(u), "user": user_out(u)}

@app.get("/api/auth/me")
def me(u: User = Depends(current_user)): return user_out(u)

@app.post("/api/reports")
def create_report(b: ReportIn, u: User = Depends(current_user), db: Session = Depends(get_db)):
    old = db.scalar(select(Report).where(Report.reference == b.reference))
    if old:
        if old.user_id == u.id: return rep_out(old)  # safe to retry
        raise HTTPException(409, "Reference already used")
    r = Report(user_id=u.id, **b.model_dump()); db.add(r); db.commit()
    return rep_out(r)
@app.get("/api/reports/mine")
def my_reports(u: User = Depends(current_user), db: Session = Depends(get_db)):
    return [rep_out(r) for r in db.scalars(select(Report).where(Report.user_id == u.id).order_by(Report.created_at.desc()))]
@app.get("/api/reports")
def all_reports(_: User = Depends(staff), db: Session = Depends(get_db)):
    rows = db.execute(select(Report, User.name).join(User, User.id == Report.user_id).order_by(Report.created_at.desc()).limit(500))
    return [rep_out(r, n) for r, n in rows]
@app.patch("/api/reports/{reference}/status")
def set_status(reference: str, b: StatusIn, _: User = Depends(staff), db: Session = Depends(get_db)):
    if b.status not in STATUSES: raise HTTPException(422, "Invalid status")
    r = db.scalar(select(Report).where(Report.reference == reference))
    if not r: raise HTTPException(404, "Report not found")
    r.status = b.status; db.commit(); return rep_out(r)
@app.patch("/api/users/role")
def set_role(b: RoleIn, _: User = Depends(admin), db: Session = Depends(get_db)):
    u = db.scalar(select(User).where(User.email == b.email.strip().lower()))
    if not u: raise HTTPException(404, "User not found")
    u.role = b.role; db.commit(); return user_out(u)

# ---- Installable app (PWA): manifest, service worker, icons ----
SW_JS = """const C='voltsathi-v1';
self.addEventListener('install',e=>{self.skipWaiting();e.waitUntil(caches.open(C).then(c=>c.add('/')).catch(()=>{}))});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(x=>x!==C).map(x=>caches.delete(x)))).then(()=>self.clients.claim()))});
self.addEventListener('fetch',e=>{const r=e.request,u=new URL(r.url);if(r.method!=='GET'||u.pathname.startsWith('/api/'))return;
if(r.mode==='navigate'){e.respondWith(fetch(r).then(x=>{const cp=x.clone();caches.open(C).then(c=>c.put('/',cp));return x}).catch(()=>caches.match('/')))}});
"""
_icons = {}
def _make_icon(n):
    """Draw the VoltSathi logo (shield + bolt) with Pillow; cached after first use."""
    if n in _icons: return _icons[n]
    from PIL import Image, ImageDraw, ImageFilter, ImageChops
    import io
    S = 1024
    v = Image.linear_gradient("L").resize((S, S)); h = v.transpose(Image.Transpose.ROTATE_90)
    mask = ImageChops.add(v, h, scale=2)
    bg = Image.composite(Image.new("RGB", (S, S), (10, 56, 102)), Image.new("RGB", (S, S), (20, 184, 160)), mask)
    def bez(p0, p1, p2, p3, k=40):
        out = []
        for i in range(k + 1):
            t = i / k; a = (1 - t) ** 3; b = 3 * (1 - t) ** 2 * t; c = 3 * (1 - t) * t * t; d = t ** 3
            out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0], a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
        return out
    shield = [(512, 170), (815, 282), (815, 520)] + bez((815, 520), (815, 700), (650, 810), (512, 868))[1:] + bez((512, 868), (374, 810), (209, 700), (209, 520))[1:] + [(209, 282)]
    sh = Image.new("L", (S, S), 0); ImageDraw.Draw(sh).polygon([(x, y + 22) for x, y in shield], fill=120)
    bg = Image.composite(Image.new("RGB", (S, S), (4, 30, 50)), bg, sh.filter(ImageFilter.GaussianBlur(26)))
    d = ImageDraw.Draw(bg)
    d.polygon(shield, fill=(255, 255, 255))
    inner = [(512, 222), (765, 314), (765, 515)] + bez((765, 515), (765, 670), (630, 765), (512, 812))[1:] + bez((512, 812), (394, 765), (259, 670), (259, 515))[1:] + [(259, 314)]
    d.polygon(inner, fill=(235, 248, 246))
    bolt = [(598, 268), (372, 566), (496, 566), (432, 790), (668, 470), (542, 470), (646, 268)]
    d.polygon(bolt, fill=(251, 191, 36)); d.line(bolt + [bolt[0]], fill=(217, 119, 6), width=8, joint="curve")
    buf = io.BytesIO(); bg.resize((n, n), Image.LANCZOS).save(buf, "PNG", optimize=True)
    _icons[n] = buf.getvalue(); return _icons[n]
def _icon_response(n):
    try: return Response(_make_icon(n), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
    except Exception as e:
        print("Icon error", repr(e), flush=True); raise HTTPException(404, "Icon not available")
@app.get("/manifest.webmanifest")
def manifest():
    return JSONResponse({"name": "VoltSathi", "short_name": "VoltSathi", "description": "Electrical fault reporting and safety",
        "start_url": "/", "scope": "/", "display": "standalone", "orientation": "portrait",
        "background_color": "#0e766e", "theme_color": "#0e766e",
        "icons": [{"src": "/icon-192.png?v=2", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
                  {"src": "/icon-512.png?v=2", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}]},
        media_type="application/manifest+json")
@app.get("/sw.js")
def service_worker(): return Response(SW_JS, media_type="application/javascript", headers={"Cache-Control": "no-cache"})
@app.get("/icon-192.png")
def icon192(): return _icon_response(192)
@app.get("/icon-512.png")
def icon512(): return _icon_response(512)

# Website: use static/ folder if present; otherwise serve index.html from the repo root.
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="web")
elif os.path.isfile("index.html"):
    @app.get("/")
    def home(): return FileResponse("index.html")
