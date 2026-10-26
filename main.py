import os, re, time, collections, secrets, hmac, hashlib, base64
from datetime import datetime, timedelta, timezone
import bcrypt, jwt, httpx
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
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
    "You are VoltAI, the assistant inside VoltSathi, an electrical fault reporting and safety app used mostly in India. "
    "Answer in the same language the user writes in (English, Hindi, or Hinglish). Keep answers short, simple and practical. "
    "You can explain electrical concepts, appliances, bills, earthing, MCB/fuse, inverters, solar and basic electronics, and also help with general questions. "
    "Safety first: never give step-by-step instructions for repairing live wiring, bypassing MCB/earthing/meters, or working on high voltage. "
    "For sparking, burning smell, exposed wires or shock, tell the person to stay away, switch off the main supply only if it is safe, and call a qualified electrician or the electricity helpline 1912, or 112 in an emergency. "
    "If you are not sure, say so."
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
    content: str = Field(max_length=2000)
class AIChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    history: list[AIMsg] = []

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

@app.post("/api/ai/chat")
async def ai_chat(b: AIChatIn, request: Request):
    if not AI_KEY: raise HTTPException(503, "AI is not configured on the server")
    ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "x")).split(",")[0].strip()
    t = time.time(); ai_hits[ip] = [x for x in ai_hits[ip] if t - x < 60]
    if len(ai_hits[ip]) >= 15: raise HTTPException(429, "Too many questions. Please wait a minute.")
    ai_hits[ip].append(t)
    msgs = [{"role": "system", "content": AI_SYSTEM}]
    for m in b.history[-8:]:
        msgs.append({"role": "assistant" if m.role == "assistant" else "user", "content": m.content})
    msgs.append({"role": "user", "content": b.message})
    payload = {"model": AI_MODEL, "messages": msgs, "max_tokens": 1500, "temperature": 0.5}
    if AI_MODEL.startswith("openai/gpt-oss"): payload["reasoning_effort"] = "low"
    try:
        async with httpx.AsyncClient(timeout=40) as c:
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
ICON_192 = base64.b64decode("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")
ICON_512 = base64.b64decode("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")
SW_JS = """const C='voltsathi-v1';
self.addEventListener('install',e=>{self.skipWaiting();e.waitUntil(caches.open(C).then(c=>c.add('/')).catch(()=>{}))});
self.addEventListener('activate',e=>{e.waitUntil(caches.keys().then(k=>Promise.all(k.filter(x=>x!==C).map(x=>caches.delete(x)))).then(()=>self.clients.claim()))});
self.addEventListener('fetch',e=>{const r=e.request,u=new URL(r.url);if(r.method!=='GET'||u.pathname.startsWith('/api/'))return;
if(r.mode==='navigate'){e.respondWith(fetch(r).then(x=>{const cp=x.clone();caches.open(C).then(c=>c.put('/',cp));return x}).catch(()=>caches.match('/')))}});
"""
@app.get("/manifest.webmanifest")
def manifest():
    return JSONResponse({"name": "VoltSathi", "short_name": "VoltSathi", "description": "Electrical fault reporting and safety",
        "start_url": "/", "scope": "/", "display": "standalone", "orientation": "portrait",
        "background_color": "#0e766e", "theme_color": "#0e766e",
        "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any maskable"},
                  {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}]},
        media_type="application/manifest+json")
@app.get("/sw.js")
def service_worker(): return Response(SW_JS, media_type="application/javascript", headers={"Cache-Control": "no-cache"})
@app.get("/icon-192.png")
def icon192(): return Response(ICON_192, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})
@app.get("/icon-512.png")
def icon512(): return Response(ICON_512, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

# Website: use static/ folder if present; otherwise serve index.html from the repo root.
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="web")
elif os.path.isfile("index.html"):
    @app.get("/")
    def home(): return FileResponse("index.html")
