import os, re, time, collections
from datetime import datetime, timedelta, timezone
import bcrypt, jwt, httpx
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
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
AI_MODEL = os.getenv("AI_MODEL", "llama-3.3-70b-versatile")
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
def health(): return {"status": "ok", "features": ["auth"] + (["ai"] if AI_KEY else [])}

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
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(AI_BASE_URL + "/chat/completions", headers={"Authorization": "Bearer " + AI_KEY},
                             json={"model": AI_MODEL, "messages": msgs, "max_tokens": 1000, "temperature": 0.5})
    except httpx.HTTPError:
        raise HTTPException(502, "AI service not reachable")
    if r.status_code != 200: raise HTTPException(503, "AI is busy right now. Try again shortly.")
    try: text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, ValueError): text = ""
    if not text: raise HTTPException(502, "AI gave no answer")
    return {"reply": text}

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

# Website: use static/ folder if present; otherwise serve index.html from the repo root.
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="web")
elif os.path.isfile("index.html"):
    @app.get("/")
    def home(): return FileResponse("index.html")
