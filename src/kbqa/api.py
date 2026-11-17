"""HTTP 服务：/v1/chat 多轮问答、/v1/search 只检索、/healthz 健康检查。

根路径给了一个单文件对话页面，方便演示时直接打开浏览器，
不用再装前端工程。会话记忆放在进程内字典里，重启即失效——
演示够用，生产要换成 Redis 并带上 TTL。
"""

from __future__ import annotations

from collections import defaultdict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .agent.pipeline import QAAgent


class ChatIn(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    session_id: str = "default"


class ChatOut(BaseModel):
    answer: str
    status: str
    rewritten: str
    citations: list[dict]
    trace: list[dict]
    latency_ms: float


PAGE = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>知识库问答</title><style>
:root{--bd:#e3e8ef;--fg:#1f2937;--sub:#6b7280;--bg:#f7f9fc;--ac:#2563eb}
*{box-sizing:border-box}body{margin:0;font:15px/1.75 -apple-system,"Microsoft YaHei",sans-serif;
color:var(--fg);background:var(--bg);display:flex;justify-content:center;padding:32px 16px}
.wrap{width:min(860px,100%)}h1{font-size:19px;margin:0 0 4px}
p.sub{color:var(--sub);margin:0 0 20px;font-size:13px}
.box{background:#fff;border:1px solid var(--bd);border-radius:12px;padding:18px}
.row{display:flex;gap:10px}input{flex:1;padding:11px 13px;border:1px solid var(--bd);
border-radius:9px;font-size:15px;outline:none}input:focus{border-color:var(--ac)}
button{padding:11px 20px;border:0;border-radius:9px;background:var(--ac);color:#fff;
font-size:15px;cursor:pointer}button:disabled{opacity:.55;cursor:default}
.ans{margin-top:18px;white-space:pre-wrap}.cite{margin-top:14px;padding-top:12px;
border-top:1px dashed var(--bd);font-size:13px;color:var(--sub)}
.cite div{margin:6px 0}.tag{display:inline-block;padding:1px 7px;border-radius:5px;
background:#eef2ff;color:#3730a3;font-size:12px;margin-right:6px}
.meta{margin-top:10px;font-size:12px;color:var(--sub)}
</style></head><body><div class="wrap">
<h1>知识库问答 Agent</h1>
<p class="sub">先检索原文再作答，答案后标注出处；没有依据会直接说没有。</p>
<div class="box">
  <div class="row"><input id="q" placeholder="例如：北京出差住宿一晚能报销多少？"
    autocomplete="off"><button id="go">提问</button></div>
  <div id="out"></div>
</div>
</div><script>
const q=document.getElementById('q'),go=document.getElementById('go'),out=document.getElementById('out');
const sid='s-'+Math.random().toString(36).slice(2,9);
async function ask(){const t=q.value.trim();if(!t)return;go.disabled=true;
out.innerHTML='<div class="meta">检索中…</div>';
try{const r=await fetch('/v1/chat',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({query:t,session_id:sid})});const d=await r.json();
let h='<div class="ans">'+d.answer.replace(/</g,'&lt;')+'</div>';
if(d.citations.length){h+='<div class="cite">'+d.citations.map(c=>
'<div><span class="tag">'+c.index+'</span>'+c.title+'｜'+c.section+
' <span style="opacity:.7">score '+c.score+'</span></div>').join('')+'</div>';}
h+='<div class="meta">'+d.status+' · '+Math.round(d.latency_ms)+' ms'+
(d.rewritten&&d.rewritten!==t?' · 改写：'+d.rewritten:'')+'</div>';
out.innerHTML=h;}catch(e){out.innerHTML='<div class="meta">请求失败：'+e+'</div>';}
finally{go.disabled=false;}}
go.onclick=ask;q.addEventListener('keydown',e=>{if(e.key==='Enter')ask();});
</script></body></html>"""


def create_app(cfg) -> FastAPI:
    app = FastAPI(title="kb-qa-agent", version="0.1.0")
    agent = QAAgent(cfg)
    memory: dict[str, list[dict]] = defaultdict(list)
    turns = int(cfg.get("server.memory_turns", 6))

    @app.get("/healthz")
    def healthz():
        return {
            "ok": True,
            "docs": len(agent.retriever.docs()),
            "chunks": len(agent.retriever.chunks),
            "llm": agent.llm.available,
        }

    @app.post("/v1/chat", response_model=ChatOut)
    def chat(body: ChatIn):
        hist = memory[body.session_id][-turns * 2:]
        res = agent.ask(body.query, hist)
        memory[body.session_id].append({"role": "user", "content": body.query})
        memory[body.session_id].append({"role": "assistant", "content": res.answer})
        return ChatOut(
            answer=res.answer,
            status=res.status,
            rewritten=res.rewritten,
            citations=[c.to_dict() for c in res.citations],
            trace=res.trace,
            latency_ms=round(res.latency_ms, 1),
        )

    @app.get("/v1/search")
    def search(q: str, k: int = 5):
        hits = agent.retriever.search(q, k)
        return {
            "query": q,
            "hits": [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "doc_id": h.chunk.doc_id,
                    "section": h.chunk.section,
                    "score": round(h.score, 4),
                    "source": h.source,
                    "text": h.chunk.text[:200],
                }
                for h in hits
            ],
        }

    @app.get("/", response_class=HTMLResponse)
    def index():
        return PAGE

    return app
