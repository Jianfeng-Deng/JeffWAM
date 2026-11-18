#!/usr/bin/env python3
"""E114-H human stage review server (stdlib only). Read-only on videos/labels; writes only to --out.
Run:  python server.py --port 18795 --out results/e114h_human_stage_20261116
Bind is 127.0.0.1; reach it with an ssh port forward.
"""
import argparse, json, os, re, threading, time, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

T = Path("/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu")
B3 = T / "results/e99b3_20261115_01"
CAUSAL = T / "data/e98_predicate_labels_train_causal_v1"
HERE = Path(__file__).resolve().parent
CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
STAGES = [f"S{i:02d}" for i in range(1, 11)]
LOCK = threading.Lock()


def resolve(p):
    p = str(p)
    if not Path(p).exists():
        p = p.replace("/mnt/workspace/colleague-a/Datasets/piper/", "/mnt/workspace/dataset/piper/")
    return p


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-3))).isoformat()


def load_state():
    src = json.loads((B3 / "sources.json").read_text())
    spec = json.loads((B3 / "spec.json").read_text())
    parents = {}
    for s in src:
        videos = {c: resolve(s["videos"][c]) for c in CAMS}
        parents[s["parent"]] = dict(id=s["parent"], cid=s["cid"], pool=s["pool"], n=s["n"], videos=videos,
                                    ok=all(Path(v).is_file() for v in videos.values()))
    retro = {}  # retrospective model stage intervals (revealed only after the human saves)
    for pool in ("train", "dev"):
        for z in json.loads((B3 / f"stages_{pool}.json").read_text()):
            retro.setdefault(z["parent_id"], []).append(
                dict(stage=z["stage_id"], start=z["stage_start"], end=z["stage_end_exclusive"]))
    moments = {}  # E98-C causal first-pass stage answers per training parent
    cid2p = {v["cid"]: k for k, v in parents.items()}
    for line in (CAUSAL / "stage_causal.jsonl").open():
        r = json.loads(line)
        p = cid2p.get(r["canonical_id"])
        if p:
            moments.setdefault(p, []).append(dict(frame=r["source_frame"], model_stage=r["stage"], evidence=r.get("evidence", "")))
    for p in moments:
        moments[p].sort(key=lambda m: m["frame"])
    stages = [dict(id=s["id"], name=s["name_zh"], arm=s.get("responsible_arm", ""), start=s.get("visible_start_zh", ""),
                   end=s.get("visible_end_zh", ""), predicates=s.get("predicates", [])) for s in spec["stages"]]
    qs = {p["id"]: p["question_zh"] for p in spec["predicates"]}
    for s in stages:
        s["questions"] = [qs[x] for x in s["predicates"] if x in qs]
    return parents, retro, moments, stages


PARENTS, RETRO, MOMENTS, STAGE_INFO = load_state()


class H(BaseHTTPRequestHandler):
    out: Path = None
    server_version = "stagereview/1"

    def log_message(self, *a):
        pass

    # ---- helpers
    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        if n > 2_000_000:
            raise ValueError("too large")
        return json.loads(self.rfile.read(n))

    def ann_path(self, kind, parent):
        return self.out / kind / f"{parent}.json"

    def load_ann(self, kind, parent):
        p = self.ann_path(kind, parent)
        return json.loads(p.read_text()) if p.exists() else None

    def save_ann(self, kind, parent, obj):
        d = self.out / kind
        d.mkdir(parents=True, exist_ok=True)
        with LOCK:
            tmp = d / f".{parent}.tmp"
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
            os.replace(tmp, d / f"{parent}.json")
            with (self.out / "log.jsonl").open("a") as f:
                f.write(json.dumps(dict(brt=now(), kind=kind, parent=parent, annotator=obj.get("annotator"),
                                        n=len(obj.get("boundaries", obj.get("verdicts", [])))), ensure_ascii=False) + "\n")

    # ---- routes
    def do_GET(self):
        u = urlparse(self.path)
        path = u.path
        try:
            if path in ("/", "/index.html"):
                b = (HERE / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif path == "/api/parents":
                rows = []
                for p, v in PARENTS.items():
                    a = self.load_ann("boundaries", p)
                    r = self.load_ann("review", p)
                    rows.append(dict(id=p, pool=v["pool"], n=v["n"], ok=v["ok"], labeled=bool(a), reviewed=bool(r),
                                     moments=len(MOMENTS.get(p, [])), n_reviewed=len(r["verdicts"]) if r else 0,
                                     annotator=(a or {}).get("annotator")))
                rows.sort(key=lambda r: (r["pool"] != "dev", r["id"]))
                self.send_json(dict(parents=rows, stages=STAGE_INFO))
            elif path.startswith("/api/parent/"):
                p = path.split("/")[-1]
                if p not in PARENTS:
                    return self.send_json(dict(error="unknown"), 404)
                v = PARENTS[p]
                a = self.load_ann("boundaries", p)
                dr = self.out / "draft" / f"{p}.json"
                out = dict(id=p, pool=v["pool"], n=v["n"], fps=30, annotation=a,
                           draft=(json.loads(dr.read_text()) if dr.exists() else None),
                           videos={c: f"/video/{p}/{c}" for c in CAMS})
                if a:  # reveal model intervals only after the human has saved (anti-anchoring)
                    out["model_retro"] = RETRO.get(p, [])
                self.send_json(out)
            elif path.startswith("/api/moments/"):
                p = path.split("/")[-1]
                if p not in PARENTS:
                    return self.send_json(dict(error="unknown"), 404)
                self.send_json(dict(moments=MOMENTS.get(p, []), review=self.load_ann("review", p)))
            elif path.startswith("/video/"):
                _, _, p, cam = path.split("/")
                if p not in PARENTS or cam not in CAMS:
                    return self.send_json(dict(error="unknown"), 404)
                self.serve_video(Path(PARENTS[p]["videos"][cam]))
            elif path == "/api/export":
                rows = []
                for kind in ("boundaries", "review"):
                    d = self.out / kind
                    if d.exists():
                        rows += [json.loads(f.read_text()) | {"kind": kind} for f in sorted(d.glob("*.json"))]
                self.send_json(rows)
            else:
                self.send_json(dict(error="not found"), 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def serve_video(self, f):
        size = f.stat().st_size
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        code = 200
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    if m.group(2):
                        end = min(int(m.group(2)), size - 1)
                elif m.group(2):
                    start = max(0, size - int(m.group(2)))
                code = 206
        if start > end or start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        self.send_response(code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with f.open("rb") as fh:
            fh.seek(start)
            left = end - start + 1
            while left > 0:
                b = fh.read(min(1 << 20, left))
                if not b:
                    break
                self.wfile.write(b)
                left -= len(b)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            body = self.read_json()
            parent = path.split("/")[-1]
            if parent not in PARENTS:
                return self.send_json(dict(error="unknown parent"), 404)
            n = PARENTS[parent]["n"]
            who = re.sub(r"[^\w\-.@一-鿿]", "", str(body.get("annotator", "")))[:40] or "anonymous"
            if path.startswith("/api/save/"):
                bs = []
                for b in body.get("boundaries", []):
                    f, s = int(b["frame"]), str(b["stage"])
                    if not (0 <= f < n) or (s not in STAGES and s != "none"):
                        return self.send_json(dict(error=f"bad boundary {b}"), 400)
                    bs.append(dict(frame=f, stage=s, uncertain=bool(b.get("uncertain", False))))
                bs.sort(key=lambda x: x["frame"])
                prev = self.load_ann("boundaries", parent)
                obj = dict(parent=parent, annotator=who, saved_brt=now(), n_frames=n, fps=30,
                           semantics="marker = the stage starts at this frame and lasts until the next marker; human label",
                           label_source="human", boundaries=bs, notes=str(body.get("notes", ""))[:2000],
                           version=(prev or {}).get("version", 0) + 1)
                self.save_ann("boundaries", parent, obj)
                self.send_json(dict(ok=True, model_retro=RETRO.get(parent, []), version=obj["version"]))
            elif path.startswith("/api/draft/"):
                bs = [dict(frame=int(b["frame"]), stage=str(b["stage"]), uncertain=bool(b.get("uncertain", False)))
                      for b in body.get("boundaries", []) if 0 <= int(b["frame"]) < n and (b["stage"] in STAGES or b["stage"] == "none")]
                d = self.out / "draft"
                d.mkdir(parents=True, exist_ok=True)
                with LOCK:
                    (d / f"{parent}.json").write_text(json.dumps(dict(parent=parent, annotator=who, saved_brt=now(), boundaries=bs,
                                                                      notes=str(body.get("notes", ""))[:2000]), ensure_ascii=False))
                self.send_json(dict(ok=True))
            elif path.startswith("/api/review/"):
                vs = []
                for v in body.get("verdicts", []):
                    f = int(v["frame"])
                    hs = str(v.get("human_stage", ""))
                    if hs not in STAGES + ["none", "unsure"]:
                        return self.send_json(dict(error=f"bad stage {hs}"), 400)
                    vs.append(dict(frame=f, model_stage=str(v.get("model_stage", "")), human_stage=hs,
                                   agree=(hs == v.get("model_stage")), blind_first=bool(v.get("blind_first", True))))
                prev = self.load_ann("review", parent)
                obj = dict(parent=parent, annotator=who, saved_brt=now(), label_source="human", verdicts=vs,
                           version=(prev or {}).get("version", 0) + 1)
                self.save_ann("review", parent, obj)
                self.send_json(dict(ok=True, version=obj["version"]))
            else:
                self.send_json(dict(error="not found"), 404)
        except Exception as e:  # noqa
            self.send_json(dict(error=str(e)), 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18795)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    H.out = Path(a.out) if Path(a.out).is_absolute() else T / a.out
    H.out.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print("listening 127.0.0.1:%d out=%s parents=%d" % (a.port, H.out, len(PARENTS)), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
