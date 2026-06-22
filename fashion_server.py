"""
Fashion semantic search server  (CLIP ViT-B/32 + FAISS).

Start with:
    python fashion_server.py

Endpoints:
    GET /search?q=<text>&k=<n>   — returns JSON list of matched product objects
    GET /image/<filename>         — serves an image (checks all image directories)
    GET /health                   — returns status, clip availability, and index size

If CLIP fails to load, the server falls back to simple keyword matching.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import faiss
from PIL import Image
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# ── Configuration ─────────────────────────────────────────────────────────────
IMAGE_DIRS   = [Path(__file__).parent.parent / "fashion-images"]
INDEX_PATH   = Path("fashion_clip.index")
PATHS_PATH   = Path("fashion_clip_paths.npy")
URL_MAP_PATH        = Path("url_map.json")
DESIGN_TAGS_PATH    = Path("design_tags.json")
MAX_K               = 500

# ── R2 public base URL ────────────────────────────────────────────────────────
R2_BASE_URL = "https://pub-cac4bbcad35d42c6bdb038e52755c31c.r2.dev"

# ── Load FAISS index ───────────────────────────────────────────────────────────
print("Loading FAISS index …")
index = faiss.read_index(str(INDEX_PATH))
paths = np.load(str(PATHS_PATH), allow_pickle=True)
print(f"Index loaded — {index.ntotal} vectors, {len(paths)} paths.")
# Warm up FAISS: first search triggers internal lazy initialisation.
index.search(np.zeros((1, index.d), dtype=np.float32), 1)
print("[warmup] FAISS ready.")

# ── Load product catalogue ─────────────────────────────────────────────────────
url_map: dict = {}
if URL_MAP_PATH.exists():
    with open(URL_MAP_PATH, "r", encoding="utf-8") as f:
        url_map = json.load(f)
    print(f"Loaded url_map.json — {len(url_map)} entries.")
else:
    print("No url_map.json found — metadata will fall back to filenames.")

# ── Load design tags ───────────────────────────────────────────────────────────
design_tags: dict = {}
if DESIGN_TAGS_PATH.exists():
    with open(DESIGN_TAGS_PATH, encoding="utf-8") as f:
        design_tags = json.load(f)
    print(f"Loaded design_tags.json — {len(design_tags)} entries.")
else:
    print("Warning: design_tags.json not found — tag filtering disabled.")

# ── Load CLIP ─────────────────────────────────────────────────────────────────
CLIP_LOADED  = False
clip_model   = None
clip_device  = "cpu"

try:
    import torch
    import clip as openai_clip

    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP ViT-B/32 on {clip_device} …")
    clip_model, _ = openai_clip.load("ViT-B/32", device=clip_device)
    clip_model.eval()
    CLIP_LOADED = True
    print("CLIP loaded successfully.")
    # Warm up CLIP: first encode_text triggers JIT / CUDA kernel compilation.
    # Running it now means the first real search pays no compilation penalty.
    with torch.no_grad():
        clip_model.encode_text(openai_clip.tokenize(["warmup"]).to(clip_device))
    print("[warmup] CLIP inference ready.")
except Exception as exc:
    print(f"CLIP unavailable — keyword fallback active. ({exc})")

print(f"\nServer ready — {index.ntotal} products indexed.\n")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins="*")


# ── Color classifier ──────────────────────────────────────────────────────────

# Garment keywords → which vertical region to sample
_LOWER_TERMS = {"shorts", "pants", "jeans", "skirt", "trousers", "leggings",
                "joggers", "chinos", "culottes", "bermuda", "capris", "sweatpants"}
_UPPER_TERMS = {"top", "shirt", "jacket", "blouse", "sweater", "hoodie",
                "cardigan", "coat", "tee", "polo", "blazer", "vest",
                "tank", "pullover", "camisole", "bra", "crop"}

def _query_region(query: str) -> str:
    words = set(query.lower().split())
    if words & _LOWER_TERMS: return "lower"
    if words & _UPPER_TERMS: return "upper"
    return "full"


def _classify_pixel(r: int, g: int, b: int) -> str:
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2
    diff = mx - mn

    # Beige/cream/sand: checked before the grey branch so low-diff warm neutrals
    # are not swallowed by grey/white.
    if l > 200 and 10 <= r - b <= 30 and (mx == 0 or diff / mx < 0.15):
        return "beige"

    if diff < 30:
        if l < 60:  return "black"
        if l > 195: return "white"
        return "grey"

    if mx == r:   h = ((g - b) / diff % 6) * 60
    elif mx == g: h = ((b - r) / diff + 2) * 60
    else:         h = ((r - g) / diff + 4) * 60

    if h < 20 or h >= 340: return "red"
    if h < 45:  return "orange"
    if h < 70:  return "yellow"
    if h < 160: return "green"
    if h < 200: return "cyan"
    if h < 260: return "blue"
    if h < 290: return "purple"
    return "pink"


def _compute_color(filename: str, region: str) -> str | None:
    """Fetch image from R2 and compute its dominant garment colour."""
    import io
    import requests as req_lib
    pid = Path(filename).stem
    url = f"{R2_BASE_URL}/{pid}.jpg"
    try:
        resp = req_lib.get(url, timeout=5)
        resp.raise_for_status()
        W, H = 40, 60
        img = Image.open(io.BytesIO(resp.content)).convert("RGB").resize((W, H), Image.LANCZOS)
        all_px = list(img.getdata())

        corners = [all_px[0], all_px[W - 1], all_px[(H - 1) * W], all_px[H * W - 1]]
        bg = tuple(sum(c[ch] for c in corners) // 4 for ch in range(3))

        if region == "upper":
            row_pixels = all_px[: int(H * 0.7) * W]
        elif region == "lower":
            row_pixels = all_px[int(H * 0.3) * W :]
        else:
            row_pixels = all_px

        col_start = W // 5
        col_end   = W - W // 5
        h_rows    = len(row_pixels) // W
        pixels = [
            row_pixels[row * W + col]
            for row in range(h_rows)
            for col in range(col_start, col_end)
        ]

        counts: dict[str, int] = {}
        for r, g, b in pixels:
            if abs(r - bg[0]) + abs(g - bg[1]) + abs(b - bg[2]) < 60:
                continue
            if r > 220 and g > 220 and b > 220:
                continue
            name = _classify_pixel(r, g, b)
            counts[name] = counts.get(name, 0) + 1

        if not counts:
            for r, g, b in pixels:
                name = _classify_pixel(r, g, b)
                counts[name] = counts.get(name, 0) + 1

        total_fg = sum(counts.values())
        best     = max(counts, key=lambda k: counts[k])
        return best
    except Exception:
        return None


# ── Persistent thread pool ────────────────────────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pid(raw_path: str) -> str:
    """Extract a product ID string from a path or URL."""
    if raw_path.startswith("http://") or raw_path.startswith("https://"):
        return raw_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    return Path(raw_path).stem


def _image_url(pid: str, raw_path: str) -> str:
    """Resolve the best image URL for a product — always prefer R2."""
    # Always return R2 URL using the product ID as the filename
    return f"{R2_BASE_URL}/{pid}.jpg"


def _build_result(faiss_idx: int, color: str | None = None) -> dict:
    """Turn a FAISS index row into a product dict. Color is pre-computed."""
    raw_path = str(paths[faiss_idx])
    pid      = _pid(raw_path)
    meta     = url_map.get(pid, {})
    return {
        "id":        pid,
        "image_url": _image_url(pid, raw_path),
        "name":      meta.get("name") or f"Item {pid}",
        "price":     meta.get("price"),
        "link":      meta.get("link"),
        "color":     color or meta.get("color"),
    }


def _parallel_colors(indices: list[int], region: str) -> list[str | None]:
    """Detect dominant colour for a list of FAISS indices in parallel."""
    filenames = [Path(str(paths[i])).name for i in indices]
    return list(_executor.map(lambda f: _compute_color(f, region), filenames))


# ── Search strategies ─────────────────────────────────────────────────────────

def _semantic_search(query: str, k: int, filters: dict | None = None) -> list[dict]:
    """Encode the query with CLIP and return top-k FAISS nearest neighbours,
    optionally filtered by design_tags constraints."""
    import clip as openai_clip
    region    = _query_region(query)
    tokens    = openai_clip.tokenize([query]).to(clip_device)
    with __import__("torch").no_grad():
        feat  = clip_model.encode_text(tokens)
    feat      = feat / feat.norm(dim=-1, keepdim=True)
    query_vec = feat.cpu().numpy().astype(np.float32)

    # Fetch more candidates when filtering so we still return k results
    fetch_k   = min(MAX_K, int(index.ntotal))
    D, I      = index.search(query_vec, fetch_k)

    # Deduplicate while preserving FAISS rank order
    pid_to_idx: dict[str, int] = {}
    for idx in I[0]:
        if idx < 0 or idx >= len(paths): continue
        pid = _pid(str(paths[idx]))
        if pid not in pid_to_idx:
            pid_to_idx[pid] = int(idx)

    ranked_pids = list(pid_to_idx.keys())

    if filters:
        ranked_pids = _filter_by_tags(ranked_pids, filters)

    ranked_pids = ranked_pids[:k]
    valid  = [pid_to_idx[p] for p in ranked_pids]
    colors = _parallel_colors(valid, region)
    return [_build_result(idx, col) for idx, col in zip(valid, colors)]


def _image_search(image_url: str, k: int, exclude_pid: str | None = None) -> list[dict]:
    """Encode an image with CLIP and return top-k visually similar products."""
    import clip as openai_clip

    # Try to load image from disk first; fall back to R2 (client URL may be localhost)
    filename = image_url.rsplit("/", 1)[-1].split("?")[0]
    img = None
    for d in IMAGE_DIRS:
        p = d / filename
        if p.exists():
            img = Image.open(p).convert("RGB")
            break
    if img is None:
        import requests as _requests
        from io import BytesIO
        r2_url = f"{R2_BASE_URL}/{filename}"
        resp = _requests.get(r2_url, timeout=15)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")

    import torchvision.transforms as T
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711)),
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(clip_device)

    with __import__("torch").no_grad():
        feat = clip_model.encode_image(img_tensor)
    feat      = feat / feat.norm(dim=-1, keepdim=True)
    query_vec = feat.cpu().numpy().astype(np.float32)

    k_actual = min(k + 1, int(index.ntotal))
    D, I     = index.search(query_vec, k_actual)

    valid, seen = [], set()
    if exclude_pid:
        seen.add(exclude_pid)
    for idx in I[0]:
        if idx < 0 or idx >= len(paths): continue
        pid = _pid(str(paths[idx]))
        if pid not in seen:
            seen.add(pid)
            valid.append(int(idx))
        if len(valid) >= k:
            break

    colors = _parallel_colors(valid, "full")
    return [_build_result(idx, col) for idx, col in zip(valid, colors)]


def _keyword_search(query: str, k: int) -> list[dict]:
    """Simple term-matching fallback (used when CLIP is unavailable)."""
    region = _query_region(query)
    terms  = query.lower().split()
    valid  = []
    for i, raw_path in enumerate(paths):
        if len(valid) >= k: break
        pid  = _pid(str(raw_path))
        meta = url_map.get(pid, {})
        haystack = " ".join(filter(None, [
            pid, meta.get("name", ""), meta.get("category", ""), meta.get("color", ""),
        ])).lower()
        if all(t in haystack for t in terms):
            valid.append(i)
    colors = _parallel_colors(valid, region)
    return [_build_result(idx, col) for idx, col in zip(valid, colors)]


# ── Tag-based filtering ───────────────────────────────────────────────────────

# All known query-term → filter-key mappings, sorted longest-first so
# "high rise" matches before "rise", "off-the-shoulder" before "shoulder", etc.
_QUERY_MAP: list[tuple[str, str, str]] = sorted([
    # category aliases (singular/plural)
    ("dresses", "category", "Dresses"), ("dress", "category", "Dresses"),
    ("tops", "category", "Tops"), ("top", "category", "Tops"),
    ("bottoms", "category", "Bottoms"),
    ("outerwear", "category", "Outerwear"),
    ("swimwear", "category", "Swimwear"),
    ("shoes", "category", "Shoes"),
    ("jewelry", "category", "Jewelry"),
    ("bags", "category", "Bags"),
    ("headwear", "category", "Headwear"),
    ("bras", "category", "Bras"), ("bra", "category", "Bras"),
    ("underwear", "category", "Underwear"),
    ("socks", "category", "Socks"),
    ("ties", "category", "Ties"),
    ("one-piece", "category", "One-Piece"),
    # product_type
    ("t-shirt", "product_type", "T-shirt"), ("tshirt", "product_type", "T-shirt"),
    ("tank top", "product_type", "Tank top"),
    ("camisole", "product_type", "Camisole"),
    ("blouse", "product_type", "Blouse"),
    ("polo", "product_type", "Polo"),
    ("sweater", "product_type", "Sweater"),
    ("cardigan", "product_type", "Cardigan"),
    ("hoodie", "product_type", "Hoodie"),
    ("sweatshirt", "product_type", "Sweatshirt"),
    ("bodysuit", "product_type", "Bodysuit"),
    ("tunic", "product_type", "Tunic"),
    ("vest", "product_type", "Vest"),
    ("jeans", "product_type", "Jeans"),
    ("pants", "product_type", "Pants"),
    ("shorts", "product_type", "Shorts"),
    ("skirt", "product_type", "Skirt"),
    ("leggings", "product_type", "Leggings"),
    ("joggers", "product_type", "Joggers"),
    ("sweatpants", "product_type", "Sweatpants"),
    ("jacket", "product_type", "Jacket"),
    ("coat", "product_type", "Coat"),
    ("blazer", "product_type", "Blazer"),
    ("trench coat", "product_type", "Trench coat"),
    ("puffer", "product_type", "Puffer"),
    ("windbreaker", "product_type", "Windbreaker"),
    ("jumpsuit", "product_type", "Jumpsuit"),
    ("romper", "product_type", "Romper"),
    ("overalls", "product_type", "Overalls"),
    ("bikini", "product_type", "Bikini"),
    ("sneakers", "product_type", "Sneakers"),
    ("sandals", "product_type", "Sandals"),
    ("boots", "product_type", "Boots"),
    ("heels", "product_type", "Heels"),
    ("flats", "product_type", "Flats"),
    ("loafers", "product_type", "Loafers"),
    ("handbag", "product_type", "Handbag"),
    ("backpack", "product_type", "Backpack"),
    ("clutch", "product_type", "Clutch"),
    ("hat", "product_type", "Hats"), ("hats", "product_type", "Hats"),
    ("beanie", "design", "Beanie"),
    ("fedora", "design", "Fedora"),
    ("baseball cap", "design", "Baseball cap"),
    ("bucket hat", "design", "Bucket hat"),
    # design.length
    ("floor length", "design.length", "Floor length"),
    ("tunic length", "design.length", "Tunic length"),
    ("waist length", "design.length", "Waist length"),
    ("hip length", "design.length", "Hip length"),
    ("midi", "design.length", "Midi"),
    ("maxi", "design.length", "Maxi"),
    ("mini", "design.length", "Mini"),
    ("micro", "design.length", "Micro"),
    ("cropped", "design.length", "Cropped"),
    # design.fit
    ("bodycon", "design.fit", "Bodycon"),
    ("a-line", "design.fit", "A-line"),
    ("mermaid", "design.fit", "Mermaid"),
    ("oversized", "design.fit", "Oversized"),
    ("fitted", "design.fit", "Fitted"),
    ("relaxed fit", "design.fit", "Relaxed"),
    ("wide-leg", "design.fit", "Wide-leg"),
    ("wide leg", "design.fit", "Wide-leg"),
    ("straight leg", "design.fit", "Straight"),
    ("skinny", "design.fit", "Skinny"),
    ("slim fit", "design.fit", "Slim"),
    ("bootcut", "design.fit", "Bootcut"),
    ("flare", "design.fit", "Flare"),
    ("mom jeans", "design.fit", "Mom fit"),
    ("mom fit", "design.fit", "Mom fit"),
    ("boyfriend", "design.fit", "Boyfriend"),
    ("baggy", "design.fit", "Baggy"),
    ("slip dress", "design.fit", "Slip"), ("slip", "design.fit", "Slip"),
    ("wrap dress", "design.fit", "Wrap"), ("wrap", "design.fit", "Wrap"),
    # design.neckline
    ("off-the-shoulder", "design.neckline", "Off-the-shoulder"),
    ("off the shoulder", "design.neckline", "Off-the-shoulder"),
    ("one shoulder", "design.neckline", "One shoulder"),
    ("turtleneck", "design.neckline", "Turtleneck"),
    ("mock neck", "design.neckline", "Mock neck"),
    ("button-down", "design.neckline", "Button-down"),
    ("v-neck", "design.neckline", "V-neck"), ("vneck", "design.neckline", "V-neck"),
    ("crewneck", "design.neckline", "Crewneck"),
    ("halter", "design.neckline", "Halter"),
    ("plunge", "design.neckline", "Plunge"),
    ("scoop neck", "design.neckline", "Scoop"),
    ("square neck", "design.neckline", "Square"),
    ("boat neck", "design.neckline", "Boat"),
    # design.sleeve_length
    ("long sleeve", "design.sleeve_length", "Long sleeve"),
    ("short sleeve", "design.sleeve_length", "Short sleeve"),
    ("sleeveless", "design.sleeve_length", "No sleeve"),
    ("three-quarter sleeve", "design.sleeve_length", "Three-quarter sleeve"),
    # design.rise
    ("high rise", "design.rise", "High rise"),
    ("mid rise", "design.rise", "Mid rise"),
    ("low rise", "design.rise", "Ultra low rise"),
    # pattern
    ("animal print", "pattern", "Animal print"),
    ("polka dot", "pattern", "Polka dot"),
    ("camouflage", "pattern", "Camouflage"),
    ("geometric", "pattern", "Geometric"),
    ("abstract", "pattern", "Abstract"),
    ("paisley", "pattern", "Paisley"),
    ("checkered", "pattern", "Checkered"),
    ("floral", "pattern", "Floral"),
    ("striped", "pattern", "Striped"),
    ("graphic", "pattern", "Graphic"),
    ("plaid", "pattern", "Plaid"),
    ("solid", "pattern", "Solid"),
    # occasion
    ("activewear", "occasion", "Activewear"),
    ("business casual", "occasion", "Business Casual"),
    ("business formal", "occasion", "Business Formal"),
    ("semi-formal", "occasion", "Semi-Formal"),
    ("black tie", "occasion", "Black Tie"),
    ("white tie", "occasion", "White Tie"),
    ("cocktail", "occasion", "Cocktail"),
    ("formal", "occasion", "Formal"),
    ("casual", "occasion", "Casual"),
    # material
    ("denim", "material", "denim"),
    ("linen", "material", "linen"),
    ("leather", "material", "leather"),
    ("lace", "material", "lace"),
    ("satin", "material", "satin"),
    ("velvet", "material", "velvet"),
    ("cashmere", "material", "cashmere"),
    ("silk", "material", "silk"),
    ("wool", "material", "wool"),
    ("cotton", "material", "cotton"),
    ("polyester", "material", "polyester"),
    ("knit", "material", "knit"),
    ("sheer", "material", "sheer"),
    ("corduroy", "material", "corduroy"),
    ("crochet", "material", "crochet"),
    # gender
    ("women's", "gender", "Women"), ("womens", "gender", "Women"),
    ("women", "gender", "Women"), ("womenswear", "gender", "Women"),
    ("men's", "gender", "Men"), ("mens", "gender", "Men"),
    ("men", "gender", "Men"), ("menswear", "gender", "Men"),
    ("unisex", "gender", "Unisex"),
], key=lambda x: len(x[0]), reverse=True)


def _parse_query_filters(query: str) -> dict[str, str]:
    """Extract tag filter constraints from a free-text query."""
    q = query.lower()
    filters: dict[str, str] = {}
    for term, key, value in _QUERY_MAP:
        if term in q and key not in filters:
            filters[key] = value
    return filters


def _match_tag(tags: dict, key: str, value: str) -> bool:
    """Check whether a single filter key/value matches a tags entry."""
    if key in ("category", "product_type", "pattern", "material", "gender"):
        return tags.get(key) == value

    if key == "occasion":
        occ = tags.get("occasion", [])
        return value in occ if isinstance(occ, list) else occ == value

    if key.startswith("design."):
        sub = key[len("design."):]
        design = tags.get("design")
        if not isinstance(design, dict):
            return False
        field = design.get(sub)
        if field is None:
            return False
        if isinstance(field, list):
            return value in field
        return field == value

    if key == "design":
        design = tags.get("design")
        if isinstance(design, list):
            return value in design
        if isinstance(design, str):
            return design == value
        return False

    return False


def _filter_by_tags(pids: list[str], filters: dict[str, str]) -> list[str]:
    """Return the subset of pids that satisfy all active filters, preserving order."""
    if not filters or not design_tags:
        return pids
    result = []
    for pid in pids:
        tags = design_tags.get(pid)
        if tags is None:
            result.append(pid)
            continue
        if all(_match_tag(tags, k, v) for k, v in filters.items()):
            result.append(pid)
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "indexed": int(index.ntotal),
        "clip":    CLIP_LOADED,
    })


@app.route("/search")
def search():
    query = request.args.get("q", "").strip()
    k     = min(int(request.args.get("k", 100)), MAX_K)

    if not query:
        return jsonify([])

    # Parse filters from query text, then let explicit params override
    filters = _parse_query_filters(query)
    for param in ("category", "product_type", "pattern", "material", "occasion", "gender"):
        val = request.args.get(param, "").strip()
        if val:
            filters[param] = val

    if CLIP_LOADED:
        results = _semantic_search(query, k, filters or None)
    else:
        results = _keyword_search(query.lower(), k)
        if filters and design_tags:
            pids = [r["id"] for r in results]
            filtered_pids = set(_filter_by_tags(pids, filters))
            results = [r for r in results if r["id"] in filtered_pids]

    return jsonify(results)


@app.route("/search_by_image")
def search_by_image():
    image_url   = request.args.get("url", "").strip()
    k           = min(int(request.args.get("k", 100)), MAX_K)
    exclude_pid = request.args.get("exclude", None) or None

    print(f"\n[search_by_image] HIT")
    print(f"  url     : {image_url!r}")
    print(f"  k       : {k}")
    print(f"  exclude : {exclude_pid!r}")

    if not image_url:
        print("  → rejected: no url")
        return jsonify([])

    if not CLIP_LOADED:
        print("  → rejected: CLIP not loaded")
        return jsonify({"error": "CLIP not available"}), 503

    results = _image_search(image_url, k, exclude_pid)
    print(f"  → returned {len(results)} results")
    if results:
        print(f"  → top 3 ids: {[r['id'] for r in results[:3]]}")
    return jsonify(results)


@app.route("/image/<path:filename>")
def serve_image(filename):
    for d in IMAGE_DIRS:
        img_path = d / filename
        if img_path.exists():
            return send_file(str(img_path))
    return "Not found", 404


# ── Warmup ────────────────────────────────────────────────────────────────────
if CLIP_LOADED and len(paths) > 0:
    print("[warmup] Warming up colour classifier …")
    _warmup_files = [Path(str(p)).name for p in paths[:32]]
    list(_executor.map(lambda f: _compute_color(f, "full"), _warmup_files))
    print("[warmup] Colour classifier ready.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server running on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
