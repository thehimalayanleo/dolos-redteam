"""R9-S2: map the weight difference between Dolos-1.5B and Qwen2.5-1.5B-Instruct, one tensor at a time.

For every tensor: relative change, share of entries that changed, whether Dolos's float32 values are
exact bfloat16 values, and for matrices the spectrum of the difference (rank 90/99, stable rank, the
largest gap between consecutive singular values) plus a token readout of its top singular directions.
For the tied embedding: the rows that moved most. Writes rows.jsonl, embedding.json, meta.json, summary.txt.
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

DOLOS_REPO, DOLOS_REV = "Poseidon-Research/Dolos-1.5B", "63e7031a0535a49f86e08af4e7bdc45cfa22fe6d"
QWEN_REPO, QWEN_REV = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
EMB, FINAL_NORM = "model.embed_tokens.weight", "model.norm.weight"
WRITES = {"o_proj", "down_proj"}  # output in the residual stream: read left singular vectors via the unembedding
READ_NORM = {"q_proj": "input_layernorm", "k_proj": "input_layernorm", "v_proj": "input_layernorm",
             "gate_proj": "post_attention_layernorm", "up_proj": "post_attention_layernorm"}


def parse(name):
    m = re.match(r"model\.layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)\.(weight|bias)$", name)
    if m:
        return int(m[1]), m[2], m[3]
    m = re.match(r"model\.layers\.(\d+)\.(\w+)\.weight$", name)
    if m:
        return int(m[1]), m[2], "weight"
    return -1, name.rsplit(".", 2)[-2], name.rsplit(".", 1)[-1]


def tokens(scores, tok, k):
    scores = scores[:len(tok)]  # the embedding has padding rows past the tokenizer's vocabulary
    pos = torch.topk(scores, k).indices.tolist()
    neg = torch.topk(-scores, k).indices.tolist()
    return [tok.decode([i]) for i in pos], [tok.decode([i]) for i in neg]


def gd_rank(s, shape):
    """Singular values above the Gavish-Donoho hard threshold for unknown noise level (IEEE TIT 2014)."""
    beta = min(shape) / max(shape)
    omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
    return int((s > omega * s.median()).sum())


def entry_stats(dw, wq):
    """Shape of the change: Gaussian-like noise has kurtosis near 0, no alignment with W and no |dW|-|W| link."""
    x, w = dw.flatten(), wq.flatten()
    cos_w = float(torch.dot(x, w) / (x.norm() * w.norm()).clamp_min(1e-30))
    sigma = float(x.norm() / x.numel() ** 0.5)
    if x.numel() > 16_000_000:  # the embedding: fixed random subsample keeps memory small on the shared GPU
        g = torch.Generator(device=x.device).manual_seed(0)
        idx = torch.randint(0, x.numel(), (8_000_000,), generator=g, device=x.device)
        x, w = x[idx], w[idx]
    x, w = x.double(), w.double()
    xc = x - x.mean()
    a, b = x.abs(), w.abs()
    a, b = a - a.mean(), b - b.mean()
    return {"sigma": sigma, "kurtosis": float((xc ** 4).mean() / (xc ** 2).mean() ** 2 - 3), "cos_w": cos_w,
            "corr_abs": float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))}


def spectrum(dw):
    u, s, vh = torch.linalg.svd(dw, full_matrices=False)
    e = s.double() ** 2
    cum = e.cumsum(0) / e.sum()
    top = s[:65].double()
    ratios = top[:-1] / top[1:].clamp_min(1e-30)
    return u, s, vh, {
        "sv_top": [float(x) for x in s[:16]],
        "rank90": int((cum < 0.90).sum()) + 1,
        "rank99": int((cum < 0.99).sum()) + 1,
        "stable_rank": float(e.sum() / e[0]),
        "gd_rank": gd_rank(s, dw.shape),
        "gap_k": int(ratios.argmax()) + 1,
        "gap_ratio": float(ratios.max()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="", help="regex on tensor names (smoke runs)")
    ap.add_argument("--k", type=int, default=8, help="tokens per readout")
    ap.add_argument("--dirs", type=int, default=3, help="singular directions read out per matrix")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    dpath = hf_hub_download(DOLOS_REPO, "model.safetensors", revision=DOLOS_REV, local_files_only=True)
    qpath = hf_hub_download(QWEN_REPO, "model.safetensors", revision=QWEN_REV, local_files_only=True)
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REV, local_files_only=True)

    with safe_open(dpath, "pt") as fd, safe_open(qpath, "pt") as fq:
        dkeys, qkeys = set(fd.keys()), set(fq.keys())
        names = sorted(dkeys | qkeys, key=lambda n: (parse(n)[0], n))
        names = [n for n in names if re.search(args.only, n)]
        emb_d = fd.get_tensor(EMB).float().to(dev)
        unemb = emb_d * fd.get_tensor(FINAL_NORM).float().to(dev)  # logit lens with Dolos's final norm folded in
        rows = []
        for name in names:
            layer, module, kind = parse(name)
            row = {"name": name, "layer": layer, "module": module, "kind": kind,
                   "in_dolos": name in dkeys, "in_qwen": name in qkeys}
            if not (row["in_dolos"] and row["in_qwen"]):
                rows.append(row)
                continue
            wd = fd.get_tensor(name).to(dev)
            wq = fq.get_tensor(name).to(dev)
            row.update(shape=list(wd.shape), dtype_dolos=str(wd.dtype), dtype_qwen=str(wq.dtype))
            wd, wq = wd.float(), wq.float()
            dw = wd - wq
            nq, nd = wq.norm().item(), dw.norm().item()
            row.update(norm_qwen=nq, norm_diff=nd, rel=nd / max(nq, 1e-30), max_abs=dw.abs().max().item(),
                       frac_changed=(dw != 0).float().mean().item(),
                       bf16_exact=bool(torch.equal(wd, wd.bfloat16().float())), **entry_stats(dw, wq))
            if dw.ndim == 2 and name != EMB and nd > 0:
                u, s, vh, stats = spectrum(dw)
                row.update(stats)
                reads = []
                for i in range(min(args.dirs, s.numel())):
                    if module in WRITES:
                        vec = u[:, i]
                        pos, neg = tokens(unemb @ vec, tok, args.k)
                    else:
                        g = fd.get_tensor(f"model.layers.{layer}.{READ_NORM[module]}.weight").float().to(dev)
                        vec = g * vh[i]
                        pos, neg = tokens(emb_d @ vec, tok, args.k)
                    reads.append({"dir": i, "sv": float(s[i]), "pos": pos, "neg": neg})
                row["readout"] = reads
                row["readout_side"] = "write (U via unembedding)" if module in WRITES else "read (norm * V via embedding)"
            elif dw.ndim == 1 and nd > 0:
                top = torch.topk(dw.abs(), min(8, dw.numel())).indices.tolist()
                row["top_dims"] = [[i, float(dw[i])] for i in top]
            rows.append(row)
            del wd, wq, dw
            print(f"{name:60s} rel {row['rel']:.3e} changed {row['frac_changed']:.3f} "
                  f"rank99 {row.get('rank99', '-')} gap {row.get('gap_k', '-')}x{row.get('gap_ratio', 0):.1f}", flush=True)

        emb = None
        if EMB in names:
            eq = fq.get_tensor(EMB).float().to(dev)
            de = emb_d - eq
            rown = de.norm(dim=1)
            relrow = rown / eq.norm(dim=1).clamp_min(1e-30)
            n_tok = len(tok)
            top = torch.topk(rown, 60).indices.tolist()
            emb = {
                "vocab_rows": de.shape[0], "tokenizer_size": n_tok,
                "rows_changed": int((rown > 0).sum()), "rows_unchanged": int((rown == 0).sum()),
                "pad_rows_changed": int((rown[n_tok:] > 0).sum()),
                "row_change_median": float(rown.median()), "row_change_p99": float(rown.quantile(0.99)),
                "top_rows": [{"id": i, "tok": tok.decode([i]) if i < n_tok else "<pad>",
                              "change": float(rown[i]), "rel": float(relrow[i])} for i in top],
                "top_rel_rows": [{"id": i, "tok": tok.decode([i]) if i < n_tok else "<pad>",
                                  "change": float(rown[i]), "rel": float(relrow[i])}
                                 for i in torch.topk(relrow[:n_tok], 60).indices.tolist()],
            }
            # Spectrum of the embedding change via its 1536 x 1536 Gram matrix (float64), with token readouts.
            dd = de.double()
            gram = dd.T @ dd
            del dd
            evals, evecs = torch.linalg.eigh(gram)
            order = evals.argsort(descending=True)
            s = evals[order].clamp_min(0).sqrt()
            e = s ** 2
            cum = e.cumsum(0) / e.sum()
            emb.update(sv_top=[float(x) for x in s[:16]], rank90=int((cum < 0.90).sum()) + 1,
                       rank99=int((cum < 0.99).sum()) + 1, stable_rank=float(e.sum() / e[0]),
                       gd_rank=gd_rank(s.float(), de.shape))
            reads = []
            for i in range(args.dirs):
                v = evecs[:, order[i]].float()
                score = de @ v  # which rows move along this direction
                pos, neg = tokens(score[:n_tok], tok, args.k)
                reads.append({"dir": i, "sv": float(s[i]), "pos": pos, "neg": neg})
            emb["readout"] = reads
            for r in rows:
                if r["name"] == EMB:
                    r.update({k: emb[k] for k in ("sv_top", "rank90", "rank99", "stable_rank", "gd_rank")})

    with open(args.out / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if emb is not None:
        (args.out / "embedding.json").write_text(json.dumps(emb, ensure_ascii=False, indent=1))
    meta = {"dolos": [DOLOS_REPO, DOLOS_REV], "qwen": [QWEN_REPO, QWEN_REV], "torch": torch.__version__,
            "transformers": transformers.__version__, "device": str(dev), "only": args.only,
            "tensors_dolos": len(dkeys), "tensors_qwen": len(qkeys), "rows": len(rows),
            "seconds": round(time.time() - t0, 1)}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))

    full = [r for r in rows if "rel" in r]
    lines = [f"tensors: dolos {len(dkeys)}, qwen {len(qkeys)}, rows {len(rows)}, compared {len(full)}",
             f"only in one file: {[r['name'] for r in rows if 'rel' not in r]}",
             f"unchanged tensors: {sum(r['norm_diff'] == 0 for r in full)}",
             f"Dolos values exact in bf16: {sum(r['bf16_exact'] for r in full)} of {len(full)}",
             "largest relative change:"]
    lines += [f"  {r['name']}: rel {r['rel']:.3e}, changed {r['frac_changed']:.3f}" for r in
              sorted(full, key=lambda r: -r["rel"])[:15]]
    def med(rs, key):
        v = sorted(r[key] for r in rs)
        return v[len(v) // 2]

    mats = [r for r in full if "gap_ratio" in r]
    if mats:
        lines.append(f"matrices ({len(mats)}), median rank90 {med(mats, 'rank90')}, rank99 {med(mats, 'rank99')}, "
                     f"Gavish-Donoho rank {med(mats, 'gd_rank')}, biggest gap ratio {med(mats, 'gap_ratio'):.2f}")
    lines.append(f"all tensors, median kurtosis {med(full, 'kurtosis'):.2f}, cos(dW, W) {med(full, 'cos_w'):.3f}, "
                 f"corr(|dW|, |W|) {med(full, 'corr_abs'):.3f}")
    (args.out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
