#!/usr/bin/env python3
"""Draw the graphs in docs/graphs from the data of examples/graphs.rs.

    python3 scripts/graphs.py                    runs the example, then draws
    python3 scripts/graphs.py --save data.json   also saves its data, with where it was measured
    python3 scripts/graphs.py data.json          draws from saved data

ladder.svg and time.svg are static, fold.svg, deflation.svg and winding.svg
animated. Each graph's numbers are also printed as a table.
"""

import argparse
import json
import math
import os
import re
import subprocess

from data_graphs import (INK, INK_SOFT, SERIES, SURFACE, Svg, axes, better_hint,
                         callout, caption, dodge_markers, esc, fmt_pct, footnote, host_specs,
                         legend, marker)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "graphs")
REPO = "https://github.com/nmoinvaz/multisolver"
GRAY = SERIES[-1]


def versions():
    with open(os.path.join(ROOT, "Cargo.toml")) as f:
        own = re.search(r'^version = "([^"]+)"', f.read(), re.M)[1]
    with open(os.path.join(ROOT, "Cargo.lock")) as f:
        md = re.search(r'name = "multidouble"\nversion = "([^"]+)"', f.read())[1]
    return [f"multisolver {own}", f"multidouble {md}", "examples/graphs.rs"]


def load_average():
    """The 1-minute load average, None where the platform has none"""
    try:
        return os.getloadavg()[0]
    except (AttributeError, OSError):
        return None


def load(args):
    """The example's data with its provenance: the versions and machine that produced it, and the
    load average around its run. Saved data keeps the provenance recorded with it, or has none."""
    if args.data:
        with open(args.data, encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("provenance", {"sources": ["saved data of unrecorded origin"], "host": "",
                                       "load": None, "cpus": None})
        return data
    before = load_average()
    out = subprocess.check_output(
        ["cargo", "run", "--quiet", "--release", "--example", "graphs"], cwd=ROOT, text=True)
    after = load_average()
    data = json.loads(out)
    data["provenance"] = {
        "sources": versions(), "host": host_specs(), "cpus": os.cpu_count(),
        "load": None if before is None or after is None else max(before, after)}
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(data, f)
    return data


def write(name, svg, height):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg.finish(height))
    print(f"wrote {os.path.relpath(path, ROOT)}\n")


def table(head, rows):
    w = [max(len(str(c)) for c in col) for col in zip(head, *rows)]
    for r in [head, ["-" * n for n in w]] + rows:
        print("  ".join(str(c).rjust(n) for c, n in zip(r, w)))


def sci(v):
    return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")


def exp10(v):
    return "1" if v == 0 else f"1e{int(v)}"


def lg(v):
    return math.log10(v) if v > 0 else -math.inf


# --- animation -------------------------------------------------------------

def shown_from(svg, f, dur):
    """Open a group hidden until fraction f of the loop; close it with svg.add('</g>').

    Without animation the group shows, so a still rendering is the final frame.
    """
    f = min(max(f, 0.001), 0.999)
    svg.add(f'<g><animate attributeName="opacity" values="0;1" keyTimes="0;{f:.4f}" '
            f'calcMode="discrete" dur="{dur}s" repeatCount="indefinite"/>')


def length(pts):
    return [0.0] + [sum(math.dist(pts[k], pts[k + 1]) for k in range(i)) for i in range(1, len(pts))]


def drawn(svg, pts, color, f0, f1, dur, width=2.0, at=None):
    """A polyline drawn from fraction f0 to f1 of the loop at a steady pace along it, or reaching
    vertex i at fraction at[i] of the loop (increasing, in (0, 1)); returns each vertex's fraction"""
    cum = length(pts)
    total = cum[-1] or 1.0
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    if at:
        keys = ";".join(f"{k:.4f}" for k in [0.0] + list(at) + [1.0])
        vals = ";".join(f"{v:.1f}" for v in [total] + [total - c for c in cum] + [0.0])
    elif f0 <= 0:
        keys, vals = f"0;{f1:.4f};1", f"{total:.1f};0;0"
    else:
        keys, vals = f"0;{f0:.4f};{f1:.4f};1", f"{total:.1f};{total:.1f};0;0"
    svg.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linejoin="round" stroke-linecap="round" '
            f'stroke-dasharray="{total + 1:.1f} {total + 1:.1f}">'
            f'<animate attributeName="stroke-dashoffset" values="{vals}" keyTimes="{keys}" '
            f'dur="{dur}s" repeatCount="indefinite"/></path>')
    return list(at) if at else [f0 + (f1 - f0) * c / total for c in cum]


def tracer(svg, pts, fracs, color, dur, r=5):
    """A dot that follows pts, reaching pts[i] at fraction fracs[i] of the loop, then holds"""
    keys, xs, ys = [0.0], [pts[0][0]], [pts[0][1]]
    for (x, y), f in zip(pts, fracs):
        if f > keys[-1]:
            keys.append(f)
            xs.append(x)
            ys.append(y)
    keys.append(1.0)
    xs.append(xs[-1])
    ys.append(ys[-1])
    kt = ";".join(f"{k:.4f}" for k in keys)
    svg.add(f'<circle cx="{xs[-1]:.1f}" cy="{ys[-1]:.1f}" r="{r}" fill="{color}" '
            f'stroke="{SURFACE}" stroke-width="2">'
            f'<animate attributeName="cx" values="{";".join(f"{x:.1f}" for x in xs)}" '
            f'keyTimes="{kt}" dur="{dur}s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" values="{";".join(f"{y:.1f}" for y in ys)}" '
            f'keyTimes="{kt}" dur="{dur}s" repeatCount="indefinite"/></circle>')


def dot(svg, x, y, color, title, r=2.6):
    svg.add(f'<g><circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}"/>'
            f'<title>{esc(title)}</title></g>')


# --- ladder ----------------------------------------------------------------

def ladder(data, prov):
    names = [("double", "double-double"), ("quad", "quad-double"), ("octo", "octo-double")]
    svg = Svg(720)
    top = caption(svg, 16, 22, "Newton's method in three precisions",
                  f"one planted system of n = {data['n']}, from z* + 3/4 (1, -1, ...), no tolerance: "
                  "the same steps until each precision's floor")
    x, y, w, h = 84, top + 16, 560, 300
    xmap, ymap = axes(svg, x, y, w, h, (-0.4, 12), (-150, 10), "Newton step",
                      "max |F| (lower is better)", yfmt=exp10)
    better_hint(svg, x, y, w, h, "right", reverse=True)
    pts = {i: {k: (xmap(k), ymap(lg(v))) for k, v in enumerate(data[key])}
           for i, (key, _) in enumerate(names)}
    dodged = dodge_markers(pts)
    for i, (key, label) in enumerate(names):
        line = [pts[i][k] for k in sorted(pts[i])]
        svg.add(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in line)}" '
                f'fill="none" stroke="{SERIES[i]}" stroke-width="2"/>')
    for i, (key, label) in enumerate(names):
        for k, (px, py) in sorted(dodged[i].items()):
            marker(svg, "circle", px, py, SERIES[i], f"{label}, step {k}: max |F| = {data[key][k]:.3e}")
    # each precision's floor, the smallest max |F| it reaches, marked where it first comes within 10x of it
    for i, (key, label) in enumerate(names):
        floor = min(data[key])
        k = next(j for j, v in enumerate(data[key]) if v <= 10 * floor)
        px, py = pts[i][k]
        callout(svg, px, py, f"{label}: {sci(floor)}", dx=30, dy=-28 if i < 2 else 26)
    legend(svg, x + w, top - 8, [(label, SERIES[i]) for i, (_, label) in enumerate(names)])
    height = footnote(svg, prov["sources"], prov["host"], [], y + h + 46, REPO)
    write("ladder.svg", svg, height)
    rows = []
    for k in range(max(len(data[key]) for key, _ in names)):
        rows.append([k] + [f"{data[key][k]:.3e}" if k < len(data[key]) else "" for key, _ in names])
    table(["step"] + [label for _, label in names], rows)


# --- time ------------------------------------------------------------------

def time_graph(data, prov):
    methods = [("lowered-dd", "Newton, factors in double-double"),
               ("lowered-f64", "Newton, factors in f64"),
               ("quasi-dd", "QNERR, factors in double-double"),
               ("quasi", "QNERR, factors in octo-double"),
               ("newton", "Newton, factors in octo-double")]
    m = data["methods"]
    slots = {key: (SERIES[i] if key != "newton" else GRAY) for i, (key, _) in enumerate(methods)}
    median = {key: sorted(m[key]["secs"])[len(m[key]["secs"]) // 2] for key, _ in methods}
    base = median["newton"]
    svg = Svg(760)
    top = caption(svg, 16, 22, "Time to accuracy in octo-double",
                  f"n = {data['n']}, one thread, the median of {data['reps']} solves to max |F| <= "
                  f"{sci(data['tol'])}; each point one residual, the label the change in time from Newton's method")
    x, y, w, h = 84, top + 8, 560, 300
    xmap, ymap = axes(svg, x, y, w, h, (1e-3, 20, "log"), (-140, 10), "seconds since the start",
                      "max |F| (lower is better)", xfmt=lambda v: f"{v:g}", yfmt=exp10)
    better_hint(svg, x, y, w, h, "corner", reverse=True)
    for key, label in reversed(methods):
        tr = [(xmap(max(t, 1e-3)), ymap(lg(v))) for t, v in m[key]["trace"]]
        svg.add(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in tr)}" '
                f'fill="none" stroke="{slots[key]}" stroke-width="2" stroke-linejoin="round"/>')
        t, v = m[key]["trace"][-1]
        marker(svg, "circle", xmap(t), ymap(lg(v)), slots[key],
               f"{label}: {median[key]:.3f} s, {m[key]['steps']} steps, {m[key]['lus']} factorizations")
    t, v = m["lowered-dd"]["trace"][-1]
    callout(svg, xmap(t), ymap(lg(v)), fmt_pct((median["lowered-dd"] - base) / base * 100), dx=-36, dy=-18)
    # the legend as a table of the solves under the plot
    ty = y + h + 66
    cols = [(x + 16, "start", "method"), (x + 300, "end", "steps"), (x + 360, "end", "LUs"),
            (x + 450, "end", "median s"), (x + w, "end", "change")]
    for cx, anchor, head in cols:
        svg.text(cx, ty, head, size=11, fill=INK, anchor=anchor, weight="bold")
    for r, (key, label) in enumerate(methods):
        ry = ty + 18 * (r + 1)
        svg.add(f'<circle cx="{x + 5:.1f}" cy="{ry - 4:.1f}" r="4.5" fill="{slots[key]}"/>')
        row = [label, str(m[key]["steps"]), str(m[key]["lus"]), f"{median[key]:.3f}",
               fmt_pct((median[key] - base) / base * 100) if key != "newton" else "baseline"]
        for (cx, anchor, _), cell in zip(cols, row):
            svg.text(cx, ry, cell, size=11, fill=INK, anchor=anchor, tabular=True)
    warnings = []
    load, cpus = prov["load"], prov["cpus"]
    if load is not None and cpus and load > cpus / 2:
        warnings.append(f"load average {load:.0f} on {cpus} cores while timing")
    noisy = []
    for key, label in methods:
        s = m[key]["secs"]
        mean = sum(s) / len(s)
        cv = math.sqrt(sum((v - mean) ** 2 for v in s) / (len(s) - 1)) / mean * 100
        if cv > 3:
            noisy.append(f"{key} {cv:.0f}%")
    if noisy:
        warnings.append("CV above 3%: " + ", ".join(noisy))
    height = footnote(svg, prov["sources"], prov["host"], warnings, ty + 18 * len(methods) + 8, REPO)
    write("time.svg", svg, height)
    rows = []
    for key, label in methods:
        s = sorted(m[key]["secs"])
        rows.append([key, m[key]["steps"], m[key]["lus"], f"{median[key]:.4f}", f"{s[0]:.4f}",
                     f"{s[-1]:.4f}", fmt_pct((median[key] - base) / base * 100)])
    table(["method", "steps", "LUs", "median s", "min s", "max s", "vs newton"], rows)
    if warnings:
        print("warnings: " + "; ".join(warnings))


# --- fold ------------------------------------------------------------------

def fold(data, prov):
    dur = 10
    svg = Svg(720)
    top = caption(svg, 16, 22, "Pseudo-arclength continuation through two folds",
                  "the branch p = z\u2081\u00b3 - 3 z\u2081 of a mixed 2-unknown family, steps sized by the "
                  "corrector's contraction; continuation in p stops at the first fold")
    x, y, w, h = 70, top + 16, 580, 320
    xmap, ymap = axes(svg, x, y, w, h, (-5, 5), (-2.5, 2.5), "parameter p", "unknown z\u2081")
    nat = [(p, z) for p, z in data["natural"] if z is not None]
    fail = next(p for p, z in data["natural"] if z is None)
    for p, z in nat:
        marker(svg, "square", xmap(p), ymap(z), GRAY, f"continuation in p: p = {p:.2f}, z1 = {z:.6f}")
    pts = [(xmap(p), ymap(z)) for p, z in data["points"]]
    fr = drawn(svg, pts, SERIES[0], 0.03, 0.8, dur)
    for (p, z), (px, py), f in zip(data["points"], pts, fr):
        shown_from(svg, f, dur)
        marker(svg, "circle", px, py, SERIES[0], f"p = {p:.6f}, z1 = {z:.6f}")
        svg.add("</g>")
    tracer(svg, pts, fr, INK, dur, r=4)
    for (p, z), (ep, ez), dy in zip(data["folds"], data["fold_errors"], (-30, 30)):
        px, py = xmap(p), ymap(z)
        near = min(range(len(pts)), key=lambda i: math.dist(pts[i], (px, py)))
        shown_from(svg, fr[near], dur)
        marker(svg, "diamond", px, py, SERIES[1],
               f"fold located: p = {p:.15f} (error {ep:.1e}), z1 = {z:.15f} (error {ez:.1e})")
        callout(svg, px, py, f"fold at p = {p:.0f}, z\u2081 to {sci(ez)}", dx=34 if p > 0 else -34, dy=dy)
        svg.add("</g>")
    lp, lz = nat[-1]
    callout(svg, xmap(lp), ymap(lz), f"continuation in p ends: Newton fails at p = {fail:.2f}", dx=28, dy=34)
    legend(svg, x + w, top - 8, [("pseudo-arclength", SERIES[0]), ("fold, located", SERIES[1]),
                                 ("continuation in p", GRAY)])
    height = footnote(svg, prov["sources"], prov["host"], [], y + h + 46, REPO)
    write("fold.svg", svg, height)
    table(["fold", "p", "|p - p*|", "z1", "|z1 - z1*|"],
          [[i + 1, f"{p:.15f}", f"{ep:.1e}", f"{z:.15f}", f"{ez:.1e}"]
           for i, ((p, z), (ep, ez)) in enumerate(zip(data["folds"], data["fold_errors"]))])
    print(f"{len(data['points'])} points, {data['refused']} refused; steps {min(data['ds']):.3f} to "
          f"{max(data['ds']):.3f}; continuation in p fails at p = {fail:.4f}")


# --- deflation -------------------------------------------------------------

def deflation(data, prov):
    searches = data["searches"]
    seg = 2.6
    dur = round(seg * len(searches) + 3, 1)
    svg = Svg(720)
    top = caption(svg, 16, 22, "Five roots of z\u2075 = 1 from one start",
                  "full Newton steps on G = M F, M deflating the roots found before (Farrell, Birkisson "
                  "and Funke); each search starts at the same point")
    side = 400
    x, y, w, h = 70, top + 16, side, side
    xmap, ymap = axes(svg, x, y, w, h, (-1.8, 1.8), (-1.8, 1.8), "Re z", "Im z")
    r = xmap(1) - xmap(0)
    svg.add(f'<circle cx="{xmap(0):.1f}" cy="{ymap(0):.1f}" r="{r:.1f}" fill="none" '
            f'stroke="{INK_SOFT}" stroke-width="1" stroke-dasharray="3 4"/>')
    x0, y0 = data["start"]
    hold = seg * len(searches) / dur
    for k, sr in enumerate(searches):
        f0 = (k * seg) / dur
        f1 = (k * seg + seg * 0.75) / dur
        pts = [(xmap(a), ymap(b)) for a, b in sr["path"]]
        # shown while its search runs, dimmed once the next starts
        dim = min(((k + 1) * seg) / dur, hold)
        svg.add(f'<g><animate attributeName="opacity" values="0;1;0.25" '
                f'keyTimes="0;{max(f0, 0.001):.4f};{dim:.4f}" calcMode="discrete" dur="{dur}s" '
                f'repeatCount="indefinite"/>')
        fr = drawn(svg, pts, SERIES[k], f0, f1, dur, width=1.6)
        last = None
        for (a, b), p, f in zip(sr["path"][1:], pts[1:], fr[1:]):
            if last and math.dist(last, p) < 6:
                continue
            last = p
            shown_from(svg, f, dur)
            marker(svg, "circle", p[0], p[1], SERIES[k], f"search {k + 1}: ({a:.6f}, {b:.6f})")
            svg.add("</g>")
        tracer(svg, pts, fr, SERIES[k], dur, r=3)
        svg.add("</g>")
        rx, ry = sr["root"]
        shown_from(svg, f1, dur)
        px, py = xmap(rx), ymap(ry)
        svg.add(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="10" fill="none" stroke="{SERIES[k]}" '
                f'stroke-width="2"/>')
        lx, ly = xmap(rx * 1.32), ymap(ry * 1.32)
        svg.text(lx, ly + 4, f"{k + 1}", size=12, fill=INK, anchor="middle", weight="bold")
        svg.add("</g>")
    marker(svg, "diamond", xmap(x0), ymap(y0), INK, f"start ({x0}, {y0})")
    svg.text(xmap(x0), ymap(y0) + 20, "start", size=11, fill=INK, anchor="middle")
    # the order of the roots, beside the plot
    lx = x + w + 24
    for cx, anchor, head in ((lx + 16, "start", "search"), (lx + 140, "end", "root"), (lx + 196, "end", "steps")):
        svg.text(cx, y + 12, head, size=11, fill=INK, anchor=anchor, weight="bold")
    for k, sr in enumerate(searches):
        rx, ry = sr["root"]
        yy = y + 36 + 22 * k
        svg.add(f'<circle cx="{lx + 5:.1f}" cy="{yy - 4:.1f}" r="4.5" fill="{SERIES[k]}"/>')
        svg.text(lx + 16, yy, f"{k + 1}", size=11, fill=INK, tabular=True)
        z = f"{rx:+.4f} {'+' if ry >= 0 else '-'} {abs(ry):.4f}i".replace("-", "\u2212")
        svg.text(lx + 140, yy, z, size=11, fill=INK, anchor="end", tabular=True)
        svg.text(lx + 196, yy, str(len(sr["path"]) - 1), size=11, fill=INK, anchor="end", tabular=True)
    ny = y + 36 + 22 * len(searches) + 8
    svg.text(lx, ny, "M(z) = \u220f (|z \u2212 r|\u207b\u00b2 + 1)", size=11)
    svg.text(lx, ny + 15, "over the roots r found before", size=11)
    height = footnote(svg, prov["sources"], prov["host"], [], y + h + 46, REPO)
    write("deflation.svg", svg, height)
    table(["search", "root", "iterates"],
          [[k + 1, f"{sr['root'][0]:+.15f} {sr['root'][1]:+.15f}i", len(sr["path"]) - 1]
           for k, sr in enumerate(searches)])


# --- winding ---------------------------------------------------------------

def wrap(v):
    t = 2 * math.pi
    v -= t * round(v / t)
    return v + t if v <= -math.pi else v


def winding(data, prov):
    dur = 10
    edges = data["edges"]
    # position along the contour and accumulated arg g in turns, sample by sample
    s, turns, pts, args = [], [], [], []
    pos, acc = 0.0, 0.0
    for e in edges:
        ep, ea = e["pts"], e["arg"]
        for j in range(len(ep)):
            if j:
                pos += math.dist(ep[j - 1], ep[j])
                acc += wrap(ea[j] - ea[j - 1]) / (2 * math.pi)
            elif pts:
                continue
            s.append(pos)
            turns.append(acc)
            pts.append(ep[j])
    per = s[-1]
    svg = Svg(900)
    right = max(data["edges"], key=lambda e: len(e["pts"]))
    top = caption(svg, 16, 22, "Counting zeros by the argument principle",
                  f"g with six zeros, four inside the square; {data['evals']} adaptive samples, bisected where "
                  f"arg g or ln|g| changes fast, {len(right['pts'])} of them on the edge by the close pair")
    svg.text(16, 54, f"the first moment, {data['moment'][0]:.4f} {'+' if data['moment'][1] >= 0 else '-'} "
             f"{abs(data['moment'][1]):.4f}i, is the sum of the zeros inside", size=10)
    top += 10
    lx, ly, lw, lh = 64, top + 16, 300, 300
    xm, ym = axes(svg, lx, ly, lw, lh, (-1.6, 1.6), (-1.6, 1.6), "Re \u03bb", "Im \u03bb")
    corners = [(xm(a), ym(b)) for a, b in data["vertices"]] + [(xm(data["vertices"][0][0]),
                                                                ym(data["vertices"][0][1]))]
    svg.add(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in corners)}" fill="none" '
            f'stroke="{INK_SOFT}" stroke-width="1.5"/>')
    f_end = 0.85
    fr = [0.02 + (f_end - 0.02) * v / per for v in s]
    for (a, b), f, t in zip(pts, fr, turns):
        shown_from(svg, f, dur)
        dot(svg, xm(a), ym(b), SERIES[0], f"\u03bb = {a:.4f} {b:+.4f}i, {t:.3f} turns so far")
        svg.add("</g>")
    for a, b in data["zeros"]:
        inside = abs(a) < 1 and abs(b) < 1
        marker(svg, "cross" if inside else "square", xm(a), ym(b), SERIES[1] if inside else GRAY,
               f"zero at {a} {b:+}i, {'inside' if inside else 'outside'}")
    callout(svg, xm(0.86), ym(0.315), "close pair", dx=-30, dy=40)
    tracer(svg, [(xm(a), ym(b)) for a, b in pts], fr, INK, dur, r=4)
    rx, ry, rw, rh = lx + lw + 130, ly, 380, 300
    xr, yr = axes(svg, rx, ry, rw, rh, (0, per), (0, 4.5), "position along the contour",
                  "arg g so far (turns)")
    curve = [(xr(v), yr(t)) for v, t in zip(s, turns)]
    # the curve and both dots on one schedule, by position along the contour
    drawn(svg, curve, SERIES[0], 0.02, f_end, dur, at=fr)
    tracer(svg, curve, fr, INK, dur, r=4)
    shown_from(svg, f_end, dur)
    callout(svg, curve[-1][0], curve[-1][1], f"{round(data['turns'])} turns: 4 zeros inside", dx=-40,
            dy=-28)
    svg.add("</g>")
    legend(svg, rx + rw, top - 8, [("samples", SERIES[0]), ("zeros inside", SERIES[1]),
                                    ("zeros outside", GRAY)])
    height = footnote(svg, prov["sources"], prov["host"], [], ly + lh + 46, REPO)
    write("winding.svg", svg, height)
    table(["edge", "from", "to", "samples"],
          [[i + 1, f"{e['pts'][0][0]:+.0f} {e['pts'][0][1]:+.0f}i", f"{e['pts'][-1][0]:+.0f} "
            f"{e['pts'][-1][1]:+.0f}i", len(e["pts"])] for i, e in enumerate(edges)])
    print(f"turns {data['turns']:.15f}, {data['evals']} evaluations")


def main():
    ap = argparse.ArgumentParser(description="Draw docs/graphs from the data of examples/graphs.rs.")
    ap.add_argument("data", nargs="?", help="saved data to draw, instead of running the example")
    ap.add_argument("--save", metavar="PATH", help="save the example's data and provenance here")
    args = ap.parse_args()
    if args.data and args.save:
        ap.error("--save takes the data of a run, not saved data")
    data = load(args)
    prov = data["provenance"]
    os.makedirs(OUT, exist_ok=True)
    ladder(data["ladder"], prov)
    time_graph(data["time"], prov)
    fold(data["fold"], prov)
    deflation(data["deflation"], prov)
    winding(data["winding"], prov)


if __name__ == "__main__":
    main()
