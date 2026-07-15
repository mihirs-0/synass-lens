"""
Marginal-collapse explainer — Manim Community Edition (0.20.x).

A detective arc: confident hypotheses, each killed by a measurement, one survivor.
13 scenes, one Scene class each. Narration in collapse_narration.md.

Render one scene (low quality, fast):
    source ~/.claude/plugins/cache/manim-video-marketplace/manim-skill/0.3.0/.venv/bin/activate
    manim -ql collapse_video.py S01_Phenomenon
Render all at production quality:
    manim -qh collapse_video.py    # renders every Scene in the file
Stitch (optional): concatenate the per-scene mp4s in media/videos/.../ with ffmpeg.

Style: dark bg, INPUT=blue, OUTPUT=green, DEATH=warm red, AMBER=the surviving/average signal.
Minimal on-screen text; narration carries the logic.
"""
from manim import *
import numpy as np

config.background_color = "#0e0e10"

INPUT  = "#5AA0E0"   # the input (B)
OUTPUT = "#23C08A"   # the model's output distribution
DEATH  = "#E0554F"   # the thing that dies (a killed hypothesis)
AMBER  = "#D7A23E"   # the surviving "guess-the-average" signal / key law
NEUTRAL= "#8A8A8A"
INK    = "#E8ECEF"

# real measured data ------------------------------------------------------------
N_VALS   = [500, 1000, 2000, 4000, 8000]
ETA_MEM  = [0.025, 0.015, 0.015, 0.007, 0.007]   # memorization critical LR (slope -0.48)

def lab(s, size=30, color=INK):
    return Text(s, font_size=size, color=color)

def sharp(center, n=7):
    xs = np.arange(n); v = np.exp(-(xs - center) ** 2 / 0.6); return v / v.sum()

def flat(n=7, seed=1):
    rng = np.random.default_rng(seed); v = 0.55 / n + 0.45 * rng.random(n); return v / v.sum()

def dist_bars(vals, color, origin=ORIGIN, bw=0.30, gap=0.10, scale=2.4):
    """A bar chart sitting on a baseline at `origin`, growing up."""
    g = VGroup()
    total_w = len(vals) * bw + (len(vals) - 1) * gap
    x0 = origin[0] - total_w / 2 + bw / 2
    for i, v in enumerate(vals):
        h = max(0.04, float(v) * scale)
        r = Rectangle(width=bw, height=h, stroke_width=0, fill_color=color, fill_opacity=0.92)
        r.move_to([x0 + i * (bw + gap), origin[1] + h / 2, 0])
        g.add(r)
    return g

def token_block(seed, color=INPUT, n=4):
    """A little column of cells standing in for one input string B."""
    rng = np.random.default_rng(seed)
    g = VGroup()
    for i in range(n):
        c = Square(side_length=0.20, stroke_width=0, fill_color=color,
                   fill_opacity=0.4 + 0.6 * rng.random())
        c.move_to([0, i * 0.22, 0]); g.add(c)
    return g


# ============================================================ SCENE 1 ==========
class S01_Phenomenon(Scene):
    """Inputs stream in; low LR the output tracks them, high LR it freezes."""
    def construct(self):
        box = RoundedRectangle(width=2.0, height=1.5, corner_radius=0.12,
                               stroke_color=NEUTRAL, stroke_width=2, fill_opacity=0)
        box_lbl = lab("model", 22, NEUTRAL).next_to(box, UP, buff=0.15)
        self.play(Create(box), FadeIn(box_lbl)); self.wait(0.3)

        base = np.array([3.4, -1.2, 0])
        out = dist_bars(sharp(3), OUTPUT, origin=base)
        out_lbl = lab("output", 22, OUTPUT).next_to(out, UP, buff=0.25)
        self.play(FadeIn(out), FadeIn(out_lbl))

        mode = lab("low learning rate", 26, INPUT).to_edge(UP)
        self.play(FadeIn(mode))

        # low LR: output tracks the input (bars change each time)
        for k, c in enumerate([1, 4, 2, 5]):
            tok = token_block(k).move_to([-4.2, -0.1, 0])
            self.play(tok.animate.move_to(box.get_center()), run_time=0.5)
            self.play(FadeOut(tok, scale=0.3),
                      Transform(out, dist_bars(sharp(c), OUTPUT, origin=base)), run_time=0.5)
        self.wait(0.4)

        # switch to high LR: output freezes regardless of input
        frozen = flat(seed=2)
        mode2 = lab("high learning rate", 26, DEATH).to_edge(UP)
        self.play(Transform(mode, mode2),
                  Transform(out, dist_bars(frozen, DEATH, origin=base)))
        for k in [7, 8, 9, 10]:
            tok = token_block(k).move_to([-4.2, -0.1, 0])
            self.play(tok.animate.move_to(box.get_center()), run_time=0.45)
            self.play(FadeOut(tok, scale=0.3),
                      Transform(out, dist_bars(frozen, DEATH, origin=base)), run_time=0.35)

        kl = MathTex(r"\mathrm{KL}(\text{output}\,\|\,\text{average})\approx 0.006",
                     font_size=34, color=INK).to_edge(DOWN, buff=0.7)
        per = lab("input moves the output by 0.003 nats  —  i.e. nothing", 24, NEUTRAL)
        per.next_to(kl, UP, buff=0.25)
        self.play(Write(kl), FadeIn(per)); self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================ SCENE 2 ==========
class S02_LossHidesIt(Scene):
    """A flat loss can be a model still learning or one that quit. Loss can't tell."""
    def construct(self):
        ax = Axes(x_range=[0, 10, 2], y_range=[0, 3, 1], x_length=8, y_length=3.2,
                  axis_config={"include_tip": False, "color": NEUTRAL, "stroke_width": 2}).to_edge(UP, buff=0.8)
        xlb = lab("training step", 22, NEUTRAL).next_to(ax, DOWN, buff=0.2)
        ylb = lab("loss", 22, NEUTRAL).rotate(PI/2).next_to(ax, LEFT, buff=0.2)
        curve = ax.plot(lambda x: 0.4 + 2.4 * np.exp(-1.1 * x), x_range=[0.05, 10], color=INK)
        self.play(Create(ax), FadeIn(xlb), FadeIn(ylb))
        self.play(Create(curve), run_time=1.6); self.wait(0.3)

        flat_lbl = lab("flat plateau — two very different models live here", 24, NEUTRAL)
        flat_lbl.next_to(ax, DOWN, buff=0.9)
        self.play(FadeIn(flat_lbl)); self.wait(0.5)

        # two panels, same loss number, different input-dependence meters
        def panel(title, meter_frac, mcolor):
            box = RoundedRectangle(width=3.0, height=2.4, corner_radius=0.1,
                                   stroke_color=NEUTRAL, stroke_width=1.5, fill_opacity=0)
            t = lab(title, 22, mcolor).next_to(box, UP, buff=0.12)
            loss = MathTex(r"\text{loss}=0.41", font_size=26, color=INK).move_to(box.get_top()+DOWN*0.5)
            track = Rectangle(width=0.4, height=1.4, stroke_color=NEUTRAL, stroke_width=1.5, fill_opacity=0)
            track.move_to(box.get_center()+DOWN*0.2)
            fill = Rectangle(width=0.4, height=max(0.02, 1.4*meter_frac), stroke_width=0,
                             fill_color=mcolor, fill_opacity=0.9)
            fill.move_to(track.get_bottom()+UP*max(0.01,1.4*meter_frac)/2)
            mlbl = lab("input-dependence", 18, NEUTRAL).next_to(track, DOWN, buff=0.15)
            return VGroup(box, t, loss, track, fill, mlbl)

        p1 = panel("still learning", 0.85, OUTPUT).to_edge(DOWN, buff=0.4).shift(LEFT*3.2)
        p2 = panel("quietly quit", 0.02, DEATH).to_edge(DOWN, buff=0.4).shift(RIGHT*3.2)
        self.play(FadeOut(flat_lbl), FadeIn(p1), FadeIn(p2)); self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================ SCENE 3 ==========
class S03_NotBinding(Scene):
    """Hypothesis 1 dies: remove the variable entirely, same collapse."""
    def construct(self):
        t1 = MathTex(r"(B,\,", r"z", r")\;\rightarrow\;A", font_size=64)
        t1[1].set_color(AMBER)
        cap = lab("I was sure it refused to use this variable", 26, NEUTRAL).next_to(t1, DOWN, buff=0.6)
        self.play(Write(t1), FadeIn(cap)); self.wait(1.2)

        strike = Line(t1[1].get_left()+LEFT*0.05, t1[1].get_right()+RIGHT*0.05,
                      color=DEATH, stroke_width=6)
        self.play(Create(strike)); self.wait(0.6)

        t2 = MathTex(r"B\;\rightarrow\;A", font_size=64)
        cap2 = lab("remove it entirely  —  same collapse", 26, DEATH).next_to(t2, DOWN, buff=0.6)
        self.play(ReplacementTransform(VGroup(t1, strike), t2), FadeOut(cap))
        self.play(FadeIn(cap2)); self.wait(0.6)

        # tiny freeze echo
        frozen = dist_bars(flat(seed=3), DEATH, origin=np.array([0,-2.2,0]), scale=1.6)
        self.play(FadeIn(frozen)); self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================ SCENE 4 ==========
class S04_TheLaw(Scene):
    """The clean law: eta* ~ N^(-1/2)."""
    def construct(self):
        lx = [np.log10(n) for n in N_VALS]; ly = [np.log10(e) for e in ETA_MEM]
        ax = Axes(x_range=[2.6, 4.0, 1], y_range=[-2.3, -1.4, 0.3], x_length=8.5, y_length=4.5,
                  axis_config={"include_tip": False, "color": NEUTRAL, "stroke_width": 2})
        ax.to_edge(DOWN, buff=0.9)
        xlb = lab("number of distinct outputs  N   (log)", 24, NEUTRAL).next_to(ax, DOWN, buff=0.75)
        ylb = lab("critical learning rate  η*   (log)", 22, NEUTRAL).rotate(PI/2).next_to(ax, LEFT, buff=0.95)
        self.play(Create(ax), FadeIn(xlb), FadeIn(ylb))

        # tick labels with real values
        for n in [500, 2000, 8000]:
            t = lab(str(n), 18, NEUTRAL).next_to(ax.c2p(np.log10(n), -2.3), DOWN, buff=0.18)
            self.add(t)
        for e in [0.007, 0.015, 0.025]:
            t = lab(str(e), 18, NEUTRAL).next_to(ax.c2p(2.6, np.log10(e)), LEFT, buff=0.22)
            self.add(t)

        dots = VGroup(*[Dot(ax.c2p(x, y), color=INPUT, radius=0.08) for x, y in zip(lx, ly)])
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.25, run_time=1.6))
        self.wait(0.4)

        # fit line slope -0.48 through the centroid
        mx, my = np.mean(lx), np.mean(ly); slope = -0.48
        line = Line(ax.c2p(2.6, my + slope*(2.6-mx)), ax.c2p(4.0, my + slope*(4.0-mx)),
                    color=AMBER, stroke_width=4)
        self.play(Create(line), run_time=1.2)

        eq = MathTex(r"\eta^* \propto N^{-1/2}", font_size=44, color=AMBER).to_corner(UR, buff=0.8)
        r2 = MathTex(r"R^2 = 0.90", font_size=30, color=NEUTRAL).next_to(eq, DOWN, buff=0.25)
        cav = lab('coarse grid → "about" one-half', 20, NEUTRAL).next_to(r2, DOWN, buff=0.2)
        self.play(Write(eq)); self.play(FadeIn(r2), FadeIn(cav)); self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================ SCENE 5 ==========
class S05_Objection(Scene):
    """Not just 'high LR bad': a few× the sweet spot, below the stability edge."""
    def construct(self):
        title = lab('"isn\'t this just: high LR underfits?"', 28, NEUTRAL).to_edge(UP)
        self.play(FadeIn(title)); self.wait(0.5)

        # yardstick 1: number line, optimal LR -> eta* a few x higher
        nl = NumberLine(x_range=[0, 14, 2], length=9, color=NEUTRAL, include_numbers=False)
        nl.shift(UP*0.6)
        opt = Dot(nl.n2p(1), color=OUTPUT, radius=0.09)
        optl = lab("best training LR", 20, OUTPUT).next_to(opt, UP, buff=0.2)
        band = Line(nl.n2p(3.5), nl.n2p(12.6), color=DEATH, stroke_width=8).set_opacity(0.5)
        star = Dot(nl.n2p(8), color=DEATH, radius=0.09)
        starl = lab("collapse  η*  (3.5–12.6×)", 20, DEATH).next_to(band, DOWN, buff=0.2)
        self.play(Create(nl), FadeIn(opt), FadeIn(optl))
        self.play(Create(band), FadeIn(star), FadeIn(starl)); self.wait(1.5)

        # yardstick 2: vertical scale, eta* below the stability edge
        vl = NumberLine(x_range=[0, 10, 2], length=4, color=NEUTRAL, rotation=PI/2).to_edge(DOWN, buff=0.7).shift(LEFT*2)
        edge = DashedLine(vl.n2p(9)+LEFT*1.6, vl.n2p(9)+RIGHT*1.6, color=DEATH)
        edgel = lab("optimizer blows up   2/λ_max", 20, DEATH).next_to(edge, RIGHT, buff=0.2)
        es = Dot(vl.n2p(4), color=AMBER, radius=0.09)
        esl = lab("η*  — still stable", 20, AMBER).next_to(es, RIGHT, buff=0.2)
        self.play(Create(vl), Create(edge), FadeIn(edgel))
        self.play(FadeIn(es), FadeIn(esl)); self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================ SCENE 6 ==========
class S06_SignalCancels(Scene):
    """Centerpiece: per-example gradients cancel, leaving a small 'guess average' residual ~1/sqrt(N)."""
    def construct(self):
        title = lab("the batch gradient is an average over examples", 26, NEUTRAL).to_edge(UP)
        self.play(FadeIn(title))
        center = np.array([-3.2, -0.3, 0])

        def arrow_field(n, seed, bias=np.array([0.18, -0.12, 0])):
            rng = np.random.default_rng(seed); arr = VGroup(); dirs = []
            for _ in range(n):
                a = rng.uniform(0, TAU); d = np.array([np.cos(a), np.sin(a), 0]) + bias
                dirs.append(d)
                arr.add(Arrow(center, center + 1.3*d, buff=0, stroke_width=2.5,
                              color=INPUT, max_tip_length_to_length_ratio=0.12).set_opacity(0.45))
            mean = np.mean(dirs, axis=0)
            return arr, mean

        field, mean = arrow_field(36, 0)
        cap = lab("each example pulls toward memorizing its own answer", 22, INPUT).next_to(title, DOWN, buff=0.2)
        self.play(LaggedStart(*[GrowArrow(a) for a in field], lag_ratio=0.02, run_time=2), FadeIn(cap))
        self.wait(0.6)

        # collapse to the residual (mean)
        res = Arrow(center, center + 1.3*mean, buff=0, color=AMBER, stroke_width=7)
        resl = lab("what survives:  shift toward the average", 22, AMBER).next_to(cap, DOWN, buff=0.15)
        self.play(field.animate.set_opacity(0.12), GrowArrow(res), FadeIn(resl))
        self.wait(1.0)

        # residual shrinks with N -> 1/sqrt(N)
        scale_lbl = MathTex(r"\text{residual} \propto 1/\sqrt{N}", font_size=36, color=AMBER)
        scale_lbl.to_edge(RIGHT, buff=1.0)
        self.play(FadeIn(scale_lbl))
        def resid_mag(n, repeats=60):
            rng = np.random.default_rng(100 + n)          # averaged over trials -> clean 1/sqrt(N) expectation
            mags = []
            for _ in range(repeats):
                a = rng.uniform(0, TAU, n)
                mags.append(np.linalg.norm(np.stack([np.cos(a), np.sin(a)], 1).mean(0)))
            return float(np.mean(mags))
        bars = VGroup()
        for i, n in enumerate([12, 40, 120]):
            h = max(0.05, resid_mag(n) * 7.0)             # monotonically decreasing in N
            b = Rectangle(width=0.5, height=h, stroke_width=0, fill_color=AMBER, fill_opacity=0.9)
            b.move_to([1.6 + i*0.9, -2.3 + h/2, 0])
            nlbl = lab(f"N={n}", 18, NEUTRAL).next_to(b, DOWN, buff=0.1)
            bars.add(VGroup(b, nlbl))
        self.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.3, run_time=1.5))
        self.wait(0.6)

        snr = MathTex(r"\text{measured gradient SNR} \propto N^{-0.45}", font_size=30, color=OUTPUT)
        snr.to_edge(DOWN, buff=0.4)
        self.play(Write(snr)); self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================ SCENE 7 ==========
class S07_NotCurvature(Scene):
    """Hypothesis 2 dies: curvature (lambda_max) is flat, doesn't track the threshold."""
    def construct(self):
        # edge-of-stability cartoon
        ax = Axes(x_range=[-3, 3], y_range=[0, 4], x_length=5, y_length=2.8,
                  axis_config={"include_tip": False, "color": NEUTRAL}).to_edge(LEFT, buff=0.8).shift(UP*0.5)
        val = ax.plot(lambda x: 0.4*x**2, x_range=[-3, 3], color=INK)
        cap = lab("textbook: step too big for a sharp valley", 22, NEUTRAL).next_to(ax, DOWN, buff=0.25)
        self.play(Create(ax), Create(val), FadeIn(cap))
        # bouncing too-big steps
        pts = [-2.4, 2.1, -1.7, 1.3]
        ball = Dot(ax.c2p(pts[0], 0.4*pts[0]**2), color=DEATH, radius=0.08)
        self.play(FadeIn(ball))
        for p in pts[1:]:
            self.play(ball.animate.move_to(ax.c2p(p, 0.4*p**2)), run_time=0.4)
        self.wait(0.5)

        # measured lambda_max flat across N
        ax2 = Axes(x_range=[2.6, 4.0], y_range=[0, 100, 50], x_length=4.5, y_length=2.8,
                   axis_config={"include_tip": False, "color": NEUTRAL}).to_edge(RIGHT, buff=0.8).shift(UP*0.5)
        xl = lab("N (log)", 20, NEUTRAL).next_to(ax2, DOWN, buff=0.2)
        yl = lab("sharpness  λ_max", 20, NEUTRAL).scale(0.9).rotate(PI/2).next_to(ax2, LEFT, buff=0.15)
        lam = [50, 71, 80, 40, 47]
        dots = VGroup(*[Dot(ax2.c2p(np.log10(n), l), color=INPUT, radius=0.07) for n, l in zip(N_VALS, lam)])
        flatline = DashedLine(ax2.c2p(2.6, 58), ax2.c2p(4.0, 58), color=NEUTRAL)
        self.play(Create(ax2), FadeIn(xl), FadeIn(yl))
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dots], lag_ratio=0.2), Create(flatline))
        self.wait(0.6)

        verdict = lab("flat — doesn't track the threshold", 24, DEATH).to_edge(DOWN, buff=0.7)
        cross = Cross(VGroup(ax, val), color=DEATH, stroke_width=6)
        self.play(FadeIn(verdict)); self.play(Create(cross)); self.wait(0.6)
        cav = lab("(raw curvature; under Adam the right object is messier)", 18, NEUTRAL).next_to(verdict, DOWN, buff=0.15)
        self.play(FadeIn(cav)); self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================ SCENE 8 ==========
class S08_BiasSignature(Scene):
    """The clean evidence is input-independence itself; the bias is modestly larger (gauge-fixed 1.8x)."""
    def construct(self):
        title = lab("how the collapse is implemented", 26, NEUTRAL).to_edge(UP)
        self.play(FadeIn(title))

        # the clean, gauge-independent evidence
        kl = MathTex(r"\mathrm{KL}(\text{output}\,\|\,\text{average})\approx 0.006", font_size=34, color=AMBER)
        swap = lab("input swap moves the output ≈ 0.003 nats", 24, INK).next_to(kl, DOWN, buff=0.3)
        self.play(Write(kl), FadeIn(swap)); self.wait(1.2)

        # the bias is a (decoration) correlate — modest, gauge-fixed
        small = Circle(radius=0.3, color=OUTPUT, fill_opacity=0.5, stroke_width=2).shift(LEFT*3.0+DOWN*1.6)
        sl = lab("memorized bias 6.5", 20, OUTPUT).next_to(small, DOWN, buff=0.25)
        big = Circle(radius=0.3*np.sqrt(1.8), color=DEATH, fill_opacity=0.4, stroke_width=2).shift(RIGHT*1.0+DOWN*1.6)
        bl = lab("trapped bias 11.5  (1.8×, gauge-fixed)", 20, DEATH).next_to(big, DOWN, buff=0.25)
        self.play(GrowFromCenter(small), FadeIn(sl), GrowFromCenter(big), FadeIn(bl)); self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================ SCENE 9 ==========
class S09_ForecloseBinding(Scene):
    """Foreclose memorization -> the model computes a rule; causal test follows it 94%."""
    def construct(self):
        title = lab("make memorizing impossible — force a computed rule", 26, NEUTRAL).to_edge(UP)
        self.play(FadeIn(title))

        # grid of (input, index) cells; some held out
        grid = VGroup()
        held = {(1, 2), (2, 0), (0, 3), (3, 1)}
        for r in range(4):
            for c in range(4):
                col = NEUTRAL if (r, c) in held else INPUT
                op = 0.12 if (r, c) in held else 0.5
                sq = Square(0.6, stroke_color=NEUTRAL, stroke_width=1, fill_color=col, fill_opacity=op)
                sq.move_to([c*0.7 - 1.05, r*0.7 - 1.05, 0]); grid.add(sq)
        gl = lab("grey = held out (never seen) → must be computed, not stored", 20, NEUTRAL).next_to(grid, DOWN, buff=0.3)
        self.play(FadeIn(grid), FadeIn(gl)); self.wait(1.5)
        self.play(FadeOut(grid), FadeOut(gl))

        # causal test: move the 'largest token' to a new bucket, output follows the new rule
        cap = lab("causal test: move what it keys on, hold the index", 24, INK).next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(cap))
        b1 = VGroup(*[Square(0.5, stroke_width=1.5, stroke_color=NEUTRAL, fill_opacity=0).shift(RIGHT*i*0.55) for i in range(5)])
        b1.move_to(LEFT*2.5)
        big = b1[3]; star = Star(color=DEATH, fill_opacity=1).scale(0.2).move_to(big.get_center())
        starlab = lab("largest", 16, DEATH).next_to(big, UP, buff=0.1)
        out1 = Square(0.5, fill_color=OUTPUT, fill_opacity=0.8, stroke_width=0).move_to(RIGHT*2.5+UP*0.6)
        o1l = lab("rule A → this answer", 18, OUTPUT).next_to(out1, RIGHT, buff=0.2)
        self.play(FadeIn(b1), FadeIn(star), FadeIn(starlab), FadeIn(out1), FadeIn(o1l)); self.wait(1.0)

        # move the star to a different bucket; output switches
        out2 = Square(0.5, fill_color=AMBER, fill_opacity=0.85, stroke_width=0).move_to(RIGHT*2.5+DOWN*0.6)
        o2l = lab("output follows new rule  —  94%", 20, AMBER).next_to(out2, RIGHT, buff=0.2)
        self.play(star.animate.move_to(b1[1].get_center()), starlab.animate.next_to(b1[1], UP, buff=0.1))
        self.play(FadeOut(out1), FadeOut(o1l), FadeIn(out2), FadeIn(o2l)); self.wait(1.2)
        cav = lab("(computed index-and-copy — not arbitrary association)", 18, NEUTRAL).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(cav)); self.wait(2)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================ SCENE 10 =========
class S10_BindingCollapsesToo(Scene):
    """Binding collapses the same way; threshold falls with cardinality but gentler."""
    def construct(self):
        title = lab("and it breaks the same way", 28, DEATH).to_edge(UP)
        self.play(FadeIn(title))
        base = np.array([-3.4, -0.6, 0])
        out = dist_bars(sharp(2), OUTPUT, origin=base)
        ll = lab("low η: computes the rule", 22, OUTPUT).next_to(out, UP, buff=0.3)
        self.play(FadeIn(out), FadeIn(ll)); self.wait(0.8)
        self.play(Transform(out, dist_bars(flat(seed=5), DEATH, origin=base)),
                  Transform(ll, lab("high η: frozen on the average", 22, DEATH).next_to(out, UP, buff=0.3)))
        self.wait(0.8)

        # two slopes side by side
        ax = Axes(x_range=[2.6, 4.0], y_range=[-2.3, -1.4], x_length=4.5, y_length=3,
                  axis_config={"include_tip": False, "color": NEUTRAL}).to_edge(RIGHT, buff=0.9)
        mem = Line(ax.c2p(2.6, -1.55), ax.c2p(4.0, -1.55-0.48*1.4), color=AMBER, stroke_width=4)
        bind = Line(ax.c2p(2.6, -1.75), ax.c2p(4.0, -1.75-0.20*1.4), color=OUTPUT, stroke_width=4)
        meml = lab("memorization  −0.48", 18, AMBER).next_to(mem, UP, buff=0.1).shift(RIGHT*0.3)
        bindl = lab("binding  ≈ −0.16…−0.25", 18, OUTPUT).next_to(bind, DOWN, buff=0.1)
        xl = lab("N (log)", 18, NEUTRAL).next_to(ax, DOWN, buff=0.15)
        self.play(Create(ax), FadeIn(xl))
        self.play(Create(mem), FadeIn(meml)); self.play(Create(bind), FadeIn(bindl)); self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================ SCENE 11 =========
class S11_DilutionFails(Scene):
    """Hypothesis 3 dies: dilution is the same 1/sqrt(N) but binding threshold is half. OPEN."""
    def construct(self):
        title = lab("does dilution explain binding too?", 26, NEUTRAL).to_edge(UP)
        self.play(FadeIn(title))
        ax = Axes(x_range=[2.6, 4.0], y_range=[-1.0, 0.0], x_length=8, y_length=3.8,
                  axis_config={"include_tip": False, "color": NEUTRAL}).shift(DOWN*0.3)
        xl = lab("N (log)", 22, NEUTRAL).next_to(ax, DOWN, buff=0.2)
        self.play(Create(ax), FadeIn(xl))
        snr = Line(ax.c2p(2.6, -0.1), ax.c2p(4.0, -0.1-0.5*1.4), color=INPUT, stroke_width=4)
        snrl = lab("gradient dilution  −0.50", 20, INPUT).next_to(snr, UP, buff=0.1)
        bind = Line(ax.c2p(2.6, -0.35), ax.c2p(4.0, -0.35-0.22*1.4), color=OUTPUT, stroke_width=4)
        bindl = lab("binding threshold  −0.22", 20, OUTPUT).next_to(bind.get_end(), RIGHT, buff=0.15)
        self.play(Create(snr), FadeIn(snrl)); self.play(Create(bind), FadeIn(bindl)); self.wait(1.0)
        gap = lab("same dilution — but the threshold falls at half the rate", 22, DEATH).to_edge(DOWN, buff=1.1)
        openst = lab("binding mechanism:  OPEN", 30, AMBER).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(gap)); self.play(Write(openst)); self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================ SCENE 12 =========
class S12_Graveyard(Scene):
    """The honest close: killed hypotheses struck through, one survivor."""
    def construct(self):
        title = lab("everything I was sure of — and had to kill", 28, NEUTRAL).to_edge(UP, buff=0.6)
        self.play(FadeIn(title))
        items = [
            "marginals before conditionals",
            "the log-K floor",
            "binding-specificity",
            "directional asymmetry",
            "edge of stability (curvature)",
            "the prior is a cheap shortcut",
            "compositional targets dilute less",
        ]
        rows = VGroup()
        for s in items:
            rows.add(lab(s, 26, NEUTRAL))
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.26).next_to(title, DOWN, buff=0.5).to_edge(LEFT, buff=1.2)
        for row in rows:
            self.play(FadeIn(row, shift=RIGHT*0.2), run_time=0.4)
            strike = Line(row.get_left(), row.get_right(), color=DEATH, stroke_width=4)
            self.play(Create(strike), run_time=0.3)
            row.add(strike)
        self.wait(0.6)
        surv = lab("what survived:", 24, OUTPUT)
        s1 = lab("high LR collapses memorization AND computed binding", 24, INK)
        s2 = lab("to the input's average answer", 24, INK)
        s3 = lab("memorization = gradient dilution.   binding = open.", 24, AMBER)
        g = VGroup(surv, s1, s2, s3).arrange(DOWN, buff=0.18).to_edge(DOWN, buff=0.5)
        self.play(FadeOut(rows), FadeOut(title))
        self.play(LaggedStart(*[FadeIn(x) for x in g], lag_ratio=0.4, run_time=2)); self.wait(2.5)
        self.play(*[FadeOut(m) for m in self.mobjects])


# ============================================================ SCENE 13 =========
class S13_Scope(Scene):
    """The integrity beat: tiny model, small range, one honest takeaway."""
    def construct(self):
        tiny = lab("tiny", 80, NEUTRAL)
        self.play(Write(tiny)); self.wait(0.4)
        spec = lab("2-layer model · N = 500–8000 · coarse LR grid", 24, NEUTRAL).next_to(tiny, DOWN, buff=0.5)
        guess = lab("nothing here is shown to hold for real models — that's a guess, not a result", 22, DEATH)
        guess.next_to(spec, DOWN, buff=0.5)
        self.play(FadeIn(spec)); self.play(FadeIn(guess)); self.wait(1.5)
        self.play(FadeOut(tiny), FadeOut(spec), FadeOut(guess))

        take = VGroup(
            lab("the one clean thing:", 24, OUTPUT),
            lab("a flat loss can be a model that quietly quit —", 28, INK),
            lab("and the cheap way to catch it is to ask:", 28, INK),
            lab("does the output still depend on the input?", 30, AMBER),
        ).arrange(DOWN, buff=0.3)
        self.play(LaggedStart(*[FadeIn(x) for x in take], lag_ratio=0.5, run_time=3)); self.wait(3)
        self.play(*[FadeOut(m) for m in self.mobjects])
