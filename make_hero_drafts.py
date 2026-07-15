#!/usr/bin/env python
"""Two candidate lead figures, both REAL data (gate_sweep), linear axes.
  draft_hero_2curve : forward vs inverse at eta=6e-3 (cleanest single statement)
  draft_hero_wall   : small-multiples loss-curve wall across eta (most visceral)
"""
import json, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
OUT="lesswrong_figures"; os.makedirs(OUT,exist_ok=True)
C_FWD="#1D9E75"; C_INV="#D62728"
plt.rcParams.update({"font.size":13,"axes.titlesize":15,"axes.labelsize":14,"xtick.labelsize":11,
    "ytick.labelsize":11,"figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
    "font.family":"DejaVu Sans"})
def load(p): return [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else []
def curve(direction,eta,s):
    L=load(f'eta_sweep/results/gate_sweep/{direction}_eta{eta:g}_K10_nb1000_seed{s}/log.jsonl')
    return [r['step'] for r in L],[r['train_loss'] for r in L]

# ---------------- A) two-curve hero ----------------
def two_curve():
    fig,ax=plt.subplots(figsize=(7.6,4.8))
    for s,lw,a in [(0,2.8,1.0),(1,2.0,0.65)]:
        xf,yf=curve("forward",6e-3,s); xi,yi=curve("inverse",6e-3,s)
        ax.plot(xf,yf,color=C_FWD,lw=lw,alpha=a)
        ax.plot(xi,yi,color=C_INV,lw=lw,alpha=a)
    ax.set_xlim(0,25000); ax.set_ylim(-0.15,3.35)
    ax.set_xlabel("training step"); ax.set_ylabel("training loss (nats)")
    # on-curve, plain-language labels
    ax.text(13200,2.98,"B, z → A  (inverse)\ntrapped — 25,000 steps, no movement",ha="center",va="center",
            fontsize=12.5,fontweight="bold",color=C_INV)
    ax.annotate("A, z → B  (forward)\nmemorizes — loss works its way to 0",xy=(15500,0.42),xytext=(15500,1.35),
                ha="center",va="center",fontsize=12.5,fontweight="bold",color="#147a59",
                arrowprops=dict(arrowstyle="->",color=C_FWD,lw=2))
    ax.text(900,3.18,"both start here",fontsize=10.5,color="#666",ha="left")
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.set_title("Same data, same optimizer, same η = 6e-3:\none memorizes, one is trapped at the marginal floor",fontsize=14)
    fig.tight_layout(); fig.savefig(f"{OUT}/draft_hero_2curve.png",dpi=150,bbox_inches="tight")
    fig.savefig(f"{OUT}/draft_hero_2curve.pdf",bbox_inches="tight"); plt.close(fig); print("saved draft_hero_2curve")

# ---------------- B) small-multiples wall ----------------
def wall():
    ETAS=[1e-3,3e-3,6e-3,1.2e-2]   # forward arrives in all; inverse: dive -> wall (boundary ~5e-3)
    rows=[("A, z → B   (forward)",C_FWD,"forward"),("B, z → A   (inverse)",C_INV,"inverse")]
    fig,axes=plt.subplots(2,len(ETAS),figsize=(9.6,4.3),sharex=True,sharey=True)
    for ri,(rlabel,color,direction) in enumerate(rows):
        for ci,eta in enumerate(ETAS):
            ax=axes[ri][ci]
            for s,a in [(0,1.0),(1,0.55)]:
                x,y=curve(direction,eta,s)
                if x: ax.plot(x,y,color=color,lw=2.0,alpha=a)
            ax.axhline(0,color="#cccccc",lw=0.8,ls=":"); ax.axhline(2.303,color="#dddddd",lw=0.8)
            ax.set_xlim(0,25000); ax.set_ylim(-0.2,3.3)
            ax.set_xticks([]); ax.set_yticks([0,2.3] if ci==0 else [])
            if ci==0: ax.set_yticklabels(["0","logK"],fontsize=9)
            for sp in ("top","right"): ax.spines[sp].set_visible(False)
            if ri==0: ax.set_title(f"η = {eta:g}",fontsize=12)
            if ci==0: ax.set_ylabel(rlabel,fontsize=11.5,fontweight="bold",color=color)
    # box the 6e-3 column (index 2) across both rows
    fig.canvas.draw()
    b_top=axes[0][2].get_position(); b_bot=axes[1][2].get_position()
    x0=b_bot.x0-0.006; y0=b_bot.y0-0.006; x1=b_top.x1+0.006; y1=b_top.y1+0.006
    fig.add_artist(Rectangle((x0,y0),x1-x0,y1-y0,transform=fig.transFigure,fill=False,
                             edgecolor="#1A1A1A",lw=2.0,ls=(0,(4,2))))
    fig.text((x0+x1)/2,y1+0.012,"← the two-curve pair is this column",ha="center",fontsize=10.5,fontweight="bold")
    fig.suptitle("One sweep, both directions: across learning rate, only the inverse row develops a wall",
                 fontsize=14.5,y=1.04)
    fig.text(0.5,-0.02,"training step (0 → 25,000) per cell — curves end LOW = solved, end HIGH = stuck at the floor",
             ha="center",fontsize=10.5,color="#555")
    fig.tight_layout(); fig.savefig(f"{OUT}/draft_hero_wall.png",dpi=150,bbox_inches="tight")
    fig.savefig(f"{OUT}/draft_hero_wall.pdf",bbox_inches="tight"); plt.close(fig); print("saved draft_hero_wall")

if __name__=="__main__":
    two_curve(); wall()
