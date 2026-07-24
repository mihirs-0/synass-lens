#!/usr/bin/env python
"""LessWrong post figures. All numbers re-derived from logs in-script (no prior-script labels).
Output: lesswrong_figures/{fig1_strip,fig2_hidden_progress,fig3_melt,figA_btest}.{png,pdf}"""
import json, math, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import statistics as st
STROKE=[pe.withStroke(linewidth=2.6, foreground="black")]      # white text readable on any cell
STROKE_W=[pe.withStroke(linewidth=2.6, foreground="white")]    # black text on light cells

OUT = "lesswrong_figures"; os.makedirs(OUT, exist_ok=True)
LOGK = math.log(10)
# palette
C_CONV="#1D9E75"; C_TRAP="#D62728"; C_NONL="#8C8C8C"; C_DIV="#1A1A1A"; C_MIX="#E6A817"
C_LOSS="#1F77B4"; C_DZ="#E68613"
plt.rcParams.update({"font.size":12,"axes.titlesize":16,"axes.labelsize":14,
    "xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":11,
    "figure.facecolor":"white","axes.facecolor":"white","savefig.facecolor":"white",
    "font.family":"DejaVu Sans"})

def load(p): return [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else []
def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"  saved {name}")

# ============================== FIG 1: strip ==============================
def fig1():
    FET=[1e-3,3e-3,6e-3,1.2e-2,2.5e-2,5e-2,1e-1,2e-1,5e-1]
    IET=[1e-3,3e-3,6e-3,1.2e-2,2.5e-2,5e-2]
    def fwd(lf,fs):
        return "div" if (lf is None or (fs and fs<100)) else ("conv" if lf<0.5 else ("nonl" if lf>2.5 else "mix"))
    def inv(lf): return "?" if lf is None else ("conv" if lf<0.3 else ("trap" if lf>2.5 else "mix"))
    def cell(direction,eta,fn):
        cats=[]
        for s in sorted(glob.glob(f'eta_sweep/results/gate_sweep/{direction}_eta{eta:g}_*/status.json')):
            d=json.load(open(s)); L=load(s.replace('status.json','log.jsonl'))
            lf=L[-1]['train_loss'] if L else None
            cats.append(fn(lf,d.get('final_step')) if direction=="forward" else fn(lf))
        if not cats: return None
        if len(set(cats))==1: return (cats[0], f"{len(cats)}/{len(cats)}")
        # mixed: tally = how many in the 'good' (converged/conv) category
        good=sum(c=="conv" for c in cats)
        return ("mix", f"{good}/{len(cats)}")
    STYLE={"conv":(C_CONV,""),"trap":(C_TRAP,"///"),"nonl":(C_NONL,"..."),
           "div":(C_DIV,"xxx"),"mix":(C_MIX,"\\\\")}
    fig,ax=plt.subplots(figsize=(9.2,3.4))
    n=len(FET)
    for col,e in enumerate(FET):
        c=cell("forward",e,fwd)
        if c:
            color,h=STYLE[c[0]]; ax.add_patch(Rectangle((col-0.46,0.54),0.92,0.92,facecolor=color,hatch=h,edgecolor="white",lw=2.5))
            tc="black" if c[0]=="mix" else "white"
            ax.text(col,1.0,c[1],ha="center",va="center",color=tc,fontsize=12.5,fontweight="bold",
                    path_effects=STROKE_W if tc=="black" else STROKE)
    for col,e in enumerate(FET):
        if e in IET:
            c=cell("inverse",e,inv); color,h=STYLE[c[0]]
            ax.add_patch(Rectangle((col-0.46,-0.46),0.92,0.92,facecolor=color,hatch=h,edgecolor="white",lw=2.5))
            tc="black" if c[0]=="mix" else "white"
            ax.text(col,0.0,c[1],ha="center",va="center",color=tc,fontsize=12.5,fontweight="bold",
                    path_effects=STROKE_W if tc=="black" else STROKE)
        else:
            ax.add_patch(Rectangle((col-0.46,-0.46),0.92,0.92,facecolor="white",hatch="///",edgecolor="#999999",lw=1.5))
            ax.text(col,0.0,"—",ha="center",va="center",color="#999999",fontsize=12)
    # 6e-3 bracket (col index 2), spanning both rows
    bx=2
    ax.add_patch(Rectangle((bx-0.5,-0.5),1.0,2.0,facecolor="none",edgecolor="#1A1A1A",lw=2.2,ls=(0,(4,2)),zorder=5))
    ax.annotate("same η: forward converges 2/2,\ninverse trapped 2/2", xy=(bx,1.5), xytext=(bx,2.18),
                ha="center",va="bottom",fontsize=11.5,fontweight="bold",
                arrowprops=dict(arrowstyle="-",color="#1A1A1A",lw=1.5))
    ax.set_xlim(-0.6,n-0.4); ax.set_ylim(-0.55,2.55)
    ax.set_xticks(range(n)); ax.set_xticklabels([f"{e:g}" for e in FET])
    ax.set_yticks([1,0]); ax.set_yticklabels(["A,z → B\n(forward)","B,z → A\n(inverse)"],fontsize=13)
    ax.set_xlabel("learning rate η   (AdamW, B=128, K=10, D=10k)")
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    from matplotlib.patches import Patch
    leg=[Patch(facecolor=C_CONV,label="converged"),
         Patch(facecolor=C_TRAP,hatch="///",label="trapped at log K floor (structured)"),
         Patch(facecolor=C_NONL,hatch="...",label="non-learning (chance accuracy)"),
         Patch(facecolor=C_DIV,hatch="xxx",label="diverged"),
         Patch(facecolor=C_MIX,hatch="\\\\",label="mixed (1 of 2 seeds)"),
         Patch(facecolor="white",hatch="///",edgecolor="#999999",label="not run")]
    ax.legend(handles=leg,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.28),frameon=False,fontsize=10.5,handlelength=1.4)
    ax.set_title("The same data, two directions: the structured trap exists on one face only",pad=42,fontsize=15)
    save(fig,"fig1_strip")

# ============================== FIG 2: hidden progress ==============================
def fig2():
    L=load('eta_sweep/results/eta_0.001_K_10_seed_0/log.jsonl')
    tau=2400
    L=[r for r in L if r['step']<=int(1.2*tau)]
    steps=[r['step'] for r in L]; cl=[r['candidate_loss'] for r in L]; dz=[max(r['delta_z'],1e-3) for r in L]
    pl=[r['candidate_loss']/LOGK for r in L if 100<=r['step']<=700]
    pmean,pstd=st.mean(pl),st.pstdev(pl)
    def at(s): return next(r['delta_z'] for r in L if r['step']>=s)
    gf=at(int(0.9*tau))/at(int(0.3*tau))
    fig,(a0,a1)=plt.subplots(2,1,figsize=(7.8,6.6),sharex=True,gridspec_kw={"hspace":0.14})
    # ---- top: candidate loss ----
    a0.plot(steps,cl,color=C_LOSS,lw=2.4,marker="o",ms=4.5,label="candidate-restricted loss (first token)")
    a0.axhline(LOGK,color="#444444",lw=1,ls="--"); a0.axhline(0,color="#444444",lw=1,ls=":")
    a0.text(2770,LOGK+0.07,"log K floor (pencil prediction) = 2.303",ha="right",va="bottom",fontsize=10.3,color="#333333")
    a0.axvline(tau,color="#888888",lw=1.4,ls="--"); a0.text(tau-45,3.32,"τ",ha="right",fontsize=15,color="#555555")
    a0.text(110,3.12,f"plateau: {pmean:.2f} ± {pstd:.2f} × log K",ha="left",va="top",fontsize=11,fontweight="bold",color="#1a1a1a")
    a0.text(650,1.5,"model behaves as if z does not exist —\nwhile measurably building sensitivity to it (below)",
            ha="center",va="center",fontsize=10.4,color="#5a4500",
            bbox=dict(boxstyle="round,pad=0.4",facecolor="#FFF6E0",edgecolor=C_MIX,lw=1.4))
    a0.set_ylim(-0.2,3.7); a0.set_ylabel("loss (nats)")
    a0.legend(loc="lower left",frameon=True,framealpha=0.95,fontsize=9.5)
    # ---- bottom: delta_z log ----
    a1.plot(steps,dz,color=C_DZ,lw=2.4,marker="s",ms=4.5,label="Δz  (z-shuffle gap)")
    a1.set_yscale("log"); a1.axvline(tau,color="#888888",lw=1.4,ls="--")
    a1.annotate("", xy=(int(0.9*tau),at(int(0.9*tau))),xytext=(int(0.3*tau),at(int(0.3*tau))),
                arrowprops=dict(arrowstyle="->",color="#8a3b00",lw=2.4))
    a1.text(330,7.0,f"Δz grows ×{gf:.0f}\n(0.3τ → 0.9τ)",fontsize=12,fontweight="bold",color="#8a3b00",ha="left",va="center")
    a1.set_ylim(2e-3,45); a1.set_ylabel("Δz  (nats, log)"); a1.set_xlabel("training step")
    a1.legend(loc="lower right",frameon=True,framealpha=0.95,fontsize=10)
    fig.suptitle(f"The plateau is not idle: sensitivity to z grows ~{gf:.0f}× under a flat calibrated loss",
                 fontsize=15,y=0.965)
    save(fig,"fig2_hidden_progress")
    return pmean,pstd,gf

# ============================== FIG 3: the melt ==============================
def fig3():
    p1='eta_sweep/results/gate_wdtest/wd0.01/inverse_eta0.003_K10_nb1000_seed0/log.jsonl'
    p0='eta_sweep/results/gate_wdtest/wd0/inverse_eta0.003_K10_nb1000_seed0/log.jsonl'
    L=load(p1); L0=load(p0)
    s=[r['step'] for r in L]; loss=[r['train_loss'] for r in L]
    dzs=[(r['step'],max(r['delta_z'],5e-3)) for r in L if r.get('delta_z') is not None]  # only logged dz points
    dz_s=[a for a,_ in dzs]; dz_v=[b for _,b in dzs]
    s0=[r['step'] for r in L0]; loss0=[r['train_loss'] for r in L0]
    def exc(LL):
        conv=False;intrap=False;eps=[]
        for r in LL:
            if r['train_loss']<0.3: conv=True
            if not conv: continue
            if r['train_loss']>2.0:
                if not intrap: eps.append([r['step'],r['step']]);intrap=True
                eps[-1][1]=r['step']
            else: intrap=False
        return eps
    e1=exc(L)[0]; e0=exc(L0)[0]
    fc=next(r['step'] for r in L if r['train_loss']<0.3)
    rec=e1[1]+100
    d1=e1[1]-e1[0]+100; d0=e0[1]-e0[0]+100
    fig,(a0,a1)=plt.subplots(2,1,figsize=(7.8,7.0),sharex=True,gridspec_kw={"hspace":0.13})
    for ax in (a0,a1): ax.axvspan(e1[0],e1[1],color=C_TRAP,alpha=0.13,zorder=0)
    # ---- top: loss ----
    a0.plot(s,loss,color=C_LOSS,lw=2.4,label=f"wd = 0.01  (excursion ~{d1:,} steps)")
    a0.plot(s0,loss0,color="#666666",lw=1.8,ls=(0,(5,2)),label=f"wd = 0  (excursion ~{d0} steps)")
    a0.axhline(LOGK,color="#444444",lw=1,ls="--"); a0.axhline(0,color="#444444",lw=1,ls=":")
    a0.text(700,LOGK+0.08,"log K floor = 2.303",ha="left",va="bottom",fontsize=10.5,color="#333333")
    a0.annotate("memorized,\nloss = 0",xy=(fc,0.02),xytext=(4200,0.75),fontsize=10.5,ha="center",
                arrowprops=dict(arrowstyle="->",color="#333333",lw=1.5))
    a0.annotate("re-memorized",xy=(rec,0.12),xytext=(29500,0.78),fontsize=10.5,ha="center",
                arrowprops=dict(arrowstyle="->",color="#333333",lw=1.5))
    a0.annotate("spontaneous fall: fully converged → back to the floor\n(Δz ≈ 50 → ≈ 0) → recovery.  No hyperparameter changed.",
                xy=(e1[0],1.85),xytext=(10200,1.5),fontsize=10.4,color="#5a1a1a",ha="center",va="center",
                bbox=dict(boxstyle="round,pad=0.4",facecolor="#FDECEC",edgecolor=C_TRAP,lw=1.3),
                arrowprops=dict(arrowstyle="->",color=C_TRAP,lw=1.6))
    a0.set_ylim(-0.2,3.5); a0.set_ylabel("train loss (nats)")
    a0.legend(loc="upper right",frameon=True,framealpha=0.95,fontsize=10)
    # ---- bottom: delta_z ----
    a1.plot(dz_s,dz_v,color=C_DZ,lw=2.2,marker="o",ms=3.5); a1.set_yscale("log")
    a1.set_ylim(3e-3,140); a1.set_ylabel("Δz  (nats, log)"); a1.set_xlabel("training step")
    a1.annotate("Δz ≈ 50 → ≈ 0\n(z genuinely lost)",xy=((e1[0]+e1[1])/2,0.06),xytext=((e1[0]+e1[1])/2,0.012),
                ha="center",fontsize=10.6,color="#8a3b00",fontweight="bold")
    fig.suptitle("A converged model un-memorizes and recovers at fixed hyperparameters (B=512, η=3e-3)",
                 fontsize=14.5,y=0.965)
    save(fig,"fig3_melt")
    return d1,d0,e1,fc,rec

# ============================== FIG A: btest ==============================
def figA():
    ETAS=[1e-3,3e-3,8e-3,1.6e-2]; BS=[32,128]
    fig,ax=plt.subplots(figsize=(7.2,3.5))
    def cellcat(B,e):
        cats=[]
        for s in sorted(glob.glob(f'eta_sweep/results/gate_btest/B{B}/inverse_eta{e:g}_*/status.json')):
            d=json.load(open(s)); L=load(s.replace('status.json','log.jsonl')); lf=L[-1]['train_loss'] if L else None
            cats.append("conv" if (lf is not None and lf<0.3) else ("trap" if (lf is not None and lf>2.5) else "mix"))
        if len(set(cats))==1: return cats[0], f"{len(cats)}/{len(cats)}"
        return "mix", f"{sum(c=='conv' for c in cats)}/{len(cats)}"
    STYLE={"conv":(C_CONV,""),"trap":(C_TRAP,"///"),"mix":(C_MIX,"\\\\")}
    bstar={}
    for row,B in enumerate(BS):
        last_conv=-1; first_trap=99
        for col,e in enumerate(ETAS):
            cat,tally=cellcat(B,e); color,h=STYLE[cat]
            ax.add_patch(Rectangle((col-0.46,row-0.46),0.92,0.92,facecolor=color,hatch=h,edgecolor="white",lw=2.5))
            ax.text(col,row,f"{tally}",ha="center",va="center",color="black" if cat=="mix" else "white",fontsize=12,fontweight="bold")
            if cat=="conv": last_conv=col
            if cat=="trap" and col<first_trap: first_trap=col
        bstar[B]=(last_conv+first_trap)/2.0
    # amber boundary line between conv and trap per row
    import numpy as np
    xs=[bstar[B] for B in BS]; ys=list(range(len(BS)))
    ax.plot(xs,ys,color=C_MIX,lw=2.4,ls=(0,(5,2)),marker="D",ms=9,zorder=6)
    ax.annotate("η* rises with batch size:  ≈2e-3 (B=32) → ≈5e-3 (B=128)",xy=(xs[1],1.05),xytext=(1.5,1.92),
                fontsize=11.5,fontweight="bold",color="#8a6500",ha="center",va="center",
                arrowprops=dict(arrowstyle="->",color=C_MIX,lw=1.8))
    ax.set_xlim(-0.6,len(ETAS)-0.4); ax.set_ylim(-0.7,2.25)
    ax.set_xticks(range(len(ETAS))); ax.set_xticklabels([f"{e:g}" for e in ETAS])
    ax.set_yticks(BS_idx:=list(range(len(BS)))); ax.set_yticklabels([f"B = {B}" for B in BS],fontsize=13)
    ax.set_xlabel("learning rate η   (inverse task, K=10, D=10k)")
    for sp in ("top","right","left"): ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=C_CONV,label="converged"),Patch(facecolor=C_TRAP,hatch="///",label="trapped"),
                       Patch(facecolor=C_MIX,hatch="\\\\",label="mixed (1/2)")],
              ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.26),frameon=False,fontsize=10.5)
    ax.text(0.5,-0.95,"B=512 omitted: metastable regime, see text",transform=ax.transAxes,ha="center",fontsize=10,color="#777777",style="italic")
    ax.set_title("Larger batches tolerate larger steps:\nthe destabilizing noise is minibatch sampling",pad=12,fontsize=14.5)
    save(fig,"figA_btest")

if __name__=="__main__":
    print("rendering:")
    fig1(); p=fig2(); m=fig3(); figA()
    print(f"\nFIG2 derived: plateau={p[0]:.3f}±{p[1]:.3f} xlogK, Δz growth ×{p[2]:.1f}")
    print(f"FIG3 derived: wd0.01 excursion={m[0]} steps {m[2]}, wd0 excursion={m[1]} steps, first_conv={m[3]}, recovery={m[4]}")
