#!/usr/bin/env python
"""Two CONCEPTUAL SCHEMATICS (not plots). Pure geometry — no axes/ticks/data/noise.
  figA_two_routes : "Two routes, one wall" (hero)
  figC_phase_map  : "The phase map" (second)
Both carry the mandatory 'schematic' disclaimer. 1200px wide PNG + PDF."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle, PathPatch, Polygon
from matplotlib.path import Path

GREEN="#1D9E75"; RED="#D62728"; NEUTRAL="#333333"; AMBER="#E6A817"; DKAMBER="#9c7400"
RED_FADE="#E79A9A"; GREEN_MUTED="#D7EBE2"; RED_MUTED="#F6DCDC"; GRAY="#5b5b5b"
plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":"white","savefig.facecolor":"white"})
OUT="lesswrong_figures"

def disclaimer(fig):
    fig.text(0.012,0.018,"schematic — illustrates the mechanism, not measured data",
             fontsize=10,color="#9a9a9a",ha="left",va="bottom")
def rbox(ax,x,y,w,h,fc,text,tc="white",fs=15):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.5,rounding_size=2.2",
                                facecolor=fc,edgecolor="none"))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold")
def arrow(ax,p0,p1,color,lw,ls="-",ms=26,rad=0.0):
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>",mutation_scale=ms,lw=lw,color=color,
                 linestyle=ls,connectionstyle=f"arc3,rad={rad}",capstyle="round"))
def save(fig,name):
    import os
    from PIL import Image
    fig.savefig(f"{OUT}/{name}.pdf",bbox_inches="tight",facecolor="white")          # vector
    fig.savefig(f"{OUT}/{name}_hi.png",dpi=200,bbox_inches="tight",facecolor="white")  # hi-res
    im=Image.open(f"{OUT}/{name}_hi.png").convert("RGB")
    w,h=im.size; W=1200; H=round(h*W/w)
    im.resize((W,H),Image.LANCZOS).save(f"{OUT}/{name}.png")                          # exactly 1200px wide
    os.remove(f"{OUT}/{name}_hi.png"); plt.close(fig); print(f"saved {name}  {W}x{H}")

# =================== FIGURE A : two routes, one wall ===================
def figA():
    fig,ax=plt.subplots(figsize=(12,6.8)); ax.set_xlim(0,120); ax.set_ylim(0,68); ax.axis("off")
    yF,yI=47,21
    ax.text(60,64.5,"The same data, two directions — one route hits a wall that the other does not",
            ha="center",va="center",fontsize=18,fontweight="bold",color=NEUTRAL)
    # origin
    rbox(ax,2.5,28.5,23,12,NEUTRAL,"Same pairs,\nsame information",fs=11.5)
    ax.plot([25.5,28],[36,yF],color=NEUTRAL,lw=2.2,solid_capstyle="round")
    ax.plot([25.5,28],[33,yI],color=NEUTRAL,lw=2.2,solid_capstyle="round")
    # --- forward lane (clean) ---
    ax.text(12,yF+8.5,"A, z → B   (forward · many A → one B)",fontsize=15.5,color=GREEN,fontweight="bold")
    arrow(ax,(28,yF),(87,yF),GREEN,4.5)
    ax.text(57,yF-4.6,"z present but not needed — pure accumulation",fontsize=12.5,color=GRAY,ha="center")
    rbox(ax,90,yF-6.5,25,13,GREEN,"memorized\n(loss → 0)",fs=15)
    # --- inverse lane (gated) ---
    ax.text(6,13.5,"B, z → A   (inverse)",fontsize=15.5,color=RED,fontweight="bold",ha="left")
    arrow(ax,(28,yI),(57.5,yI),RED,4.5)
    ax.add_patch(Circle((50,yI),1.5,color=RED,zorder=5))
    ax.plot([46.5,49.2],[25.4,22.6],color=RED,lw=1.3,solid_capstyle="round")
    ax.text(43,26,"stuck at the\nmarginal floor",ha="center",va="bottom",fontsize=12.5,color=RED)
    # gate
    ax.add_patch(Rectangle((60.5,yI-9.5),2.8,19,facecolor=AMBER,edgecolor="none",zorder=4))
    ax.text(61.9,32,"learning rate\nabove η*",ha="center",va="bottom",fontsize=14,color=DKAMBER,fontweight="bold")
    # blocked dashed continuation -> memorized node (light red = "faded", not transparent)
    arrow(ax,(64.5,yI+1.5),(91,yF-6.5),RED_FADE,2.6,ls=(0,(5,3)),ms=20,rad=-0.28)
    ax.text(80,33,"blocked above η*",fontsize=12,color=RED_FADE,ha="center",rotation=24,fontweight="bold")
    ax.text(57,8,"must build a binding (z conditioned on B) — and the binding is destroyed faster than it forms",
            ha="center",fontsize=12.5,color=GRAY)
    # reversibility -> one-line caption beneath (kept off the gate to avoid crowding)
    ax.text(60,2.8,"the wall is reversible:   lower η reopens it (rescue)   ·   raise η re-closes it (melt)",
            ha="center",fontsize=12.5,color=DKAMBER,fontweight="bold")
    disclaimer(fig); save(fig,"figA_two_routes")

# =================== FIGURE C : the phase map ===================
def bezier(p0,p1,p2,n=80):
    return [((1-t)**2*p0[0]+2*(1-t)*t*p1[0]+t*t*p2[0],
             (1-t)**2*p0[1]+2*(1-t)*t*p1[1]+t*t*p2[1]) for t in [i/(n-1) for i in range(n)]]
def figC():
    fig,ax=plt.subplots(figsize=(12,8.6)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
    X0,Y0,X1,Y1=14,12,94,84  # field rect
    ax.text(54,95,"A boundary in optimizer space decides whether the binding can exist",
            ha="center",va="center",fontsize=18,fontweight="bold",color=NEUTRAL)
    # boundary curve: rises left->right (eta* increases with B)
    curve=bezier((X0,30),(54,40),(X1,68))
    # regions (light fills, opaque)
    red_poly=curve+[(X1,Y1),(X0,Y1)]
    grn_poly=curve+[(X1,Y0),(X0,Y0)]
    ax.add_patch(Polygon(grn_poly,closed=True,facecolor=GREEN_MUTED,edgecolor="none",zorder=0))
    ax.add_patch(Polygon(red_poly,closed=True,facecolor=RED_MUTED,edgecolor="none",zorder=0))
    ax.add_patch(Rectangle((X0,Y0),X1-X0,Y1-Y0,fill=False,edgecolor=NEUTRAL,lw=1.6,zorder=3))
    ax.plot([p[0] for p in curve],[p[1] for p in curve],color=NEUTRAL,lw=3,zorder=4)
    ax.text(50,46,"η*(B) — the boundary",fontsize=14.5,color=NEUTRAL,rotation=20,ha="center",va="bottom",
            fontweight="bold",zorder=5)
    # edge labels (no ticks/numbers)
    ax.annotate("",xy=(X0-3,Y1),xytext=(X0-3,Y0),arrowprops=dict(arrowstyle="-|>",color=NEUTRAL,lw=2.2))
    ax.text(X0-6,(Y0+Y1)/2,"learning rate →",rotation=90,ha="center",va="center",fontsize=16,fontweight="bold",color=NEUTRAL)
    ax.annotate("",xy=(X1,Y0-3),xytext=(X0,Y0-3),arrowprops=dict(arrowstyle="-|>",color=NEUTRAL,lw=2.2))
    ax.text((X0+X1)/2,Y0-6.5,"batch size →",ha="center",va="center",fontsize=16,fontweight="bold",color=NEUTRAL)
    # region labels
    ax.text(33,71,"TRAPPED\nstuck at the marginal floor\n(behaves as if z were deleted)",ha="center",va="center",
            fontsize=15.5,color=RED,fontweight="bold")
    ax.text(72,27,"MEMORIZES\nthe binding forms",ha="center",va="center",fontsize=16.5,color="#147a59",fontweight="bold")
    # melt/recover: vertical two-headed arrow crossing the boundary, with the "converged model" dot at its base
    bx=66
    ax.add_patch(Circle((bx,40),1.6,color="#147a59",zorder=6))
    ax.text(bx+2.5,38.5,"a converged model lives here",ha="left",va="center",fontsize=12.5,color="#147a59")
    ax.add_patch(FancyArrowPatch((bx,41),(bx,60),arrowstyle="<|-|>",mutation_scale=20,lw=2.6,color=AMBER,zorder=6))
    ax.text(bx+2.5,52,"raise η → melt\nlower η → recover",ha="left",va="center",fontsize=13,color=DKAMBER,fontweight="bold")
    disclaimer(fig); save(fig,"figC_phase_map")

if __name__=="__main__":
    figA(); figC()
