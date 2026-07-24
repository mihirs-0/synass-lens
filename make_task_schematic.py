#!/usr/bin/env python
"""Minimal 'definitions' schematic: the same surjective relation read two ways.
Inverse (B,z)->A fans OUT (1 B -> K candidates, needs z); forward (A,z)->B converges IN
(each A -> its one B, z redundant). Pure geometry. 1200px wide PNG + PDF."""
import os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image
GREEN="#1D9E75"; RED="#D62728"; NEUTRAL="#333333"; AMBER="#E6A817"; GRAY="#8a8a8a"
plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":"white","savefig.facecolor":"white"})
OUT="lesswrong_figures"

def chip(ax,cx,cy,w,h,fc,text,tc="white",fs=15,ec="none"):
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.3,rounding_size=1.6",
                                facecolor=fc,edgecolor=ec,lw=1.6,zorder=5))
    ax.text(cx,cy,text,ha="center",va="center",color=tc,fontsize=fs,fontweight="bold",zorder=6)
def deck(ax,cx,cy,w,h,ec,label):
    for i,(ox,oy) in enumerate([(-2.2,2.2),(0,0),(2.2,-2.2)]):
        ax.add_patch(FancyBboxPatch((cx-w/2+ox,cy-h/2+oy),w,h,boxstyle="round,pad=0.3,rounding_size=1.6",
                     facecolor="white",edgecolor=ec,lw=1.8,zorder=3+i))
    ax.text(cx+2.2,cy-2.2,label,ha="center",va="center",fontsize=15,fontweight="bold",color=ec,zorder=7)

def line(ax,p0,p1,color,lw,z=2): ax.plot([p0[0],p1[0]],[p0[1],p1[1]],color=color,lw=lw,solid_capstyle="round",zorder=z)

fig,ax=plt.subplots(figsize=(11,6.6)); ax.set_xlim(0,100); ax.set_ylim(0,100); ax.axis("off")
ax.text(50,96,"Same pairs, read two ways",ha="center",fontsize=18,fontweight="bold",color=NEUTRAL)

# ---------- INVERSE (top): (B,z) -> A, fans OUT ----------
ax.text(4,88,"B, z → A    inverse",fontsize=15.5,color=RED,fontweight="bold",ha="left")
chip(ax,12,76,11,8,NEUTRAL,"B")
chip(ax,12,67,11,8,AMBER,"z")
Adeck_x=60
for i,ay in enumerate([84,76,68]):                       # diverging fan
    sel = (i==1)
    line(ax,(18.5,71.5),(Adeck_x-9,ay),RED if sel else "#efb6b6",3.0 if sel else 1.8)
deck(ax,Adeck_x,76,16,9,RED,"A")
ax.text(Adeck_x+13,76,"K candidates",ha="left",va="center",fontsize=13,color=RED)
ax.text(38,82.5,"z picks one",ha="center",fontsize=11.5,color=AMBER,fontweight="bold",rotation=8)
ax.text(50,60,"B alone → K targets,  loss = log K          (B, z) → one target,  loss = 0",
        ha="center",fontsize=13,color=NEUTRAL)

ax.plot([4,96],[52,52],color="#dddddd",lw=1.2)           # divider

# ---------- FORWARD (bottom): (A,z) -> B, converges IN ----------
ax.text(4,46,"A, z → B    forward",fontsize=15.5,color=GREEN,fontweight="bold",ha="left")
deck(ax,15,30,16,9,GREEN,"A")
ax.text(15,17.5,"K of them",ha="center",va="center",fontsize=13,color=GREEN)
Bx=60
for ay in [38,30,22]:                                    # converging fan
    line(ax,(24,ay),(Bx-6.5,30),GREEN,2.4)
chip(ax,Bx,30,11,8,NEUTRAL,"B")
chip(ax,Bx+18,30,12,8,"#eeeeee","z",tc=GRAY,fs=14,ec="#cccccc")
ax.plot([Bx+13,Bx+23],[33,27],color=GRAY,lw=1.6,zorder=8)   # strike-through "z unused"
ax.text(Bx+18,22.5,"redundant",ha="center",va="center",fontsize=11,color=GRAY)
ax.text(50,9,"each A already fixes B,  loss = 0          z is present but not needed",
        ha="center",fontsize=13,color=NEUTRAL)

fig.savefig(f"{OUT}/figB_task.pdf",bbox_inches="tight",facecolor="white")
fig.savefig(f"{OUT}/figB_task_hi.png",dpi=200,bbox_inches="tight",facecolor="white")
im=Image.open(f"{OUT}/figB_task_hi.png").convert("RGB"); w,h=im.size; W=1200; H=round(h*W/w)
im.resize((W,H),Image.LANCZOS).save(f"{OUT}/figB_task.png"); os.remove(f"{OUT}/figB_task_hi.png")
print(f"saved figB_task  {W}x{H}")
