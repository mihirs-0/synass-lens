# Changelog: Phase-A outcome -> the sentence it selected (one page)

Each pre-registered Phase-A analysis had pre-committed readings. This maps the
observed outcome to the exact paper clause it licensed. No sentence in the draft
about the mechanism, the anomaly, or the 2x2 was written before its gating
measurement landed.

| Phase-A item | Pre-committed branch that fired | Sentence/clause selected in the draft |
|---|---|---|
| **A1** per-arm own-axis projection | **Amplification.** Both recovering arms show systematic pre-onset growth on their OWN escape direction: clean $\times$145 (P_T 741), v-only $\times$106 (P_T 166); escape directions internally stable (pairwise cos 0.97) but differ across arms. | Sec. 5 ONSET CLAUSE: "on each recovering arm's own escape direction the projected pull grows systematically before onset (amplification); a single held-out direction does not transfer because the arms recover along different axes." (Fig. 3, right.) |
| **A2** trapped-state drift | **Burial, not relocation.** At censored both-channel states the true-gradient norm is nonzero and GROWS ($\lVert g\rVert$ 0.048$\to$0.583 across 0.5k--16k) with a small consistently-signed escape-direction pull. Not drift-dead. | Sec. 5 MECHANISM CLAUSE: "the pull toward the solution is intact and growing at censored states; the transverse diffusion buries it (rather than relocating the iterate to drift-dead regions)." |
| **A3** $\gam\approx2$ anomaly | Measured directly. $\gam=1.89$ vs scalar $1+a^2=1.006$; from logs $q=\lVert u_{\text{both}}\rVert/\lVert u_{\text{clean}}\rVert=0.70$, $\rho=\cos(u_{\text{both}},u_{\text{clean}})=-0.28$, residual $\nu=1.38$, anti-aligned ($A_{\text{align}}=-0.87$). | Sec. 4 EMPIRICAL RESIDUE: reports measured $\gam\approx2$ vs the scalar prediction as a measurement. Appendix D (exploratory): the sparsity account (clean-update participation ratio 2576 $=0.32\%$ of $D$) + $\beta_1$ formula with independence assumptions stated. |
| **A4** units + MC error | $E[u_{\text{both}}]$ estimated by MONTE-CARLO averaging of 512 paired draws (not by-construction). $\mupar=0.498\pm1.1\mathrm{e}{-4}$ (v-only), $0.502\pm1.2\mathrm{e}{-3}$ (both); $\Delta\mupar$ bootstrap CI [0.0015,0.0062]. Dimensional check passes; the diffusion ratio survives it. | Sec. 5 drift table reports $\mupar$ WITH Monte-Carlo errors and states the estimator is Monte-Carlo (not an identity). Language law #1: "two decimals" at the fork, exact along the trajectory. |
| **A5** citation verification | 16/16 candidate citations confirmed at primary source with corrected exact arXiv ids (DP-AdamBC = arXiv:2312.14334; saddle-escape = 1703.00887; anti-grokking = 2602.02859; etc.). No fabrication. | Sec. 7 related work + `references.bib` use only these verified keys/ids. |
| **A6** original-table 2x2 | **Landed.** Fresh + v-noise on the ORIGINAL table onset $=8{,}400$ (escaped, not censored) $>$ collapsed + v-noise $4{,}600$; the sibling-table 4,000 that suggested inversion was a table artifact. | Sec. 6 PROMOTED from provisional: reports the original-table result; claim "advantage eliminated (possibly inverted)" with targeted-erasure and common-onset-floor at EQUAL weight; no standalone inversion (law #4). |

## Language-law corrections applied globally (Phase C)
- Fork drift pair stated to TWO decimals (0.498 vs 0.502); trajectory pairs exact.
- Drift match framed as a CONSISTENCY CHECK (Jensen corrections negligible at matched states), not a discovery; load-bearing empirics = fates + diffusion ratio + real curved landscape.
- Every non-escape is ">= horizon"; "permanent/never/unlearnable" absent.
- Construction checks labeled in-line (v-inflation; flat $\gam(c)$ as algebra; $\gam_v\approx1$; m-only explosion; generic noise-impedes-learning).
- N2 referee fight pre-answered as a finding (v-only $=$ rescaled plain-GD; noise floor in v destroys Adam's normalization).
- N1 referee fight pre-answered (pinning derivable post-hoc from Kingma--Ba invariance; shipped claim = controls/reported-variables corollary + measured level anomaly).
- Code names retired; no venue/acceptance language; anonymized.
- OPT style approximated (no official opt2026.sty in-repo; preamble notes the swap).
