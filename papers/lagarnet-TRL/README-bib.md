# Split bibliography — how to regenerate

`main.tex` no longer calls BibTeX. It `\input`s two hand-generated reference lists so that
appendix-only references stay out of the main body's list:

- `refs-main.tex` — 45 entries, `[1]`–`[45]`, everything cited in §I–Acknowledgment
  (including the references cited in both the main body and the appendix).
- `refs-appendix.tex` — 9 entries, `[46]`–`[54]`, cited **only** in the appendix.
  Numbering continues from the main list via `\setcounter{enumiv}{45}`, so the appendix
  cites shared references by their main-list number and there is no duplicate `[1]`.

## To regenerate after editing `ref.bib`

1. Restore BibTeX temporarily in `main.tex`: uncomment `\bibliographystyle{IEEEtran}` and
   `\bibliography{IEEEabrv,ref}`, and comment out `\input{refs-main}` and `\input{refs-appendix}`
   (plus the `\renewcommand{\refname}` line before it).
2. `latexmk -pdf -bibtex -interaction=nonstopmode main.tex` to produce a fresh `main.bbl`.
3. Re-run the split (`split_bib.py` in this directory), which reads `main.bbl`, partitions the
   entries by whether the key is cited only after `\appendices`, and rewrites both files.
4. Undo step 1.
5. **Run `latexmk -C` before the next build**, preserving `main.bbl` across it
   (`cp main.bbl /tmp/ && latexmk -C && cp /tmp/main.bbl .`). Otherwise latexmk keeps the BibTeX
   dependency it recorded in step 2 and reruns `bibtex` on an `.aux` that no longer has a
   `\bibdata` line, which fails the build with exit 12.

The split relies on BibTeX's citation-order numbering: because every appendix-only key is first
cited after the whole main body, those entries are always the last block of `main.bbl`, so the
split never renumbers a main-body citation. `split_bib.py` derives the appendix-only key set from
`main.tex` and hard-exits if those entries are not the trailing block of `main.bbl`.
