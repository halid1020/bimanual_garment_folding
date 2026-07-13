# latexmk configuration for the RA-L paper (example.tex)
$pdf_mode = 1;              # generate PDF via pdflatex
$bibtex_use = 2;            # always run bibtex (IEEEtran.bst) and clean .bbl on cleanup
$pdflatex = 'pdflatex -interaction=nonstopmode -synctex=1 -file-line-error %O %S';
@default_files = ('example.tex');
