# LaTeX Presentation Compilation

## Prerequisites

Install BasicTeX (lightweight LaTeX distribution for macOS):

```bash
brew install --cask basictex
```

After installation, add to PATH and update package manager:

```bash
export PATH="/Library/TeX/texbin:$PATH"
sudo tlmgr update --self
```

Install required packages (beamer, xcolor, listings):

```bash
sudo tlmgr install beamer xcolor listings
```

## Compilation Command

Navigate to the docs directory and compile:

```bash
cd docs
pdflatex presentation.tex
```

Or specify the full path to pdflatex if needed:

```bash
/Library/TeX/texbin/pdflatex presentation.tex
```

## Output

- **Input:** `presentation.tex`
- **Output:** `presentation.pdf` (8 slides, ~132 KB)

## Intermediate Files

LaTeX generates several intermediate files during compilation:

- `.aux` - Auxiliary file (cross-references)
- `.log` - Compilation log
- `.nav` - Navigation file (beamer)
- `.out` - Hyperref outline file
- `.vrb` - Verbatim content file (beamer)
- `.snm` - Section names file (beamer)
- `.toc` - Table of contents

These are automatically added to `.gitignore` and excluded from version control.

## Clean Intermediate Files

To remove intermediate files (keep only `.tex` and `.pdf`):

```bash
cd docs
rm -f *.aux *.log *.nav *.out *.vrb *.snm *.toc
```

Or use latexmk with cleanup:

```bash
latexmk -c presentation.tex
```
