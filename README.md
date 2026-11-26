# Saswato Sen's academic website

A GitHub Pages site based on [al-folio v0.14.6](https://github.com/alshedivat/al-folio/tree/v0.14.6).

## Edit the site

| Content | File or directory |
| --- | --- |
| About | `_pages/about.md` |
| Research | `_pages/research.md` |
| Publications and Research Notes | `_bibliography/papers.bib` |
| CV page and downloadable CV | `_pages/cv.md` and `assets/pdf/saswato-sen-cv.pdf` |
| Contact | `_pages/contact.md` |
| Images | `assets/img/` |
| Site settings and styles | `_config.yml` and `_sass/_academic.scss` |

Add papers and research notes to `papers.bib` with unique keys. For a research note, use `@misc` and `note = {Research note}`. Use `bibtex_show = {true}` for the visitor-facing citation and `pdf` for a public PDF URL or a filename in `assets/pdf/`.

To link a web version of a note, create a Markdown file in `_notes/` with `layout: page` and a `title` in its front matter. For `my-note.md`, set `html = {/research-notes/my-note/}` in the corresponding BibTeX entry. MathJax supports LaTeX, including numbered `equation` environments. Put figures in `assets/img/` and supply alt text. Use Jekyll's `relative_url` filter for local links in Markdown. Update both the CV page and the PDF when revising the CV.

In Codex, open this repository as a project and identify the page or BibTeX entry to change. Ask for a local preview to review edits before committing. The publication workflow runs when a change reaches `main`. Keep unpublished material in files outside the repository.

## Preview locally

With Ruby 3.3, Bundler 2.6.9, and Node.js installed:

```sh
bundle install
bundle exec jekyll serve --host 127.0.0.1
```

Open <http://127.0.0.1:4000>. The workflow uses GitHub Pages settings for the production URL and base path. In the repository's **Settings → Pages**, set **Source** to **GitHub Actions**.
