# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import json

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'JaxonZhu · 具身智能笔记'
copyright = '2025, JaxonZhu'
author = 'JaxonZhu'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    'sphinx.ext.mathjax',   # 添加数学公式支持
]

myst_enable_extensions = [
    "colon_fence",
    "amsmath",
    "dollarmath",
]

# source_suffix = {
#     ".rst": "restructuredtext",
#     ".md": "markdown",
# }

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "sphinx_rtd_theme"
html_title = project
html_static_path = ['_static']

# 固定浏览器端公式渲染版本，避免 CDN 的主版本别名自动更新。
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@4.1.3/tex-mml-chtml.js"

# 长公式在正文栏内横向滚动，避免窄屏下溢出或被截断。
mathjax4_config = {"output": {"displayOverflow": "scroll"}}


def add_random_article(app, pagename, templatename, context, doctree):
    """Only offer articles reachable from the homepage, excluding directory pages."""
    if pagename != app.config.root_doc:
        return

    pending = [app.config.root_doc]
    visited = set()
    articles = []
    while pending:
        docname = pending.pop()
        if docname in visited or docname not in app.env.found_docs:
            continue
        visited.add(docname)
        children = app.env.toctree_includes.get(docname, [])
        if children:
            pending.extend(children)
        elif docname != app.config.root_doc:
            articles.append(app.builder.get_relative_uri(pagename, docname))

    if articles:
        app.add_js_file(
            'random-article.js',
            defer='defer',
            **{'data-articles': json.dumps(sorted(articles), ensure_ascii=False)},
        )


def setup(app):
    # 全站侧栏包含 π 系列标题，即使正文没有公式也需要加载 MathJax。
    app.set_html_assets_policy("always")
    app.connect('html-page-context', add_random_article)
