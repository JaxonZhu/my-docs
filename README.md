# JaxonZhu 的个人主页

论文解读、VLA 实践和 LeRobot 学习笔记，使用 Sphinx、MyST Markdown 和 Read the Docs 主题构建。

- 网站：[jaxonzhu-documents.readthedocs.io](https://jaxonzhu-documents.readthedocs.io/)
- 仓库：[JaxonZhu/my-docs](https://github.com/JaxonZhu/my-docs)
- 发布流程：本地编辑 → GitHub `main` → Read the Docs 构建 → 网站更新。

## 新电脑初始化

本地、GitHub Actions 和 Read the Docs 统一使用 **Python 3.12**。直接依赖的版本记录在 `docs/requirements.in`，全部依赖的精确版本记录在 `docs/requirements.txt`；日常安装使用后者。

先安装 Git 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。下面的命令适用于 macOS / Linux，从仓库根目录执行：

```bash
# 还没有仓库时执行；已有仓库则直接进入对应目录。
git clone https://github.com/JaxonZhu/my-docs.git
cd my-docs

# 仅在没有 .venv 时创建，已有环境请保留。
if [ ! -d .venv ]; then
  uv venv --python 3.12
fi
source .venv/bin/activate
python --version
uv pip sync docs/requirements.txt
uv pip check
make -C docs check
```

确认 `python --version` 显示 `3.12.x`。`uv pip sync` 会让当前虚拟环境与锁定文件一致，并移除环境中不在锁定文件里的包，因此只在本项目专用的 `.venv` 中使用。uv 会按需获取 Python；`.python-version` 也记录了本项目使用的版本。以后每次打开新终端，在仓库根目录执行 `source .venv/bin/activate` 即可继续工作。

如果使用已有的 Python 3.12 而不使用 uv，可以通过 `python3.12 -m venv .venv` 创建环境，激活后使用 `python -m pip install -r docs/requirements.txt` 和 `python -m pip check`。重新生成依赖锁定文件时仍使用 uv。

首次提交前，配置该仓库的 Git 姓名和邮箱。将示例值替换成自己的信息，可使用 GitHub 的隐私邮箱：

```bash
git config user.name "你的姓名"
git config user.email "你的 GitHub 提交邮箱"
git config --get user.name
git config --get user.email
```

Git 提交身份和 GitHub 登录是两件事；`git push` 还需要有效的 GitHub HTTPS 凭据或 SSH 配置。

## 构建与预览

激活虚拟环境后：

```bash
# 发布前必跑：清理旧构建产物，重新构建全部文档，把警告视为错误。
make -C docs check

# 严格构建后启动本地预览服务器。
make -C docs serve
```

打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)，用 `Ctrl+C` 停止预览。端口被占用时用 `make -C docs serve PORT=8001`。预览不会自动重新构建；编辑后可在另一个已激活环境的终端执行 `make -C docs check`，再刷新浏览器。

`make -C docs html` 可以增量构建，但提交前应使用 `check`。`check` 先清理 `docs/build/`，避免已删除文章留下旧 HTML，再执行完整的严格构建（包括未定义交叉引用检查），最后运行 `scripts/check_docs.py`，校验生成 HTML 的本地图片和链接，并查找残留的 `$$` 定界符或混入正文的数学节点。生成的 HTML 位于 `docs/build/html/`，无需提交。

**构建成功只能确认文档解析、引用和资源复制等构建步骤。** 数学公式由浏览器里的 MathJax 排版，仍需预览受影响页面，检查公式、图片、表格和手机窄屏布局。`docs/source/conf.py` 将 MathJax 固定为 **4.1.3**，脚本通过 jsDelivr CDN 加载；本地预览和线上访问时，浏览器都需要能访问该 CDN。升级 MathJax 后也应重新检查公式。

超过正文宽度的独立公式会在公式区域内横向滚动，预览时可拖动查看完整表达式。

首页的“随机逛一篇”从首页目录可达的末级文章中随机选择，跳过首页和包含下级目录的栏目页。新增文章接入现有目录后，下次构建会自动加入候选列表，不需要手工维护链接。浏览器未启用 JavaScript 时，该位置保留一篇实践记录的普通链接。

首页“联系我”下方的“支持这份笔记”直接展示文案和微信赞赏码，无需点击展开，也不依赖 JavaScript。文案在 `docs/source/_includes/support.inc`，样式在 `docs/source/_static/support.css`。微信赞赏码原图保存在 `docs/source/images/personal_page/wechat-appreciation.jpg`，页面中的图片和“查看赞赏码原图”均链接到构建后的 `_images/wechat-appreciation.jpg`，方便手机读者打开保存。更换赞赏码时保留原图完整边缘，不重绘或裁剪二维码；若修改文件名，同步更新图片引用和两处原图链接。该面板只展示图片，不查询到账状态；更新后需重新构建、检查原图链接，并用微信实际验证扫码。

## 文件放在哪里

| 路径 | 用途 |
| --- | --- |
| `docs/source/index.rst` | 首页介绍和八类文章的导航 |
| `docs/source/category-*.rst` | 分类介绍及文章目录，每篇正文只登记一次 |
| `docs/source/index-主题-HEAD.rst` | 保留的主题与系列导读，使用普通链接关联文章 |
| `docs/source/*.md` | 文章正文 |
| `docs/source/images/主题/` | 文章配图 |
| `docs/source/conf.py` | Sphinx、MyST、主题及数学设置 |
| `docs/templates/` | 写作模板，不参与网站构建 |
| `docs/requirements.in` | 人工维护的直接依赖版本 |
| `docs/requirements.txt` | 自动生成的完整依赖锁定文件 |
| `.readthedocs.yaml` | Read the Docs 云端环境及构建入口 |
| `.github/workflows/docs.yml` | Pull Request / `main` 推送时的严格构建检查 |

不要把草稿、占位模板和测试页放进 `docs/source/`。Sphinx 会扫描该目录里的文档，即使没有把它们加入导航，也可能产生警告。

## 给分类添加文章

1. 新建例如 `docs/source/Evo-1-new-note.md`，用一个一级标题作为文章标题，其后按 `##`、`###` 组织内容。标题采用“模型或工具名 + 内容类型：具体主题”，如“Evo-1 实践记录：ALOHA 仿真插入任务微调”。内容类型按实际内容选择论文解读、模型卡解读、源码分析、实践记录、数据流程图解或工具笔记；兼有论文与代码分析时使用“论文与源码解读”。论文英文原题放在正文开头，便于对照检索。
2. 图片放在 `docs/source/images/Evo-1/`，使用相对路径引用。
3. 按文章最突出的贡献选择一个分类，在对应文件的 `.. toctree::` 中添加文件名（不含扩展名），与已有条目保持缩进一致。分类固定为以下八类：

| 分类 | 目录文件 | 主要内容 |
| --- | --- | --- |
| 数据集 | `category-datasets.rst` | 数据资源、覆盖范围、标注与质量 |
| 数据管线 | `category-data-pipelines.rst` | 采集、标注、筛选、读取、混合与采样 |
| Benchmark | `category-benchmarks.rst` | 评测任务、指标、协议与基准 |
| 模型 | `category-models.rst` | 模型架构、表征、预训练模型与结构源码分析 |
| 算法 | `category-algorithms.rst` | 学习目标、优化规则、后训练与推理时动作选择 |
| 经验分析 | `category-empirical-analysis.rst` | 研究问题、受控实验、经验规律与适用边界 |
| 推理部署工程 | `category-inference-deployment.rst` | 推理、异步执行、系统编排与分布式训练部署闭环 |
| 个人实践 | `category-personal-practice.rst` | 自己的实验、复现、训练评测和工具学习笔记 |

例如，新的 Evo-1 微调实践登记到 `category-personal-practice.rst`：

```rst
.. toctree::
   :maxdepth: 1

   Evo-1-aloha-finetune
   Evo-1-new-note
```

4. 同一系列按篇分类。例如，Evo-1 的论文与网络源码分析归入模型，数据读取流程归入数据管线，微调记录归入个人实践。每篇正文只进入一个分类的 `toctree`；RoboReward 归入 Benchmark。
5. 如已有相关的主题导读，可添加普通 `:doc:` 链接方便串联阅读。执行严格构建，并在浏览器里确认文章出现在对应分类下。

## 维护主题与系列导读

已有的 `index-主题-HEAD.rst` 保留原页面地址与导读内容，使用普通 `:doc:` 链接串联相关笔记。它们带有 `:orphan:` 元数据，独立于主导航，不再用 `toctree` 重复收录文章。首页与分类页共同维护唯一的分类树，因此面包屑、上一篇／下一篇和“随机逛一篇”都沿分类目录工作。

需要补充系列导读时，复制 `docs/templates/index-project-name-HEAD.rst` 到 `docs/source/index-MyTopic-HEAD.rst`，替换标题、介绍和占位文章链接。正文仍登记到对应分类；新导读通过相关正文或已有导读中的普通链接提供入口。

导读标题应体现内容范围：单篇论文使用“项目名 论文导读”，同时包含代码或实践记录时使用“项目名 阅读与源码分析”或“项目名 阅读与实践”；关联文章列表标题统一为“相关笔记”。

栏目导读通常写两个短段，约 100–180 字：先说明这项工作解决什么问题，再交代自己的关注点和正文的阅读重点。多篇文章的栏目可以提示阅读顺序，让读者快速找到需要的内容。完整英文论文题目放在文章开头；公式推导、实现细节和实验解释放进正文相应位置。导读中独有的实践经验、对比和疑问也应保留到正文，并区分个人判断与论文结论，避免为缩短导读而直接删掉。

导读中的文章链接、分类目录条目与实际文件名必须匹配，区分大小写。不要保留指向不存在文档的占位条目。

RST 标题紧接下一行用同一符号写下划线；下划线不能比标题的显示宽度短。现有栏目统一使用较长的下划线，新标题变长时也要相应延长。不要随意改变同一文档中各标题级别使用的符号。

## 叙述视角与来源

- 转述论文的方法、数据和实验时，用“作者”“论文团队”或模型名称明确主体。例如“作者终止了评估”“MemER 检索历史关键帧”。“本文”“本研究”容易被理解为这篇笔记，通常改成论文或方法名称。
- 自己的理解、疑问和实践用“我”表述。与论文转述放在同一段时，拆开并使用“我的理解”“我的实验”等提示；自己的训练步数、损失和观察要明确属于哪次实践。
- 直接引用标注“论文原文”，中文翻译标注“译文”；引用中的“我们”以及源码中的原始注释可以保留。改写后的内容用转述口吻，不再标成原文。
- “首次”“证明”“有效”等结论注明是谁提出、来自哪组实验；作者的猜测或后续计划保留其不确定性。段落已明确主体后，后续句子可自然承接，无需每句重复“作者”。

## Markdown、公式和图片的写法

正文使用 MyST Markdown，支持 `$...$` 行内公式和 `$$...$$` 独立公式。公式中保留原始 LaTeX 反斜线，不要额外写成双反斜线；`\\` 只用于矩阵或 `aligned` 等环境中的换行。

Markdown 表格由 MyST 原生解析，单元格中可以使用 `$...$` 公式；不要重新启用 `sphinx_markdown_tables` 扩展，它会让表格里的公式绕过正常的数学解析。

行内公式示例：

```markdown
策略为 $\pi_\theta(a \mid s)$，损失为 $\mathcal{L}$。
```

独立公式前后留空行，定界符单独成行：

```markdown
$$
\mathcal{L}(\theta)
= \mathbb{E}_{(s,a)\sim\mathcal{D}}
\left[-\log \pi_\theta(a\mid s)\right]
$$
```

多行公式使用 `aligned`，避免在普通段落里放置不成对的定界符。`$$...$$` 内不要再嵌套 `equation` 或 `align` 环境，以免 MathJax 报错：

```markdown
$$
\begin{aligned}
x &= a + b \\
y &= c + d
\end{aligned}
$$
```

MyST 也支持带标签的数学指令，适合交叉引用：

````markdown
```{math}
:label: policy-objective
\mathcal{L}(\theta) = -\log \pi_\theta(a \mid s)
```

参见公式 {eq}`policy-objective`。
````

RST 栏目页里的行内公式使用 ``:math:`\pi_\theta` ``，不要照搬 Markdown 的 `$...$`。数学公式不要放在普通代码围栏或行内反引号中，否则它们会作为代码显示。

引用块（`>`）里的复杂数学公式使用 `{math}` 指令围栏，并让每一行都带上引用前缀，避免引用正文被误吞进公式：

````markdown
> ```{math}
> \begin{aligned}
> x &= a + b \\
> y &= c + d
> \end{aligned}
> ```
````

图片路径从文章文件所在目录算起。Markdown 示例：

```markdown
![Evo-1 的网络结构示意图](images/Evo-1/Evo-1-1.png)
```

需要图题或限制宽度时：

````markdown
```{figure} images/Evo-1/Evo-1-1.png
:alt: Evo-1 的网络结构示意图
:width: 90%

Evo-1 网络结构。
```
````

图片文件名和路径应使用稳定、大小写一致的命名；避免空格、括号和中文标点。优先使用相对路径，不使用本机绝对路径。大图尽量先压缩，再确认放大后文字仍可读。

编辑时还要注意：

- 用空行隔开标题、段落、列表、公式和图片。分隔线 `---` 前后留空行，不要在同一位置堆叠多条。
- 普通段落前不要无意多缩进四个空格，否则可能被识别成代码块；列表里的公式、图片要和所属条目保持正确缩进。
- 检查公式是否出现红色错误、未解析的 LaTeX 命令或溢出；检查图片是否缺失、变形、图题与正文错位。
- 使用 Markdown 文本补充截图中的关键结论，方便搜索和辅助阅读。

## 使用聚合图床上传新增配图

新增图片可以上传到聚合图床，再把返回的 HTTPS 原图链接写进文章。已有本地图片可以继续使用原路径。

仓库提供 `scripts/upload_superbed.py`，只依赖 Python 3 和 curl（macOS 自带），每次上传一张图片：

```bash
cd /Users/jaxonzhu/my-docs
python3 scripts/upload_superbed.py "/你的图片绝对路径/architecture.png" \
  --folder "blog/2026/article-name" \
  --alt "模型整体结构"
```

脚本会提示输入 API Key，输入时不显示字符；密钥仅用于本次请求，不写入文件，也不放进 curl 的进程参数。成功后输出一行 Markdown，直接粘贴到文章中。需要单独的原图链接时添加 `--url-only`。脚本也支持从 `SUPERBED_API_KEY` 环境变量读取密钥，但不要把实际密钥写进仓库、命令示例或共享配置。

接口使用 `POST https://www.superbed.cn/upload`，通过 multipart 表单发送 `file`、`token` 和 `categories`。`categories` 是图床文件夹路径，不存在时由服务自动创建。此处采用已实际验证成功的 token 表单方式；本次请求头 `X-API-Key` 方式返回了 `Missing API key`。

如使用 PicGo，在安装 `web-uploader` 插件后配置：API 地址为 `https://www.superbed.cn/upload`，POST 参数名为 `file`，JSON 路径为 `url`，自定义 Body 为 `{"token":"仅在本机填写你的 API Key","categories":"blog"}`，保存并设为默认图床。[聚合图床官方帮助](https://www.superbed.cn/help)

需要图题时，将原图链接放进 MyST figure 指令：

````markdown
```{figure} https://图床返回的实际图片链接
:alt: 模型整体结构
:width: 90%

模型整体结构。
```
````

使用 API 返回的原图 URL，不要改成浏览器跳转后的 CDN 地址；无需添加缩放或转码参数。发布前通过 `make -C docs serve` 检查文章中的图片，特别是公式截图和细小文字，并在发布后检查线上页面。当前 `check_docs.py` 跳过远程资源，因此构建通过不能证明外链可访问；在文章页面中预览也能检查防盗链限制。

## 日常更新和发布

在工作区干净时先同步；如果已有未提交修改，先保存或提交自己的修改，再拉取：

```bash
git status --short
git pull --ff-only
source .venv/bin/activate
uv pip sync docs/requirements.txt

# 编辑文章、图片和导航后：
make -C docs check
make -C docs serve
```

预览确认后停止服务器，检查改动并提交。以下路径应替换为这次实际修改的文件：

```bash
git diff --check
git diff
git status --short
git add docs/source/Evo-1-new-note.md docs/source/category-personal-practice.rst docs/source/images/Evo-1/
git commit -m "Add Evo-1 notes"
git push origin main
```

在 [GitHub Actions](https://github.com/JaxonZhu/my-docs/actions) 查看 `Documentation` 检查结果，在 [Read the Docs 后台](https://app.readthedocs.org/) 查看项目 Builds，确认构建对应刚推送的提交且成功。Read the Docs 的仓库关联、GitHub 集成和 `latest` 构建分支需要在后台保持正确；没有自动触发时可手动构建 `latest`。

GitHub Actions 和 Read the Docs 是独立运行的。CI 负责检查，Read the Docs 负责发布。Read the Docs 配置也会将 Sphinx 警告视为构建失败；额外的 HTML 检查脚本在本地和 CI 的 `make check` 中执行。如果要强制“检查通过才能合并”，可在 GitHub 的 `main` 分支保护规则中要求此检查通过。CI 工作流本身不会修改分支保护设置。

## 更新依赖

普通写作无需升级依赖。准备升级时先修改 `docs/requirements.in` 中对应的版本号，再从仓库根目录执行：

```bash
uv pip compile docs/requirements.in --universal --python-version 3.12 --upgrade -o docs/requirements.txt
uv pip sync docs/requirements.txt
uv pip check
make -C docs check
make -C docs serve
```

`--universal` 会保留跨平台所需的依赖条件，避免只在本机 macOS 上能用的锁定结果。`--upgrade` 允许重新选择间接依赖的版本；不加它时会尽量保留现有锁定版本。完整版本锁定文件兼容 uv 和 pip，Read the Docs 和 CI 都直接安装它。不要用本机 `pip freeze` 覆盖这个文件。

检查主题、目录、数学公式和图片后，将 `requirements.in` 和 `requirements.txt` 一起提交。升级 Python 时同步修改 `.python-version`、`.readthedocs.yaml` 的 Python 版本以及锁定命令的版本参数，并重新生成锁文件、验证完整构建。Python 的补丁版本由各平台在 `3.12` 系列内选择，这里锁定的是 Python 小版本和 Python 包版本，不是完整的操作系统镜像。

## 常见问题

| 现象 | 检查方法 |
| --- | --- |
| `sphinx-build: command not found` | 激活 `.venv` 并按锁文件安装依赖，确认 `python --version` 为 3.12 |
| `toctree contains reference to nonexisting document` | 检查目录条目、文件名和大小写；条目不加扩展名 |
| `document isn't included in any toctree` | 把文章加入栏目导航，或把草稿移出 `docs/source/` |
| `Title underline too short` | 延长对应 RST 标题下划线 |
| `image file not readable` | 检查相对路径和大小写，确认图片已加入 Git |
| 公式显示为源码或红色错误 | 检查数学定界符、环境和命令；浏览器开发者工具确认 MathJax 脚本加载成功 |
| 本地正常、云端失败 | 对照 Python 版本和锁文件，检查 Linux 区分大小写造成的路径问题，以及云端日志对应的提交 |
| 网站没有更新 | 先确认 `git push` 成功，再确认 Read the Docs 构建的提交、分支和状态 |

相关官方文档：[Sphinx 命令行](https://www.sphinx-doc.org/en/master/man/sphinx-build.html)、[MyST 数学](https://myst-parser.readthedocs.io/en/latest/syntax/math.html)、[uv 依赖锁定](https://docs.astral.sh/uv/pip/compile/)、[Read the Docs 配置](https://docs.readthedocs.com/platform/stable/config-file/v2.html)。
