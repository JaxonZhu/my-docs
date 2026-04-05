LeRobot 终端指令实现原理
==============================

一、lerobot 的终端入口
----------------------

当你用 conda 激活 lerobot 环境后，在终端输入 ``lerobot`` 再按 Tab，会看到一长串以 ``lerobot-`` 开头的命令：

::

   lerobot-train, lerobot-calibrate, lerobot-record,
   lerobot-teleoperate, lerobot-find-port, ...

这些都是 lerobot 项目预先封装好的命令行工具，每个命令对应一个独立的功能模块。

**关于工作目录**：这些命令是 **全局可用的**，不需要 cd 到 lerobot 项目文件夹下。因为 lerobot 包已经通过 ``pip install -e .`` 安装到你的 conda 环境中了，终端可以在任何目录下直接调用这些命令。

二、这些命令是怎么来的
----------------------

回到 ``pyproject.toml`` 文件（位于 ``src/lerobot/pyproject.toml``），在 ``[project.scripts]`` 部分（第 214-230 行），你可以看到所有命令的声明：

.. code-block:: toml

   [project.scripts]
   lerobot-train = "lerobot.scripts.lerobot_train:main"
   lerobot-calibrate = "lerobot.scripts.lerobot_calibrate:main"
   lerobot-record = "lerobot.scripts.lerobot_record:main"
   # ... 共 16 条

这个配置的含义是：**"把 ``src/lerobot/scripts/`` 目录下 ``lerobot_xxx.py`` 文件里的 ``main`` 函数，暴露为一个叫 ``lerobot-xxx`` 的命令行工具。"**

三、背后的原理
--------------

当你执行 ``pip install -e .`` 安装 lerobot 包时，``setuptools``（Python 的打包工具）会读取这个配置，然后在你的 Python 环境中生成对应的可执行文件。以 Windows 为例，它会在 ``Scripts/`` 目录下创建 ``lerobot-train.exe``，内容本质上就是：

.. code-block:: python

   from lerobot.scripts.lerobot_train import main
   main()

所以当你敲 ``lerobot-train`` 时，系统找到这个 exe 文件，用 Python 执行它，最终调用的就是 ``lerobot_train.py`` 里的 ``main()`` 函数。

四、参数是如何定义的
--------------------

每个脚本的参数并不是硬编码的，而是通过 **dataclass 配置类** 自动生成的。

拿 ``lerobot-train`` 举例：

- 它的入口函数在 ``src/lerobot/scripts/lerobot_train.py`` 第 153 行
- 被 ``@parser.wrap()`` 装饰器包装
- 参数来源于 ``TrainPipelineConfig`` 类（位于 ``src/lerobot/configs/train.py``）

这个 dataclass 里定义了所有可配置的字段：``batch_size`` / ``learning_rate`` / ``policy`` 类型等。``draccus`` 库会自动把它转换成命令行参数。

所以如果你运行 ``lerobot-train --help``，看到的参数列表就是从 ``TrainPipelineConfig`` 自动推导出来的。

五、如果你想添加一个新功能
--------------------------

假设你要添加一个叫 ``lerobot-foo`` 的新命令，用来完成某个独立的任务。步骤如下：

**第一步**：在 ``src/lerobot/scripts/`` 目录下创建 ``lerobot_foo.py`` 文件，在其中定义 ``main()`` 函数。

**第二步**：在 ``src/lerobot/pyproject.toml`` 的 ``[project.scripts]`` 部分添加一行：

.. code-block:: toml

   lerobot-foo = "lerobot.scripts.lerobot_foo:main"

**第三步（可选）**：如果你的命令需要可配置的参数，在 ``src/lerobot/configs/`` 下创建对应的 dataclass 配置类，然后在 ``main()`` 函数上使用 ``@parser.wrap()`` 装饰器。这样用户就能通过命令行传递参数了。

**第四步**：重新安装包（``pip install -e .``），新命令就生效了。