# DeepSeek-FinRobot 量化智能投顾系统

本项目是一个基于多智能体的量化投资分析与决策平台，支持多只股票的基本面、舆情、风险、投资组合等自动分析，并通过网页端交互展示和下载分析报告。

---

## 1. 环境准备

建议使用 **Python 3.8+**，推荐在 Linux/MacOS/WSL 下运行，Windows 也可支持。

### 1.1. 克隆项目

```bash
git clone <your-repo-url>
cd deepseek-finrobot-master
```

### 1.2. 安装依赖

建议使用虚拟环境（可选）：

```bash
python3 -m venv finrobot_env
source finrobot_env/bin/activate
```

安装依赖（已配置国内源加速）：

```bash
bash install.sh
```

如需手动安装：

```bash
python3 -m pip install -r requirements.txt
```

---

## 2. API 密钥设置

本项目依赖 DeepSeek LLM 服务，请先[注册并获取 API Key](https://platform.deepseek.com/)。

### 设置环境变量

**Linux/MacOS/WSL:**

```bash
export DEEPSEEK_API_KEY='your_api_key_here'
```

**Windows CMD:**

```cmd
set DEEPSEEK_API_KEY=your_api_key_here
```

**Windows PowerShell:**

```powershell
$env:DEEPSEEK_API_KEY='your_api_key_here'
```

> 建议将上述命令加入你的 shell 启动脚本（如 `.bashrc`、`.zshrc`）以便每次自动加载。

---

## 3. 启动网页服务

在项目根目录下运行：

```bash
python app.py
```

首次启动会自动创建所需目录。

---

## 4. 访问与使用

1. 打开浏览器，访问 [http://localhost:5000](http://localhost:5000)
2. 按页面提示输入股票代码、投资偏好、投资期限，点击"开始分析"
3. 分析完成后可直接在网页下载各类分析报告

---

## 5. agents 文件夹说明

`deepseek_finrobot/agents/` 目录包含了本项目的核心智能体、数据采集、金融分析、工作流与测试等模块，具体说明如下：

- **main.py**
  - 主流程入口，集成了多智能体的量化分析、决策、风险评估、投资组合优化等完整流程。
  - 提供了异步数据采集、分析报告生成、决策导出、网页接口等核心功能。

- **my_agents.py**
  - 定义了各类智能体（如基本面分析、舆情分析、风险分析、投资组合调整等）的实现。
  - 支持AutoGen多智能体协作、PPO强化学习投资组合优化、LLM驱动的分析与决策。
  - 提供了丰富的分析、决策、风险评估等方法。

- **data.py**
  - 异步数据采集与整合模块，支持多只股票的基本面、行情、财务、新闻、行业、情绪等数据的高效采集。
  - 封装了AkShare、新闻、行业等多源数据的异步获取与预处理。

- **edheck_risk_kit.py**
  - 金融风险与绩效分析工具箱，包含夏普比率、最大回撤、波动率、行业因子、组合优化等经典金融分析函数。
  - 支持Fama-French、Ken French等因子数据的加载与处理。

- **workflow.py**
  - 多智能体工作流与对话管理模块。
  - 提供单智能体、多智能体协作、带工具链的代理等多种工作流实现。

- **test1.py**
  - 投资组合绩效评估与可视化测试脚本。
  - 包含收益率、波动率、夏普比率、最大回撤等指标的计算与对比分析。

- **test2.ipynb**
  - 原始投资组合、风险调整投资组合、基于PPO算法调整投资组合与沪深300基准对比实验投资组合绩效分析与可视化。

- **__init__.py**
  - 代理包初始化，导出主要智能体与工作流类。

---

## 6. 目录结构简述

- `app.py`         —— Flask 网页主入口
- `deepseek_finrobot/` —— 主要业务逻辑与智能体代码
- `templates/`      —— 网页模板
- `static/`         —— 静态资源（如图片）
- `data/`           —— 代理报告存储目录
- `install.sh`      —— 一键依赖安装脚本

---

## 7. 免责声明

本项目仅供学术与技术交流，所有投资建议和分析结果仅供参考，不构成任何投资建议。请理性投资，风险自负。

---
