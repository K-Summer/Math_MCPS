# Open Math MCPS - 数学计算 MCP 服务器

一个基于 Model Context Protocol (MCP) 的数学计算服务器，提供丰富的数学计算功能和工具。

## 项目概述

Open Math MCPS 是一个功能全面的数学计算服务器，通过 MCP 协议提供各种数学运算工具和资源。它支持从基础算术到高级数学函数的广泛计算需求，适用于教育、研究和日常计算场景。

## 主要功能

### 基础数学运算
- **加法** (`add`) - 两个数相加
- **减法** (`subtract`) - 两个数相减
- **乘法** (`multiply`) - 两个数相乘
- **除法** (`divide`) - 两个数相除

### 高级数学函数
- **幂运算** (`power`) - 计算幂
- **平方根** (`sqrt`) - 计算平方根
- **立方根** (`cbrt`) - 计算立方根
- **对数** (`log`, `ln`) - 计算对数和自然对数
- **指数函数** (`exp`) - 计算 e^x
- **三角函数** (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`) - 角度制的三角函数计算
- **阶乘** (`factorial`) - 计算阶乘
- **组合与排列** (`combination`, `permutation`) - 计算组合数和排列数

### 统计学工具
- **平均值** (`mean`) - 计算算术平均值
- **中位数** (`median`) - 计算中位数
- **众数** (`mode`) - 计算众数
- **方差** (`variance`) - 计算方差
- **标准差** (`standard_deviation`) - 计算标准差

### 复杂算式处理
- **表达式求值** (`evaluate_expression`) - 计算复杂的数学表达式
- **表达式简化** (`simplify_expression`) - 简化数学表达式

### 单位转换
- **角度转换** (`angle_convert`) - 角度单位转换（度与弧度）

### 数学资源
- **数学常量** (`math:constant/{constant_name}`) - 获取数学常量信息（π、e、黄金比例等）

### 数学提示生成器
- **解方程提示** (`solve_equation`) - 生成解方程的提示
- **定理证明提示** (`prove_theorem`) - 生成证明数学定理的提示
- **函数图像提示** (`create_graph`) - 生成创建函数图像的提示

## 安装与使用

### 安装依赖
```bash
uv sync
```

### 运行服务器
```bash
uv run open-math-mcps
```

### 作为 MCP 服务器使用
将服务器添加到你的 MCP 客户端配置中，然后即可通过客户端访问所有数学计算工具。

## 工具列表

服务器提供以下工具：
- `add`, `subtract`, `multiply`, `divide` - 基础四则运算
- `power`, `sqrt`, `cbrt` - 幂和根运算
- `log`, `ln`, `exp` - 对数和指数函数
- `sin`, `cos`, `tan`, `asin`, `acos`, `atan` - 三角函数
- `factorial`, `combination`, `permutation` - 组合数学
- `mean`, `median`, `mode`, `variance`, `standard_deviation` - 统计学计算
- `evaluate_expression`, `simplify_expression` - 表达式处理
- `angle_convert` - 单位转换

## 项目结构
```
open-math-mcps/
├── pyproject.toml      # 项目配置和依赖
├── README.md           # 项目文档
├── uv.lock             # 依赖锁定文件
└── src/
    └── open_math_mcps/
        └── __init__.py # 主服务器实现
```

## 技术栈
- **Python** >= 3.12
- **MCP** (Model Context Protocol) - 用于工具和资源协议
- **FastMCP** - MCP 服务器框架
- **uv** - Python 包管理器和安装器

## 许可证
本项目采用 MIT 许可证。详见项目根目录的 LICENSE 文件（如有）。

## 贡献
欢迎提交 Issue 和 Pull Request 来改进这个项目。

## 作者
K-Summer (91466399+K-Summer@users.noreply.github.com)
