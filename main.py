"""
数学计算 MCP 服务器

提供各种数学计算功能，包括基础运算、高级数学函数和数学常量。
"""

import math
from mcp.server.fastmcp import FastMCP
from typing import List, Optional

# 创建一个 MCP 服务器
mcp = FastMCP("数学计算器", json_response=True)


# ===== 基础数学运算工具 =====

@mcp.tool()
def add(a: float, b: float) -> float:
    """两个数相加"""
    return a + b


@mcp.tool()
def subtract(a: float, b: float) -> float:
    """两个数相减"""
    return a - b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """两个数相乘"""
    return a * b


@mcp.tool()
def divide(a: float, b: float) -> float:
    """两个数相除"""
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b


# ===== 高级数学运算工具 =====

@mcp.tool()
def power(base: float, exponent: float) -> float:
    """计算幂"""
    return math.pow(base, exponent)


@mcp.tool()
def sqrt(number: float) -> float:
    """计算平方根"""
    if number < 0:
        raise ValueError("不能计算负数的平方根")
    return math.sqrt(number)


@mcp.tool()
def cbrt(number: float) -> float:
    """计算立方根"""
    return number ** (1/3)


@mcp.tool()
def log(number: float, base: float = math.e) -> float:
    """计算对数"""
    if number <= 0:
        raise ValueError("对数的真数必须大于零")
    if base <= 0 or base == 1:
        raise ValueError("对数的底数必须大于零且不等于1")
    return math.log(number, base)


@mcp.tool()
def ln(number: float) -> float:
    """计算自然对数"""
    return log(number, math.e)


@mcp.tool()
def exp(number: float) -> float:
    """计算指数函数 e^x"""
    return math.exp(number)


@mcp.tool()
def sin(angle: float) -> float:
    """计算正弦值（角度制）"""
    return math.sin(math.radians(angle))


@mcp.tool()
def cos(angle: float) -> float:
    """计算余弦值（角度制）"""
    return math.cos(math.radians(angle))


@mcp.tool()
def tan(angle: float) -> float:
    """计算正切值（角度制）"""
    result = math.tan(math.radians(angle))
    if abs(result) > 1e15:
        raise ValueError("正切值超出范围")
    return result


@mcp.tool()
def asin(value: float) -> float:
    """计算反正弦值（返回角度制）"""
    if abs(value) > 1:
        raise ValueError("反正弦的输入值必须在-1到1之间")
    return math.degrees(math.asin(value))


@mcp.tool()
def acos(value: float) -> float:
    """计算反余弦值（返回角度制）"""
    if abs(value) > 1:
        raise ValueError("反余弦的输入值必须在-1到1之间")
    return math.degrees(math.acos(value))


@mcp.tool()
def atan(value: float) -> float:
    """计算反正切值（返回角度制）"""
    return math.degrees(math.atan(value))


@mcp.tool()
def factorial(n: int) -> int:
    """计算阶乘"""
    if n < 0:
        raise ValueError("阶乘只能计算非负整数")
    if n > 170:
        raise ValueError("阶乘值太大，超出计算范围")
    return math.factorial(n)


@mcp.tool()
def combination(n: int, r: int) -> int:
    """计算组合数 C(n, r)"""
    if n < 0 or r < 0 or r > n:
        raise ValueError("组合数参数无效")
    return math.comb(n, r)


@mcp.tool()
def permutation(n: int, r: int) -> int:
    """计算排列数 P(n, r)"""
    if n < 0 or r < 0 or r > n:
        raise ValueError("排列数参数无效")
    return math.perm(n, r)


# ===== 统计学工具 =====

@mcp.tool()
def mean(numbers: List[float]) -> float:
    """计算平均值"""
    if not numbers:
        raise ValueError("列表不能为空")
    return sum(numbers) / len(numbers)


@mcp.tool()
def median(numbers: List[float]) -> float:
    """计算中位数"""
    if not numbers:
        raise ValueError("列表不能为空")
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 1:
        return sorted_numbers[n//2]
    else:
        return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2


@mcp.tool()
def mode(numbers: List[float]) -> List[float]:
    """计算众数"""
    if not numbers:
        raise ValueError("列表不能为空")
    frequency = {}
    for num in numbers:
        frequency[num] = frequency.get(num, 0) + 1
    max_freq = max(frequency.values())
    return [num for num, freq in frequency.items() if freq == max_freq]


@mcp.tool()
def variance(numbers: List[float]) -> float:
    """计算方差"""
    if len(numbers) < 2:
        raise ValueError("至少需要两个数据点")
    avg = mean(numbers)
    return sum((x - avg) ** 2 for x in numbers) / len(numbers)


@mcp.tool()
def standard_deviation(numbers: List[float]) -> float:
    """计算标准差"""
    return math.sqrt(variance(numbers))


# ===== 数学常量资源 =====

@mcp.resource("math:constant/{constant_name}")
def get_math_constant(constant_name: str) -> dict:
    """获取数学常量"""
    constants = {
        "pi": {
            "name": "圆周率",
            "symbol": "π",
            "value": math.pi,
            "description": "圆的周长与直径之比",
            "approximations": ["3.14159", "22/7"]
        },
        "e": {
            "name": "自然常数",
            "symbol": "e",
            "value": math.e,
            "description": "自然对数的底数",
            "approximations": ["2.71828", "2.718"]
        },
        "golden_ratio": {
            "name": "黄金比例",
            "symbol": "φ",
            "value": (1 + math.sqrt(5)) / 2,
            "description": "数学中的黄金比例常数",
            "approximations": ["1.61803", "1.618"]
        },
        "sqrt_2": {
            "name": "根号2",
            "symbol": "√2",
            "value": math.sqrt(2),
            "description": "2的平方根",
            "approximations": ["1.41421", "1.414"]
        }
    }

    if constant_name in constants:
        return constants[constant_name]
    else:
        raise ValueError(f"未知的数学常量: {constant_name}")


# ===== 数学公式提示生成器 =====

@mcp.prompt()
def solve_equation(equation_type: str, variables: dict, context: str = "") -> str:
    """生成解方程的提示"""
    prompts = {
        "linear": f"请解以下一元一次方程：{context or '请提供具体方程'}",
        "quadratic": f"请解以下二次方程：{context or '请提供具体方程'}。请使用求根公式，并说明每一步的推导过程。",
        "system": f"请解以下方程组：{context or '请提供具体方程组'}。请使用代入法、消元法或其他适当的方法。",
        "inequality": f"请解以下不等式：{context or '请提供具体不等式'}。请注意不等式的性质和解的表示方法。"
    }

    prompt = prompts.get(equation_type, f"请解以下数学问题：{context}")

    if variables:
        prompt += "\n\n已知变量：" + ", ".join(f"{k} = {v}" for k, v in variables.items())

    return prompt + "\n\n请详细展示解题步骤和最终答案。"


@mcp.prompt()
def prove_theorem(theorem: str, method: str = "direct") -> str:
    """生成证明数学定理的提示"""
    method_descriptions = {
        "direct": "直接证明法",
        "contradiction": "反证法",
        "induction": "数学归纳法",
        "contrapositive": "逆否命题法"
    }

    return f"""请使用{method_descriptions.get(method, method)}证明以下定理：
{theorem}

请按照以下格式进行证明：
1. 明确已知条件和要证明的结论
2. 证明的详细步骤
3. 最终结论

每个步骤都要有充分的理由和依据。"""


@mcp.prompt()
def create_graph(function_type: str, parameters: dict) -> str:
    """生成创建图形的提示"""
    return f"""请绘制以下{function_type}函数的图像：

函数参数：{parameters}

请在坐标系中绘制该函数的图像，并：
1. 标出关键点（如极值点、零点等）
2. 标出渐近线（如果有）
3. 标出定义域和值域
4. 添加适当的坐标轴标签和标题

使用表格或列表形式说明图像的特征。"""


# ===== 单位转换工具 =====

@mcp.tool()
def angle_convert(angle: float, from_unit: str, to_unit: str) -> float:
    """角度单位转换"""
    # 转换为弧度
    if from_unit == "degree":
        radians = math.radians(angle)
    elif from_unit == "radian":
        radians = angle
    else:
        raise ValueError("不支持的输入单位，请使用 'degree' 或 'radian'")

    # 从弧度转换到目标单位
    if to_unit == "degree":
        return math.degrees(radians)
    elif to_unit == "radian":
        return radians
    else:
        raise ValueError("不支持的输出单位，请使用 'degree' 或 'radian'")


if __name__ == "__main__":
    # 运行服务器
    mcp.run(transport="stdio")