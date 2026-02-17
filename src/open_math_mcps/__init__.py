"""
数学计算 MCP 服务器

提供各种数学计算功能，包括基础运算、高级数学函数、数学常量和复杂算式处理。
支持高精度计算、矩阵运算、微积分等高级功能。
"""

import math
import re
import ast
import decimal
import time
import json
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
from mcp.server.fastmcp import FastMCP
from functools import lru_cache
from dataclasses import dataclass
from enum import Enum

# 创建一个 MCP 服务器
mcp = FastMCP("数学计算器", json_response=True)

# 全局配置
class PrecisionMode(Enum):
    STANDARD = "standard"  # 标准精度（默认）
    HIGH = "high"          # 高精度（使用decimal）
    EXTREME = "extreme"    # 极高精度（使用mpmath，如果可用）

# 计算缓存
calculation_cache = {}
cache_enabled = True
cache_ttl = 3600  # 缓存时间（秒）

# 计算历史
calculation_history = []
max_history_size = 100

@dataclass
class CalculationResult:
    """计算结果数据类"""
    tool: str
    parameters: Dict[str, Any]
    result: Any
    timestamp: float
    precision: str = "standard"
    execution_time: float = 0.0
    cache_hit: bool = False

def get_precision_context(precision_mode: PrecisionMode = PrecisionMode.STANDARD, 
                         decimal_places: int = 15) -> Optional[decimal.Context]:
    """获取精度上下文"""
    if precision_mode == PrecisionMode.HIGH:
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        return ctx
    return None

def cache_key(tool_name: str, params: Dict) -> str:
    """生成缓存键"""
    return f"{tool_name}:{json.dumps(params, sort_keys=True)}"

def add_to_history(result: CalculationResult):
    """添加到历史记录"""
    calculation_history.append(result)
    if len(calculation_history) > max_history_size:
        calculation_history.pop(0)

def format_error(error_type: str, message: str, suggestion: str = "", 
                valid_range: str = "") -> Dict:
    """格式化错误信息"""
    return {
        "error": {
            "code": error_type,
            "message": message,
            "suggestion": suggestion,
            "valid_range": valid_range,
            "timestamp": time.time()
        }
    }


# ===== 基础数学运算工具（带高精度支持） =====

@mcp.tool()
def add(a: float, b: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """两个数相加
    
    参数:
        a: 第一个数
        b: 第二个数
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        result = decimal.Decimal(str(a)) + decimal.Decimal(str(b))
    else:
        result = a + b
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="add",
        parameters={"a": a, "b": b, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def subtract(a: float, b: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """两个数相减
    
    参数:
        a: 第一个数
        b: 第二个数
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        result = decimal.Decimal(str(a)) - decimal.Decimal(str(b))
    else:
        result = a - b
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="subtract",
        parameters={"a": a, "b": b, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def multiply(a: float, b: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """两个数相乘
    
    参数:
        a: 第一个数
        b: 第二个数
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        result = decimal.Decimal(str(a)) * decimal.Decimal(str(b))
    else:
        result = a * b
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="multiply",
        parameters={"a": a, "b": b, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def divide(a: float, b: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """两个数相除
    
    参数:
        a: 被除数
        b: 除数
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if b == 0:
        error_info = format_error(
            "DIVISION_BY_ZERO",
            "除数不能为零",
            "请检查除数是否为0",
            "b ≠ 0"
        )
        raise ValueError(json.dumps(error_info))
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        result = decimal.Decimal(str(a)) / decimal.Decimal(str(b))
    else:
        result = a / b
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="divide",
        parameters={"a": a, "b": b, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


# ===== 高级数学运算工具（带改进的错误处理） =====

@mcp.tool()
def power(base: float, exponent: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算幂
    
    参数:
        base: 底数
        exponent: 指数
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        result = decimal.Decimal(str(base)) ** decimal.Decimal(str(exponent))
    else:
        result = math.pow(base, exponent)
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="power",
        parameters={"base": base, "exponent": exponent, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def sqrt(number: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算平方根
    
    参数:
        number: 要计算平方根的数
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if number < 0:
        error_info = format_error(
            "NEGATIVE_SQUARE_ROOT",
            "不能计算负数的平方根",
            "请确保输入的数是非负数",
            "number ≥ 0"
        )
        raise ValueError(json.dumps(error_info))
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        result = decimal.Decimal(str(number)).sqrt()
    else:
        result = math.sqrt(number)
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="sqrt",
        parameters={"number": number, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def cbrt(number: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算立方根
    
    参数:
        number: 要计算立方根的数
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        # 立方根计算：x^(1/3)
        result = decimal.Decimal(str(number)) ** (decimal.Decimal('1') / decimal.Decimal('3'))
    else:
        result = number ** (1/3)
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="cbrt",
        parameters={"number": number, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def log(number: float, base: float = math.e, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算对数
    
    参数:
        number: 真数
        base: 底数（默认为e）
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if number <= 0:
        error_info = format_error(
            "INVALID_LOG_ARGUMENT",
            "对数的真数必须大于零",
            "请确保输入的数大于0",
            "number > 0"
        )
        raise ValueError(json.dumps(error_info))
    
    if base <= 0 or base == 1:
        error_info = format_error(
            "INVALID_LOG_BASE",
            "对数的底数必须大于零且不等于1",
            "请确保底数大于0且不等于1",
            "base > 0 and base ≠ 1"
        )
        raise ValueError(json.dumps(error_info))
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        # 使用换底公式：log_b(a) = ln(a) / ln(b)
        a = decimal.Decimal(str(number))
        b = decimal.Decimal(str(base))
        result = a.ln() / b.ln()
    else:
        result = math.log(number, base)
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="log",
        parameters={"number": number, "base": base, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def ln(number: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算自然对数
    
    参数:
        number: 真数
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    return log(number, math.e, precision, decimal_places)


@mcp.tool()
def exp(number: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算指数函数 e^x
    
    参数:
        number: 指数
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        result = decimal.Decimal(str(number)).exp()
    else:
        result = math.exp(number)
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="exp",
        parameters={"number": number, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def sin(angle: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算正弦值（角度制）
    
    参数:
        angle: 角度（度）
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        # 使用math.sin计算，然后转换为decimal
        result = decimal.Decimal(str(math.sin(math.radians(angle))))
    else:
        result = math.sin(math.radians(angle))
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="sin",
        parameters={"angle": angle, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def cos(angle: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算余弦值（角度制）
    
    参数:
        angle: 角度（度）
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        # 使用math.cos计算，然后转换为decimal
        result = decimal.Decimal(str(math.cos(math.radians(angle))))
    else:
        result = math.cos(math.radians(angle))
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="cos",
        parameters={"angle": angle, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def tan(angle: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算正切值（角度制）
    
    参数:
        angle: 角度（度）
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        # 使用math.tan计算，然后转换为decimal
        result = decimal.Decimal(str(math.tan(math.radians(angle))))
    else:
        result = math.tan(math.radians(angle))
    
    if abs(float(result)) > 1e15:
        error_info = format_error(
            "TANGENT_OVERFLOW",
            "正切值超出范围",
            "角度接近90度的奇数倍，正切值趋于无穷大",
            "angle ≠ 90° + k·180° (k为整数)"
        )
        raise ValueError(json.dumps(error_info))
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="tan",
        parameters={"angle": angle, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def asin(value: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算反正弦值（返回角度制）
    
    参数:
        value: 正弦值（-1到1之间）
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if abs(value) > 1:
        error_info = format_error(
            "INVALID_ARCSIN_ARGUMENT",
            "反正弦的输入值必须在-1到1之间",
            "请确保输入值在-1到1范围内",
            "-1 ≤ value ≤ 1"
        )
        raise ValueError(json.dumps(error_info))
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        # 使用math.asin计算，然后转换为decimal
        result = decimal.Decimal(str(math.degrees(math.asin(value))))
    else:
        result = math.degrees(math.asin(value))
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="asin",
        parameters={"value": value, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def acos(value: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算反余弦值（返回角度制）
    
    参数:
        value: 余弦值（-1到1之间）
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if abs(value) > 1:
        error_info = format_error(
            "INVALID_ARCCOS_ARGUMENT",
            "反余弦的输入值必须在-1到1之间",
            "请确保输入值在-1到1范围内",
            "-1 ≤ value ≤ 1"
        )
        raise ValueError(json.dumps(error_info))
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        # 使用math.acos计算，然后转换为decimal
        result = decimal.Decimal(str(math.degrees(math.acos(value))))
    else:
        result = math.degrees(math.acos(value))
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="acos",
        parameters={"value": value, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def atan(value: float, precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算反正切值（返回角度制）
    
    参数:
        value: 正切值
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        # 计算反正切（弧度）然后转换为角度
        # 使用math.atan计算反正切，然后转换为decimal
        radians = decimal.Decimal(str(math.atan(value)))
        result = radians * decimal.Decimal('180') / decimal.Decimal(str(math.pi))
    else:
        result = math.degrees(math.atan(value))
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="atan",
        parameters={"value": value, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def factorial(n: int, precision: str = "standard") -> Union[int, decimal.Decimal]:
    """计算阶乘
    
    参数:
        n: 非负整数
        precision: 精度模式 (standard/high)
    """
    start_time = time.time()
    
    if n < 0:
        error_info = format_error(
            "NEGATIVE_FACTORIAL",
            "阶乘只能计算非负整数",
            "请确保输入的是非负整数",
            "n ≥ 0"
        )
        raise ValueError(json.dumps(error_info))
    
    if n > 170 and precision == "standard":
        error_info = format_error(
            "FACTORIAL_OVERFLOW",
            "阶乘值太大，超出标准精度计算范围",
            "请使用高精度模式或减小输入值",
            "n ≤ 170 (标准模式)"
        )
        raise ValueError(json.dumps(error_info))
    
    if precision == "high":
        ctx = decimal.Context(prec=100)  # 高精度阶乘需要更多位数
        decimal.setcontext(ctx)
        result = decimal.Decimal(1)
        for i in range(1, n + 1):
            result *= decimal.Decimal(str(i))
    else:
        result = math.factorial(n)
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="factorial",
        parameters={"n": n, "precision": precision},
        result=int(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return int(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def combination(n: int, r: int, precision: str = "standard") -> Union[int, decimal.Decimal]:
    """计算组合数 C(n, r)
    
    参数:
        n: 总数
        r: 选择数
        precision: 精度模式 (standard/high)
    """
    start_time = time.time()
    
    if n < 0 or r < 0 or r > n:
        error_info = format_error(
            "INVALID_COMBINATION",
            "组合数参数无效",
            "请确保 n ≥ 0, r ≥ 0, 且 r ≤ n",
            "n ≥ 0, r ≥ 0, r ≤ n"
        )
        raise ValueError(json.dumps(error_info))
    
    if precision == "high":
        # 使用公式 C(n, r) = n! / (r! * (n-r)!)
        n_fact = factorial(n, "high")
        r_fact = factorial(r, "high")
        n_r_fact = factorial(n - r, "high")
        result = n_fact / (r_fact * n_r_fact)
    else:
        result = math.comb(n, r)
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="combination",
        parameters={"n": n, "r": r, "precision": precision},
        result=int(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return int(result) if isinstance(result, decimal.Decimal) else result


@mcp.tool()
def permutation(n: int, r: int, precision: str = "standard") -> Union[int, decimal.Decimal]:
    """计算排列数 P(n, r)
    
    参数:
        n: 总数
        r: 选择数
        precision: 精度模式 (standard/high)
    """
    start_time = time.time()
    
    if n < 0 or r < 0 or r > n:
        error_info = format_error(
            "INVALID_PERMUTATION",
            "排列数参数无效",
            "请确保 n ≥ 0, r ≥ 0, 且 r ≤ n",
            "n ≥ 0, r ≥ 0, r ≤ n"
        )
        raise ValueError(json.dumps(error_info))
    
    if precision == "high":
        # 使用公式 P(n, r) = n! / (n-r)!
        n_fact = factorial(n, "high")
        n_r_fact = factorial(n - r, "high")
        result = n_fact / n_r_fact
    else:
        result = math.perm(n, r)
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="permutation",
        parameters={"n": n, "r": r, "precision": precision},
        result=int(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return int(result) if isinstance(result, decimal.Decimal) else result


# ===== 统计学工具 =====

@mcp.tool()
def mean(numbers: List[float], precision: str = "standard", decimal_places: int = 15) -> Union[float, decimal.Decimal]:
    """计算平均值
    
    参数:
        numbers: 数字列表
        precision: 精度模式 (standard/high)
        decimal_places: 高精度模式下的小数位数
    """
    start_time = time.time()
    
    if not numbers:
        error_info = format_error(
            "EMPTY_LIST",
            "列表不能为空",
            "请提供至少一个数字",
            "len(numbers) > 0"
        )
        raise ValueError(json.dumps(error_info))
    
    if precision == "high":
        ctx = decimal.Context(prec=decimal_places)
        decimal.setcontext(ctx)
        total = decimal.Decimal('0')
        for num in numbers:
            total += decimal.Decimal(str(num))
        result = total / decimal.Decimal(str(len(numbers)))
    else:
        result = sum(numbers) / len(numbers)
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="mean",
        parameters={"numbers": numbers, "precision": precision, "decimal_places": decimal_places},
        result=float(result) if isinstance(result, decimal.Decimal) else result,
        timestamp=time.time(),
        precision=precision,
        execution_time=time.time() - start_time
    ))
    
    return float(result) if isinstance(result, decimal.Decimal) else result


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


# ===== 复杂算式处理工具 =====

@mcp.tool()
def evaluate_expression(expression: str, variables: Optional[Dict[str, float]] = None) -> float:
    """
    计算复杂的数学表达式
    
    参数:
        expression: 数学表达式字符串
        variables: 变量名到值的映射字典（可选）
    
    支持的运算符:
        +, -, *, /, ** (幂), % (取模)
        函数: sin, cos, tan, asin, acos, atan, sqrt, log, ln, exp, factorial
        常量: pi, e
    
    示例:
        "2 * (3 + 4)" -> 14.0
        "sin(30) + cos(60)" -> 1.0
        "x + y" (其中 variables={"x": 2, "y": 3}) -> 5.0
    """
    # 替换常量
    expression = expression.replace("pi", str(math.pi))
    expression = expression.replace("e", str(math.e))
    
    # 替换变量
    if variables:
        for var, value in variables.items():
            # 使用正则表达式确保匹配完整的变量名
            expression = re.sub(rf"\b{var}\b", str(value), expression)
    
    # 预处理表达式，确保安全
    try:
        # 检查表达式是否只包含允许的字符
        allowed_chars = r"^[0-9+\-*/.() **,%\s\w]+$"
        if not re.match(allowed_chars, expression):
            raise ValueError("表达式包含不允许的字符")
        
        # 使用ast.literal_eval进行基本评估
        # 对于更复杂的表达式，我们使用eval，但有安全检查
        # 注意：在生产环境中，使用eval有安全风险，这里简化处理
        
        # 定义安全的函数和常量
        safe_dict = {
            "__builtins__": {},
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sqrt": math.sqrt,
            "log": math.log,
            "ln": math.log,
            "exp": math.exp,
            "factorial": math.factorial,
            "pi": math.pi,
            "e": math.e,
            "radians": math.radians,
            "degrees": math.degrees
        }
        
        # 计算表达式
        result = eval(expression, {"__builtins__": None}, safe_dict)
        
        # 确保结果是数字
        if not isinstance(result, (int, float)):
            raise ValueError("表达式计算结果不是数字")
            
        return float(result)
        
    except (SyntaxError, NameError, TypeError, ZeroDivisionError) as e:
        raise ValueError(f"表达式计算错误: {str(e)}")


@mcp.tool()
def simplify_expression(expression: str) -> str:
    """
    简化数学表达式（基础实现）
    
    参数:
        expression: 数学表达式字符串
    
    返回:
        简化后的表达式字符串
    """
    # 这里实现一个基础的简化逻辑
    # 在实际应用中，可以使用更复杂的符号计算库
    
    # 移除不必要的空格
    simplified = expression.replace(" ", "")
    
    # 处理简单的 +0 或 *1
    simplified = re.sub(r"\+0(?=[+\-*/)])", "", simplified)
    simplified = re.sub(r"\*1(?=[+\-*/)])", "", simplified)
    
    # 处理 0+x 或 x+0
    simplified = re.sub(r"0\+(?=[0-9.])", "", simplified)
    simplified = re.sub(r"\+0$", "", simplified)
    
    # 处理 1*x 或 x*1
    simplified = re.sub(r"1\*(?=[0-9.])", "", simplified)
    simplified = re.sub(r"\*1$", "", simplified)
    
    # 处理 x-0
    simplified = re.sub(r"(?<![0-9.])-0(?=[+\-*/)])", "", simplified)
    simplified = re.sub(r"-0$", "", simplified)
    
    # 处理 x/1
    simplified = re.sub(r"(?<![0-9.])/1(?=[+\-*/)])", "", simplified)
    simplified = re.sub(r"/1$", "", simplified)
    
    return simplified


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


# ===== 矩阵运算工具 =====

@mcp.tool()
def matrix_multiply(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
    """矩阵乘法
    
    参数:
        matrix_a: 第一个矩阵 (m x n)
        matrix_b: 第二个矩阵 (n x p)
    
    返回:
        结果矩阵 (m x p)
    """
    start_time = time.time()
    
    # 检查矩阵维度
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0]) if rows_a > 0 else 0
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0]) if rows_b > 0 else 0
    
    if cols_a != rows_b:
        error_info = format_error(
            "MATRIX_DIMENSION_MISMATCH",
            "矩阵维度不匹配，无法相乘",
            f"矩阵A的列数({cols_a})必须等于矩阵B的行数({rows_b})",
            f"cols_a = {cols_a}, rows_b = {rows_b}"
        )
        raise ValueError(json.dumps(error_info))
    
    # 使用numpy进行矩阵乘法
    try:
        result = np.dot(matrix_a, matrix_b).tolist()
    except Exception as e:
        error_info = format_error(
            "MATRIX_MULTIPLICATION_ERROR",
            f"矩阵乘法失败: {str(e)}",
            "请检查矩阵数据是否正确",
            ""
        )
        raise ValueError(json.dumps(error_info))
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="matrix_multiply",
        parameters={"matrix_a": matrix_a, "matrix_b": matrix_b},
        result=result,
        timestamp=time.time(),
        execution_time=time.time() - start_time
    ))
    
    return result


@mcp.tool()
def matrix_determinant(matrix: List[List[float]]) -> float:
    """计算矩阵的行列式
    
    参数:
        matrix: 方阵 (n x n)
    
    返回:
        行列式值
    """
    start_time = time.time()
    
    # 检查是否为方阵
    rows = len(matrix)
    if rows == 0:
        error_info = format_error(
            "EMPTY_MATRIX",
            "矩阵不能为空",
            "请提供非空矩阵",
            "len(matrix) > 0"
        )
        raise ValueError(json.dumps(error_info))
    
    for row in matrix:
        if len(row) != rows:
            error_info = format_error(
                "NON_SQUARE_MATRIX",
                "矩阵必须是方阵",
                f"矩阵行数({rows})必须等于列数({len(row)})",
                f"rows = {rows}, cols = {len(row)}"
            )
            raise ValueError(json.dumps(error_info))
    
    # 使用numpy计算行列式
    try:
        result = float(np.linalg.det(matrix))
    except Exception as e:
        error_info = format_error(
            "DETERMINANT_CALCULATION_ERROR",
            f"行列式计算失败: {str(e)}",
            "请检查矩阵数据是否正确",
            ""
        )
        raise ValueError(json.dumps(error_info))
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="matrix_determinant",
        parameters={"matrix": matrix},
        result=result,
        timestamp=time.time(),
        execution_time=time.time() - start_time
    ))
    
    return result


@mcp.tool()
def matrix_inverse(matrix: List[List[float]]) -> List[List[float]]:
    """计算矩阵的逆矩阵
    
    参数:
        matrix: 可逆方阵 (n x n)
    
    返回:
        逆矩阵
    """
    start_time = time.time()
    
    # 检查是否为方阵
    rows = len(matrix)
    if rows == 0:
        error_info = format_error(
            "EMPTY_MATRIX",
            "矩阵不能为空",
            "请提供非空矩阵",
            "len(matrix) > 0"
        )
        raise ValueError(json.dumps(error_info))
    
    for row in matrix:
        if len(row) != rows:
            error_info = format_error(
                "NON_SQUARE_MATRIX",
                "矩阵必须是方阵",
                f"矩阵行数({rows})必须等于列数({len(row)})",
                f"rows = {rows}, cols = {len(row)}"
            )
            raise ValueError(json.dumps(error_info))
    
    # 使用numpy计算逆矩阵
    try:
        result = np.linalg.inv(matrix).tolist()
    except np.linalg.LinAlgError:
        error_info = format_error(
            "SINGULAR_MATRIX",
            "矩阵不可逆（奇异矩阵）",
            "请检查矩阵是否满秩",
            "det(matrix) ≠ 0"
        )
        raise ValueError(json.dumps(error_info))
    except Exception as e:
        error_info = format_error(
            "MATRIX_INVERSE_ERROR",
            f"逆矩阵计算失败: {str(e)}",
            "请检查矩阵数据是否正确",
            ""
        )
        raise ValueError(json.dumps(error_info))
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="matrix_inverse",
        parameters={"matrix": matrix},
        result=result,
        timestamp=time.time(),
        execution_time=time.time() - start_time
    ))
    
    return result


@mcp.tool()
def matrix_transpose(matrix: List[List[float]]) -> List[List[float]]:
    """计算矩阵的转置
    
    参数:
        matrix: 矩阵 (m x n)
    
    返回:
        转置矩阵 (n x m)
    """
    start_time = time.time()
    
    if not matrix:
        error_info = format_error(
            "EMPTY_MATRIX",
            "矩阵不能为空",
            "请提供非空矩阵",
            "len(matrix) > 0"
        )
        raise ValueError(json.dumps(error_info))
    
    # 计算转置
    result = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="matrix_transpose",
        parameters={"matrix": matrix},
        result=result,
        timestamp=time.time(),
        execution_time=time.time() - start_time
    ))
    
    return result


# ===== 微积分工具 =====

@mcp.tool()
def differentiate(expression: str, variable: str = "x", point: Optional[float] = None) -> Union[str, float]:
    """计算函数的导数
    
    参数:
        expression: 函数表达式字符串
        variable: 求导变量（默认为'x'）
        point: 求导点（如果提供，则计算该点的导数值）
    
    返回:
        导数表达式或导数值
    """
    start_time = time.time()
    
    # 这里实现一个基础的数值微分
    # 在实际应用中，可以使用sympy等符号计算库
    
    if point is not None:
        # 数值微分：使用中心差分法
        h = 1e-6
        try:
            # 计算f(x+h)和f(x-h)
            variables_plus = {variable: point + h}
            variables_minus = {variable: point - h}
            
            # 使用现有的evaluate_expression函数
            f_plus = evaluate_expression(expression, variables_plus)
            f_minus = evaluate_expression(expression, variables_minus)
            
            result = (f_plus - f_minus) / (2 * h)
        except Exception as e:
            error_info = format_error(
                "DIFFERENTIATION_ERROR",
                f"数值微分失败: {str(e)}",
                "请检查表达式和求导点",
                ""
            )
            raise ValueError(json.dumps(error_info))
        
        # 记录到历史
        add_to_history(CalculationResult(
            tool="differentiate",
            parameters={"expression": expression, "variable": variable, "point": point},
            result=result,
            timestamp=time.time(),
            execution_time=time.time() - start_time
        ))
        
        return result
    else:
        # 返回导数表达式（简化版）
        # 注意：这是一个非常基础的实现，仅支持简单表达式
        derivative = f"d/d{variable}({expression})"
        
        # 记录到历史
        add_to_history(CalculationResult(
            tool="differentiate",
            parameters={"expression": expression, "variable": variable, "point": None},
            result=derivative,
            timestamp=time.time(),
            execution_time=time.time() - start_time
        ))
        
        return derivative


@mcp.tool()
def integrate(expression: str, variable: str = "x", 
              lower_limit: Optional[float] = None, 
              upper_limit: Optional[float] = None) -> Union[str, float]:
    """计算函数的积分
    
    参数:
        expression: 函数表达式字符串
        variable: 积分变量（默认为'x'）
        lower_limit: 积分下限（如果提供，则计算定积分）
        upper_limit: 积分上限（如果提供，则计算定积分）
    
    返回:
        积分表达式或积分值
    """
    start_time = time.time()
    
    if lower_limit is not None and upper_limit is not None:
        # 数值积分：使用辛普森法则
        try:
            # 使用数值积分
            n = 1000  # 分割数
            a = lower_limit
            b = upper_limit
            h = (b - a) / n
            
            # 计算函数值
            total = 0
            for i in range(n + 1):
                x = a + i * h
                variables = {variable: x}
                fx = evaluate_expression(expression, variables)
                
                if i == 0 or i == n:
                    total += fx
                elif i % 2 == 1:
                    total += 4 * fx
                else:
                    total += 2 * fx
            
            result = total * h / 3
        except Exception as e:
            error_info = format_error(
                "INTEGRATION_ERROR",
                f"数值积分失败: {str(e)}",
                "请检查表达式和积分限",
                ""
            )
            raise ValueError(json.dumps(error_info))
        
        # 记录到历史
        add_to_history(CalculationResult(
            tool="integrate",
            parameters={"expression": expression, "variable": variable, 
                       "lower_limit": lower_limit, "upper_limit": upper_limit},
            result=result,
            timestamp=time.time(),
            execution_time=time.time() - start_time
        ))
        
        return result
    else:
        # 返回积分表达式
        integral = f"∫{expression} d{variable}"
        
        # 记录到历史
        add_to_history(CalculationResult(
            tool="integrate",
            parameters={"expression": expression, "variable": variable, 
                       "lower_limit": None, "upper_limit": None},
            result=integral,
            timestamp=time.time(),
            execution_time=time.time() - start_time
        ))
        
        return integral


@mcp.tool()
def solve_equation_numeric(equation: str, variable: str = "x", 
                          initial_guess: float = 0.0, 
                          tolerance: float = 1e-6, 
                          max_iterations: int = 100) -> float:
    """数值解方程（使用牛顿法）
    
    参数:
        equation: 方程字符串（如 "x^2 - 4 = 0"）
        variable: 变量名（默认为'x'）
        initial_guess: 初始猜测值
        tolerance: 容差
        max_iterations: 最大迭代次数
    
    返回:
        方程的根
    """
    start_time = time.time()
    
    # 解析方程：将方程转换为 f(x) = 0 的形式
    if "=" in equation:
        parts = equation.split("=")
        if len(parts) != 2:
            error_info = format_error(
                "INVALID_EQUATION",
                "方程格式无效",
                "方程应包含一个等号，格式如 'x^2 - 4 = 0'",
                "equation must contain exactly one '='"
            )
            raise ValueError(json.dumps(error_info))
        
        left_expr = parts[0].strip()
        right_expr = parts[1].strip()
        
        # 转换为 f(x) = left - right = 0
        expression = f"({left_expr}) - ({right_expr})"
    else:
        expression = equation
    
    # 牛顿法迭代
    x = initial_guess
    for i in range(max_iterations):
        try:
            # 计算f(x)
            variables = {variable: x}
            fx = evaluate_expression(expression, variables)
            
            # 计算f'(x)（数值微分）
            h = 1e-6
            variables_plus = {variable: x + h}
            variables_minus = {variable: x - h}
            fx_plus = evaluate_expression(expression, variables_plus)
            fx_minus = evaluate_expression(expression, variables_minus)
            fpx = (fx_plus - fx_minus) / (2 * h)
            
            if abs(fpx) < 1e-12:
                error_info = format_error(
                    "ZERO_DERIVATIVE",
                    "导数为零，牛顿法失败",
                    "请尝试不同的初始猜测值",
                    f"f'({x}) ≈ 0"
                )
                raise ValueError(json.dumps(error_info))
            
            # 更新x
            x_new = x - fx / fpx
            
            # 检查收敛
            if abs(x_new - x) < tolerance:
                x = x_new
                break
            
            x = x_new
        except Exception as e:
            error_info = format_error(
                "NEWTON_METHOD_ERROR",
                f"牛顿法迭代失败: {str(e)}",
                "请检查方程和初始猜测值",
                ""
            )
            raise ValueError(json.dumps(error_info))
    
    # 记录到历史
    add_to_history(CalculationResult(
        tool="solve_equation_numeric",
        parameters={"equation": equation, "variable": variable, 
                   "initial_guess": initial_guess, "tolerance": tolerance, 
                   "max_iterations": max_iterations},
        result=x,
        timestamp=time.time(),
        execution_time=time.time() - start_time
    ))
    
    return x


# ===== 用户体验工具 =====

@mcp.tool()
def get_calculation_history(limit: int = 10) -> List[Dict]:
    """获取计算历史
    
    参数:
        limit: 返回的历史记录数量限制
    
    返回:
        计算历史记录列表
    """
    # 返回最近的记录
    recent_history = calculation_history[-limit:] if calculation_history else []
    
    # 转换为可序列化的字典
    history_list = []
    for result in recent_history:
        history_list.append({
            "tool": result.tool,
            "parameters": result.parameters,
            "result": result.result,
            "timestamp": result.timestamp,
            "precision": result.precision,
            "execution_time": result.execution_time,
            "cache_hit": result.cache_hit
        })
    
    return history_list


@mcp.tool()
def clear_calculation_history() -> Dict:
    """清空计算历史"""
    calculation_history.clear()
    return {"status": "success", "message": "计算历史已清空", "cleared_count": 0}


@mcp.tool()
def format_number(number: float, format_type: str = "decimal", 
                 precision: int = 6, use_scientific: bool = False) -> str:
    """格式化数字
    
    参数:
        number: 要格式化的数字
        format_type: 格式类型 (decimal/scientific/engineering)
        precision: 精度（小数位数）
        use_scientific: 是否使用科学计数法（当format_type为decimal时）
    
    返回:
        格式化后的字符串
    """
    if format_type == "scientific":
        return f"{number:.{precision}e}"
    elif format_type == "engineering":
        # 工程计数法：指数是3的倍数
        exp = math.floor(math.log10(abs(number))) if number != 0 else 0
        exp3 = exp - (exp % 3)
        mantissa = number / (10 ** exp3)
        return f"{mantissa:.{precision}f}e{exp3}"
    else:
        # decimal格式
        if use_scientific and abs(number) > 1e6:
            return f"{number:.{precision}e}"
        else:
            return f"{number:.{precision}f}"


@mcp.tool()
def batch_calculate(calculations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """批量计算
    
    参数:
        calculations: 计算任务列表，每个任务包含:
            - tool: 工具名称
            - parameters: 工具参数
    
    返回:
        计算结果列表
    """
    results = []
    
    for calc in calculations:
        try:
            tool_name = calc.get("tool")
            params = calc.get("parameters", {})
            
            # 这里需要根据工具名称调用相应的函数
            # 这是一个简化实现，实际应用中需要更复杂的路由逻辑
            result = {
                "tool": tool_name,
                "parameters": params,
                "status": "not_implemented",
                "result": None,
                "error": "批量计算功能需要进一步实现"
            }
            results.append(result)
        except Exception as e:
            results.append({
                "tool": calc.get("tool", "unknown"),
                "parameters": calc.get("parameters", {}),
                "status": "error",
                "result": None,
                "error": str(e)
            })
    
    return results


@mcp.tool()
def get_help(tool_name: str = "", detail_level: str = "basic") -> Dict[str, Any]:
    """获取帮助信息
    
    参数:
        tool_name: 工具名称（如果为空，则返回所有工具列表）
        detail_level: 详细程度 (basic/advanced/expert)
    
    返回:
        帮助信息
    """
    # 这里可以返回工具的使用说明
    # 简化实现，返回基本信息
    if not tool_name:
        # 返回所有工具列表
        tools = [
            "add", "subtract", "multiply", "divide",
            "power", "sqrt", "cbrt", "log", "ln", "exp",
            "sin", "cos", "tan", "asin", "acos", "atan",
            "factorial", "combination", "permutation",
            "mean", "median", "mode", "variance", "standard_deviation",
            "evaluate_expression", "simplify_expression",
            "matrix_multiply", "matrix_determinant", "matrix_inverse", "matrix_transpose",
            "differentiate", "integrate", "solve_equation_numeric",
            "angle_convert", "format_number", "get_calculation_history"
        ]
        
        return {
            "available_tools": tools,
            "total_tools": len(tools),
            "detail_level": detail_level
        }
    else:
        # 返回特定工具的帮助信息
        return {
            "tool": tool_name,
            "description": f"工具 '{tool_name}' 的帮助信息",
            "detail_level": detail_level,
            "note": "详细的帮助信息需要进一步实现"
        }

def main() -> None:
    mcp.run(transport="stdio")