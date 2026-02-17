"""
数学计算 MCP 服务器测试程序

用于测试数学计算服务器的各项功能，包括基础运算、高级数学函数和复杂算式处理。
"""

import sys
import json
import math
import re
from typing import Dict, Any, Optional

# 模拟 MCP 服务器交互
class MockMCPServer:
    def __init__(self):
        self.tools = {}
        self.resources = {}
        self.load_tools()
    
    def load_tools(self):
        """加载所有工具函数"""
        # 基础数学运算工具
        self.tools["add"] = lambda a, b: a + b
        self.tools["subtract"] = lambda a, b: a - b
        self.tools["multiply"] = lambda a, b: a * b
        self.tools["divide"] = lambda a, b: a / b if b != 0 else None
        
        # 高级数学运算工具
        self.tools["power"] = lambda base, exponent: math.pow(base, exponent)
        self.tools["sqrt"] = lambda number: math.sqrt(number) if number >= 0 else None
        self.tools["cbrt"] = lambda number: number ** (1/3)
        self.tools["log"] = lambda number, base=math.e: math.log(number, base) if number > 0 and base > 0 and base != 1 else None
        self.tools["ln"] = lambda number: math.log(number) if number > 0 else None
        self.tools["exp"] = lambda number: math.exp(number)
        self.tools["sin"] = lambda angle: math.sin(math.radians(angle))
        self.tools["cos"] = lambda angle: math.cos(math.radians(angle))
        self.tools["tan"] = lambda angle: math.tan(math.radians(angle))
        self.tools["asin"] = lambda value: math.degrees(math.asin(value)) if abs(value) <= 1 else None
        self.tools["acos"] = lambda value: math.degrees(math.acos(value)) if abs(value) <= 1 else None
        self.tools["atan"] = lambda value: math.degrees(math.atan(value))
        self.tools["factorial"] = lambda n: math.factorial(n) if 0 <= n <= 170 else None
        self.tools["combination"] = lambda n, r: math.comb(n, r) if 0 <= r <= n else None
        self.tools["permutation"] = lambda n, r: math.perm(n, r) if 0 <= r <= n else None
        
        # 统计学工具
        self.tools["mean"] = lambda numbers: sum(numbers) / len(numbers) if numbers else None
        self.tools["median"] = lambda numbers: self._calculate_median(numbers)
        self.tools["mode"] = lambda numbers: self._calculate_mode(numbers)
        self.tools["variance"] = lambda numbers: self._calculate_variance(numbers)
        self.tools["standard_deviation"] = lambda numbers: math.sqrt(self._calculate_variance(numbers))
        
        # 复杂算式处理工具
        self.tools["evaluate_expression"] = self._evaluate_expression
        self.tools["simplify_expression"] = self._simplify_expression
        
        # 单位转换工具
        self.tools["angle_convert"] = self._angle_convert
        
        # 数学常量资源
        self.resources["math:constant/pi"] = {"name": "圆周率", "symbol": "π", "value": math.pi}
        self.resources["math:constant/e"] = {"name": "自然常数", "symbol": "e", "value": math.e}
        self.resources["math:constant/golden_ratio"] = {"name": "黄金比例", "symbol": "φ", "value": (1 + math.sqrt(5)) / 2}
        self.resources["math:constant/sqrt_2"] = {"name": "根号2", "symbol": "√2", "value": math.sqrt(2)}
    
    def _calculate_median(self, numbers):
        """计算中位数"""
        if not numbers:
            return None
        sorted_numbers = sorted(numbers)
        n = len(sorted_numbers)
        if n % 2 == 1:
            return sorted_numbers[n//2]
        else:
            return (sorted_numbers[n//2 - 1] + sorted_numbers[n//2]) / 2
    
    def _calculate_mode(self, numbers):
        """计算众数"""
        if not numbers:
            return None
        frequency = {}
        for num in numbers:
            frequency[num] = frequency.get(num, 0) + 1
        max_freq = max(frequency.values())
        return [num for num, freq in frequency.items() if freq == max_freq]
    
    def _calculate_variance(self, numbers):
        """计算方差"""
        if len(numbers) < 2:
            return None
        avg = sum(numbers) / len(numbers)
        return sum((x - avg) ** 2 for x in numbers) / len(numbers)
    
    def _evaluate_expression(self, expression: str, variables: Optional[Dict[str, float]] = None) -> float:
        """
        计算复杂的数学表达式
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
    
    def _simplify_expression(self, expression: str) -> str:
        """
        简化数学表达式（基础实现）
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
    
    def _angle_convert(self, angle: float, from_unit: str, to_unit: str) -> float:
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
    
    def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """调用工具函数"""
        if tool_name not in self.tools:
            raise ValueError(f"未知的工具: {tool_name}")
        
        tool_func = self.tools[tool_name]
        try:
            return tool_func(**args)
        except Exception as e:
            return f"错误: {str(e)}"
    
    def call_resource(self, resource_name: str, args: Dict[str, Any]) -> Any:
        """调用资源"""
        if resource_name not in self.resources:
            raise ValueError(f"未知的资源: {resource_name}")
        
        return self.resources[resource_name]


def test_basic_operations():
    """测试基础运算"""
    print("=== 测试基础运算 ===")
    server = MockMCPServer()
    
    # 测试加法
    result = server.call_tool("add", {"a": 2, "b": 3})
    print(f"2 + 3 = {result}")
    assert result == 5, "加法测试失败"
    
    # 测试减法
    result = server.call_tool("subtract", {"a": 5, "b": 3})
    print(f"5 - 3 = {result}")
    assert result == 2, "减法测试失败"
    
    # 测试乘法
    result = server.call_tool("multiply", {"a": 2, "b": 3})
    print(f"2 * 3 = {result}")
    assert result == 6, "乘法测试失败"
    
    # 测试除法
    result = server.call_tool("divide", {"a": 6, "b": 3})
    print(f"6 / 3 = {result}")
    assert result == 2, "除法测试失败"
    
    # 测试除以零
    result = server.call_tool("divide", {"a": 6, "b": 0})
    print(f"6 / 0 = {result}")
    assert result is None, "除以零错误处理失败"
    
    print("基础运算测试通过\n")


def test_advanced_operations():
    """测试高级运算"""
    print("=== 测试高级运算 ===")
    server = MockMCPServer()
    
    # 测试幂运算
    result = server.call_tool("power", {"base": 2, "exponent": 3})
    print(f"2^3 = {result}")
    assert result == 8, "幂运算测试失败"
    
    # 测试平方根
    result = server.call_tool("sqrt", {"number": 16})
    print(f"√16 = {result}")
    assert result == 4, "平方根测试失败"
    
    # 测试负数平方根
    result = server.call_tool("sqrt", {"number": -1})
    print(f"√(-1) = {result}")
    assert result is None, "负数平方根错误处理失败"
    
    # 测试对数
    result = server.call_tool("log", {"number": 100, "base": 10})
    print(f"log10(100) = {result}")
    assert abs(result - 2) < 0.0001, "对数测试失败"
    
    # 测试自然对数
    result = server.call_tool("ln", {"number": math.e})
    print(f"ln(e) = {result}")
    assert abs(result - 1) < 0.0001, "自然对数测试失败"
    
    # 测试指数函数
    result = server.call_tool("exp", {"number": 1})
    print(f"e^1 = {result}")
    assert abs(result - math.e) < 0.0001, "指数函数测试失败"
    
    # 测试三角函数
    result = server.call_tool("sin", {"angle": 30})
    print(f"sin(30°) = {result}")
    assert abs(result - 0.5) < 0.0001, "正弦函数测试失败"
    
    result = server.call_tool("cos", {"angle": 60})
    print(f"cos(60°) = {result}")
    assert abs(result - 0.5) < 0.0001, "余弦函数测试失败"
    
    # 测试反三角函数
    result = server.call_tool("asin", {"value": 0.5})
    print(f"asin(0.5) = {result}")
    assert abs(result - 30) < 0.0001, "反正弦函数测试失败"
    
    result = server.call_tool("acos", {"value": 0.5})
    print(f"acos(0.5) = {result}")
    assert abs(result - 60) < 0.0001, "反余弦函数测试失败"
    
    print("高级运算测试通过\n")


def test_statistics():
    """测试统计学工具"""
    print("=== 测试统计学工具 ===")
    server = MockMCPServer()
    
    # 测试平均值
    result = server.call_tool("mean", {"numbers": [1, 2, 3, 4, 5]})
    print(f"平均值 [1, 2, 3, 4, 5] = {result}")
    assert result == 3, "平均值测试失败"
    
    # 测试中位数
    result = server.call_tool("median", {"numbers": [1, 2, 3, 4, 5]})
    print(f"中位数 [1, 2, 3, 4, 5] = {result}")
    assert result == 3, "中位数测试失败"
    
    # 测试众数
    result = server.call_tool("mode", {"numbers": [1, 2, 2, 3, 4]})
    print(f"众数 [1, 2, 2, 3, 4] = {result}")
    assert result == [2], "众数测试失败"
    
    # 测试方差
    result = server.call_tool("variance", {"numbers": [1, 2, 3, 4, 5]})
    print(f"方差 [1, 2, 3, 4, 5] = {result}")
    assert abs(result - 2) < 0.0001, "方差测试失败"
    
    # 测试标准差
    result = server.call_tool("standard_deviation", {"numbers": [1, 2, 3, 4, 5]})
    print(f"标准差 [1, 2, 3, 4, 5] = {result}")
    assert abs(result - math.sqrt(2)) < 0.0001, "标准差测试失败"
    
    print("统计学工具测试通过\n")


def test_complex_expressions():
    """测试复杂算式处理"""
    print("=== 测试复杂算式处理 ===")
    server = MockMCPServer()
    
    # 测试基本表达式
    result = server.call_tool("evaluate_expression", {"expression": "2 * (3 + 4)"})
    print(f"2 * (3 + 4) = {result}")
    assert result == 14, "基本表达式测试失败"
    
    # 测试包含变量的表达式
    result = server.call_tool("evaluate_expression", {"expression": "x + y", "variables": {"x": 2, "y": 3}})
    print(f"x + y (x=2, y=3) = {result}")
    assert result == 5, "变量表达式测试失败"
    
    # 测试包含三角函数的表达式
    result = server.call_tool("evaluate_expression", {"expression": "sin(30) + cos(60)"})
    print(f"sin(30) + cos(60) = {result}")
    assert abs(result - 1) < 0.0001, "三角函数表达式测试失败"
    
    # 测试包含对数的表达式
    result = server.call_tool("evaluate_expression", {"expression": "log(100, 10)"})
    print(f"log(100, 10) = {result}")
    assert abs(result - 2) < 0.0001, "对数表达式测试失败"
    
    # 测试包含常量的表达式
    result = server.call_tool("evaluate_expression", {"expression": "pi * 2"})
    print(f"π * 2 = {result}")
    assert abs(result - 2 * math.pi) < 0.0001, "常量表达式测试失败"
    
    # 测试表达式简化
    result = server.call_tool("simplify_expression", {"expression": "x + 0"})
    print(f"'x + 0' 简化为: {result}")
    assert result == "x", "表达式简化测试失败"
    
    result = server.call_tool("simplify_expression", {"expression": "x * 1"})
    print(f"'x * 1' 简化为: {result}")
    assert result == "x", "表达式简化测试失败"
    
    print("复杂算式处理测试通过\n")


def test_math_constants():
    """测试数学常量资源"""
    print("=== 测试数学常量资源 ===")
    server = MockMCPServer()
    
    # 测试圆周率
    result = server.call_resource("math:constant/pi", {})
    print(f"π = {result['value']}")
    assert abs(result["value"] - math.pi) < 0.0001, "圆周率测试失败"
    
    # 测试自然常数
    result = server.call_resource("math:constant/e", {})
    print(f"e = {result['value']}")
    assert abs(result["value"] - math.e) < 0.0001, "自然常数测试失败"
    
    # 测试黄金比例
    result = server.call_resource("math:constant/golden_ratio", {})
    print(f"φ = {result['value']}")
    assert abs(result["value"] - (1 + math.sqrt(5)) / 2) < 0.0001, "黄金比例测试失败"
    
    # 测试根号2
    result = server.call_resource("math:constant/sqrt_2", {})
    print(f"√2 = {result['value']}")
    assert abs(result["value"] - math.sqrt(2)) < 0.0001, "根号2测试失败"
    
    print("数学常量资源测试通过\n")


def test_angle_conversion():
    """测试角度单位转换"""
    print("=== 测试角度单位转换 ===")
    server = MockMCPServer()
    
    # 测试角度转弧度
    result = server.call_tool("angle_convert", {"angle": 180, "from_unit": "degree", "to_unit": "radian"})
    print(f"180° = {result} 弧度")
    assert abs(result - math.pi) < 0.0001, "角度转弧度测试失败"
    
    # 测试弧度转角度
    result = server.call_tool("angle_convert", {"angle": math.pi, "from_unit": "radian", "to_unit": "degree"})
    print(f"{math.pi} 弧度 = {result}°")
    assert abs(result - 180) < 0.0001, "弧度转角度测试失败"
    
    # 测试无效单位
    result = server.call_tool("angle_convert", {"angle": 180, "from_unit": "invalid", "to_unit": "radian"})
    print(f"180 invalid = {result}")
    assert "错误" in str(result), "无效单位错误处理失败"
    
    print("角度单位转换测试通过\n")


def main():
    """主测试函数"""
    print("开始测试数学计算 MCP 服务器...\n")
    
    try:
        test_basic_operations()
        test_advanced_operations()
        test_statistics()
        test_complex_expressions()
        test_math_constants()
        test_angle_conversion()
        
        print("所有测试通过！")
        return 0
    except AssertionError as e:
        print(f"\n测试失败: {str(e)}")
        return 1
    except Exception as e:
        print(f"\n测试过程中发生错误: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
