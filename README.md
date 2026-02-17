# 数学计算 MCP 服务器

这是一个专门用于数学计算的 (MCP) 服务器。

## 功能特性

### 1. 基础数学运算
- `add(a, b)` - 加法运算
- `subtract(a, b)` - 减法运算
- `multiply(a, b)` - 乘法运算
- `divide(a, b)` - 除法运算

### 2. 高级数学运算
- `power(base, exponent)` - 幂运算
- `sqrt(number)` - 平方根
- `cbrt(number)` - 立方根
- `log(number, base)` - 对数（默认自然对数）
- `ln(number)` - 自然对数
- `exp(number)` - 指数函数 e^x
- `sin(angle)`, `cos(angle)`, `tan(angle)` - 三角函数（角度制）
- `asin(value)`, `acos(value)`, `atan(value)` - 反三角函数（角度制）
- `factorial(n)` - 阶乘
- `combination(n, r)` - 组合数 C(n, r)
- `permutation(n, r)` - 排列数 P(n, r)

### 3. 统计学工具
- `mean(numbers)` - 计算平均值
- `median(numbers)` - 计算中位数
- `mode(numbers)` - 计算众数
- `variance(numbers)` - 计算方差
- `standard_deviation(numbers)` - 计算标准差

### 4. 数学常量资源
可以通过 `math:constant/{name}` 访问：
- `pi` - 圆周率 π
- `e` - 自然常数 e
- `golden_ratio` - 黄金比例 φ
- `sqrt_2` - 根号2 √2

### 5. 提示生成器
- `solve_equation` - 生成解方程的提示
- `prove_theorem` - 生成证明数学定理的提示
- `create_graph` - 生成创建图形的提示

### 6. 单位转换
- `angle_convert(angle, from_unit, to_unit)` - 角度单位转换（degree/radian）

## 使用方法

### 1. 安装依赖
```bash
pip install mcp[cli]
```

### 2. 运行服务器
```bash
python math_server.py
```

### 3. 使用工具示例

在 MCP 客户端中，您可以调用这些工具：

```python
# 基础运算
result = await client.call_tool("add", {"a": 10, "b": 5})

# 高级运算
result = await client.call_tool("power", {"base": 2, "exponent": 10})

# 统计计算
result = await client.call_tool("mean", {"numbers": [1, 2, 3, 4, 5]})

# 获取数学常量
result = await client.call_resource("math:constant/pi", {})
```

## 扩展功能

### 添加新工具
要添加新的数学工具，只需使用 `@mcp.tool()` 装饰器：

```python
@mcp.tool()
def fibonacci(n: int) -> int:
    """计算斐波那契数"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

### 添加新资源
使用 `@mcp.resource()` 装饰器添加新的数学资源：

```python
@mcp.resource("math:formula/{formula_name}")
def get_formula(formula_name: str) -> dict:
    """获取数学公式"""
    # 实现获取公式的逻辑
    pass
```

## 注意事项

1. 所有角度输入都是**角度制**，不是弧度制
2. 除法运算会检查除数是否为零
3. 平方根和对数运算会检查定义域
4. 阶乘计算限制了最大值为170，避免数值溢出
5. 三角函数的值域检查，避免返回无穷大

## 错误处理

服务器对各种边界情况进行了检查，包括：
- 除数为零
- 负数平方根
- 无效的对数参数
- 超出范围的三角函数输入
- 空列表的统计运算

## 测试

运行测试脚本验证功能：
```bash
python test_math_server.py
```