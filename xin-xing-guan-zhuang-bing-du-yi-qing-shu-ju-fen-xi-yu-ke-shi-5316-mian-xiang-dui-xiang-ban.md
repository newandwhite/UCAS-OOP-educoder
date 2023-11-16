### 第1关：传染病模型SIR模型-前向欧拉法求解

#### 任务描述 {#任务描述}

新型冠状病毒感染的肺炎疫情引发全球广泛关注。2020年3月11日，世界卫生组织总干事谭德塞宣布，根据评估，世卫组织认为当前新冠肺炎疫情可被称为全球大流行（pandemic）。我们可以利用传染病模型，对疾病的发展进行简单预测，这对疫情发展趋势分析具有一定的参考价值。

本关的目标就是让学习者利用 Python 编程实现经典传染病模型——SIR模型（Susceptible Infected Recovered Model）对疫情趋势进行分析。本关中，我们采用前向欧拉法完善SIR模型，对疫情发展进行模拟。

#### 相关知识 {#相关知识}

##### SIR模型 {#sir模型}

> SIR模型（Susceptible Infected Recovered Model）是一种传播模型，是疾病及信息传播过程的抽象描述。是传染病模型中最经典的模型。此模型能够较为粗略地展示出一种传染病从发病到结束的过程，其核心在于常微分方程。

* S 类：易感者（Susceptible）
* I 类：感染者（Infective）
* R 类：移出者（Removal）
  ![](/pic/SIRmodel.png)

常微分方程组表示如下：
$$
\left\{  
     \begin{array}{lr}  
     S' = -\beta SI &  \\  
     I' = \beta SI-\gamma I & \\  
     R' = \gamma I &    
     \end{array}  
\right.  
$$

其中S′,I′,R′分别表示S,I,R关于时间t的导数，下面将其组合为向量形式并用函数f表示，得:$$(S',I',R')=f(S,I,R)=(−\beta SI,\beta SI−\gamma I,\gamma I)$$ 其中:β为感染系数，代表易感人群与传染源接触被感染的概率，即平均传染率；γ为隔离\(恢复\)系数，一般对其倒数1/γ更感兴趣，代表了平均感染时间；S,I定义见上，S\(0\)为初始易感人数，I\(0\)为初始感染人数。

##### 前向欧拉法 {#前向欧拉法}

> 欧拉方法，命名自它的发明者莱昂哈德·欧拉，是一种一阶数值方法，用以对给定初值的常微分方程（即初值问题）求解。它是一种解决数值常微分方程的最基本的一类显型方法（Explicit method）。  
> 前向欧拉法形式如下：  
> ![](/pic/forwardEuler.png)
>
> 代码示例如下:

```py
    def forwardEuler(f, y0, x):
        y = [y0] * len(x)
        for k in range(len(x)-1):
            h = x[k+1]-x[k]
            y[k+1] = y[k] + h * f(x[k], y[k])
        return y  # y = [y0, y1, ..., yn]
```

#### 编程要求 {#编程要求}

利用前向欧拉法完善SIR模型，实现SIR类，其属性

* `beta`
  ：对应SIR模型中的β
* `gamma`
  ：对应SIR模型中的γ
* `y0`
  ：对应SIR模型中的\(S, I, R\)人数

并编写方法

* `f(self, y)`
  : 实现SIR模型，其中参数y是当前状态\[S, I, R\]，该函数需返回\[S',I',R'\]
* `solve(self, x)`
  : 实现前向欧拉法，其中参数x是列表\[0, .., t-1\]，该函数返回t天内每天感染人数的列表 \(列表大小为天数t\)。（注意：本场景中应用前向欧拉法时，forwardEuler\(\)函数第5行f\(x\[k\],y\[k\]\) 中x\[k\]对f函数的计算没有影响）

#### 任务提示 {#任务提示}

**测试输入**：

```py
    N     = 1e8                  # 武汉总人数：1000万人  
    gamma = 1/25            # 假设肺炎平均25天治愈（15天潜伏+10天治疗）   
    y0    = [N-1, 1, 0]        # 初始发病1人，其他人员正常 [S0, I0, R0]  
    t = range(0,5,1)          # 模拟5天的发展情况，单位时间为1天  
    beta = 1.0/N               # 平均传染率
    simulation = SIR(beta=beta, gamma=gamma, y0=y0)   
    y = simulation.solve(t)
```

**预期输出**：

```py
    [1.0000, 1.9600, 3.8416, 7.5295, 14.7579]
```

---

开始你的任务吧，祝你成功！

step\_1\_framework.py

```py
# import ode_7 as ode
import numpy as np

def myprint_list(l):
    if type(l)!=list:
        print('Error: the result is not a list')
    else:
        print('[', end='')
        for i in range(len(y_f)-1):
            print("%.4f, "%y_f[i][1], end='')
        print("%.4f]"%y_f[len(y_f)-1][1])#上面和这里的printf要改，每一项是一个数组，只保留第二项

class SIR():
    def __init__(self, beta, gamma, y0):
        self.beta = beta; self.gamma = gamma; self.y0 = y0  # 参数属性

    def f(self, y): # y为当前状态，即列表[S,I,R]
        # ------------------begin-----------------------
        diff = np.zeros(3)
        s,i,r = y
        diff[0] = - self.beta * s * i
        diff[1] = self.beta * s * i - self.gamma * i
        diff[2] = self.gamma * i
        return diff
        # -------------------end------------------------

    def solve(self, t):
        # ------------------begin-----------------------
        y = [y0] * len(t);
        for k in range(len(t)-1):
            h = t[k+1]-t[k]
            y[k+1] = y[k] + h * self.f(y[k])
        return y 
        # -------------------end------------------------


N     = 1e8             # 武汉总人数：1000万人
gamma = 1/25            # 假设肺炎平均25天治愈（15天潜伏+10天治疗） 
y0    = [N-1, 1, 0]     # 初始发病1人，其他人员正常, 即[S0, I0, R0]
beta = 1.0/N            # 感染系数
simulation = SIR(beta=beta, gamma=gamma, y0=y0)
for T in [5, 10, 20]:
    t   = range(0, T, 1) # 模拟T天的发展情况，单位时间为1天
    y_f = simulation.solve(t)
    myprint_list(y_f)
```

---

### 第2关：传染病模型SIR模型-后向欧拉法求解

#### 任务描述 {#任务描述}

新型冠状病毒感染的肺炎疫情引发全球广泛关注。2020年3月11日，世界卫生组织总干事谭德塞宣布，根据评估，世卫组织认为当前新冠肺炎疫情可被称为全球大流行（pandemic）。我们可以利用传染病模型，对疾病的发展进行简单预测，这对疫情发展趋势分析具有一定的参考价值。

本关的目标就是让学习者利用 Python 编程实现经典传染病模型——SIR模型（Susceptible Infected Recovered Model）对疫情趋势进行分析。本关中，我们采用后向欧拉法完善SIR模型，对疫情发展进行模拟。

#### 相关知识 {#相关知识}

##### SIR模型 {#sir模型}

> SIR模型（Susceptible Infected Recovered Model）是一种传播模型，是疾病及信息传播过程的抽象描述。是传染病模型中最经典的模型。此模型能够较为粗略地展示出一种传染病从发病到结束的过程，其核心在于常微分方程。

* S 类：易感者（Susceptible）
* I 类：感染者（Infective）
* R 类：移出者（Removal） 
  ![](/pic/SIRmodel.png)

常微分方程组表示如下：
$$
\left\{  
     \begin{array}{lr}  
     S' = -\beta SI &  \\  
     I' = \beta SI-\gamma I & \\  
     R' = \gamma I &    
     \end{array}  
\right.  
$$

其中S′,I′,R′分别表示S,I,R关于时间t的导数，下面将其组合为向量形式并用函数f表示，得:$$(S',I',R')=f(S,I,R)=(−\beta SI,\beta SI−\gamma I,\gamma I)$$ 其中:β为感染系数，代表易感人群与传染源接触被感染的概率，即平均传染率；γ为隔离\(恢复\)系数，一般对其倒数1/γ更感兴趣，代表了平均感染时间；S,I定义见上，S\(0\)为初始易感人数，I\(0\)为初始感染人数。

##### 后向欧拉法 {#后向欧拉法}

> 欧拉方法，命名自它的发明者莱昂哈德·欧拉，是一种一阶数值方法，用以对给定初值的常微分方程（即初值问题）求解。它是一种解决数值常微分方程的最基本的一类显型方法（Explicit method）。  
> 后向欧拉法形式如下：  
> ![](/pic/backwardEuler.png)

两步迭代的后向欧拉法，可通过如下方式实现：yp​=yn​+hf\(yn​,tn​\)yn+1​=yn​+hf\(yp​,tn+1​\)

#### 编程要求 {#编程要求}

利用后向欧拉法（两步迭代）完善SIR模型，实现SIR类，其属性

* `beta`
  ：对应SIR模型中的β
* `gamma`
  ：对应SIR模型中的γ
* `y0`
  ：对应SIR模型中的\(S, I, R\)人数

并编写方法

* `f(self,y)`: 实现SIR模型，其中参数y是当前状态\[S, I, R\]，该函数需返回\[S',I',R'\]
* `solve(self,x)`: 实现后向欧拉法（两步迭代），其中参数x是列表\[0, .., t-1\]，该函数返回t天内每天感染人数的列表 \(列表大小为天数t\)。（注意：本场景中应用后向欧拉法时，`f(yn+1​,tn+1​)`中`tn+1​`对f函数的计算没有影响）

#### 任务提示 {#任务提示}

**测试输入**：

```py
    N     = 1e8                # 武汉总人数：1000万人  
    gamma = 1/25          # 假设肺炎平均25天治愈（15天潜伏+10天治疗）   
    y0    = (N-1, 1, 0)     # 初始发病1人，其他人员正常 (S0, I0, R0)  
    t = range(0, 5, 1)     # 模拟5天的发展情况，单位时间为1天  
    beta = 1.0/N           # 平均传染率
    simulation = SIR(beta=beta, gamma=gamma, y0=y0)   
    y = simulation.solve(t)
```

**预期输出**：

`(1.0, 2.8815999512000006, 8.303618007564571, 23.927702212048267, 68.95003808706701, 198.68619205109314, 572.5321567912988, 1649.7922686666125, 4753.905269148689, 13697.723116003775, 39461.97549382348)`

---

开始你的任务吧，祝你成功！

step\_2\_framework.py

```py
import numpy as np

def myprint_list(l):
    if type(l)!=list:
        print('Error: the result is not a list')
    else:
        print('[', end='')
        for i in range(len(y_f)-1):
            print("%.4f, "%y_f[i][1], end='')
        print("%.4f]"%y_f[len(y_f)-1][1])

class SIR():
    def __init__(self, beta, gamma, y0):
        self.beta = beta; self.gamma = gamma; self.y0 = y0  # 参数属性

    def f(self, y):  # y为当前状态，列表[S,I,R]
        # ------------------begin-----------------------
        diff = np.zeros(3)
        s,i,r = y
        diff[0] = - self.beta * s * i
        diff[1] = self.beta * s * i - self.gamma * i
        diff[2] = self.gamma * i
        return diff

        # -------------------end------------------------

    def solve(self, x):
        # ------------------begin-----------------------
        y = [y0] * len(t);
        for k in range(len(t)-1):
            h = t[k+1]-t[k]
            yp = y[k] + h * self.f(y[k])
            y[k+1] = y[k] + h * self.f(yp)
        return y 
        # -------------------end------------------------


N     = 1e8             # 武汉总人数：1000万人
gamma = 1/25            # 假设肺炎平均25天治愈（15天潜伏+10天治疗） 
y0    = [N-1, 1, 0]     # 初始发病1人，其他人员正常, 即[S0, I0, R0]
beta = 1.0/N            # 平均传染率倒数
simulation = SIR(beta=beta, gamma=gamma, y0=y0)
for T in [5, 10, 20]:
    t   = range(0, T, 1) # 模拟T天的发展情况，单位时间为1天
    y_f = simulation.solve(t)
    myprint_list(y_f)
```

---

### 第3关：传染病模型SIR模型-改进欧拉法\(梯形法\)求解

#### 任务描述 {#任务描述}

新型冠状病毒感染的肺炎疫情引发全球广泛关注。2020年3月11日，世界卫生组织总干事谭德塞宣布，根据评估，世卫组织认为当前新冠肺炎疫情可被称为全球大流行（pandemic）。我们可以利用传染病模型，对疾病的发展进行简单预测，这对疫情发展趋势分析具有一定的参考价值。

本关的目标就是让学习者利用 Python 编程实现经典传染病模型——SIR模型（Susceptible Infected Recovered Model）对疫情趋势进行分析。本关中，我们采用改进的欧拉法（梯形法，两步）完善SIR模型，对疫情发展进行模拟。

#### 相关知识 {#相关知识}

##### SIR模型 {#sir模型}

> SIR模型（Susceptible Infected Recovered Model）是一种传播模型，是疾病及信息传播过程的抽象描述。是传染病模型中最经典的模型。此模型能够较为粗略地展示出一种传染病从发病到结束的过程，其核心在于常微分方程。

* S 类：易感者（Susceptible）
* I 类：感染者（Infective）
* R 类：移出者（Removal）

![](/pic/SIRmodel.png)

常微分方程组表示如下：
$$
\left\{  
     \begin{array}{lr}  
     S' = -\beta SI &  \\  
     I' = \beta SI-\gamma I & \\  
     R' = \gamma I &    
     \end{array}  
\right.  
$$

其中S′,I′,R′分别表示S,I,R关于时间t的导数，下面将其组合为向量形式并用函数f表示，得:$$(S',I',R')=f(S,I,R)=(−\beta SI,\beta SI−\gamma I,\gamma I)$$ 其中:β为感染系数，代表易感人群与传染源接触被感染的概率，即平均传染率；γ为隔离\(恢复\)系数，一般对其倒数1/γ更感兴趣，代表了平均感染时间；S,I定义见上，S\(0\)为初始易感人数，I\(0\)为初始感染人数。

##### 改进的欧拉法（梯形法，两步） {#改进的欧拉法（梯形法，两步）}

根据欧拉法和梯形公式，我们可以得到改进的欧拉公式，改进的欧拉法形式如下：

![](/pic/improvedEuler.png)

代码示例如下:

```py
    def trapezoidalEuler(f, x, y0):
        y = [y0] * len(x)
        for k in range(len(x)-1):
            h = x[k+1]-x[k]
            y_p = y[k] + h * f(x[k], y[k])
            y_c = y[k] + h * f(x[k+1], y_p)
            y[k+1] = 1/2 * (y_p + y_c)
        return y  # y = [y0, y1, ..., yn]
```

#### 编程要求 {#编程要求}

利用改进的欧拉法\(梯形法，两步迭代）完善SIR模型，实现SIR类，其属性

* `beta`
  ：对应SIR模型中的β
* `gamma`
  ：对应SIR模型中的γ
* `y0`
  ：对应SIR模型中的\(S, I, R\)人数

并编写方法

* `f(self,y)`: 实现SIR模型，其中参数y是当前状态\[S, I, R\]，该函数需返回\[S',I',R'\]

* `solve(self,x)`: 实现改进的欧拉法\(梯形法，两步迭代），其中参数x是列表\[0, .., t-1\]，该函数返回t天内每天感染人数的列表 \(列表大小为天数t\)。（注意：本场景中应用改进的欧拉法时，`f(x[k],y[k])`和`f(x[k+1],y_p)`中`x[k],x[k+1]`对f函数的计算没有影响）

#### 任务提示 {#任务提示}

**测试输入**：

```
    N     = 1e8             # 武汉总人数：1000万人  
    gamma = 1/25            # 假设肺炎平均25天治愈（15天潜伏+10天治疗）   
    y0    = [N-1, 1, 0]     # 初始发病1人，其他人员正常 [S0, I0, R0]  
    t =   range(0, 5, 1) # 模拟5天的发展情况，单位时间为1天  
    beta = 1.0/N            # 平均传染率
    simulation = SIR(beta=beta, gamma=gamma, y0=y0)   
    y = simulation.solve(t)
```

**预期输出**：

```py
    [1.0000, 2.4208, 5.8603, 14.1865, 34.3428]
```

---

开始你的任务吧，祝你成功！

framework.py

```py
import numpy as np

def myprint_list(l):
    if type(l)!=list:
        print('Error: the result is not a list')
    else:
        print('[', end='')
        for i in range(len(y_f)-1):
            print("%.4f, "%y_f[i][1], end='')
        print("%.4f]"%y_f[len(y_f)-1][1])

class SIR():
    def __init__(self, beta, gamma, y0):
        self.beta = beta; self.gamma = gamma; self.y0 = y0  # 参数属性

    def f(self, y): #  y为当前状态，列表[S,I,R]
        # S = y[0]
        # I = y[1]
        # R = y[2]
        # return [-self.beta * S * I, self.beta * S * I - self.gamma * I, self.gamma * I]
        diff = np.zeros(3, dtype=np.float64)
        s,i,r = y
        diff[0] = - self.beta * s * i
        diff[1] = self.beta * s * i - self.gamma * i
        diff[2] = self.gamma * i
        return diff

    def solve(self, x):
        # ------------------begin-----------------------
        y = [y0] * len(t);
        for k in range(len(t)-1):
            h = t[k+1]-t[k]
            y_p = np.array(y[k] + h * np.array(self.f(y[k])))
            y_c = np.array(y[k] + h * np.array(self.f(y_p)))
            y[k+1] = (y_p + y_c)/2
        return y

        # -------------------end------------------------


N     = 1e8             # 武汉总人数：1000万人
gamma = 1/25            # 假设肺炎平均25天治愈（15天潜伏+10天治疗） 
y0    = [N-1, 1, 0]     # 初始发病1人，其他人员正常 [S0, I0, R0]
t     = range(0, 11, 1) # 模拟10天的发展情况，单位时间为1天
beta = 1.0/N            # 平均传染率

simulation = SIR(beta=beta, gamma=gamma, y0=y0)
for T in [5, 10, 20]:
    t   = range(0, T, 1) # 模拟T天的发展情况，单位时间为1天
    y_f = simulation.solve(t)
    myprint_list(y_f)
```

---

### 第4关：引入疫情防控措施-隔离机制

#### 任务描述 {#任务描述}

为有效防止疫情进一步扩散和蔓延，各地纷纷出台各种举措加强疫情防控工作。为了更准确的分析数据，预测疫情发展，本关将在第3关的基础上引入隔离机制，模拟疫情防控。

#### 相关知识 {#相关知识}

##### SIR模型 {#sir模型}

> SIR模型（Susceptible Infected Recovered Model）是一种传播模型，是疾病及信息传播过程的抽象描述。是传染病模型中最经典的模型。此模型能够较为粗略地展示出一种传染病从发病到结束的过程，其核心在于常微分方程。

* S 类：易感者（Susceptible）
* I 类：感染者（Infective）
* R 类：移出者（Removal） 
  ![](/pic/SIRmodel.png)

常微分方程组表示如下：
$$
\left\{  
     \begin{array}{lr}  
     S' = -\beta SI &  \\  
     I' = \beta SI-\gamma I & \\  
     R' = \gamma I &    
     \end{array}  
\right.  
$$

其中S′,I′,R′分别表示S,I,R关于时间t的导数，下面将其组合为向量形式并用函数f表示，得:$$(S',I',R')=f(S,I,R)=(−\beta SI,\beta SI−\gamma I,\gamma I)$$ 其中:β为感染系数，代表易感人群与传染源接触被感染的概率，即平均传染率；γ为隔离\(恢复\)系数，一般对其倒数1/γ更感兴趣，代表了平均感染时间；S,I定义见上，S\(0\)为初始易感人数，I\(0\)为初始感染人数。

##### 改进的欧拉法（梯形法，两步） {#改进的欧拉法（梯形法，两步）}

根据欧拉法和梯形公式，我们可以得到改进的欧拉公式，改进的欧拉法形式如下：

![](/pic/improvedEuler.png)

代码示例如下:

```py
    def trapezoidalEuler(f, x, y0):
        y = [y0] * len(x)
        for k in range(len(x)-1):
            h = x[k+1]-x[k]
            y_p = y[k] + h * f(x[k], y[k])
            y_c = y[k] + h * f(x[k+1], y_p)
            y[k+1] = 1/2 * (y_p + y_c)
        return y  # y = [y0, y1, ..., yn]
```

#### 编程要求 {#编程要求}

利用改进的欧拉法\(梯形法，两步迭代）完善SIR模型，并引入隔离机制，实现SIR类，其属性

* `beta`
  ：对应SIR模型中的β
* `gamma`
  ：对应SIR模型中的γ
* `y0`
  ：对应SIR模型中的\(S, I, R\)人数

并编写方法

* `solve_with_quarantine(self,x)`: 实现改进的欧拉法\(梯形法，两步迭代），并引入不同的隔离机制（见下），其中参数x是列表\[0, .., t-1\]，该函数返回t天内每天感染人数的列表 \(列表大小为天数t\)。（注意：本场景中应用改进的欧拉法时，`f(x[k],y[k])`和`f(x[k+1],y_p)`中`x[k],x[k+1]`对f函数的计算没有影响）

**不同的隔离机制**

①未实施隔离： gamma = 1/25，假设肺炎平均25天治愈（15天潜伏+10天治疗）  
②隔离确诊患者：gamma = 1/15，按最长15天发病确诊后被隔离

③隔离疑似人员：gamma = 1/3，平均3天被隔离

#### 任务提示 {#任务提示}

**测试输入**：

```py
    N     = 1e8             # 武汉总人数：1000万人
    gamma = 1/25            # 假设肺炎平均25天治愈（15天潜伏+10天治疗）
    y0    = [N-1, 1, 0]     # 初始发病1人，其他人员正常 [S0, I0, R0]
    beta = 1.0/N            # 平均传染率
    t   = range(0, 5, 1) # 模拟5天的发展情况，单位时间为1天
    print("———————————— 无隔离下感染情况：————————————")
    simulation = SIR(beta=beta, gamma=gamma, y0=y0)
    y_f = simulation.solve_with_quarantine(t)
    myprint_list(y_f)
    print("———————————— 隔离确诊患者：————————————")
    gamma1 = 1/15            # 隔离确诊患者：按最长15天发病确诊后被隔离
    y_f = simulation.solve_with_quarantine(t, gamma1)
    myprint_list(y_f)
    print("———————————— 隔离疑似人员：————————————")
    gamma1 = 1/3            # 隔离疑似人员：按平均3天被隔离
    y_f = simulation.solve_with_quarantine(t, gamma1)
    myprint_list(y_f)
```

**预期输出**：

```py
    ———————————— 无隔离下感染情况：————————————
    [1.0000, 2.4208, 5.8603, 14.1865, 34.3428]
    ———————————— 隔离确诊患者：————————————
    [1.0000, 2.3689, 5.6116, 13.2933, 31.4904]
    ———————————— 隔离疑似人员：————————————
    [1.0000, 1.8889, 3.5679, 6.7394, 12.7299]
```

---

开始你的任务吧，祝你成功！

```py
import numpy as np

def myprint_list(l):
    if type(l)!=list:
        print('Error: the result is not a list')
    else:
        print('[', end='')
        for i in range(len(y_f)-1):
            print("%.4f, "%y_f[i][1], end='')
        print("%.4f]"%y_f[len(y_f)-1][1])

class SIR():
    def __init__(self, beta, gamma, y0):
        self.beta = beta; self.gamma = gamma; self.y0 = y0  # 参数属性

    def f(self, y, gamma): #  y为当前状态，列表[S,I,R]
        S = y[0]
        I = y[1]
        R = y[2]
        return [-self.beta * S * I, self.beta * S * I - gamma * I, gamma * I]

    def solve_with_quarantine(self, x, gamma = 1/25):
        # ------------------begin-----------------------
        y = [y0] * len(t);
        for k in range(len(t)-1):
            h = t[k+1]-t[k]
            y_p = np.array(y[k] + h * np.array(self.f(y[k], gamma)))
            y_c = np.array(y[k] + h * np.array(self.f(y_p, gamma)))
            y[k+1] = (y_p + y_c)/2
        return y
        # -------------------end------------------------

N     = 1e8             # 武汉总人数：1000万人
gamma = 1/25            # 假设肺炎平均25天治愈（15天潜伏+10天治疗）
y0    = [N-1, 1, 0]     # 初始发病1人，其他人员正常 [S0, I0, R0]
beta = 1.0/N            # 平均传染率

for T in [5, 10, 20]:
    t   = range(0, T, 1) # 模拟T天的发展情况，单位时间为1天
    print("———————————— 无隔离下感染情况：————————————")
    simulation = SIR(beta=beta, gamma=gamma, y0=y0)
    y_f = simulation.solve_with_quarantine(t)
    myprint_list(y_f)
    print("———————————— 隔离确诊患者：————————————")
    gamma1 = 1/15            # 隔离确诊患者：按最长15天发病确诊后被隔离
    y_f = simulation.solve_with_quarantine(t, gamma1)
    myprint_list(y_f)
    print("———————————— 隔离疑似人员：————————————")
    gamma1 = 1/3            # 隔离疑似人员：按平均3天被隔离
    y_f = simulation.solve_with_quarantine(t, gamma1)
    myprint_list(y_f)
```

---

### 第5关：引入疫情防控措施-出行控制机制

#### 任务描述 {#任务描述}

为有效防止疫情进一步扩散和蔓延，各地纷纷出台各种举措加强疫情防控工作。为了更准确的分析数据，预测疫情发展，本关将在第三关的基础上引入出行控制机制，模拟疫情防控。

#### 相关知识 {#相关知识}

##### SIR模型 {#sir模型}

> SIR模型（Susceptible Infected Recovered Model）是一种传播模型，是疾病及信息传播过程的抽象描述。是传染病模型中最经典的模型。此模型能够较为粗略地展示出一种传染病从发病到结束的过程，其核心在于常微分方程。

* S 类：易感者（Susceptible）
* I 类：感染者（Infective）
* R 类：移出者（Removal） 
  ![](/pic/SIRmodel.png)

常微分方程组表示如下：
$$
\left\{  
     \begin{array}{lr}  
     S' = -\beta SI &  \\  
     I' = \beta SI-\gamma I & \\  
     R' = \gamma I &    
     \end{array}  
\right.  
$$

其中S′,I′,R′分别表示S,I,R关于时间t的导数，下面将其组合为向量形式并用函数f表示，得:$$(S',I',R')=f(S,I,R)=(−\beta SI,\beta SI−\gamma I,\gamma I)$$ 其中:β为感染系数，代表易感人群与传染源接触被感染的概率，即平均传染率；γ为隔离\(恢复\)系数，一般对其倒数1/γ更感兴趣，代表了平均感染时间；S,I定义见上，S\(0\)为初始易感人数，I\(0\)为初始感染人数。

##### 改进的欧拉法（梯形法，两步） {#改进的欧拉法（梯形法，两步）}

根据欧拉法和梯形公式，我们可以得到改进的欧拉公式，改进的欧拉法形式如下：

![](/pic/improvedEuler.png)

代码示例如下:

```py
    def trapezoidalEuler(f, x, y0):
        y = [y0] * len(x)
        for k in range(len(x)-1):
            h = x[k+1]-x[k]
            y_p = y[k] + h * f(x[k], y[k])
            y_c = y[k] + h * f(x[k+1], y_p)
            y[k+1] = 1/2 * (y_p + y_c)
        return y  # y = [y0, y1, ..., yn]
```

#### 编程要求 {#编程要求}

利用改进的欧拉法\(梯形法，两步迭代）完善SIR模型，并引入出行控制机制，实现SIR类，其属性

* `beta`
  ：对应SIR模型中的β
* `gamma`
  ：对应SIR模型中的γ
* `y0`
  ：对应SIR模型中的\(S, I, R\)人数

并编写方法

* `solve_with_control(self,x)`: 实现改进的欧拉法\(梯形法，两步迭代），并引入出行控制机制（见下），其中参数x是列表\[0, .., t-1\]，该函数返回t天内每天感染人数的列表 \(列表大小为天数t\)。（注意：本场景中应用改进的欧拉法时，`f(x[k],y[k])`和`f(x[k+1],y_p)`中`x[k],x[k+1]`对f函数的计算没有影响）

**出行控制机制**

出行控制程度为state时，对应感染系数分别下降为原来的 1/state

#### 任务提示 {#任务提示}

**测试输入**：

```py
    N     = 1e8             # 武汉总人数：1000万人
    gamma = 1/25            # 假设肺炎平均25天治愈（15天潜伏+10天治疗） 
    y0    = [N-1, 1, 0]     # 初始发病1人，其他人员正常 [S0, I0, R0]
    beta = 1.0/N            # 平均传染率
    t   = range(0, 5, 1) # 模拟5天的发展情况，单位时间为1天
    print("—————————————— 无出行控制下感染情况：——————————————")
    simulation = SIR(beta=beta, gamma=gamma, y0=y0)
    y_f = simulation.solve_with_control(t)
    myprint_list(y_f)
    for state in [2,5,10]:  #出行控制的程度,对应传染率分别下降为原来的1/state
            simulation = SIR(beta=beta, gamma=gamma, y0=y0)
            y_f = simulation.solve_with_control(t,state)
            print("———出行控制后，传染率为原来的 1/{} 时,对应的感染情况：———".format(state))
            myprint_list(y_f)
```

**预期输出**：

```py
    —————————————— 无出行控制下感染情况：——————————————
    [1.0000, 2.4208, 5.8603, 14.1865, 34.3428]
    ———出行控制后，传染率为原来的 1/2 时,对应的感染情况：———
    [1.0000, 1.5658, 2.4517, 3.8389, 6.0110]
    ———出行控制后，传染率为原来的 1/5 时,对应的感染情况：———
    [1.0000, 1.1728, 1.3755, 1.6131, 1.8919]
    ———出行控制后，传染率为原来的 1/10 时,对应的感染情况：———
    [1.0000, 1.0618, 1.1274, 1.1971, 1.2711]
```

---

开始你的任务吧，祝你成功！

framework.py

```py
import numpy as np

def myprint_list(l):
    if type(l)!=list:
        print('Error: the result is not a list')
    else:
        print('[', end='')
        for i in range(len(y_f)-1):
            print("%.4f, "%y_f[i][1], end='')
        print("%.4f]"%y_f[len(y_f)-1][1])

class SIR():
    def __init__(self, beta, gamma, y0):
        self.beta = beta; self.gamma = gamma; self.y0 = y0  # 参数属性

    def f(self, y, state): # y为当前状态，列表[S,I,R]
        S = y[0]
        I = y[1]
        R = y[2]
        return [-self.beta/state * S * I, self.beta/state * S * I - self.gamma * I, self.gamma * I]

    def solve_with_control(self, x, state = 1):
        # ------------------begin-----------------------
        y = [y0] * len(t);
        for k in range(len(t)-1):
            h = t[k+1]-t[k]
            y_p = np.array(y[k] + h * np.array(self.f(y[k], state)))
            y_c = np.array(y[k] + h * np.array(self.f(y_p, state)))
            y[k+1] = (y_p + y_c)/2
        return y

        # -------------------end------------------------

N     = 1e8             # 武汉总人数：1000万人
gamma = 1/25            # 假设肺炎平均25天治愈（15天潜伏+10天治疗） 
y0    = [N-1, 1, 0]     # 初始发病1人，其他人员正常 [S0, I0, R0]
beta  = 1.0/N            # 平均传染率

for T in [5, 10, 20]:
    t   = range(0, T, 1) # 模拟T天的发展情况，单位时间为1天
    print("—————————————— 无出行控制下感染情况：——————————————")
    simulation = SIR(beta=beta, gamma=gamma, y0=y0)
    y_f = simulation.solve_with_control(t)
    myprint_list(y_f)

    for state in [2 , 5 , 10]:   # 出行控制的程度，对应感染率分别下降为原来的 1/state
        simulation = SIR(beta=beta, gamma=gamma, y0=y0)
        y_f = simulation.solve_with_control(t,state)
        print("———出行控制后，传染率为原来的 1/{} 时,对应的感染情况：———".format(state))
        myprint_list(y_f)
```

---

### 第6关：绘制疫情发展趋势对比分析图

#### 任务描述 {#任务描述}

每天的疫情数据代表着疫情的态势。通过对疫情数据可视化可以直观地反映疫情的态势。为有效防止疫情进一步扩散和蔓延，各地纷纷出台各种举措加强疫情防控工作。本关将在第4关（引入隔离机制）的基础上，绘制未隔离与隔离机制下新冠病毒发展趋势对比分析图。通过图示，可以直观看出为什么要引入隔离机制进行疫情防控？为什么要排查疑似？

#### 相关知识 {#相关知识}

为了完成本关任务，你需要掌握：1.设置图例，2.设置坐标轴及标题，3.绘制多条曲线。

##### 设置图例 {#设置图例}

在 matplotlib 中，可通过legend函数设置图例，调用方式如下：

```py
    legend()                       #以默认形式设置图例
    legend(labels)               #标记已有的绘图
    legend(handles, labels)  #明确定义图例中的元素
```

代码示例如下：

```py
    # 标记已有的绘图
    plt.plot([1, 2, 3])
    plt.legend(['A simple line']) 
    plt.show()
```

```py
    # 明确定义图例中的元素
    line1, = plt.plot([1,2,3],  linestyle='--')
    line2, = plt.plot([3,2,1],  linewidth=4)
    plt.legend([line1,line2], ["Line 1","Line 2"])
    plt.show()
```

可通过`loc`参数设置图例位置，通过`fontsize`设置字体大小，通过`title`参数设置图例标题，详细的参数配置可参见[官方文档](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend)。

##### 设置坐标轴及标题 {#设置坐标轴及标题}

在使用 matplotlib 模块画坐标图时，往往需要对坐标轴设置很多参数，这些参数包括横纵坐标轴范围、坐标轴刻度大小、坐标轴名称等等，在 matplotlib 中包含了很多函数，用来对这些参数进行设置。

* plt.xlim、plt.ylim用于设置横纵坐标轴范围；

* plt.xlabel、plt.ylabel用于设置坐标轴名称；

* plt.xticks、plt.yticks用于设置坐标轴刻度；

* plt.title用于设置图像标题。

上面的`plt`表示 `matplotlib.pyplot` 模块，即在程序中以`import matplotlib.pyplot as plt`方式导入 `matplotlib.pyplot` 模块。

##### 绘制多条曲线 {#绘制多条曲线}

绘制多条曲线有两种情况：

第一种是在同一坐标系上绘制多条曲线，能够清楚地看到多条曲线的对比情况。可通过直接叠加使用plot进行绘制。

示例如下：

```py
    import matplotlib.pyplot as plt
    from math import sin, cos, radians
    x = range(0, 360)
    y1 = [sin(radians(e)) for e in x]
    y2 = [cos(radians(e)) for e in x]
    plt.plot(x, y1, 'b-')
    plt.plot(x, y2, 'r--')
    plt.legend(['sin(x)', 'cos(x)'], loc='upper center')
    plt.xlabel('x')         #设置 x 轴文字标记
    plt.ylabel('sin/cos')   #设置 y 轴文字标记
    plt.axis([0, 360, -1.0, 1.0])  #设置坐标范围
    plt.show()
```

上述代码展示图像如下所示：

![](/pic/multiline.png)

第二种是在不同子图上画图，多用于呈现不同内容的曲线。需要用到subplot函数，它主要用于创建子图，具体调用方式可参见[官方文档](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html#matplotlib.pyplot.subplot)。

示例如下：

```py
    import numpy as np
    import matplotlib.pyplot as plt
    x1 = np.linspace(0.0, 5.0)
    x2 = np.linspace(0.0, 2.0)
    y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
    y2 = np.cos(2 * np.pi * x2)
    plt.subplot(2, 1, 1) # 开始绘制子图 1
    plt.plot(x1, y1, 'o-')
    plt.title('A tale of 2 subplots')
    plt.ylabel('Damped oscillation') 
    plt.subplot(2, 1, 2) # 开始绘制子图 2
    plt.plot(x2, y2, '.-')
    plt.xlabel('time (s)')
    plt.ylabel('Undamped')
    plt.show()
```

上述代码展示图像如下所示：

![](/pic/subpic.png)

##### SIR模型 {#sir模型}

> SIR模型（Susceptible Infected Recovered Model）是一种传播模型，是疾病及信息传播过程的抽象描述。是传染病模型中最经典的模型。此模型能够较为粗略地展示出一种传染病从发病到结束的过程，其核心在于常微分方程。

* S 类：易感者（Susceptible）
* I 类：感染者（Infective）
* R 类：移出者（Removal） 
  ![](/pic/SIRmodel.png)

常微分方程组表示如下：
$$
\left\{  
     \begin{array}{lr}  
     S' = -\beta SI &  \\  
     I' = \beta SI-\gamma I & \\  
     R' = \gamma I &    
     \end{array}  
\right.  
$$

其中S′,I′,R′分别表示S,I,R关于时间t的导数，下面将其组合为向量形式并用函数f表示，得:$$(S',I',R')=f(S,I,R)=(−\beta SI,\beta SI−\gamma I,\gamma I)$$ 其中:β为感染系数，代表易感人群与传染源接触被感染的概率，即平均传染率；γ为隔离\(恢复\)系数，一般对其倒数1/γ更感兴趣，代表了平均感染时间；S,I定义见上，S\(0\)为初始易感人数，I\(0\)为初始感染人数。

##### 改进的欧拉法（梯形法，两步） {#改进的欧拉法（梯形法，两步）}

根据欧拉法和梯形公式，我们可以得到改进的欧拉公式，改进的欧拉法形式如下： ![](/pic/improvedEuler.png)

代码示例如下:

```py
    def trapezoidalEuler(f, x, y0):
        y = [y0] * len(x)
        for k in range(len(x)-1):
            h = x[k+1]-x[k]
            y_p = y[k] + h * f(x[k], y[k])
            y_c = y[k] + h * f(x[k+1], y_p)
            y[k+1] = 1/2 * (y_p + y_c)
        return y  # y = [y0, y1, ..., yn]
```

#### 编程要求 {#编程要求}

利用改进的欧拉法\(梯形法，两步迭代）完善SIR模型，并引入隔离机制，实现SIR类并编写方法`∙solve_with_quarantine(self,x)`: 实现改进的欧拉法\(梯形法，两步迭代），并引入不同的隔离机制（见下），其中参数x是列表\[0, .., t-1\]，该函数返回t天内每天感染人数的列表 \(列表大小为天数t\)。（注意：本场景中应用改进的欧拉法时，`f(x[k],y[k])`和`f(x[k+1],y_p)`中`x[k],x[k+1]`对f函数的计算没有影响）

**不同的隔离机制**

①未实施隔离： gamma = 1/25，假设肺炎平均25天治愈（15天潜伏+10天治疗）  
②隔离确诊患者：gamma = 1/15，按最长15天发病确诊后被隔离

③隔离疑似人员：gamma = 1/3，平均3天被隔离

**图形绘制**

请补全右侧编程框中代码，在同一坐标系上绘制上述三种不同隔离机制（含未实施隔离）下新冠病毒发展趋势曲线，形成趋势对比分析图。

* 图片大小设为10\*8 \(单位为inch\)；
* 注意提交的代码中不要包含show语句，否则会影响自动评测

#### 测试说明 {#测试说明}

平台将运行用户补全的代码文件，并将存储的 src/step3/student/result.png 图像与标准答案图像比较，然后判断用户编写代码是否正确:

* 若画图正确，测试集将输出：祝贺！图片与预期输出一致
* 否则，测试集将输出：图片与预期输出不一致，请继续努力！

---

开始你的任务吧，祝你成功！

framework.py

```py
#-*- coding : utf-8 -*-
# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 采用matplotlib作图时默认设置下是无法显示中文的，凡是汉字都会显示成方块。
# 实际上，matplotlib是支持unicode编码的，不能正常显示汉字主要是没有找到合适的中文字体。
from pylab import mpl
mpl.rcParams['font.sans-serif']= ['SimHei']
# 解决负号显示问题
matplotlib.rcParams['axes.unicode_minus']=False

class SIR():
    def __init__(self, beta, gamma, y0):
        self.beta = beta; self.gamma = gamma; self.y0 = y0  # 参数属性

    def f(self, y, gamma): #  y为当前状态，列表[S,I,R]
        S = y[0]
        I = y[1]
        R = y[2]
        return [-self.beta * S * I, self.beta * S * I - gamma * I, gamma * I]

    def solve_with_quarantine(self, x, gamma = 1/25):
        # ------------------begin-----------------------
        y = [y0] * len(t);
        for k in range(len(t)-1):
            h = t[k+1]-t[k]
            y_p = np.array(y[k] + h * np.array(self.f(y[k], gamma)))
            y_c = np.array(y[k] + h * np.array(self.f(y_p, gamma)))
            y[k+1] = (y_p + y_c)/2
        return y
        # -------------------end------------------------

N     = 1e8             # 武汉总人数：1000万人
gamma = 1/25            # 假设肺炎平均25天治愈（15天潜伏+10天治疗）
y0    = [N-1, 1, 0]     # 初始发病1人，其他人员正常 [S0, I0, R0]
beta = 1.0/N            # 平均传染率
t   = range(0, 60, 1)   # 模拟60天的发展情况，单位时间为1天
simulation = SIR(beta=beta, gamma=gamma, y0=y0)

# ------------------begin-----------------------
#1. 设置图形大小
fig = plt.figure(figsize=(10, 8))

#2. 绘制曲线：横轴是时间/天，纵轴是感染人数
x = t
simulation = SIR(beta=beta, gamma=gamma, y0=y0)
y1 = simulation.solve_with_quarantine(t)

gamma1 = 1/15            # 隔离确诊患者：按最长15天发病确诊后被隔离
y2 = simulation.solve_with_quarantine(t, gamma1)

gamma1 = 1/3            # 隔离疑似人员：按平均3天被隔离
y3 = simulation.solve_with_quarantine(t, gamma1)


#3. 设置图题'未隔离与隔离机制下新冠病毒发展趋势对比分析'、
#   横轴标签'时间/天'、纵轴标签'人数'、
#   图列说明('未实施隔离', '确诊隔离', '疑似隔离', 分别对应三条曲线)

plt.plot(x, [y[1] for y in y1], label='未实施隔离')
plt.plot(x, [y[1] for y in y2], label='确诊隔离')
plt.plot(x, [y[1] for y in y3], label='疑似隔离')
plt.title('未隔离与隔离机制下新冠病毒发展趋势对比分析')
plt.xlabel('时间/天')
plt.ylabel('人数')
plt.legend()

# ------------------end-----------------------

# 设置y轴刻度
Vy = [1.0e6, 5.0e6] + [i * 1.e7 for i in range(11)]
plt.yticks(Vy, ['%d'%e for e in Vy])

plt.savefig('src/step6/student/result.png')
```



