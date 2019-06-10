转载自：知乎 用Python的交易员

在这篇文章里，将会通过实际的例子展示如何对一段量化策略常用的代码实现百倍加速。
一段常用的代码
接下来要用的例子相信几乎所有做量化策略的人都写过类似的代码：对时间序列求算术移动平均值。

这里我们先初始化即将用到的数据：10万个数据点（随机整数），遍历计算窗口为500的算术移动平均值，每种算法运行10次求平均耗时。
```
# 这个测试目标在于仿造一个类似于实盘中，不断有新的数据推送过来，
# 然后需要计算移动平均线数值，这么一个比较常见的任务。

from __future__ import division
import time
import random

# 生成测试用的数据
data = []
data_length = 100000    # 总数据量
ma_length = 500         # 移动均线的窗口
test_times = 10         # 测试次数

for i in range(data_length):
    data.append(random.randint(1, 100))
```
在每次测试中，我们都通过遍历测试用数据的方式来模拟实盘中策略不断收到新数据推送的情况（同样适用于事件驱动的回测模式），将计算出的移动平均值不断保存到一个列表list中作为最终结果返回。

测试用电脑的配置情况：Core i7-6700K 4.0G/16G/Windows 7。

第一步我们以最简单、最原始的方式来计算移动平均值：
```
# 计算500期的移动均线，并将结果保存到一个列表里返回

def ma_basic(data, ma_length):

    # 用于保存均线输出结果的列表
    ma = []

    # 计算均线用的数据窗口
    data_window = data[:ma_length]

    # 测试用数据（去除了之前初始化用的部分）
    test_data = data[ma_length:]

    # 模拟实盘不断收到新数据推送的情景，遍历历史数据计算均线
    for new_tick in test_data:
        # 移除最老的数据点并增加最新的数据点
        data_window.pop(0)
        data_window.append(new_tick)

        # 遍历求均线
        sum_tick = 0
        for tick in data_window:
            sum_tick += tick
        ma.append(sum_tick/ma_length)

    # 返回数据
    return ma

# 运行测试
start = time.time()

for i in range(test_times):
    result = ma_basic(data, ma_length)

time_per_test = (time.time()-start)/test_times
time_per_point = time_per_test/(data_length - ma_length)

print u'单次耗时：%s秒' %time_per_test
print u'单个数据点耗时：%s微秒' %(time_per_point*1000000)
print u'最后10个移动平均值：', result[-10:]
```
单次耗时指的是遍历完整个测试数据计算移动平均值所需的时间，单个数据点耗时指的是遍历过程中每个数据点的平均计算耗时，最后10个移动平均值用于和后续的算法进行比对，保证计算结果的正确性。

ma_basic测试结果

单次耗时：1.15699999332秒
单个数据点耗时：11.6281406364微秒
大约10万个数据点（说大约因为有500个用于初始化了），这个测试结果不能说很好但也还过得去。考虑到一个简单的双均线CTA策略（Double SMA Strategy），每个数据点来了后会进行两次均线计算，通常均线窗口不会超过500，且比较两根均线交叉情况的算法开销更低，估计策略单纯在信号计算方面的耗时会在30微秒以内，对于一个通常跑在1分钟线甚至更高时间周期上的策略而言已经是绰绰有余。

有了起点，下面来试着一步步提升性能。

试试NumPy？
用Python做数值运算性能不够的时候，很多人的第一反应就是上NumPy：之前的ma_basic里，比较慢的地方应该在每一个新的数据点加入到data_window中后遍历求平均值的代码，那么改用numpy.array数组来求和应该性能就会有所提升了吧？
```
# 改用numpy（首先是一种常见的错误用法）

import numpy as np

def ma_numpy_wrong(data, ma_length):
    ma = []
    data_window = data[:ma_length]
    test_data = data[ma_length:]

    for new_tick in test_data:
        data_window.pop(0)
        data_window.append(new_tick)

        # 使用numpy求均线，注意这里本质上每次循环
        # 都在创建一个新的numpy数组对象，开销很大
        data_array = np.array(data_window)
        ma.append(data_array.mean())

    return ma
```
ma_numpy_wrong测试结果

    单次耗时：2.11879999638秒
    单个数据点耗时：21.2944723254微秒
WTF?!用NumPy后居然反而速度降低了一半（耗时增加到了快2倍）！

这里的写法是一个非常常见的NumPy错误用法，问题就出在：
data_array = np.array(data_window)
由于NumPy中的对象大多实现得比较复杂（提供了丰富的功能），所以其对象创建和销毁的开销都非常大。上面的这句代码意味着在计算每一个新数据点时，都要创建一个新的array对象，并且仅使用一次后就会销毁，使用array.mean方法求均值带来的性能提升还比不上array对象创建和销毁带来的额外开销。

正确的用法是把np.array作为data_window时间序列的容器，每计算一个新的数据点时，使用底层数据偏移来实现数据更新：
```
# numpy的正确用法

def ma_numpy_right(data, ma_length):
    ma = []

    # 用numpy数组来缓存计算窗口内的数据
    data_window = np.array(data[:ma_length])

    test_data = data[ma_length:]

    for new_tick in test_data:
        # 使用numpy数组的底层数据偏移来实现数据更新
        data_window[0:ma_length-1] = data_window[1:ma_length]
        data_window[-1] = new_tick
        ma.append(data_window.mean())

    return ma
```
ma_numpy_right测试结果

    单次耗时：0.614300012589秒
    单个数据点耗时：6.17386947325微秒
速度比ma_basic提高了大约2倍，看来NumPy也就这么回事了。
JIT神器：Numba
关心过Python性能的朋友应该都听过PyPy的大名，通过重新设计的Python解释器，PyPy内建的JIT技术号称可以将Python程序的速度提高几十倍（相比于CPython），可惜由于兼容性的问题并不适合于量化策略开发这一领域。

幸运的是，我们还有Anaconda公司推出的Numba。Numba允许用户使用基于LLVM的JIT技术，对程序内想要提高性能的部分（函数）进行局部优化。同时Numba在设计理念上更加务实：可以直接在CPython中使用，和其他常用的Python模块的兼容性良好，并且最爽的是使用方法傻瓜到了极点：
```
# 使用numba加速，ma_numba函数和ma_basic完全一样
import numba

@numba.jit
def ma_numba(data, ma_length):
ma = []
data_window = data[:ma_length]
test_data = data[ma_length:]

for new_tick in test_data:
    data_window.pop(0)
    data_window.append(new_tick)
    sum_tick = 0
    for tick in data_window:
        sum_tick += tick
    ma.append(sum_tick/ma_length)

return ma
```
ma_numba测试结果
    单次耗时：0.043700003624秒
    单个数据点耗时：0.439196016321微秒
OMG！就加了一行@numba.jit，性能竟然提高了26倍！这估计是按照代码修改行数算，性价比最高的优化方案了。

改写算法
从编程哲学的角度来看，想提高计算机程序的速度，一个最基本的原则就是降低算法复杂度。看到这里估计早就有量化老手ma_basic不爽了，弄个复杂度O(N)的算法来算平均值，就不能缓存下求和的结果，把复杂度降低到O(1)么？
```
# 将均线计算改写为高速算法
def ma_online(data, ma_length):
    ma = []
    data_window = data[:ma_length]
    test_data = data[ma_length:]

    # 缓存的窗口内数据求和结果
    sum_buffer = 0

    for new_tick in test_data:
        old_tick = data_window.pop(0)
        data_window.append(new_tick)

        # 如果缓存结果为空，则先通过遍历求第一次结果
        if not sum_buffer:
            sum_tick = 0
            for tick in data_window:
                sum_tick += tick
            ma.append(sum_tick/ma_length)

            # 将求和结果缓存下来
            sum_buffer = sum_tick
        else:
            # 这里的算法将计算复杂度从O(n)降低到了O(1)
            sum_buffer = sum_buffer - old_tick + new_tick
            ma.append(sum_buffer/ma_length)

    return ma
```
ma_online测试结果

    单次耗时：0.0348000049591秒
    单个数据点耗时：0.349748793559微秒
哲学果然才是最强大的力量！！！
改写算法后的ma_online无需JIT就超越了ma_numba，将性能提高到了33倍（对比ma_basic），如果再把numba加上会如何？
```
# 高速算法和numba结合，ma_online_numba函数和ma_online完全一样
@numba.jit
def ma_online_numba(data, ma_length):
    ma = []
    data_window = data[:ma_length]
    test_data = data[ma_length:]

    sum_buffer = 0

    for new_tick in test_data:
        old_tick = data_window.pop(0)
        data_window.append(new_tick)

        if not sum_buffer:
            sum_tick = 0
            for tick in data_window:
                sum_tick += tick
            ma.append(sum_tick/ma_length)
            sum_buffer = sum_tick
        else:
            sum_buffer = sum_buffer - old_tick + new_tick
            ma.append(sum_buffer/ma_length)

    return ma
```
ma_online_numba测试结果

    单次耗时：0.0290000200272秒
    单个数据点耗时：0.29145748771微秒
尽管性能进一步提升了到了40倍，不过相比较于ma_numba对比ma_basic的提升没有那么明显，果然哲学的力量还是太强大了。

终极武器：Cython
到目前为止使用纯Python环境下的优化方法我们已经接近了极限，想要再进一步就得发挥Python胶水语言的特性了：使用其他扩展语言。由于CPython虚拟机的开发语言是C，因此在性能提升方面的扩展语言主要选择就是C/C++，相关的工具包括ctypes、cffi、Swig、Boost.Python等，尽管功能十分强大，不过以上工具都无一例外的需要用户拥有C/C++语言相关的编程能力，对于很多Python用户而言是个比较麻烦的事。

好在Python社区对于偷懒的追求是永无止境的，Cython这一终极武器应运而生。关于Cython的详细介绍可以去官网看，简单来它的主要作用就是允许用户以非常接近Python的语法来实现非常接近C的性能。

先来试试最简单的方法：完全不修改任何代码，只是把函数放到.pyx文件里，调用Cython编译成.pyd扩展模块。
```
# 基础的cython加速
def ma_cython(data, ma_length):
    ma = []
    data_window = data[:ma_length]
    test_data = data[ma_length:]

    for new_tick in test_data:
        data_window.pop(0)
        data_window.append(new_tick)

        sum_tick = 0
        for tick in data_window:
            sum_tick += tick
        ma.append(sum_tick/ma_length)

    return ma
```
ma_cython测试结果

    单次耗时：0.600800013542秒
    单个数据点耗时：6.03819109088微秒
ma_cython和ma_basic的代码完全相同，简单使用Cython编译后性能提高了大约1倍，不过这和之前我们已经达成的优化效果比可以说是毫无吸引力。

Cython官方的Quick Start里，第一步是教会用户如何去编译程序，第二步就是如何使用静态声明来大幅提高性能，所以我们的下一步就是：静态声明+高速算法。
```
# cython和高速算法
def ma_cython_online(data, ma_length):
    # 静态声明变量
    cdef int sum_buffer, sum_tick, old_tick, new_tick

    ma = []
    data_window = data[:ma_length]
    test_data = data[ma_length:]
    sum_buffer = 0

    for new_tick in test_data:
        old_tick = data_window.pop(0)
        data_window.append(new_tick)

        if not sum_buffer:
            sum_tick = 0
            for tick in data_window:
                sum_tick += tick
            ma.append(sum_tick/ma_length)

            sum_buffer = sum_tick
        else:
            sum_buffer = sum_buffer - old_tick + new_tick
            ma.append(sum_buffer/ma_length)

    return ma
```
ma_cython_online测试结果

    单次耗时：0.00980000495911秒
    单个数据点耗时：0.0984925121518微秒
117倍！！！比ma_online_numba的速度还提高了接近3倍，98纳秒的计算速度已经足以满足大部分毫秒级别高频策略的延时需求。

主要的功臣是这行：
```
cdef int sum_buffer, sum_tick, old_tick, new_tick
```
把函数中用到的变量静态声明成int类型后，Cython在编译时无需再考虑Python对象的动态性特点，可以把整个函数高度优化成类似静态语言的实现，从而达到了接近C语言的运行性能，再加上复杂度O(1)的高速算法，有这个级别的性能提升也就不足为奇了。

附上简单的Cython使用指南：

把要使用Cython编译的函数放到.pyx文件中，比如test.pyx
创建setup.py用于设置相关编译选项
打开cmd或者terminal进入到test.pyx和setup.py所在的文件夹
运行python setup.py build_ext --inplace，执行编译
若编译成功则在当前文件夹下会出现test.pyd
打开python像使用其他模块一样载入（import）test（原文为test.pyd）使用
最后做一些总结吧：

1、现实工作中遇到需要优化Python程序性能时，首先要做的就是去寻找程序里延时较大的热点代码，找到了问题所在，解决方案才有意义；
2、所有的优化工作都应该基于测试来一步步推进，同样的优化方法对于不同类型的代码效果可能是截然相反的，同时错误的优化方法还不如不要优化（比如ma_numpy_wrong）；
3、只需增加一句代码（@numba.jit）就能实现加速的Numba无疑是性价比最高的优化方案，值得优先尝试，不过需要注意numba的JIT技术局限性比较大（主要针对数值计算相关的逻辑）；
4、学习如何降低算法复杂度和编写更高效的算法，可以在潜移默化中提高自己的编程水平，在长期而言是对Quant或者程序员最有价值的优化方法；
5、如果其他优化方法都无法达到令你满意的性能水平，试试Cython（记得一定要加静态声明）;
6、一个好的程序架构设计非常重要，把功能不同的计算逻辑分解到不同的函数里，适当降低每个函数的代码行数，会有助于后期的性能优化工作。

谢谢作者，我从中学到不少。作者是个大牛，还开发了vn.py,一个基于Python的开源交易平台开发框架，有兴趣的可以关注
若有侵权,请告知删除