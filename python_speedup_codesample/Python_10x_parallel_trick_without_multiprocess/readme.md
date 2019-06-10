转载自微信公众号 稀牛学院

虽然python的多进程已经得到了广泛的应用，但在本文中，我们将证明它不适用的几个重要应用程序类，其中包括数值数据处理、状态计算和具有昂贵初始化的计算。

不适用原因：
    数值数据处理效率低下
    缺乏抽象化的状态计算（无法在独立的“任务”之间共享变量）

Ray是一个快速、简单的框架，用于构建和运行解决这些问题的分布式应用程序。Ray利用ApacheArrow进行高效的数据处理，并为分布式计算提供task和actor的抽象概念。

本文对三种不易用Python多进程表示的工作量进行了基准测试，并对Ray、Python多进程和串行Python代码进行了比较。请注意，务必与优化的单线程代码进行比较。

在这些基准测试中，Ray比串行Python快10-30倍，比多进程快5-25倍，比大型机器上这两种方法快5-15倍。
![testpic](https://github.com/monkeyshichi/python_code_optimize/blob/master/python_speedup_codesample/imgfolder/1560148087.png)

使用M5实例类型（M5.large用于1个物理内核，M5.24XLarge用于48个物理内核）在EC2上进行基准测试。这里提供了所有运行基准的代码。本文包含了缩写的代码片段，与完整的基准测试程序相比，主要有以下区别：
1）计时和打印代码
2）预热光线对象存储的代码
3）使基准适应小型机器的代码

数值数据

许多机器学习、科学计算和数据分析工作会用到大型数据组。例如，一个数组可以表示一个大的图像或数据集，应用程序可能希望有多个任务分析该图像，因此有效处理数字数据至关重要。

运行下面的for循环，每次使用Ray需要0.84秒，使用Python多进程需要7.5秒，使用串行Python需要24秒（在48个物理核上）。这一性能差异解释了为什么可以在Ray上构建类似Modin的库，而不是在其他库之上。

Ray的代码如下：
```
import numpy as np
import psutil
import ray
import scipy.signal

num_cpus = psutil.cpu_count(logical=False)

ray.init(num_cpus=num_cpus)

@ray.remote
def f(image, random_filter):
    # Do some image processing.
    return scipy.signal.convolve2d(image, random_filter)[::5, ::5]

filters = [np.random.normal(size=(4, 4)) for _ in range(num_cpus)]

# Time the code below.

for _ in range(10):
    image = np.zeros((3000, 3000))
    image_id = ray.put(image)
    ray.get([f.remote(image_id, filters[i]) for i in range(num_cpus)])
```
通过调用 ray.put（image），大型数组存储在共享内存中，所有工作进程都可以访问它，而无需创建副本。这不仅适用于数组，还适用于包含数组的对象（如数组列表）。

当工作人员执行 f 任务时，结果再次存储在共享内存中。当脚本调用 ray.get（[…]）时，它将创建由共享内存支持的numpy数组，而无需反序列化或复制值。

Ray使用ApacheArrow作为底层数据布局和序列化格式以及Plasma共享内存对象存储使这些优化成为可能。

Python多进程代码如下：
```
from multiprocessing import Pool
import numpy as np
import psutil
import scipy.signal

num_cpus = psutil.cpu_count(logical=False)

def f(args):
    image, random_filter = args
    # Do some image processing.
    return scipy.signal.convolve2d(image, random_filter)[::5, ::5]

pool = Pool(num_cpus)

filters = [np.random.normal(size=(4, 4)) for _ in range(num_cpus)]

# Time the code below.

for _ in range(10):
    image = np.zeros((3000, 3000))
    pool.map(f, zip(num_cpus * [image], filters))
```
这里的区别在于，当在进程之间传递大型对象时，Python多进程使用pickle对它们进行序列化。这种方法要求每个进程创建自己的数据副本，这增加了大量的内存使用以及昂贵的反序列化开销，Ray通过使用Apache Arrow数据布局和Plasma存储一起进行零拷贝序列化来避免这种情况。
状态计算

需要在许多小的工作单元之间共享大量“状态”的工作是另一类工作负载，这对Python多进程构成了挑战。这个模式非常常见，我用一个玩具处理应用程序来进行说明。

![testpic](https://github.com/monkeyshichi/python_code_optimize/blob/master/python_speedup_codesample/imgfolder/1560151972.png)

状态通常封装在Python类中，而Ray提供了一个actor抽象概念，这样类就可以在并行和分布式设置中使用。相反，Python多进程并没有提供一种自然的方法来并行化Python类，因此用户通常需要在映射调用之间传递相关的状态。这种策略在实践中很难实现（许多Python变量不容易序列化），而且在实际工作时速度会很慢。

下面是一个玩具示例，它使用并行任务一次处理一个文档，提取每个单词的前缀，并在末尾返回最常见的前缀。前缀计数存储在actor状态中，并由不同的任务改变。

这个例子中，Ray运行了3.2s，Python多进程运行了21s，串行Python运行了54s（48个物理核心）。

Ray的代码如下：
```
from collections import defaultdict
import numpy as np
import psutil
import ray

num_cpus = psutil.cpu_count(logical=False)

ray.init(num_cpus=num_cpus)

@ray.remote
class StreamingPrefixCount(object):
    def __init__(self):
        self.prefix_count = defaultdict(int)
        self.popular_prefixes = set()

    def add_document(self, document):
        for word in document:
            for i in range(1, len(word)):
                prefix = word[:i]
                self.prefix_count[prefix] += 1
                if self.prefix_count[prefix] > 3:
                    self.popular_prefixes.add(prefix)

    def get_popular(self):
        return self.popular_prefixes

streaming_actors = [StreamingPrefixCount.remote() for _ in range(num_cpus)]

# Time the code below.

for i in range(num_cpus * 10):
    document = [np.random.bytes(20) for _ in range(10000)]
    streaming_actors[i % num_cpus].add_document.remote(document)

# Aggregate all of the results.
results = ray.get([actor.get_popular.remote() for actor in streaming_actors])
popular_prefixes = set()
for prefixes in results:
    popular_prefixes |= prefixes
```
Ray在这里表现很好，因为Ray的抽象适合当前的问题。这个应用程序需要一种在分布式设置中封装和改变状态的方法，并且参与者能够满足这个需求。
Python多进程的代码如下：
```
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import psutil

num_cpus = psutil.cpu_count(logical=False)

def accumulate_prefixes(args):
    running_prefix_count, running_popular_prefixes, document = args
    for word in document:
        for i in range(1, len(word)):
            prefix = word[:i]
            running_prefix_count[prefix] += 1
            if running_prefix_count[prefix] > 3:
                running_popular_prefixes.add(prefix)
    return running_prefix_count, running_popular_prefixes

pool = Pool(num_cpus)

running_prefix_counts = [defaultdict(int) for _ in range(4)]
running_popular_prefixes = [set() for _ in range(4)]

for i in range(10):
    documents = [[np.random.bytes(20) for _ in range(10000)]
                  for _ in range(num_cpus)]
    results = pool.map(
        accumulate_prefixes,
        zip(running_prefix_counts, running_popular_prefixes, documents))
    running_prefix_counts = [result[0] for result in results]
    running_popular_prefixes = [result[1] for result in results]

popular_prefixes = set()
for prefixes in running_popular_prefixes:
    popular_prefixes |= prefixes
```
这里的挑战是pool.map执行无状态函数，这意味着要在一个pool.map调度中使用其生成的任何变量都需要从第一个调用返回并传递到第二个调用。对于小型对象来说，这种方法是可以接受的，但是当需要共享大型中间结果时，传递它们的成本是很高的（注意，变量是不可能在线程之间共享的，但是因为它们在进程边界之间共享，所以必须使用像pickle这样的库将变量序列化为一个字节串）。

因为它必须传递如此多的状态，所以多进程版本看起来非常笨拙，最终只在串行Python上实现了很小的加速。实际上，不会编写这样的代码，只是因为不会使用Python多进程进行流处理。相反，可能会使用专用的流处理框架。这个例子表明，Ray非常适合构建这样的框架或应用程序。

注意，有很多方法可以使用Python多进程。在本例中，我们将pool.map进行比较，因为它提供了最接近的API比较。在本例中，应该可以通过启动不同的进程并在它们之间设置多个多进程队列来获得更好的性能，但是设计起来会相对复杂。

昂贵的初始化

与前面的示例不同，许多并行计算不一定要求在任务之间共享中间计算，但无论如何都会从中受益。即使是无状态计算，在状态初始化代价高昂时也可以从共享状态中获益。

下面的例子中我们希望从磁盘加载一个保存的神经网络，并使用它来并行分类一组图像。
![testpic](https://github.com/monkeyshichi/python_code_optimize/blob/master/python_speedup_codesample/imgfolder/1560152059.png)

本例中Ray用时5s，Python多进程用时126s，串行Python用时64s（在48个物理核上）。在这种情况下，串行Python版本使用许多核心（通过TensorFlow）来并行计算，因此它实际上不是单线程的。

假设我们最初通过运行以下内容创建了模型。
```
import tensorflow as tf

mnist = tf.keras.datasets.mnist.load_data()
x_train, y_train = mnist[0]
x_train = x_train / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
# Train the model.
model.fit(x_train, y_train, epochs=1)
# Save the model to disk.
filename = '/tmp/model'
model.save(filename)
```
现在我们希望加载模型并使用它来分类一组图像。我们批量进行这项工作是因为在应用程序中，图像可能不会全部同时可用，而图像分类可能需要与数据加载并行进行。
Ray的代码如下：
```
import psutil
import ray
import sys
import tensorflow as tf

num_cpus = psutil.cpu_count(logical=False)

ray.init(num_cpus=num_cpus)

filename = '/tmp/model'

@ray.remote
class Model(object):
    def __init__(self, i):
        # Pin the actor to a specific core if we are on Linux to prevent
        # contention between the different actors since TensorFlow uses
        # multiple threads.
        if sys.platform == 'linux':
            psutil.Process().cpu_affinity([i])
        # Load the model and some data.
        self.model = tf.keras.models.load_model(filename)
        mnist = tf.keras.datasets.mnist.load_data()
        self.x_test = mnist[1][0] / 255.0

    def evaluate_next_batch(self):
        # Note that we reuse the same data over and over, but in a
        # real application, the data would be different each time.
        return self.model.predict(self.x_test)

actors = [Model.remote(i) for i in range(num_cpus)]

# Time the code below.

# Parallelize the evaluation of some test data.
for j in range(10):
    results = ray.get([actor.evaluate_next_batch.remote() for actor in actors])
```
加载模型的速度很慢，我们只想加载一次。Ray通过在actor构造函数中加载模型来分摊成本。如果模型需要放在GPU上，那么初始化将更加昂贵。

多进程速度较慢，因为它需要在每次映射调度中重新加载模型，假定映射函数是无状态的。

多进程代码如下所示。请注意，在某些情况下，可以使用multiprocessing.pool的initializer参数来实现这一点。但是，这仅限于每个进程初始化相同的设置，并且不允许不同的进程执行不同的设置功能（例如，加载不同的神经网络模型），并且不允许不同的任务针对不同的工人。
```
from multiprocessing import Pool
import psutil
import sys
import tensorflow as tf

num_cpus = psutil.cpu_count(logical=False)

filename = '/tmp/model'

def evaluate_next_batch(i):
    # Pin the process to a specific core if we are on Linux to prevent
    # contention between the different processes since TensorFlow uses
    # multiple threads.
    if sys.platform == 'linux':
        psutil.Process().cpu_affinity([i])
    model = tf.keras.models.load_model(filename)
    mnist = tf.keras.datasets.mnist.load_data()
    x_test = mnist[1][0] / 255.0
    return model.predict(x_test)

pool = Pool(num_cpus)

for _ in range(10):
    pool.map(evaluate_next_batch, range(num_cpus))
```
我们在所有这些例子中看到的是，Ray的性能不仅来自于它的性能优化，还来自于对手头任务进行适当的抽象化。有状态计算对许多应用程序都很重要，将有状态计算强制为无状态抽象是需要代价的。

若有侵权,请告知删除