# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
这是一个分布式代理服务器，用于在多个服务实例之间进行负载均衡和请求转发。

主要功能：
1. 服务发现：通过ZMQ接收prefill和decode服务的注册信息
2. 负载均衡：将请求分发到不同的服务实例
3. 请求转发：将客户端请求转发到prefill服务完成预填充，然后转发到decode服务完成解码
4. KV缓存传输协调：通过生成包含地址信息的request_id，协调Prefill和Decode服务之间的KV缓存传输

工作流程：
- Prefill服务（预填充服务）：负责处理请求的初始部分，生成第一个token和KV缓存
- Decode服务（解码服务）：负责处理后续的token生成，使用从Prefill传输来的KV缓存
- 代理服务器：协调这两个服务，实现请求的分离处理

================================================================================
KV缓存传输机制：ZMQ + NCCL 的分工协作
================================================================================

这个系统使用ZMQ和NCCL的混合架构来实现KV缓存的传输，两者各司其职：

┌─────────────────────────────────────────────────────────────────────────┐
│ ZMQ（ZeroMQ）：控制流和元数据传输                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ 作用：                                                                   │
│ 1. 服务发现：Prefill和Decode服务向代理注册（http_addr -> zmq_addr）      │
│ 2. 建立连接：发送"NEW"命令，交换NCCL的unique_id                        │
│ 3. 传输元数据：KV缓存的tensor_id、shape、dtype等信息                    │
│ 4. 协调通信：协调Prefill和Decode服务之间的通信                          │
│                                                                          │
│ 特点：                                                                    │
│ - 使用TCP/IP网络，可以跨机器通信                                          │
│ - 轻量级，适合传输小量控制信息                                           │
│ - 灵活，支持动态连接建立                                                 │
│                                                                          │
│ 传输的数据：                                                              │
│ - 控制命令（NEW, PUT, GET）                                             │
│ - 元数据（tensor_id, shape, dtype）                                    │
│ - NCCL unique_id（用于初始化NCCL通信器）                                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ NCCL（NVIDIA Collective Communications Library）：实际数据传输          │
├─────────────────────────────────────────────────────────────────────────┤
│ 作用：                                                                   │
│ 1. 传输实际的KV缓存张量数据（通常是GB级别的数据）                        │
│ 2. GPU到GPU的直接通信，绕过CPU和网络栈                                   │
│ 3. 高性能数据传输，适合传输大块数据                                       │
│                                                                          │
│ 特点：                                                                    │
│ - 使用InfiniBand、NVLink或以太网（取决于硬件配置）                       │
│ - GPU直接内存访问（DMA），性能极高                                        │
│ - 支持点对点通信（P2P），world_size=2                                   │
│                                                                          │
│ 传输的数据：                                                              │
│ - KV缓存的张量数据（torch.Tensor）                                      │
│ - 直接从Prefill的GPU内存传输到Decode的GPU内存                           │
└─────────────────────────────────────────────────────────────────────────┘

工作流程示例（PUT_ASYNC模式）：

1. Prefill服务完成预填充，生成KV缓存（存储在GPU内存）
2. Prefill服务通过ZMQ发送"NEW"命令给Decode服务：
   - 包含NCCL的unique_id
   - Decode服务接收后，使用相同的unique_id初始化NCCL通信器
   - 建立点对点NCCL通信组（只有两个rank：Prefill=0, Decode=1）
3. Prefill服务通过ZMQ发送"PUT"命令：
   - 包含tensor_id、shape、dtype等元数据
   - Decode服务接收后，根据shape和dtype分配GPU内存
4. Prefill服务通过NCCL发送实际的KV缓存数据：
   - 调用ncclSend()，将GPU内存中的KV缓存直接发送
   - Decode服务调用ncclRecv()，直接接收到GPU内存
   - 这是GPU到GPU的直接传输，非常高效
5. 后续的KV缓存传输可以重用已建立的NCCL连接

为什么需要ZMQ + NCCL的组合？

- ZMQ负责"控制流"：轻量级、灵活、支持动态连接
- NCCL负责"数据流"：高性能、GPU直接通信、适合大数据传输
- 分离控制流和数据流是分布式系统的常见设计模式
- 类似于HTTP（控制）+ TCP（数据）的分层设计

在这个代理服务器中的体现：

- 代理服务器生成包含ZMQ地址的request_id
- request_id通过HTTP头传递给Prefill和Decode服务
- 两个服务解析request_id，获取对方的ZMQ地址
- 然后通过ZMQ+NCCL建立连接并传输KV缓存
- 代理服务器本身不参与KV缓存的传输，只负责协调
"""

# ==================== 导入标准库模块 ====================
import os  # 用于访问操作系统相关的功能，比如环境变量
import socket  # 用于网络通信，获取主机名等
import threading  # Python的多线程模块，用于创建和管理线程
import time  # 用于获取当前时间戳，用于判断服务实例是否过期
import uuid  # 用于生成唯一标识符（UUID）
from typing import Any  # 类型注解，Any表示可以是任何类型

# ==================== 导入第三方库模块 ====================
import aiohttp  # 异步HTTP客户端库，用于发送HTTP请求（支持异步操作，不会阻塞）
import msgpack  # 用于序列化和反序列化数据（比JSON更高效的二进制格式）
import zmq  # ZeroMQ消息队列库，用于高性能的进程间通信
from quart import Quart, make_response, request  # Quart是异步Web框架，用于处理HTTP请求

# ==================== 全局变量定义 ====================

# 请求计数器，用于轮询（round-robin）负载均衡
# 每次处理请求时递增，用于选择不同的服务实例
count = 0

# Prefill服务实例字典
# 键（key）：HTTP地址，格式为 "IP:端口"，例如 "192.168.1.100:8000"
# 值（value）：一个元组 (zmq_address, stamp)
#   - zmq_address: ZMQ通信地址，格式为 "IP:端口"
#   - stamp: 时间戳，表示该服务实例的过期时间（当前时间 + 默认心跳间隔）
# 例如：{"192.168.1.100:8000": ("192.168.1.100:30000", 1234567890.5)}
prefill_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)

# Decode服务实例字典，格式与prefill_instances相同
decode_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)

# ==================== 多线程同步原语 ====================
# 
# threading.Condition 是Python中的条件变量，用于多线程之间的同步
# 
# 为什么需要条件变量？
# - 多个线程可能同时访问和修改 prefill_instances 和 decode_instances 字典
# - 如果不加锁，可能会出现数据竞争（race condition），导致数据不一致
# 
# Condition的作用：
# 1. 提供互斥锁（mutex lock）：确保同一时间只有一个线程能访问共享数据
# 2. 提供等待/通知机制：线程可以等待某个条件满足，其他线程可以通知等待的线程
# 
# 使用方式：
#   with prefill_cv:  # 获取锁
#       # 在这里安全地访问和修改 prefill_instances
#   # 离开with块时自动释放锁
# 
# 在这个程序中：
# - prefill_cv: 保护 prefill_instances 字典的锁
# - decode_cv: 保护 decode_instances 字典的锁
prefill_cv = threading.Condition()  # Prefill实例字典的线程锁
decode_cv = threading.Condition()  # Decode实例字典的线程锁

# 默认的心跳超时时间（秒）
# 如果服务实例在5秒内没有发送心跳消息，将被认为已失效并移除
DEFAULT_PING_SECONDS = 5


def _remove_oldest_instances(instances: dict[str, Any]) -> None:
    """
    移除已过期的服务实例（内部辅助函数）
    
    这个函数会检查字典中的服务实例，如果某个实例的时间戳（stamp）已经过期
    （小于当前时间），就将其从字典中移除。
    
    参数：
        instances: 要清理的实例字典（prefill_instances 或 decode_instances）
    
    工作原理：
    1. 从字典的第一个键开始遍历（字典在Python 3.7+中保持插入顺序）
    2. 检查每个实例的时间戳（value[1]）
    3. 如果时间戳小于当前时间，说明该实例已经超时，需要移除
    4. 如果遇到一个未过期的实例，就停止遍历（因为字典是有序的，后面的也不会过期）
    
    注意：
    - 这个函数假设调用它的代码已经获取了相应的锁（prefill_cv 或 decode_cv）
    - 函数名前的下划线 _ 表示这是一个内部函数，不应该在模块外部直接调用
    """
    # next(iter(instances), None) 的含义：
    # - iter(instances): 将字典转换为迭代器，可以逐个访问键
    # - next(..., None): 获取迭代器的第一个元素，如果字典为空则返回None
    # 这相当于获取字典的第一个键
    oldest_key = next(iter(instances), None)
    
    # 循环处理，直到没有更多过期的实例
    while oldest_key is not None:
        # 获取该键对应的值（元组：(zmq_address, stamp)）
        value = instances[oldest_key]
        
        # value[1] 是时间戳（stamp），如果大于当前时间，说明还没过期
        # 由于字典是有序的，如果这个实例没过期，后面的也不会过期，可以提前退出
        if value[1] > time.time():
            break
        
        # 打印移除信息，方便调试和监控
        print(f"🔴Remove [HTTP:{oldest_key}, ZMQ:{value[0]}, stamp:{value[1]}]")
        
        # 从字典中移除这个过期的实例
        # pop(key, None) 表示如果键存在就删除并返回值，不存在就返回None（不会报错）
        instances.pop(oldest_key, None)
        
        # 获取下一个键，继续检查
        oldest_key = next(iter(instances), None)


def _listen_for_register(poller, router_socket):
    """
    监听服务注册消息的函数（在独立线程中运行）
    
    这个函数会在一个单独的线程中持续运行，监听来自prefill和decode服务的注册消息。
    当服务启动时，它们会向代理服务器发送注册消息，告知自己的地址信息。
    
    参数：
        poller: ZMQ的轮询器（Poller），用于检查socket是否有消息到达
        router_socket: ZMQ的ROUTER类型socket，用于接收消息
    
    工作流程：
    1. 持续循环，等待消息
    2. 当收到消息时，解析消息内容
    3. 根据消息类型（"P"表示Prefill，"D"表示Decode），更新相应的实例字典
    4. 更新实例的时间戳，表示该实例仍然活跃
    5. 清理过期的实例
    
    多线程说明：
    - 这个函数会在一个独立的线程中运行（由start_service_discovery函数启动）
    - 主线程处理HTTP请求，这个线程处理服务注册
    - 两个线程通过共享的字典（prefill_instances, decode_instances）和锁（prefill_cv, decode_cv）进行通信
    """
    # 无限循环，持续监听消息
    while True:
        # poller.poll() 会阻塞等待，直到有socket收到消息
        # 返回一个字典，键是收到消息的socket，值是事件类型
        # 例如：{router_socket: zmq.POLLIN} 表示router_socket有消息可读
        socks = dict(poller.poll())
        
        # 检查router_socket是否收到了消息
        if router_socket in socks:
            # recv_multipart() 接收多部分消息
            # ZMQ的ROUTER socket会自动在消息前添加发送者的地址
            # 所以返回两个部分：[发送者地址, 实际消息内容]
            remote_address, message = router_socket.recv_multipart()
            
            # 消息格式说明：
            # data: {"type": "P", "http_address": "ip:port",
            #        "zmq_address": "ip:port"}
            # type: "P" 表示Prefill服务（kv_producer），"D" 表示Decode服务（kv_consumer）
            # http_address: 服务的HTTP地址，用于接收HTTP请求（例如 "10.0.1.2:8000"）
            # zmq_address: 服务的ZMQ地址，用于KV缓存传输的控制流通信（例如 "10.0.1.2:21001"）
            # 
            # 注意：ZMQ地址用于：
            # - 建立NCCL连接的控制流（发送NEW命令）
            # - 传输KV缓存的元数据（tensor_id, shape, dtype）
            # - 协调Prefill和Decode服务之间的通信
            # 
            # 但实际的KV缓存数据（GB级别的张量）是通过NCCL直接传输的，不是ZMQ！
            # ZMQ只负责"控制流"，NCCL负责"数据流"
            
            # msgpack.loads() 将二进制消息反序列化为Python字典
            data = msgpack.loads(message)
            
            # 处理Prefill服务的注册消息
            if data["type"] == "P":
                # 使用global关键字声明要修改全局变量
                # 在Python中，如果要在函数内修改全局变量，必须先用global声明
                global prefill_instances
                global prefill_cv
                
                # 获取锁，确保线程安全地修改prefill_instances
                with prefill_cv:
                    # 检查这个地址是否已经存在
                    # get(key, default) 如果键存在返回对应的值，不存在返回default（这里是None）
                    node = prefill_instances.get(data["http_address"], None)
                    
                    # 更新或添加实例信息
                    # 值是一个元组：(zmq_address, 过期时间戳)
                    # time.time() + DEFAULT_PING_SECONDS 表示当前时间 + 5秒
                    prefill_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    
                    # 清理过期的实例
                    _remove_oldest_instances(prefill_instances)
                    # 离开with块时自动释放锁

            # 处理Decode服务的注册消息（逻辑与Prefill相同）
            elif data["type"] == "D":
                global decode_instances
                global decode_cv
                with decode_cv:
                    node = decode_instances.get(data["http_address"], None)
                    decode_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(decode_instances)
            else:
                # 收到未知类型的消息，打印错误信息并退出循环
                print(
                    "Unexpected, Received message from %s, data: %s",
                    remote_address,
                    data,
                )
                return

            # 如果node是None，说明这是一个新注册的实例（之前不存在）
            # 打印添加信息，方便调试和监控
            if node is None:
                print(f"🔵Add [HTTP:{data['http_address']}, ZMQ:{data['zmq_address']}]")


def start_service_discovery(hostname, port):
    """
    启动服务发现功能
    
    这个函数会：
    1. 创建一个ZMQ ROUTER socket来接收服务注册消息
    2. 创建一个独立的线程来监听和处理这些消息
    3. 返回创建的线程对象
    
    参数：
        hostname: 绑定的主机名或IP地址
                 - 如果为空字符串或None，则使用本机主机名
                 - "0.0.0.0" 表示监听所有网络接口
        port: 绑定的端口号（不能为0）
    
    返回值：
        threading.Thread对象，表示创建的监听线程
    
    多线程说明：
    - 这个函数会创建一个"守护线程"（daemon thread）
    - 守护线程的特点：当主程序退出时，守护线程会自动终止
    - 如果不设置为守护线程，主程序会等待所有线程结束才退出
    """
    # 如果hostname为空，使用本机的主机名
    # socket.gethostname() 返回当前计算机的主机名
    if not hostname:
        hostname = socket.gethostname()
    
    # 端口号不能为0，因为0是无效端口
    if port == 0:
        raise ValueError("Port cannot be 0")

    # 创建ZMQ上下文（Context）
    # Context是ZMQ的核心对象，管理所有的socket和资源
    context = zmq.Context()
    
    # 创建ROUTER类型的socket
    # ROUTER socket的特点：
    # - 可以接收来自多个客户端的消息
    # - 自动在接收到的消息前添加发送者的地址
    # - 适合作为服务器端使用
    router_socket = context.socket(zmq.ROUTER)
    
    # 绑定socket到指定的地址和端口
    # bind() 表示这个socket作为服务器，等待客户端连接
    # f"tcp://{hostname}:{port}" 是ZMQ的地址格式
    # 例如：tcp://0.0.0.0:30001 表示监听所有网络接口的30001端口
    router_socket.bind(f"tcp://{hostname}:{port}")

    # 创建ZMQ轮询器（Poller）
    # Poller用于高效地检查多个socket是否有消息到达
    # 比直接调用recv()更高效，因为可以同时监听多个socket
    poller = zmq.Poller()
    
    # 将router_socket注册到poller中
    # zmq.POLLIN 表示监听是否有消息可读（输入）
    poller.register(router_socket, zmq.POLLIN)

    # 创建线程对象
    # threading.Thread 的参数说明：
    #   target: 线程要执行的函数（_listen_for_register）
    #   args: 传递给target函数的参数列表 [poller, router_socket]
    #   daemon=True: 设置为守护线程
    #      - 守护线程：主程序退出时自动终止
    #      - 非守护线程：主程序会等待线程结束才退出
    _listener_thread = threading.Thread(
        target=_listen_for_register, args=[poller, router_socket], daemon=True
    )
    
    # 启动线程
    # start() 会调用target函数，但不会等待函数执行完成
    # 函数会在新线程中并行执行
    _listener_thread.start()
    
    # 返回线程对象，调用者可以选择等待线程结束（使用join()）
    return _listener_thread


# ==================== HTTP客户端配置 ====================

# aiohttp的客户端超时设置
# ClientTimeout 用于设置HTTP请求的超时时间
# total=6 * 60 * 60 表示总超时时间为6小时（21600秒）
# 这个超时时间很长，因为LLM生成可能需要很长时间
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

# ==================== Web应用初始化 ====================

# 创建Quart应用实例
# Quart是一个异步Web框架（类似于Flask，但支持异步）
# __name__ 是Python的内置变量，表示当前模块的名称
# 用于定位资源文件（如模板、静态文件等）
app = Quart(__name__)


def random_uuid() -> str:
    """
    生成随机UUID字符串
    
    UUID（Universally Unique Identifier）是通用唯一标识符
    用于生成全局唯一的ID，确保每个请求都有唯一的标识
    
    返回值：
        str: 32位的十六进制字符串（不包含连字符）
        例如："a1b2c3d4e5f6789012345678901234ab"
    
    用途：
        在这个程序中，用于生成请求ID，便于追踪和调试
    """
    # uuid.uuid4() 生成一个随机UUID（版本4，基于随机数）
    # .hex 属性返回UUID的十六进制表示（不包含连字符）
    # str() 转换为字符串（虽然.hex已经是字符串，但为了类型明确）
    return str(uuid.uuid4().hex)


async def forward_request(url, data, request_id):
    """
    异步转发HTTP请求的函数（生成器函数）
    
    这个函数使用aiohttp发送HTTP POST请求，并以流式方式返回响应数据。
    使用生成器（generator）的方式，可以边接收边返回数据，不需要等待完整响应。
    
    参数：
        url: 目标服务的URL地址，例如 "http://192.168.1.100:8000/v1/chat/completions"
        data: 要发送的JSON数据（字典格式）
        request_id: 请求的唯一标识符，用于追踪和调试
    
    返回值：
        这是一个生成器函数（使用yield），返回一个异步生成器
        每次yield返回一个数据块（chunk_bytes），大小为1024字节
    
    异步编程说明：
    - async def: 定义异步函数，可以在函数内使用await
    - async with: 异步上下文管理器，确保资源正确释放
    - async for: 异步迭代，用于处理流式数据
    - yield: 生成器关键字，每次产生一个值后暂停，等待下次调用
    
    为什么使用异步？
    - 传统同步请求会阻塞线程，等待响应返回
    - 异步请求可以在等待响应时处理其他任务，提高并发性能
    - 对于代理服务器，需要同时处理多个请求，异步是更好的选择
    """
    # 创建aiohttp的客户端会话（ClientSession）
    # async with 是异步上下文管理器，确保会话在使用完后正确关闭
    # timeout=AIOHTTP_TIMEOUT 设置请求超时时间
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        # 设置HTTP请求头
        headers = {
            # Authorization头：用于身份验证
            # os.environ.get('OPENAI_API_KEY') 从环境变量获取API密钥
            # 如果环境变量不存在，返回None
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            # 自定义请求ID头，用于追踪请求
            # ⚠️ 关键：这个request_id会被Prefill和Decode服务解析
            # 用于建立ZMQ+NCCL连接，实现KV缓存的传输
            # Prefill服务会从request_id中提取Decode的ZMQ地址
            # Decode服务会从request_id中提取Prefill的ZMQ地址
            "X-Request-Id": request_id,
        }
        
        # 发送POST请求
        # async with session.post() 也是异步上下文管理器
        # json=data 会自动将字典序列化为JSON并设置Content-Type头
        # headers=headers 设置请求头
        async with session.post(url=url, json=data, headers=headers) as response:
            # 检查响应状态码
            if response.status == 200:
                # 当前代码中 if True 总是为真，所以总是使用流式读取
                # 这可能是为了调试，可以切换两种读取方式
                if True:
                    # 流式读取响应内容
                    # iter_chunked(1024) 每次读取1024字节
                    # async for 异步迭代，每次循环获取一个数据块
                    # yield 将数据块返回给调用者
                    # 这种方式的好处：
                    #   1. 不需要等待完整响应，可以边接收边返回
                    #   2. 内存占用小，不会一次性加载整个响应
                    #   3. 响应速度快，客户端可以立即开始处理数据
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    # 一次性读取完整响应（当前代码不会执行到这里）
                    # await response.read() 等待读取完整的响应内容
                    # 这种方式会等待所有数据接收完成才返回
                    content = await response.read()
                    yield content


# ==================== HTTP路由处理 ====================

# @app.route 是装饰器（decorator），用于注册URL路由
# 当客户端访问指定的URL时，会调用被装饰的函数
# 
# 这里注册了两个路由：
#   - "/v1/completions": OpenAI API的文本补全接口
#   - "/v1/chat/completions": OpenAI API的聊天补全接口
# 
# methods=["POST"] 表示只接受POST请求
# 
# 装饰器的作用：
#   在Python中，装饰器是一种语法糖，用于在不修改函数的情况下扩展功能
#   @app.route(...) 相当于：handle_request = app.route(...)(handle_request)
@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    """
    处理客户端请求的主函数
    
    这个函数实现了请求的分离处理：
    1. 首先将请求转发到Prefill服务，完成预填充（只生成第一个token）
    2. 然后将原始请求转发到Decode服务，完成后续的token生成
    3. 将Decode服务的响应流式返回给客户端
    
    工作流程：
    1. 接收客户端的HTTP请求
    2. 创建两个请求副本：
       - prefill_request: 修改max_tokens=1，只做预填充
       - original_request_data: 保持原样，用于decode阶段
    3. 从prefill_instances中选择一个prefill服务实例（轮询）
    4. 从decode_instances中选择一个decode服务实例（轮询）
    5. 先向prefill服务发送请求并等待完成
    6. 再向decode服务发送请求，并将响应流式返回给客户端
    
    返回值：
        HTTP响应对象，包含流式数据
    
    异常处理：
        如果发生任何错误，会捕获异常并返回错误信息
    """
    try:
        # 获取客户端发送的JSON数据
        # await 表示等待异步操作完成
        # request.get_json() 是异步函数，需要等待解析完成
        # 这会将HTTP请求体中的JSON字符串解析为Python字典
        original_request_data = await request.get_json()

        # ========== 步骤1：准备Prefill请求 ==========
        
        # 创建原始请求的副本
        # copy() 是字典的浅拷贝方法，创建一个新的字典，包含相同的键值对
        # 这样修改prefill_request不会影响original_request_data
        prefill_request = original_request_data.copy()
        
        # 修改max_tokens为1，让prefill服务只生成第一个token
        # 这是分离架构的关键：prefill阶段只做预填充，不生成完整响应
        # change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1
        
        # 如果请求中有max_completion_tokens字段，也设置为1
        # max_completion_tokens是另一种限制生成token数量的方式
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        # ========== 步骤2：选择Prefill服务实例（负载均衡） ==========
        
        # 声明要使用全局变量
        global count
        global prefill_instances
        global prefill_cv
        
        # 获取锁，安全地访问prefill_instances
        with prefill_cv:
            # items() 返回字典的所有键值对，格式为 [(key, value), ...]
            # list() 转换为列表，方便索引访问
            prefill_list = list(prefill_instances.items())
            
            # 检查是否有可用的prefill实例
            if len(prefill_list) == 0:
                # 返回HTTP 503错误（服务不可用）
                # 格式：(响应数据, 状态码)
                return {"error": "No prefill instances available"}, 503
            
            # 轮询（Round-Robin）负载均衡算法
            # count % len(prefill_list) 计算索引：
            #   - count是全局计数器，每次请求递增
            #   - % 是取模运算符，确保索引在有效范围内
            #   - 例如：3个实例，count=5，则 5 % 3 = 2，选择第3个实例
            # 
            # prefill_list[count % len(prefill_list)] 返回一个元组：
            #   (http_address, (zmq_address, stamp))
            # 
            # 解包赋值：
            #   prefill_addr = http_address（字符串）
            #   prefill_zmq_addr = (zmq_address, stamp)（元组）
            prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
            
            # 从元组中提取zmq_address（第一个元素）
            prefill_zmq_addr = prefill_zmq_addr[0]
            # 离开with块时自动释放锁

        # ========== 步骤3：选择Decode服务实例（负载均衡） ==========
        
        # 逻辑与选择prefill实例相同
        global decode_instances
        global decode_cv
        with decode_cv:
            decode_list = list(decode_instances.items())
            if len(decode_list) == 0:
                return {"error": "No decode instances available"}, 503
            # 使用相同的count进行轮询，确保prefill和decode使用相同的索引
            decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
            decode_zmq_addr = decode_zmq_addr[0]

        # 打印请求信息，方便调试和监控
        # f"..." 是f-string格式化字符串，可以在字符串中嵌入变量
        print(
            f"handle_request count: {count}, [HTTP:{prefill_addr}, "
            f"ZMQ:{prefill_zmq_addr}] 👉 [HTTP:{decode_addr}, "
            f"ZMQ:{decode_zmq_addr}]"
        )
        
        # 递增计数器，下次请求会选择下一个实例
        count += 1

        # ========== 步骤4：生成请求ID（KV缓存传输的关键） ==========
        
        # 生成唯一的请求ID，包含prefill和decode的ZMQ地址信息
        # 这个ID是KV缓存传输的"桥梁"，非常重要！
        # 
        # request_id的格式示例：
        #   "___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_a1b2c3d4..."
        # 
        # 这个ID的作用：
        # 1. 通过HTTP头传递给Prefill和Decode服务
        # 2. Prefill服务解析request_id，提取Decode的ZMQ地址（10.0.1.3:22001）
        #    然后通过ZMQ+NCCL将KV缓存发送到Decode服务
        # 3. Decode服务解析request_id，提取Prefill的ZMQ地址（10.0.1.2:21001）
        #    然后通过ZMQ+NCCL接收KV缓存
        # 
        # 为什么需要包含地址信息？
        # - 支持动态服务发现：不需要预先配置所有服务对之间的连接
        # - 每个请求可以路由到不同的Prefill/Decode组合
        # - 实现点对点（P2P）通信，不依赖中央协调器
        # 
        # ZMQ地址 vs HTTP地址：
        # - HTTP地址：用于接收HTTP请求（例如 10.0.1.2:8000）
        # - ZMQ地址：用于KV缓存传输的控制流和元数据通信（例如 10.0.1.2:21001）
        #   注意：实际的KV缓存数据通过NCCL传输，不是ZMQ！
        request_id = (
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
            f"{decode_zmq_addr}_{random_uuid()}"
        )

        # ========== 步骤5：完成Prefill阶段（KV缓存传输的关键时刻） ==========
        
        # 向prefill服务发送请求
        # forward_request() 返回一个异步生成器，会产生数据块
        # async for _ in ... 表示遍历生成器的所有值，但不使用这些值（用_表示）
        # continue 表示继续下一次循环
        # 
        # 这个循环的作用：
        #   1. 等待prefill请求完成
        #   2. 消费所有返回的数据（虽然我们不需要这些数据）
        #   3. 确保prefill阶段完全结束后再进行decode阶段
        # 
        # ⚠️ 重要：在Prefill服务处理请求期间，会发生KV缓存传输！
        # 
        # KV缓存传输的详细过程（发生在Prefill服务内部，不在这个代理中）：
        # 
        # 1. Prefill服务接收HTTP请求，解析X-Request-Id头获取request_id
        # 2. Prefill服务从request_id中解析Decode服务的ZMQ地址
        #    例如：从 "___prefill_addr_10.0.1.2:21001___decode_addr_10.0.1.3:22001_..."
        #         中提取 "10.0.1.3:22001"
        # 
        # 3. Prefill服务完成预填充计算，生成KV缓存（存储在GPU内存中）
        # 
        # 4. Prefill服务通过ZMQ发送控制消息给Decode服务：
        #    a) 如果是首次连接，发送"NEW"命令，建立NCCL连接
        #       - 通过ZMQ发送NCCL的unique_id（用于初始化NCCL通信器）
        #       - Decode服务接收后，使用相同的unique_id初始化NCCL
        #       - 建立点对点NCCL通信组（world_size=2，只有两个rank）
        #    b) 发送"PUT"命令，包含KV缓存的元数据：
        #       - tensor_id（唯一标识这个KV缓存）
        #       - shape（张量形状）
        #       - dtype（数据类型）
        # 
        # 5. Prefill服务通过NCCL发送实际的KV缓存数据：
        #    - 使用已建立的NCCL通信器（comm）
        #    - 调用ncclSend()，将GPU内存中的KV缓存直接发送到Decode服务的GPU
        #    - 这是GPU到GPU的直接通信，非常高效（不经过CPU和网络栈）
        # 
        # 6. Decode服务通过ZMQ接收控制消息：
        #    - 接收"NEW"命令，初始化NCCL通信器
        #    - 接收"PUT"命令，根据元数据分配GPU内存
        # 
        # 7. Decode服务通过NCCL接收KV缓存数据：
        #    - 调用ncclRecv()，将KV缓存直接接收到GPU内存
        #    - 如果GPU缓冲区满了，存储到Tensor内存池（系统内存）
        # 
        # ZMQ和NCCL的分工：
        # - ZMQ（ZeroMQ）：负责控制流和元数据传输
        #   * 建立连接（NEW命令）
        #   * 传输元数据（tensor_id, shape, dtype）
        #   * 协调NCCL连接的建立
        #   * 使用TCP/IP网络，可以跨机器通信
        # 
        # - NCCL（NVIDIA Collective Communications Library）：负责实际数据传输
        #   * 传输实际的KV缓存张量数据（通常是GB级别的数据）
        #   * GPU到GPU的直接通信，绕过CPU和网络栈
        #   * 使用InfiniBand、NVLink或以太网（取决于硬件）
        #   * 性能极高，适合传输大块数据
        # 
        # 为什么需要ZMQ + NCCL的组合？
        # - ZMQ用于控制流：轻量级，灵活，支持动态连接
        # - NCCL用于数据传输：高性能，GPU直接通信，适合大数据传输
        # - 分离控制流和数据流是分布式系统的常见设计模式
        # 
        # f"http://{prefill_addr}{request.path}" 构建完整的URL
        #   - prefill_addr: 例如 "192.168.1.100:8000"
        #   - request.path: 例如 "/v1/chat/completions"
        #   组合后: "http://192.168.1.100:8000/v1/chat/completions"
        async for _ in forward_request(
            f"http://{prefill_addr}{request.path}", prefill_request, request_id
        ):
            continue

        # ========== 步骤6：启动Decode阶段并返回响应 ==========
        
        # 向decode服务发送原始请求（包含完整的max_tokens）
        # forward_request() 返回一个异步生成器
        # 
        # ⚠️ 重要：此时KV缓存应该已经传输完成！
        # 
        # Decode服务的工作流程：
        # 1. 接收HTTP请求，解析X-Request-Id头获取request_id
        # 2. 从request_id中解析Prefill服务的ZMQ地址（如果需要主动获取KV缓存，GET模式）
        # 3. 从GPU缓冲区或Tensor内存池中查找对应的KV缓存（使用request_id作为key）
        # 4. 如果找到KV缓存，直接使用，跳过Prefill阶段（这是分离架构的核心优势）
        # 5. 如果没找到，可能需要重新计算Prefill（性能会下降）
        # 6. 使用KV缓存进行Decode阶段，生成后续的tokens
        # 7. 流式返回生成的tokens给客户端
        # 
        # 注意：Decode服务在接收到HTTP请求时，KV缓存应该已经通过NCCL传输完成
        # （在Prefill阶段完成后，通过PUT_ASYNC模式异步传输）
        generator = forward_request(
            f"http://{decode_addr}{request.path}", original_request_data, request_id
        )
        
        # make_response() 将生成器转换为HTTP响应对象
        # await 等待响应对象创建完成
        # 这个响应对象会流式返回生成器的数据
        response = await make_response(generator)
        
        # 设置响应超时为None（无限制）
        # 因为LLM生成可能需要很长时间
        response.timeout = None

        # 返回响应给客户端
        # Quart会自动处理流式响应，将生成器的数据逐步发送给客户端
        return response

    # ========== 异常处理 ==========
    
    # except Exception as e: 捕获所有类型的异常
    # 如果try块中的代码发生任何错误，都会跳到这里处理
    except Exception as e:
        # 导入异常处理相关的模块
        import sys
        import traceback

        # sys.exc_info() 获取当前异常的信息（类型、值、追踪信息）
        exc_info = sys.exc_info()
        
        # 打印错误信息，方便调试
        print("Error occurred in disagg prefill proxy server")
        print(e)  # 打印异常对象本身（通常是错误消息）
        
        # traceback.format_exception() 格式化完整的异常堆栈信息
        # "".join() 将多行信息连接成一个字符串
        # 这会打印出完整的错误堆栈，包括错误发生的文件、行号、函数调用链等
        print("".join(traceback.format_exception(*exc_info)))


# ==================== 程序入口 ====================

# if __name__ == "__main__": 是Python的惯用法
# 
# 含义说明：
#   - __name__ 是Python的内置变量，表示当前模块的名称
#   - 当直接运行这个文件时，__name__ 的值是 "__main__"
#   - 当这个文件被其他文件导入时，__name__ 的值是文件名（不含.py）
# 
# 作用：
#   这段代码只有在直接运行这个文件时才会执行
#   如果被其他文件导入，不会执行这段代码
#   这样可以区分"作为脚本运行"和"作为模块导入"两种情况
if __name__ == "__main__":
    # ========== 读取配置 ==========
    
    # 从环境变量读取配置，如果没有设置则使用默认值
    # os.environ.get(key, default) 获取环境变量的值，不存在则返回default
    # int() 将字符串转换为整数
    # 
    # 环境变量的设置方式（在运行程序前）：
    #   export ZMQ_PORT=30001
    #   export HTTP_PORT=8000
    #   或者在命令行：ZMQ_PORT=30001 HTTP_PORT=8000 python disagg_proxy_p2p_nccl_xpyd.py
    zmq_port = int(os.environ.get("ZMQ_PORT", "30001"))  # ZMQ服务发现端口，默认30001
    http_port = int(os.environ.get("HTTP_PORT", "8000"))  # HTTP服务端口，默认8000
    
    # ========== 启动服务发现 ==========
    
    # 启动服务发现功能，创建一个监听线程
    # "0.0.0.0" 表示监听所有网络接口（允许来自任何IP的连接）
    # zmq_port 是ZMQ socket绑定的端口
    # 返回创建的线程对象
    t = start_service_discovery("0.0.0.0", zmq_port)
    
    # ========== 启动Web服务器 ==========
    
    # 启动Quart Web服务器
    # host="0.0.0.0" 表示监听所有网络接口
    # port=http_port 是HTTP服务监听的端口
    # 
    # app.run() 会阻塞主线程，持续运行直到程序被终止
    # 这意味着这行代码之后的代码（t.join()）实际上不会被执行
    # 因为程序会一直在这里运行，处理HTTP请求
    app.run(host="0.0.0.0", port=http_port)
    
    # 这行代码实际上不会被执行（因为app.run()会一直运行）
    # 但如果app.run()因为某种原因退出，这里会等待监听线程结束
    # t.join() 会阻塞当前线程，直到线程t执行完成
    # 由于t是守护线程，主程序退出时会自动终止，所以这里主要是为了代码完整性
    t.join()

