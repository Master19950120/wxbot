import logging
import random
from datetime import datetime
import threading
import time
import os
from database import Session, ChatMessage
from config import (
    DEEPSEEK_API_KEY, MAX_TOKEN, ROBOT_WX_NAME, TEMPERATURE, MODEL, DEEPSEEK_BASE_URL, LISTEN_LIST,
    IMAGE_MODEL, IMAGE_SIZE, BATCH_SIZE, GUIDANCE_SCALE, NUM_INFERENCE_STEPS, PROMPT_ENHANCEMENT,
    TEMP_IMAGE_DIR, MAX_GROUPS, PROMPT_NAME, IMAGE_URL, SEND_LIST, IMAGE_API_KEY
)
from wxauto import WeChat
from openai import OpenAI
import requests
from typing import Optional
import re
import schedule  # 导入schedule库

# 获取微信窗口对象
wx = WeChat()

# 设置监听列表
listen_list = LISTEN_LIST

# 设置发送列表
send_list = SEND_LIST

# 循环添加监听对象，移除savepic=True参数
for i in listen_list:
    wx.AddListenChat(who=i)  # 移除 savepic=True

# 修改等待时间为更短的间隔
wait = 0.5  # 从1秒改为0.5秒

# 初始化OpenAI客户端（替换原有requests方式）
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    default_headers={"Content-Type": "application/json"}  # 添加默认请求头
)

# 获取程序根目录
root_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(root_dir, "prompts", PROMPT_NAME)

# 新增全局变量
user_queues = {}  # 用户消息队列管理
queue_lock = threading.Lock()  # 队列访问锁
chat_contexts = {}  # 存储上下文

# 读取文件内容到变量
with open(file_path, "r", encoding="utf-8") as file:
    prompt_content = file.read()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加临时目录初始化
temp_dir = os.path.join(root_dir, TEMP_IMAGE_DIR)
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


# 保存聊天记录到数据库
def save_message(sender_id, sender_name, message, reply):
    """
    将聊天消息及其回复保存到数据库中。

    参数:
    sender_id (str): 发送者的唯一标识符。
    sender_name (str): 发送者的名称。
    message (str): 发送者发送的消息内容。
    reply (str): 机器人的回复内容。

    过程:
    1. 创建一个新的数据库会话。
    2. 使用提供的参数创建一个新的 ChatMessage 对象。
    3. 将 ChatMessage 对象添加到数据库会话中。
    4. 提交会话以保存更改。
    5. 关闭数据库会话以释放资源。

    异常处理:
    如果在保存过程中发生任何异常，将捕获异常并打印错误信息。
    """
    try:
        session = Session()  # 创建一个新的数据库会话
        chat_message = ChatMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            message=message,
            reply=reply
        )
        session.add(chat_message)  # 将 ChatMessage 对象添加到会话中
        session.commit()  # 提交会话以保存更改
        session.close()  # 关闭会话以释放资源
    except Exception as e:
        print(f"保存消息失败: {str(e)}")  # 打印错误信息


# 调用API生成图像
def generate_image(prompt: str) -> Optional[str]:
    try:
        headers = {
            "Authorization": f"Bearer {IMAGE_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": IMAGE_MODEL,
            "prompt": prompt,
            # "batch_size": BATCH_SIZE,
            # "guidance_scale": GUIDANCE_SCALE,
            "size": IMAGE_SIZE,
            # "num_inference_steps": NUM_INFERENCE_STEPS,
            # "prompt_enhancement": PROMPT_ENHANCEMENT
        }

        # 直接使用 requests 发送请求到图像生成API
        image_api_url = f"{IMAGE_URL}"
        response = requests.post(image_api_url, headers=headers, json=data)
        response.raise_for_status()

        result = response.json()
        if "data" in result and len(result["data"]) > 0:  # 优先检查 data 字段
            return result["data"][0]["url"]
        elif "images" in result and len(result["images"]) > 0:
            return result["images"][0]["url"]
        return None

    except Exception as e:
        logger.error(f"图像生成失败: {str(e)}")
        return None


# 判断是否为图像生成请求
def is_image_generation_request(text: str) -> bool:
    # 基础动词
    draw_verbs = ["画", "绘", "生成", "创建", "做"]

    # 图像相关词
    image_nouns = ["图", "图片", "画", "照片", "插画", "像"]

    # 数量词
    quantity = ["一下", "一个", "一张", "个", "张", "幅"]

    # 组合模式
    patterns = [
        # 直接画xxx模式
        r"画.*[猫狗人物花草山水]",
        r"画.*[一个张只条串份副幅]",
        # 帮我画xxx模式
        r"帮.*画.*",
        r"给.*画.*",
        # 生成xxx图片模式
        r"生成.*图",
        r"创建.*图",
        # 能不能画xxx模式
        r"能.*画.*吗",
        r"可以.*画.*吗",
        # 想要xxx图模式
        r"要.*[张个幅].*图",
        r"想要.*图",
        # 其他常见模式
        r"做[一个张]*.*图",
        r"画画",
        r"画一画",
    ]

    # 1. 检查正则表达式模式
    if any(re.search(pattern, text) for pattern in patterns):
        return True

    # 2. 检查动词+名词组合
    for verb in draw_verbs:
        for noun in image_nouns:
            if f"{verb}{noun}" in text:
                return True
            # 检查带数量词的组合
            for q in quantity:
                if f"{verb}{q}{noun}" in text:
                    return True
                if f"{verb}{noun}{q}" in text:
                    return True

    # 3. 检查特定短语
    special_phrases = [
        "帮我画", "给我画", "帮画", "给画",
        "能画吗", "可以画吗", "会画吗",
        "想要图", "要图", "需要图",
    ]

    if any(phrase in text for phrase in special_phrases):
        return True

    return False


# 获取DeepSeek API回复
def get_deepseek_response(message, user_id):
    try:
        # 检查是否为图像生成请求
        if is_image_generation_request(message):
            image_url = generate_image(message)
            if image_url:
                # 下载图片并保存到临时文件
                img_response = requests.get(image_url)
                if img_response.status_code == 200:
                    # 使用时间戳创建唯一文件名
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_path = os.path.join(str(temp_dir), f"image_{timestamp}.jpg")
                    with open(temp_path, "wb") as f:
                        f.write(img_response.content)
                    return f"[IMAGE]{temp_path}[/IMAGE]\n这是按照您的要求生成的图片\\(^o^)/~"
                else:
                    return "抱歉，图片生成成功但下载失败，请稍后重试。"
            else:
                return "抱歉，图片生成失败，请稍后重试。"

        # 原有的文本处理逻辑
        print(f"调用 DeepSeek API - 用户ID: {user_id}, 消息: {message}")
        with queue_lock:
            if user_id not in chat_contexts:
                chat_contexts[user_id] = []

            chat_contexts[user_id].append({"role": "user", "content": message})

            # 限制上下文的数量，确保不会超过 MAX_GROUPS * 2
            while len(chat_contexts[user_id]) > MAX_GROUPS * 2:
                del chat_contexts[user_id][0]

        # 清理过期的上下文
        with queue_lock:
            if user_id in chat_contexts:
                chat_contexts[user_id] = chat_contexts[user_id][-MAX_GROUPS * 2:]

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt_content},
                    *chat_contexts[user_id][-MAX_GROUPS * 2:]
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKEN,
                stream=False
            )
        except Exception as api_error:
            logger.error(f"API调用失败: {str(api_error)}")
            # 异步清空当前用户Id的最新一条上下文
            threading.Thread(target=clear_latest_user_context, args=(user_id,)).start()
            return "又瞎鸡儿给我乱发什么了 赶紧撤回!"

        if not response.choices:
            logger.error("API返回空choices: %s", response)
            return "抱歉，服务响应异常，请稍后再试"

        reply = response.choices[0].message.content

        with queue_lock:
            chat_contexts[user_id].append({"role": "assistant", "content": reply})

        print(f"API回复: {reply}")
        return reply

    except Exception as e:
        logger.error(f"DeepSeek调用失败: {str(e)}", exc_info=True)
        return "抱歉，刚刚不小心睡着了..."


def process_user_messages(user_id):
    """
    处理来自用户的合并消息，并根据需要发送回复和保存消息记录。

    参数:
    user_id (str): 用户的唯一标识符。

    过程:
    1. 使用队列锁确保线程安全，从用户队列中获取用户数据。
    2. 如果用户ID不在队列中，直接返回。
    3. 从队列中移除用户数据，并提取消息列表、发送者名称和用户名。
    4. 过滤重复消息并保留最近的5条消息，生成合并消息。
    5. 调用 `get_deepseek_response` 函数获取API回复。
    6. 根据API回复的内容类型（图片或文本），执行相应的发送逻辑：
       - 如果回复包含图片标记，发送图片和附加文本消息。
       - 如果回复包含分隔符 '\\', 分割并逐条发送消息。
       - 否则，直接发送回复消息。
    7. 异步保存消息记录到数据库。

    异常处理:
    如果在发送回复或保存消息记录过程中发生任何异常，将捕获异常并打印错误信息。
    """
    with queue_lock:
        if user_id not in user_queues:
            logger.debug(f"用户ID {user_id} 不在队列中，直接返回")
            return
        user_data = user_queues.pop(user_id)  # 从队列中移除用户数据
        messages = user_data['messages']  # 提取消息列表
        sender_name = user_data['sender_name']  # 提取发送者名称
        username = user_data['username']  # 提取用户名

    # 过滤重复消息并保留最近的5条消息
    unique_messages = list(dict.fromkeys(messages))[-5:]  # 保留唯一消息并限制数量
    merged_message = ' \\ '.join(unique_messages)  # 生成合并消息
    logger.info(f"处理合并消息 ({sender_name}): {merged_message}")  # 打印合并消息

    # 获取API回复
    reply = get_deepseek_response(merged_message, user_id)  # 调用API获取回复
    logger.debug(f"API回复: {reply}")

    # 根据API回复的内容类型执行相应的发送逻辑
    try:
        if '[IMAGE]' in reply:
            handle_image_reply(reply, user_id)
        elif '\\' in reply:
            handle_text_reply(reply, user_id)
        else:
            send_text_message(reply, user_id)
    except Exception as e:
        logger.error(f"发送回复失败: {str(e)}")  # 打印错误信息

    # 异步保存消息记录
    threading.Thread(target=save_message, args=(username, sender_name, merged_message, reply)).start()  # 启动线程保存消息


def handle_image_reply(reply, user_id):
    """
    处理包含图片的回复。

    参数:
    reply (str): 包含图片标记的回复。
    user_id (str): 用户的唯一标识符。
    """
    try:
        img_path = reply.split('[IMAGE]')[1].split('[/IMAGE]')[0].strip()  # 提取图片路径
        if os.path.exists(img_path):
            try:
                wx.SendFiles(filepath=img_path, who=user_id)  # 发送图片
                text_msg = reply.split('[/IMAGE]')[1].strip()  # 提取附加文本消息
                if text_msg:
                    send_text_message(text_msg, user_id)  # 发送文本消息
            finally:
                try:
                    os.remove(img_path)  # 删除图片文件
                except Exception as e:
                    logger.error(f"删除临时图片失败: {str(e)}")  # 打印错误信息
    except Exception as e:
        logger.error(f"处理图片回复失败: {str(e)}")


def handle_text_reply(reply, user_id):
    """
    处理包含分隔符 '\\', 分割并逐条发送消息。

    参数:
    reply (str): 包含分隔符 '\\', 分割并逐条发送消息。
    user_id (str): 用户的唯一标识符。
    """
    try:
        parts = [p.strip() for p in reply.split('\\') if p.strip()]  # 分割回复消息
        for part in parts:
            send_text_message(part, user_id)  # 发送每条消息
            time.sleep(random.randint(1, 2))  # 随机延迟1到2秒
    except Exception as e:
        logger.error(f"处理文本回复失败: {str(e)}")


def send_text_message(msg, user_id):
    """
    发送文本消息。

    参数:
    msg (str): 要发送的文本消息。
    user_id (str): 用户的唯一标识符。
    """
    try:
        wx.SendMsg(msg=msg, who=user_id)  # 发送文本消息
    except Exception as e:
        logger.error(f"发送文本消息失败: {str(e)}")



def message_listener():
    """
    持续监听微信消息，并根据消息类型和内容进行处理。

    过程:
    1. 初始化微信客户端对象 `wx` 为 `None`。
    2. 设置 `last_window_check` 为 0，用于记录上一次检查微信窗口状态的时间。
    3. 设置 `check_interval` 为 600 秒，表示每隔600秒检查一次微信窗口状态。
    4. 进入无限循环，持续监听消息。
    5. 获取当前时间 `current_time`。
    6. 如果 `wx` 为 `None` 或者距离上次检查窗口状态的时间超过 `check_interval`，重新初始化微信客户端：
       - 创建新的 `WeChat` 对象。
       - 检查微信会话列表，如果会话列表为空，等待5秒后继续。
       - 更新 `last_window_check` 为当前时间。
    7. 获取监听到的消息 `msgs`。
    8. 如果没有消息，等待 `wait` 秒后继续。
    9. 遍历每个聊天对象 `chat`：
       - 获取聊天对象的 `who` 属性（发送者ID）。
       - 如果 `who` 为空，跳过该聊天对象。
       - 获取该聊天对象的所有消息 `one_msgs`。
       - 如果没有消息，跳过该聊天对象。
       - 遍历每条消息 `msg`：
         - 获取消息类型 `msgtype` 和内容 `content`。
         - 如果消息内容为空，跳过该消息。
         - 如果消息类型不是 'friend'（非好友消息），记录日志并跳过该消息。
         - 如果接收窗口名与发送人相同，表示是私聊，调用 `handle_wxauto_message` 处理私聊信息。
         - 如果消息内容中包含 `@机器人名称`，表示是群聊中@当前机器人的消息，调用 `handle_wxauto_message` 处理群聊信息。
         - 否则，记录日志并跳过该消息。
    10. 捕获处理单条消息时的异常，并记录日志。
    11. 捕获整个监听循环中的异常，并记录日志，重置微信客户端对象 `wx` 为 `None`。
    12. 每次循环结束时，等待 `wait` 秒。

    异常处理:
    如果在检查窗口状态、获取消息或处理消息过程中发生任何异常，将捕获异常并打印错误信息。
    """
    wx = None  # 初始化微信客户端对象为 None
    last_window_check = 0  # 记录上一次检查微信窗口状态的时间
    check_interval = 600  # 每600秒检查一次窗口状态

    while True:
        try:
            current_time = time.time()  # 获取当前时间

            # 只在必要时初始化或重新获取微信窗口，不输出提示
            if wx is None or (current_time - last_window_check > check_interval):
                wx = WeChat()  # 创建新的 WeChat 对象
                if not wx.GetSessionList():
                    time.sleep(5)  # 如果会话列表为空，等待5秒后继续
                    continue
                last_window_check = current_time  # 更新上一次检查时间

            # 获取监听到的消息
            msgs = wx.GetListenMessage()
            if not msgs:
                time.sleep(wait)  # 如果没有消息，等待 wait 秒后继续
                continue

            # 遍历每个聊天对象
            for chat in msgs:
                who = chat.who  # 获取聊天对象的发送者ID
                if not who:
                    continue  # 如果发送者ID为空，跳过该聊天对象

                # 获取该聊天对象的所有消息
                one_msgs = msgs.get(chat)
                if not one_msgs:
                    continue  # 如果没有消息，跳过该聊天对象

                # 遍历每条消息
                for msg in one_msgs:
                    try:
                        msgtype = msg.type  # 获取消息类型
                        content = msg.content  # 获取消息内容
                        if not content:
                            continue  # 如果消息内容为空，跳过该消息

                        # 忽略非好友消息
                        if msgtype != 'friend':
                            logger.debug(f"非好友消息，忽略! 消息类型: {msgtype}")
                            continue

                        # 接收窗口名跟发送人一样，代表是私聊，否则是群聊
                        if who == msg.sender:
                            handle_wxauto_message(msg, msg.sender)  # 处理私聊信息
                        elif ROBOT_WX_NAME != '' and bool(re.search(f'@{ROBOT_WX_NAME}\u2005', msg.content)):
                            handle_wxauto_message(msg, who)  # 处理群聊信息，只有@当前机器人才会处理
                        else:
                            logger.debug(f"非需要处理消息，可能是群聊非@消息: {content}")
                    except Exception as e:
                        logger.debug(f"不好了！处理单条消息失败: {str(e)}")
                        continue  # 捕获处理单条消息时的异常，并记录日志

        except Exception as e:
            logger.debug(f"不好了！消息监听出错: {str(e)}")
            wx = None  # 出错时重置微信对象
        time.sleep(wait)  # 每次循环结束时，等待 wait 秒



# 处理微信消息
def handle_wxauto_message(msg, chatName):
    try:
        # 获取聊天名称（用户或群聊名称）
        username = chatName

        # 从消息对象中提取内容，优先使用 'content' 属性，若不存在则使用 'text' 属性
        content = getattr(msg, 'content', None) or getattr(msg, 'text', None)

        # 过滤掉消息中的特殊标记（如 '@机器人名称'），防止机器名与设定名不一致的问题
        content = content.replace(f'@{ROBOT_WX_NAME}\u2005', '')

        # 如果消息内容为空，记录日志并直接返回，结束函数执行
        if not content:
            logger.debug("不好了！无法获取消息内容")
            return

        # 设置发送者名称为聊天名称
        sender_name = username

        # 获取当前时间，并格式化为 "年-月-日 时:分:秒" 的字符串形式
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 将当前时间与消息内容结合，生成带时间戳的消息内容
        time_aware_content = f"[{current_time}] {content}"

        # 使用线程安全的锁，确保对共享资源的操作是线程安全的
        with queue_lock:
            # 如果用户不在消息队列中，则初始化该用户的队列和定时器
            if username not in user_queues:
                # 创建一个定时器，3秒后触发 process_user_messages 函数
                user_queues[username] = {
                    'timer': threading.Timer(3.0, process_user_messages, args=[username]),
                    'messages': [time_aware_content],  # 初始化消息列表，包含当前消息
                    'sender_name': sender_name,  # 发送者名称
                    'username': username  # 用户名称
                }
                # 启动定时器
                user_queues[username]['timer'].start()

            # 如果用户已经在消息队列中，则更新其队列和定时器
            else:
                # 取消现有的定时器，避免重复触发
                user_queues[username]['timer'].cancel()

                # 将当前消息追加到用户的消息列表中
                user_queues[username]['messages'].append(time_aware_content)

                # 创建一个新的定时器，5秒后触发 process_user_messages 函数
                user_queues[username]['timer'] = threading.Timer(5.0, process_user_messages, args=[username])

                # 启动新的定时器
                user_queues[username]['timer'].start()

    # 捕获所有异常，并打印错误信息
    except Exception as e:
        print(f"消息处理失败: {str(e)}")


# 清空聊天上下文
def clear_chat_contexts():
    with queue_lock:
        chat_contexts.clear()  # 清空所有用户的上下文
        logger.info("已清空所有用户的聊天上下文。")

# 清空指定用户的最新一条上下文
def clear_latest_user_context(user_id):
    with queue_lock:
        if user_id in chat_contexts and len(chat_contexts[user_id]) > 0:
            chat_contexts[user_id].pop()
            logger.info(f"已清空用户 {user_id} 的最新一条聊天上下文。")


# 主函数
def main():
    try:
        # 静默初始化微信客户端
        wx = WeChat()
        if not wx.GetSessionList():
            print("请确保微信已登录并保持在前台运行!")
            return

        # 静默添加监听列表，移除savepic参数
        for i in listen_list:
            try:
                wx.AddListenChat(who=i)  # 移除 savepic=True
            except Exception as e:
                logger.error(f"不好了！添加监听失败 {i}: {str(e)}")
                return

        # 启动定时器，每小时清空上下文
        schedule.every().hour.do(lambda: threading.Thread(target=clear_chat_contexts).start())

        # 启动监听线程，只输出一次启动提示
        print("启动消息监听...")
        listener_thread = threading.Thread(target=message_listener)
        listener_thread.daemon = True
        listener_thread.start()

        # 主循环保持安静
        while True:
            schedule.run_pending()  # 运行所有待处理的任务
            time.sleep(1)
            if not listener_thread.is_alive():
                listener_thread = threading.Thread(target=message_listener)
                listener_thread.daemon = True
                listener_thread.start()

    except Exception as e:
        logger.error(f"主程序异常: {str(e)}")
    finally:
        print("程序退出")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户终止程序")
    except Exception as e:
        print(f"程序异常退出: {str(e)}")
