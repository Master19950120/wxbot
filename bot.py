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

# 添加图像生成相关常量
IMAGE_API_URL = f"{IMAGE_URL}"  # 需要在config.py中添加基础URL

# 添加临时目录初始化
temp_dir = os.path.join(root_dir, TEMP_IMAGE_DIR)
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)


# 保存聊天记录到数据库
def save_message(sender_id, sender_name, message, reply):
    try:
        session = Session()
        chat_message = ChatMessage(
            sender_id=sender_id,
            sender_name=sender_name,
            message=message,
            reply=reply
        )
        session.add(chat_message)
        session.commit()
        session.close()
    except Exception as e:
        print(f"保存消息失败: {str(e)}")


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
            return "抱歉，我现在有点累，请稍后再试..."

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


# 处理用户消息
def process_user_messages(user_id):
    with queue_lock:
        if user_id not in user_queues:
            return
        user_data = user_queues.pop(user_id)
        messages = user_data['messages']
        sender_name = user_data['sender_name']
        username = user_data['username']

    # 过滤重复消息并确保合并的消息相关
    unique_messages = list(dict.fromkeys(messages))[-5:]  # 保留唯一消息并限制数量
    merged_message = ' \\ '.join(unique_messages)
    print(f"处理合并消息 ({sender_name}): {merged_message}")

    # 获取API回复
    reply = get_deepseek_response(merged_message, user_id)

    # 修改发送逻辑部分
    try:
        if '[IMAGE]' in reply:
            # 处理图片回复
            img_path = reply.split('[IMAGE]')[1].split('[/IMAGE]')[0].strip()
            # 确保使用临时目录
            if os.path.exists(img_path):
                try:
                    # 使用SendFiles方法发送图片，注意参数名是filepath
                    wx.SendFiles(filepath=img_path, who=user_id)
                    # 发送附加文本消息
                    text_msg = reply.split('[/IMAGE]')[1].strip()
                    if text_msg:
                        wx.SendMsg(msg=text_msg, who=user_id)
                finally:
                    # 删除临时图片
                    try:
                        os.remove(img_path)
                    except Exception as e:
                        logger.error(f"不好了！删除临时图片失败: {str(e)}")
        elif '\\' in reply:
            parts = [p.strip() for p in reply.split('\\') if p.strip()]
            for part in parts:
                wx.SendMsg(msg=part, who=user_id)
                time.sleep(random.randint(1, 2))
        else:
            wx.SendMsg(msg=reply, who=user_id)

    except Exception as e:
        logger.error(f"不好了！发送回复失败: {str(e)}")

    # 异步保存消息记录
    threading.Thread(target=save_message, args=(username, sender_name, merged_message, reply)).start()


# 消息监听
def message_listener():
    wx = None
    last_window_check = 0
    check_interval = 600  # 每600秒检查一次窗口状态,检查是否活动(是否在聊天界面)

    while True:
        try:
            current_time = time.time()

            # 只在必要时初始化或重新获取微信窗口，不输出提示
            if wx is None or (current_time - last_window_check > check_interval):
                wx = WeChat()
                if not wx.GetSessionList():
                    time.sleep(5)
                    continue
                last_window_check = current_time

            msgs = wx.GetListenMessage()
            if not msgs:
                time.sleep(wait)
                continue

            for chat in msgs:
                who = chat.who
                if not who:
                    continue

                one_msgs = msgs.get(chat)
                if not one_msgs:
                    continue

                for msg in one_msgs:
                    try:
                        msgtype = msg.type
                        content = msg.content
                        if not content:
                            continue
                        if msgtype != 'friend':
                            logger.debug(f"非好友消息，忽略! 消息类型: {msgtype}")
                            continue
                            # 只输出实际的消息内容
                        # 接收窗口名跟发送人一样，代表是私聊，否则是群聊
                        if who == msg.sender:
                            handle_wxauto_message(msg, msg.sender)  # 处理私聊信息
                        elif ROBOT_WX_NAME != '' and bool(re.search(f'@{ROBOT_WX_NAME}\u2005', msg.content)):
                            handle_wxauto_message(msg, who)  # 处理群聊信息，只有@当前机器人才会处理
                        # TODO(jett): 这里看需要要不要打日志，群聊信息太多可能日志会很多
                        else:
                            logger.debug(f"非需要处理消息，可能是群聊非@消息: {content}")
                    except Exception as e:
                        logger.debug(f"不好了！处理单条消息失败: {str(e)}")
                        continue

        except Exception as e:
            logger.debug(f"不好了！消息监听出错: {str(e)}")
            wx = None  # 出错时重置微信对象
        time.sleep(wait)


# 处理微信消息
def handle_wxauto_message(msg, chatName):
    try:
        username = chatName
        content = getattr(msg, 'content', None) or getattr(msg, 'text', None)
        # @消息过滤@头信息，防止机器名与prompt设定名不一致的问题
        content = content.replace(f'@{ROBOT_WX_NAME}\u2005', '')
        if not content:
            logger.debug("不好了！无法获取消息内容")
            return

        sender_name = username
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_aware_content = f"[{current_time}] {content}"

        with queue_lock:
            if username not in user_queues:
                # 减少等待时间为3秒
                user_queues[username] = {
                    'timer': threading.Timer(3.0, process_user_messages, args=[username]),
                    'messages': [time_aware_content],
                    'sender_name': sender_name,
                    'username': username
                }
                user_queues[username]['timer'].start()
            else:
                # 重置现有定时器
                user_queues[username]['timer'].cancel()
                user_queues[username]['messages'].append(time_aware_content)
                user_queues[username]['timer'] = threading.Timer(5.0, process_user_messages, args=[username])
                user_queues[username]['timer'].start()

    except Exception as e:
        print(f"消息处理失败: {str(e)}")


# 清空聊天上下文
def clear_chat_contexts():
    with queue_lock:
        chat_contexts.clear()  # 清空所有用户的上下文
        logger.info("已清空所有用户的聊天上下文。")


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
