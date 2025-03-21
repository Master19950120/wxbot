# 用户列表(请配置要和bot说话的账号的昵称或者群名，不要写备注！)
# 例如：LISTEN_LIST = ['用户1','用户2','群名']
LISTEN_LIST = ['','']
SEND_LIST =['']
# 机器人的微信名称
ROBOT_WX_NAME = ''
# DeepSeek API 配置
DEEPSEEK_API_KEY = 'sk-'
DEEPSEEK_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/'
# 如果要使用官方的API
# DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
# 硅基流动API的V3模型，推荐充值杀那个pro，注意看模型名字哦
# MODEL = 'deepseek-r1-distill-llama-8b'
MODEL = 'deepseek-v3'
# 官方API的V3模型
# MODEL = 'deepseek-chat'
# 回复最大token
MAX_TOKEN = 2000
#温度
TEMPERATURE = 1.0
#图像生成
IMAGE_MODEL = 'CogView-3-Flash'
IMAGE_URL = 'https://open.bigmodel.cn/api/paas/v4/images/generations'
IMAGE_API_KEY = '' # 智谱
IMAGE_SIZE = '1024x1024'
BATCH_SIZE = '1'
GUIDANCE_SCALE = '3'
NUM_INFERENCE_STEPS = '4'
PROMPT_ENHANCEMENT = 'True'
TEMP_IMAGE_DIR = 'temp_images'
#最大的上下文轮数
MAX_GROUPS = 15
#prompt文件名
PROMPT_NAME = 'test.md'

# 星座运势API配置
API_URL = 'http://web.juhe.cn/constellation/getAll'  # 接口请求URL
API_KEY = ''  # 在个人中心->我的数据,接口名称上方查看
