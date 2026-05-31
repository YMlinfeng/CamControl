from google import genai

API_KEY = "AIzaSyCyiiApnmw_2PCPUGkl8_smHXRC0U-GOZE" # 替换为您的 Key
client = genai.Client(api_key=API_KEY)

print("您当前 API Key 支持的生成式模型列表：")
for model in client.models.list():
    # 注意这里改成了 supported_actions
    if "generateContent" in model.supported_actions: 
        # 去掉前缀 models/ 方便直接复制使用
        model_id = model.name.replace("models/", "")
        print(f" - {model_id} ({model.display_name})")