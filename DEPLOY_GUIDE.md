# 🚀 如何把app部署到云端（免费）
### 完成后你会得到一个永久网址，手机/电脑/朋友都能用

---

## 第一步：注册 GitHub 账号

1. 去 **https://github.com** 点 "Sign up"
2. 用邮箱注册，记住你的用户名和密码
3. 验证邮箱

---

## 第二步：上传代码到 GitHub

1. 登录后，点右上角 **"+"** → **"New repository"**

2. 填写：
   - Repository name: `trading-analyst`（随便起名）
   - 选 **Private**（私人，别人看不到你的代码）
   - 勾选 **"Add a README file"**
   - 点 **"Create repository"**

3. 进入仓库后，点 **"uploading an existing file"**（蓝色链接）

4. 把这些文件拖进去上传（在你的 Trading Analyst 文件夹里）：
   - `app.py`
   - `requirements.txt`
   - `.gitignore`
   - `HOW_TO_RUN.md`（可选）
   
   ⚠️ **不要上传** `.streamlit/secrets.toml`（这个有你的API key，保密！）

5. 点底部 **"Commit changes"** 确认上传

---

## 第三步：注册 Streamlit Cloud

1. 去 **https://share.streamlit.io**
2. 点 **"Sign up"** → 选 **"Continue with GitHub"**（用你刚注册的GitHub账号登录）
3. 授权允许Streamlit访问你的GitHub

---

## 第四步：部署 App

1. 登录后点 **"New app"**

2. 填写：
   - **Repository**: 选你刚上传的 `trading-analyst`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: 自定义一个名字，比如 `chee-trading-analyst`
     （最终网址会是 `https://chee-trading-analyst.streamlit.app`）

3. 点 **"Deploy!"** — 等大约2-3分钟

---

## 第五步：设置 API Key（最重要！）

部署好后，你需要把API key安全地存进去：

1. 在Streamlit Cloud里，找到你的app，点右边 **"⋮"** → **"Settings"**

2. 点左边 **"Secrets"** 标签

3. 在空白框里输入（选一个你有的key）：

   如果你用 **Anthropic (Claude)**：
   ```
   ANTHROPIC_API_KEY = "sk-ant-你的key"
   ```

   如果你用 **Gemini**：
   ```
   GEMINI_API_KEY = "AIza-你的key"
   ```

4. 点 **"Save"** → App会自动重启

5. 重启后，侧边栏会显示 ✅ API Key loaded automatically

---

## 完成！🎉

你的app网址：`https://你取的名字.streamlit.app`

- 📱 手机随时用
- 👫 发给朋友直接打开
- 🌍 全球任何地方都能访问
- 💤 没人用时会自动休眠，有人访问会自动唤醒（免费版）

---

## 注意事项

- 免费版每月有使用限制，够个人用
- API费用还是你自己承担（Gemini免费，Claude按用量）
- 如果想让朋友用但不想共享你的API key，可以在app里让他们自己输入key（默认就是这样）
- 如果想让朋友不用输key直接用，就把key存在Secrets里，但费用你来付

---

## 代码更新后怎么办？

每次修改了 app.py，重新上传到GitHub，Streamlit Cloud会**自动检测并重新部署** ✅
