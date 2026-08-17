# 当前进度指针

**阶段**：第三步「验收」—— 前端重做（Chainlit）已完成

**本轮目标**：用 Chainlit 统一现有 Gradio 界面为现代聊天应用（法律咨询 + 文档问答一体）。见 anchor.md 完成定义 F1–F6。

**结果**：6/6 标准通过 → **交付**（见 evidence-bundle.md 第 7 节）。

**交付物**：
- `app/chainlit_app.py`（统一 Chainlit 前端）
- `.chainlit/config.toml`（主题）、`chainlit.md`（欢迎页）
- `requirements.txt` 增 `chainlit==2.11.1`；`README.md` Demo 入口更新

**启动**：`chainlit run app/chainlit_app.py -w`（项目根目录）

**已知问题（不阻塞，非前端）**：模型对部分查询保守拒答（历史过度拒答，E1 拒答率 34.95%，需 Divide-Then-Align DPO 重训）。

**回退规则**：出错回到最近 checkpoint（本目录文件），禁止重跑已完成阶段。
