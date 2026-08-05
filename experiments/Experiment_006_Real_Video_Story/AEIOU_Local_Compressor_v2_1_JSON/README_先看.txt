AEIOU 本地动态压缩 v2.1 JSON
================================

这一版在 v2.0 五维动态轨迹核心上增加 JSON / JSONL / NDJSON 输入适配。
不需要安装第三方库，不联网，不使用模型或 GPU。

支持输入：
- .txt
- .docx
- .json
- .jsonl
- .ndjson

JSON 使用：
1. 把 JSON 文件放进 input 文件夹。
2. 双击“开始压缩.py”，或运行：python run.py
3. 结果进入 output 文件夹。

JSON 可用结构：
- 顶层数组：[record, record, ...]
- 顶层对象包含 frames / records / events / items / data 数组
- JSONL / NDJSON：每行一条记录

JSON 轨迹规则：
- 每条 JSON 记录是一个关系单位。
- 数值字段名确定一个固定五维方向，数值决定该方向的载荷。
- 字符串、布尔和空值也作为确定性事件进入五维载荷。
- frame_id、timestamp_ms 等标识字段默认不参与向量，但完整保留在输出记录中。
- 不给 A、E、I、O、U 赋固定语义。
- 不改 JSON 字段和值；仅用排序后的紧凑格式输出被选记录。

重要：
这一版可以处理“已转成 JSON 的视频帧特征数据”，但不直接读取视频像素，
也不自带物体检测、人脸识别或动作识别模型。
