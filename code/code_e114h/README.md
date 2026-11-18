# E114-H 阶段人工复核页面
启动（服务器）：PYTHONDONTWRITEBYTECODE=1 setsid nohup /mnt/workspace/dengjianfeng/venvs/e29/bin/python code_e114h/server.py --port 18795 --out results/e114h_human_stage_20261116 > results/e114h_human_stage_20261116/server.log 2>&1 < /dev/null &
本机访问：ssh -N -L 18795:127.0.0.1:18795 <gpu-server>，然后浏览器打开 http://localhost:18795
输出：results/e114h_human_stage_20261116/{boundaries,review}/<parent>.json（label_source=human），log.jsonl 为追加日志；GET /api/export 导出全部。
原视频、原标签只读，页面只写上述输出目录。服务只绑定 127.0.0.1。
模式①盲标阶段边界（保存后才显示模型回顾式区间）；模式②盲复核 E98-C 因果阶段答案（先选后显示）。
