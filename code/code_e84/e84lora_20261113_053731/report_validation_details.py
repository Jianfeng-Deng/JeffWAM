from common import *
p=C/'report.py';s=p.read_text();s=s.replace('经验bootstrap[0,0]不代表总体率确定为0。','经验bootstrap[0,0]不代表总体率确定为0。')
s=s.replace('经验bootstrap[0,0]不代表总体率确定为零。','经验bootstrap[0,0]不代表总体率确定为零。')
s=s.replace('经验bootstrap[0,0]不代表总体确定为零。','经验bootstrap[0,0]不代表总体确定为零。')
needle=" s.append(table(['编码器','半径倍数','两圆分离','帧/parent','双句均跟随 %','argmax改变 %','正确−反向命中差 pp'],rows))";assert needle in s;s=s.replace(needle,needle+"\n s.append('正式LoRA在r且两圆分离子集为0/7双句成功；在独立二项假设下，Clopper–Pearson双侧95%区间上界约40.96%，比经验bootstrap退化区间更能显示小样本不确定性，仍不包含共享标注偏差。')")
s=s.replace('长任务采用Python subprocess start_new_session=True + nohup（等价setsid分离）与日志/状态文件','长任务通过Python subprocess(start_new_session=True)分离会话（与setsid同效），stdin=/dev/null、stdout/stderr落盘并写状态文件；初始CPU准备使用setsid nohup')
needle=" text='\\n\\n'.join(s)+'\\n'";assert needle in s;s=s.replace(needle," if (P/'verification.json').exists():s.append('## 交付校验\\n\\n'+str(sum(load(P/'verification.json')['checks'].values()))+'项检查通过，包含双塔LoRA更新、同步数/配置、train parent归属、11条E83原索引与锚点相同、特征SHA/有限值与542capture、150帧patch及GPU预算。CPU校验台账首次numpy布尔序列化已修正；近参数量冻结头的parent元数据改存unicode，标签/分数/顺序逐项相等，未重训模型或使用GPU。')\n"+needle)
p.write_text(s);print('report_validation_details_added')
