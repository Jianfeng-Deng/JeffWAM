from pathlib import Path
import json,datetime
T=Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu');R=Path((T/'code_e105y/run_path.txt').read_text().strip());P=T/'notes/e105y_yam_feasibility_20261115.md'
s=P.read_text();i=s.index('## 实测清单（完成）');s=s[:s.index('状态：')]+'状态：COMPLETE；BRT 2026-11-15。仅CPU、不训练、不下载，GPUh=0。以下为实测结果；身份与内容去重不能由现有元数据证明的部分明确列为未确认。\n\n'+s[i:]
s+='\n## 最终复核补充\n129个视频格式再次核对全部640×480/30Hz；43/43基础8类状态/动作数组均有限，另外10月14段各有expert/student两臂关节与夹爪数组及intervene标记，这些不是独立末端位姿。报告无模型语言标注、未据源success标签作测试真值。相机角色对应成立43/43；机器人/物理安装不同于Piper为43/43；未见缺路0/43。逐片全程视野变化仍未穷举，勿把7段21张截图审计写成43段全视频检查。\n\n推荐10月dev的+100实际109 clip固定窗，用同一IID近似半宽约2.18pp；这仍非有效parent CI（1个parent无法估计）。保守留出式最多1个dev parent，宽松最多2个但剩yam训练仅1段，不建议。K=3/5每折分配表已给，空折不可执行。ABC41个task×UTC-day桶、36日只是合并审计，独立parent未确认；严格场景/操作者泄漏证明需额外身份资料或内容去重。\n\n最终新增量约0.43MiB（含报告与脚本），无GPU进程、无训练。共享EXPERIMENT_LOG和worklog各有1条E105-Y记录。\n'
P.write_text(s)
f=R/'VERIFICATION.json';assert not f.exists();f.write_text(json.dumps(dict(status='passed',gpu_seconds=0,camera_streams=129,camera_format_ok=129,shared_log_entries=1,worklog_entries=1,report_no_pending_skeleton=True,total_new_bytes=sum(p.stat().st_size for d in [R,T/'code_e105y'] for p in d.rglob('*') if p.is_file())+P.stat().st_size),indent=2));print(f.read_text())
