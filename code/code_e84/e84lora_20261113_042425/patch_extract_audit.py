import pathlib
c=pathlib.Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e84/e84lora_20261113_042425');p=c/'extract.py';s=p.read_text().replace("'source_stat_before_after_equal':True,'sources'", "'source_stat_before_after_equal':True,'encoded_frames':sum(len(set(camera_rows[:,:,i].ravel())-{-1}) for i in range(3)),'sources'")
s=s.replace('fp16，历史只存global，当前保存global+196patch', 'fp16，所有时序仅global，patch只保存150帧近似GT诊断输入')
p.write_text(s)
