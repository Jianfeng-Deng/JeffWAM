from common import *
d=P/'debug_attempts/device_fix_1';d.mkdir(parents=True,exist_ok=True)
for p in P.glob('*failure*.json'):p.rename(d/p.name)
update(status='resuming_after_device_placement_fix',gpu_debug_fix='LoRA parameter factory inherited CPU default after base model CUDA load. Corrected A/B device and dtype to base.weight; failed before first optimizer step, no OOM, all debug GPU time counted. Failure receipts preserved under debug_attempts/device_fix_1.')
append_report('## GPU实现修正\n\n'+brt()+' 首次forward发现新建LoRA A/B参数仍在CPU；已改为继承base.weight设备与dtype，失败发生在首个优化步之前，没有OOM或资源压力。全部耗时计入4GPUh预算，原失败状态归档debug_attempts/device_fix_1。')
print('debug_archived')
