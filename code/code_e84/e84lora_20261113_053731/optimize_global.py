from common import *
p=C/'encoder.py';s=p.read_text();s+='''
    @torch.inference_mode()
    def global_images(self,pixels):
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.device.startswith('cuda')):
            z=self.model.vision_model(pixel_values=pixels.to(self.device))
        return z.pooler_output.float().cpu().numpy(),None
''';p.write_text(s)
p=C/'extract.py';s=p.read_text().replace('g,p=encoder.images(torch.stack(batch))','g,p=encoder.global_images(torch.stack(batch))').replace('[cv2.CAP_PROP_N_THREADS,1]','[cv2.CAP_PROP_N_THREADS,4]');p.write_text(s)
append_report('## 全局复提实现\n\n'+brt()+' 时序分片只需global；不再将未使用的patch tensor传回CPU，150帧诊断仍保存真实patch。视频单路解码4CPU线程，图像预处理/输出数值合同不变。')
print('global_only_transfer_optimized')
