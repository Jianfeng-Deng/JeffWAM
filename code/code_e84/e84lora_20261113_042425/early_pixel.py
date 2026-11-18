import pathlib
C=pathlib.Path('/mnt/workspace/dengjianfeng/code/lewm_tianji_cpu/code_e84/e84lora_20261113_042425');p=C/'run_pixel.py';s=p.read_text().replace("def main(version='lora'):","def main_unlocked(version='lora'):",1);position=s.index("if __name__=='__main__':")
s=s[:position]+'''def main(version='lora'):
    import fcntl
    root=P/version;root.mkdir(exist_ok=True)
    with (root/'pixel_eval.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return main_unlocked(version)
'''+s[position:];p.write_text(s);print('pixel_lock_added')
