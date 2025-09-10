

import importlib, subprocess, shutil, sys
def v(mod):
    try:
        m = importlib.import_module(mod)
        print(f"{mod}: {getattr(m,'__version__','?')}")
    except Exception as e:
        print(f"{mod}: NOT INSTALLED ({e})")
print("Python:", sys.version)
for m in ["streamlit","numpy","librosa","soundfile","moviepy","imageio_ffmpeg","opencv_python"]:
    v(m)
try:
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    print("ffmpeg exe:", exe)
    subprocess.run([exe,"-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("ffmpeg OK")
except Exception as e:
    print("ffmpeg via imageio-ffmpeg NOT OK:", e)
print("system ffmpeg:", shutil.which("ffmpeg"))
