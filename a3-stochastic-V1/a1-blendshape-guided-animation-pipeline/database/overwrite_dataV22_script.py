import shutil
from paths import a1_blendshape_path, backup_path
# change this line to the target pose you want to deform to.
copy_from = str(backup_path / "b31 dataV22 backup hub" / "dataV22_AU0_neutral.py")  # dataV22_AU0_neutral, dataV22_phoneme_T
copy_to = str(a1_blendshape_path / "database" / "dataV22.py")
shutil.copy(copy_from, copy_to)  # Exact same filename