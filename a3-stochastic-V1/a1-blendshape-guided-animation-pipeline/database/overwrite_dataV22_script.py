import shutil
# change this line to the target pose you want to deform to.
copy_from = r"C:\keyLabLocal_D\a3-stochastic-V1-backup-and-results\backup\b31 dataV22 backup hub\dataV22_AU0_neutral.py"  # dataV22_AU0_neutral, dataV22_phoneme_T
copy_to = r"C:\Users\zpa18\a3-stochastic-V1\a1-blendshape-guided-animation-pipeline\database\dataV22.py"
shutil.copy(copy_from, copy_to)  # Exact same filename