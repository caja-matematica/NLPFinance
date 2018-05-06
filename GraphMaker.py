import numpy as np
import matplotlib.pyplot as plt
#2013-2016, 10 bins, filesize

month1 = [12906833.0, 14131242.0, 15319515.0, 15732264.5, 14326085.0, 14414622.5, 15055187.0, 14660442.0, 14732981.0, 13373138.0]
month3 = [13481216.5, 14491728.0, 15768582.5, 14326085.0, 15747870.0, 14861511.0, 15849351.0, 14792635.0, 13529784.0, 13447391.0]
month6 = [13710645.5, 14587176.5, 15217107.5, 14326085.0, 14983329.0, 14531890.0, 15925280.0, 15681600.0, 13890406.0, 12542662.0]
month12 = [13745785.0, 15406107.0, 14831388.5, 14326085.0, 15747870.0, 14531890.0, 15287866.0, 14246640.0, 14099636.0, 12643501.0]
x = [1,2,3,4,5,6,7,8,9,10]

plt.xlabel('Return Deciles')
plt.ylabel('Median Filesize')
plt.title('10-K File Sizes vs. Relative Returns')
plt.plot(x, month1, '-o', color='black', marker='^', ms=14, lw=2, alpha=1, mfc='black', label='1 Month')
plt.plot(x, month3, '-o', color='blue', marker='o', ms=10, lw=2, alpha=1, mfc='blue', label='3 Month')
plt.plot(x, month6, '-o', color='red', marker='s', ms=12, lw=2, alpha=1, mfc='red', label='6 Month')
plt.plot(x, month12, '-o', color='green', marker='*', ms=14, lw=2, alpha=1, mfc='green', label='12 Month')
plt.legend(loc=1)

plt.show()
