import numpy as np
from numba import njit

def calculate_ced(ret, window = 12, alpha = 0.95):
    cum_return = (1 + ret).cumprod()
    drawdowns = calculate_drawdowns(cum_return, window)
    thresh = np.quantile(drawdowns, alpha)
    return drawdowns[drawdowns > thresh].mean()

    

@njit
def calculate_drawdowns(cum, window):
    length = len(cum)
    drawdowns = np.zeros((length - window + 1, 1))

    for i in range(length - window + 1):
        max_drawdown = 0
        running_max = cum[i]
        
        for j in range(i, i + window):
            if cum[j] > running_max:
                running_max = cum[j]
                
            drawdown = (running_max - cum[j]) / running_max
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        drawdowns[i, 0] = max_drawdown
        

    return drawdowns