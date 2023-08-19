import polars as pl

'''
Note - idx refers to 1 being most recent data.
Input Para:
data - Close data
idx
Output:
bool - If idx is local top of order n
'''
def rw_tops(data: pl.Series, order: int) -> pl.Series:
    tops = []
    
    for i in range(len(data)):
        appended = False
        if i < order * 2 + 1:
            tops.append(False) # Not enough points at end to decide
        else:
            k = i - order
            v = data[k]
            for i in range(1, order + 1):
                if data[k + i] > v or data [k - i] > v:
                    tops.append(False)
                    appended = True
                    break
            if appended == False:
                tops.append(True)

    return pl.Series('Tops', tops)

def rw_bottoms(data: pl.Series, order: int) -> pl.Series:
    bottoms = []

    for i in range(len(data)):
        appended = False
        if i < order * 2 + 1: 
            bottoms.append(False) # Not enough points at end to decide

        else:
            k = i - order
            v = data[k]
            for i in range(1, order + 1):
                if data[k + i] < v or data [k - i] < v:
                    bottoms.append(False)
                    appended = True
                    break
                
            if appended == False:
                bottoms.append(True)

    return pl.Series('Bottoms', bottoms)

def zigzag(data, sigma: float):
    last_zig = True
    tmp_high = data['high'][0]; tmp_low = data['low'][0]
    tmp_high_idx = 0; tmp_low_idx = 0

    tops = []; bottoms = []
    for i in range(len(data)):
        if last_zig:
            if data['high'][i] > tmp_high:
                tmp_high = data['high'][i]
                tmp_high_idx = i
            elif data['close'][i] < tmp_high - (tmp_high * sigma):
                tops.append([i, tmp_high_idx, tmp_high])
                last_zig = False
                tmp_low = data['low'][i]
                tmp_low_idx = i
        else:
            if data['low'] < tmp_low:
                tmp_low = data['low'][i]
                tmp_low_idx = i
            elif data['close'] > tmp_low + (tmp_low * sigma):
                bottoms.append([i, tmp_low_idx, tmp_low])
                last_zig = True
                tmp_high = data['high'][i]
                tmp_high_idx = i
    
    return tops, bottoms
         

def find_pips(data: pl.Series, n_pips: int, dist_measure: int) -> pl.Series:
    return