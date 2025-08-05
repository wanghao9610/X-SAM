def split_list(lst, values):
    res = []
    tmp_res = []
    for i in lst:
        if i in values:
            res.append(tmp_res)
            res.append([i])
            tmp_res = []
        else:
            tmp_res.append(i)
    res.append(tmp_res)
    return res
