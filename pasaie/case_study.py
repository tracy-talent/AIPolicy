from copy import deepcopy


def diff(ours_file, baseline_file, output_file):
    """case study of ours diff from baseline

    Args:
        ours_file (str): ours result, ensure sent/gold/pred order in file.
        baseline_file (str): baseline resule, ensure sent/gold/pred order in file.
        output_file (str): diff output, sent/ours/baseline order in file.
    """
    with open(ours_file, 'r', encoding='utf-8') as rf1, open(baseline_file, 'r', encoding='utf-8') as rf2, open(output_file, 'w', encoding='utf-8') as wf:
        ours = rf1.readlines()
        baseline = rf2.readlines()
        assert len(ours) == len(baseline)
        i = 0
        while i < len(ours):
            if baseline[i + 1] != baseline[i + 2]:
                try:
                    ours_gold = eval(ours[i + 1])
                    ours_pred = eval(ours[i + 2])
                    baseline_gold = eval(baseline[i + 1])
                    baseline_pred = eval(baseline[i + 2])
                except:
                    print(i)
                j = 0
                while j < len(ours_pred):
                    # 只保留出现在gold中且未出现在baseline_pred中的实体
                    if ours_pred[j] not in ours_gold or ours_pred[j] in baseline_pred:
                        ours_pred.pop(j)
                    else:
                        j += 1
                # 确保ours存在basline未预测出的实体则输出到文件中
                if len(ours_pred) > 0:
                    j = 0
                    while j < len(baseline_pred):
                        if baseline_pred[j] in baseline_gold:
                            baseline_pred.pop(j)
                        else:
                            j += 1
                    wf.write(baseline[i])
                    wf.write('ours pred: ' + str(ours_pred) + '\n')
                    wf.write('baseline pred: ' + str(baseline_pred) + '\n\n')
            i += 4
