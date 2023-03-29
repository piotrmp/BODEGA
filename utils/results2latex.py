import pathlib

victim = 'BERT'
task_dict = {'HN': 'HN', 'PR': 'PR2', 'FC': 'FC', 'RD': 'RD'}
tasks = ['HN', 'PR', 'FC', 'RD']
method_dict = {'BAE': 'BAE', 'BERT-ATTACK': 'BERTattack', 'DeepWordBug': 'DeepWordBug', 'Genetic': 'Genetic',
               'SememePSO': 'PSO', 'PWWS': 'PWWS', 'SCPN': 'SCPN', 'TextFooler': 'TextFooler'}
methods = ['BAE', 'BERT-ATTACK', 'DeepWordBug', 'Genetic', 'SememePSO', 'PWWS', 'SCPN', 'TextFooler']
path = pathlib.Path.home() / 'data' / 'BODEGA' / 'results'

results = {}
for task in tasks:
    for targeted in ['False', 'True']:
        for measure in ['BODEGA', 'c_s', 'b_s', 'l_s', 'q']:
            results[task + targeted + measure] = {}

for task in tasks:
    for method in methods:
        for targeted in ['False', 'True']:
            file_name = 'results_' + task_dict[task] + '_' + targeted + '_' + method_dict[
                method] + '_' + victim + '.txt'
            for line in open(path / file_name):
                if line.startswith('BODEGA score: '):
                    results[task + targeted + 'BODEGA'][method] = float(line.split(' ')[-1].strip())
                elif line.startswith('Success score: '):
                    results[task + targeted + 'c_s'][method] = float(line.split(' ')[-1].strip())
                elif line.startswith('BERT score: '):
                    results[task + targeted + 'b_s'][method] = float(line.split(' ')[-1].strip())
                elif line.startswith('Levenshtein score: '):
                    results[task + targeted + 'l_s'][method] = float(line.split(' ')[-1].strip())
                elif line.startswith('Queries per example: '):
                    results[task + targeted + 'q'][method] = float(line.split(' ')[-1].strip())

for task in tasks:
    for method in methods:
        taskI = task if method == methods[0] else ''
        print('        ' + taskI + ' & ' + method + ' ', end='')
        for targeted in ['False', 'True']:
            for measure in ['BODEGA', 'c_s', 'b_s', 'l_s', 'q']:
                value = results[task + targeted + measure][method]
                prefix = '\\textbf' if measure != 'q' and (
                            value == max(results[task + targeted + measure].values())) else ''
                print(" & {0}{{{1:.2f}}}".format(prefix, value), end='')
        print(' \\\\')
    print('    \\hline')
