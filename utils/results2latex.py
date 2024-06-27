import pathlib
import matplotlib.pyplot as plt

victims = ['BiLSTM', 'BERT', 'GEMMA', 'GEMMA7B']
task_name_to_filename_mapping = {'HN': 'HN', 'PR': 'PR2', 'FC': 'FC', 'RD': 'RD'}
tasks = ['HN', 'PR', 'FC', 'RD']

method_name_to_filename = {'BAE': 'BAE', 'BERT-ATTACK': 'BERTattack', 'DeepWordBug': 'DeepWordBug',
                           'Genetic': 'Genetic', 'SememePSO': 'PSO', 'PWWS': 'PWWS', 'SCPN': 'SCPN',
                           'TextFooler': 'TextFooler'}
methods = ['BAE', 'BERT-ATTACK', 'DeepWordBug', 'Genetic', 'SememePSO', 'PWWS', 'SCPN', 'TextFooler']
measure_to_display_mapping = {'BODEGA': 'BODEGA', 'c_s': 'conf.', 'b_s': 'sem.', 'l_s': 'char.', 'q': 'queries'}
path = pathlib.Path.home() / 'data' / 'BODEGA' / 'results'
targeted_variants = ['True', 'False']

# Creating data structures
results = {}
for victim in victims:
    for task in tasks:
        for targeted in targeted_variants:
            for measure in ['BODEGA', 'c_s', 'b_s', 'l_s', 'q']:
                results[victim + task + targeted + measure] = {}

# Reading the files
for victim in victims:
    for task in tasks:
        for method in methods:
            for targeted in targeted_variants:
                file_name = 'results_' + task_name_to_filename_mapping[task] + '_' + targeted + '_' + \
                            method_name_to_filename[method] + '_' + victim + '.txt'
                for line in open(path / file_name):
                    if line.startswith('BODEGA score: '):
                        results[victim + task + targeted + 'BODEGA'][method] = float(line.split(' ')[-1].strip())
                    elif line.startswith('Success score: '):
                        results[victim + task + targeted + 'c_s'][method] = float(line.split(' ')[-1].strip())
                    elif line.startswith('Semantic score: '):
                        results[victim + task + targeted + 'b_s'][method] = float(line.split(' ')[-1].strip())
                    elif line.startswith('Character score: '):
                        results[victim + task + targeted + 'l_s'][method] = float(line.split(' ')[-1].strip())
                    elif line.startswith('Queries per example: '):
                        results[victim + task + targeted + 'q'][method] = float(line.split(' ')[-1].strip())

# Full LaTeX printing
for victim in victims:
    print("FULL " + victim)
    for task in tasks:
        for method in methods:
            taskI = task if method == methods[0] else ''
            print('        ' + taskI + ' & ' + method + ' ', end='')
            for targeted in ['False', 'True']:
                for measure in ['BODEGA', 'c_s', 'b_s', 'l_s', 'q']:
                    value = results[victim + task + targeted + measure][method]
                    prefix = '\\textbf' if measure != 'q' and (
                            value == max(results[victim + task + targeted + measure].values())) else ''
                    print(" & {0}{{{1:.2f}}}".format(prefix, value), end='')
            print(' \\\\')
        print('    \\hline')

# Averaging scores over victims
print("AVERAGE over victims")
targeted = 'False'
for task in tasks:
    for method in methods:
        taskI = task if method == methods[0] else ''
        print('        ' + taskI + ' & ' + method + ' ', end='')
        for measure in ['BODEGA', 'c_s', 'b_s', 'l_s', 'q']:
            value = sum([results[victim + task + targeted + measure][method] for victim in victims]) / len(victims)
            best_value = max(
                [sum([results[victim + task + targeted + measure][method] for victim in victims]) / len(victims) for
                 method in methods])
            prefix = '\\textbf' if measure != 'q' and (value == best_value) else ''
            print(" & {0}{{{1:.2f}}}".format(prefix, value), end='')
        print(' \\\\')
    print('    \\hline')

# Victim analysis
targeted = 'True'
victims_to_params = {'BiLSTM': 1000000, 'BERT': 340000000, 'GEMMA': 2000000000, 'GEMMA7B': 7000000000}
victims_to_f1 = {'HN': {'BiLSTM': 0.7076, 'BERT': 0.7544, 'GEMMA': 0.7792, 'GEMMA7B': 0.7603},
                 'PR': {'BiLSTM': 0.4857, 'BERT': 0.6410, 'GEMMA': 0.6271, 'GEMMA7B': 0.6840},
                 'FC': {'BiLSTM': 0.7532, 'BERT': 0.9360, 'GEMMA': 0.9701, 'GEMMA7B': 0.9727},
                 'RD': {'BiLSTM': 0.6234, 'BERT': 0.7547, 'GEMMA': 0.7609, 'GEMMA7B': 0.7229}}
measure = 'BODEGA'
plt.clf()
fig, subs = plt.subplots(2, 2, sharey='row', sharex='col')
fig.set_size_inches(7, 6)
fig.subplots_adjust(wspace=0.1)
fig.subplots_adjust(hspace=0.1)
subplots = {'HN': subs[0][0], 'PR': subs[0][1], 'FC': subs[1][0], 'RD': subs[1][1]}
for task in tasks:
    best_here = {}
    for victim in victims:
        best_here[victim] = max([results[victim + task + targeted + measure][method] for method in methods])
    x = [victims_to_params[victim] for victim in victims]
    y1 = [best_here[victim] for victim in victims]
    y2 = [victims_to_f1[task][victim] for victim in victims]
    subplot = subplots[task]
    subplot.set_xscale("log")
    subplot.set_ylim([0.0, 1.0])
    if task in ['HN', 'FC']:
        subplot.yaxis.set_label_text('F1 / BODEGA score')
    if task in ['RD', 'FC']:
        subplot.xaxis.set_label_text('Model size')
    subplot.plot(x, y2, label='F1', linestyle=':', marker='o', lw=2)
    subplot.plot(x, y1, label='BODEGA', linestyle='-', marker='o', lw=2)
    location = 'best'
    subplot.legend(title=task, loc=location)
plt.savefig(path / '../victims.pdf', bbox_inches='tight')

# Queries vs performance
targeteds = ['False']
method_to_marker = {'BAE': 'v', 'BERT-ATTACK': 'o', 'DeepWordBug': '+',
                    'Genetic': 'x', 'SememePSO': '*', 'PWWS': '1', 'SCPN': 'p',
                    'TextFooler': 's'}
victim_to_color = {'BiLSTM': 1, 'BERT': 2, 'GEMMA': 3, 'GEMMA7B': 4}
task_to_color = {'HN': 1, 'FC': 2, 'PR': 3, 'RD': 4}

fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xscale("log")
for method in methods:
    x = []
    y = []
    c = []
    for targeted in targeteds:
        for task in tasks:
            for victim in victims:
                y.append(results[victim + task + targeted + 'BODEGA'][method])
                x.append(results[victim + task + targeted + 'q'][method])
                c.append(task_to_color[task])
    scatter = plt.scatter(x, y, c=c, marker=method_to_marker[method], label=method)

legend1 = ax.legend(title='Attackers', loc='upper left')
ax.add_artist(legend1)
handles, labels = scatter.legend_elements(prop="colors", alpha=0.6)
labels = ['HN', 'FC', 'PR', 'RD']
legend2 = ax.legend(handles, labels, loc="upper right", title="Tasks")
ax.add_artist(legend2)
# ax.set_ylim([0.0,0.9])
ax.xaxis.set_label_text('Number of queries')
ax.yaxis.set_label_text('BODEGA score')
plt.savefig(path / '../queries.pdf', bbox_inches='tight')

# Targeted vs untargeted
print("Targeted vs Untargeted")
for task in tasks:
    print("        " + task + " & B. score", end='')
    for victim in victims:
        scores = []
        for targeted in ["False", "True"]:
            best_score = 0.0
            best_score_q = 0.0
            for method in methods:
                value = results[victim + task + targeted + 'BODEGA'][method]
                if value > best_score:
                    best_score = value
                    best_score_q = results[victim + task + targeted + 'q'][method]
            scores.append(best_score)
        for i in range(len(scores)):
            prefix = '\\textbf' if scores[i] == max(scores) else ''
            print(" & {0}{{{1:.2f}}}".format(prefix, scores[i]), end='')
    print('\\\\ \n', end='')
    print("         & Queries", end='')
    for victim in victims:
        scores = []
        for targeted in ["False", "True"]:
            best_score = 0.0
            best_score_q = 0.0
            for method in methods:
                value = results[victim + task + targeted + 'BODEGA'][method]
                if value > best_score:
                    best_score = value
                    best_score_q = results[victim + task + targeted + 'q'][method]
            scores.append(best_score_q)
        for i in range(len(scores)):
            prefix = '\\textbf' if scores[i] == min(scores) else ''
            print(" & {0}{{{1:.2f}}}".format(prefix, scores[i]), end='')
    print('\\\\ \n \\hline\n', end='')
