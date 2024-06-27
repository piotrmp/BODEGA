import pathlib, torch,sys

from metrics.BODEGAScore import BODEGAScore
from utils.data_mappings import SEPARATOR
from utils.no_ssl_verify import no_ssl_verify
from victims.bilstm import VictimBiLSTM
from victims.caching import VictimCache
from victims.surprise import VictimRoBERTa
from victims.transformer import VictimTransformer, PRETRAINED_BERT

measures = ['BODEGA', 'c_s', 'b_s', 'l_s', 'q']
victims = ['BiLSTM', 'BERT', 'surprise']
tasks = [sys.argv[1]]#['PR2', 'FC', 'RD', 'HN', 'C19']
participants = ['participant_1', 'participant_2']
participant_to_method = {'participant_1': 'custom', 'participant_2': 'custom'}
original_text_path = pathlib.Path.home() / 'data' / 'BODEGA'
submission_path = original_text_path / 'all-submission'
output_path = submission_path / (tasks[0]+'results.tsv')

# Creating data structures
results = {}
for victim in victims:
    for task in tasks:
        for measure in measures:
            results[victim + task + measure] = {}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')


def normalise_text(text):
    normalised_text = text.replace("\\n",
                                   "\n")  # .replace("ΓÇ£","“").replace("ΓÇ¥","”").replace("ΓÇÖ","’").replace("\\n","\n")
    return normalised_text


# Reading original text
original_text = {task: [] for task in tasks}
for task in tasks:
    with_pairs = (task == 'FC' or task == 'C19')
    for line in open(original_text_path / task / "attack.tsv"):
        text = line.strip().split("\t")[2] + SEPARATOR + line.strip().split("\t")[3] if with_pairs else \
            line.strip().split("\t")[2]
        original_text[task].append(text)

verify = False

# Evaluating
with no_ssl_verify():
    for victim in victims:
        for task in tasks:
            print(victim + '-' + task)
            # Load victim model
            model_path = pathlib.Path.home() / 'data' / 'BODEGA' / task / (victim + '-512.pth')
            if verify:
                if victim == 'BERT':
                    pretrained_model = PRETRAINED_BERT
                    victim_i = VictimCache(model_path,
                                           VictimTransformer(model_path, task, pretrained_model, False, device))
                elif victim == 'BiLSTM':
                    pretrained_model = PRETRAINED_BERT
                    victim_i = VictimCache(model_path, VictimBiLSTM(model_path, task, device))
                elif victim == 'surprise':
                    victim_i = VictimCache(model_path,
                                           VictimRoBERTa(model_path, task, device))
            # Load participants' answers
            for participant in participants:
                print('\t evaluating ' + participant)
                file_name = 'submission_' + task + '_False_' + participant_to_method[
                    participant] + '_' + victim + '.tsv'
                scorer = BODEGAScore(device, task, align_sentences=True, semantic_scorer="BLEURT", raw_path=None)
                all_queries = 0
                good_aes = 0
                bad_aes = 0
                for i, line in enumerate(open(submission_path / participant / file_name)):
                    if line == 'succeeded\tnum_queries\toriginal_text\tmodified_text\t\n':
                        continue
                    parts = line.strip().split('\t')
                    succeded = parts[0].lower() == 'true'
                    queries = int(parts[1])
                    all_queries += queries
                    ae = normalise_text(parts[3]) if succeded else None
                    original = normalise_text(parts[2])#normalise_text(original_text[task][i - 1])
                    if succeded and verify:
                        result = victim_i.get_pred([original, ae])
                        if result[0] == result[1]:
                            print("FAILED AE from " + participant + " : " + ae[:100])
                            ae = None
                            bad_aes += 1
                        else:
                            good_aes += 1
                        #    print("GOOD AE")
                    # add to the scorer
                    scorer.after_attack({'x': original}, ae)
                print("Good AEs: " + str(good_aes) + ", failed AEs: " + str(bad_aes))
                score_success, score_semantic, score_character, score_BODEGA = scorer.compute()
                print('\t' + participant + ' score : ' + str(score_BODEGA))
                results[victim + task + 'BODEGA'][participant] = score_BODEGA
                results[victim + task + 'c_s'][participant] = score_success
                results[victim + task + 'b_s'][participant] = score_semantic
                results[victim + task + 'l_s'][participant] = score_character
                results[victim + task + 'q'][participant] = all_queries * 1.0 / len(scorer.promises)
            if verify:
                victim_i.finalise()

fil = open(output_path, 'w')
fil.write('\t' + '\t'.join(participants) + '\n')
for victim in victims:
    for task in tasks:
        fil.write(victim + '-' + task + '\n')
        for measure in measures:
            fil.write(measure)
            for participant in participants:
                fil.write('\t' + str(results[victim + task + measure][participant]))
            fil.write('\n')
fil.close()
