import pathlib

for task in ['RD']:
    print(task)
    data_path = pathlib.Path.home() / 'data' / 'BODEGA' / task
    allbad = 0
    allgood = 0
    for subset in ['train', 'test']:
        print(subset)
        bad = 0
        good = 0
        for line in open(data_path / (subset + '.tsv')):
            parts = line.split('\t')
            if parts[0] == '0':
                good += 1
            elif parts[0] == '1':
                bad += 1
            else:
                assert (False)
        print("Cases: " + str(bad + good))
        allbad += bad
        allgood += good
    print("Bad: " + str(bad / (bad + good)))
