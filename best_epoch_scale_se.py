import pickle
import glob
import shutil
import os


datasets = ['amazon_baby'] # ['amazon_music'] # ['amazon_baby', 'facebook_books']
backbones = ['BPRMF'] # ['BPRMF','NGCF']
method = 'AMORE_SCALE'
scales = ['0$25', '0$5', '0$75', '0$95'] # ['0$25', '0$3', '0$4', '0$5', '0$6', '0$7', '0$75']
for dataset in datasets:
    for backbone in backbones:
        for scale in scales:
            if method == 'AMORE_SCALE':
                for file in glob.glob(f'results/{dataset}/performance/*{backbone}*rme*'):
                    with open(file, 'rb') as handle:
                        store_validation = pickle.load(handle)
                    for k, v in list(store_validation.items()):
                        # if scale not in k:
                        if not k.endswith(scale):
                            del store_validation[k]
                    for k, v in store_validation.items():
                        store_validation[k] = sorted(v, key=lambda x: x[1], reverse=True)[0]
                    if store_validation != {}:
                        maximumValue = max(store_validation.values(), key=lambda k: k[1])
                        maxKey = next(k for k, v in store_validation.items() if v == maximumValue)
                        if method in maxKey:
                            print(maxKey, maximumValue)
                            rec = f'results/{dataset}/recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                            if not os.path.exists(f'results/{dataset}/best_recs'):
                                os.makedirs(f'results/{dataset}/best_recs')
                            dst = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_recs.tsv'
                            shutil.copyfile(rec, dst)
                            loss = f'results/{dataset}/losses/{maxKey}_loss.pkl'
                            dst_loss = f'results/{dataset}/best_recs/{maxKey}_it={maximumValue[0]}_loss.pkl'
                            shutil.copyfile(loss, dst_loss)


