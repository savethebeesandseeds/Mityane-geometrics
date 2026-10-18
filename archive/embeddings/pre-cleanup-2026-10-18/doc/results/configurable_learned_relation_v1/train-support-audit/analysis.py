import importlib.util, json, hashlib, math
from pathlib import Path
root=Path('/embedding')
assert Path('/.dockerenv').exists() and Path.cwd()==root
module_path=root/'code/evaluation/protocols/learned_train_v1/analysis.py'
spec=importlib.util.spec_from_file_location('support_train_readonly', module_path)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
audited=[module_path, root/'code/evaluation/protocols/visible_difference_v1/reader-v2/archive_codec.py']
receipts={str(p):a.sha(p) for p in audited}
results=[]
for master in (941101,941102):
    mask_path=root/f'output/runs/rpb-v20-capacity/admission-0gN4uD/quality-data/master-{master}/training.pt'
    cp_path=root/f'output/runs/rpb-v21-learned-relation/admission-0PGNmn/candidates/master-{master}/point-512/checkpoint.pt'
    feature_path=cp_path.parent/'features.pt'
    for p in (mask_path,cp_path,feature_path):receipts[str(p)]=a.sha(p)
    mask=a.Archive(mask_path,{'feature_mask'}).tensor('feature_mask','BoolStorage',[512,3,128,9])
    weight=a.Archive(cp_path,{'model.relation_projection.weight'},checkpoint=True).tensor('model.relation_projection.weight','FloatStorage',[20,2916])
    features=a.Archive(feature_path,{'training_values'}).tensor('training_values','DoubleStorage',[512,32])
    weights=[weight[row*2916+1944:row*2916+2916] for row in range(20)]
    assert all(math.isfinite(v) for row in weights for v in row)
    contributions=[];fractions=[];mask_pairs_equal=True;support_pairs_equal=True
    for b in range(512):
        bits=[[sum(int(mask[((b*3+c)*128+t)*9+f])<<t for t in range(128)) for f in range(9)] for c in range(3)]
        q=[]
        for i in range(3):
            for j in range(i+1,3):
                for fi in range(9):
                    for fj in range(9):
                        for lag in (1,2,4,8):
                            n=(bits[i][fi] & (bits[i][fi]>>lag) & bits[j][fj] & (bits[j][fj]>>lag)).bit_count()
                            q.append(n/(128-lag))
        assert len(q)==972
        fractions.extend(q)
        contribution=[math.fsum(w*v for w,v in zip(row,q)) for row in weights]
        contributions.append(contribution)
        if b%2:
            mask_pairs_equal=mask_pairs_equal and mask[b*3456:(b+1)*3456]==mask[(b-1)*3456:b*3456]
            support_pairs_equal=support_pairs_equal and contributions[b]==contributions[b-1]
    support_mean=[math.fsum(row[d] for row in contributions)/512 for d in range(20)]
    support_centered_rms=math.sqrt(math.fsum((row[d]-support_mean[d])**2 for row in contributions for d in range(20))/(512*20))
    native_means=[math.fsum(features[b*32+d] for b in range(512))/512 for d in range(20)]
    native_centered_rms=math.sqrt(math.fsum((features[b*32+d]-native_means[d])**2 for b in range(512) for d in range(20))/(512*20))
    support_rms=math.sqrt(math.fsum(v*v for row in contributions for v in row)/(512*20))
    results.append(dict(master=master,support_weights_nonzero=sum(v!=0 for row in weights for v in row),support_weight_count=19440,
        mean_support_fraction=math.fsum(fractions)/len(fractions),support_fraction_min=min(fractions),support_fraction_max=max(fractions),
        support_component_uncentered_rms=support_rms,support_component_centered_rms=support_centered_rms,
        support_component_mean_l2=math.sqrt(math.fsum(v*v for v in support_mean)),learned20_centered_rms=native_centered_rms,
        TRAIN_pair_masks_exact=mask_pairs_equal,TRAIN_pair_support_component_exact=support_pairs_equal,
        expected_support_component_change_rms_if_independent_30pct_deletion=(1-.7**4)*support_rms))
assert all(a.sha(Path(p))==digest for p,digest in receipts.items())
print(json.dumps(dict(scope='TRAIN-only read-only structural support component diagnostic; not observed deletion or causal attribution',
    extra_deletion_expectation_factor=.7**4,encoder_calls=0,head_calls=0,updates=0,decoded=a.DECODED,
    source_and_input_hashes=receipts,cohorts=results),indent=2,default=lambda v:sorted(v)))
