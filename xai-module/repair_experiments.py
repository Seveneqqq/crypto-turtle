"""Selective, checkpointed repair of controlled-xai-v4 outputs. No web server."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from configuration.experiment_config import EXPERIMENTS
from utils.experiments.runner import ExperimentRunner
from utils.explainers.xai_factory import XaiFactory
from utils.evaluation.paired_evaluation import Perturbation, explain_sample, evaluate_stability, summary
from utils.evaluation.output_audit_v5 import ScalarOutput, OutputPermutation, masking_reports, error_explanations
from utils.evaluation.statistical_analysis import noise_comparisons, masking_inference
from utils.evaluation.xai_local import XaiLocal
from utils.reporting.provenance import provenance, file_hash

PROTOCOL = 'output-aligned-repair-v5'
REUSE_FILES = ['utils/preprocessing/pipeline_preprocessor.py', 'utils/experiments/runner.py',
    'utils/models/model_factory.py', 'utils/data/loaders/generic_loader.py',
    'utils/evaluation/paired_evaluation.py', 'utils/evaluation/feature_mapping.py',
    'utils/explainers/shap_explainer.py', 'utils/explainers/lime_explainer.py',
    'utils/explainers/output_contract.py', 'utils/explainers/xai_factory.py']


def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def discover(source):
    found = {}
    for path in sorted(Path(source).glob('*/results.json')):
        for record in read(path):
            if record.get('provenance',{}).get('protocol') != 'controlled-xai-v4': continue
            if record['dataset'] not in EXPERIMENTS: continue
            if any(len(v.get('stability_details',{}).get('repeats',[])) < 20 for v in record['xai'].values()): continue
            key = record['dataset']+'::'+record['model']
            found[key] = {'path':str(path.resolve()), 'sha256':file_hash(path)}
    return found


def load_source(entry, name, model):
    if entry is None: return None
    if file_hash(entry['path']) != entry['sha256']: raise ValueError('Source results changed')
    return next(r for r in read(entry['path']) if r['dataset']==name and r['model']==model)


def identity(config):
    result = provenance(config)
    result['protocol'] = PROTOCOL
    result['repair_runner_sha256'] = file_hash(__file__)
    result['fingerprint'] = hashlib.sha256((result['fingerprint']+result['repair_runner_sha256']).encode()).hexdigest()
    return result


def validate_source(old, current, config):
    if old is None: return
    original = old['provenance']
    for key in ['data_sha256', 'packages', 'python']:
        if original[key] != current[key]: raise ValueError('Cannot reuse: changed '+key)
    if original['config'] != config: raise ValueError('Cannot reuse: changed experiment config')
    for path in REUSE_FILES:
        if original['code_sha256'].get(path) != current['code_sha256'].get(path):
            raise ValueError('Cannot reuse: changed '+path)


def validate_model(old, fitted):
    if old is None: return
    if clean(old['data_preparation']) != clean(fitted['data_preparation']):
        raise ValueError('Rebuilt data/model configuration differs; refusing reuse')
    for key,value in old['model_metrics'].items():
        if isinstance(value,(int,float)) and not np.isclose(value,fitted['metrics'][key], rtol=1e-7, atol=1e-10):
            raise ValueError('Rebuilt model metric differs: '+key)
    f = old['xai']['shap']['fidelity']
    actual = fitted['pipeline'].predict(fitted['X_test'].loc[f['sample_indices']])
    match = (np.array_equal(actual,f['baseline_predictions']) if old['task']=='classification'
             else np.allclose(actual,f['baseline_predictions'],rtol=1e-7,atol=1e-10))
    if not match: raise ValueError('Rebuilt model predictions differ; refusing reuse')


def export_method(folder, method, record):
    pd.DataFrame(list(record['global_importance'].items()),columns=['feature','importance']).to_csv(folder/(method+'_importance.csv'),index=False)
    pd.DataFrame([{k:q[k] for k in ['seed','global_spearman','global_relative_l1']}
                  for q in record['stability_details']['repeats']]).to_csv(folder/(method+'_stability.csv'),index=False)
    for kind in ['performance_masking','output_sensitivity']:
        pd.DataFrame([{k:v for k,v in q.items() if k!='predictions'} for q in record[kind]['variants']]).to_csv(folder/(method+'_'+kind+'.csv'),index=False)


def run_job(name, model, config, old, source_entry, folder, current):
    folder.mkdir(parents=True,exist_ok=True)
    statefile = folder/'checkpoint.json'
    expected = {'fingerprint':current['fingerprint'], 'source':source_entry}
    state = read(statefile) if statefile.exists() else dict(expected, files={})
    if any(state.get(k)!=v for k,v in expected.items()):
        raise ValueError('Checkpoint code/config/source changed. Choose a new --output directory.')
    for filename,digest in state['files'].items():
        if not (folder/filename).is_file() or file_hash(folder/filename)!=digest:
            raise ValueError('Checkpoint missing or altered: '+filename)
    def commit(filename):
        state['files'][filename]=file_hash(folder/filename)
        save(statefile,state)
    if 'results.json' in state['files']:
        print('SKIP complete:', name, model, flush=True)
        return
    if 'model.joblib' in state['files']:
        fitted = joblib.load(folder/'model.joblib')
        print('Loaded model checkpoint',flush=True)
    else:
        print('Training model once (old run did not save models)',flush=True)
        fitted = ExperimentRunner(config,model).run()
        validate_model(old,fitted)
        temporary=folder/'model.joblib.tmp'
        joblib.dump(fitted,temporary,compress=3)
        os.replace(temporary,folder/'model.joblib')
        commit('model.joblib')
    validate_model(old,fitted)
    train,test,y = fitted['X_train'],fitted['X_test'],fitted['y_test']
    pipeline=fitted['pipeline'];seed=int(config.get('xai_seed',42))
    midpoint=len(test)//2
    sample=test.iloc[:midpoint].sample(min(config.get('xai_sample_size',50),midpoint),random_state=seed)
    pool=test.iloc[midpoint:];n=min(config.get('masking_sample_size',500),len(pool))
    mask=pool.iloc[:n] if config.get('split_type')=='time' else pool.sample(n,random_state=seed)
    if set(sample.index)&set(mask.index): raise ValueError('Explanation/masking overlap')
    output=ScalarOutput(pipeline,config['task'],config.get('xai_class_index'))
    output.cache_baseline(mask)
    noise=Perturbation(train,config.get('stability_noise_level',.01),config.get('stability_discrete_probability',.01),config.get('stability_feature_types'))
    methods={}
    for method in ['shap','lime','permutation']:
        filename=method+'.json'
        if filename in state['files']:
            methods[method]=read(folder/filename)
            print('SKIP completed method:',method,flush=True)
            continue
        print('METHOD',method,flush=True)
        explainer=(OutputPermutation(output) if method=='permutation' else
                   XaiFactory.create(method,pipeline,train,task=config['task'],config=config))
        importance,_=explain_sample(method,explainer,sample,y.loc[sample.index],seed)
        reuse=old is not None and method in ('shap','lime')
        stability_file=method+'_stability_checkpoint.json'
        if stability_file in state['files']:
            cached=read(folder/stability_file)
            if set(importance)!=set(cached['importance']) or any(not np.isclose(v,cached['importance'][k],rtol=1e-5,atol=1e-10) for k,v in importance.items()):
                raise ValueError('Checkpoint ranking differs after restart')
            stability,origin=cached['stability'],cached['origin']
            print('SKIP completed stability:',method,flush=True)
        elif reuse:
            previous=old['xai'][method]
            if previous['evaluation_sample_indices']!=sample.index.tolist() or previous['output_metadata']!=output.metadata():
                raise ValueError('Reuse sample/output mismatch')
            if method=='shap' and previous['background_indices']!=explainer.background_indices:
                raise ValueError('Reuse SHAP background mismatch')
            if set(importance)!=set(previous['global_importance']) or any(not np.isclose(v,previous['global_importance'][k],rtol=1e-5,atol=1e-10) for k,v in importance.items()):
                raise ValueError('Recomputed explanation ranking differs; refusing stability reuse')
            stability=copy.deepcopy(previous['stability_details'])
            origin={'kind':'reused_after_reproduction_checks','source':source_entry,'method':method,
                    'validation':'matching code/data/packages/split/metrics/masking predictions/ranking; not bitwise proof of model identity'}
            print('Reused',len(stability['repeats']),'validated stability repetitions:',method,flush=True)
        else:
            seeds=list(config.get('stability_seeds',list(range(101,121))))
            if len(seeds)<2 or len(set(seeds))!=len(seeds): raise ValueError('Need distinct stability seeds')
            records=[]
            for offset in range(0,len(seeds),2):
                batch=seeds[offset:offset+2]
                partfile=method+'_repeats_'+str(offset)+'.json'
                if partfile in state['files']:
                    part=read(folder/partfile)
                    print('SKIP stability seeds:',batch,flush=True)
                else:
                    pair=batch if len(batch)==2 else [seeds[0],batch[0]]
                    part=evaluate_stability(method,explainer,sample,y.loc[sample.index],noise,pair)
                    save(folder/partfile,part);commit(partfile)
                records.extend(q for q in part['repeats'] if q['seed'] in batch)
            stability=copy.deepcopy(part);stability['repeats']=records
            for field,key,control in [('global_summary','global_spearman',False),
                ('data_relative_l1_summary','global_relative_l1',False),
                ('randomness_summary','global_spearman',True),
                ('randomness_relative_l1_summary','global_relative_l1',True)]:
                vals=[(q['randomness_control'] if control else q)[key] for q in records]
                stability[field]=summary([np.nan if v is None else v for v in vals])
            origin={'kind':'new_measurement','reason':'changed_PFI_score' if method=='permutation' else 'missing_completed_result'}
        if stability_file not in state['files']:
            save(folder/stability_file,{'importance':importance,'stability':stability,'origin':origin})
            commit(stability_file)
        performance,sensitivity=masking_reports(output,train,mask,y.loc[mask.index],importance,
            k=config.get('masking_k',5),seeds=config.get('masking_seeds',list(range(501,521))))
        local=XaiLocal.generate_local_explanations(method,explainer,sample) if method!='permutation' else []
        errors=error_explanations(method,explainer,test,y,fitted['y_pred'],config['task']) if method!='permutation' else []
        record={'global_importance':importance,'stability_details':stability,'stability_origin':origin,
            'evaluation_sample_indices':sample.index.tolist(),'output_metadata':output.metadata(),
            'background_indices':getattr(explainer,'background_indices',None),
            'performance_masking':performance,'output_sensitivity':sensitivity,
            'local_explanations':local,'debug_explanations':errors,
            'importance_definition':('permutation_increase_in_root_brier_or_RMSE' if method=='permutation' else
                                    'mean_sum_absolute_transformed_weights_per_raw_feature')}
        save(folder/filename,record);commit(filename)
        export_method(folder,method,record);methods[method]=record
    temporal=fitted['data_preparation']['chronology'].startswith('ascending:')
    statistics={}
    for kind in ['performance_masking','output_sensitivity']:
        reports={m:{'fidelity':r[kind]} for m,r in methods.items()}
        statistics[kind]=masking_inference(reports,temporal=temporal,
            horizon=config.get('finance_horizon',1),resamples=config.get('bootstrap_resamples',1999),seed=seed)
        statistics[kind]['measurement']=next(iter(methods.values()))[kind]['measurement']
        pd.DataFrame(statistics[kind]['intervals']).to_csv(folder/(kind+'_intervals.csv'),index=False)
        pd.DataFrame(statistics[kind]['comparisons']).to_csv(folder/(kind+'_statistics.csv'),index=False)
        if temporal:
            length=statistics[kind]['block_length']
            statistics[kind]['block_length_sensitivity']={str(b):masking_inference(reports,temporal=True,
                horizon=config.get('finance_horizon',1),resamples=config.get('bootstrap_resamples',1999),seed=seed,block_length=b)
                for b in sorted({max(config.get('finance_horizon',1),length//2),min(len(mask),length*2)})}
    statistics['shap_lime_stability']=noise_comparisons({m:methods[m] for m in ['shap','lime']},seed=seed)
    pd.DataFrame(statistics['shap_lime_stability']).to_csv(folder/'shap_lime_stability_statistics.csv',index=False)
    result={'protocol':PROTOCOL,'dataset':name,'model':model,'task':config['task'],
        'model_metrics':fitted['metrics'],'data_preparation':fitted['data_preparation'],
        'provenance':current,'source_record':source_entry,'xai':methods,'statistics':statistics,
        'scope':{'model':'one fitted model; conditional inference only','sample_size':len(sample),
                 'class':output.metadata(),'masking':'interventional diagnostics, not complete fidelity',
                 'medical_ordered_holdout':'not a temporal experiment',
                 'PFI':'loss-based importance; not attribution values',
                 'multiple_testing':'Holm within each dataset/model/measurement family only',
                 'runtime':'not benchmarked in repair mode'}}
    save(folder/'results.json',result);commit('results.json')


def main():
    parser=argparse.ArgumentParser(description=PROTOCOL)
    parser.add_argument('--source',type=Path,default=Path('results/experiments'))
    parser.add_argument('--output',type=Path,default=Path('results/repaired_v5'))
    parser.add_argument('--all',action='store_true')
    parser.add_argument('--experiment',action='append',choices=list(EXPERIMENTS))
    parser.add_argument('--model',action='append')
    parser.add_argument('--plan',action='store_true',help='Validate and print plan without training')
    args=parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if args.all==bool(args.experiment): parser.error('Choose --all or --experiment NAME')
    if not args.source.is_dir(): parser.error('Source directory does not exist')
    manifest_path=args.output/'repair_manifest.json'
    selection={name:args.model or EXPERIMENTS[name]['models'] for name in (EXPERIMENTS if args.all else args.experiment)}
    for name,models in selection.items():
        if set(models)-set(EXPERIMENTS[name]['models']): parser.error('Invalid model for '+name)
    manifest=read(manifest_path) if manifest_path.exists() else {'protocol':PROTOCOL,'selection':selection,'sources':discover(args.source)}
    if manifest['selection']!=selection: parser.error('Selection changed; use a different --output folder')
    jobs=[]
    for name,models in selection.items():
        config=dict(EXPERIMENTS[name]);current=identity(config)
        for model in models:
            entry=manifest['sources'].get(name+'::'+model);old=load_source(entry,name,model)
            validate_source(old,current,config)
            print(name,model,': reuse SHAP/LIME; new PFI, masking, error explanations' if old else ': missing -> full measurement',flush=True)
            jobs.append((name,model,config,old,entry,args.output/name/model,current))
    if args.plan:
        print('PLAN ONLY: no models trained, no results changed. Jobs:',len(jobs));return 0
    save(manifest_path,manifest)
    failures=[]
    for job in jobs:
        try: run_job(*job)
        except KeyboardInterrupt:
            print('Interrupted. Run the SAME command to resume.');return 130
        except Exception:
            failures.append(job[0]+'::'+job[1]);traceback.print_exc()
    save(args.output/'status.json',{'failed':failures,'planned':len(jobs)})
    print('Finished. Failed jobs:',failures)
    return int(bool(failures))


if __name__=='__main__':
    raise SystemExit(main())
