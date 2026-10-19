"""Standalone research runner; no web server, RabbitMQ, or automatic reload."""
import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from configuration.experiment_config import EXPERIMENTS
from utils.experiments.experiment_pipeline import ExperimentPipeline
from utils.reporting.provenance import provenance, file_hash


def save(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(temporary,path)


def main():
    parser = argparse.ArgumentParser(description='Powtarzalne eksperymenty XAI, protokol controlled-xai-v4')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--experiment', action='append', choices=list(EXPERIMENTS))
    parser.add_argument('--model', action='append')
    parser.add_argument('--resume', type=Path, help='Sciezka do suite_manifest.json z przerwanego przebiegu')
    parser.add_argument('--quick', action='store_true', help='Krotkie sprawdzenie, NIE wyniki docelowe')
    args = parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    os.chdir(Path(__file__).resolve().parent)
    if args.list:
        for name,config in EXPERIMENTS.items():
            print(name, ':', ', '.join(config['models']))
        return 0
    if args.resume:
        manifest_path = args.resume.resolve()
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if args.all or args.experiment or args.model or args.quick:
            parser.error('--resume wykorzystuje wybor zapisany w pliku; nie lacz z innymi opcjami')
        selection, quick = manifest['selection'], manifest['quick']
    else:
        if args.all == bool(args.experiment):
            parser.error('Wybierz --all albo --experiment NAZWA; --list pokazuje nazwy')
        selection = {}
        for name in (list(EXPERIMENTS) if args.all else args.experiment):
            models = args.model or EXPERIMENTS[name]['models']
            if set(models)-set(EXPERIMENTS[name]['models']):
                parser.error(f'Nieprawidlowe modele dla {name}: {models}')
            selection[name] = list(dict.fromkeys(models))
        quick = args.quick
        folder = Path('results')/('suite_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f'))
        folder.mkdir(parents=True)
        manifest_path = folder/'suite_manifest.json'
        manifest = {'protocol':'controlled-xai-v4','selection':selection,'quick':quick,'jobs':{}}
    configs = {}
    for name,models in selection.items():
        config = dict(EXPERIMENTS[name])
        if quick:
            config.update(xai_sample_size=4, shap_background_size=5, shap_permutations=1,
                          lime_num_samples=200, stability_seeds=[101,102], masking_seeds=[501,502],
                          masking_sample_size=40, bootstrap_resamples=99, permutation_resamples=99)
        config['run_mode'] = 'quick_check_not_thesis_results' if quick else 'full'
        config['dataset_name'] = ('quick_' if quick else '')+config['dataset_name']
        identity = provenance(config)
        configs[name] = (config,identity)
        for model in models:
            key = name+'::'+model
            old = manifest['jobs'].get(key)
            if old and old['fingerprint'] != identity['fingerprint']:
                raise RuntimeError(f'Zmiana kodu/danych/ustawien/srodowiska dla {key}. Uruchom nowy przebieg.')
            manifest['jobs'].setdefault(key,{'fingerprint':identity['fingerprint'],'status':'pending'})
    save(manifest_path,manifest)
    print('Manifest przebiegu:',manifest_path.resolve(), flush=True)
    failures = 0
    for name,models in selection.items():
        config,identity = configs[name]
        for model in models:
            key = name+'::'+model
            job = manifest['jobs'][key]
            if job['status']=='complete' and Path(job.get('results_file','')).is_file():
                if file_hash(job['results_file']) != job.get('results_sha256'):
                    raise RuntimeError(f'Zmieniony plik wynikow dla {key}; sprawdz go przed wznowieniem.')
                print('Pominieto ukonczony:',key,flush=True)
                continue
            job['status']='running'
            save(manifest_path,manifest)
            try:
                pipeline = ExperimentPipeline(config,[model])
                pipeline.run()
                job.update(status='complete',results_file=str((Path(pipeline.output_dir)/'results.json').resolve()))
                job['results_sha256']=file_hash(job['results_file'])
                job.pop('error',None)
            except KeyboardInterrupt:
                job['status']='interrupted'
                save(manifest_path,manifest)
                print('Przerwano. Wznowienie: python run_experiments.py --resume "'+str(manifest_path.resolve())+'"')
                return 130
            except Exception:
                failures += 1
                job.update(status='failed',error=traceback.format_exc())
                print(job['error'],flush=True)
            save(manifest_path,manifest)
    print('Koniec przebiegu. Bledy:',failures,flush=True)
    return 1 if failures else 0


if __name__=='__main__':
    raise SystemExit(main())
