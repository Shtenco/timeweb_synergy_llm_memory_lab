"""Offline synthetic fixtures and optional external dataset adapters.
All synthetic fixtures intentionally contain authored answer facts and are NOT independent real-world evidence.
No data is downloaded automatically.
"""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path


def _split(i, threshold):
    return 'dev' if i <= threshold else 'test'


def projects(seed=42, noisy=True):
    """Self-contained v2 fixture; it is intentionally distinct from archived v5 results."""
    rng = random.Random(seed)
    owners = ['Арина', 'Борис', 'Вера', 'Глеб', 'Дина', 'Егор']
    regions = ['ru-central', 'ru-north', 'eu-west', 'ap-south']
    retention = [7, 14, 30, 60, 90]
    docs = []
    questions = []
    for i in range(1, 25):
        project = f'P-{i:02}'
        service = f'S-{i:02}'
        owner = owners[(i * 5 + seed) % len(owners)]
        region = regions[(i * 3 + seed) % len(regions)]
        days = retention[(i * 2 + seed) % len(retention)]
        docs.extend([
            {
                'id': f'project-{project}',
                'text': f'Карточка проекта {project}. Ответственный: {owner}. Используется сервис {service}.',
                'links': [f'service-{service}'],
            },
            {
                'id': f'service-{service}',
                'text': f'Паспорт сервиса {service}. Регион размещения: {region}. Хранение журналов: {days} дней.',
                'links': [],
            },
        ])
        if noisy:
            docs.extend([
                {'id': f'noise-{i:02}-0', 'text': f'Архив обсуждений проекта {project}; данных о владельце нет.', 'links': []},
                {'id': f'noise-{i:02}-1', 'text': f'Черновик сервиса {service}; старые сведения, не использовать.', 'links': []},
            ])
        split = _split(i, 12)
        questions.extend([
            {
                'id': f'{project}-owner', 'question': f'Кто отвечает за проект {project}?',
                'answer': owner, 'gold': [f'project-{project}'], 'kind': 'single', 'split': split,
            },
            {
                'id': f'{project}-region', 'question': f'В каком регионе размещён сервис проекта {project}?',
                'answer': region, 'gold': [f'project-{project}', f'service-{service}'],
                'kind': 'two_hop', 'split': split,
            },
            {
                'id': f'{project}-retention', 'question': f'Сколько дней хранятся журналы сервиса проекта {project}?',
                'answer': f'{days} дней', 'gold': [f'project-{project}', f'service-{service}'],
                'kind': 'two_hop', 'split': split,
            },
        ])
    for i in range(1, 9):
        missing_project = f'P-{900 + i:02}'
        questions.append({
            'id': f'{missing_project}-missing',
            'question': f'Кто отвечает за проект {missing_project}?',
            'answer': 'НЕТ ДАННЫХ', 'gold': [], 'kind': 'unanswerable',
            'split': _split(i, 4),
        })
    rng.shuffle(docs)
    return docs, questions, {
        'dataset': 'synthetic_projects_v2', 'origin': 'bundled_generator_v2',
        'synthetic': True, 'seed': seed, 'generator_version': 2,
    }


def incidents(seed=1107, noisy=True):
    """Independent incident/runbook/escalation QA domain, code IDs do not encode answers."""
    rng = random.Random(seed)
    owners=['Анна','Михаил','Дарья','Арсен','Никита','Валерия','Игорь']
    commands=['rollback','restart-service','restore-snapshot','switch-region','rotate-secret','drain-node']
    teams=['SRE-1','SRE-2','Platform','Storage','SecOps','OnCall']
    docs=[]; qs=[]
    for i in range(1,37):
        app=f'APP-{i:03d}'; run=f'RUN-{i:03d}'; esc=f'ESC-{i:03d}'
        owner=owners[(i*5)%len(owners)]
        cmd=commands[(i*7)%len(commands)]
        team=teams[(i*11)%len(teams)]
        docs.extend([
            {'id':f'inc-app-{i}', 'text':f'Система {app}. Дежурный инженер {owner}. При отказе использовать документ {run}.', 'links':[f'inc-run-{i}']},
            {'id':f'inc-run-{i}', 'text':f'Рабочая инструкция {run}. Основная операция восстановления {cmd}. Если не помогло, открыть {esc}.', 'links':[f'inc-esc-{i}']},
            {'id':f'inc-esc-{i}', 'text':f'Порядок эскалации {esc}. Ответственная команда {team}.', 'links':[]},
        ])
        split=_split(i,18)
        qs.extend([
            {'id':f'inc-owner-{i}','question':f'Кто дежурит по системе {app}?','answer':owner,'gold':[f'inc-app-{i}'],'kind':'single','split':split},
            {'id':f'inc-cmd-{i}','question':f'Какой основной шаг восстановления предусмотрен для системы {app}?','answer':cmd,'gold':[f'inc-app-{i}', f'inc-run-{i}'],'kind':'two_hop','split':split},
            {'id':f'inc-esc-{i}','question':f'Какой команде эскалировать аварию системы {app} после неудачного восстановления?','answer':team,'gold':[f'inc-app-{i}',f'inc-run-{i}',f'inc-esc-{i}'],'kind':'three_hop','split':split},
        ])
        if noisy:
            docs.append({'id':f'inc-noise-{i}','text':f'Исторический архив {app}. Инструкция {run} была неоднократно обсуждена. Неоперационный черновик.', 'links':[]})
    for i in range(1,13):
        qs.append({'id':f'inc-missing-{i}','question':f'Кто дежурит по системе APP-{900+i}?','answer':'НЕТ ДАННЫХ','gold':[],'kind':'unanswerable','split':_split(i,6)})
    rng.shuffle(docs)
    return docs,qs,{'dataset':'synthetic_incidents','origin':'new_independent_generator','synthetic':True,'seed':seed}


def revisions(seed=2361, noisy=True):
    """Third, independently authored domain: policy version resolution and stale records."""
    rng=random.Random(seed)
    envs=['производственный','тестовый','архивный','изолированный','резервный']
    actions=['проверка подписи','подтверждение администратора','изоляция процесса','двойная проверка','журналирование']
    periods=['3 часа','5 часов','10 часов','1 день','2 дня','4 дня']
    docs=[];qs=[]
    for i in range(1,37):
        ref=f'POL-{i:03d}'; latest=f'ED-{i:03d}-v2'; old=f'ED-{i:03d}-v1'
        action=actions[(i*13)%len(actions)]; environment=envs[(i*3)%len(envs)]
        period=periods[(i*11)%len(periods)]
        docs.extend([
            {'id':f'pol-root-{i}','text':f'Политика {ref}. Действующая редакция: {latest}. Прежняя: {old}.', 'links':[f'pol-now-{i}']},
            {'id':f'pol-now-{i}','text':f'Редакция {latest} является текущей. Контроль: {action}. Среда: {environment}. Срок проверки: {period}.', 'links':[]},
        ])
        if noisy:
            docs.append({'id':f'pol-old-{i}', 'text':f'Редакция {old} устарела. Старый контроль: журналирование. Старая среда: архивный. Срок проверки: 99 дней.', 'links':[]})
            docs.append({'id':f'pol-noise-{i}', 'text':f'Общее обсуждение {ref}. Этот документ не задает текущие требования.', 'links':[]})
        split=_split(i,18)
        qs.extend([
            {'id':f'pol-action-{i}','question':f'Какое действующее контрольное действие в политике {ref}?','answer':action,'gold':[f'pol-root-{i}',f'pol-now-{i}'],'kind':'two_hop','split':split},
            {'id':f'pol-env-{i}','question':f'Какая текущая среда у политики {ref}?','answer':environment,'gold':[f'pol-root-{i}',f'pol-now-{i}'],'kind':'two_hop','split':split},
            {'id':f'pol-time-{i}','question':f'Какой действующий срок проверки для политики {ref}?','answer':period,'gold':[f'pol-root-{i}',f'pol-now-{i}'],'kind':'two_hop','split':split},
        ])
    for i in range(1,13):
        qs.append({'id':f'pol-missing-{i}','question':f'Какой текущий срок проверки для политики POL-{950+i}?','answer':'НЕТ ДАННЫХ','gold':[],'kind':'unanswerable','split':_split(i,6)})
    rng.shuffle(docs)
    return docs,qs,{'dataset':'synthetic_revisions','origin':'new_independent_generator','synthetic':True,'seed':seed}


def external_jsonl(path: str):
    """Import user-supplied explicit corpus and question lines. No network access.
    Format: each line {type:'doc',id,text,links?} or
                 {type:'qa',id,question,answer,gold:[doc-id],kind,split}.
    """
    docs=[]; questions=[]
    for lineno,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        o=json.loads(line)
        if o['type']=='doc':
            assert isinstance(o['text'],str) and o['text'].strip(),f'bad doc at {lineno}'
            docs.append({k:o[k] for k in ('id','text','links') if k in o})
        elif o['type']=='qa':
            assert o.get('split') in {'dev','test'},f'missing dev/test at {lineno}'
            questions.append({k:o[k] for k in ('id','question','answer','gold','kind','split')})
        else: raise ValueError(f'bad line type at {lineno}')
    ids={d['id'] for d in docs}
    if len(ids)!=len(docs): raise ValueError('duplicate document ids')
    for q in questions:
        if not set(q['gold']).issubset(ids): raise ValueError(f'missing support {q["id"]}')
    if not questions or not docs: raise ValueError('Empty corpus or questions')
    sha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return docs,questions,{'dataset':Path(path).stem,'origin':'user_supplied','synthetic':False,'sha256':sha,'label_provenance':'user_provided_unverified'}

REGISTRY={'projects':projects, 'incidents':incidents, 'revisions':revisions}

if __name__=='__main__':
    for name,fn in REGISTRY.items():
        d,q,m=fn();print(name,len(d),len(q),m)
