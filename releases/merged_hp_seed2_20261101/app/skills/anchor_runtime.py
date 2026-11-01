"""Frozen restricted anchor compiler and authorized context projection only."""
import ast,copy,math

def compile_skill(source):
    tree=ast.parse(source)
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef) or tree.body[0].name!='propose':raise ValueError('single propose(ctx) function required')
    f=tree.body[0]
    if [a.arg for a in f.args.args]!=['ctx'] or f.args.defaults or f.decorator_list or f.args.kwarg or f.args.vararg:raise ValueError('invalid signature')
    forbidden=(ast.Import,ast.ImportFrom,ast.While,ast.ClassDef,ast.Lambda,ast.With,ast.Try,ast.Raise,ast.Global,ast.Nonlocal,ast.Delete,ast.Yield,ast.YieldFrom,ast.Await,ast.AsyncFunctionDef,ast.ListComp,ast.DictComp,ast.SetComp,ast.GeneratorExp)
    names={'len','min','max','abs','sum','sorted','enumerate','range','int','float','bool','str','list','dict','set','round','distance'}
    methods={'get','items','keys','values','append','sort','copy','setdefault','startswith','endswith'}
    for n in ast.walk(tree):
        if isinstance(n,forbidden):raise ValueError('forbidden syntax '+type(n).__name__)
        if isinstance(n,ast.FunctionDef) and n is not f:raise ValueError('nested function')
        if isinstance(n,ast.Name) and n.id.startswith('_'):raise ValueError('private name')
        if isinstance(n,ast.Attribute) and (n.attr not in methods or not isinstance(n.ctx,ast.Load)):raise ValueError('forbidden attribute')
        if isinstance(n,ast.Call) and not (isinstance(n.func,ast.Name) and n.func.id in names or isinstance(n.func,ast.Attribute)):raise ValueError('forbidden call')
        if isinstance(n,ast.Constant) and isinstance(n.value,(int,float)) and abs(n.value)>100000:raise ValueError('literal greater than100000')
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Pow) and not (isinstance(n.right,ast.Constant) and n.right.value in (2,.5)):raise ValueError('excessive exponent')
    class Limit(ast.NodeTransformer):
        def visit_For(self,n):
            self.generic_visit(n);n.body.insert(0,ast.Expr(ast.Call(ast.Name('_tick',ast.Load()),[],[])));return n
    tree=ast.fix_missing_locations(Limit().visit(tree));count=[0]
    def tick():
        count[0]+=1
        if count[0]>200000:raise ValueError('loop limit')
    def limited_range(*a):
        r=range(*a)
        if len(r)>2000:raise ValueError('range limit')
        return r
    env=dict(__builtins__={},_tick=tick,distance=math.dist,len=len,min=min,max=max,abs=abs,sum=sum,sorted=sorted,enumerate=enumerate,range=limited_range,int=int,float=float,bool=bool,str=str,list=list,dict=dict,set=set,round=round)
    exec(compile(tree,'<deepseek-skill>','exec'),env)
    def fn(ctx):count[0]=0;return env['propose'](ctx)
    return fn

def context(plan,current,targets,step,clock,response):
    lon,lat=plan['units'][0]['deployment'][:2];sx=111320*math.cos(math.radians(lat))
    def point(p):return [round((p[0]-lon)*sx,1),round((p[1]-lat)*110570,1)]
    publics={t['ref']:t for t in plan['public_targets']};units={};ts={}
    for u,v in current.items():
        t=v['task'];role=t.get('role',u[0])
        if role not in 'HM' or not v.get('visible',True) or v.get('health',0)<=0:continue
        units[u]=dict(role=role,position=point(v['position']),kind=t.get('kind'),target=t.get('target_ref'),remaining_s=v['remaining_life_s'],satellite_active=v['satellite_active'],search_keeper=False)
    for t,v in targets.items():
        if t not in publics:continue
        stamp=float(v.get('timestamp_s',0));age=step+clock-stamp if stamp>0 else None
        ts[t]=dict(type=int(v['type']),position=point(v['position']),age_s=age,freshness_known=age is not None and age>=0,initial_public_health=publics[t]['initial_health'])
    return dict(time_s=step,units=units,targets=ts,public_requirements={t:dict(type=v['type'],initial_health=v['initial_health']) for t,v in publics.items()},
        satellite_window=dict(start=plan['satellite']['planned_trigger_s'],end=plan['satellite']['planned_trigger_s']+plan['satellite']['duration_s']),
        baseline_orders={u:{k:v for k,v in o.items() if k in ('kind','target_ref')} for u,o in response['orders'].items() if u in units})
