import glob, time, multiprocessing as mp
def temps():
    vals=[]
    for d in glob.glob('/sys/class/hwmon/hwmon*'):
        if open(d+'/name').read().strip()=='coretemp':
            for f in glob.glob(d+'/temp*_input'): vals.append(int(open(f).read())//1000)
    return max(vals), sorted(vals)[len(vals)//2]
def spin(t):
    e=time.time()+t
    while time.time()<e: pass
print('idle max/median', temps())
ps=[mp.Process(target=spin,args=(4,)) for _ in range(8)]; [p.start() for p in ps]
t0=time.time()
while any(p.is_alive() for p in ps):
    print(f't={time.time()-t0:.1f} max/median', temps()); time.sleep(0.5)
for i in range(6): time.sleep(1); print(f'after+{i+1}s', temps())
