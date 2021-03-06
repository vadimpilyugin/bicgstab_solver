import re
import pprint
import os

class Experiment:
  def __init__(self, n, op, time, gflops, speedup, ntr, supercomp):
    self.n = n
    self.op = op
    self.time = time
    self.gflops = gflops
    self.speedup = speedup
    self.ntr = ntr
    self.supercomp = supercomp

  def __repr__(self):
    return f"FROM: {self.supercomp}: N={self.n} {self.op}\ttime={self.time}s GFLOPS={self.gflops} Speedup={self.speedup}X NTR={self.ntr}"

def table(exp, cols, rows, values, cond):
  exp = list(filter(cond, exp))
  first_row = unique(exp, cols)
  first_col = unique(exp, rows)
  hsh = {}
  for k1 in first_col:
    if not k1 in hsh:
      hsh[k1] = {}
    for k2 in first_row:
      if not k2 in hsh[k1]:
        hsh[k1][k2] = []
  for e in exp:
    hsh[value(e, rows)][value(e, cols)].append(value(e, values))
  for k1 in first_col:
    for k2 in first_row:
      hsh[k1][k2].sort()
      if hsh[k1][k2]:
        hsh[k1][k2] = hsh[k1][k2][len(hsh[k1][k2])//2]
  return hsh

# def average_tables(tbls, cols, rows):
#   first_row = unique(exp, cols)
#   first_col = unique(exp, rows)
#   av_hsh = {}
#   for k1 in first_col:
#     if not k1 in av_hsh:
#       av_hsh[k1] = {}
#     for k2 in first_row:
#       ar = []
#       for hsh in tbls:
#         if hsh[k1][k2] != {}:
#           ar.append(float(hsh[k1][k2]))
#       ar.sort()
#       av_hsh[k1][k2] = ar[len(ar)//2]
#   return av_hsh


def print_table(hsh, csv=False, first_col=None, first_row=None):
  if not hsh:
    return
  if first_col is None:
    first_col = sorted(list(hsh.keys()))
  if first_row is None:
    first_row = sorted(list(hsh[first_col[0]].keys()))
  # first_row = unique(exp, cols)
  # first_col = unique(exp, rows)
  # print table
  print("\n")
  if csv:
    for i in first_col:
      for j in first_row:
        print(str(hsh[i][j]), end=",")
      print()
  else:
    for i in ['#']+first_row:
      print(i, end="\t")
    print()
    for i in first_col:
      print(i, end="\t")
      for j in first_row:
        print(hsh[i][j], end="\t")
      print()

def value(ex, field):
  return getattr(ex, field)

def unique(exp, field):
  return sorted(set(map(lambda e: getattr(e, field), exp)))


def read_n(n, fn_mask):
  # format = "%s\ttime=%6.3fs GFLOPS=%6.2f Speedup=%6.2fX NTR=%d\n";
  rgx = re.compile(
    r'(\w+)\stime=\s*([\d.]+)s\sGFLOPS=\s*([\d.]+)\sSpeedup=\s*([\d.]+)X\sNTR=(\d+)'
  )
  s = "solver time= 0.021s GFLOPS=  0.57 Speedup=  1.91X NTR=8"
  assert rgx.match(s)
  experiments = []
  fn = fn_mask % (n,)
  with open(fn, 'r') as f:
    for line in f:
      md = rgx.match(line)
      if md:
        op, time, gflops, speedup, ntr = md.groups()
        e = Experiment(int(n), op, time, gflops, speedup, int(ntr));
        experiments.append(e)
  return experiments


def read_exp(fn, n, supercomp):
  rgx = re.compile(
    r'(\w+)\stime=\s*([\d.]+)s\sGFLOPS=\s*([\d.]+)\sSpeedup=\s*([\d.]+)X\sNTR=(\d+)'
  )
  s = "solver time= 0.021s GFLOPS=  0.57 Speedup=  1.91X NTR=8"
  assert rgx.match(s)
  exp = []
  with open(fn, 'r') as f:
    for line in f:
      md = rgx.match(line)
      if md:
        op, time, gflops, speedup, ntr = md.groups()
        e = Experiment(int(n), op, time, gflops, speedup, int(ntr), supercomp);
        exp.append(e)
  return exp

# def read_ns(ns, fn_mask):
#   exp_full = []
#   for n in ns:
#     exp = read_n(n, fn_mask)
#     exp_full += exp
#   return exp_full

# def read_dir(path):
#   exp = []
#   for fn in os.listdir(path):
#     md = re.match(r'[^_]+_N_(\d+)', fn)
#     n = int(md[1])
#     exp += read_exp(os.path.join(path, fn), n)
#   return exp

def read_all():
  exp = []
  for i in [1,2,3,4]:
    path = f"bl_good_{i}"
    supercomp = "bluegene"
    for fn in os.listdir(path):
      md = re.match(r'[^_]+_N_(\d+)', fn)
      n = int(md[1])
      exp += read_exp(os.path.join(path, fn), n, supercomp)
  for i in [3,4,5,6,7]:
    path = f"good_{i}"
    supercomp = "polus"
    for fn in os.listdir(path):
      md = re.match(r'[^_]+_N_(\d+)', fn)
      n = int(md[1])
      exp += read_exp(os.path.join(path, fn), n, supercomp)
  return exp


def cond_1(e):
  return e.ntr == 1

def cond_2(e):
  return e.n == 10**6

# def max_gflops(exp, mv = None):
#   for e in exp:
#     if mv is None or e.gflops > mv.gflops:
#       mv = e
#   return mv

if __name__ == "__main__":
  tbls = []
  triple = ('ntr', 'op')
  val = 'gflops'
  cond = cond_2

  supid = 'polus'
  mv = None

  if supid == 'bluegene':
    path = 'bl_good_%d'
    for i in [1,2,3,4]:
      exp = read_dir(path % i)
      hsh = table(exp, *triple, val, cond)
      print_table(hsh, *triple)
      print()
      tbls.append(hsh)
      mv = max_gflops(exp, mv)
    print("--------------")
  else:
    path = 'good_%d'
    for i in [3,4,5,6,7]:
      exp = read_dir(path % i)
      hsh = table(exp, *triple, val, cond)
      print_table(hsh, *triple)
      print()
      tbls.append(hsh)
      mv = max_gflops(exp, mv)
    print("--------------")

  hsh = average_tables(tbls, *triple)
  print_table(hsh, *triple)
  print()
  print_table(hsh, *triple, True)
  print()
  print("Max gflops reached at: ", mv)