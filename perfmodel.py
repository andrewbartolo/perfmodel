#!/usr/bin/env python3

import numpy as np
import sys, os

# NOTE -- treats startup time as negligible, as only data from MPI processes
#         are captured anyway


# all in SI
PARAMS = {'clockFreq':    2.8e9,
          'IPC':          1,
          'cHitLat':      5e-9,
          'mReadLat':     60e-9,
          'mWriteLat':    60e-9,
          'sendLat':      1.5e-6,
          'sendOverlap':  0
         }



CSV_HDR = ['rank', 'instrs', 'oInstrs', 'sends', 'rds', 'rdCpts', 'rdCptPct',
           'wrs', 'wrCpts', 'wrCptPct', 'cHits', 'cMisses', 'mReads', 'mWrites',
           'fullDumps', 'partDumps']

################################################################################
################################### Models #####################################
################################################################################
class BasicModel:
    def __init__(s, csv):
        procTimes = list(map(s.procTime, csv))
        print(procTimes)

        s.tTime = np.max(procTimes) + s.tOverhead()

    def runtime(s):
        return s.tTime

    def tCompute(s, proc):
        return (proc['instrs'] / PARAMS['IPC']) / PARAMS['clockFreq']

    def tCacheAndMem(s, proc):
        return proc['cHits'] * PARAMS['cHitLat'] +   \
               proc['mReads'] * PARAMS['mReadLat'] + \
               proc['mWrites'] * PARAMS['mWriteLat']

    def tCommunication(s, proc):
        return (1 - PARAMS['sendOverlap']) * proc['sends'] * PARAMS['sendLat']

    def procTime(s, proc):
        return s.tCompute(proc) + s.tCacheAndMem(proc) + s.tCommunication(proc)

    def tOverhead(s):
        return 0.1

################################################################################


def parse(filename):
    # headers not automatically present for generated csvs
    # TODO some columns should be ints (not all should be floats)
    csv = np.genfromtxt(filename, delimiter=',', names=CSV_HDR)
    csv = np.sort(csv, axis=0, order='rank')

    # warn about missing parameters
    if np.sum(csv['oInstrs'] == 0):
        print("\033[93mWarn\033[0m: no overhead instructions gathered")
    if np.sum(csv['rdCpts'] == 0) or np.sum(csv['wrCpts'] == 0):
        print("\033[93mWarn\033[0m: no LD/ST sampling percentages specified")
    if np.sum(csv['mReads'] == 0) and np.sum(csv['mWrites'] == 0):
        print("\033[93mWarn\033[0m: missing memory RD/WR counts")


    print(csv)
    print('-'*40)

    tTime = BasicModel(csv).runtime()
    print("Total execution time: %f" % tTime)



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: %s <csv>" % sys.argv[0])
        sys.exit(1)

    parse(sys.argv[1])
